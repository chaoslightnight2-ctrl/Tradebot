#!/usr/bin/env python3
"""
Sontrade dashboard-free tradebot.

Modes:
  python sontrade_bot.py train
  python sontrade_bot.py backtest
  python sontrade_bot.py score
  python sontrade_bot.py trade
  python sontrade_bot.py loop

API keys are read from .env or environment variables. Do not hardcode keys in public repos.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, StopLossRequest, TakeProfitRequest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "sontrade_model.joblib"
ENV_PATH = APP_DIR / ".env"

FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ema12_dist", "ema26_dist",
    "ema200_dist", "rsi14", "atr_pct", "vol_z20", "range_pct", "breakout20",
    "drawdown20", "market_ret_5", "rel_ret_spy_5", "rel_ret_spy_20",
]

@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...] = ("AAPL",)
    benchmark: str = "SPY"
    timeframe: str = "1Day"
    lookback_days: int = 2500
    data_provider: str = "yfinance"
    data_feed: str = "iex"
    horizon_bars: int = 8
    label_threshold: float = 0.003
    long_threshold: float = 0.20
    short_threshold: float = 0.48
    min_edge_gap: float = 0.03
    trade_direction: str = "both"
    stop_atr_mult: float = 3.2
    tp_r: float = 0.6
    min_stop_pct: float = 0.002
    max_stop_pct: float = 0.066
    risk_per_trade: float = 0.005
    max_notional: float = 6000.0
    max_positions: int = 4
    min_order_dollars: float = 50.0


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> Config:
    symbols = tuple(s.strip().upper() for s in os.getenv("ALPACA_SYMBOLS", "AAPL").split(",") if s.strip())
    return Config(
        symbols=symbols or ("AAPL",),
        benchmark=os.getenv("SONTRADE_BENCHMARK", "SPY").strip().upper(),
        timeframe=os.getenv("ALPACA_TIMEFRAME", "1Day"),
        lookback_days=int(os.getenv("SONTRADE_LOOKBACK_DAYS", "2500")),
        data_provider=os.getenv("SONTRADE_DATA_PROVIDER", "yfinance").strip().lower(),
        data_feed=os.getenv("ALPACA_DATA_FEED", "iex").strip().lower(),
        horizon_bars=int(os.getenv("SONTRADE_HORIZON_BARS", "8")),
        label_threshold=float(os.getenv("SONTRADE_LABEL_THRESHOLD", "0.003")),
        long_threshold=float(os.getenv("SONTRADE_LONG_THRESHOLD", "0.20")),
        short_threshold=float(os.getenv("SONTRADE_SHORT_THRESHOLD", "0.48")),
        min_edge_gap=float(os.getenv("SONTRADE_MIN_EDGE_GAP", "0.03")),
        trade_direction=os.getenv("SONTRADE_TRADE_DIRECTION", "both").strip().lower(),
        stop_atr_mult=float(os.getenv("SONTRADE_STOP_ATR_MULT", "3.2")),
        tp_r=float(os.getenv("SONTRADE_TP_R", "0.6")),
        min_stop_pct=float(os.getenv("SONTRADE_MIN_STOP_PCT", "0.002")),
        max_stop_pct=float(os.getenv("SONTRADE_MAX_STOP_PCT", "0.066")),
        risk_per_trade=float(os.getenv("SONTRADE_RISK_PER_TRADE", "0.005")),
        max_notional=float(os.getenv("SONTRADE_MAX_NOTIONAL", "6000")),
        max_positions=int(os.getenv("SONTRADE_MAX_POSITIONS", "4")),
        min_order_dollars=float(os.getenv("SONTRADE_MIN_ORDER_DOLLARS", "50")),
    )


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_timeframe(value: str) -> TimeFrame:
    clean = value.strip().lower()
    if clean in {"1d", "1day", "day"}:
        return TimeFrame.Day
    if clean in {"1h", "1hour", "hour"}:
        return TimeFrame.Hour
    if clean.endswith("min"):
        return TimeFrame(int(clean[:-3]), TimeFrameUnit.Minute)
    if clean.endswith("m") and clean[:-1].isdigit():
        return TimeFrame(int(clean[:-1]), TimeFrameUnit.Minute)
    raise ValueError(f"Unsupported timeframe: {value}")


def yf_interval(timeframe: str) -> str:
    clean = timeframe.strip().lower()
    if clean in {"1d", "1day", "day"}:
        return "1d"
    if clean in {"1h", "1hour", "hour"}:
        return "1h"
    if clean.endswith("min"):
        return clean[:-3] + "m"
    if clean.endswith("m"):
        return clean
    return "1d"


def require_keys() -> tuple[str, str]:
    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env or environment variables.")
    return key, secret


def alpaca_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    key, secret = require_keys()
    return TradingClient(key, secret, paper=True), StockHistoricalDataClient(key, secret)


def fetch_yfinance(symbols: tuple[str, ...], cfg: Config) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    end = utc_now()
    start = end - timedelta(days=cfg.lookback_days)
    for symbol in symbols:
        df = yf.download(symbol, start=start.date().isoformat(), end=(end + timedelta(days=1)).date().isoformat(), interval=yf_interval(cfg.timeframe), auto_adjust=False, progress=False, threads=False)
        if df.empty:
            df = yf.download(symbol, period=f"{max(30, cfg.lookback_days)}d", interval=yf_interval(cfg.timeframe), auto_adjust=False, progress=False, threads=False)
        if df.empty:
            logging.warning("No yfinance data for %s", symbol)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index().rename(columns={"Date": "timestamp", "Datetime": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = symbol
        frames.append(df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]])
    if not frames:
        raise RuntimeError("No market data returned.")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])


def fetch_alpaca(symbols: tuple[str, ...], cfg: Config) -> pd.DataFrame:
    _, data = alpaca_clients()
    req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=parse_timeframe(cfg.timeframe), start=utc_now() - timedelta(days=cfg.lookback_days), end=utc_now(), feed=cfg.data_feed)
    df = data.get_stock_bars(req).df.reset_index()
    if df.empty:
        raise RuntimeError("Alpaca returned empty bars.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].sort_values(["symbol", "timestamp"])


def fetch_bars(cfg: Config) -> pd.DataFrame:
    universe = tuple(dict.fromkeys(list(cfg.symbols) + [cfg.benchmark]))
    if cfg.data_provider == "alpaca":
        return fetch_alpaca(universe, cfg)
    return fetch_yfinance(universe, cfg)


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_features(raw: pd.DataFrame, cfg: Config, require_targets: bool = True) -> pd.DataFrame:
    frames = []
    for symbol, df in raw.groupby("symbol", sort=False):
        df = df.sort_values("timestamp").copy()
        c = df["close"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        v = df["volume"].astype(float)
        prev = c.shift(1)
        tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df["ret_1"] = c.pct_change(1)
        df["ret_3"] = c.pct_change(3)
        df["ret_5"] = c.pct_change(5)
        df["ret_10"] = c.pct_change(10)
        df["ret_20"] = c.pct_change(20)
        df["ema12_dist"] = c / ema(c, 12) - 1
        df["ema26_dist"] = c / ema(c, 26) - 1
        df["ema200_dist"] = c / ema(c, 200) - 1
        df["rsi14"] = (rsi(c, 14) - 50) / 50
        df["atr_pct"] = atr / c
        df["vol_z20"] = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
        df["range_pct"] = (h - l) / c
        df["breakout20"] = c / h.shift(1).rolling(20).max() - 1
        df["drawdown20"] = c / c.rolling(20).max() - 1
        fut = c.shift(-cfg.horizon_bars) / c - 1
        df["future_return"] = fut
        df["future_high_max"] = h.shift(-1).rolling(cfg.horizon_bars, min_periods=cfg.horizon_bars).max()
        df["future_low_min"] = l.shift(-1).rolling(cfg.horizon_bars, min_periods=cfg.horizon_bars).min()
        df["long_target"] = (fut > cfg.label_threshold).astype(int)
        df["short_target"] = ((-fut) > cfg.label_threshold).astype(int)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    bench = out[out["symbol"] == cfg.benchmark][["timestamp", "ret_5", "ret_20"]].rename(columns={"ret_5": "market_ret_5", "ret_20": "market_ret_20"})
    out = out.merge(bench, on="timestamp", how="left")
    out["market_ret_5"] = out["market_ret_5"].fillna(0)
    out["rel_ret_spy_5"] = out["ret_5"] - out["market_ret_5"]
    out["rel_ret_spy_20"] = out["ret_20"] - out["market_ret_20"].fillna(0)
    out = out[out["symbol"].isin(cfg.symbols)].copy()
    cols = FEATURES + (["future_return", "future_high_max", "future_low_min"] if require_targets else [])
    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=cols).reset_index(drop=True)


def build_dataset(cfg: Config, require_targets: bool = True) -> pd.DataFrame:
    return add_features(fetch_bars(cfg), cfg, require_targets=require_targets)


def split_time(df: pd.DataFrame, frac: float = 0.78) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = sorted(df["timestamp"].unique())
    cut = times[max(1, min(len(times)-1, int(len(times)*frac)))]
    return df[df["timestamp"] < cut].copy(), df[df["timestamp"] >= cut].copy()


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=350, max_depth=6, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1)


def train(cfg: Config) -> dict[str, Any]:
    df = build_dataset(cfg)
    train_df, test_df = split_time(df)
    long_model = make_model().fit(train_df[FEATURES], train_df["long_target"])
    short_model = make_model().fit(train_df[FEATURES], train_df["short_target"])
    bundle = {"long_model": long_model, "short_model": short_model, "features": FEATURES, "config": asdict(cfg), "trained_at": utc_now().isoformat()}
    joblib.dump(bundle, MODEL_PATH)
    scored = score_frame(test_df, bundle)
    metrics = {"rows": len(df), "train_rows": len(train_df), "test_rows": len(test_df), "long": model_metrics(test_df["long_target"], scored["long_score"]), "short": model_metrics(test_df["short_target"], scored["short_score"])}
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def model_metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    out = {"base_rate": float(y.mean()), "accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0))}
    if y.nunique() > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
    return out


def load_model(cfg: Config) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        train(cfg)
    return joblib.load(MODEL_PATH)


def score_frame(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["long_score"] = bundle["long_model"].predict_proba(out[FEATURES])[:, 1]
    out["short_score"] = bundle["short_model"].predict_proba(out[FEATURES])[:, 1]
    out["edge_gap"] = (out["long_score"] - out["short_score"]).abs()
    return out


def realized_return(row: pd.Series, side: str, cfg: Config) -> float:
    entry = float(row["close"])
    atr_pct = float(row.get("atr_pct", 0.0))
    stop_pct = min(cfg.max_stop_pct, max(cfg.min_stop_pct, atr_pct * cfg.stop_atr_mult))
    tp_pct = stop_pct * cfg.tp_r
    hi = float(row.get("future_high_max", np.nan))
    lo = float(row.get("future_low_min", np.nan))
    fallback = float(row["future_return"]) if side == "long" else -float(row["future_return"])
    if not np.isfinite(hi) or not np.isfinite(lo):
        return fallback
    if side == "long":
        stop_hit = lo <= entry * (1 - stop_pct)
        tp_hit = hi >= entry * (1 + tp_pct)
    else:
        stop_hit = hi >= entry * (1 + stop_pct)
        tp_hit = lo <= entry * (1 - tp_pct)
    if stop_hit and tp_hit:
        return -stop_pct
    if stop_hit:
        return -stop_pct
    if tp_hit:
        return tp_pct
    return fallback


def backtest(cfg: Config) -> dict[str, Any]:
    df = build_dataset(cfg)
    train_df, test_df = split_time(df)
    bundle = load_model(cfg)
    scored = score_frame(test_df, bundle)
    trades = []
    for _, row in scored.iterrows():
        long_ok = cfg.trade_direction in {"both", "long"} and row.long_score >= cfg.long_threshold and row.long_score > row.short_score and row.edge_gap >= cfg.min_edge_gap
        short_ok = cfg.trade_direction in {"both", "short"} and row.short_score >= cfg.short_threshold and row.short_score > row.long_score and row.edge_gap >= cfg.min_edge_gap
        if cfg.trade_direction == "long":
            long_ok = row.long_score >= cfg.long_threshold
        if cfg.trade_direction == "short":
            short_ok = row.short_score >= cfg.short_threshold
        if long_ok:
            trades.append({"timestamp": row.timestamp, "symbol": row.symbol, "side": "long", "return": realized_return(row, "long", cfg)})
        elif short_ok:
            trades.append({"timestamp": row.timestamp, "symbol": row.symbol, "side": "short", "return": realized_return(row, "short", cfg)})
    if not trades:
        return {"trades": 0, "note": "No trades with current thresholds."}
    t = pd.DataFrame(trades)
    r = t["return"].clip(-0.08, 0.08)
    eq = (1 + r).cumprod()
    result = {
        "symbols": list(cfg.symbols),
        "start_date": str(pd.Timestamp(scored["timestamp"].min()).date()),
        "end_date": str(pd.Timestamp(scored["timestamp"].max()).date()),
        "bars": int(scored["timestamp"].nunique()),
        "trades": int(len(t)),
        "long_trades": int((t["side"] == "long").sum()),
        "short_trades": int((t["side"] == "short").sum()),
        "win_rate": float((r > 0).mean()),
        "avg_return": float(r.mean()),
        "median_return": float(r.median()),
        "total_compounded_return": float(eq.iloc[-1] - 1),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def latest_score(cfg: Config) -> pd.DataFrame:
    df = build_dataset(cfg, require_targets=False)
    idx = df.groupby("symbol")["timestamp"].idxmax()
    latest = df.loc[idx].copy()
    bundle = load_model(cfg)
    scored = score_frame(latest, bundle)
    scored["decision"] = "hold"
    long_ok = (scored["long_score"] >= cfg.long_threshold) & (scored["long_score"] > scored["short_score"]) & (scored["edge_gap"] >= cfg.min_edge_gap)
    short_ok = (scored["short_score"] >= cfg.short_threshold) & (scored["short_score"] > scored["long_score"]) & (scored["edge_gap"] >= cfg.min_edge_gap)
    if cfg.trade_direction == "long":
        long_ok = scored["long_score"] >= cfg.long_threshold
        short_ok = False
    if cfg.trade_direction == "short":
        short_ok = scored["short_score"] >= cfg.short_threshold
        long_ok = False
    scored.loc[long_ok, "decision"] = "long"
    scored.loc[short_ok, "decision"] = "short"
    return scored.sort_values("edge_gap", ascending=False)


def round_price(x: float) -> float:
    return round(x, 2) if x >= 1 else round(x, 4)


def position_size(equity: float, price: float, atr_pct: float, cfg: Config) -> int:
    stop_pct = min(cfg.max_stop_pct, max(cfg.min_stop_pct, atr_pct * cfg.stop_atr_mult))
    risk_qty = (equity * cfg.risk_per_trade) / max(price * stop_pct, 0.01)
    notional_qty = cfg.max_notional / price
    qty = math.floor(min(risk_qty, notional_qty))
    return qty if qty * price >= cfg.min_order_dollars else 0


def trade_once(cfg: Config) -> pd.DataFrame:
    trading, _ = alpaca_clients()
    clock = trading.get_clock()
    if not clock.is_open:
        logging.info("Market is closed. Next open: %s", clock.next_open)
        return pd.DataFrame()
    account = trading.get_account()
    equity = float(account.equity)
    positions = {p.symbol.upper() for p in trading.get_all_positions()}
    try:
        orders = {o.symbol.upper() for o in trading.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))}
    except Exception:
        orders = set()
    blocked = positions | orders
    scored = latest_score(cfg)
    sent = []
    for _, row in scored.iterrows():
        if len(positions) + len(sent) >= cfg.max_positions:
            break
        symbol = str(row.symbol).upper()
        decision = str(row.decision)
        if decision == "hold" or symbol in blocked:
            continue
        price = float(row.close)
        qty = position_size(equity, price, float(row.atr_pct), cfg)
        if qty <= 0:
            continue
        stop_pct = min(cfg.max_stop_pct, max(cfg.min_stop_pct, float(row.atr_pct) * cfg.stop_atr_mult))
        tp_pct = stop_pct * cfg.tp_r
        if decision == "long":
            side = OrderSide.BUY
            tp = round_price(price * (1 + tp_pct))
            sl = round_price(price * (1 - stop_pct))
        else:
            side = OrderSide.SELL
            tp = round_price(price * (1 - tp_pct))
            sl = round_price(price * (1 + stop_pct))
        order = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET, take_profit=TakeProfitRequest(limit_price=tp), stop_loss=StopLossRequest(stop_price=sl))
        res = trading.submit_order(order_data=order)
        sent.append({"symbol": symbol, "decision": decision, "qty": qty, "tp": tp, "sl": sl, "order_id": str(res.id)})
    return pd.DataFrame(sent)


def loop(cfg: Config) -> None:
    while True:
        try:
            out = trade_once(cfg)
            if out.empty:
                logging.info("No order this cycle.")
            else:
                logging.info("Orders:\n%s", out.to_string(index=False))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logging.exception("Loop error: %s", exc)
        time.sleep(int(os.getenv("SONTRADE_LOOP_SECONDS", "900")))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sontrade dashboard-free CLI")
    p.add_argument("mode", nargs="?", default="score", choices=["train", "backtest", "score", "trade", "loop"])
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    setup_logging(args.verbose)
    cfg = load_config()
    logging.info("Config: %s", json.dumps(asdict(cfg), ensure_ascii=False))
    if args.mode == "train":
        train(cfg)
    elif args.mode == "backtest":
        backtest(cfg)
    elif args.mode == "score":
        scored = latest_score(cfg)
        cols = ["symbol", "timestamp", "decision", "long_score", "short_score", "edge_gap", "close"]
        if args.json:
            print(json.dumps(scored[cols].to_dict(orient="records"), indent=2, ensure_ascii=False, default=str))
        else:
            print(scored[cols].to_string(index=False))
    elif args.mode == "trade":
        sent = trade_once(cfg)
        print("Bu turda emir yok." if sent.empty else sent.to_string(index=False))
    elif args.mode == "loop":
        loop(cfg)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDurduruldu.")
        raise SystemExit(130)
