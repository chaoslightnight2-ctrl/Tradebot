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
import hashlib
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

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "sontrade_model.joblib"
ENV_PATH = APP_DIR / ".env"

FEATURES = [
    # Price momentum
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    "ema12_dist", "ema26_dist", "ema200_dist",

    # MACD + trend quality
    "macd_hist_pct", "macd_line_pct", "macd_signal_pct", "trend_quality", "ema_slope_20",

    # VWAP
    "vwap_dist", "vwap_slope_5",

    # Oscillator / volatility
    "rsi14", "atr_pct", "range_pct",

    # Bollinger Band
    "bb_position", "bb_width", "bb_squeeze", "bb_upper_dist", "bb_lower_dist",

    # Gap
    "gap_pct", "gap_abs",

    # ADX / directional movement
    "adx14", "plus_di", "minus_di", "di_spread",

    # Volume confirmation
    "vol_z20", "relative_volume_20", "dollar_volume_z20", "obv_change_5", "volume_price_trend_5",

    # Breakout / drawdown
    "breakout20", "breakout55", "drawdown20", "proximity_high_252",

    # Opening range, useful on intraday data; zeroed on daily data.
    "opening_range_breakout", "opening_range_breakdown",

    # Market regime
    "spy_ret_5", "spy_ret_20", "spy_ema200_dist", "spy_trend_up",
    "qqq_ret_5", "qqq_ret_20", "qqq_ema200_dist", "qqq_trend_up",
    "market_regime_score",

    # Relative strength
    "rel_ret_spy_5", "rel_ret_spy_20", "rel_ret_qqq_5", "rel_ret_qqq_20",
]


@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...] = ("AAPL",)
    benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ")
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
    commission_bps_per_side: float = 0.0
    spread_bps_round_trip: float = 2.0
    slippage_bps_per_side: float = 1.0
    short_borrow_apr: float = 0.03
    fill_probability: float = 0.98
    max_atr_pct_for_fill: float = 0.12
    min_dollar_volume: float = 1_000_000.0
    bars_per_year: int = 252
    random_seed: int = 42


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_symbols(raw: str) -> tuple[str, ...]:
    return tuple(s.strip().upper() for s in raw.split(",") if s.strip())


def load_config() -> Config:
    symbols = parse_symbols(os.getenv("ALPACA_SYMBOLS", "AAPL"))
    benchmark_raw = os.getenv("SONTRADE_BENCHMARK_SYMBOLS", os.getenv("SONTRADE_BENCHMARK", "SPY,QQQ"))
    benchmarks = parse_symbols(benchmark_raw) or ("SPY", "QQQ")
    return Config(
        symbols=symbols or ("AAPL",),
        benchmark_symbols=benchmarks,
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
        commission_bps_per_side=float(os.getenv("SONTRADE_COMMISSION_BPS_PER_SIDE", "0.0")),
        spread_bps_round_trip=float(os.getenv("SONTRADE_SPREAD_BPS_ROUND_TRIP", "2.0")),
        slippage_bps_per_side=float(os.getenv("SONTRADE_SLIPPAGE_BPS_PER_SIDE", "1.0")),
        short_borrow_apr=float(os.getenv("SONTRADE_SHORT_BORROW_APR", "0.03")),
        fill_probability=float(os.getenv("SONTRADE_FILL_PROBABILITY", "0.98")),
        max_atr_pct_for_fill=float(os.getenv("SONTRADE_MAX_ATR_PCT_FOR_FILL", "0.12")),
        min_dollar_volume=float(os.getenv("SONTRADE_MIN_DOLLAR_VOLUME", "1000000")),
        bars_per_year=int(os.getenv("SONTRADE_BARS_PER_YEAR", "252")),
        random_seed=int(os.getenv("SONTRADE_RANDOM_SEED", "42")),
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


def timeframe_is_daily(value: str) -> bool:
    return value.strip().lower() in {"1d", "1day", "day"}


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


def feature_universe(cfg: Config) -> tuple[str, ...]:
    return tuple(dict.fromkeys(list(cfg.symbols) + list(cfg.benchmark_symbols)))


def fetch_yfinance(symbols: tuple[str, ...], cfg: Config) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    end = utc_now()
    start = end - timedelta(days=cfg.lookback_days)
    for symbol in symbols:
        df = yf.download(
            symbol,
            start=start.date().isoformat(),
            end=(end + timedelta(days=1)).date().isoformat(),
            interval=yf_interval(cfg.timeframe),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df.empty:
            df = yf.download(
                symbol,
                period=f"{max(30, cfg.lookback_days)}d",
                interval=yf_interval(cfg.timeframe),
                auto_adjust=False,
                progress=False,
                threads=False,
            )
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
    req = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=parse_timeframe(cfg.timeframe),
        start=utc_now() - timedelta(days=cfg.lookback_days),
        end=utc_now(),
        feed=cfg.data_feed,
    )
    df = data.get_stock_bars(req).df.reset_index()
    if df.empty:
        raise RuntimeError("Alpaca returned empty bars.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].sort_values(["symbol", "timestamp"])


def fetch_bars(cfg: Config) -> pd.DataFrame:
    universe = feature_universe(cfg)
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


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    df["adx14"] = (adx / 100).clip(0, 1)
    df["plus_di"] = (plus_di / 100).clip(0, 1)
    df["minus_di"] = (minus_di / 100).clip(0, 1)
    df["di_spread"] = ((plus_di - minus_di) / 100).clip(-1, 1)
    return df


def add_opening_range_features(df: pd.DataFrame, daily: bool) -> pd.DataFrame:
    if daily:
        df["opening_range_breakout"] = 0.0
        df["opening_range_breakdown"] = 0.0
        return df

    out = df.copy()
    session = out["timestamp"].dt.strftime("%Y-%m-%d")
    bar_no = out.groupby(session).cumcount()
    opening_mask = bar_no < 2
    opening_high = out["high"].where(opening_mask).groupby(session).transform("max")
    opening_low = out["low"].where(opening_mask).groupby(session).transform("min")
    opening_high = opening_high.groupby(session).ffill()
    opening_low = opening_low.groupby(session).ffill()

    close = out["close"].astype(float)
    out["opening_range_breakout"] = close / opening_high.replace(0, np.nan) - 1
    out["opening_range_breakdown"] = close / opening_low.replace(0, np.nan) - 1
    out[["opening_range_breakout", "opening_range_breakdown"]] = out[["opening_range_breakout", "opening_range_breakdown"]].fillna(0.0)
    return out


def add_symbol_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)
    prev_close = close.shift(1)

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    ema200 = ema(close, 200)

    macd_line = ema12 - ema26
    macd_signal = ema(macd_line, 9)
    macd_hist = macd_line - macd_signal

    true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()

    typical = (open_ + high + low + close) / 4
    dollar_volume = close * volume
    if timeframe_is_daily(cfg.timeframe):
        vwap = (typical * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    else:
        session = df["timestamp"].dt.strftime("%Y-%m-%d")
        vwap = (typical * volume).groupby(session).cumsum() / volume.groupby(session).cumsum().replace(0, np.nan)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_den = (bb_upper - bb_lower).replace(0, np.nan)

    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    vol20_sum = volume.rolling(20).sum().replace(0, np.nan)
    volume_mean20 = volume.rolling(20).mean()
    volume_std20 = volume.rolling(20).std().replace(0, np.nan)
    dollar_log = np.log1p(dollar_volume)

    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_20"] = close.pct_change(20)
    df["ret_60"] = close.pct_change(60)

    df["ema12_dist"] = close / ema12 - 1
    df["ema26_dist"] = close / ema26 - 1
    df["ema200_dist"] = close / ema200 - 1
    df["ema_slope_20"] = ema26.pct_change(20)

    df["macd_hist_pct"] = macd_hist / close
    df["macd_line_pct"] = macd_line / close
    df["macd_signal_pct"] = macd_signal / close
    df["trend_quality"] = ((ema12 / ema26 - 1) * np.sign(df["ret_5"])).fillna(0.0)

    df["vwap_dist"] = close / vwap - 1
    df["vwap_slope_5"] = vwap.pct_change(5)

    df["rsi14"] = (rsi(close, 14) - 50) / 50
    df["atr_pct"] = atr / close
    df["range_pct"] = (high - low) / close

    df["bb_position"] = ((close - bb_lower) / bb_den - 0.5).clip(-2, 2)
    df["bb_width"] = bb_den / close
    df["bb_squeeze"] = df["bb_width"] / df["bb_width"].rolling(120).mean() - 1
    df["bb_upper_dist"] = close / bb_upper - 1
    df["bb_lower_dist"] = close / bb_lower - 1

    df["gap_pct"] = open_ / prev_close - 1
    df["gap_abs"] = df["gap_pct"].abs()

    df["vol_z20"] = (volume - volume_mean20) / volume_std20
    df["relative_volume_20"] = volume / volume_mean20.replace(0, np.nan) - 1
    df["dollar_volume"] = dollar_volume
    df["dollar_volume_z20"] = (dollar_log - dollar_log.rolling(20).mean()) / dollar_log.rolling(20).std().replace(0, np.nan)
    df["obv_change_5"] = obv.diff(5) / vol20_sum
    df["volume_price_trend_5"] = (df["ret_1"] * volume).rolling(5).sum() / volume_mean20.replace(0, np.nan)

    df["breakout20"] = close / high.shift(1).rolling(20).max() - 1
    df["breakout55"] = close / high.shift(1).rolling(55).max() - 1
    df["drawdown20"] = close / close.rolling(20).max() - 1
    df["proximity_high_252"] = close / high.rolling(252).max() - 1

    df = add_adx(df)
    df = add_opening_range_features(df, daily=timeframe_is_daily(cfg.timeframe))

    future_return = close.shift(-cfg.horizon_bars) / close - 1
    df["future_return"] = future_return
    df["future_high_max"] = high.shift(-1).rolling(cfg.horizon_bars, min_periods=cfg.horizon_bars).max()
    df["future_low_min"] = low.shift(-1).rolling(cfg.horizon_bars, min_periods=cfg.horizon_bars).min()
    df["long_target"] = (future_return > cfg.label_threshold).astype(int)
    df["short_target"] = ((-future_return) > cfg.label_threshold).astype(int)
    return df


def add_market_context(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = features.sort_values(["symbol", "timestamp"]).copy()

    for col in [
        "spy_ret_5", "spy_ret_20", "spy_ema200_dist", "spy_trend_up",
        "qqq_ret_5", "qqq_ret_20", "qqq_ema200_dist", "qqq_trend_up",
        "rel_ret_spy_5", "rel_ret_spy_20", "rel_ret_qqq_5", "rel_ret_qqq_20",
        "market_regime_score",
    ]:
        out[col] = 0.0

    bench_specs = []
    for symbol in cfg.benchmark_symbols:
        prefix = symbol.lower()
        if symbol == "SPY":
            prefix = "spy"
        elif symbol == "QQQ":
            prefix = "qqq"
        bench_specs.append((symbol, prefix))

    trend_cols: list[str] = []
    for symbol, prefix in bench_specs:
        bench = out[out["symbol"] == symbol]
        if bench.empty:
            continue
        ctx = bench[["timestamp", "ret_5", "ret_20", "ema200_dist"]].rename(
            columns={
                "ret_5": f"{prefix}_ret_5",
                "ret_20": f"{prefix}_ret_20",
                "ema200_dist": f"{prefix}_ema200_dist",
            }
        )
        out = out.merge(ctx, on="timestamp", how="left", suffixes=("", "_ctx"))

        for col in [f"{prefix}_ret_5", f"{prefix}_ret_20", f"{prefix}_ema200_dist"]:
            ctx_col = f"{col}_ctx"
            if ctx_col in out.columns:
                out[col] = out[ctx_col].fillna(out.get(col, 0.0)).fillna(0.0)
                out = out.drop(columns=[ctx_col])

        trend_col = f"{prefix}_trend_up"
        if f"{prefix}_ema200_dist" in out.columns:
            out[trend_col] = (out[f"{prefix}_ema200_dist"] > 0).astype(float)
            trend_cols.append(trend_col)

        if prefix in {"spy", "qqq"}:
            out[f"rel_ret_{prefix}_5"] = out["ret_5"] - out[f"{prefix}_ret_5"]
            out[f"rel_ret_{prefix}_20"] = out["ret_20"] - out[f"{prefix}_ret_20"]

    if trend_cols:
        out["market_regime_score"] = out[trend_cols].mean(axis=1)

    return out


def add_features(raw: pd.DataFrame, cfg: Config, require_targets: bool = True) -> pd.DataFrame:
    frames = []
    for _, df in raw.groupby("symbol", sort=False):
        frames.append(add_symbol_features(df, cfg))

    out = pd.concat(frames, ignore_index=True)
    out = add_market_context(out, cfg)
    out = out[out["symbol"].isin(cfg.symbols)].copy()

    for col in FEATURES:
        if col not in out.columns:
            out[col] = 0.0

    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    required = FEATURES + (["future_return", "future_high_max", "future_low_min"] if require_targets else [])
    return out.dropna(subset=required).reset_index(drop=True)


def build_dataset(cfg: Config, require_targets: bool = True) -> pd.DataFrame:
    return add_features(fetch_bars(cfg), cfg, require_targets=require_targets)


def split_time(df: pd.DataFrame, frac: float = 0.78) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = sorted(df["timestamp"].unique())
    cut = times[max(1, min(len(times) - 1, int(len(times) * frac)))]
    return df[df["timestamp"] < cut].copy(), df[df["timestamp"] >= cut].copy()


def make_model(cfg: Config) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=450,
        max_depth=7,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=cfg.random_seed,
        n_jobs=-1,
    )


def train(cfg: Config) -> dict[str, Any]:
    df = build_dataset(cfg)
    train_df, test_df = split_time(df)
    long_model = make_model(cfg).fit(train_df[FEATURES], train_df["long_target"])
    short_model = make_model(cfg).fit(train_df[FEATURES], train_df["short_target"])
    bundle = {
        "long_model": long_model,
        "short_model": short_model,
        "features": FEATURES,
        "config": asdict(cfg),
        "trained_at": utc_now().isoformat(),
    }
    joblib.dump(bundle, MODEL_PATH)
    scored = score_frame(test_df, bundle)
    metrics = {
        "rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "long": model_metrics(test_df["long_target"], scored["long_score"]),
        "short": model_metrics(test_df["short_target"], scored["short_score"]),
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def model_metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    out = {
        "base_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
    }
    if y.nunique() > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
    return out


def load_model(cfg: Config) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        train(cfg)
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("features") != FEATURES:
        logging.info("Model feature list changed; retraining model.")
        train(cfg)
        bundle = joblib.load(MODEL_PATH)
    return bundle


def score_frame(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    features = bundle.get("features", FEATURES)
    out["long_score"] = bundle["long_model"].predict_proba(out[features])[:, 1]
    out["short_score"] = bundle["short_model"].predict_proba(out[features])[:, 1]
    out["edge_gap"] = (out["long_score"] - out["short_score"]).abs()
    return out


def deterministic_unit_interval(*parts: object) -> float:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:12]
    return int(digest, 16) / float(16**12 - 1)


def should_fill(row: pd.Series, side: str, cfg: Config) -> bool:
    if float(row.get("dollar_volume", 0.0)) < cfg.min_dollar_volume:
        return False
    atr_pct = float(row.get("atr_pct", 0.0))
    if np.isfinite(atr_pct) and atr_pct > cfg.max_atr_pct_for_fill:
        return False
    fill_probability = min(1.0, max(0.0, cfg.fill_probability))
    if np.isfinite(atr_pct):
        fill_probability -= max(0.0, atr_pct - 0.04) * 1.5
    fill_probability = min(1.0, max(0.0, fill_probability))
    key = deterministic_unit_interval(row.get("timestamp"), row.get("symbol"), side, cfg.random_seed)
    return key <= fill_probability


def realized_gross_return(row: pd.Series, side: str, cfg: Config) -> float:
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


def backtest_costs(row: pd.Series, side: str, cfg: Config) -> dict[str, float]:
    commission = 2.0 * cfg.commission_bps_per_side / 10_000.0
    spread = cfg.spread_bps_round_trip / 10_000.0
    slippage = 2.0 * cfg.slippage_bps_per_side / 10_000.0
    borrow = 0.0
    if side == "short":
        borrow = cfg.short_borrow_apr * (cfg.horizon_bars / max(1, cfg.bars_per_year))
    total = commission + spread + slippage + borrow
    return {"commission": commission, "spread": spread, "slippage": slippage, "borrow": borrow, "total": total}


def realized_net_return(row: pd.Series, side: str, cfg: Config) -> tuple[float, float, dict[str, float]]:
    gross = realized_gross_return(row, side, cfg)
    costs = backtest_costs(row, side, cfg)
    net = gross - costs["total"]
    return gross, net, costs


def backtest(cfg: Config) -> dict[str, Any]:
    df = build_dataset(cfg)
    _, test_df = split_time(df)
    bundle = load_model(cfg)
    scored = score_frame(test_df, bundle)
    trades = []
    skipped_no_fill = 0

    for _, row in scored.iterrows():
        long_ok = cfg.trade_direction in {"both", "long"} and row.long_score >= cfg.long_threshold and row.long_score > row.short_score and row.edge_gap >= cfg.min_edge_gap
        short_ok = cfg.trade_direction in {"both", "short"} and row.short_score >= cfg.short_threshold and row.short_score > row.long_score and row.edge_gap >= cfg.min_edge_gap
        if cfg.trade_direction == "long":
            long_ok = row.long_score >= cfg.long_threshold
        if cfg.trade_direction == "short":
            short_ok = row.short_score >= cfg.short_threshold

        side = "long" if long_ok else "short" if short_ok else None
        if side is None:
            continue
        if not should_fill(row, side, cfg):
            skipped_no_fill += 1
            continue

        gross, net, costs = realized_net_return(row, side, cfg)
        trades.append({
            "timestamp": row.timestamp,
            "symbol": row.symbol,
            "side": side,
            "gross_return": gross,
            "return": net,
            **{f"cost_{k}": v for k, v in costs.items()},
        })

    if not trades:
        result = {"trades": 0, "skipped_no_fill": skipped_no_fill, "note": "No filled trades with current thresholds and fill filters."}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    t = pd.DataFrame(trades)
    gross_r = t["gross_return"].clip(-0.08, 0.08)
    net_r = t["return"].clip(-0.08, 0.08)
    gross_eq = (1 + gross_r).cumprod()
    net_eq = (1 + net_r).cumprod()
    result = {
        "symbols": list(cfg.symbols),
        "start_date": str(pd.Timestamp(scored["timestamp"].min()).date()),
        "end_date": str(pd.Timestamp(scored["timestamp"].max()).date()),
        "bars": int(scored["timestamp"].nunique()),
        "feature_count": len(FEATURES),
        "signals_not_filled": int(skipped_no_fill),
        "trades": int(len(t)),
        "long_trades": int((t["side"] == "long").sum()),
        "short_trades": int((t["side"] == "short").sum()),
        "win_rate": float((net_r > 0).mean()),
        "avg_gross_return": float(gross_r.mean()),
        "avg_net_return": float(net_r.mean()),
        "median_net_return": float(net_r.median()),
        "gross_compounded_return": float(gross_eq.iloc[-1] - 1),
        "net_compounded_return": float(net_eq.iloc[-1] - 1),
        "max_drawdown": float((net_eq / net_eq.cummax() - 1).min()),
        "avg_total_cost": float(t["cost_total"].mean()),
        "avg_spread_cost": float(t["cost_spread"].mean()),
        "avg_slippage_cost": float(t["cost_slippage"].mean()),
        "avg_commission_cost": float(t["cost_commission"].mean()),
        "avg_borrow_cost": float(t["cost_borrow"].mean()),
        "cost_model": {
            "commission_bps_per_side": cfg.commission_bps_per_side,
            "spread_bps_round_trip": cfg.spread_bps_round_trip,
            "slippage_bps_per_side": cfg.slippage_bps_per_side,
            "short_borrow_apr": cfg.short_borrow_apr,
            "fill_probability": cfg.fill_probability,
            "max_atr_pct_for_fill": cfg.max_atr_pct_for_fill,
            "min_dollar_volume": cfg.min_dollar_volume,
        },
        "features": FEATURES,
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

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=tp),
            stop_loss=StopLossRequest(stop_price=sl),
        )
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
