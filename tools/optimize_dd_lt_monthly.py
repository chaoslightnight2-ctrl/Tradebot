#!/usr/bin/env python3
"""Backtest optimizer for configs where monthly return is greater than max drawdown.

This script runs offline backtests only. It never submits orders.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sontrade_bot import (  # noqa: E402
    FEATURES,
    Config,
    add_features,
    backtest_costs,
    fetch_bars,
    make_model,
    realized_gross_return,
    score_frame,
    setup_logging,
    should_fill,
)

REPORT_DIR = ROOT / "reports"
TRIALS_JSONL = REPORT_DIR / "dd_lt_monthly_optimizer_trials.jsonl"
VALID_JSON = REPORT_DIR / "dd_lt_monthly_valid_candidates.json"
BEST_JSON = REPORT_DIR / "dd_lt_monthly_best_config.json"
REPORT_MD = REPORT_DIR / "DD_LT_MONTHLY_OPTIMIZER_REPORT.md"


def pct(x: float) -> float:
    return round(float(x) * 100.0, 6)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def parse_symbols(raw: str) -> tuple[str, ...]:
    return tuple(s.strip().upper() for s in raw.split(",") if s.strip())


def base_config(args: argparse.Namespace) -> Config:
    return Config(
        symbols=parse_symbols(args.symbols) or ("AAPL",),
        benchmark_symbols=parse_symbols(args.benchmarks) or ("SPY", "QQQ"),
        timeframe=args.timeframe,
        lookback_days=max(args.lookback_days, 1825),
        data_provider=args.data_provider,
        data_feed=args.data_feed,
        bars_per_year=args.bars_per_year,
        random_seed=args.seed,
        commission_bps_per_side=args.commission_bps_per_side,
        spread_bps_round_trip=args.spread_bps_round_trip,
        slippage_bps_per_side=args.slippage_bps_per_side,
        short_borrow_apr=args.short_borrow_apr,
        fill_probability=args.fill_probability,
        min_dollar_volume=args.min_dollar_volume,
    )


def random_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    space = {
        "trade_direction": ["long", "short", "both"],
        "horizon_bars": [2, 3, 5, 8, 10, 13, 21],
        "label_threshold": [0.002, 0.003, 0.004, 0.006, 0.008, 0.01],
        "long_threshold": [0.20, 0.26, 0.32, 0.38, 0.44, 0.50, 0.58, 0.66],
        "short_threshold": [0.20, 0.26, 0.32, 0.38, 0.44, 0.50, 0.58, 0.66],
        "min_edge_gap": [0.00, 0.02, 0.04, 0.07, 0.10],
        "stop_atr_mult": [1.0, 1.4, 1.8, 2.2, 2.8, 3.2, 3.8, 4.4],
        "tp_r": [0.5, 0.7, 1.0, 1.3, 1.8, 2.4, 3.0],
        "min_stop_pct": [0.002, 0.004, 0.006, 0.01],
        "max_stop_pct": [0.018, 0.025, 0.035, 0.05, 0.066, 0.08],
        "risk_per_trade": [0.0025, 0.005, 0.0075],
        "max_positions": [1, 2, 3, 4],
    }
    keys = list(space)
    trials = [{k: rng.choice(space[k]) for k in keys} for _ in range(args.max_trials)]
    trials = [t for t in trials if t["min_stop_pct"] < t["max_stop_pct"]]
    for t in trials:
        if t["trade_direction"] == "long":
            t["short_threshold"] = 0.99
        if t["trade_direction"] == "short":
            t["long_threshold"] = 0.99
    trials.insert(0, {
        "trade_direction": "both",
        "horizon_bars": 8,
        "label_threshold": 0.003,
        "long_threshold": 0.20,
        "short_threshold": 0.48,
        "min_edge_gap": 0.03,
        "stop_atr_mult": 3.2,
        "tp_r": 0.6,
        "min_stop_pct": 0.002,
        "max_stop_pct": 0.066,
        "risk_per_trade": 0.005,
        "max_positions": 4,
    })
    return trials


def split_by_time(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = sorted(pd.to_datetime(df["timestamp"]).unique())
    cut = times[max(1, min(len(times) - 1, int(len(times) * frac)))]
    return df[df["timestamp"] < cut].copy(), df[df["timestamp"] >= cut].copy()


def train_bundle(train_df: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    return {
        "long_model": make_model(cfg).fit(train_df[FEATURES], train_df["long_target"]),
        "short_model": make_model(cfg).fit(train_df[FEATURES], train_df["short_target"]),
        "features": FEATURES,
        "config": asdict(cfg),
        "trained_at": datetime.now(UTC).isoformat(),
    }


def select_trades(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in scored.sort_values("timestamp").iterrows():
        long_score = safe_float(row.get("long_score"))
        short_score = safe_float(row.get("short_score"))
        edge_gap = abs(long_score - short_score)
        side: str | None = None
        if cfg.trade_direction in {"long", "both"} and long_score >= cfg.long_threshold and edge_gap >= cfg.min_edge_gap:
            side = "long"
        if cfg.trade_direction in {"short", "both"} and short_score >= cfg.short_threshold and edge_gap >= cfg.min_edge_gap:
            if side is None or short_score > long_score:
                side = "short"
        if side is None or not should_fill(row, side, cfg):
            continue
        gross = realized_gross_return(row, side, cfg)
        costs = backtest_costs(row, side, cfg)
        net = gross - costs["total_cost"]
        atr_stop = safe_float(row.get("atr_pct")) * cfg.stop_atr_mult
        stop_pct = max(cfg.min_stop_pct, min(cfg.max_stop_pct, atr_stop))
        position_return = net * cfg.risk_per_trade / stop_pct
        rows.append({
            "timestamp": str(row["timestamp"]),
            "symbol": row.get("symbol"),
            "side": side,
            "gross_return": gross,
            "net_return": net,
            "position_return": position_return,
            "long_score": long_score,
            "short_score": short_score,
            "edge_gap": edge_gap,
        })
    return pd.DataFrame(rows)


def metrics_from_trades(trades: pd.DataFrame, start_ts: Any, end_ts: Any, min_monthly_return_pct: float) -> dict[str, Any]:
    if trades.empty or start_ts is None or end_ts is None:
        return {
            "trade_count": 0, "long_trades": 0, "short_trades": 0,
            "win_rate_pct": 0.0, "total_return_pct": 0.0, "monthly_return_pct": 0.0,
            "max_drawdown_pct": 0.0, "profit_factor": 0.0, "expectancy_pct": 0.0,
            "dd_to_monthly_ratio": None, "passes_dd_lt_monthly": False, "passes_min_monthly": False,
        }
    r = trades["position_return"].astype(float).clip(lower=-0.95)
    equity = (1.0 + r).cumprod()
    max_dd = ((equity / equity.cummax()) - 1.0).min()
    total_return = equity.iloc[-1] - 1.0
    start = pd.to_datetime(start_ts)
    end = pd.to_datetime(end_ts)
    months = max(1.0 / 30.4375, (end - start).days / 30.4375)
    monthly_return = (1.0 + total_return) ** (1.0 / months) - 1.0 if total_return > -0.999 else -1.0
    wins = r[r > 0]
    losses = r[r < 0]
    monthly_pct = pct(monthly_return)
    dd_pct = pct(max_dd)
    ratio = abs(dd_pct) / monthly_pct if monthly_pct > 0 else None
    return {
        "trade_count": int(len(trades)),
        "long_trades": int((trades["side"] == "long").sum()),
        "short_trades": int((trades["side"] == "short").sum()),
        "win_rate_pct": round(float((r > 0).mean() * 100.0), 6),
        "total_return_pct": pct(total_return),
        "monthly_return_pct": monthly_pct,
        "max_drawdown_pct": dd_pct,
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 6) if abs(losses.sum()) > 0 else None,
        "expectancy_pct": pct(r.mean()),
        "dd_to_monthly_ratio": round(float(ratio), 6) if ratio is not None else None,
        "passes_dd_lt_monthly": bool(monthly_pct > 0 and abs(dd_pct) < monthly_pct),
        "passes_min_monthly": bool(monthly_pct > min_monthly_return_pct),
    }


def evaluate(scored: pd.DataFrame, cfg: Config, min_monthly_return_pct: float) -> dict[str, Any]:
    if scored.empty:
        return metrics_from_trades(pd.DataFrame(), None, None, min_monthly_return_pct)
    trades = select_trades(scored, cfg)
    return metrics_from_trades(trades, scored["timestamp"].min(), scored["timestamp"].max(), min_monthly_return_pct)


def walk_forward(df: pd.DataFrame, cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    times = sorted(pd.to_datetime(df["timestamp"]).unique())
    windows: list[dict[str, Any]] = []
    n = args.wf_train_bars + args.wf_test_bars
    if len(times) < n:
        return {"window_count": 0, "pass_rate": 0.0, "windows": []}
    for start in range(0, len(times) - n + 1, args.wf_step_bars):
        tr0, tr1 = times[start], times[start + args.wf_train_bars - 1]
        te0, te1 = times[start + args.wf_train_bars], times[start + n - 1]
        train_df = df[(pd.to_datetime(df["timestamp"]) >= tr0) & (pd.to_datetime(df["timestamp"]) <= tr1)]
        test_df = df[(pd.to_datetime(df["timestamp"]) >= te0) & (pd.to_datetime(df["timestamp"]) <= te1)]
        if len(train_df) < 100 or len(test_df) < 20:
            continue
        bundle = train_bundle(train_df, cfg)
        metrics = evaluate(score_frame(test_df, bundle), cfg, args.min_monthly_return_pct)
        metrics.update({"test_start": str(te0), "test_end": str(te1)})
        windows.append(metrics)
    pass_count = sum(1 for w in windows if w["passes_dd_lt_monthly"] and w["passes_min_monthly"])
    return {"window_count": len(windows), "pass_rate": round(pass_count / len(windows), 6) if windows else 0.0, "windows": windows}
