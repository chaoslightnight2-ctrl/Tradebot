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
