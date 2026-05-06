#!/usr/bin/env python3
"""1-hour optimizer focused on monthly return near 15% with the lowest DD.

Target:
- timeframe = 1h
- monthly_return_pct around 15, accepted band defaults to 12-20
- DD < monthly return gate is intentionally disabled
- among valid candidates, choose lowest full-period max DD
- prove with full backtest, OOS and walk-forward monthly-return gates
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.optimize_dd_lt_monthly as opt  # noqa: E402

TARGET_MONTHLY = 15.0
MAX_MONTHLY = 20.0


def monthly_in_band(metrics: dict[str, Any], args: Any) -> bool:
    monthly = float(metrics.get("monthly_return_pct", 0.0))
    return args.min_monthly_return_pct <= monthly <= MAX_MONTHLY


def low_dd_no_dd_gate_status(
    full: dict[str, Any],
    oos: dict[str, Any],
    wf: dict[str, Any],
    args: Any,
) -> tuple[bool, float, list[str]]:
    reasons: list[str] = []
    full_monthly = float(full["monthly_return_pct"])
    oos_monthly = float(oos["monthly_return_pct"])
    full_dd = abs(float(full["max_drawdown_pct"]))
    oos_dd = abs(float(oos["max_drawdown_pct"]))

    if full_monthly < args.min_monthly_return_pct:
        reasons.append("full_monthly_below_target_band")
    if full_monthly > MAX_MONTHLY:
        reasons.append("full_monthly_too_far_above_15_target")
    if full["trade_count"] < args.min_total_trades:
        reasons.append("full_trade_count_too_low")
    if oos["trade_count"] < args.min_oos_trades:
        reasons.append("oos_trade_count_too_low")
    if oos_monthly < args.min_monthly_return_pct:
        reasons.append("oos_monthly_below_target_band")
    if wf.get("window_count", 0) and wf.get("pass_rate", 0.0) < 0.60:
        reasons.append("walk_forward_monthly_pass_rate_below_60pct")

    if reasons:
        return False, -999.0, reasons

    distance_to_target = abs(full_monthly - TARGET_MONTHLY)
    oos_distance = abs(oos_monthly - TARGET_MONTHLY)
    wf_pass = float(wf.get("pass_rate", 0.0))

    # Higher score wins. Lowest DD is primary; closeness to 15% monthly is secondary.
    score = (
        1_000_000.0
        - full_dd * 20_000.0
        - oos_dd * 2_500.0
        - distance_to_target * 1_000.0
        - oos_distance * 250.0
        + wf_pass * 500.0
    )
    return True, round(score, 6), reasons


def walk_forward_no_dd_gate(df: pd.DataFrame, cfg: Any, args: Any) -> dict[str, Any]:
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
        if len(train_df) < 200 or len(test_df) < 40:
            continue
        metrics = opt.evaluate(opt.score_frame(test_df, opt.train_bundle(train_df, cfg)), cfg, args.min_monthly_return_pct)
        metrics.update({"test_start": str(te0), "test_end": str(te1)})
        windows.append(metrics)

    pass_count = sum(1 for w in windows if monthly_in_band(w, args))
    return {
        "window_count": len(windows),
        "pass_rate": round(pass_count / len(windows), 6) if windows else 0.0,
        "windows": windows,
    }


def main() -> None:
    args = opt.parse_args()
    args.timeframe = "1h"
    args.timeout_seconds = max(60, args.timeout_seconds)
    args.max_trials = max(args.max_trials, 6_000)
    args.min_monthly_return_pct = max(args.min_monthly_return_pct, 12.0)
    args.min_total_trades = max(args.min_total_trades, 80)
    args.min_oos_trades = max(args.min_oos_trades, 25)
    args.bars_per_year = max(args.bars_per_year, 1638)
    args.wf_train_bars = max(args.wf_train_bars, 1000)
    args.wf_test_bars = max(args.wf_test_bars, 240)
    args.wf_step_bars = max(args.wf_step_bars, 240)

    opt.candidate_status = low_dd_no_dd_gate_status
    opt.walk_forward = walk_forward_no_dd_gate
    opt.setup_logging(args.verbose)
    opt.run(args)


if __name__ == "__main__":
    main()
