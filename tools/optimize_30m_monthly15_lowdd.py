#!/usr/bin/env python3
"""30-minute optimizer focused on monthly return near 15% with the lowest DD.

Target:
- timeframe = 30m
- monthly_return_pct around 15, accepted band defaults to 12-20
- abs(max_drawdown_pct) < monthly_return_pct
- among valid candidates, choose lowest full-period max DD
- prove with full backtest, OOS and walk-forward gates
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.optimize_dd_lt_monthly as opt  # noqa: E402

TARGET_MONTHLY = 15.0
MAX_MONTHLY = 20.0


def low_dd_30m_status(
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
    if not full["passes_dd_lt_monthly"]:
        reasons.append("full_dd_not_below_monthly")
    if full["trade_count"] < args.min_total_trades:
        reasons.append("full_trade_count_too_low")
    if oos["trade_count"] < args.min_oos_trades:
        reasons.append("oos_trade_count_too_low")
    if not oos["passes_dd_lt_monthly"] or not oos["passes_min_monthly"]:
        reasons.append("oos_gate_failed")
    if wf.get("window_count", 0) and wf.get("pass_rate", 0.0) < 0.60:
        reasons.append("walk_forward_pass_rate_below_60pct")

    if reasons:
        return False, -999.0, reasons

    distance_to_target = abs(full_monthly - TARGET_MONTHLY)
    oos_distance = abs(oos_monthly - TARGET_MONTHLY)
    wf_pass = float(wf.get("pass_rate", 0.0))

    score = (
        1_000_000.0
        - full_dd * 20_000.0
        - oos_dd * 2_500.0
        - distance_to_target * 1_000.0
        - oos_distance * 250.0
        + wf_pass * 500.0
    )
    return True, round(score, 6), reasons


def main() -> None:
    args = opt.parse_args()
    args.timeframe = "30m"
    # Respect the workflow/user timeout. Do not force it back to 5 hours.
    args.timeout_seconds = max(60, args.timeout_seconds)
    args.max_trials = max(args.max_trials, 6_000)
    args.min_monthly_return_pct = max(args.min_monthly_return_pct, 12.0)
    args.min_total_trades = max(args.min_total_trades, 120)
    args.min_oos_trades = max(args.min_oos_trades, 40)
    args.bars_per_year = max(args.bars_per_year, 3276)

    opt.candidate_status = low_dd_30m_status
    opt.setup_logging(args.verbose)
    opt.run(args)


if __name__ == "__main__":
    main()
