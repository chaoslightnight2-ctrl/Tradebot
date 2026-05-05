#!/usr/bin/env python3
"""Run the DD optimizer with a stricter objective.

Target:
- monthly_return_pct > 15
- abs(max_drawdown_pct) < monthly_return_pct
- among valid candidates, choose the lowest full-period max DD
- use OOS DD and monthly return only as tie-breakers
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.optimize_dd_lt_monthly as opt  # noqa: E402


def low_dd_candidate_status(
    full: dict[str, Any],
    oos: dict[str, Any],
    wf: dict[str, Any],
    args: Any,
) -> tuple[bool, float, list[str]]:
    reasons: list[str] = []

    if full["monthly_return_pct"] <= args.min_monthly_return_pct:
        reasons.append("full_monthly_return_not_above_target")
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

    full_dd = abs(float(full["max_drawdown_pct"]))
    oos_dd = abs(float(oos["max_drawdown_pct"]))
    full_monthly = float(full["monthly_return_pct"])
    oos_monthly = float(oos["monthly_return_pct"])
    wf_pass = float(wf.get("pass_rate", 0.0))

    # Higher score wins. The large DD penalties make lowest DD the main objective.
    score = (
        1_000_000.0
        - full_dd * 10_000.0
        - oos_dd * 1_000.0
        + full_monthly * 10.0
        + oos_monthly
        + wf_pass * 100.0
    )
    return True, round(score, 6), reasons


def main() -> None:
    args = opt.parse_args()
    args.timeout_seconds = max(args.timeout_seconds, 18_000)
    args.min_monthly_return_pct = max(args.min_monthly_return_pct, 15.0)
    args.max_trials = max(args.max_trials, 5_000)

    opt.candidate_status = low_dd_candidate_status
    opt.setup_logging(args.verbose)
    opt.run(args)


if __name__ == "__main__":
    main()
