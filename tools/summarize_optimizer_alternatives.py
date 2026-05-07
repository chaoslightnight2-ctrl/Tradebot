#!/usr/bin/env python3
"""Append near-miss alternatives to optimizer reports.

The main optimizer is intentionally strict. This post-process step makes reports useful
when no candidate passes every gate by listing the strongest alternatives from
reports/dd_lt_monthly_optimizer_trials.jsonl.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
TRIALS_JSONL = REPORT_DIR / "dd_lt_monthly_optimizer_trials.jsonl"
REPORT_MD = REPORT_DIR / "DD_LT_MONTHLY_OPTIMIZER_REPORT.md"
ALTERNATIVES_JSON = REPORT_DIR / "optimizer_top_alternatives.json"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def load_trials() -> list[dict[str, Any]]:
    if not TRIALS_JSONL.exists():
        return []
    rows: list[dict[str, Any]] = []
    with TRIALS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "full_metrics" in item and "oos_metrics" in item:
                rows.append(item)
    return rows


def alternative_score(item: dict[str, Any]) -> float:
    full = item.get("full_metrics", {})
    oos = item.get("oos_metrics", {})
    wf = item.get("walk_forward", {})
    trial = item.get("trial", {})

    full_monthly = safe_float(full.get("monthly_return_pct"))
    oos_monthly = safe_float(oos.get("monthly_return_pct"))
    full_dd = abs(safe_float(full.get("max_drawdown_pct")))
    oos_dd = abs(safe_float(oos.get("max_drawdown_pct")))
    full_trades = safe_float(full.get("trade_count"))
    oos_trades = safe_float(oos.get("trade_count"))
    wf_pass = safe_float(wf.get("pass_rate"))

    # Prefer monthly return around 15%, but strongly penalize high DD.
    target_gap = abs(full_monthly - 15.0)
    score = 0.0
    score += full_monthly * 20.0
    score += oos_monthly * 8.0
    score -= full_dd * 25.0
    score -= oos_dd * 10.0
    score -= target_gap * 12.0
    score += wf_pass * 100.0
    score += min(full_trades, 200.0) * 0.15
    score += min(oos_trades, 80.0) * 0.30

    # Reward candidates that at least satisfy the user's broad target band.
    if 12.0 <= full_monthly <= 20.0:
        score += 200.0
    if 12.0 <= oos_monthly:
        score += 120.0
    if full_monthly > 0 and full_dd < full_monthly:
        score += 150.0
    if oos_monthly > 0 and oos_dd < oos_monthly:
        score += 75.0

    # Slightly prefer simpler / lower exposure configs.
    score -= safe_float(trial.get("max_positions"), 1.0) * 2.0
    score -= safe_float(trial.get("risk_per_trade"), 0.0) * 200.0
    return round(score, 6)


def compact_candidate(item: dict[str, Any]) -> dict[str, Any]:
    full = item.get("full_metrics", {})
    oos = item.get("oos_metrics", {})
    return {
        "trial_index": item.get("trial_index"),
        "alternative_score": alternative_score(item),
        "passed_strict": bool(item.get("passed")),
        "rejection_reasons": item.get("rejection_reasons", []),
        "trial": item.get("trial", {}),
        "full_metrics": full,
        "oos_metrics": oos,
        "walk_forward": item.get("walk_forward", {}),
    }


def markdown_table(candidates: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Top Alternatives / Near-miss Candidates",
        "",
        "These are not guaranteed paper-ready. They are the strongest alternatives found when strict gates fail or when you want the best nearby setup.",
        "",
        "| Rank | Trial | Monthly % | DD % | DD/monthly | OOS monthly % | OOS DD % | Trades | OOS trades | WF pass | Reasons |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(candidates[:10], start=1):
        full = item.get("full_metrics", {})
        oos = item.get("oos_metrics", {})
        wf = item.get("walk_forward", {})
        reasons = ", ".join(item.get("rejection_reasons", [])[:3]) or "passed_or_near"
        lines.append(
            f"| {rank} | {item.get('trial_index')} | "
            f"{full.get('monthly_return_pct')} | {full.get('max_drawdown_pct')} | {full.get('dd_to_monthly_ratio')} | "
            f"{oos.get('monthly_return_pct')} | {oos.get('max_drawdown_pct')} | "
            f"{full.get('trade_count')} | {oos.get('trade_count')} | {wf.get('pass_rate')} | {reasons} |"
        )

    if candidates:
        best = candidates[0]
        lines += [
            "",
            "### Best Alternative Settings",
            "",
            "```json",
            json.dumps(best.get("trial", {}), indent=2, ensure_ascii=False),
            "```",
        ]
    return lines


def main() -> None:
    trials = load_trials()
    candidates = [compact_candidate(x) for x in trials]
    candidates.sort(key=lambda x: x["alternative_score"], reverse=True)
    top = candidates[:25]
    ALTERNATIVES_JSON.write_text(json.dumps(top, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    extra = markdown_table(top)
    if REPORT_MD.exists():
        current = REPORT_MD.read_text(encoding="utf-8")
    else:
        current = "# DD < Monthly Return Optimizer Report\n\nNo base report was produced.\n"
    REPORT_MD.write_text(current.rstrip() + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    print(json.dumps({"alternatives_written": len(top), "best_trial_index": top[0].get("trial_index") if top else None}, indent=2))


if __name__ == "__main__":
    main()
