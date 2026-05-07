#!/usr/bin/env python3
"""Append near-miss alternatives and diagnostics to optimizer reports."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
TRIALS_JSONL = REPORT_DIR / "dd_lt_monthly_optimizer_trials.jsonl"
REPORT_MD = REPORT_DIR / "DD_LT_MONTHLY_OPTIMIZER_REPORT.md"
ALTERNATIVES_JSON = REPORT_DIR / "optimizer_top_alternatives.json"
ERROR_SUMMARY_JSON = REPORT_DIR / "optimizer_error_summary.json"


def sf(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def load_rows() -> list[dict[str, Any]]:
    if not TRIALS_JSONL.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in TRIALS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"error": "json_decode_error", "raw": line[:500]})
    return rows


def metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [x for x in rows if "full_metrics" in x and "oos_metrics" in x]


def alt_score(item: dict[str, Any]) -> float:
    full = item.get("full_metrics", {})
    oos = item.get("oos_metrics", {})
    wf = item.get("walk_forward", {})
    trial = item.get("trial", {})
    fm = sf(full.get("monthly_return_pct"))
    om = sf(oos.get("monthly_return_pct"))
    fdd = abs(sf(full.get("max_drawdown_pct")))
    odd = abs(sf(oos.get("max_drawdown_pct")))
    ft = sf(full.get("trade_count"))
    ot = sf(oos.get("trade_count"))
    wf_pass = sf(wf.get("pass_rate"))
    target_gap = abs(fm - 15.0)
    score = fm * 20 + om * 8 - fdd * 25 - odd * 10 - target_gap * 12
    score += wf_pass * 100 + min(ft, 200) * 0.15 + min(ot, 80) * 0.30
    if 12 <= fm <= 20:
        score += 200
    if om >= 12:
        score += 120
    if fm > 0 and fdd < fm:
        score += 150
    if om > 0 and odd < om:
        score += 75
    score -= sf(trial.get("max_positions"), 1) * 2
    score -= sf(trial.get("risk_per_trade"), 0) * 200
    return round(score, 6)


def compact(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_index": item.get("trial_index"),
        "alternative_score": alt_score(item),
        "passed_strict": bool(item.get("passed")),
        "rejection_reasons": item.get("rejection_reasons", []),
        "trial": item.get("trial", {}),
        "full_metrics": item.get("full_metrics", {}),
        "oos_metrics": item.get("oos_metrics", {}),
        "walk_forward": item.get("walk_forward", {}),
    }


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    err = Counter(str(x.get("error")) for x in rows if x.get("error"))
    reasons: Counter[str] = Counter()
    for item in rows:
        for r in item.get("rejection_reasons", []) or []:
            reasons[str(r)] += 1
    return {
        "trials_file_exists": TRIALS_JSONL.exists(),
        "total_rows": len(rows),
        "metric_rows": len(metric_rows(rows)),
        "error_rows": sum(err.values()),
        "top_errors": err.most_common(10),
        "top_rejection_reasons": reasons.most_common(15),
        "sample_error_rows": [x for x in rows if x.get("error")][:5],
    }


def md(top: list[dict[str, Any]], s: dict[str, Any]) -> str:
    lines = ["", "## Top Alternatives / Near-miss Candidates", ""]
    if not top:
        lines += [
            "No metric-bearing alternatives were available.",
            "",
            "### Optimizer Diagnostics",
            "",
            f"- Trials file exists: `{s['trials_file_exists']}`",
            f"- Total rows: `{s['total_rows']}`",
            f"- Rows with full/OOS metrics: `{s['metric_rows']}`",
            f"- Error rows: `{s['error_rows']}`",
            "",
            "#### Top errors",
        ]
        lines += [f"- `{e}`: {c}" for e, c in s["top_errors"]] or ["- None captured; run likely stopped before first completed trial."]
        lines += ["", "#### Top rejection reasons"]
        lines += [f"- `{r}`: {c}" for r, c in s["top_rejection_reasons"]] or ["- None captured."]
        if s["sample_error_rows"]:
            lines += ["", "#### Sample error rows", "", "```json", json.dumps(s["sample_error_rows"], indent=2, ensure_ascii=False, default=str), "```"]
        return "\n".join(lines) + "\n"

    lines += [
        "These are not guaranteed paper-ready; they are the strongest nearby alternatives.",
        "",
        "| Rank | Trial | Monthly % | DD % | DD/monthly | OOS monthly % | OOS DD % | Trades | OOS trades | WF pass | Reasons |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(top[:10], 1):
        f = item.get("full_metrics", {})
        o = item.get("oos_metrics", {})
        w = item.get("walk_forward", {})
        reasons = ", ".join(item.get("rejection_reasons", [])[:3]) or "passed_or_near"
        lines.append(f"| {rank} | {item.get('trial_index')} | {f.get('monthly_return_pct')} | {f.get('max_drawdown_pct')} | {f.get('dd_to_monthly_ratio')} | {o.get('monthly_return_pct')} | {o.get('max_drawdown_pct')} | {f.get('trade_count')} | {o.get('trade_count')} | {w.get('pass_rate')} | {reasons} |")
    lines += ["", "### Best Alternative Settings", "", "```json", json.dumps(top[0].get("trial", {}), indent=2, ensure_ascii=False), "```"]
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    s = summary(rows)
    top = [compact(x) for x in metric_rows(rows)]
    top.sort(key=lambda x: x["alternative_score"], reverse=True)
    top = top[:25]
    ALTERNATIVES_JSON.write_text(json.dumps(top, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    ERROR_SUMMARY_JSON.write_text(json.dumps(s, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    base = REPORT_MD.read_text(encoding="utf-8") if REPORT_MD.exists() else "# DD < Monthly Return Optimizer Report\n\nNo base report was produced.\n"
    REPORT_MD.write_text(base.rstrip() + "\n" + md(top, s), encoding="utf-8")
    print(json.dumps({"alternatives_written": len(top), "best_trial_index": top[0].get("trial_index") if top else None, **s}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
