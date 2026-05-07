#!/usr/bin/env python3
"""Robust OPEN both-direction optimizer.

Runs random/grid search for OPEN with trade_direction='both' and validates every
candidate across 1Y, 3Y and 5Y windows. It writes Top 3 valid candidates plus
near misses to Markdown and JSON reports.

This script runs backtests only. It never submits paper/live orders.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
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
    score_frame,
    setup_logging,
    should_fill,
)

REPORT_DIR = ROOT / "reports"
REPORT_MD = REPORT_DIR / "OPEN_BOTH_TOP3_OPTIMIZER_REPORT.md"
REPORT_JSON = REPORT_DIR / "OPEN_BOTH_TOP3_OPTIMIZER_REPORT.json"
NEAR_MISS_JSON = REPORT_DIR / "OPEN_BOTH_TOP3_NEAR_MISSES.json"

WINDOW_DAYS = {"1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5}
MIN_TRADES = {"1Y": 30, "3Y": 80, "5Y": 120}


def sf(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def pct(x: float) -> float:
    return round(float(x) * 100.0, 6)


def total_cost_value(costs: dict[str, Any]) -> float:
    if "total_cost" in costs:
        return sf(costs.get("total_cost"))
    if "total" in costs:
        return sf(costs.get("total"))
    if "total_bps" in costs:
        return sf(costs.get("total_bps")) / 10_000.0
    return sum(sf(costs.get(k)) for k in ("commission", "spread", "slippage", "borrow", "borrow_cost"))


def candidate_space(trials: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    space = {
        "trade_direction": ["both"],
        "horizon_bars": [2, 3, 4, 5, 8, 12],
        "label_threshold": [0.002, 0.003, 0.004, 0.006, 0.008, 0.01],
        "long_threshold": [0.18, 0.20, 0.22, 0.25, 0.28, 0.32, 0.36],
        "short_threshold": [0.42, 0.45, 0.48, 0.52, 0.56, 0.60],
        "min_edge_gap": [0.02, 0.03, 0.04, 0.05, 0.07],
        "stop_atr_mult": [1.8, 2.2, 2.6, 3.0, 3.4, 4.0],
        "tp_r": [0.5, 0.7, 1.0, 1.3, 1.6, 2.0],
        "min_stop_pct": [0.002, 0.004, 0.006, 0.008],
        "max_stop_pct": [0.018, 0.025, 0.035, 0.05],
        "risk_per_trade": [0.0025, 0.005, 0.0075, 0.01],
        "max_positions": [1, 2, 3],
        "cooldown_bars": [0, 2, 4, 8],
        "max_daily_trades": [1, 2, 3, 5],
    }
    keys = list(space)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(out) < trials:
        item = {k: rng.choice(space[k]) for k in keys}
        if item["min_stop_pct"] >= item["max_stop_pct"]:
            continue
        sig = json.dumps(item, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)
    return out


def build_config(args: argparse.Namespace, params: dict[str, Any] | None = None) -> Config:
    params = params or {}
    return Config(
        symbols=(args.symbol.upper(),),
        benchmark_symbols=("SPY", "QQQ"),
        timeframe=args.timeframe,
        lookback_days=max(args.lookback_days, 365 * 5 + 120),
        data_provider=args.data_provider,
        data_feed=args.data_feed,
        bars_per_year=args.bars_per_year,
        random_seed=args.seed,
        trade_direction="both",
        horizon_bars=int(params.get("horizon_bars", 8)),
        label_threshold=float(params.get("label_threshold", 0.003)),
        long_threshold=float(params.get("long_threshold", 0.20)),
        short_threshold=float(params.get("short_threshold", 0.48)),
        min_edge_gap=float(params.get("min_edge_gap", 0.03)),
        stop_atr_mult=float(params.get("stop_atr_mult", 3.2)),
        tp_r=float(params.get("tp_r", 0.6)),
        min_stop_pct=float(params.get("min_stop_pct", 0.002)),
        max_stop_pct=float(params.get("max_stop_pct", 0.05)),
        risk_per_trade=float(params.get("risk_per_trade", 0.005)),
        max_positions=int(params.get("max_positions", 1)),
        commission_bps_per_side=args.commission_bps_per_side,
        spread_bps_round_trip=args.spread_bps_round_trip,
        slippage_bps_per_side=args.slippage_bps_per_side,
        short_borrow_apr=args.short_borrow_apr,
        fill_probability=args.fill_probability,
        min_dollar_volume=args.min_dollar_volume,
    )


def filter_window(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    end = pd.to_datetime(df["timestamp"]).max()
    start = end - pd.Timedelta(days=WINDOW_DAYS[label])
    return df[pd.to_datetime(df["timestamp"]) >= start].copy().reset_index(drop=True)


def split_time(df: pd.DataFrame, frac: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = sorted(pd.to_datetime(df["timestamp"]).unique())
    if len(times) < 3:
        return df.iloc[:0].copy(), df.iloc[:0].copy()
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


def realized_return_reason(row: pd.Series, side: str, cfg: Config) -> tuple[float, str, int]:
    entry = sf(row["close"])
    atr_pct = sf(row.get("atr_pct"))
    stop_pct = min(cfg.max_stop_pct, max(cfg.min_stop_pct, atr_pct * cfg.stop_atr_mult))
    tp_pct = stop_pct * cfg.tp_r
    hi = sf(row.get("future_high_max"), float("nan"))
    lo = sf(row.get("future_low_min"), float("nan"))
    fallback = sf(row.get("future_return")) if side == "long" else -sf(row.get("future_return"))
    if not np.isfinite(hi) or not np.isfinite(lo) or entry <= 0:
        return fallback, "time_stop", cfg.horizon_bars
    if side == "long":
        stop_hit = lo <= entry * (1 - stop_pct)
        tp_hit = hi >= entry * (1 + tp_pct)
    else:
        stop_hit = hi >= entry * (1 + stop_pct)
        tp_hit = lo <= entry * (1 - tp_pct)
    if stop_hit and tp_hit:
        return -stop_pct, "stop_loss", max(1, cfg.horizon_bars // 2)
    if stop_hit:
        return -stop_pct, "stop_loss", max(1, cfg.horizon_bars // 2)
    if tp_hit:
        return tp_pct, "take_profit", max(1, cfg.horizon_bars // 2)
    return fallback, "time_stop", cfg.horizon_bars


def select_trades(scored: pd.DataFrame, cfg: Config, params: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cooldown_bars = int(params.get("cooldown_bars", 0))
    max_daily_trades = int(params.get("max_daily_trades", 999999))
    cooldown_left = 0
    day_counts: dict[str, int] = {}
    for _, row in scored.sort_values("timestamp").iterrows():
        if cooldown_left > 0:
            cooldown_left -= 1
            continue
        day_key = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d")
        if day_counts.get(day_key, 0) >= max_daily_trades:
            continue
        long_score = sf(row.get("long_score"))
        short_score = sf(row.get("short_score"))
        edge_gap = abs(long_score - short_score)
        side: str | None = None
        if long_score >= cfg.long_threshold and edge_gap >= cfg.min_edge_gap:
            side = "long"
        if short_score >= cfg.short_threshold and edge_gap >= cfg.min_edge_gap:
            if side is None or short_score > long_score:
                side = "short"
        if side is None or not should_fill(row, side, cfg):
            continue
        gross, exit_reason, hold_bars = realized_return_reason(row, side, cfg)
        net = gross - total_cost_value(backtest_costs(row, side, cfg))
        atr_stop = sf(row.get("atr_pct")) * cfg.stop_atr_mult
        stop_pct = max(cfg.min_stop_pct, min(cfg.max_stop_pct, atr_stop))
        position_return = net * cfg.risk_per_trade / max(stop_pct, 1e-9)
        rows.append({
            "timestamp": str(row["timestamp"]),
            "symbol": row.get("symbol"),
            "side": side,
            "gross_return": gross,
            "net_return": net,
            "position_return": position_return,
            "exit_reason": exit_reason,
            "hold_bars": hold_bars,
            "long_score": long_score,
            "short_score": short_score,
        })
        day_counts[day_key] = day_counts.get(day_key, 0) + 1
        cooldown_left = cooldown_bars
    return pd.DataFrame(rows)


def metrics_from_trades(trades: pd.DataFrame, start_ts: Any, end_ts: Any) -> dict[str, Any]:
    if trades.empty or start_ts is None or end_ts is None:
        return {
            "monthly_return_pct": 0.0, "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
            "dd_to_monthly_ratio": None, "win_rate_pct": 0.0, "profit_factor": 0.0,
            "trade_count": 0, "long_trade_count": 0, "short_trade_count": 0,
            "avg_trade_return_pct": 0.0, "median_trade_return_pct": 0.0, "avg_hold_bars": 0.0,
            "stop_loss_exit_count": 0, "take_profit_exit_count": 0, "time_stop_exit_count": 0,
        }
    r = trades["position_return"].astype(float).clip(lower=-0.95)
    equity = (1.0 + r).cumprod()
    dd = ((equity / equity.cummax()) - 1.0).min()
    total = equity.iloc[-1] - 1.0
    months = max(1.0 / 30.4375, (pd.to_datetime(end_ts) - pd.to_datetime(start_ts)).days / 30.4375)
    monthly = (1.0 + total) ** (1.0 / months) - 1.0 if total > -0.999 else -1.0
    wins = r[r > 0]
    losses = r[r < 0]
    monthly_pct = pct(monthly)
    dd_pct = pct(dd)
    exits = trades["exit_reason"].value_counts().to_dict() if "exit_reason" in trades else {}
    return {
        "monthly_return_pct": monthly_pct,
        "total_return_pct": pct(total),
        "max_drawdown_pct": dd_pct,
        "dd_to_monthly_ratio": round(abs(dd_pct) / monthly_pct, 6) if monthly_pct > 0 else None,
        "win_rate_pct": round(float((r > 0).mean() * 100.0), 6),
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 6) if abs(losses.sum()) > 0 else None,
        "trade_count": int(len(trades)),
        "long_trade_count": int((trades["side"] == "long").sum()),
        "short_trade_count": int((trades["side"] == "short").sum()),
        "avg_trade_return_pct": pct(r.mean()),
        "median_trade_return_pct": pct(r.median()),
        "avg_hold_bars": round(float(trades.get("hold_bars", pd.Series([0])).mean()), 4),
        "stop_loss_exit_count": int(exits.get("stop_loss", 0)),
        "take_profit_exit_count": int(exits.get("take_profit", 0)),
        "time_stop_exit_count": int(exits.get("time_stop", 0)),
    }


def evaluate_period(df: pd.DataFrame, cfg: Config, params: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    train_df, oos_df = split_time(df)
    if len(train_df) < 120 or len(oos_df) < 20:
        raise RuntimeError("not_enough_rows")
    bundle = train_bundle(train_df, cfg)
    scored_full = score_frame(df, bundle)
    trades_full = select_trades(scored_full, cfg, params)
    metrics = metrics_from_trades(trades_full, df["timestamp"].min(), df["timestamp"].max())
    scored_oos = score_frame(oos_df, bundle)
    trades_oos = select_trades(scored_oos, cfg, params)
    oos = metrics_from_trades(trades_oos, oos_df["timestamp"].min(), oos_df["timestamp"].max())
    metrics["oos_monthly_return_pct"] = oos["monthly_return_pct"]
    metrics["oos_max_drawdown_pct"] = oos["max_drawdown_pct"]
    return metrics, trades_full


def walk_forward_pass_rate(df: pd.DataFrame, cfg: Config, params: dict[str, Any], max_dd: float) -> float:
    times = sorted(pd.to_datetime(df["timestamp"]).unique())
    train_bars = 252 if cfg.bars_per_year <= 300 else min(1000, max(252, cfg.bars_per_year // 2))
    test_bars = 63 if cfg.bars_per_year <= 300 else min(240, max(63, cfg.bars_per_year // 8))
    if len(times) < train_bars + test_bars:
        return 0.0
    passed = 0
    total = 0
    step = test_bars
    for start in range(0, len(times) - train_bars - test_bars + 1, step):
        tr0, tr1 = times[start], times[start + train_bars - 1]
        te0, te1 = times[start + train_bars], times[start + train_bars + test_bars - 1]
        tr = df[(pd.to_datetime(df["timestamp"]) >= tr0) & (pd.to_datetime(df["timestamp"]) <= tr1)]
        te = df[(pd.to_datetime(df["timestamp"]) >= te0) & (pd.to_datetime(df["timestamp"]) <= te1)]
        if len(tr) < 120 or len(te) < 20:
            continue
        bundle = train_bundle(tr, cfg)
        trades = select_trades(score_frame(te, bundle), cfg, params)
        m = metrics_from_trades(trades, te["timestamp"].min(), te["timestamp"].max())
        total += 1
        if m["monthly_return_pct"] > 0 and abs(m["max_drawdown_pct"]) < max_dd:
            passed += 1
    return round(100.0 * passed / total, 6) if total else 0.0


def overtrade_penalty(period_results: dict[str, dict[str, Any]]) -> float:
    penalty = 0.0
    limits = {"1Y": 700, "3Y": 1800, "5Y": 3000}
    for label, limit in limits.items():
        count = sf(period_results.get(label, {}).get("trade_count"))
        if count > limit:
            penalty += (count - limit) / limit
    return round(penalty, 6)


def check_hard_gates(candidate: dict[str, Any], max_dd: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    periods = candidate.get("period_results", {})
    for label in ("1Y", "3Y", "5Y"):
        m = periods.get(label, {})
        if abs(sf(m.get("max_drawdown_pct"))) >= max_dd:
            reasons.append(f"{label}_dd_gte_{max_dd}")
        if sf(m.get("total_return_pct")) <= 0:
            reasons.append(f"{label}_total_not_positive")
        if sf(m.get("monthly_return_pct")) <= 0:
            reasons.append(f"{label}_monthly_not_positive")
        if sf(m.get("trade_count")) < MIN_TRADES[label]:
            reasons.append(f"{label}_trade_count_below_{MIN_TRADES[label]}")
    if sf(candidate.get("oos_monthly_return_pct")) <= 0:
        reasons.append("oos_monthly_not_positive")
    return not reasons, reasons


def compute_score(candidate: dict[str, Any]) -> float:
    periods = candidate["period_results"]
    monthly = [sf(periods[p]["monthly_return_pct"]) for p in ("1Y", "3Y", "5Y")]
    worst_dd = max(abs(sf(periods[p]["max_drawdown_pct"])) for p in ("1Y", "3Y", "5Y"))
    trade_count = max(1.0, sf(periods["5Y"].get("trade_count")))
    long_count = sf(periods["5Y"].get("long_trade_count"))
    short_count = sf(periods["5Y"].get("short_trade_count"))
    balance_penalty = abs(long_count - short_count) / trade_count
    penalty = sf(candidate.get("overtrade_penalty"))
    score = (
        4.0 * median(monthly)
        + 2.0 * min(monthly)
        + 1.5 * sf(candidate.get("oos_monthly_return_pct"))
        + 1.0 * sf(candidate.get("walk_forward_monthly_pass_rate_pct"))
        - 3.0 * worst_dd
        - 1.0 * balance_penalty
        - 0.5 * penalty
    )
    if all(m > abs(sf(periods[p]["max_drawdown_pct"])) for m, p in zip(monthly, ("1Y", "3Y", "5Y"))):
        score += 10.0
    return round(float(score), 6)


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda x: sf(x.get("score")), reverse=True)


def evaluate_candidate(raw: pd.DataFrame, args: argparse.Namespace, params: dict[str, Any]) -> dict[str, Any]:
    cfg = build_config(args, params)
    featured = add_features(raw, cfg, require_targets=True)
    period_results: dict[str, dict[str, Any]] = {}
    all_trades: dict[str, int] = {}
    for label in ("1Y", "3Y", "5Y"):
        wdf = filter_window(featured, label)
        metrics, trades = evaluate_period(wdf, cfg, params)
        period_results[label] = metrics
        all_trades[label] = int(len(trades))
    wf_rate = walk_forward_pass_rate(filter_window(featured, "5Y"), cfg, params, args.max_dd)
    candidate = {
        "settings": params,
        "period_results": period_results,
        "oos_monthly_return_pct": period_results["5Y"].get("oos_monthly_return_pct", 0.0),
        "oos_max_drawdown_pct": period_results["5Y"].get("oos_max_drawdown_pct", 0.0),
        "walk_forward_monthly_pass_rate_pct": wf_rate,
        "overtrade_penalty": overtrade_penalty(period_results),
    }
    ok, reasons = check_hard_gates(candidate, args.max_dd)
    candidate["passed"] = ok
    candidate["rejection_reasons"] = reasons
    candidate["score"] = compute_score(candidate) if ok else near_miss_score(candidate)
    return candidate


def near_miss_score(candidate: dict[str, Any]) -> float:
    periods = candidate.get("period_results", {})
    if not periods:
        return -999999.0
    monthly = [sf(periods.get(p, {}).get("monthly_return_pct")) for p in ("1Y", "3Y", "5Y")]
    worst_dd = max(abs(sf(periods.get(p, {}).get("max_drawdown_pct"))) for p in ("1Y", "3Y", "5Y"))
    reasons = len(candidate.get("rejection_reasons", []))
    return round(2 * median(monthly) + min(monthly) - 2 * worst_dd - 5 * reasons + sf(candidate.get("walk_forward_monthly_pass_rate_pct")) * 0.2, 6)


def aggregate_top_row(candidate: dict[str, Any]) -> dict[str, Any]:
    periods = candidate["period_results"]
    monthly = [sf(periods[p]["monthly_return_pct"]) for p in ("1Y", "3Y", "5Y")]
    worst_dd = max(abs(sf(periods[p]["max_drawdown_pct"])) for p in ("1Y", "3Y", "5Y"))
    return {
        "score": candidate["score"],
        "monthly_median_pct": round(float(median(monthly)), 6),
        "monthly_min_pct": round(float(min(monthly)), 6),
        "worst_dd_pct": round(worst_dd, 6),
        "dd_to_monthly_ratio": round(worst_dd / median(monthly), 6) if median(monthly) > 0 else None,
        "oos_monthly_return_pct": candidate.get("oos_monthly_return_pct"),
        "oos_max_drawdown_pct": candidate.get("oos_max_drawdown_pct"),
        "walk_forward_monthly_pass_rate_pct": candidate.get("walk_forward_monthly_pass_rate_pct"),
        "trades_1y_3y_5y": [periods[p]["trade_count"] for p in ("1Y", "3Y", "5Y")],
        "long_short_5y": [periods["5Y"]["long_trade_count"], periods["5Y"]["short_trade_count"]],
        "win_rate_5y_pct": periods["5Y"]["win_rate_pct"],
        "profit_factor_5y": periods["5Y"]["profit_factor"],
    }


def write_reports(valid: list[dict[str, Any]], near_misses: list[dict[str, Any]], args: argparse.Namespace) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    valid = rank_candidates(valid)[: args.top]
    near_misses = rank_candidates(near_misses)[:10]
    payload = {
        "summary": {
            "symbol": args.symbol.upper(),
            "direction": args.direction,
            "max_dd_gate": f"abs(max_drawdown_pct) < {args.max_dd}",
            "backtest_windows": ["1Y", "3Y", "5Y"],
            "best_valid_candidates_found": len(valid),
            "created_at": datetime.now(UTC).isoformat(),
        },
        "top_candidates": valid,
        "near_misses": near_misses,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    NEAR_MISS_JSON.write_text(json.dumps(near_misses, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    lines = [
        "# OPEN Both Direction Optimizer Report",
        "",
        "## Summary",
        f"- Symbol: {args.symbol.upper()}",
        f"- Direction: {args.direction}",
        f"- Max DD gate: abs(max_drawdown_pct) < {args.max_dd}",
        "- Backtest windows: 1Y / 3Y / 5Y",
        f"- Best valid candidates found: {len(valid)}",
    ]
    if not valid:
        lines += ["", "NO_VALID_BOT_FOUND", "", "No candidate passed every hard gate. See Near Misses for closest alternatives."]
    lines += [
        "", "## Top 3 Valid Settings", "",
        "| Rank | Score | Monthly % median | Worst DD % | DD/monthly | OOS monthly % | WF pass % | Trades 1Y/3Y/5Y | Long/Short trades | Win rate | PF |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for rank, c in enumerate(valid, 1):
        row = aggregate_top_row(c)
        lines.append(
            f"| {rank} | {row['score']} | {row['monthly_median_pct']} | {row['worst_dd_pct']} | {row['dd_to_monthly_ratio']} | "
            f"{row['oos_monthly_return_pct']} | {row['walk_forward_monthly_pass_rate_pct']} | {row['trades_1y_3y_5y']} | "
            f"{row['long_short_5y']} | {row['win_rate_5y_pct']} | {row['profit_factor_5y']} |"
        )
    lines += ["", "## Candidate Details"]
    for rank, c in enumerate(valid, 1):
        lines += [
            "", f"### Rank {rank}", "", "#### Settings", "", "```json",
            json.dumps(c["settings"], indent=2, ensure_ascii=False), "```", "",
            "#### Period Results", "", "```json",
            json.dumps(c["period_results"], indent=2, ensure_ascii=False, default=str), "```", "",
            "#### OOS / Walk-forward", "",
            f"- OOS monthly %: {c.get('oos_monthly_return_pct')}",
            f"- OOS DD %: {c.get('oos_max_drawdown_pct')}",
            f"- Walk-forward pass %: {c.get('walk_forward_monthly_pass_rate_pct')}",
            "", "#### Why selected", "",
            "Passed all hard gates, then ranked by monthly consistency, OOS result, walk-forward pass rate, DD control, trade balance and overtrade penalty.",
        ]
    lines += ["", "## Near Misses", "", "| Rank | Score | Reasons | Worst DD | Monthly median | OOS monthly | Trades 1Y/3Y/5Y |", "|---:|---:|---|---:|---:|---:|---|"]
    for rank, c in enumerate(near_misses, 1):
        row = aggregate_top_row(c)
        lines.append(f"| {rank} | {c['score']} | {', '.join(c.get('rejection_reasons', [])[:4])} | {row['worst_dd_pct']} | {row['monthly_median_pct']} | {row['oos_monthly_return_pct']} | {row['trades_1y_3y_5y']} |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.symbol.upper() != "OPEN":
        raise ValueError("This optimizer is intentionally restricted to --symbol OPEN")
    if args.direction != "both":
        raise ValueError("This optimizer requires --direction both")
    cfg = build_config(args, {})
    raw = fetch_bars(cfg)
    valid: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    for i, params in enumerate(candidate_space(args.trials, args.seed), 1):
        try:
            cand = evaluate_candidate(raw, args, params)
            cand["trial_index"] = i
            if cand["passed"]:
                valid.append(cand)
            else:
                near.append(cand)
        except Exception as exc:
            near.append({"trial_index": i, "settings": params, "passed": False, "score": -999999.0, "rejection_reasons": [repr(exc)], "period_results": {}})
    write_reports(valid, near, args)
    result = {"valid_count": len(valid), "near_miss_count": len(near), "report_md": str(REPORT_MD), "report_json": str(REPORT_JSON)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="OPEN")
    p.add_argument("--windows", default="1y,3y,5y")
    p.add_argument("--direction", default="both", choices=["both"])
    p.add_argument("--max-dd", type=float, default=20.0)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--timeframe", default="1Day")
    p.add_argument("--data-provider", default="yfinance", choices=["yfinance", "alpaca"])
    p.add_argument("--data-feed", default="iex")
    p.add_argument("--lookback-days", type=int, default=365 * 5 + 160)
    p.add_argument("--bars-per-year", type=int, default=252)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--commission-bps-per-side", type=float, default=0.0)
    p.add_argument("--spread-bps-round-trip", type=float, default=2.0)
    p.add_argument("--slippage-bps-per-side", type=float, default=1.0)
    p.add_argument("--short-borrow-apr", type=float, default=0.03)
    p.add_argument("--fill-probability", type=float, default=0.98)
    p.add_argument("--min-dollar-volume", type=float, default=50_000.0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    run(args)
