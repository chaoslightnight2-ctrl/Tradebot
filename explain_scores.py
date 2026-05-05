#!/usr/bin/env python3
"""Explain Sontrade scores on a 0-100 scale.

Usage:
  python explain_scores.py
  python explain_scores.py --json

This file imports sontrade_bot.py and prints:
- latest long/short score as 0-100 points
- feature importance percentages for long and short models
- grouped analysis importance percentages
- latest feature percentile points on a 0-100 scale
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

import sontrade_bot as bot


ANALYSIS_GROUPS: dict[str, list[str]] = {
    "price_momentum": [
        "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    ],
    "ema_trend": [
        "ema12_dist", "ema26_dist", "ema200_dist", "ema_slope_20", "trend_quality",
    ],
    "macd": [
        "macd_hist_pct", "macd_line_pct", "macd_signal_pct",
    ],
    "vwap": [
        "vwap_dist", "vwap_slope_5",
    ],
    "rsi_atr_range": [
        "rsi14", "atr_pct", "range_pct",
    ],
    "bollinger_band": [
        "bb_position", "bb_width", "bb_squeeze", "bb_upper_dist", "bb_lower_dist",
    ],
    "gap": [
        "gap_pct", "gap_abs",
    ],
    "adx_directional_movement": [
        "adx14", "plus_di", "minus_di", "di_spread",
    ],
    "volume_confirmation": [
        "vol_z20", "relative_volume_20", "dollar_volume_z20", "obv_change_5", "volume_price_trend_5",
    ],
    "breakout_drawdown": [
        "breakout20", "breakout55", "drawdown20", "proximity_high_252",
    ],
    "opening_range": [
        "opening_range_breakout", "opening_range_breakdown",
    ],
    "market_regime_spy": [
        "spy_ret_5", "spy_ret_20", "spy_ema200_dist", "spy_trend_up",
    ],
    "market_regime_qqq": [
        "qqq_ret_5", "qqq_ret_20", "qqq_ema200_dist", "qqq_trend_up",
    ],
    "combined_market_regime": [
        "market_regime_score",
    ],
    "relative_strength": [
        "rel_ret_spy_5", "rel_ret_spy_20", "rel_ret_qqq_5", "rel_ret_qqq_20",
    ],
}


def model_importance(model: Any, features: list[str]) -> dict[str, float]:
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        return {feature: 0.0 for feature in features}
    raw = np.asarray(raw, dtype=float)
    total = float(raw.sum())
    if total <= 0:
        return {feature: 0.0 for feature in features}
    return {
        feature: round(float(value / total * 100.0), 4)
        for feature, value in zip(features, raw, strict=True)
    }


def group_importance(feature_importance: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for group, cols in ANALYSIS_GROUPS.items():
        out[group] = round(float(sum(feature_importance.get(col, 0.0) for col in cols)), 4)
    return dict(sorted(out.items(), key=lambda item: item[1], reverse=True))


def percentile_point(series: pd.Series, value: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or not np.isfinite(value):
        return 50.0
    return round(float((clean <= value).mean() * 100.0), 2)


def latest_feature_points(history: pd.DataFrame, latest_row: pd.Series, features: list[str]) -> dict[str, float]:
    symbol = str(latest_row["symbol"])
    symbol_history = history[history["symbol"] == symbol]
    out: dict[str, float] = {}
    for feature in features:
        out[feature] = percentile_point(symbol_history[feature], float(latest_row[feature]))
    return out


def weighted_average(points: dict[str, float], weights: dict[str, float], cols: list[str]) -> float:
    w = np.array([weights.get(col, 0.0) for col in cols], dtype=float)
    p = np.array([points.get(col, 50.0) for col in cols], dtype=float)
    if w.sum() <= 0:
        return round(float(p.mean()) if len(p) else 50.0, 2)
    return round(float(np.average(p, weights=w)), 2)


def group_points(feature_points: dict[str, float], feature_importance: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for group, cols in ANALYSIS_GROUPS.items():
        out[group] = weighted_average(feature_points, feature_importance, cols)
    return dict(sorted(out.items(), key=lambda item: item[1], reverse=True))


def impact_rows(feature_importance: dict[str, float], feature_points: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for feature, impact in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True):
        rows.append(
            {
                "feature": feature,
                "impact_pct": impact,
                "latest_point_0_100": feature_points.get(feature, 50.0),
                "weighted_point": round(impact * feature_points.get(feature, 50.0) / 100.0, 4),
            }
        )
    return rows


def build_report() -> dict[str, Any]:
    bot.load_dotenv()
    cfg = bot.load_config()
    bundle = bot.load_model(cfg)
    features = list(bundle.get("features", bot.FEATURES))

    history = bot.build_dataset(cfg, require_targets=False)
    latest = bot.latest_score(cfg)

    long_imp = model_importance(bundle["long_model"], features)
    short_imp = model_importance(bundle["short_model"], features)

    rows = []
    for _, latest_row in latest.iterrows():
        feature_points = latest_feature_points(history, latest_row, features)
        rows.append(
            {
                "symbol": str(latest_row["symbol"]),
                "timestamp": str(latest_row["timestamp"]),
                "decision": str(latest_row["decision"]),
                "long_score_0_100": round(float(latest_row["long_score"]) * 100.0, 2),
                "short_score_0_100": round(float(latest_row["short_score"]) * 100.0, 2),
                "edge_gap_0_100": round(float(latest_row["edge_gap"]) * 100.0, 2),
                "long_group_points_0_100": group_points(feature_points, long_imp),
                "short_group_points_0_100": group_points(feature_points, short_imp),
                "long_top_feature_impacts": impact_rows(long_imp, feature_points)[:20],
                "short_top_feature_impacts": impact_rows(short_imp, feature_points)[:20],
            }
        )

    return {
        "feature_count": len(features),
        "features": features,
        "long_feature_importance_pct": dict(sorted(long_imp.items(), key=lambda item: item[1], reverse=True)),
        "short_feature_importance_pct": dict(sorted(short_imp.items(), key=lambda item: item[1], reverse=True)),
        "long_group_importance_pct": group_importance(long_imp),
        "short_group_importance_pct": group_importance(short_imp),
        "latest_scores": rows,
        "note": "impact_pct is global RandomForest feature importance normalized to sum 100. latest_point_0_100 is the latest feature percentile versus that symbol's history, not a live-profit guarantee.",
    }


def print_report(report: dict[str, Any]) -> None:
    print("\n=== LATEST 0-100 SCORES ===")
    for row in report["latest_scores"]:
        print(
            f"{row['symbol']} | {row['timestamp']} | decision={row['decision']} | "
            f"long={row['long_score_0_100']} | short={row['short_score_0_100']} | edge={row['edge_gap_0_100']}"
        )

    print("\n=== LONG GROUP IMPACT % (SUM=100) ===")
    for group, value in report["long_group_importance_pct"].items():
        print(f"{group:30s} {value:7.2f}%")

    print("\n=== SHORT GROUP IMPACT % (SUM=100) ===")
    for group, value in report["short_group_importance_pct"].items():
        print(f"{group:30s} {value:7.2f}%")

    for row in report["latest_scores"]:
        print(f"\n=== {row['symbol']} LONG TOP FEATURE IMPACTS ===")
        for item in row["long_top_feature_impacts"]:
            print(
                f"{item['feature']:28s} impact={item['impact_pct']:6.2f}% "
                f"point={item['latest_point_0_100']:6.2f} weighted={item['weighted_point']:6.3f}"
            )
        print(f"\n=== {row['symbol']} SHORT TOP FEATURE IMPACTS ===")
        for item in row["short_top_feature_impacts"]:
            print(
                f"{item['feature']:28s} impact={item['impact_pct']:6.2f}% "
                f"point={item['latest_point_0_100']:6.2f} weighted={item['weighted_point']:6.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show Sontrade 0-100 scores and model feature impacts.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
