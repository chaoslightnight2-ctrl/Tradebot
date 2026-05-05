#!/usr/bin/env python3
"""Train/search Sontrade until a monthly-return and drawdown target is found.

Default target:
  monthly return > 15%
  max drawdown better than -10%
  timeout = 3 hours

Usage:
  python optimize_until_target.py
  python optimize_until_target.py --target-monthly 15 --max-dd 10 --timeout-hours 3
  python optimize_until_target.py --max-trials 50 --verify-runs 3

Outputs:
  sonayarlar.json   -> best model settings, metrics, feature groups, weights, verification backtests
  sonayarlar.env    -> copy/paste environment settings for the best model
  sontrade_model_best.joblib -> trained best model snapshot
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import joblib

import sontrade_bot as bot


OUT_JSON = Path("sonayarlar.json")
OUT_ENV = Path("sonayarlar.env")
BEST_MODEL_PATH = Path("sontrade_model_best.joblib")

FEATURE_GROUPS: dict[str, list[str]] = {
    "price_momentum": ["ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60"],
    "ema_trend": ["ema12_dist", "ema26_dist", "ema200_dist", "ema_slope_20", "trend_quality"],
    "macd": ["macd_hist_pct", "macd_line_pct", "macd_signal_pct"],
    "vwap": ["vwap_dist", "vwap_slope_5"],
    "rsi_atr_range": ["rsi14", "atr_pct", "range_pct"],
    "bollinger_band": ["bb_position", "bb_width", "bb_squeeze", "bb_upper_dist", "bb_lower_dist"],
    "gap": ["gap_pct", "gap_abs"],
    "adx_directional_movement": ["adx14", "plus_di", "minus_di", "di_spread"],
    "volume_confirmation": ["vol_z20", "relative_volume_20", "dollar_volume_z20", "obv_change_5", "volume_price_trend_5"],
    "breakout_drawdown": ["breakout20", "breakout55", "drawdown20", "proximity_high_252"],
    "opening_range": ["opening_range_breakout", "opening_range_breakdown"],
    "market_regime_spy": ["spy_ret_5", "spy_ret_20", "spy_ema200_dist", "spy_trend_up"],
    "market_regime_qqq": ["qqq_ret_5", "qqq_ret_20", "qqq_ema200_dist", "qqq_trend_up"],
    "combined_market_regime": ["market_regime_score"],
    "relative_strength": ["rel_ret_spy_5", "rel_ret_spy_20", "rel_ret_qqq_5", "rel_ret_qqq_20"],
}

CORE_GROUPS = ["price_momentum", "ema_trend", "rsi_atr_range", "volume_confirmation"]
ALL_GROUPS = list(FEATURE_GROUPS.keys())


@dataclass(frozen=True)
class Candidate:
    cfg: bot.Config
    enabled_groups: tuple[str, ...]
    features: tuple[str, ...]


def pct(x: float | None) -> float:
    if x is None:
        return 0.0
    return round(float(x) * 100.0, 4)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def calc_monthly_return(result: dict[str, Any]) -> float:
    net = safe_float(result.get("net_compounded_return", result.get("total_compounded_return", 0.0)))
    try:
        start = datetime.fromisoformat(str(result["start_date"]))
        end = datetime.fromisoformat(str(result["end_date"]))
        days = max(1, (end - start).days)
    except Exception:
        days = 365
    if net <= -0.999:
        return -0.999
    return (1.0 + net) ** (30.4375 / days) - 1.0


def trade_ratio_summary(result: dict[str, Any]) -> dict[str, Any]:
    trades = int(result.get("trades", 0) or 0)
    long_trades = int(result.get("long_trades", 0) or 0)
    short_trades = int(result.get("short_trades", 0) or 0)
    if trades <= 0:
        long_ratio = 0.0
        short_ratio = 0.0
    else:
        long_ratio = long_trades / trades
        short_ratio = short_trades / trades
    return {
        "total_trades": trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_trade_ratio_pct": pct(long_ratio),
        "short_trade_ratio_pct": pct(short_ratio),
    }


def clear_backtest_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = trade_ratio_summary(result)
    summary.update(
        {
            "status": "PASS" if int(result.get("trades", 0) or 0) > 0 else "NO_TRADES",
            "start_date": result.get("start_date"),
            "end_date": result.get("end_date"),
            "bars": result.get("bars"),
            "monthly_net_profit_pct": result.get("monthly_return_pct", pct(calc_monthly_return(result))),
            "net_compounded_profit_pct": result.get(
                "net_compounded_return_pct",
                pct(result.get("net_compounded_return", result.get("total_compounded_return", 0.0))),
            ),
            "gross_compounded_profit_pct": pct(result.get("gross_compounded_return", result.get("total_compounded_return", 0.0))),
            "win_rate_pct": pct(result.get("win_rate", 0.0)),
            "max_drawdown_pct": result.get("max_drawdown_pct", pct(result.get("max_drawdown", 0.0))),
            "signals_not_filled": int(result.get("signals_not_filled", 0) or 0),
            "avg_total_cost_pct": pct(result.get("avg_total_cost", 0.0)),
            "objective_score": result.get("objective_score"),
        }
    )
    return summary


def objective(result: dict[str, Any], target_monthly: float, max_dd: float) -> float:
    monthly = calc_monthly_return(result)
    dd = abs(safe_float(result.get("max_drawdown", 1.0), 1.0))
    trades = safe_float(result.get("trades", 0), 0.0)
    win_rate = safe_float(result.get("win_rate", 0.0), 0.0)
    short_trades = safe_float(result.get("short_trades", 0), 0.0)
    long_trades = safe_float(result.get("long_trades", 0), 0.0)

    score = monthly * 100.0
    score -= max(0.0, dd - max_dd) * 300.0
    score -= dd * 45.0
    score += min(trades, 250.0) / 250.0 * 5.0
    score += win_rate * 8.0
    score += min(short_trades + long_trades, 250.0) / 250.0 * 2.0

    if monthly >= target_monthly and dd <= max_dd:
        score += 1000.0
    if trades < 8:
        score -= 100.0
    return round(score, 6)


def target_hit(result: dict[str, Any], target_monthly: float, max_dd: float) -> bool:
    return (
        calc_monthly_return(result) >= target_monthly
        and abs(safe_float(result.get("max_drawdown", 1.0), 1.0)) <= max_dd
        and int(result.get("trades", 0) or 0) > 0
    )


def features_from_groups(enabled_groups: tuple[str, ...]) -> tuple[str, ...]:
    selected = set()
    for group in enabled_groups:
        selected.update(FEATURE_GROUPS.get(group, []))
    ordered = tuple(feature for feature in bot.FEATURES if feature in selected)
    return ordered or tuple(bot.FEATURES)


@contextlib.contextmanager
def use_feature_subset(features: tuple[str, ...]) -> Iterator[None]:
    old_features = bot.FEATURES
    bot.FEATURES = list(features)
    try:
        yield
    finally:
        bot.FEATURES = old_features


def run_train_backtest(candidate: Candidate, quiet: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    with use_feature_subset(candidate.features):
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                train_metrics = bot.train(candidate.cfg)
                bt_metrics = bot.backtest(candidate.cfg)
        else:
            train_metrics = bot.train(candidate.cfg)
            bt_metrics = bot.backtest(candidate.cfg)
    bt_metrics["monthly_return"] = calc_monthly_return(bt_metrics)
    bt_metrics["monthly_return_pct"] = pct(bt_metrics["monthly_return"])
    bt_metrics["max_drawdown_pct"] = pct(bt_metrics.get("max_drawdown"))
    bt_metrics["net_compounded_return_pct"] = pct(bt_metrics.get("net_compounded_return", bt_metrics.get("total_compounded_return", 0.0)))
    bt_metrics["trade_summary"] = trade_ratio_summary(bt_metrics)
    bt_metrics["clear_summary"] = clear_backtest_summary(bt_metrics)
    bt_metrics["objective_score"] = None
    return train_metrics, bt_metrics


def random_feature_groups(rng: random.Random, optimize_feature_groups: bool) -> tuple[str, ...]:
    if not optimize_feature_groups:
        return tuple(ALL_GROUPS)
    roll = rng.random()
    if roll < 0.18:
        return tuple(ALL_GROUPS)
    if roll < 0.30:
        return tuple(CORE_GROUPS)

    enabled = set(CORE_GROUPS)
    optional = [group for group in ALL_GROUPS if group not in enabled]
    for group in optional:
        probability = {
            "macd": 0.70,
            "vwap": 0.65,
            "bollinger_band": 0.65,
            "gap": 0.55,
            "adx_directional_movement": 0.65,
            "breakout_drawdown": 0.65,
            "opening_range": 0.35,
            "market_regime_spy": 0.70,
            "market_regime_qqq": 0.70,
            "combined_market_regime": 0.60,
            "relative_strength": 0.70,
        }.get(group, 0.55)
        if rng.random() < probability:
            enabled.add(group)
    if len(enabled) < 5:
        enabled.update(rng.sample(optional, k=min(3, len(optional))))
    return tuple(group for group in ALL_GROUPS if group in enabled)


def candidate_configs(base: bot.Config, rng: random.Random, optimize_feature_groups: bool):
    full_groups = tuple(ALL_GROUPS)
    yield Candidate(
        cfg=replace(
            base,
            trade_direction="both",
            horizon_bars=8,
            label_threshold=0.003,
            long_threshold=0.20,
            short_threshold=0.48,
            stop_atr_mult=3.2,
            tp_r=0.6,
            min_stop_pct=0.002,
            max_stop_pct=0.066,
        ),
        enabled_groups=full_groups,
        features=features_from_groups(full_groups),
    )

    horizons = [3, 5, 8, 10, 13, 16, 20]
    labels = [0.002, 0.003, 0.004, 0.006, 0.008, 0.010]
    directions = ["both", "long", "short"]

    while True:
        min_stop = rng.choice([0.001, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.008])
        max_stop = rng.choice([0.020, 0.030, 0.040, 0.050, 0.066, 0.080])
        if max_stop <= min_stop:
            max_stop = min_stop + 0.02
        groups = random_feature_groups(rng, optimize_feature_groups)
        yield Candidate(
            cfg=replace(
                base,
                trade_direction=rng.choices(directions, weights=[0.60, 0.20, 0.20], k=1)[0],
                horizon_bars=rng.choice(horizons),
                label_threshold=rng.choice(labels),
                long_threshold=round(rng.uniform(0.12, 0.72), 3),
                short_threshold=round(rng.uniform(0.12, 0.78), 3),
                min_edge_gap=round(rng.uniform(0.0, 0.12), 3),
                stop_atr_mult=round(rng.uniform(1.4, 5.2), 3),
                tp_r=round(rng.uniform(0.30, 2.20), 3),
                min_stop_pct=round(min_stop, 4),
                max_stop_pct=round(max_stop, 4),
                random_seed=rng.randint(1, 999_999),
            ),
            enabled_groups=groups,
            features=features_from_groups(groups),
        )


def feature_importance(model: Any, features: list[str]) -> dict[str, float]:
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        return {}
    total = float(sum(raw))
    if total <= 0:
        return {f: 0.0 for f in features}
    return {
        feature: round(float(value / total * 100.0), 4)
        for feature, value in zip(features, raw, strict=True)
    }


def group_importance(feature_importance_pct: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for group, features in FEATURE_GROUPS.items():
        out[group] = round(sum(feature_importance_pct.get(feature, 0.0) for feature in features), 4)
    return dict(sorted(out.items(), key=lambda item: item[1], reverse=True))


def read_current_weights() -> dict[str, Any]:
    if not bot.MODEL_PATH.exists():
        return {"error": "model file not found"}
    bundle = joblib.load(bot.MODEL_PATH)
    features = list(bundle.get("features", bot.FEATURES))
    long_imp = dict(sorted(feature_importance(bundle["long_model"], features).items(), key=lambda item: item[1], reverse=True))
    short_imp = dict(sorted(feature_importance(bundle["short_model"], features).items(), key=lambda item: item[1], reverse=True))
    return {
        "feature_count": len(features),
        "features": features,
        "long_feature_importance_pct": long_imp,
        "short_feature_importance_pct": short_imp,
        "long_group_importance_pct": group_importance(long_imp),
        "short_group_importance_pct": group_importance(short_imp),
    }


def env_text(candidate: Candidate) -> str:
    cfg = candidate.cfg
    return "\n".join(
        [
            f"ALPACA_SYMBOLS={','.join(cfg.symbols)}",
            f"SONTRADE_BENCHMARK_SYMBOLS={','.join(cfg.benchmark_symbols)}",
            f"SONTRADE_ACTIVE_FEATURE_GROUPS={','.join(candidate.enabled_groups)}",
            f"SONTRADE_ACTIVE_FEATURES={','.join(candidate.features)}",
            f"SONTRADE_DATA_PROVIDER={cfg.data_provider}",
            f"ALPACA_DATA_FEED={cfg.data_feed}",
            f"ALPACA_TIMEFRAME={cfg.timeframe}",
            f"SONTRADE_LOOKBACK_DAYS={cfg.lookback_days}",
            f"SONTRADE_TRADE_DIRECTION={cfg.trade_direction}",
            f"SONTRADE_HORIZON_BARS={cfg.horizon_bars}",
            f"SONTRADE_LABEL_THRESHOLD={cfg.label_threshold}",
            f"SONTRADE_LONG_THRESHOLD={cfg.long_threshold}",
            f"SONTRADE_SHORT_THRESHOLD={cfg.short_threshold}",
            f"SONTRADE_MIN_EDGE_GAP={cfg.min_edge_gap}",
            f"SONTRADE_STOP_ATR_MULT={cfg.stop_atr_mult}",
            f"SONTRADE_TP_R={cfg.tp_r}",
            f"SONTRADE_MIN_STOP_PCT={cfg.min_stop_pct}",
            f"SONTRADE_MAX_STOP_PCT={cfg.max_stop_pct}",
            f"SONTRADE_RISK_PER_TRADE={cfg.risk_per_trade}",
            f"SONTRADE_MAX_NOTIONAL={cfg.max_notional}",
            f"SONTRADE_MAX_POSITIONS={cfg.max_positions}",
            f"SONTRADE_MIN_ORDER_DOLLARS={cfg.min_order_dollars}",
            f"SONTRADE_COMMISSION_BPS_PER_SIDE={cfg.commission_bps_per_side}",
            f"SONTRADE_SPREAD_BPS_ROUND_TRIP={cfg.spread_bps_round_trip}",
            f"SONTRADE_SLIPPAGE_BPS_PER_SIDE={cfg.slippage_bps_per_side}",
            f"SONTRADE_SHORT_BORROW_APR={cfg.short_borrow_apr}",
            f"SONTRADE_FILL_PROBABILITY={cfg.fill_probability}",
            f"SONTRADE_MAX_ATR_PCT_FOR_FILL={cfg.max_atr_pct_for_fill}",
            f"SONTRADE_MIN_DOLLAR_VOLUME={cfg.min_dollar_volume}",
            f"SONTRADE_BARS_PER_YEAR={cfg.bars_per_year}",
            f"SONTRADE_RANDOM_SEED={cfg.random_seed}",
            "",
        ]
    )


def save_outputs(
    *,
    best_candidate: Candidate,
    best_train: dict[str, Any],
    best_backtest: dict[str, Any],
    verification: list[dict[str, Any]],
    target_monthly: float,
    max_dd: float,
    timeout_hit: bool,
    found_target: bool,
    trial_count: int,
    elapsed_sec: float,
) -> None:
    run_train_backtest(best_candidate, quiet=True)
    if bot.MODEL_PATH.exists():
        shutil.copy2(bot.MODEL_PATH, BEST_MODEL_PATH)

    weights = read_current_weights()
    verification_clear = [item["clear_summary"] for item in verification]
    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "found_target": found_target,
        "timeout_hit": timeout_hit,
        "trial_count": trial_count,
        "elapsed_sec": round(elapsed_sec, 2),
        "target": {
            "monthly_return_gt_pct": round(target_monthly * 100, 4),
            "max_drawdown_abs_lt_pct": round(max_dd * 100, 4),
        },
        "net_ozet": {
            "durum": "HEDEF_BULUNDU" if found_target else "TIMEOUT_EN_IYI_MODEL" if timeout_hit else "EN_IYI_MODEL",
            **clear_backtest_summary(best_backtest),
        },
        "aktif_analiz_gruplari": list(best_candidate.enabled_groups),
        "pasif_analiz_gruplari": [group for group in ALL_GROUPS if group not in best_candidate.enabled_groups],
        "aktif_feature_sayisi": len(best_candidate.features),
        "aktif_featureler": list(best_candidate.features),
        "best_config": asdict(best_candidate.cfg),
        "best_train_metrics": best_train,
        "best_backtest": best_backtest,
        "verification_ozetleri": verification_clear,
        "verification_backtests": verification,
        "weights": weights,
        "env_file": str(OUT_ENV),
        "best_model_file": str(BEST_MODEL_PATH),
        "notes": [
            "monthly_net_profit_pct is converted to monthly from the test period net compounded return.",
            "max_drawdown_pct is based on net returns after spread/slippage/borrow/commission settings.",
            "weights are RandomForest feature_importances_ normalized to sum to 100 separately for long and short models.",
            "SONTRADE_ACTIVE_FEATURE_GROUPS and SONTRADE_ACTIVE_FEATURES are exported for record-keeping; optimizer uses them directly during search.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_ENV.write_text(env_text(best_candidate), encoding="utf-8")


def verify_best(best_candidate: Candidate, runs: int, quiet: bool = True) -> list[dict[str, Any]]:
    out = []
    base_seed = best_candidate.cfg.random_seed
    for i in range(max(1, runs)):
        cfg = replace(best_candidate.cfg, random_seed=base_seed + i * 101)
        candidate = Candidate(cfg=cfg, enabled_groups=best_candidate.enabled_groups, features=best_candidate.features)
        _, bt = run_train_backtest(candidate, quiet=quiet)
        clear = clear_backtest_summary(bt)
        out.append(
            {
                "run": i + 1,
                "random_seed": cfg.random_seed,
                "monthly_return_pct": bt["monthly_return_pct"],
                "max_drawdown_pct": bt["max_drawdown_pct"],
                "net_compounded_return_pct": bt["net_compounded_return_pct"],
                "trades": bt.get("trades", 0),
                "long_trades": bt.get("long_trades", 0),
                "short_trades": bt.get("short_trades", 0),
                "long_trade_ratio_pct": clear["long_trade_ratio_pct"],
                "short_trade_ratio_pct": clear["short_trade_ratio_pct"],
                "win_rate_pct": pct(bt.get("win_rate", 0.0)),
                "clear_summary": clear,
                "raw": bt,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for a Sontrade model until monthly return/DD target is hit.")
    parser.add_argument("--target-monthly", type=float, default=15.0, help="Target monthly return in percent. Default: 15")
    parser.add_argument("--max-dd", type=float, default=10.0, help="Max absolute drawdown in percent. Default: 10")
    parser.add_argument("--timeout-hours", type=float, default=3.0, help="Search timeout in hours. Default: 3")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional trial cap. 0 means unlimited until timeout/target.")
    parser.add_argument("--verify-runs", type=int, default=3, help="Number of verification backtests for the best config.")
    parser.add_argument("--seed", type=int, default=42, help="Optimizer random seed.")
    parser.add_argument("--quiet", action="store_true", help="Suppress train/backtest JSON during each trial.")
    parser.add_argument("--no-feature-group-search", action="store_true", help="Disable feature-group on/off search and use all indicators.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bot.load_dotenv()
    bot.setup_logging(False)
    base_cfg = bot.load_config()

    target_monthly = args.target_monthly / 100.0
    max_dd = args.max_dd / 100.0
    timeout_sec = max(1, int(args.timeout_hours * 3600))
    deadline = time.time() + timeout_sec
    rng = random.Random(args.seed)
    optimize_feature_groups = not args.no_feature_group_search

    best_candidate: Candidate | None = None
    best_train: dict[str, Any] = {}
    best_bt: dict[str, Any] = {"objective_score": -10**9}
    found_target = False
    trial_count = 0
    start = time.time()

    print(
        f"Optimizer başladı | hedef aylık > %{args.target_monthly:.2f}, "
        f"DD < %{args.max_dd:.2f}, timeout={args.timeout_hours:.2f} saat, "
        f"feature_group_search={optimize_feature_groups}"
    )

    for candidate in candidate_configs(base_cfg, rng, optimize_feature_groups):
        if time.time() >= deadline:
            break
        if args.max_trials and trial_count >= args.max_trials:
            break

        trial_count += 1
        try:
            train_metrics, bt_metrics = run_train_backtest(candidate, quiet=args.quiet)
            bt_metrics["objective_score"] = objective(bt_metrics, target_monthly, max_dd)
            bt_metrics["clear_summary"] = clear_backtest_summary(bt_metrics)
        except Exception as exc:
            print(f"[trial {trial_count}] HATA: {exc}")
            continue

        is_best = bt_metrics["objective_score"] > safe_float(best_bt.get("objective_score"), -10**9)
        if is_best:
            best_candidate = candidate
            best_train = train_metrics
            best_bt = bt_metrics
            summary = clear_backtest_summary(bt_metrics)
            print(
                f"[trial {trial_count}] YENI EN IYI | monthly={summary['monthly_net_profit_pct']}% "
                f"dd={summary['max_drawdown_pct']}% win={summary['win_rate_pct']}% "
                f"trades={summary['total_trades']} L/S={summary['long_trades']}/{summary['short_trades']} "
                f"score={bt_metrics['objective_score']} groups={len(candidate.enabled_groups)}/{len(ALL_GROUPS)} "
                f"direction={candidate.cfg.trade_direction} long_th={candidate.cfg.long_threshold} "
                f"short_th={candidate.cfg.short_threshold} tp_r={candidate.cfg.tp_r} stop_atr={candidate.cfg.stop_atr_mult}"
            )

        if target_hit(bt_metrics, target_monthly, max_dd):
            found_target = True
            best_candidate = candidate
            best_train = train_metrics
            best_bt = bt_metrics
            print(f"[trial {trial_count}] HEDEF BULUNDU.")
            break

    if best_candidate is None:
        raise RuntimeError("Hiç geçerli backtest üretilemedi; veri/API ayarlarını kontrol et.")

    timeout_hit = not found_target and time.time() >= deadline
    print("En iyi model doğrulanıyor...")
    verification = verify_best(best_candidate, runs=args.verify_runs, quiet=args.quiet)
    elapsed = time.time() - start

    save_outputs(
        best_candidate=best_candidate,
        best_train=best_train,
        best_backtest=best_bt,
        verification=verification,
        target_monthly=target_monthly,
        max_dd=max_dd,
        timeout_hit=timeout_hit,
        found_target=found_target,
        trial_count=trial_count,
        elapsed_sec=elapsed,
    )

    final_summary = clear_backtest_summary(best_bt)
    print("\n=== SONUC ===")
    print(f"found_target={found_target} timeout_hit={timeout_hit} trials={trial_count} elapsed_sec={elapsed:.2f}")
    print(
        f"best monthly={final_summary['monthly_net_profit_pct']}% | dd={final_summary['max_drawdown_pct']}% | "
        f"net={final_summary['net_compounded_profit_pct']}% | win={final_summary['win_rate_pct']}% | "
        f"trades={final_summary['total_trades']} | long={final_summary['long_trades']} "
        f"({final_summary['long_trade_ratio_pct']}%) | short={final_summary['short_trades']} "
        f"({final_summary['short_trade_ratio_pct']}%)"
    )
    print(f"aktif analiz grupları: {', '.join(best_candidate.enabled_groups)}")
    print(f"ayar dosyası: {OUT_JSON}")
    print(f"env dosyası: {OUT_ENV}")
    print(f"model dosyası: {BEST_MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
