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
  sonayarlar.json   -> best model settings, metrics, feature weights, verification backtests
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
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

import sontrade_bot as bot


OUT_JSON = Path("sonayarlar.json")
OUT_ENV = Path("sonayarlar.env")
BEST_MODEL_PATH = Path("sontrade_model_best.joblib")


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


def objective(result: dict[str, Any], target_monthly: float, max_dd: float) -> float:
    monthly = calc_monthly_return(result)
    dd = abs(safe_float(result.get("max_drawdown", 1.0), 1.0))
    trades = safe_float(result.get("trades", 0), 0.0)
    win_rate = safe_float(result.get("win_rate", 0.0), 0.0)

    score = monthly * 100.0
    score -= max(0.0, dd - max_dd) * 300.0
    score -= dd * 45.0
    score += min(trades, 250.0) / 250.0 * 5.0
    score += win_rate * 8.0

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


def run_train_backtest(cfg: bot.Config, quiet: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            train_metrics = bot.train(cfg)
            bt_metrics = bot.backtest(cfg)
    else:
        train_metrics = bot.train(cfg)
        bt_metrics = bot.backtest(cfg)
    bt_metrics["monthly_return"] = calc_monthly_return(bt_metrics)
    bt_metrics["monthly_return_pct"] = pct(bt_metrics["monthly_return"])
    bt_metrics["max_drawdown_pct"] = pct(bt_metrics.get("max_drawdown"))
    bt_metrics["net_compounded_return_pct"] = pct(bt_metrics.get("net_compounded_return", bt_metrics.get("total_compounded_return", 0.0)))
    bt_metrics["objective_score"] = None
    return train_metrics, bt_metrics


def candidate_configs(base: bot.Config, rng: random.Random):
    yield replace(
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
    )

    horizons = [3, 5, 8, 10, 13, 16, 20]
    labels = [0.002, 0.003, 0.004, 0.006, 0.008, 0.010]
    directions = ["both", "long", "short"]

    while True:
        min_stop = rng.choice([0.001, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.008])
        max_stop = rng.choice([0.020, 0.030, 0.040, 0.050, 0.066, 0.080])
        if max_stop <= min_stop:
            max_stop = min_stop + 0.02

        yield replace(
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


def read_current_weights() -> dict[str, Any]:
    if not bot.MODEL_PATH.exists():
        return {"error": "model file not found"}
    bundle = joblib.load(bot.MODEL_PATH)
    features = list(bundle.get("features", bot.FEATURES))
    return {
        "feature_count": len(features),
        "features": features,
        "long_feature_importance_pct": dict(
            sorted(feature_importance(bundle["long_model"], features).items(), key=lambda item: item[1], reverse=True)
        ),
        "short_feature_importance_pct": dict(
            sorted(feature_importance(bundle["short_model"], features).items(), key=lambda item: item[1], reverse=True)
        ),
    }


def env_text(cfg: bot.Config) -> str:
    return "\n".join(
        [
            f"ALPACA_SYMBOLS={','.join(cfg.symbols)}",
            f"SONTRADE_BENCHMARK_SYMBOLS={','.join(cfg.benchmark_symbols)}",
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
    best_cfg: bot.Config,
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
    run_train_backtest(best_cfg, quiet=True)
    if bot.MODEL_PATH.exists():
        shutil.copy2(bot.MODEL_PATH, BEST_MODEL_PATH)

    weights = read_current_weights()
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
        "best_config": asdict(best_cfg),
        "best_train_metrics": best_train,
        "best_backtest": best_backtest,
        "verification_backtests": verification,
        "weights": weights,
        "env_file": str(OUT_ENV),
        "best_model_file": str(BEST_MODEL_PATH),
        "notes": [
            "monthly_return_pct is converted to monthly from the test period net compounded return.",
            "max_drawdown_pct is based on net returns after spread/slippage/borrow/commission settings.",
            "weights are RandomForest feature_importances_ normalized to sum to 100 separately for long and short models.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_ENV.write_text(env_text(best_cfg), encoding="utf-8")


def verify_best(best_cfg: bot.Config, runs: int, quiet: bool = True) -> list[dict[str, Any]]:
    out = []
    base_seed = best_cfg.random_seed
    for i in range(max(1, runs)):
        cfg = replace(best_cfg, random_seed=base_seed + i * 101)
        _, bt = run_train_backtest(cfg, quiet=quiet)
        out.append(
            {
                "run": i + 1,
                "random_seed": cfg.random_seed,
                "monthly_return_pct": bt["monthly_return_pct"],
                "max_drawdown_pct": bt["max_drawdown_pct"],
                "net_compounded_return_pct": bt["net_compounded_return_pct"],
                "trades": bt.get("trades", 0),
                "win_rate_pct": pct(bt.get("win_rate", 0.0)),
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

    best_cfg: bot.Config | None = None
    best_train: dict[str, Any] = {}
    best_bt: dict[str, Any] = {"objective_score": -10**9}
    found_target = False
    trial_count = 0
    start = time.time()

    print(
        f"Optimizer başladı | hedef aylık > %{args.target_monthly:.2f}, "
        f"DD < %{args.max_dd:.2f}, timeout={args.timeout_hours:.2f} saat"
    )

    for cfg in candidate_configs(base_cfg, rng):
        if time.time() >= deadline:
            break
        if args.max_trials and trial_count >= args.max_trials:
            break

        trial_count += 1
        try:
            train_metrics, bt_metrics = run_train_backtest(cfg, quiet=args.quiet)
            bt_metrics["objective_score"] = objective(bt_metrics, target_monthly, max_dd)
        except Exception as exc:
            print(f"[trial {trial_count}] HATA: {exc}")
            continue

        is_best = bt_metrics["objective_score"] > safe_float(best_bt.get("objective_score"), -10**9)
        if is_best:
            best_cfg = cfg
            best_train = train_metrics
            best_bt = bt_metrics
            print(
                f"[trial {trial_count}] YENI EN IYI | monthly={bt_metrics['monthly_return_pct']}% "
                f"dd={bt_metrics['max_drawdown_pct']}% trades={bt_metrics.get('trades', 0)} "
                f"score={bt_metrics['objective_score']} direction={cfg.trade_direction} "
                f"long_th={cfg.long_threshold} short_th={cfg.short_threshold} "
                f"tp_r={cfg.tp_r} stop_atr={cfg.stop_atr_mult}"
            )

        if target_hit(bt_metrics, target_monthly, max_dd):
            found_target = True
            best_cfg = cfg
            best_train = train_metrics
            best_bt = bt_metrics
            print(f"[trial {trial_count}] HEDEF BULUNDU.")
            break

    if best_cfg is None:
        raise RuntimeError("Hiç geçerli backtest üretilemedi; veri/API ayarlarını kontrol et.")

    timeout_hit = not found_target and time.time() >= deadline
    print("En iyi model doğrulanıyor...")
    verification = verify_best(best_cfg, runs=args.verify_runs, quiet=args.quiet)
    elapsed = time.time() - start

    save_outputs(
        best_cfg=best_cfg,
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

    print("\n=== SONUC ===")
    print(f"found_target={found_target} timeout_hit={timeout_hit} trials={trial_count} elapsed_sec={elapsed:.2f}")
    print(
        f"best monthly={best_bt['monthly_return_pct']}% | dd={best_bt['max_drawdown_pct']}% | "
        f"net={best_bt['net_compounded_return_pct']}% | trades={best_bt.get('trades', 0)}"
    )
    print(f"ayar dosyası: {OUT_JSON}")
    print(f"env dosyası: {OUT_ENV}")
    print(f"model dosyası: {BEST_MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
