#!/usr/bin/env python3
"""Zero-input optimizer runner.

Just run:
  python run_max_profit_optimizer.py

Goal:
  Maximize monthly net profit while keeping max drawdown under 10%.

Defaults:
  timeout = 3 hours
  max drawdown = 10%
  feature-group search = enabled
  verification backtests = 3

Outputs:
  sonayarlar.json
  sonayarlar.env
  sontrade_model_best.joblib
"""
from __future__ import annotations

import random
import time
from typing import Any

import optimize_until_target as opt
import sontrade_bot as bot

MAX_DD = 0.10
TIMEOUT_HOURS = 3.0
VERIFY_RUNS = 3
MAX_TRIALS = 0
SEED = 42
QUIET = True


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def max_profit_objective(result: dict[str, Any]) -> float:
    monthly = opt.calc_monthly_return(result)
    dd = abs(safe_float(result.get("max_drawdown", 1.0), 1.0))
    trades = safe_float(result.get("trades", 0), 0.0)
    win_rate = safe_float(result.get("win_rate", 0.0), 0.0)

    # Main objective: among models with DD <= 10%, maximize monthly return.
    if dd <= MAX_DD and trades > 0:
        score = 1_000_000.0
        score += monthly * 10_000.0
        score += win_rate * 50.0
        score += min(trades, 250.0)
        score -= dd * 200.0
        return round(score, 6)

    # Fallback objective if no DD-valid model has been found yet.
    score = monthly * 100.0
    score -= max(0.0, dd - MAX_DD) * 20_000.0
    score -= dd * 300.0
    if trades < 8:
        score -= 100.0
    return round(score, 6)


def dd_valid(result: dict[str, Any]) -> bool:
    return (
        int(result.get("trades", 0) or 0) > 0
        and abs(safe_float(result.get("max_drawdown", 1.0), 1.0)) <= MAX_DD
    )


def main() -> int:
    bot.load_dotenv()
    bot.setup_logging(False)
    base_cfg = bot.load_config()

    timeout_sec = int(TIMEOUT_HOURS * 3600)
    deadline = time.time() + timeout_sec
    rng = random.Random(SEED)

    best_candidate: opt.Candidate | None = None
    best_train: dict[str, Any] = {}
    best_bt: dict[str, Any] = {"objective_score": -10**18}
    best_valid_under_dd = False
    trial_count = 0
    start = time.time()

    print(
        "Sıfır-parametre optimizer başladı | hedef: DD <%10 şartıyla aylık net kârı maksimize et | "
        f"timeout={TIMEOUT_HOURS:.2f} saat | feature_group_search=True"
    )

    for candidate in opt.candidate_configs(base_cfg, rng, optimize_feature_groups=True):
        if time.time() >= deadline:
            break
        if MAX_TRIALS and trial_count >= MAX_TRIALS:
            break

        trial_count += 1
        try:
            train_metrics, bt_metrics = opt.run_train_backtest(candidate, quiet=QUIET)
            bt_metrics["objective_score"] = max_profit_objective(bt_metrics)
            bt_metrics["clear_summary"] = opt.clear_backtest_summary(bt_metrics)
        except Exception as exc:
            print(f"[trial {trial_count}] HATA: {exc}")
            continue

        valid = dd_valid(bt_metrics)
        previous_valid = best_valid_under_dd
        current_score = safe_float(bt_metrics.get("objective_score"), -10**18)
        best_score = safe_float(best_bt.get("objective_score"), -10**18)

        # Prefer any DD-valid model over DD-invalid models. Among DD-valid models, maximize monthly profit.
        is_best = False
        if valid and not previous_valid:
            is_best = True
        elif valid == previous_valid and current_score > best_score:
            is_best = True

        if is_best:
            best_candidate = candidate
            best_train = train_metrics
            best_bt = bt_metrics
            best_valid_under_dd = valid
            summary = opt.clear_backtest_summary(bt_metrics)
            print(
                f"[trial {trial_count}] YENİ EN İYİ | dd_valid={valid} "
                f"monthly={summary['monthly_net_profit_pct']}% dd={summary['max_drawdown_pct']}% "
                f"win={summary['win_rate_pct']}% trades={summary['total_trades']} "
                f"L/S={summary['long_trades']}/{summary['short_trades']} "
                f"score={bt_metrics['objective_score']} groups={len(candidate.enabled_groups)}/{len(opt.ALL_GROUPS)} "
                f"direction={candidate.cfg.trade_direction} long_th={candidate.cfg.long_threshold} "
                f"short_th={candidate.cfg.short_threshold} tp_r={candidate.cfg.tp_r} stop_atr={candidate.cfg.stop_atr_mult}"
            )

    if best_candidate is None:
        raise RuntimeError("Hiç geçerli backtest üretilemedi; veri/API ayarlarını kontrol et.")

    print("En iyi model doğrulanıyor...")
    verification = opt.verify_best(best_candidate, runs=VERIFY_RUNS, quiet=QUIET)
    elapsed = time.time() - start

    opt.save_outputs(
        best_candidate=best_candidate,
        best_train=best_train,
        best_backtest=best_bt,
        verification=verification,
        target_monthly=0.0,
        max_dd=MAX_DD,
        timeout_hit=True,
        found_target=best_valid_under_dd,
        trial_count=trial_count,
        elapsed_sec=elapsed,
    )

    final_summary = opt.clear_backtest_summary(best_bt)
    print("\n=== SONUÇ ===")
    print(f"dd_valid_under_10={best_valid_under_dd} trials={trial_count} elapsed_sec={elapsed:.2f}")
    print(
        f"best monthly={final_summary['monthly_net_profit_pct']}% | dd={final_summary['max_drawdown_pct']}% | "
        f"net={final_summary['net_compounded_profit_pct']}% | win={final_summary['win_rate_pct']}% | "
        f"trades={final_summary['total_trades']} | long={final_summary['long_trades']} "
        f"({final_summary['long_trade_ratio_pct']}%) | short={final_summary['short_trades']} "
        f"({final_summary['short_trade_ratio_pct']}%)"
    )
    print(f"aktif analiz grupları: {', '.join(best_candidate.enabled_groups)}")
    print(f"ayar dosyası: {opt.OUT_JSON}")
    print(f"env dosyası: {opt.OUT_ENV}")
    print(f"model dosyası: {opt.BEST_MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
