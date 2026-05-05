#!/usr/bin/env python3
"""Zero-input optimizer runner.

Just run:
  python run_max_profit_optimizer.py

Goal:
  Find a model where:
    monthly net profit > 5%
    monthly net profit > absolute max drawdown
    absolute max drawdown < 10%

If multiple valid models are found, keep the highest monthly net profit.
If timeout happens, save the best model found so far.

Defaults:
  timeout = 3 hours
  target monthly return = 5%
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

TARGET_MONTHLY = 0.05
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


def metrics(result: dict[str, Any]) -> tuple[float, float, float, float]:
    monthly = opt.calc_monthly_return(result)
    dd = abs(safe_float(result.get("max_drawdown", 1.0), 1.0))
    trades = safe_float(result.get("trades", 0), 0.0)
    win_rate = safe_float(result.get("win_rate", 0.0), 0.0)
    return monthly, dd, trades, win_rate


def target_valid(result: dict[str, Any]) -> bool:
    monthly, dd, trades, _ = metrics(result)
    return trades > 0 and monthly >= TARGET_MONTHLY and monthly > dd and dd <= MAX_DD


def dd_valid(result: dict[str, Any]) -> bool:
    monthly, dd, trades, _ = metrics(result)
    return trades > 0 and dd <= MAX_DD and monthly > 0


def target_objective(result: dict[str, Any]) -> float:
    monthly, dd, trades, win_rate = metrics(result)
    monthly_minus_dd = monthly - dd

    # Best case: monthly >= 5%, monthly > DD, and DD <= 10%.
    # Among valid models, maximize monthly profit, then margin above DD.
    if trades > 0 and monthly >= TARGET_MONTHLY and monthly > dd and dd <= MAX_DD:
        score = 3_000_000.0
        score += monthly * 30_000.0
        score += monthly_minus_dd * 20_000.0
        score += win_rate * 75.0
        score += min(trades, 300.0)
        score -= dd * 250.0
        return round(score, 6)

    # Second priority: DD-valid and profitable, but not yet monthly > 5 and > DD.
    if trades > 0 and dd <= MAX_DD and monthly > 0:
        score = 1_000_000.0
        score += monthly * 10_000.0
        score += monthly_minus_dd * 8_000.0
        score -= max(0.0, TARGET_MONTHLY - monthly) * 30_000.0
        score -= max(0.0, dd - monthly) * 30_000.0
        score += win_rate * 50.0
        score += min(trades, 300.0)
        score -= dd * 200.0
        return round(score, 6)

    # Fallback: no acceptable model yet, heavily punish DD over 10% and non-positive monthly profit.
    score = monthly * 100.0
    score += monthly_minus_dd * 100.0
    score -= max(0.0, dd - MAX_DD) * 20_000.0
    score -= dd * 300.0
    if trades < 8:
        score -= 100.0
    return round(score, 6)


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
    best_dd_valid = False
    best_target_valid = False
    trial_count = 0
    start = time.time()

    print(
        "Optimizer başladı | hedef: aylık net kâr > %5, aylık net kâr > DD, DD <%10 | "
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
            bt_metrics["objective_score"] = target_objective(bt_metrics)
            bt_metrics["clear_summary"] = opt.clear_backtest_summary(bt_metrics)
        except Exception as exc:
            print(f"[trial {trial_count}] HATA: {exc}")
            continue

        valid_target = target_valid(bt_metrics)
        valid_dd = dd_valid(bt_metrics)
        current_score = safe_float(bt_metrics.get("objective_score"), -10**18)
        best_score = safe_float(best_bt.get("objective_score"), -10**18)

        is_best = False
        if valid_target and not best_target_valid:
            is_best = True
        elif valid_target == best_target_valid:
            if valid_dd and not best_dd_valid:
                is_best = True
            elif valid_dd == best_dd_valid and current_score > best_score:
                is_best = True

        if is_best:
            best_candidate = candidate
            best_train = train_metrics
            best_bt = bt_metrics
            best_target_valid = valid_target
            best_dd_valid = valid_dd
            summary = opt.clear_backtest_summary(bt_metrics)
            monthly = opt.calc_monthly_return(bt_metrics)
            dd = abs(safe_float(bt_metrics.get("max_drawdown", 0.0), 0.0))
            print(
                f"[trial {trial_count}] YENİ EN İYİ | target_valid={valid_target} dd_valid={valid_dd} "
                f"monthly={summary['monthly_net_profit_pct']}% dd={summary['max_drawdown_pct']}% "
                f"monthly_minus_dd_pct={round((monthly - dd) * 100, 4)}% "
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

    # Add monthly-minus-DD values for readability.
    monthly = opt.calc_monthly_return(best_bt)
    dd = abs(safe_float(best_bt.get("max_drawdown", 0.0), 0.0))
    best_bt["monthly_minus_abs_drawdown"] = monthly - dd
    best_bt["monthly_minus_abs_drawdown_pct"] = round((monthly - dd) * 100, 4)

    opt.save_outputs(
        best_candidate=best_candidate,
        best_train=best_train,
        best_backtest=best_bt,
        verification=verification,
        target_monthly=TARGET_MONTHLY,
        max_dd=MAX_DD,
        timeout_hit=True,
        found_target=best_target_valid,
        trial_count=trial_count,
        elapsed_sec=elapsed,
    )

    final_summary = opt.clear_backtest_summary(best_bt)
    print("\n=== SONUÇ ===")
    print(
        f"target_monthly_gt_5_and_monthly_gt_dd_and_dd_under_10={best_target_valid} "
        f"dd_valid_under_10={best_dd_valid} trials={trial_count} elapsed_sec={elapsed:.2f}"
    )
    print(
        f"best monthly={final_summary['monthly_net_profit_pct']}% | dd={final_summary['max_drawdown_pct']}% | "
        f"monthly_minus_dd={best_bt['monthly_minus_abs_drawdown_pct']}% | "
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
