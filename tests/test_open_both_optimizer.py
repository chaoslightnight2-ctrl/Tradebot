from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import tools.open_both_optimizer as opt


def make_metrics(monthly=5.0, total=50.0, dd=-10.0, trades=150, longs=75, shorts=75):
    return {
        "monthly_return_pct": monthly,
        "total_return_pct": total,
        "max_drawdown_pct": dd,
        "dd_to_monthly_ratio": abs(dd) / monthly if monthly > 0 else None,
        "win_rate_pct": 55.0,
        "profit_factor": 1.4,
        "trade_count": trades,
        "long_trade_count": longs,
        "short_trade_count": shorts,
        "avg_trade_return_pct": 0.1,
        "median_trade_return_pct": 0.05,
        "avg_hold_bars": 4.0,
        "stop_loss_exit_count": 10,
        "take_profit_exit_count": 20,
        "time_stop_exit_count": 30,
        "oos_monthly_return_pct": 2.0,
        "oos_max_drawdown_pct": -8.0,
    }


def make_candidate(monthly=5.0, dd=-10.0):
    return {
        "settings": {"trade_direction": "both", "horizon_bars": 3},
        "period_results": {
            "1Y": make_metrics(monthly=monthly, dd=dd, trades=40),
            "3Y": make_metrics(monthly=monthly, dd=dd, trades=100),
            "5Y": make_metrics(monthly=monthly, dd=dd, trades=150),
        },
        "oos_monthly_return_pct": 2.0,
        "oos_max_drawdown_pct": -8.0,
        "walk_forward_monthly_pass_rate_pct": 75.0,
        "overtrade_penalty": 0.0,
    }


def test_dd_gte_20_candidate_is_rejected():
    cand = make_candidate(monthly=5.0, dd=-20.0)
    ok, reasons = opt.check_hard_gates(cand, max_dd=20.0)
    assert not ok
    assert any("dd_gte" in r for r in reasons)


def test_non_positive_monthly_candidate_is_rejected():
    cand = make_candidate(monthly=0.0, dd=-5.0)
    ok, reasons = opt.check_hard_gates(cand, max_dd=20.0)
    assert not ok
    assert any("monthly_not_positive" in r for r in reasons)


def test_report_contains_1y_3y_5y_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(opt, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(opt, "REPORT_MD", tmp_path / "OPEN_BOTH_TOP3_OPTIMIZER_REPORT.md")
    monkeypatch.setattr(opt, "REPORT_JSON", tmp_path / "OPEN_BOTH_TOP3_OPTIMIZER_REPORT.json")
    monkeypatch.setattr(opt, "NEAR_MISS_JSON", tmp_path / "OPEN_BOTH_TOP3_NEAR_MISSES.json")
    args = Namespace(symbol="OPEN", direction="both", max_dd=20.0, top=3)
    cand = make_candidate(monthly=6.0, dd=-10.0)
    cand["passed"] = True
    cand["score"] = opt.compute_score(cand)
    cand["rejection_reasons"] = []
    opt.write_reports([cand], [], args)
    md = (tmp_path / "OPEN_BOTH_TOP3_OPTIMIZER_REPORT.md").read_text()
    js = json.loads((tmp_path / "OPEN_BOTH_TOP3_OPTIMIZER_REPORT.json").read_text())
    assert "1Y" in md and "3Y" in md and "5Y" in md
    assert set(js["top_candidates"][0]["period_results"].keys()) == {"1Y", "3Y", "5Y"}


def test_top_candidates_sorted_by_score():
    low = make_candidate(monthly=3.0, dd=-10.0)
    high = make_candidate(monthly=8.0, dd=-8.0)
    low["score"] = opt.compute_score(low)
    high["score"] = opt.compute_score(high)
    ranked = opt.rank_candidates([low, high])
    assert ranked[0]["score"] >= ranked[1]["score"]
    assert ranked[0] is high


def test_json_and_markdown_reports_are_created(tmp_path, monkeypatch):
    monkeypatch.setattr(opt, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(opt, "REPORT_MD", tmp_path / "OPEN_BOTH_TOP3_OPTIMIZER_REPORT.md")
    monkeypatch.setattr(opt, "REPORT_JSON", tmp_path / "OPEN_BOTH_TOP3_OPTIMIZER_REPORT.json")
    monkeypatch.setattr(opt, "NEAR_MISS_JSON", tmp_path / "OPEN_BOTH_TOP3_NEAR_MISSES.json")
    args = Namespace(symbol="OPEN", direction="both", max_dd=20.0, top=3)
    cand = make_candidate(monthly=6.0, dd=-10.0)
    cand["passed"] = True
    cand["score"] = opt.compute_score(cand)
    cand["rejection_reasons"] = []
    opt.write_reports([cand], [], args)
    assert (tmp_path / "OPEN_BOTH_TOP3_OPTIMIZER_REPORT.md").exists()
    assert (tmp_path / "OPEN_BOTH_TOP3_OPTIMIZER_REPORT.json").exists()
    assert (tmp_path / "OPEN_BOTH_TOP3_NEAR_MISSES.json").exists()
