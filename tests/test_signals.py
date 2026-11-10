"""The contrarian-signals pipeline (src/signals): scoring is point-in-time and the evidence adds up."""
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from signals import companies as C
from signals import fetch as F
from signals import model as M


def _cal(n=3000):
    return pd.bdate_range("2000-01-03", periods=n)


def test_expanding_pct_uses_only_the_past():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(size=1200), index=_cal(1200))
    full = M.expanding_pct(s, minp=100)
    cut = M.expanding_pct(s.iloc[:800], minp=100)
    pd.testing.assert_series_equal(full.iloc[:800], cut)          # later data never changes an earlier rank
    assert full.iloc[:99].isna().all() and 0 < full.iloc[500] <= 1


def test_iv_percentile_and_rank():
    s = pd.Series(np.arange(1.0, 301.0), index=_cal(300))         # always a new high
    assert M.iv_percentile(s, 252).iloc[-1] == 100
    assert M.iv_rank(s, 252).iloc[-1] == 100
    assert M.iv_percentile(s[::-1].set_axis(s.index), 252).iloc[-1] == 0


def test_zone_edges():
    assert M.zone_of(0)["key"] == "buy" and M.zone_of(9.99)["key"] == "buy"
    assert M.zone_of(10)["key"] == "fear" and M.zone_of(25)["key"] == "neutral"
    assert M.zone_of(89.9)["key"] == "calm" and M.zone_of(90)["key"] == "caution" and M.zone_of(100)["key"] == "caution"


def test_composite_low_score_means_fear():
    cal = _cal(1500)
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(size=1500), index=cal)
    b = pd.Series(rng.normal(size=1500), index=cal)
    a.iloc[-1] = b.iloc[-1] = 50                                   # both inputs at record fear
    fear = pd.DataFrame({"a": M.expanding_pct(a), "b": M.expanding_pct(b)})
    comp = M.composite(fear, {"p1": ["a"], "p2": ["b"]}, cal)
    assert comp["score"].iloc[-1] < 1 and comp["n"].iloc[-1] == 2
    assert comp["score"].dropna().between(0, 100).all()


def test_composite_needs_two_inputs():
    cal = _cal(1200)
    fear = pd.DataFrame({"a": M.expanding_pct(pd.Series(np.random.default_rng(3).normal(size=1200), index=cal))})
    assert M.composite(fear, {"p": ["a", "missing"]}, cal)["score"].isna().all()


def test_episode_counting_merges_close_visits():
    m = np.zeros(200, bool)
    m[[10, 11, 20, 100, 150, 160]] = True
    assert M.count_episodes(m, gap=21) == 3
    assert M.count_episodes(np.zeros(5, bool)) == 0


def test_episodes_report_forward_returns_from_first_day():
    cal = _cal(600)
    px = pd.Series(np.linspace(100, 160, 600), index=cal)
    score = pd.Series(50.0, index=cal)
    score.iloc[[100, 101, 105]] = 5
    score.iloc[-1] = 3
    eps = M.episodes(score, px, 10, below=True)
    assert len(eps) == 2
    first = eps[0]
    assert first["start"] == cal[100].date().isoformat() and first["days"] == 3 and first["extreme"] == 5
    assert first["1m"] == round((px.iloc[121] / px.iloc[100] - 1) * 100, 1)
    assert first["worst"] == 0 and first["open"] is False
    assert eps[1]["open"] is True and eps[1]["12m"] is None        # still running, no outcome yet


def test_odds_table_partitions_every_day():
    cal = _cal(2500)
    rng = np.random.default_rng(4)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 2500))), index=cal)
    score = pd.Series(rng.uniform(0, 100, 2500), index=cal)
    t = M.odds_table(score, px)
    assert sum(z["days"] for z in t["zones"]) == t["days"] == 2500
    assert set(t["base"]) == {"1m", "3m", "6m", "12m"}
    assert t["base"]["12m"]["n"] == 2500 - 252


def test_tail_evidence_detects_a_planted_edge():
    cal = _cal(4000)
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.01, 4000)
    level = pd.Series(rng.uniform(0, 100, 4000), index=cal).rolling(15).mean().bfill()
    lo = level <= level.quantile(0.1)
    r[np.flatnonzero(lo.values)] += 0.004                          # fear days are followed by drift
    px = pd.Series(100 * np.exp(np.cumsum(r)), index=cal)
    ev = M.tail_evidence(level, px, float(level.quantile(0.1)), float(level.quantile(0.9)))
    assert ev["low"]["h"]["1m"]["excess"] > 0
    assert M.grade({"h": {"3m": {"excess": 3, "pre": 2, "post": 4}}, "p": 0.01}, "above") == "strong"
    assert M.grade({"h": {"3m": {"excess": 3, "pre": -2, "post": 4}}, "p": 0.01}, "above") == "weak"
    assert M.grade({"h": {"3m": {"excess": -1, "pre": -2, "post": -1}}, "p": 0.9}, "above") == "none"
    assert M.grade({"h": {"3m": {"excess": -2, "pre": -1, "post": -2}}, "p": 0.05}, "below") == "moderate"


@pytest.mark.parametrize("now, expected", [
    (datetime(2026, 11, 9, 17, 0, tzinfo=F.ET), date(2026, 11, 9)),    # Monday after the close
    (datetime(2026, 11, 9, 11, 0, tzinfo=F.ET), date(2026, 11, 6)),    # Monday mid-session -> Friday
    (datetime(2026, 11, 7, 12, 0, tzinfo=F.ET), date(2026, 11, 6)),    # Saturday -> Friday
    (datetime(2026, 11, 10, 2, 0, tzinfo=F.ET), date(2026, 11, 9)),    # early Tuesday -> Monday
])
def test_completed_session(now, expected):
    assert F.completed_session(now) == expected


def test_thin_keeps_recent_days_and_thins_older_ones():
    cal = _cal(3000)
    axis = M.thin(cal, keep_daily=500)
    assert list(axis[-500:]) == list(cal[-500:]) and len(axis) < 1100 and axis.is_monotonic_increasing


def test_clean_makes_json_safe():
    assert M.clean({"a": float("nan"), "b": [np.float64(1.5), np.int64(2), np.bool_(True)]}) == {"a": None, "b": [1.5, 2, True]}


def test_catalysts_window(tmp_path):
    cal = tmp_path / "calendar.yaml"
    cal.write_text("- {date: 2026-11-25, event: GDP, kind: macro}\n- {date: 2027-03-01, event: Far away, kind: macro}\n")
    out = C.catalysts([{"date": "2026-11-17", "label": "NVDA earnings", "kind": "earnings"},
                       {"date": "2026-10-01", "label": "old", "kind": "earnings"}], date(2026, 11, 9), cal)
    assert [c["label"] for c in out] == ["NVDA earnings", "GDP"]


def test_gpu_spot_index_ignores_series_that_start_late():
    idx = pd.date_range("2026-10-08", periods=5)
    df = pd.DataFrame({"a": [1.0, 1.1, 1.21, 1.21, 1.21], "b": [np.nan, np.nan, 5.0, 5.0, 5.0]}, index=idx)
    g = C.gpu_spot(df)
    assert g["index"][0] == 100 and g["index"][2] == 121 and g["value"] == 121   # b's arrival does not move the level
