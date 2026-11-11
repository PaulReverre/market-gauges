"""The contrarian-signals pipeline (src/signals): scoring is point-in-time and the evidence adds up."""
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from signals import companies as C
from signals import fetch as F
from signals import model as M

pytestmark = pytest.mark.filterwarnings("ignore")      # synthetic series trip numpy fit warnings


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


# ------------------------------------------------------------------ alerts
from signals import alerts as A


def _g(gid, score, spark=None):
    stats = {"hit": 79, "median": 5.0}
    return {"id": gid, "name": "S&P 500" if gid == "spx" else "AI & semiconductors", "score": score, "as_of": "2026-11-09",
            "zone": {"label": "Extreme fear"}, "spark": spark or [score],
            "now": {"zone": {"episodes": 51, "h": {"3m": stats}}, "base": {"3m": {"hit": 70, "median": 3.4}}}}


CFG = {"spx": {"below": 10.0, "above": 90.0}, "ai": {"below": 15.0, "above": 85.0}, "dashboard_url": "https://x/"}


def test_alert_state_uses_each_gauges_thresholds():
    assert A.state_of(10, CFG["spx"]) == "below" and A.state_of(10.1, CFG["spx"]) == "none"
    assert A.state_of(90, CFG["spx"]) == "above" and A.state_of(12, CFG["ai"]) == "below"


def test_evaluate_lists_only_gauges_in_range_and_counts_days():
    out = A.evaluate({"spx": _g("spx", 8, [40, 12, 9, 8]), "ai": _g("ai", 50)}, CFG)
    assert out["thresholds"]["ai"] == {"below": 15.0, "above": 85.0}
    assert [(a["gauge"], a["kind"], a["days"], a["threshold"]) for a in out["active"]] == [("spx", "below", 2, 10.0)]


def test_transitions_fire_once_per_change():
    gauges = {"spx": _g("spx", 8), "ai": _g("ai", 50)}
    assert A.transitions({}, gauges, CFG) == [{"gauge": "spx", "was": "none", "now": "below"}]
    assert A.transitions({"spx": "below", "ai": "none"}, gauges, CFG) == []                       # already told
    assert A.transitions({"spx": "below", "ai": "above"}, gauges, CFG) == [{"gauge": "ai", "was": "above", "now": "none"}]


def test_issue_text_states_score_threshold_and_odds():
    title, body = A.issue_text(_g("spx", 8), "below", 10.0, "https://x/")
    assert title == "Buy alert: S&P 500 gauge at 8 (at or below 10)"
    assert "79% of the time" in body and "https://x/#spx" in body and "not a forecast" in body


def test_alert_config_rejects_inverted_thresholds(tmp_path):
    p = tmp_path / "alerts.yaml"
    p.write_text("spx: {below: 95, above: 90}\n")
    with pytest.raises(ValueError):
        A.load_config(p)
    p.write_text("ai: {below: 20}\n")
    assert A.load_config(p)["ai"] == {"below": 20.0, "above": 90.0} and A.load_config(p)["spx"] == {"below": 10.0, "above": 90.0}


# ------------------------------------------------------------ fundamentals
from signals import fundamentals as FU


def _monthly(n=900, seed=0):
    idx = pd.date_range("1951-01-01", periods=n, freq="MS")
    return idx, np.random.default_rng(seed)


def test_fundamentals_forward_returns_are_annualized():
    idx = pd.date_range("2000-01-01", periods=200, freq="MS")
    tr = pd.Series(1.01 ** np.arange(200), index=idx)                 # 1% a month
    Y = FU.forward_returns(tr)
    assert abs(Y["1y"].iloc[0] - (1.01 ** 12 - 1) * 100) < 1e-6 and abs(Y["10y"].iloc[0] - Y["1y"].iloc[0]) < 1e-6
    assert Y["10y"].iloc[-119:].isna().all()                          # no outcome yet for the last decade


def test_out_of_sample_fit_never_sees_unfinished_windows():
    idx, rng = _monthly()
    x = pd.Series(rng.normal(size=len(idx)), index=idx).rolling(24).mean()
    y = -5 * x + 10 + rng.normal(0, .5, len(idx))
    y.iloc[-120:] = np.nan                                            # like a 10-year forward return
    fc = FU.out_of_sample(x, y)
    y2 = y.copy(); y2.loc["2000":] = 999                              # corrupt outcomes that finish after 2010
    fc2 = FU.out_of_sample(x, y2)
    cut = fc.loc[:"2009-12"].index
    pd.testing.assert_series_equal(fc.loc[cut], fc2.loc[cut])         # forecasts up to 2009 use windows that ended by then
    rec = FU.record(fc, y)
    assert rec["corr"] > 0.7 and rec["mae"] < rec["mae_naive"]


def test_fundamentals_grade_needs_both_halves_and_a_better_record():
    good = {"corr": 0.8, "mae": 2.6, "mae_naive": 4.9}
    assert FU.grade(0.7, 0.8, 0.6, good) == "strong"
    assert FU.grade(0.5, 0.8, 0.8, {"corr": 0.7, "mae": 4.3, "mae_naive": 4.9}) == "moderate"
    assert FU.grade(0.35, 0.8, 0.3, {"corr": 0.7, "mae": 5.5, "mae_naive": 4.9}) == "weak"     # ranks well, misses by more
    assert FU.grade(0.5, 0.8, -0.2, good) == "weak"                                             # sign flips after the split
    assert FU.grade(-0.01, 0.0, -0.02, {"corr": -0.7, "mae": 6.2, "mae_naive": 4.9}) == "none"
    assert FU.grade(0.04, 0.01, 0.11, {"corr": -0.6, "mae": 6.2, "mae_naive": 4.7}) == "none"


def _fund_inputs():
    """Synthetic world: one valuation ratio drives everything, so every 'high = expensive' measure should work."""
    idx, rng = _monthly(n=909)                                        # 1951-01 .. 2026-09
    v = np.cumsum(rng.normal(0, .03, len(idx)))
    v = v - np.linspace(v[0], v[-1], len(idx))                        # mean-reverting log valuation
    earn = 10 * 1.005 ** np.arange(len(idx))
    price = earn * 15 * np.exp(v)
    sh = pd.DataFrame({"price": price, "dividend": price * .03 / np.exp(v), "earnings": earn, "cpi": 100 * 1.002 ** np.arange(len(idx)),
                       "cape": 15 * np.exp(v)}, index=idx)
    q = idx[::3]
    lvl = pd.Series(price, index=idx).reindex(q)
    z1 = pd.DataFrame({"NCBEILQ027S": lvl * 1000, "FBCELLQ027S": lvl * 300, **{c: 1.5e6 * 1.01 ** np.arange(len(q)) for c in FU.DEBT}}, index=q)
    g = 2000 * 1.012 ** np.arange(len(q))
    fr = pd.DataFrame({"A455RC1Q027SBEA": g * .5, "TNWMVBSNNCB": g * 800, "NCBCEBQ027S": -lvl.values * 10, "BOGZ1FA106121075Q": lvl.values * 20,
                       "NFCPATAX": g * .05, "CP": g * .08, "GDP": g}, index=q)
    fr = pd.concat([fr, pd.Series(5.0, index=idx, name="GS10")], axis=1, sort=True)
    spx = pd.Series(price, index=idx).resample("B").ffill()
    return sh, z1, fr, spx


def test_fundamentals_build_scores_expensive_markets_high():
    sh, z1, fr, spx = _fund_inputs()
    out, charts = FU.build(sh, z1, fr, spx, None)
    assert 0 <= out["score"] <= 100 and out["zone"]["label"] in FU.ZONE_LABEL.values()
    assert out["expected"]["models"] == 10 and out["expected"]["low"] <= out["expected"]["mid"] <= out["expected"]["high"]
    v = out["views"]
    assert min(v["history"]["pred"], v["era"]["pred"]) <= out["expected"]["mid"] <= max(v["history"]["pred"], v["era"]["pred"])
    assert abs(out["score"] - (v["history"]["score"] + v["era"]["score"]) / 2) < 0.2          # the score is the two yardsticks averaged
    by = {r["id"]: r for r in out["inputs"]}
    assert [r["id"] for r in out["inputs"] if r["role"] == "scored"] == ["alloc", "tobin_q", "ecy", "cape", "payout"]
    assert by["cape"]["r2"]["10y"] > 0.3 and by["cape"]["grade"] in ("strong", "moderate")   # valuation mean-reverts by construction
    assert sum(z["months"] for z in out["odds"]["zones"]) == out["odds"]["months"]
    cheap, dear = out["odds"]["zones"][0]["h"]["10y"], out["odds"]["zones"][-1]["h"]["10y"]
    if cheap and dear:
        assert cheap["median"] > dear["median"]
    assert set(charts["_gauge"]) == {"dates", "score", "score_history", "score_era", "estimate", "realized"} and "cape" in charts
    assert by["cape"]["era"] and 0 <= by["cape"]["era"]["pct"] <= 100 and by["pe"]["era"] is None   # only scored measures get the era yardstick
    assert out["support"] is None                                                             # no credit series passed


def test_era_yardstick_reads_a_new_plateau_as_less_extreme():
    """A measure that steps up to a permanently higher level: record-high forever against all history,
    but back to ordinary against its own 30-year average once the plateau has lasted."""
    idx = pd.date_range("1951-01-01", periods=900, freq="MS")
    rng = np.random.default_rng(3)
    level = np.where(np.arange(900) < 400, 15.0, 30.0) + rng.normal(0, 1.0, 900)
    m = FU.Metric("x", "X", "x", 1, "high", "scored", what="", why="", source="", series=pd.Series(level, index=idx),
                  now=float(level[-1]), official=float(level[-1]), log=True)
    y = pd.DataFrame({h: pd.Series(rng.normal(8, 3, 900), index=idx) for h in FU.HZ.values()})
    row, _, _ = FU.evaluate(m, y, era=True)
    assert row["pct"] > 60 and row["era"]["pct"] < row["pct"] and abs(row["era"]["avg"] - 30) < 2


def test_support_check_splits_expensive_months():
    idx = pd.date_range("1960-01-01", periods=480, freq="MS")
    score = pd.Series(np.where(np.arange(480) % 40 < 20, 80.0, 50.0), index=idx)
    eps = pd.Series(np.where(np.arange(480) % 80 < 40, 5.0, -5.0), index=idx)
    baa = pd.Series(2.0, index=pd.date_range("1959-06-01", periods=600, freq="MS"))
    Y = pd.DataFrame({h: pd.Series(np.where(eps > 0, 10.0, 2.0), index=idx) for h in FU.HZ.values()})
    s = FU.support_check(score, Y, eps, baa)
    yes, no = (next(r for r in s["rows"] if r["key"] == k) for k in ("yes", "no"))
    assert yes["months"] + no["months"] == int((score >= 75).sum())
    assert yes["h"]["1y"]["mean"] == 10 and no["h"]["1y"]["mean"] == 2 and s["now"]["supported"] in (True, False)
    assert FU.support_check(score, Y, eps, None) is None


def test_fundamentals_percentile_orientation():
    sh, z1, fr, spx = _fund_inputs()
    sh.loc[sh.index[-6:], ["price", "cape"]] *= 3                     # a sudden bubble
    spx = pd.Series(sh["price"].values, index=sh.index).resample("B").ffill()
    out, _ = FU.build(sh, z1, fr, spx, None)
    by = {r["id"]: r for r in out["inputs"]}
    assert by["cape"]["pct"] > 95 and by["ecy"]["pct"] > 90 and by["dy"]["pct"] > 95   # high P/E, low yields: all read "expensive"
    assert out["score"] > 70 and out["zone"]["key"] in ("calm", "caution")     # both yardsticks agree it is expensive
