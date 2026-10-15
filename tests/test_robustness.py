"""Robustness study helpers (no store, no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gauges import robustness as R


def series(values, start="2020-01-01") -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=pd.bdate_range(start, periods=len(values)))


def test_turning_points_follow_the_20pct_rule():
    # 130 (peak) -> 100 (-23%, confirms the peak) -> 125 (+25%, confirms the trough) -> 110
    p = series([120, 130, 120, 104, 100, 110, 125, 110])
    pts = R.turning_points(p, 20)
    assert pts == [(p.index[1], "peak"), (p.index[4], "trough")]
    # A 20% rise straight from the first close makes the first close a trough
    assert R.turning_points(series([100, 115, 130]), 20) == [(pd.Timestamp("2020-01-01"), "trough")]
    # A 19% dip is not a bear market
    assert R.turning_points(series([120, 140, 113.4, 130]), 20) == []


def test_swings_close_the_open_leg():
    p = series([130, 100, 125, 110, 128])
    sw = R.swings(p, 20)
    assert [s["kind"] for s in sw] == ["decline", "advance"]
    assert sw[0]["change"] == pytest.approx((100 / 130 - 1) * 100)
    assert sw[1]["ongoing"] and sw[1]["end"] == p.index[-1]
    # An open decline ends at its low so far, not at the latest close
    q = series([100, 130, 100, 95, 101])
    last = R.swings(q, 20)[-1]
    assert last["kind"] == "decline" and last["end"] == q.index[3] and last["ongoing"]


def test_episodes_merge_close_visits_and_find_the_extreme():
    x = [50] * 5 + [15, 12, 18] + [30] * 3 + [10] + [50] * 40 + [19] + [50] * 3
    ep = R.episodes(series(x), "fear", 20, gap=5)
    assert len(ep) == 2
    first = ep.iloc[0]
    assert (first.start, first.end, first.extreme, first.exit) == (5, 11, 11, 12)
    assert first.days == 4 and first.extreme_value == 10
    assert ep.iloc[1].start == 52
    greed = R.episodes(series([50, 85, 90, 70]), "greed", 80)
    assert list(greed.iloc[0][["start", "extreme", "exit"]]) == [1, 2, 3]
    assert R.episodes(series([np.nan, 50]), "fear", 20).empty


def test_random_entry_p_is_small_when_picks_are_the_best_days():
    rng = np.random.default_rng(1)
    values = np.r_[np.zeros(990), np.full(10, 5.0)]
    eligible = np.ones(1000, bool)
    obs, p = R.random_entry_p(values, np.arange(990, 1000), eligible, True, rng, 2000)
    assert obs == 5.0 and p < 0.01
    _, p_low = R.random_entry_p(values, np.arange(990, 1000), eligible, False, rng, 2000)
    assert p_low > 0.99


def test_greed_trim_has_hysteresis():
    g = series([np.nan, 50, 85, 70, 61, 60, 85])
    e = R.greed_trim(g, hi=80, back=60, trim=0.5)
    assert np.isnan(e.iloc[0])
    assert e.iloc[1:].tolist() == [1.0, 0.5, 0.5, 0.5, 1.0, 0.5]


def test_fear_add_holds_after_each_fear_reading():
    g = series([50, 15, 50, 50, 50, 50])
    assert R.fear_add(g, lo=20, hold=3, base=0.6).tolist() == pytest.approx([0.6, 1.0, 1.0, 1.0, 0.6, 0.6])


def test_tilt_is_linear_between_the_edges():
    g = series([10, 20, 50, 80, 95, np.nan])
    assert R.tilt(g, 20, 80, 0.5).tolist() == pytest.approx([1.0, 1.0, 0.75, 0.5, 0.5, 0.5])


def test_level_filter_and_win_share():
    s = series([45, 35, 50, np.nan])
    assert R.level_filter(s, 40, below=True).tolist() == [1.0, 0.5, 1.0, 1.0]
    a, b = pd.Series([0.1, 0.05, 0.2]), pd.Series([0.1, 0.07, 0.1])
    assert R.win_share(a, b) == 50.0   # the tie is ignored


def test_run_strategy_trades_the_day_after_the_signal():
    r = series([0.0, 0.01, 0.02, 0.03, 0.04])
    cash = series([0.001] * 5)
    expo = series([0.0, 1.0, 1.0, 1.0, 1.0])     # signal turns on at the close of day 1
    run = R.run_strategy(expo, r, cash, delay=1)
    # Trade at the close of day 2, so day 3 is the first return earned in the asset
    assert run["ret"].loc[r.index[2]] == pytest.approx(0.001)
    assert run["ret"].loc[r.index[3]] == pytest.approx(0.03)
    costly = R.run_strategy(expo, r, cash, delay=1, cost_bps=10)
    assert costly["ret"].loc[r.index[3]] == pytest.approx(0.03 - 0.001)


def test_metrics_on_a_steady_path():
    idx = pd.bdate_range("2020-01-01", periods=504)
    ret = pd.Series(0.0004, index=idx)
    ret.iloc[100] = -0.2
    m = R.metrics(ret, pd.Series(0.0, index=idx))
    assert m["max_dd"] == pytest.approx(-20.0, abs=0.01)
    assert m["years"] == pytest.approx(2.0)
    assert R.metrics(ret.iloc[:100], pd.Series(0.0, index=idx)) == {}


def test_band_helpers_match_the_dashboard():
    assert [R.band_index(v) for v in (0, 19.9, 20, 59.99, 80, 100)] == [0, 0, 1, 2, 4, 4]
    assert R.band_index(float("nan")) is None
    assert R.bands(series([10, np.nan, 65])).tolist() == [0, -1, 3]


def test_composite_without_rebuilds_and_drops_pillars():
    idx = pd.bdate_range("2020-01-01", periods=4)
    P = pd.DataFrame({"a": [10.0, 20, 30, 40], "b": [50.0, np.nan, 70, 80]}, index=idx)
    D = R.Data(idx, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"G1": P},
               {"G1": {"a": 3.0, "b": 1.0}}, {"G1": "none"}, {}, pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float),
               pd.Series(dtype=float))
    full = R.composite_without(D, "G1", None)
    assert full.tolist() == pytest.approx([20.0, 20.0, 40.0, 50.0])   # day 2 renormalizes over the live pillar
    assert R.composite_without(D, "G1", "b").tolist() == pytest.approx([10, 20, 30, 40])


def test_block_bootstrap_ci_and_stationary_indices():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    lo, hi = R.block_bootstrap_ci(x, 2 * x + 1, 20, 200, rng)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    y = rng.normal(size=500)
    lo, hi = R.block_bootstrap_ci(x, y, 20, 500, rng)
    assert lo < R.pearson(x, y) < hi and hi - lo < 0.5
    idx = R.stationary_bootstrap(100, 300, 50, 10, rng)
    assert idx.shape == (50, 300) and idx.min() >= 0 and idx.max() < 100
    steps = np.diff(idx, axis=1)
    assert ((steps == 1) | (steps == -99)).mean() > 0.85      # mostly continues the block


def test_placebo_finds_real_timing():
    rng = np.random.default_rng(3)
    n = 3000
    r = rng.normal(0.0003, 0.01, n)
    good = (r > 0).astype(float)              # perfect foresight: in only on up days
    out = R.placebo(good, r, np.zeros(n), rng, 200)
    assert out["p_sharpe"] < 0.01 and out["p_cagr"] < 0.01
    assert R.placebo(good[:500], r[:500], np.zeros(500), rng, 50) == {}


def test_state_fan_uses_real_blocks_from_the_current_state():
    rng = np.random.default_rng(5)
    r = np.r_[np.full(300, 0.01), np.full(300, -0.01)]
    state = np.r_[np.zeros(300, int), np.ones(300, int)]
    fan = R.state_fan(r, state, 0, rng, paths=200, steps=2, block=21)
    # Blocks that start in state 0 and end in state 0 come from the rising half only
    assert fan["final"]["p50"] > 0
    assert fan["p_loss"] < 50
    assert len(fan["months"]) == 2
    assert R.state_fan(r, np.full(600, -1), 0, rng) == {}


def test_market_events_file_is_valid():
    events = R.load_market_events()
    assert len(events) >= 25
    assert {e["kind"] for e in events} <= {"peak", "trough", "shock"}
    assert events == sorted(events, key=lambda e: e["date"])
