"""Transforms vs hand-computed values."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gauges import transforms as T

IDX = pd.bdate_range("2026-01-05", periods=6)


def s(values, index=IDX):
    return pd.Series(values, index=index[:len(values)], dtype=float)


def test_pct_from_ma():
    # MA(3) of 10, 11, 12 = 11; 12/11 - 1 = 9.0909%
    out = T.pct_from_ma(s([10, 11, 12]), 3)
    assert out.iloc[:2].isna().all() and out.iloc[2] == pytest.approx(100 / 11)


def test_change_pct_change_and_avg_pct_change():
    x = s([100, 102, 104, 110])
    assert T.change(x, 2, scale=100).iloc[3] == pytest.approx(800)          # (110 - 102) * 100
    assert T.pct_change(x, 3).iloc[3] == pytest.approx(10.0)                # 110/100 - 1
    # 2-period averages: 101, 103, 107; change over 1 period: 107/103 - 1
    assert T.avg_pct_change(x, 2, 1).iloc[3] == pytest.approx((107 / 103 - 1) * 100)


def test_zscore_and_clip():
    x = s([1, 2, 3, 4, 100, 5])
    z = T.zscore(x, window=4)
    window = np.array([1, 2, 3, 4])   # the window includes the current value
    assert z.iloc[3] == pytest.approx((4 - window.mean()) / window.std(ddof=1))
    spike = pd.Series([0.0, 1.0] * 10 + [100.0], index=pd.bdate_range("2026-01-01", periods=21))
    assert T.zscore(spike, window=20).iloc[-1] == 3.0   # raw z is about 4.3, winsorized at 3


def test_ratio_diff_and_ratio_change():
    df = pd.DataFrame({"a": [20.0, 22.0, 24.0], "b": [10.0, 10.0, 12.0]}, index=IDX[:3])
    assert T.c_ratio(df).tolist() == [2.0, 2.2, 2.0]
    assert T.c_diff(df).tolist() == [10.0, 12.0, 12.0]
    assert T.c_ratio_change(df, 1).iloc[1] == pytest.approx(10.0)


def test_diff_max_and_diff_mean():
    df = pd.DataFrame({"sofr": [5.30, 5.33, 5.31], "iorb": [5.40, 5.30, 5.30]}, index=IDX[:3])
    # diffs -0.10, 0.03, 0.01 -> max over 3 = 0.03 -> 3 bp
    assert T.c_diff_max(df, window=3, scale=100).iloc[2] == pytest.approx(3.0)
    assert T.c_diff_mean(df, window=2).iloc[2] == pytest.approx(0.02)


def test_realized_vol_and_vrp():
    close = s([100, 101, 99, 102])
    r = np.log(np.array([101 / 100, 99 / 101, 102 / 99]))
    expected = r.std(ddof=1) * math.sqrt(252) * 100
    assert T.realized_vol(close, 3).iloc[3] == pytest.approx(expected)
    df = pd.DataFrame({"vix": [20.0] * 4, "spx": close.values}, index=IDX[:4])
    assert T.c_vrp(df, 3).iloc[3] == pytest.approx(20 - expected)


def test_return_spread_and_earnings_yield_gap():
    df = pd.DataFrame({"spx": [100.0, 105.0], "ief": [100.0, 101.0]}, index=IDX[:2])
    assert T.c_return_spread(df, 1).iloc[1] == pytest.approx(4.0)
    gap = T.c_earnings_yield_gap(pd.DataFrame({"pe": [20.0], "real": [2.0]}, index=IDX[:1]))
    assert gap.iloc[0] == pytest.approx(3.0)   # 5% earnings yield - 2%


def test_net_liquidity_change():
    df = pd.DataFrame({"walcl": [7_000_000.0, 6_900_000.0], "tga": [800_000.0, 750_000.0], "rrp": [100.0, 50.0]},
                      index=IDX[:2])
    # net: 7000-800-100 = 6100bn -> 6900-750-50 = 6100bn: change 0
    assert T.c_net_liquidity_change(df, 1).iloc[1] == pytest.approx(0.0)


def test_sum_avg_pct_change_skips_incomplete_months():
    months = pd.date_range("2025-01-01", periods=14, freq="MS")
    a = pd.Series(np.arange(10, 24, dtype=float), index=months)
    b = pd.Series(np.full(14, 10.0), index=months)
    b.iloc[0] = np.nan
    out = T.c_sum_avg_pct_change(pd.DataFrame({"a": a, "b": b}), window=1, periods=12)
    # first complete month is Feb 2025 (a=11) so only Feb 2026 has a 12-month base: (23+10)/(11+10) - 1
    assert out.dropna().iloc[-1] == pytest.approx((33 / 21 - 1) * 100)


def test_breadth_panels():
    closes = pd.DataFrame({"A": [1, 2, 3, 4.0], "B": [4, 3, 2, 1.0], "C": [1, 1, 1, 2.0]}, index=IDX[:4])
    # MA(2) at last row: A 3.5 (4 above), B 1.5 (1 below), C 1.5 (2 above) -> 2 of 3
    assert T.pct_above_ma(closes, 2).iloc[3] == pytest.approx(200 / 3)
    # window 3 at last row: A new high, B new low, C high (2 = max) -> 2 highs - 1 low
    assert T.net_highs_lows(closes, 3, 1).iloc[3] == 1.0


def test_mcclellan_summation():
    closes = pd.DataFrame({"A": [10, 11, 12.0], "B": [10, 9, 10.0]}, index=IDX[:3])
    volumes = pd.DataFrame({"A": [0, 300, 100.0], "B": [0, 100, 100.0]}, index=IDX[:3])
    # day 2: adv 300, dec 100 -> RANA 500; day 3: adv 200, dec 0 -> 1000
    # EMA10 (alpha .1): 500, 550; EMA05: 500, 525 -> osc 0, 25 -> summation 0, 25
    out = T.mcclellan_summation(closes, volumes)
    assert np.isnan(out.iloc[0]) and out.iloc[1] == pytest.approx(0.0) and out.iloc[2] == pytest.approx(25.0)


def test_chain_index_ignores_columns_entering_and_leaving():
    prices = pd.DataFrame({"a": [1.0, 2.0, 2.0], "b": [np.nan, 4.0, 8.0], "c": [1.0, 1.0, np.nan]}, index=IDX[:3])
    # step 1: a doubles, c flat, b has no prior price -> mean log change ln2/2; step 2: a flat, b doubles
    assert T.chain_index(prices).tolist() == pytest.approx([100.0, 100 * math.sqrt(2), 200.0])
    assert T.chain_index(pd.DataFrame()).empty


def test_basket_index_converts_krw_and_renormalizes():
    closes = pd.DataFrame({"US": [100, 110.0, 110.0], "KR": [1000, 1000.0, 1100.0]}, index=IDX[:3])
    fx = pd.Series([1000, 1100.0, 1100.0], index=IDX[:3])          # KRW per USD
    r = T.usd_returns(closes, {"KR": fx})
    assert r["KR"].iloc[1] == pytest.approx(1000 / 1100 - 1)       # flat in KRW, won weakened
    idx = T.basket_index(r, {"us": (50, ["US"]), "kr": (50, ["KR"])})
    day2 = 0.5 * 0.10 + 0.5 * (1000 / 1100 - 1)
    assert idx.iloc[0] == pytest.approx(1 + day2)
    assert idx.iloc[1] == pytest.approx((1 + day2) * (1 + 0.5 * 0.0 + 0.5 * 0.10))


def test_drawdown():
    assert T.drawdown(s([100, 120, 90]), window=3).iloc[2] == pytest.approx(-25.0)


def test_avg_pairwise_corr():
    rng = np.random.default_rng(0)
    base = rng.normal(size=80)
    df = pd.DataFrame({"a": base, "b": base, "c": base}, index=pd.bdate_range("2026-01-01", periods=80))
    assert T.avg_pairwise_corr(df, window=40, step=40).iloc[-1] == pytest.approx(1.0)


def test_bs_delta_atm_and_risk_reversal():
    d = T.bs_delta(100, np.array([100.0]), 1.0, np.array([0.2]), "call")
    assert d[0] == pytest.approx(0.5398, abs=1e-4)                  # N(0.1)
    snap = pd.Timestamp("2026-10-14")
    expiry = pd.Timestamp("2026-11-13")                              # 30 days
    strikes = np.arange(80, 121, 5, dtype=float)
    rows = [{"expiry": expiry, "option_type": k, "strike": x, "implied_vol": 0.40} for k in ("call", "put") for x in strikes]
    chain = pd.DataFrame(rows)
    assert T.atm_iv(chain, 101.0, snap) == pytest.approx(40.0)
    assert T.risk_reversal(chain, 100.0, snap) == pytest.approx(0.0)
    skewed = chain.assign(implied_vol=np.where(chain["option_type"] == "put", 0.50, 0.40))
    assert T.risk_reversal(skewed, 100.0, snap) == pytest.approx(-10.0)


def test_atm_iv_interpolates_total_variance():
    snap = pd.Timestamp("2026-10-01")
    chain = pd.DataFrame([
        {"expiry": snap + pd.Timedelta(days=20), "option_type": "call", "strike": 100.0, "implied_vol": 0.30},
        {"expiry": snap + pd.Timedelta(days=40), "option_type": "call", "strike": 100.0, "implied_vol": 0.40}])
    expected = math.sqrt((0.30 ** 2 * 20 + (0.40 ** 2 * 40 - 0.30 ** 2 * 20) * 0.5) / 30) * 100
    assert T.atm_iv(chain, 100.0, snap) == pytest.approx(expected)


def test_quarters_from_ytd():
    facts = pd.DataFrame([
        ("2025-07-01", "2025-09-30", 10.0, "2025-10-30"),
        ("2025-07-01", "2025-12-31", 25.0, "2026-01-29"),
        ("2025-07-01", "2026-03-31", 45.0, "2026-04-29"),
        ("2025-07-01", "2026-06-30", 70.0, "2026-07-29"),
        ("2025-07-01", "2026-06-30", 71.0, "2026-08-15"),   # later restatement ignored (point in time)
    ], columns=["period_start", "period_end", "value", "filed"])
    for col in ("period_start", "period_end", "filed"):
        facts[col] = pd.to_datetime(facts[col]).dt.date
    q = T.quarters_from_ytd(facts)
    assert q["value"].tolist() == [10.0, 15.0, 20.0, 25.0]
    assert str(q["filed"].iloc[-1]) == "2026-07-29"


@pytest.mark.parametrize("d, expected", [("2026-05-31", "2026-06-30"), ("2026-08-31", "2026-09-30"),
                                         ("2026-02-28", "2026-03-31"), ("2025-12-31", "2025-12-31")])
def test_calendar_quarter(d, expected):
    assert T.calendar_quarter(d) == pd.Timestamp(expected)


def test_rolling_percentile_is_point_in_time():
    x = s([1, 3, 2, 5, 4, 0])
    p = T.rolling_percentile(x, window=None, min_periods=3)
    # day 3: values [1,3,2], 2 ranks 2nd of 3 -> (2-0.5)/3
    assert p.iloc[2] == pytest.approx(50.0)
    assert p.iloc[3] == pytest.approx(87.5)            # 5 is the max of 4: (4-0.5)/4
    assert p.iloc[5] == pytest.approx(0.5 / 6 * 100)   # 0 is the min of 6
    w = T.rolling_percentile(x, window=3, min_periods=3)
    assert w.iloc[5] == pytest.approx(0.5 / 3 * 100)


def test_staleness_multiplier():
    assert T.staleness_multiplier([0, 7, 15, 20], freq_bdays=5).tolist() == [1.0, 1.0, 0.0, 0.0]
    assert T.staleness_multiplier([11.25], freq_bdays=5)[0] == pytest.approx(0.5)


def test_piecewise_and_band_and_ema():
    assert T.piecewise(12.5, [(0, 10), (25, 40)]) == pytest.approx(25.0)
    assert T.piecewise(-5, [(0, 10), (25, 40)]) == 10.0
    labels = ["EF", "F", "N", "G", "EG"]
    assert [T.band(v, labels) for v in (0, 19.99, 20, 59.9, 80, 100)] == ["EF", "EF", "F", "N", "EG", "EG"]
    assert T.ema(s([10, 20, 30]), 3).tolist() == [10.0, 15.0, 22.5]


def test_share_pct_diff_and_event_window_mean():
    df = pd.DataFrame({"neg": [44.0, 0.0], "pos": [72.0, 0.0]}, index=IDX[:2])
    assert T.c_share(df).iloc[0] == pytest.approx(44 / 116 * 100) and np.isnan(T.c_share(df).iloc[1])
    assert T.c_pct_diff(pd.DataFrame({"pe": [22.0], "avg": [20.0]}, index=IDX[:1])).iloc[0] == pytest.approx(10.0)
    cal = pd.bdate_range("2026-01-01", "2026-03-31")
    ev = pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-02-20"]), "value": [2.0, 4.0, 9.0]})
    m = T.event_window_mean(ev, cal, days=30, min_events=2)
    assert np.isnan(m.loc["2026-01-05"]) and m.loc["2026-01-06"] == pytest.approx(3.0)
    assert np.isnan(m.loc["2026-02-20"])            # earlier events aged out, only one left
