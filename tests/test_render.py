"""Dashboard JSON helpers."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from gauges import render as R
from gauges.registry import Indicator


def ind(**kw) -> Indicator:
    base = dict(id="x", gauge="G1", pillar="volatility", name="x", inputs=["cboe:VIX"], direction="lower_is_greed",
                weight=1, expected_freq="1d", role="leading")
    return Indicator.model_validate({**base, **kw})


def test_next_refresh_follows_the_cron():
    # Thursday 2026-10-15 20:41 UTC -> Friday 02:00 UTC (Tue-Sat only)
    assert R.next_refresh(datetime(2026, 10, 15, 20, 41, tzinfo=timezone.utc)) == datetime(2026, 10, 16, 2, tzinfo=timezone.utc)
    # Saturday after the run -> Tuesday 02:00 UTC
    assert R.next_refresh(datetime(2026, 10, 17, 3, 0, tzinfo=timezone.utc)) == datetime(2026, 10, 20, 2, tzinfo=timezone.utc)


def test_drawdown_periods_peak_to_recovery():
    idx = pd.bdate_range("2026-01-01", periods=8)
    price = pd.Series([100, 95, 85, 90, 101, 99, 92, 95], index=idx, dtype=float)
    # 100 -> 85 (-15%) recovered on day 5; 101 -> 92 (-8.9%) never reaches -10%
    assert R.drawdown_periods(price) == [[idx[0].date().isoformat(), idx[4].date().isoformat()]]
    open_dd = pd.Series([100, 80, 85], index=idx[:3], dtype=float)
    assert R.drawdown_periods(open_dd) == [[idx[0].date().isoformat(), idx[2].date().isoformat()]]


def test_badges_need_every_input_cross_checked():
    row = pd.Series({"stale": 1.0, "short_history": False})
    assert R.badges(ind(inputs=["cboe:VIX"]), row, set()) == ["verified"]
    assert R.badges(ind(inputs=["cboe:VIX", "cboe:VIX3M"]), row, set()) == ["single-source"]
    assert R.badges(ind(inputs=["cboe:VIX"]), row, {"cboe:VIX"}) == ["single-source"]   # failed today's cross-check
    stale = pd.Series({"stale": 0.4, "short_history": True})
    assert R.badges(ind(inputs=["aaii:bullish"]), stale, set()) == ["single-source", "stale", "short-history"]
    event = ind(kind="event", inputs=[], gauge="G4", pillar="spend", direction="higher_is_healthy",
                event={"kpi": "sp_guide_next", "categories": {"raise": 85}})
    assert R.badges(event, None, set()) == ["manual"]


def test_source_labels_and_links():
    label, url = R.source_of(ind(inputs=["fred:BAMLH0A0HYM2", "alfred:NFCI"]))
    assert label == "FRED BAMLH0A0HYM2 + FRED NFCI (ALFRED vintages)"
    assert url == "https://fred.stlouisfed.org/series/BAMLH0A0HYM2"
    assert R.source_of(ind(inputs=["yfinance:^GSPC.close"]))[1] == "https://finance.yahoo.com/quote/^GSPC"
    assert R.source_of(ind(kind="panel", inputs=[], panel="xbrl_aggregate"))[0] == "SEC EDGAR XBRL company facts"


def test_clean_makes_json_safe_values():
    values = [float("nan"), np.float64(1.234567), np.int64(3), np.bool_(True), pd.Timestamp("2026-10-14"),
              date(2026, 10, 14), None, "x"]
    out = [R._clean(v) for v in values]
    assert out == [None, 1.2346, 3, True, "2026-10-14", "2026-10-14", None, "x"]
    json.dumps(out)
    assert R._round_list([1.26, math.nan, None]) == [1.3, None, None]


def test_stretch_reports_window_means_and_z():
    idx = pd.date_range("2000-01-01", periods=27 * 12, freq="MS")
    s = pd.Series(np.r_[np.full(len(idx) - 1, 20.0) + np.tile([-1.0, 1.0], (len(idx) - 1) // 2 + 1)[:len(idx) - 1], 30.0],
                  index=idx)
    out = R.stretch(s, "CAPE", "x", "test", "https://example.com")
    w10 = out["windows"]["10y"]
    assert out["value"] == 30.0 and set(out["windows"]) == {"5y", "10y", "20y", "all"}
    window = s[s.index > idx[-1] - pd.DateOffset(years=10)]
    assert w10["z"] == pytest.approx(round((30 - window.mean()) / window.std(), 2))
    assert w10["pct"] > 99
    assert R.stretch(s.iloc[:5], "x", "x", "t", "u") is None
