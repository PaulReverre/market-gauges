"""Scoring, composites and point-in-time feature plumbing on synthetic data."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gauges import score as S
from gauges.config import Event, load_series_config
from gauges.features import Features
from gauges.registry import Indicator, Registry, load_registry
from gauges.schema import Obs, coerce

CAL = pd.bdate_range("2026-01-01", periods=300)


def ind(**kw) -> Indicator:
    base = dict(id="x", gauge="G1", pillar="volatility", name="x", inputs=["cboe:VIX"], direction="lower_is_greed",
                weight=1, expected_freq="1d", role="leading")
    return Indicator.model_validate({**base, **kw})


def known(values, avail=None):
    v = pd.Series(values, index=CAL[:len(values)], dtype=float).reindex(CAL)
    a = pd.Series(CAL, index=CAL) if avail is None else avail
    return pd.DataFrame({"value": v, "as_of": a, "avail": a})


def test_score_orients_and_needs_history():
    rising = known(np.arange(300, dtype=float))
    f = S.score_indicator(ind(direction="lower_is_greed"), rising, CAL)
    assert f["score"].iloc[: S.MIN_HISTORY - 1].isna().all()
    assert f["pct"].iloc[-1] == pytest.approx(100 * (300 - 0.5) / 300)      # newest value is the max
    assert f["score"].iloc[-1] == pytest.approx(100 - f["pct"].iloc[-1])    # lower is greed -> near 0
    assert f["mult"].iloc[-1] == 1.0


def test_bounds_reject_and_staleness_decay():
    k = known(np.arange(300, dtype=float))
    k.loc[CAL[-1], "value"] = 5_000.0
    f = S.score_indicator(ind(bounds=(0, 1000)), k, CAL)
    assert f["rejected"].iloc[-1] and np.isnan(f["score"].iloc[-1])
    stale = known(np.arange(300, dtype=float), avail=pd.Series(CAL[0], index=CAL))
    g = S.score_indicator(ind(expected_freq="1w"), stale, CAL)
    assert g["stale"].iloc[7] == 1.0 and g["stale"].iloc[11] == pytest.approx((15 - 11) / 7.5)
    assert g["stale"].iloc[-1] == 0.0 and g["mult"].iloc[-1] == 0.0


def _registry(extra_g1=None) -> Registry:
    g = {"name": "G", "scale": ["a", "b", "c", "d", "e"], "smoothing": "none"}
    inds = [
        {"id": "a1", "gauge": "G1", "pillar": "p", "name": "a1", "inputs": ["cboe:VIX"], "direction": "higher_is_greed",
         "weight": 3, "expected_freq": "1d", "role": "leading"},
        {"id": "a2", "gauge": "G1", "pillar": "p", "name": "a2", "inputs": ["cboe:VIX"], "direction": "higher_is_greed",
         "weight": 1, "expected_freq": "1d", "role": "leading"},
        {"id": "b1", "gauge": "G1", "pillar": "q", "name": "b1", "inputs": ["cboe:VIX"], "direction": "higher_is_greed",
         "weight": 1, "expected_freq": "1d", "role": "leading"},
    ] + (extra_g1 or [])
    gauges = {"G1": {**g, "pillars": {"p": 60, "q": 40}}}
    for gid, greed in (("G2", False), ("G3", True), ("G4", False)):
        gauges[gid] = {**g, "pillars": {"z": 100}}
        inds.append({"id": f"z_{gid.lower()}", "gauge": gid, "pillar": "z", "name": "z", "inputs": ["cboe:VIX"],
                     "direction": "higher_is_greed" if greed else "higher_is_healthy", "weight": 1,
                     "expected_freq": "1d", "role": "leading"})
    return Registry.model_validate({"gauges": gauges, "indicators": inds, "fragility": {"divergence": {}}})


def frame(score, mult):
    idx = CAL[:1]
    return pd.DataFrame({"score": [score], "mult": [mult]}, index=idx)


def test_composite_weights_renormalization_and_coverage():
    reg = _registry()
    scored = {"a1": frame(80.0, 1.0), "a2": frame(40.0, 0.5), "b1": frame(np.nan, 0.0)}
    r = S.composite(reg, "G1", scored, CAL[:1])
    pillar_p = (3 * 80 + 0.5 * 40) / 3.5
    assert r.pillars["p"].iloc[0] == pytest.approx(pillar_p)
    assert np.isnan(r.pillars["q"].iloc[0])
    assert r.gauge["score"].iloc[0] == pytest.approx(pillar_p)               # pillar q renormalized away
    assert r.gauge["coverage"].iloc[0] == pytest.approx((60 * 3 / 4 * 1 + 60 * 1 / 4 * 0.5) / 100 * 100)
    assert r.contributions.iloc[0].sum() == pytest.approx(r.gauge["score"].iloc[0])
    assert bool(r.gauge["low_coverage"].iloc[0])                            # 52.5% of target weight < 70%


def test_low_coverage_flag():
    reg = _registry()
    scored = {"a1": frame(80.0, 0.0), "a2": frame(40.0, 1.0), "b1": frame(50.0, 1.0)}
    r = S.composite(reg, "G1", scored, CAL[:1])
    assert r.gauge["coverage"].iloc[0] == pytest.approx(15 + 40)
    assert bool(r.gauge["low_coverage"].iloc[0])


def test_infl_penalty():
    core = pd.Series([2.5, 3.5, 6.0])
    p = S.infl_penalty({"infl": pd.DataFrame({"value": core})})
    assert p.tolist() == [0.0, 5.0, 20.0]


def test_banners():
    g1 = pd.Series([75.0] * 16 + [20.0])
    g2 = pd.Series(list(np.linspace(60, 45, 16)) + [55.0])
    frag = pd.Series([72.0] * 17)
    b = S.banners(g1, g2, frag)
    assert b["complacent_and_fragile"].iloc[15]
    assert b["sentiment_ahead_of_fundamentals"].iloc[15]
    assert b["fear_without_fundamental_damage"].iloc[16]


def test_registry_rejects_inconsistent_entries():
    reg = load_registry()
    assert len(reg.indicators) > 90
    with pytest.raises(ValidationError, match="does not fit"):
        _registry([{"id": "bad", "gauge": "G1", "pillar": "p", "name": "b", "inputs": ["cboe:VIX"],
                    "direction": "higher_is_healthy", "weight": 1, "expected_freq": "1d", "role": "leading"}])
    with pytest.raises(ValidationError, match="not a pillar"):
        _registry([{"id": "bad", "gauge": "G1", "pillar": "nope", "name": "b", "inputs": ["cboe:VIX"],
                    "direction": "higher_is_greed", "weight": 1, "expected_freq": "1d", "role": "leading"}])
    with pytest.raises(ValidationError, match="need inputs"):
        ind(inputs=[])


# ---------------------------------------------------------------- features
@pytest.fixture
def features(store, fetched_at):
    days = pd.bdate_range("2026-01-01", "2026-03-31")
    rows = [Obs("yfinance", "SPY.close", d.date(), 100.0, None, "e") for d in days]
    # A monthly revised series: Jan published Feb 15 (100), revised Mar 1 to 101; Feb published Mar 15 (102).
    rows += [Obs("alfred", "X", date(2026, 1, 1), 100.0, date(2026, 2, 16), "e"),
             Obs("alfred", "X", date(2026, 1, 1), 101.0, date(2026, 3, 2), "e"),
             Obs("alfred", "X", date(2026, 2, 1), 102.0, date(2026, 3, 16), "e"),
             Obs("finra", "debit_margin", date(2026, 1, 1), 5.0, None, "e")]
    store.append_observations(coerce(rows, date(2026, 10, 15))[0], "r1", fetched_at)
    from gauges.config import load_baskets
    return Features(store, load_series_config(), load_baskets(), [], start=date(2026, 1, 1))


def test_vintaged_series_is_point_in_time(features):
    from gauges.registry import InputRef
    k = features.known(InputRef(ref="alfred:X"))
    assert np.isnan(k.loc["2026-02-13", "value"])
    assert k.loc["2026-02-16", "value"] == 100.0 and k.loc["2026-02-27", "value"] == 100.0
    assert k.loc["2026-03-02", "value"] == 101.0                       # revision becomes known on its vintage
    assert k.loc["2026-03-16", "value"] == 102.0
    assert k.loc["2026-03-16", "as_of"] == pd.Timestamp("2026-02-01")


def test_release_lag_delays_non_revised_series(features):
    from gauges.registry import InputRef
    k = features.known(InputRef(ref="finra:debit_margin"))           # lag_days 52 from series.yaml
    assert np.isnan(k.loc["2026-02-20", "value"]) and k.loc["2026-02-23", "value"] == 5.0


def test_to_calendar_ignores_late_stale_prints(features):
    idx = pd.DatetimeIndex(["2026-01-05", "2026-01-07"])
    out = features.to_calendar(pd.Series([2.0, 1.0], index=idx),
                               pd.Series(pd.DatetimeIndex(["2026-01-05", "2025-12-01"]), index=idx))
    assert out.loc["2026-01-08", "value"] == 2.0


def test_event_scores_categories_and_growth(features):
    def ev(**kw):
        base = dict(id="e", kpi="sp_guide_next", entity="Microsoft", value="raise", period="FY2027",
                    as_of=date(2026, 1, 15), status="primary", source_url="https://example.com/x")
        return Event.model_validate({**base, **kw})
    features.events = [ev(), ev(id="e2", entity="Alphabet", value="cut", as_of=date(2026, 2, 2), status="secondhand")]
    spec = ind(id="g", gauge="G4", pillar="spend", kind="event", inputs=[], direction="higher_is_healthy",
               event={"kpi": "sp_guide_next", "categories": {"raise": 85, "hold": 50, "cut": 15}})
    out = features.event_scores(spec)
    assert out.loc["2026-01-20", "score"] == 85.0 and out.loc["2026-01-20", "status_mult"] == 1.0
    assert out.loc["2026-02-03", "score"] == pytest.approx((85 * 1.0 + 15 * 0.6) / 1.6)
    assert out.loc["2026-02-03", "status_mult"] == pytest.approx(0.8)

    features.events = [ev(id="a", kpi="mo_labs", entity="Lab", metric="rev", value=10.0, period="2026Q1"),
                       ev(id="b", kpi="mo_labs", entity="Lab", metric="rev", value=12.5, period="2026Q2",
                          as_of=date(2026, 2, 2))]
    labs = ind(id="l", gauge="G4", pillar="monetization", kind="event", inputs=[], direction="higher_is_healthy",
               event={"kpi": "mo_labs", "growth": True, "score_map": [[0, 30], [25, 60], [50, 80]]})
    out = features.event_scores(labs)
    assert np.isnan(out.loc["2026-01-20", "score"]) and out.loc["2026-02-03", "score"] == pytest.approx(60.0)


def test_business_age():
    dates = pd.DatetimeIndex(["2026-10-12", "2026-10-19"])
    avail = pd.Series(pd.DatetimeIndex(["2026-10-09", "2026-10-09"]), index=dates)
    assert S.business_age(avail, dates).tolist() == [1.0, 6.0]
