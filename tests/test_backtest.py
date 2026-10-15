"""Backtest targets, statistics, verdict rules and registry edits."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gauges import backtest as B
from gauges.registry import Indicator

IDX = pd.bdate_range("2000-01-03", periods=2000)


def test_forward_return_and_max_drawdown():
    p = pd.Series([100, 110, 99, 104.5, 120], index=IDX[:5], dtype=float)
    assert B.forward_return(p, 2).iloc[0] == pytest.approx(-1.0)
    dd = B.forward_max_drawdown(p, 2)
    # window [100, 110, 99]: peak 110 -> trough 99 = -10%
    assert dd.iloc[0] == pytest.approx(-10.0)
    assert dd.iloc[2] == pytest.approx(0.0)          # 99, 104.5, 120 never falls
    assert dd.iloc[3:].isna().all()                  # window runs past the data


def test_targets_flag_ten_percent_drawdowns():
    p = pd.Series(np.r_[np.full(10, 100.0), np.full(70, 89.0)], index=IDX[:80])
    tg = B.targets(p)
    assert tg["dd10"].iloc[0] == 1.0 and tg["mdd63"].iloc[0] == pytest.approx(-11.0)
    assert tg["dd10"].iloc[12] == 0.0


def test_spearman_and_nonoverlap():
    x = pd.Series(np.arange(300, dtype=float), index=IDX[:300])
    y = x * 2
    rho, n = B.spearman(x, y)
    assert rho == pytest.approx(1.0) and n == 300
    rho_no, n_no, t = B.nonoverlap_t(x, -x + np.sin(x), 10)
    assert n_no == 30 and rho_no == pytest.approx(-1.0) and math.isnan(t)   # |rho| = 1 has no finite t
    assert math.isnan(B.spearman(x.iloc[:10], y.iloc[:10])[0])              # too few points


def test_decile_table_and_monotonicity():
    rng = np.random.default_rng(1)
    score = pd.Series(rng.uniform(0, 100, 1000), index=IDX[:1000])
    fwd = score / 10 + rng.normal(0, 0.1, 1000)
    tg = pd.DataFrame({"fwd21": fwd, "fwd63": fwd, "fwd126": fwd, "mdd63": -fwd, "dd10": (score > 90).astype(float)})
    t = B.decile_table(score, tg)
    assert len(t) == 10 and t["days"].sum() == 1000
    assert B.monotonicity(t, "fwd63") == pytest.approx(1.0)
    assert t.loc[10, "dd10"] == 100.0 and t.loc[1, "dd10"] == 0.0


def ind(**kw) -> Indicator:
    base = dict(id="x", gauge="G1", pillar="volatility", name="x", inputs=["cboe:VIX"], direction="lower_is_greed",
                weight=1, expected_freq="1d", role="contrarian")
    return Indicator.model_validate({**base, **kw})


def result(role="contrarian", ic=-0.1, regimes=None, years=10.0, ic_ret=-0.1, t_ret=-1.0, ic_mdd=0.0, t_mdd=0.0,
           ic_126=0.0, t_126=0.0, gauge="G1") -> B.IndicatorResult:
    fam = B.family(gauge)
    target, sign = B.EXPECTED[(fam, role)] if role != "display" else ("fwd63", -1)
    return B.IndicatorResult(ind(role=role, gauge=gauge, pillar="volatility" if gauge == "G1" else "growth",
                                 direction="lower_is_greed" if fam == "greed" else "higher_is_healthy"),
                             target, sign, ic, int(years * 252), ic, -1.0, ic_ret, t_ret, ic_mdd, t_mdd, ic_126, t_126,
                             regimes if regimes is not None else {"2008": -0.2, "2020": -0.1}, years)


def test_rules_keep_demote_relabel_and_low_confidence():
    r = result()
    B.decide(r, "greed")
    assert r.verdict == "keep"
    r = result(ic=0.05)                                         # contrarian with a positive IC: wrong sign
    B.decide(r, "greed")
    assert (r.verdict, r.new_role) == ("demote", "display")
    r = result(ic=-0.1, regimes={"a": 0.2, "b": 0.1, "c": -0.1})  # holds in 1 of 3 regimes
    B.decide(r, "greed")
    assert r.verdict == "demote" and "1/3" in r.reason
    r = result(role="leading", ic=-0.1, ic_ret=-0.2, t_ret=-2.5)  # leading fails, but clearly contrarian
    B.decide(r, "greed")
    assert (r.verdict, r.new_role) == ("relabel", "contrarian")
    r = result(ic=0.3, years=2.0)
    B.decide(r, "greed")
    assert r.verdict == "keep (low confidence)"
    r = result(role="display", ic=0.3)
    B.decide(r, "greed")
    assert r.verdict == "display"


def test_fragility_validation_thresholds():
    table = pd.DataFrame({"dd10": np.linspace(10, 40, 10), "days": 100}, index=range(1, 11))
    res = B.CompositeResult("f", "S&P 500", table, {}, {}, base_dd10=20.0)
    ok, detail = B.validate_fragility(res)
    assert ok and "2.00x" in detail
    flat = B.CompositeResult("f", "S&P 500", table.assign(dd10=[20.0] * 9 + [26.0]), {}, {}, base_dd10=20.0)
    assert not B.validate_fragility(flat)[0]


def test_edit_registry_changes_only_the_target_block():
    text = ("indicators:\n  - id: a\n    role: leading\n    fragility: higher_is_fragile\n    notes: keep me\n"
            "  - id: b\n    role: leading\n    fragility: higher_is_fragile\n")
    out = B.edit_registry(text, "a", {"role": "display", "fragility": None, "validation": '"why"'})
    assert out == ("indicators:\n  - id: a\n    role: display\n    notes: keep me\n    validation: \"why\"\n"
                   "  - id: b\n    role: leading\n    fragility: higher_is_fragile\n")
    with pytest.raises(KeyError):
        B.edit_registry(text, "zzz", {"role": "display"})


def test_regime_ics_need_enough_observations():
    idx = pd.bdate_range("2019-06-03", "2021-06-30")
    x = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    out = B.regime_ics(x, x)
    assert set(out) == {"2020"} and out["2020"] == pytest.approx(1.0)
