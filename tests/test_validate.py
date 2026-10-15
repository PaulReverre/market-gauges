from __future__ import annotations

from datetime import date

from gauges.schema import Obs, coerce
from gauges.validate import CROSSCHECKS, apply_bounds, run_crosschecks

TODAY = date(2026, 10, 15)


def test_coerce_rejects_bad_rows_and_dedupes():
    rows = [
        Obs("cboe", "VIX", date(2026, 10, 14), 16.0, None, "e"),
        Obs("cboe", "VIX", date(2026, 10, 14), 16.87, None, "e"),            # duplicate key: last wins
        Obs("cboe", "VIX", date(2026, 10, 13), float("nan"), None, "e"),
        Obs("cboe", "VIX", date(2026, 10, 20), 15.0, None, "e"),             # future
        Obs("cboe", "VIX", date(2026, 10, 12), 15.0, None, ""),              # no endpoint
    ]
    valid, rejected = coerce(rows, TODAY)
    assert valid["value"].tolist() == [16.87]
    assert sorted(rejected["reason"]) == ["missing endpoint", "non-finite value", "obs_date in the future"]


def test_bounds_reject_and_report_unmatched(cfg):
    rows = [Obs("fred", "VIXCLS", date(2026, 10, 14), 16.87, None, "e"),
            Obs("fred", "VIXCLS", date(2026, 10, 13), 400.0, None, "e"),     # outside [5, 150]
            Obs("nowhere", "x", date(2026, 10, 14), 1.0, None, "e")]
    valid, _ = coerce(rows, TODAY)
    kept, rejected, unmatched = apply_bounds(valid, cfg)
    assert len(kept) == 2 and rejected["value"].tolist() == [400.0]
    assert "outside bounds [5, 150]" in rejected["reason"].iloc[0]
    assert unmatched == {("nowhere", "x")}


def test_crosscheck_tolerances():
    spx, vix, tenyr, tsmc = CROSSCHECKS
    assert spx.compare(7728.0, 7718.94)[1] and not spx.compare(7735.0, 7718.94)[1]   # +0.12% vs +0.21%
    assert vix.compare(16.87, 16.87)[1] and not vix.compare(16.87, 16.88)[1]
    assert tenyr.compare(5.21, 5.19)[1] and not tenyr.compare(5.21, 5.18)[1]
    assert tsmc.compare(511857.0, 511857399.0)[1]


def test_run_crosschecks_flags_mismatch(store, fetched_at):
    def put(source, series, values):
        rows = [Obs(source, series, d, v, None, "e") for d, v in values.items()]
        store.append_observations(coerce(rows, TODAY)[0], "r1", fetched_at)

    put("cboe", "VIX", {date(2026, 10, 13): 17.10, date(2026, 10, 14): 16.87})
    put("fred", "VIXCLS", {date(2026, 10, 13): 17.10, date(2026, 10, 14): 16.95})
    put("fred", "DGS10", {date(2026, 10, 13): 5.21})
    put("treasury", "10y", {date(2026, 10, 13): 5.21, date(2026, 10, 14): 5.19})
    results = {r.name: r for r in run_crosschecks(store, today=TODAY)}
    assert results["VIX close"].ok is False and "1/2 dates" in results["VIX close"].detail
    assert results["10y yield"].ok is True and results["10y yield"].obs_date == date(2026, 10, 13)
    assert results["SPX close"].ok is None
