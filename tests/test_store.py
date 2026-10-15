from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from gauges.schema import Obs, coerce

TODAY = date(2026, 10, 15)


def frame(rows):
    valid, rejected = coerce(rows, TODAY)
    assert rejected.empty
    return valid


def obs(series="VIX", d=date(2026, 10, 14), value=16.87, vintage=None, source="cboe"):
    return Obs(source, series, d, value, vintage, "test://endpoint")


def test_append_is_idempotent(store, fetched_at):
    rows = frame([obs(), obs(d=date(2026, 10, 13), value=17.1)])
    assert store.append_observations(rows, "r1", fetched_at) == 2
    assert store.append_observations(rows, "r2", fetched_at) == 0
    assert store.con.execute("select count(*) from raw_observations").fetchone()[0] == 2


def test_revision_appends_and_latest_view_returns_it(store, fetched_at):
    store.append_observations(frame([obs(value=16.87)]), "r1", fetched_at)
    later = datetime(2026, 10, 16, 23, 0)
    assert store.append_observations(frame([obs(value=16.90)]), "r2", later) == 1
    assert store.con.execute("select count(*) from raw_observations").fetchone()[0] == 2
    assert store.latest_value("cboe", "VIX") == (date(2026, 10, 14), 16.90)
    # Re-sending the revised value is a no-op; reverting to the original is a new revision.
    assert store.append_observations(frame([obs(value=16.90)]), "r3", later) == 0
    assert store.append_observations(frame([obs(value=16.87)]), "r4", datetime(2026, 10, 17)) == 1


def test_vintages_are_separate_keys(store, fetched_at):
    rows = frame([obs("PCEPILFE", date(2026, 8, 1), 130.4, date(2026, 9, 26), "alfred"),
                  obs("PCEPILFE", date(2026, 8, 1), 130.455, date(2026, 10, 10), "alfred")])
    assert store.append_observations(rows, "r1", fetched_at) == 2
    assert store.append_observations(rows, "r2", fetched_at) == 0
    assert store.series_points("alfred", "PCEPILFE") == {date(2026, 8, 1): 130.455}


def test_last_dates(store, fetched_at):
    store.append_observations(frame([obs(), obs(d=date(2026, 10, 1))]), "r1", fetched_at)
    assert store.last_dates() == {("cboe", "VIX"): date(2026, 10, 14)}


def test_group_snapshot_written_once_per_key(store, fetched_at):
    chain = pd.DataFrame({"snapshot_date": [date(2026, 10, 15)] * 2, "underlying": ["NVDA"] * 2,
                          "expiry": [date(2026, 11, 20)] * 2, "option_type": ["call", "put"], "strike": [200.0, 200.0],
                          "implied_vol": [0.45, 0.47]})
    assert store.append_snapshot("option_snapshots", chain, "r1", fetched_at) == 2
    assert store.append_snapshot("option_snapshots", chain.assign(implied_vol=0.5), "r2", fetched_at) == 0


def test_row_snapshot_appends_changed_rows(store, fetched_at):
    event = pd.DataFrame({"ticker": ["NVDA"], "report_date": [date(2026, 11, 17)], "source": ["finnhub_calendar"],
                          "eps_actual": [None], "eps_estimate": [2.5253]})
    assert store.append_snapshot("earnings_events", event, "r1", fetched_at) == 1
    assert store.append_snapshot("earnings_events", event, "r2", fetched_at) == 0
    assert store.append_snapshot("earnings_events", event.assign(eps_actual=2.61), "r3", fetched_at) == 1
