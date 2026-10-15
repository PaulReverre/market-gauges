"""ALFRED: every vintage of revised macro series (vintage = the date a value became current).

First run: one output_type=1 pull per series, which returns a row only when a
value changes. Later runs ask for vintage dates newer than the newest stored
one, skip series with none, and for each new vintage keep only the values that
differ from the previous vintage's snapshot (the same compact representation).
NFCI and STLFSI4 revise their full history weekly, so a full re-pull would be
about a million rows each time.
"""
from __future__ import annotations

from datetime import date

from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult, each
from gauges.sources.fred import api, observations_all, parse_observations

SOURCE = "alfred"
ENDPOINT = "fred/series/observations?series_id={series}&realtime_start={vintage}"


def new_vintages(series: str, after: date) -> list[date]:
    """Vintage dates after `after`. The query starts at `after` itself because FRED answers
    HTTP 500 when no vintage follows realtime_start, and HTTP 400 when it is in the future."""
    payload = api("series/vintagedates", series_id=series, realtime_start=after.isoformat())
    return [d for d in map(date.fromisoformat, payload.get("vintage_dates", [])) if d > after]


def snapshot(series: str, vintage: date) -> dict[date, float]:
    """The series as it stood on `vintage`."""
    rows = observations_all(series_id=series, realtime_start=vintage.isoformat(), realtime_end=vintage.isoformat())
    return {date.fromisoformat(o["date"]): float(o["value"]) for o in rows if o["value"] not in (".", "")}


def diff_vintages(series: str, previous: dict[date, float], current: dict[date, float], vintage: date) -> list[Obs]:
    """Rows that are new or changed in `current` relative to `previous`."""
    endpoint = ENDPOINT.format(series=series, vintage=vintage)
    return [Obs(SOURCE, series, d, v, vintage, endpoint) for d, v in sorted(current.items())
            if d not in previous or abs(previous[d] - v) > 1e-12]


def fetch(ctx: Context) -> FetchResult:
    notes: list[str] = []
    skipped: list[str] = []

    def one(series: str) -> list[Obs]:
        last = ctx.last_vintage(SOURCE, series)
        if last is None:
            rows = observations_all(series_id=series, realtime_start="1776-07-04", realtime_end="9999-12-31",
                                    output_type=1)
            return parse_observations(rows, SOURCE, series, with_vintage=True)
        vintages = new_vintages(series, last)
        if not vintages:
            skipped.append(series)
            return []
        out, previous = [], snapshot(series, last)
        for v in vintages:
            current = snapshot(series, v)
            out += diff_vintages(series, previous, current, v)
            previous = current
        return out

    rows = each(ctx.cfg.fetch.alfred.series, one, notes)
    if skipped:
        notes.append(f"no new vintages: {len(skipped)} series")
    return FetchResult(rows, notes=notes)
