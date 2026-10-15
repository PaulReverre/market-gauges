"""GDELT 2.0 DOC API: average news tone per day for a few AI/chip queries.

Noisy and sparse (the 7-day timeline can have multi-day gaps), so it carries
low weight. Buckets are averaged per UTC day; today's partial day is dropped.
GDELT documents one request every 5 seconds but in practice throttles harder,
so requests are spaced by hand instead of using the shared retry policy.
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import date, datetime, timezone

from gauges.http import SourceError, Throttle, client
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult, each

SOURCE = "gdelt"
URL = "https://api.gdeltproject.org/api/v2/doc/doc"
THROTTLE = Throttle(20.0)   # documented limit is 5s, but 8s-spaced requests still drew 429s


def parse_timeline(payload: dict, slug: str, before: date) -> list[Obs]:
    by_day: dict[date, list[float]] = defaultdict(list)
    for timeline in payload.get("timeline", []):
        for point in timeline.get("data", []):
            day = datetime.strptime(point["date"], "%Y%m%dT%H%M%SZ").date()
            if day < before:
                by_day[day].append(float(point["value"]))
    return [Obs(SOURCE, f"tone.{slug}", d, sum(v) / len(v), None, URL) for d, v in sorted(by_day.items())]


def _request(query: str):
    params = {"query": query, "mode": "timelinetone", "format": "json", "timespan": "7d"}
    for _ in range(2):
        THROTTLE.wait()
        r = client().get(URL, params=params)
        if r.status_code != 429:
            return r
        time.sleep(30)
    return r


def fetch(ctx: Context) -> FetchResult:
    utc_today = datetime.now(timezone.utc).date()
    notes: list[str] = []

    def one(item: tuple[str, str]) -> list[Obs]:
        slug, query = item
        r = _request(query)
        if r.status_code != 200:
            raise SourceError(f"HTTP {r.status_code}")
        try:
            return parse_timeline(r.json(), slug, utc_today)
        except ValueError as e:
            raise SourceError(f"non-JSON reply: {r.text[:80]}") from e

    rows = each(ctx.cfg.fetch.gdelt.queries.items(), one, notes, label=lambda item: item[0])
    return FetchResult(rows, notes=notes)
