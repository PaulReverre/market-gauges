"""FRED: series that are not revised after publication (vintage = NULL)."""
from __future__ import annotations

from datetime import date, timedelta

from gauges.http import SourceError, Throttle, get, require
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult, each

SOURCE = "fred"
API = "https://api.stlouisfed.org/fred/"
THROTTLE = Throttle(0.55)  # FRED allows 120 requests/minute
REFETCH_DAYS = 14          # re-read recent days to catch late corrections


def api(path: str, **params) -> dict:
    key = require("FRED_API_KEY")
    THROTTLE.wait()
    r = get(API + path, params={**params, "api_key": key, "file_type": "json"})
    if r.status_code != 200:
        try:
            message = r.json().get("error_message", "")
        except ValueError:
            message = r.text[:120]
        raise SourceError(f"FRED {path} {params.get('series_id', '')}: HTTP {r.status_code} {message}")
    return r.json()


def observations_all(**params) -> list[dict]:
    """Page through series/observations (100k rows per page)."""
    rows, offset = [], 0
    while True:
        payload = api("series/observations", **params, offset=offset, limit=100000)
        page = payload.get("observations", [])
        rows += page
        offset += len(page)
        if not page or offset >= int(payload.get("count", 0)):
            return rows


def parse_observations(rows: list[dict], source: str, series: str, *, with_vintage: bool) -> list[Obs]:
    endpoint = f"fred/series/observations?series_id={series}"
    out = []
    for o in rows:
        if o.get("value") in (".", "", None):
            continue
        vintage = date.fromisoformat(o["realtime_start"]) if with_vintage else None
        out.append(Obs(source, series, date.fromisoformat(o["date"]), float(o["value"]), vintage, endpoint))
    return out


def fetch(ctx: Context) -> FetchResult:
    notes: list[str] = []

    def one(series: str) -> list[Obs]:
        last = ctx.last_date(SOURCE, series)
        params = {"series_id": series}
        if last:
            params["observation_start"] = (last - timedelta(days=REFETCH_DAYS)).isoformat()
        return parse_observations(observations_all(**params), SOURCE, series, with_vintage=False)

    return FetchResult(each(ctx.cfg.fetch.fred.series, one, notes), notes=notes)
