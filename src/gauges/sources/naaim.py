"""NAAIM Exposure Index (weekly xlsx, history since 2006)."""
from __future__ import annotations

from datetime import datetime, timedelta
from io import BytesIO

from openpyxl import load_workbook

from gauges.http import SourceError, get_ok
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult

NAAIM_MEDIA = "https://naaim.org/wp-json/wp/v2/media"

NAAIM_COLUMNS = {"Mean/Average": "mean", "Quart 2 (median)": "median", "Standard Deviation": "stdev"}


def parse_naaim(content: bytes, endpoint: str) -> list[Obs]:
    rows = load_workbook(BytesIO(content), read_only=True, data_only=True).active.iter_rows(values_only=True)
    header = [str(h).strip() if h else "" for h in next(rows)]
    idx = {name: header.index(name) for name in NAAIM_COLUMNS if name in header}
    out = []
    for r in rows:
        if not r or not isinstance(r[0], datetime):
            continue
        d = r[0].date()
        for name, i in idx.items():
            if isinstance(r[i], (int, float)):
                # Survey taken Wednesday, published Thursday.
                out.append(Obs("naaim", NAAIM_COLUMNS[name], d, float(r[i]), d + timedelta(days=1), endpoint))
    return out


def fetch(ctx: Context) -> FetchResult:
    media = get_ok(NAAIM_MEDIA, params={"search": "USE_Data", "per_page": 1, "orderby": "date", "order": "desc"},
                   what="naaim media").json()
    if not media:
        raise SourceError("no USE_Data spreadsheet found")
    url = media[0]["source_url"]
    rows = parse_naaim(get_ok(url, what="naaim xlsx").content, url)
    latest = max(r.obs_date for r in rows)
    notes = [f"file {url.rsplit('/', 1)[-1]}, latest {latest}"]
    if (ctx.today - latest).days > 10:
        notes.append(f"STALE: newest NAAIM data is {(ctx.today - latest).days} days old")
    return FetchResult(rows, notes=notes)
