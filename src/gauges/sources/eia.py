"""EIA API v2: monthly commercial electricity sales (million kWh) for data-center-heavy states."""
from __future__ import annotations

from datetime import date

from gauges.http import get_ok, month_start, require
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult

SOURCE = "eia"
URL = "https://api.eia.gov/v2/electricity/retail-sales/data/"
PAGE = 5000


def parse_rows(rows: list[dict]) -> list[Obs]:
    out = []
    for row in rows:
        if row.get("sales") in (None, ""):
            continue
        year, month = map(int, row["period"].split("-"))
        out.append(Obs(SOURCE, f"sales_com.{row['stateid']}", month_start(year, month), float(row["sales"]), None, URL))
    return out


def fetch(ctx: Context) -> FetchResult:
    key = require("EIA_API_KEY")
    last = ctx.last_date_prefix(SOURCE, "sales_com.")
    start = date(last.year - 1, last.month, 1) if last else date(2001, 1, 1)   # re-read a year of revisions
    params = {"api_key": key, "frequency": "monthly", "data[0]": "sales", "facets[sectorid][]": "COM",
              "facets[stateid][]": ctx.cfg.fetch.eia.states, "start": f"{start:%Y-%m}",
              "sort[0][column]": "period", "sort[0][direction]": "asc", "length": PAGE}
    rows, offset = [], 0
    while True:
        payload = get_ok(URL, params={**params, "offset": offset}, what="eia retail-sales").json()["response"]
        page = payload.get("data", [])
        rows += parse_rows(page)
        offset += len(page)
        if not page or offset >= int(payload.get("total", 0)):
            break
    return FetchResult(rows, notes=[f"from {start:%Y-%m}"])
