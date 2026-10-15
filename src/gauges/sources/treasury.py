"""US Treasury daily par yield curve: secondary source for the DGS10 cross-check."""
from __future__ import annotations

import csv
import io

from gauges.http import get_ok, parse_mdy
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult, each

SOURCE = "treasury"
URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all"
TENORS = {"3 Mo": "3m", "2 Yr": "2y", "10 Yr": "10y"}


def parse_csv(text: str, endpoint: str) -> list[Obs]:
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        for col, series in TENORS.items():
            if row.get(col) not in (None, ""):
                out.append(Obs(SOURCE, series, parse_mdy(row["Date"]), float(row[col]), None, endpoint))
    return out


def fetch(ctx: Context) -> FetchResult:
    last = ctx.last_date(SOURCE, "10y")
    first = last.year if last else ctx.cfg.fetch.treasury.start_year
    notes: list[str] = []

    def one(year: int) -> list[Obs]:
        url = URL.format(year=year)
        text = get_ok(url, params={"type": "daily_treasury_yield_curve", "field_tdr_date_value": year,
                                   "_format": "csv"}, what=f"treasury {year}").text
        return parse_csv(text, url)

    return FetchResult(each(range(first, ctx.today.year + 1), one, notes), notes=notes)
