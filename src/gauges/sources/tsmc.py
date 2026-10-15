"""TSMC investor-relations monthly revenue (NT$ millions), released around the 10th.

Plain curl gets HTTP 403 here; httpx's TLS fingerprint is accepted.
Cross-checked against MOPS ticker 2330 (reported in NT$ thousands).
"""
from __future__ import annotations

import re

from gauges.http import get_ok, month_start, strip_tags
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult, each

SOURCE = "tsmc"
URL = "https://investor.tsmc.com/english/monthly-revenue/{year}"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse_year(html: str, year: int) -> list[Obs]:
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = [strip_tags(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        if len(cells) >= 2 and cells[0][:3] in MONTHS and re.fullmatch(r"[\d,]+", cells[1]):
            d = month_start(year, MONTHS.index(cells[0][:3]) + 1)
            out.append(Obs(SOURCE, "revenue_ntd_m", d, float(cells[1].replace(",", "")), None, URL.format(year=year)))
    return out


def fetch(ctx: Context) -> FetchResult:
    last = ctx.last_date(SOURCE, "revenue_ntd_m")
    first_year = last.year if last else ctx.cfg.fetch.tsmc.start_year
    notes: list[str] = []
    rows = each(range(first_year, ctx.today.year + 1),
                lambda y: parse_year(get_ok(URL.format(year=y), what=f"tsmc {y}").text, y), notes)
    return FetchResult(rows, notes=notes)
