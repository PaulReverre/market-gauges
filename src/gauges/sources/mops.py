"""TWSE MOPS monthly revenue (NT$ thousands): TSMC and AI server ODMs.

Two tables per month: _0 (domestic listings) and _1 (foreign/KY listings such
as Alchip 3661). Pages are cp950-encoded; URLs use ROC years (2026 = 115).
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from gauges.http import get, month_start, strip_tags
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult

SOURCE = "mops"
URL = "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{roc}_{month}_{table}.html"


def parse_table(html: str, tickers: dict[str, str], year: int, month: int, endpoint: str) -> list[Obs]:
    """Column 2 is the current month's revenue."""
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = [strip_tags(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if len(cells) > 2 and cells[0] in tickers and re.fullmatch(r"[\d,]+", cells[2]):
            out.append(Obs(SOURCE, f"{cells[0]}.revenue_ntd_k", month_start(year, month),
                           float(cells[2].replace(",", "")), None, endpoint))
    return out


def _month(args: tuple[int, int, dict[str, str]]) -> list[Obs]:
    year, month, tickers = args
    out = []
    for table in (0, 1):
        url = URL.format(roc=year - 1911, month=month, table=table)
        r = get(url)
        if r.status_code == 200:
            out += parse_table(r.content.decode("cp950", "replace"), tickers, year, month, url)
    return out


def months_between(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch(ctx: Context) -> FetchResult:
    cfg = ctx.cfg.fetch.mops
    last = ctx.last_date(SOURCE, "2330.revenue_ntd_k")
    start = last.replace(day=1) if last else cfg.start            # re-read the latest month for revisions
    prev = ctx.today.replace(day=1)
    months = [ym for ym in months_between(start, prev) if ym != (prev.year, prev.month)]
    rows: list[Obs] = []
    with ThreadPoolExecutor(4) as pool:
        for month_rows in pool.map(_month, [(y, m, cfg.tickers) for y, m in months]):
            rows += month_rows
    found = {r.series.split(".")[0] for r in rows}
    notes = [f"{len(months)} months requested from {start:%Y-%m}"]
    missing = set(cfg.tickers) - found
    if months and missing:
        notes.append(f"tickers with no rows: {sorted(missing)}")
    return FetchResult(rows, notes=notes)
