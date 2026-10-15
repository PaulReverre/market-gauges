"""Robert Shiller's ie_data.xls (monthly): CAPE, price, dividends, earnings, CPI, GS10.

The file moved from yale.edu (frozen since 2023-10) to shillerdata.com, whose
download link carries a GUID, so it is scraped each run. The newest month is
preliminary (e.g. "Oct price is Oct 2nd close") and is revised as it completes.
"""
from __future__ import annotations

import re

import xlrd

from gauges.http import SourceError, get_ok, month_start
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult

SOURCE = "shiller"
PAGE = "https://shillerdata.com/"
COLUMNS = {"P": "price", "D": "dividend", "E": "earnings", "CPI": "cpi", "Rate GS10": "gs10", "CAPE": "cape"}


def parse_date(x: float) -> tuple[int, int]:
    """Shiller writes months as decimals: 2026.01 is January, 2026.1 is October."""
    year = int(x)
    return year, int(round((x - year) * 100))


def parse_xls(content: bytes, endpoint: str) -> list[Obs]:
    sheet = xlrd.open_workbook(file_contents=content).sheet_by_name("Data")
    header_row = next(i for i in range(min(20, sheet.nrows)) if str(sheet.cell_value(i, 0)).strip() == "Date")
    header = [str(v).strip() for v in sheet.row_values(header_row)]
    idx = {name: header.index(col) for col, name in COLUMNS.items() if col in header}
    out = []
    for i in range(header_row + 1, sheet.nrows):
        row = sheet.row_values(i)
        if not isinstance(row[0], float):
            continue
        year, month = parse_date(row[0])
        if not 1 <= month <= 12:
            continue
        d = month_start(year, month)
        for name, j in idx.items():
            if isinstance(row[j], float):
                out.append(Obs(SOURCE, name, d, row[j], None, endpoint))
    return out


def fetch(ctx: Context) -> FetchResult:
    m = re.search(r"(?:https?:)?//img1\.wsimg\.com/[^\"']+/ie_data\.xls[^\"']*", get_ok(PAGE, what="shillerdata.com").text)
    if not m:
        raise SourceError("ie_data.xls link not found on shillerdata.com")
    url = m.group(0) if m.group(0).startswith("http") else "https:" + m.group(0)
    rows = parse_xls(get_ok(url, what="ie_data.xls").content, url.split("?")[0])
    return FetchResult(rows, notes=[f"latest month {max(r.obs_date for r in rows)} (preliminary)"])
