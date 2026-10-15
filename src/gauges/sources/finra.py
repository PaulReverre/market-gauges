"""FINRA margin statistics (monthly, ~3-week release lag)."""
from __future__ import annotations

import re
from io import BytesIO

from openpyxl import load_workbook

from gauges.http import get_ok, month_start
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult

FINRA_PAGE = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"
FINRA_FALLBACK = "/sites/default/files/2021-03/margin-statistics.xlsx"

FINRA_COLUMNS = {1: "debit_margin", 2: "free_credit_cash", 3: "free_credit_margin"}


def parse_finra(content: bytes, endpoint: str) -> list[Obs]:
    out = []
    for r in load_workbook(BytesIO(content), read_only=True, data_only=True).active.iter_rows(min_row=2, values_only=True):
        m = re.fullmatch(r"(\d{4})-(\d{2})", str(r[0] or "").strip())
        if not m:
            continue
        d = month_start(int(m.group(1)), int(m.group(2)))
        for i, name in FINRA_COLUMNS.items():
            if i < len(r) and isinstance(r[i], (int, float)):
                out.append(Obs("finra", name, d, float(r[i]), None, endpoint))
    return out


def fetch(ctx: Context) -> FetchResult:
    page = get_ok(FINRA_PAGE, what="finra page").text
    m = re.search(r"/sites/default/files/[^\"']+margin-statistics[^\"']*\.xlsx", page)
    url = "https://www.finra.org" + (m.group(0) if m else FINRA_FALLBACK)
    return FetchResult(parse_finra(get_ok(url, what="finra xlsx").content, url))
