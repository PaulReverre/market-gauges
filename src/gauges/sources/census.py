"""Census Business Trends and Outlook Survey (BTOS): AI use rates."""
from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO

from openpyxl import load_workbook

from gauges.http import SourceError, get_ok
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult

BTOS_URL = "https://www.census.gov/hfp/btos/downloads/National.xlsx"


def parse_btos(content: bytes) -> list[Obs]:
    """'Yes' share for every AI question; obs_date = reference period end, vintage = publication date.

    The AI-use question was broadened on 2025-11-17: treat as a series break.
    """
    wb = load_workbook(BytesIO(content), read_only=True, data_only=True)
    dates = {}
    for r in wb["Collection and Reference Dates"].iter_rows(min_row=2, values_only=True):
        if r and r[3] and isinstance(r[7], datetime):
            dates[str(r[3])] = (r[7].date(), r[8].date() if isinstance(r[8], datetime) else None)
    rows = wb["Response Estimates"].iter_rows(values_only=True)
    periods = [str(p) if p else None for p in next(rows)[4:]]
    out = []
    for r in rows:
        if not r or "Artificial Intelligence" not in str(r[1] or "") or str(r[3] or "").strip() != "Yes":
            continue
        series = f"ai_q{r[0]}_yes"
        for period, cell in zip(periods, r[4:]):
            m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%?\s*", str(cell or ""))
            if period in dates and m:
                ref_end, published = dates[period]
                out.append(Obs("btos", series, ref_end, float(m.group(1)), published, BTOS_URL))
    return out


def fetch(ctx: Context) -> FetchResult:
    rows = parse_btos(get_ok(BTOS_URL, what="census btos").content)
    if not rows:
        raise SourceError("no AI question rows found in National.xlsx")
    return FetchResult(rows, notes=[f"{len({r.series for r in rows})} AI questions; series break 2025-11-17"])
