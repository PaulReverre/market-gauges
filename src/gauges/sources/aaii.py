"""AAII Sentiment Survey: public recent-weeks table, accumulated weekly."""
from __future__ import annotations

import re
from datetime import date, timedelta

from gauges.http import SourceError, get_ok, strip_tags
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult

AAII_URL = "https://www.aaii.com/sentimentsurvey/sent_results"
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def parse_aaii(html: str, today: date) -> list[Obs]:
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = [strip_tags(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        m = re.fullmatch(r"([A-Z][a-z]{2})\w*\s+(\d{1,2})", cells[0]) if cells else None
        if not m or len(cells) < 4 or m.group(1) not in MONTHS:
            continue
        d = date(today.year, MONTHS[m.group(1)], int(m.group(2)))
        if d > today + timedelta(days=7):   # table spans a year boundary
            d = d.replace(year=today.year - 1)
        for name, cell in zip(("bullish", "neutral", "bearish"), cells[1:4]):
            out.append(Obs("aaii", name, d, float(cell.rstrip("%")), None, AAII_URL))
    return out


def fetch(ctx: Context) -> FetchResult:
    # Weekly survey behind Incapsula, which challenges repeated fetches: ask once a week.
    last = ctx.last_date("aaii", "bullish")
    if last and (ctx.today - last).days < 6:
        return FetchResult(notes=[f"up to date (latest week {last})"])
    r = get_ok(AAII_URL, what="aaii")
    rows = parse_aaii(r.text, ctx.today)
    if not rows:
        raise SourceError("no AAII results table" + (" (Incapsula challenge)" if "Incapsula" in r.text else ""))
    return FetchResult(rows, notes=[f"{len(rows) // 3} weeks visible; full history is members-only"])
