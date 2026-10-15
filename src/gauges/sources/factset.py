"""FactSet Earnings Insight (weekly PDF): forward P/E, quarterly growth, beat rate.

Stand-in for the S&P DJI sp-500-eps-est.xlsx, which is blocked (HTTP 403,
Akamai). Values are read from the fixed "Key Metrics" and "Overview" sentences.
pypdf inserts stray spaces inside numbers ("2 9.6%", "12 -month"), so the
patterns allow whitespace inside numbers and strip it before parsing.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from io import BytesIO

from pypdf import PdfReader

from gauges.http import get
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult

SOURCE = "factset"
BASE = "https://advantage.factset.com/hubfs/Website/Resources%20Section/Research%20Desk/Earnings%20Insight/"
NUM = r"(-?\d[\d\s]*(?:\.\s*\d[\d\s]*)?)"
PATTERNS = {
    "fwd_pe": rf"forward\s*12\s*-\s*month\s*P\s*/\s*E\s*ratio\s*for\s*the\s*S&P\s*500\s*is\s*{NUM}",
    "q_growth_est": rf"Earnings\s*Growth:\s*For\s*Q\s*\d\s*\d[\d\s]*,\s*the\s*(?:estimated|blended)\s*\(year-over-year\)\s*"
                    rf"earnings\s*growth\s*rate\s*for\s*the\s*S&P\s*500\s*is\s*{NUM}\s*%",
    "eps_beat_pct": rf"{NUM}\s*%\s*have\s*reported\s*actual\s*EPS\s*above\s*the\s*mean\s*EPS\s*estimate",
    "neg_guidance": rf"{NUM}\s*S&P\s*500\s*companies\s*have\s*issued\s*negative\s*EPS\s*guidance",
    "pos_guidance": rf"{NUM}\s*S&P\s*500\s*companies\s*have\s*issued\s*positive\s*EPS\s*guidance",
}


def pdf_url(d: date) -> str:
    return f"{BASE}EarningsInsight_{d:%m%d%y}.pdf"


def _number(raw: str) -> float:
    return float(re.sub(r"\s+", "", raw).rstrip("."))


def parse_text(text: str) -> dict[str, float]:
    flat = re.sub(r"\s+", " ", text)
    out = {}
    for name, pattern in PATTERNS.items():
        m = re.search(pattern, flat, re.I)
        if m:
            out[name] = _number(m.group(1))
    return out


def pdf_text(content: bytes, pages: int = 8) -> str:
    reader = PdfReader(BytesIO(content))
    return "\n".join(p.extract_text() or "" for p in reader.pages[:pages])


def _week(friday: date) -> list[Obs]:
    """FactSet publishes on Fridays (Thursday in holiday weeks)."""
    for d in (friday, friday - timedelta(days=1), friday - timedelta(days=2)):
        r = get(pdf_url(d))
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
            values = parse_text(pdf_text(r.content))
            return [Obs(SOURCE, name, d, value, None, pdf_url(d)) for name, value in values.items()]
    return []


def fetch(ctx: Context) -> FetchResult:
    last = ctx.last_date(SOURCE, "fwd_pe")
    start = max(ctx.cfg.fetch.factset.backfill_since, (last + timedelta(days=1)) if last else date.min)
    friday = start + timedelta(days=(4 - start.weekday()) % 7)
    fridays = []
    while friday - timedelta(days=2) <= ctx.today:   # this week's Friday may still be ahead
        fridays.append(friday)
        friday += timedelta(days=7)
    rows: list[Obs] = []
    with ThreadPoolExecutor(4) as pool:
        for week_rows in pool.map(_week, fridays):
            rows += week_rows
    found = sorted({r.obs_date for r in rows})
    notes = [f"{len(found)} of {len(fridays)} weekly PDFs parsed" + (f", latest {found[-1]}" if found else "")]
    missing = {r.obs_date for r in rows if r.series == "fwd_pe"} ^ set(found)
    if missing:
        notes.append(f"forward P/E not found in {len(missing)} PDFs: {sorted(missing)[:5]}")
    return FetchResult(rows, notes=notes)
