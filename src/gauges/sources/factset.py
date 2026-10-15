"""FactSet Earnings Insight (weekly PDF, archive from January 2017).

Stand-in for the S&P DJI sp-500-eps-est.xlsx, which is blocked (HTTP 403,
Akamai). Values are read from FactSet's fixed sentences: forward P/E and its
5- and 10-year averages, quarterly growth estimate, beat rate, negative vs
positive EPS guidance counts, and calendar-year EPS growth forecasts.
pypdf inserts stray spaces inside words and numbers ("compa nies", "2 9.6%"),
so phrases are matched with optional whitespace between every character.
"""
from __future__ import annotations

import logging
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
NUM = r"(-?\s*\d[\d\s]*(?:\.\s*\d[\d\s]*)?)"
logging.getLogger("pypdf").setLevel(logging.ERROR)   # font-encoding warnings on every page


def loose(phrase: str) -> str:
    """Regex for a phrase that tolerates whitespace anywhere inside it."""
    return r"\s*".join(re.escape(c) for c in phrase if not c.isspace())


PATTERNS = {
    "fwd_pe": loose("forward 12-month P/E ratio for the S&P 500 is") + rf"\s*{NUM}",
    # "estimated (year-over-year) earnings growth rate", "blended earnings growth rate", "blended earnings decline"
    "q_growth_est": loose("For Q") + r"[\d\s,]+" + loose("the") + r"\s*(?:estimated|blended)\s*"
                    + r"(?:\(\s*year\s*-\s*over\s*-\s*year\s*\)\s*)?" + loose("earnings")
                    + r"\s*(?:g\s*r\s*o\s*w\s*t\s*h\s*r\s*a\s*t\s*e|d\s*e\s*c\s*l\s*i\s*n\s*e)\s*"
                    + loose("for the S&P 500 is") + rf"\s*{NUM}\s*%",
    "eps_beat_pct": rf"{NUM}\s*%\s*" + loose("have reported actual EPS above the mean EPS estimate"),
    "pe_avgs": loose("P/E ratio is") + r"[^.()]{0,60}?" + loose("5-year average (") + rf"{NUM}\)" + r"[^.()]{0,60}?"
               + loose("10-year average (") + rf"{NUM}\)",
    "guidance": loose("Of these") + r"[\d\s,]+" + r"c\s*o\s*m\s*p\s*a\s*n\s*i\s*e\s*s\s*,\s*" + rf"{NUM}\s*"
                + r"h\s*a\s*(?:s|v\s*e)\s*" + loose("issued negative EPS guidance and") + rf"\s*{NUM}\s*"
                + r"h\s*a\s*(?:s|v\s*e)\s*" + loose("issued positive EPS guidance"),
}
# Key Metrics wording, used when the "Of these N companies" sentence is missing.
GUIDANCE_KM = (rf"{NUM}\s*" + loose("S&P 500 companies have issued negative EPS guidance and") + rf"\s*{NUM}\s*"
               + loose("S&P 500 companies have issued positive EPS guidance"))
CY_GROWTH = (loose("For CY") + r"\s*(\d[\d\s]{3,5})\s*,\s*" + loose("analysts are") + r"\s*(?:p\s*r\s*o\s*j\s*e\s*c\s*t\s*i\s*n\s*g|"
             r"p\s*r\s*e\s*d\s*i\s*c\s*t\s*i\s*n\s*g)[^.%]{0,80}?" + loose("earnings") + r"\s*(?:g\s*r\s*o\s*w\s*t\s*h|d\s*e\s*c\s*l\s*i\s*n\s*e)"
             r"[^.%\d-]{0,20}" + rf"{NUM}\s*%")


def pdf_url(d: date) -> str:
    return f"{BASE}EarningsInsight_{d:%m%d%y}.pdf"


def _number(raw: str) -> float:
    return float(re.sub(r"\s+", "", raw).rstrip("."))


def parse_text(text: str, published: date) -> dict[str, float]:
    flat = re.sub(r"\s+", " ", text)
    out = {}
    for name, pattern in PATTERNS.items():
        m = re.search(pattern, flat, re.I)
        if not m:
            continue
        if name == "pe_avgs":
            out["fwd_pe_5y_avg"], out["fwd_pe_10y_avg"] = _number(m.group(1)), _number(m.group(2))
        elif name == "guidance":
            out["neg_guidance"], out["pos_guidance"] = _number(m.group(1)), _number(m.group(2))
        else:
            out[name] = _number(m.group(1))
    if "neg_guidance" not in out and (m := re.search(GUIDANCE_KM, flat, re.I)):
        out["neg_guidance"], out["pos_guidance"] = _number(m.group(1)), _number(m.group(2))
    for m in re.finditer(CY_GROWTH, flat, re.I):
        year = int(re.sub(r"\s+", "", m.group(1)))
        key = {published.year: "cy_growth_cur", published.year + 1: "cy_growth_next"}.get(year)
        if key and key not in out:
            out[key] = _number(m.group(2))
    return out


def pdf_text(content: bytes, pages: int = 14) -> str:
    reader = PdfReader(BytesIO(content))
    return "\n".join(p.extract_text() or "" for p in reader.pages[:pages])


def _week(friday: date) -> list[Obs]:
    """FactSet publishes on Fridays (Thursday in holiday weeks)."""
    for d in (friday, friday - timedelta(days=1), friday - timedelta(days=2)):
        r = get(pdf_url(d))
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
            values = parse_text(pdf_text(r.content), d)
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
