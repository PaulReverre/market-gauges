"""Cboe: index history CSVs and daily put/call ratios.

Put/call history is stitched from the free archive CSVs (2006-11-01 to
2019-10-04) and the daily market-statistics JSON (2019-10-07 onward).
"""
from __future__ import annotations

import csv
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from gauges.http import SourceError, get, get_ok, parse_mdy
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult, business_days, each

SOURCE = "cboe"
INDEX_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
PC_JSON = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{day}_daily_options"
PC_ARCHIVE = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{name}.csv"
ARCHIVE_SERIES = {"totalpc": "pc_total", "equitypc": "pc_equity", "indexpc": "pc_index"}
JSON_SERIES = {
    "TOTAL PUT/CALL RATIO": "pc_total",
    "EQUITY PUT/CALL RATIO": "pc_equity",
    "INDEX PUT/CALL RATIO": "pc_index",
    "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO": "pc_etp",
    "SPX + SPXW PUT/CALL RATIO": "pc_spx",
    "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO": "pc_vix",
}


def parse_index_csv(text: str, symbol: str) -> list[Obs]:
    """DATE,...,CLOSE (or DATE,VALUE for VVIX/SKEW): the close is the last column."""
    lines = text.splitlines()
    if not lines or not lines[0].upper().startswith("DATE"):
        raise SourceError(f"{symbol}: unexpected CSV header {lines[0][:60] if lines else '(empty)'}")
    endpoint = INDEX_CSV.format(sym=symbol)
    return [Obs(SOURCE, symbol, parse_mdy(row[0]), float(row[-1]), None, endpoint)
            for row in csv.reader(lines[1:]) if row and row[0].strip() and row[-1].strip()]


def parse_putcall_archive(text: str, name: str) -> list[Obs]:
    """Rows look like 11/1/2006,calls,puts,total,P/C ratio; header lines are skipped."""
    series, endpoint = ARCHIVE_SERIES[name], PC_ARCHIVE.format(name=name)
    out = []
    for line in text.splitlines():
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4},", line):
            fields = [f.strip() for f in line.split(",")]
            out.append(Obs(SOURCE, series, parse_mdy(fields[0]), float(fields[-1]), None, endpoint))
    return out


def parse_putcall_json(payload: dict, day: date) -> list[Obs]:
    endpoint = PC_JSON.format(day=day.isoformat())
    out = []
    for item in payload.get("ratios", []):
        series = JSON_SERIES.get(item.get("name", "").strip())
        if series and item.get("value") not in (None, ""):
            out.append(Obs(SOURCE, series, day, float(item["value"]), None, endpoint))
    return out


def _putcall_day(day: date) -> list[Obs]:
    r = get(PC_JSON.format(day=day.isoformat()))
    if r.status_code != 200:   # holidays and not-yet-posted days return 403
        return []
    return parse_putcall_json(r.json(), day)


def fetch(ctx: Context) -> FetchResult:
    notes: list[str] = []
    cfg = ctx.cfg.fetch.cboe
    rows = each(cfg.indices, lambda s: parse_index_csv(get_ok(INDEX_CSV.format(sym=s)).text, s), notes)

    last_pc = ctx.last_date(SOURCE, "pc_total")
    if last_pc is None or last_pc < cfg.putcall_json_start:
        rows += each(ARCHIVE_SERIES, lambda n: parse_putcall_archive(get_ok(PC_ARCHIVE.format(name=n)).text, n),
                     notes)
    start = max(cfg.putcall_json_start, (last_pc or cfg.putcall_json_start) + timedelta(days=1))
    days = business_days(start, ctx.session)
    with ThreadPoolExecutor(4) as pool:
        for day_rows in pool.map(_putcall_day, days):
            rows += day_rows
    notes.append(f"put/call JSON: {len(days)} business days requested from {start}")
    return FetchResult(rows, notes=notes)
