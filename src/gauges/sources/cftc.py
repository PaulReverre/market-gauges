"""CFTC Traders in Financial Futures (Socrata gpe5-46if) for ES and NQ.

Keyed on contract code because market names changed in 2022. Tuesday positions
are released the following Friday: vintage = report date + 3 days.
"""
from __future__ import annotations

from datetime import date, timedelta

from gauges.http import get_ok
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult

SOURCE = "cftc"
URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
REFETCH_DAYS = 28


def parse_rows(rows: list[dict], contracts: dict[str, str], fields: list[str]) -> list[Obs]:
    out = []
    for row in rows:
        alias = contracts.get(row.get("cftc_contract_market_code", ""))
        if not alias:
            continue
        report = date.fromisoformat(row["report_date_as_yyyy_mm_dd"][:10])
        for f in fields:
            if row.get(f) not in (None, ""):
                out.append(Obs(SOURCE, f"{alias}.{f}", report, float(row[f]), report + timedelta(days=3), URL))
    return out


def fetch(ctx: Context) -> FetchResult:
    cfg = ctx.cfg.fetch.cftc
    codes = ", ".join(f"'{c}'" for c in cfg.contracts)
    where = f"cftc_contract_market_code in ({codes})"
    last = ctx.last_date_prefix(SOURCE, "")
    if last:
        where += f" AND report_date_as_yyyy_mm_dd >= '{(last - timedelta(days=REFETCH_DAYS)).isoformat()}'"
    payload = get_ok(URL, params={
        "$select": ", ".join(["cftc_contract_market_code", "report_date_as_yyyy_mm_dd", *cfg.fields]),
        "$where": where, "$order": "report_date_as_yyyy_mm_dd", "$limit": 50000}, what="cftc tff").json()
    rows = parse_rows(payload, cfg.contracts, cfg.fields)
    return FetchResult(rows, notes=[f"{len(payload)} report rows"])
