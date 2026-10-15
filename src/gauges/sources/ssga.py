"""SSGA SPY daily holdings: S&P 500 constituents and weights (top-10 concentration)."""
from __future__ import annotations

import re
from datetime import date, datetime
from io import BytesIO

import pandas as pd
from openpyxl import load_workbook

from gauges.http import SourceError, get_ok
from gauges.sources.base import Context, FetchResult

SOURCE = "ssga"
URL = "https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
TICKER_RE = re.compile(r"[A-Z][A-Z0-9]*(\.[A-Z])?")  # drops cash lines ("-") and CVRs ("2602335D")


def parse_holdings(content: bytes) -> tuple[date, pd.DataFrame]:
    rows = list(load_workbook(BytesIO(content), read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    as_of = None
    for r in rows[:8]:
        m = re.search(r"As of (\d{1,2}-[A-Za-z]{3}-\d{4})", str(r[1] if r and len(r) > 1 else ""))
        if r and r[0] == "Holdings:" and m:
            as_of = datetime.strptime(m.group(1), "%d-%b-%Y").date()
    header = next((i for i, r in enumerate(rows) if r and r[0] == "Name" and len(r) > 1 and r[1] == "Ticker"), None)
    if as_of is None or header is None:
        raise SourceError("SPY holdings: as-of date or header row not found")
    col = {name: i for i, name in enumerate(rows[header]) if name}
    records = []
    for r in rows[header + 1:]:
        if not r or all(v is None for v in r):
            break
        ticker = str(r[col["Ticker"]] or "").strip()
        if not TICKER_RE.fullmatch(ticker) or r[col["Weight"]] is None:
            continue
        records.append({"ticker": ticker, "name": r[col["Name"]], "weight": float(r[col["Weight"]]),
                        "shares": float(r[col["Shares Held"]]) if r[col["Shares Held"]] is not None else None,
                        "identifier": str(r[col["Identifier"]])})
    return as_of, pd.DataFrame(records)


def holdings(ctx: Context) -> tuple[date, pd.DataFrame]:
    return ctx.cached("ssga_spy", lambda: parse_holdings(get_ok(URL).content))


def yahoo_symbol(ticker: str) -> str:
    """BRK.B -> BRK-B (Yahoo's class-share notation)."""
    return ticker.replace(".", "-") if re.fullmatch(r"[A-Z]+\.[A-Z]", ticker) else ticker


def constituents(ctx: Context) -> list[str]:
    _, df = holdings(ctx)
    return sorted({yahoo_symbol(t) for t in df["ticker"]})


def fetch(ctx: Context) -> FetchResult:
    as_of, df = holdings(ctx)
    table = df.assign(as_of=as_of, fund="SPY")[["as_of", "fund", "ticker", "name", "weight", "shares", "identifier"]]
    return FetchResult(tables={"index_holdings": table},
                       notes=[f"SPY holdings as of {as_of}: {len(df)} constituents"])
