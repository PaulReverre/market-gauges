"""Finnhub (free tier): upcoming earnings calendar and recent EPS surprises.

The free calendar only returns upcoming dates and stock/earnings only the last
four quarters, so history accumulates from the first run; historical report
dates come from EDGAR 8-K Item 2.02 filings instead. The all-company calendar is
capped at 1,500 rows (about two weeks), so tracked tickers are queried one by one.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from gauges.http import Throttle, get_ok, require
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult, each

SOURCE = "finnhub"
API = "https://finnhub.io/api/v1/"
THROTTLE = Throttle(1.1)  # ~60 calls/minute


def api(path: str, **params):
    THROTTLE.wait()
    return get_ok(API + path, params={**params, "token": require("FINNHUB_API_KEY")}, what=f"finnhub {path}").json()


def parse_calendar(payload: dict, universe: set[str]) -> pd.DataFrame:
    rows = []
    for e in payload.get("earningsCalendar", []):
        if e.get("symbol") not in universe:
            continue
        period = f"{e['year']}Q{e['quarter']}" if e.get("year") and e.get("quarter") else None
        rows.append({"ticker": e["symbol"], "report_date": date.fromisoformat(e["date"]), "source": "finnhub_calendar",
                     "hour": e.get("hour") or None, "fiscal_period": period, "eps_actual": e.get("epsActual"),
                     "eps_estimate": e.get("epsEstimate"), "revenue_actual": e.get("revenueActual"),
                     "revenue_estimate": e.get("revenueEstimate")})
    return pd.DataFrame(rows)


def parse_surprises(payload: list[dict], ticker: str) -> list[Obs]:
    out = []
    for e in payload:
        period = date.fromisoformat(e["period"])
        for key, suffix in (("actual", "eps_actual"), ("estimate", "eps_estimate")):
            if e.get(key) is not None:
                out.append(Obs(SOURCE, f"{ticker}.{suffix}", period, float(e[key]), None,
                               f"finnhub/stock/earnings?symbol={ticker}"))
    return out


def fetch(ctx: Context) -> FetchResult:
    notes: list[str] = []
    us_basket = [t for t in ctx.baskets.tickers if "." not in t]
    universe = set(us_basket + ctx.cfg.fetch.edgar.tickers)
    end = ctx.today + timedelta(days=ctx.cfg.fetch.finnhub.calendar_days_ahead)
    window = {"from": (ctx.today - timedelta(days=7)).isoformat(), "to": end.isoformat()}
    frames: list[pd.DataFrame] = []

    def calendar_for(ticker: str) -> list:
        frames.append(parse_calendar(api("calendar/earnings", symbol=ticker, **window), universe))
        return []

    each(sorted(universe), calendar_for, notes)
    calendar = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rows = each(us_basket, lambda t: parse_surprises(api("stock/earnings", symbol=t), t), notes)
    result = FetchResult(rows, notes=notes)
    result.add_table("earnings_events", calendar)
    notes.append(f"{len(calendar)} calendar rows for tracked tickers through {end}")
    return result
