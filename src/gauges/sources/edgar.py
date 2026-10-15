"""SEC EDGAR: XBRL facts (capex, OCF, RPO, D&A) and 8-K Item 2.02 earnings dates.

XBRL tags vary by company (AMZN and NVDA tag capex as
PaymentsToAcquireProductiveAssets; MSFT, GOOGL and ORCL tag only Depreciation),
so every candidate tag is stored and Phase 2 maps one per company. 10-Q values
are year-to-date; de-cumulation is a transform, not done here. `filed` is the
point-in-time availability date.
"""
from __future__ import annotations

import pandas as pd

from gauges.http import SourceError, Throttle, get_ok, require
from gauges.sources.base import Context, FetchResult

SOURCE = "edgar"
FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKERS = "https://www.sec.gov/files/company_tickers.json"
THROTTLE = Throttle(0.15)  # SEC fair access: stay under 10 requests/second
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}


def sec_json(url: str) -> dict:
    THROTTLE.wait()
    return get_ok(url, headers={"User-Agent": require("SEC_USER_AGENT"), "Accept-Encoding": "gzip, deflate"}).json()


def cik_map(ctx: Context) -> dict[str, int]:
    def load() -> dict[str, int]:
        return {row["ticker"].upper(): int(row["cik_str"]) for row in sec_json(TICKERS).values()}
    return ctx.cached("sec_cik_map", load)


def parse_companyfacts(payload: dict, ticker: str, tags: list[str]) -> pd.DataFrame:
    cik = int(payload["cik"])
    records = []
    for taxonomy, facts in payload.get("facts", {}).items():
        for tag in tags:
            for unit, items in facts.get(tag, {}).get("units", {}).items():
                for f in items:
                    if f.get("form") not in FORMS:
                        continue
                    records.append({
                        "cik": cik, "ticker": ticker, "taxonomy": taxonomy, "tag": tag, "unit": unit,
                        "period_start": f.get("start"), "period_end": f["end"], "value": float(f["val"]),
                        "form": f.get("form"), "fy": f.get("fy"), "fp": f.get("fp"), "filed": f["filed"],
                        "accn": f["accn"]})
    df = pd.DataFrame(records)
    if df.empty:
        return df
    for col in ("period_start", "period_end", "filed"):
        df[col] = pd.to_datetime(df[col]).dt.date
    df["fy"] = df["fy"].astype("Int64")
    return df


def parse_8k_earnings(payload: dict, ticker: str) -> pd.DataFrame:
    """8-K filings carrying Item 2.02 (Results of Operations) mark earnings release dates."""
    recent = payload.get("filings", {}).get("recent", {})
    rows = []
    for form, items, filed, accepted, accn in zip(recent.get("form", []), recent.get("items", []),
                                                    recent.get("filingDate", []), recent.get("acceptanceDateTime", []),
                                                    recent.get("accessionNumber", [])):
        if form == "8-K" and "2.02" in (items or "").split(","):
            rows.append({"ticker": ticker, "report_date": pd.to_datetime(filed).date(), "source": "edgar_8k_202",
                         "accepted_at": accepted, "accn": accn})
    return pd.DataFrame(rows)


def fetch(ctx: Context) -> FetchResult:
    cfg, notes, result = ctx.cfg.fetch.edgar, [], FetchResult()
    ciks = cik_map(ctx)
    us_basket = [t for t in ctx.baskets.tickers if "." not in t]
    failures = 0
    fact_tickers = list(dict.fromkeys(cfg.tickers + (us_basket if cfg.basket else [])))
    for ticker in fact_tickers:
        if ticker not in ciks:
            continue
        try:
            facts = parse_companyfacts(sec_json(FACTS.format(cik=ciks[ticker])), ticker, cfg.tags)
            result.add_table("xbrl_facts", facts)
            if facts.empty:
                notes.append(f"{ticker}: no facts for configured tags")
        except (KeyError, SourceError) as e:
            failures += 1
            notes.append(f"{ticker} facts: {type(e).__name__}: {e}")
    for ticker in dict.fromkeys(cfg.tickers + us_basket):
        if ticker not in ciks:
            notes.append(f"{ticker}: no CIK (foreign filer or not listed)")
            continue
        try:
            result.add_table("earnings_events", parse_8k_earnings(sec_json(SUBMISSIONS.format(cik=ciks[ticker])), ticker))
        except SourceError as e:
            notes.append(f"{ticker} submissions: {e}")
    if failures == len(fact_tickers):
        raise SourceError("companyfacts failed for every ticker")
    n_events = len(result.tables.get("earnings_events", []))
    notes.append(f"{len(result.tables.get('xbrl_facts', []))} XBRL facts; {n_events} 8-K 2.02 filings")
    return result
