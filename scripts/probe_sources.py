#!/usr/bin/env python3
"""Phase 0: probe every data source in BUILD_PROMPT.md Section 5.

For each source this prints status, latest observation, history start,
rate-limit headers and terms flags as a markdown table, lists failures with a
proposed substitute, runs the Section 3 cross-checks, and writes the report to
reports/source_status.md. Nothing else is persisted. Keyed sources report 🔑
until their variable is set in .env.

Usage: python scripts/probe_sources.py [--only fred,cboe_csv] [--no-write]
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import platform
import re
import statistics
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from openpyxl import load_workbook
from tenacity import retry, retry_if_exception_type, retry_if_result, stop_after_attempt, wait_exponential

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "source_status.md"
ET, PT = ZoneInfo("America/New_York"), ZoneInfo("America/Los_Angeles")
NOW = datetime.now(ET)
TODAY = NOW.date()

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0 Safari/537.36"
)
PROJECT_UA = "market-gauges/0.1 (Phase 0 source probe)"

OK, WARN, KEY, FAIL = "✅", "⚠️", "🔑", "❌"
RETRY_STATUS = {429, 500, 502, 503, 504}

KEY_ENVS = {
    "fred": "FRED_API_KEY",
    "finnhub": "FINNHUB_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "eia": "EIA_API_KEY",
    "sec": "SEC_USER_AGENT",
    "massive": "MASSIVE_API_KEY",
    "fmp": "FMP_API_KEY",
    "alphavantage": "ALPHAVANTAGE_API_KEY",
}


def env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


# --------------------------------------------------------------------------- #
# HTTP: one client per worker thread; retry transport errors, 429 and 5xx.
# --------------------------------------------------------------------------- #
_local = threading.local()


def _client() -> httpx.Client:
    if not hasattr(_local, "client"):
        _local.client = httpx.Client(follow_redirects=True, timeout=45, headers={"User-Agent": BROWSER_UA})
    return _local.client


@retry(
    retry=retry_if_exception_type(httpx.TransportError) | retry_if_result(lambda r: r.status_code in RETRY_STATUS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry_error_callback=lambda state: state.outcome.result(),
)
def get(url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
    return _client().get(url, params=params, headers=headers)


def head(url: str) -> httpx.Response:
    return _client().head(url)


def rate_headers(r: httpx.Response) -> str:
    return "; ".join(f"{k}: {v}" for k, v in r.headers.items() if re.search(r"rate-?limit|retry-after", k, re.I))


class Throttle:
    """Spaces calls to one API across threads (e.g. FRED's 120 req/min)."""

    def __init__(self, interval: float):
        self.interval, self.lock, self.next_at = interval, threading.Lock(), 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next_at - now
            self.next_at = max(now, self.next_at) + self.interval
        if delay > 0:
            time.sleep(delay)


def redact(text: str) -> str:
    """Strip any configured secret from text that may end up in the committed report."""
    for name in KEY_ENVS.values():
        secret = env(name)
        if secret and len(secret) >= 8:
            text = text.replace(secret, f"<{name}>")
    return text


def strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>|&nbsp;", " ", s)).strip()


def parse_mdy(s: str) -> date:
    return datetime.strptime(s.strip(), "%m/%d/%Y").date()


# Observations shared with the cross-checks, keyed "source:series".
_obs: dict[str, dict[date, float]] = {}
_obs_lock = threading.Lock()


def remember(key: str, points: dict[date, float]) -> None:
    with _obs_lock:
        _obs[key] = points


@dataclass
class Result:
    status: str
    latest: str = "—"
    history: str = "—"
    limits: str = ""
    terms: str = ""
    notes: str = ""
    details: list[str] = field(default_factory=list)
    source: str = ""
    key: str = ""


def key_missing(probe_key: str, r: httpx.Response | None = None, extra: str = "") -> Result:
    seen = f"unauthenticated request → HTTP {r.status_code}. " if r is not None else ""
    return Result(KEY, notes=f"{seen}Set {KEY_ENVS[probe_key]} in .env. {extra}".strip())


# --------------------------------------------------------------------------- #
# FRED / ALFRED
# --------------------------------------------------------------------------- #
FRED_GRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API = "https://api.stlouisfed.org/fred/"
FRED_SERIES = """
SP500 VIXCLS VXNCLS DGS10 BAMLH0A0HYM2EY BAMLC0A0CMEY BAMLH0A0HYM2 BAMLH0A3HYC BAMLH0A1HYBB BAMLC0A4CBBB
NFCI STLFSI4 SOFR IORB DTWEXBGS GDPNOW CFNAIMA3 ICSA SAHMREALTIME PERMIT DFII10 NCBEILQ027S GDP DFF
PCEPILFE T10Y3M WALCL WTREGEN RRPONTSYD DRTSCILM RRSFS DRCCLACBS UMCSENT T10YIE
RAMPAIALL RAMPAIANTHROPIC RAMPAIOPENAI RAMPAIXAI
GACDFSA066MSFRBPHI GACDISA066MSFRBNY BACTSAMFRBDAL
""".split()
ICE_BOFA = [s for s in FRED_SERIES if s.startswith("BAML")]
REGIONAL_SEARCH = ["Fifth District manufacturing composite", "Tenth District manufacturing composite"]
CROSSCHECK_FRED = {"SP500", "VIXCLS", "DGS10"}
FRED_THROTTLE = Throttle(0.55)  # ~110 req/min, under the 120/min cap


@dataclass
class FredSeries:
    first: date
    last: date
    last_value: float
    detail: str  # frequency or observation count


def fred_api(path: str, **params) -> httpx.Response:
    FRED_THROTTLE.wait()
    return get(FRED_API + path, params={**params, "api_key": env("FRED_API_KEY"), "file_type": "json"})


def _fred_via_api(series: str) -> FredSeries | None:
    meta = fred_api("series", series_id=series)
    if meta.status_code != 200:
        return None
    info = meta.json()["seriess"][0]
    limit = 10 if series in CROSSCHECK_FRED else 1
    obs = fred_api("series/observations", series_id=series, sort_order="desc", limit=limit)
    points = {date.fromisoformat(o["date"]): float(o["value"])
              for o in obs.json().get("observations", []) if o["value"] != "."}
    if not points:
        return None
    if series in CROSSCHECK_FRED:
        remember(f"fred:{series}", points)
    last = max(points)
    return FredSeries(date.fromisoformat(info["observation_start"]), last, points[last],
                      f"{info['frequency_short']}, updated {info['last_updated'][:10]}")


def _fred_via_graph(series: str) -> FredSeries | None:
    r = get(FRED_GRAPH, params={"id": series})
    lines = r.text.splitlines()
    if r.status_code != 200 or not lines or series not in lines[0]:
        return None
    points = {date.fromisoformat(row[0]): float(row[1])
              for row in csv.reader(lines[1:]) if len(row) == 2 and row[1] not in ("", ".")}
    if not points:
        return None
    if series in CROSSCHECK_FRED:
        remember(f"fred:{series}", points)
    last = max(points)
    return FredSeries(min(points), last, points[last], f"{len(points)} obs")


def probe_fred() -> Result:
    keyed = bool(env("FRED_API_KEY"))
    fetch = _fred_via_api if keyed else _fred_via_graph

    def one(series: str) -> tuple[str, FredSeries | None, str]:
        try:
            return series, fetch(series), ""
        except Exception as e:  # noqa: BLE001 - one slow series must not sink the rest
            return series, None, type(e).__name__

    with ThreadPoolExecutor(2 if keyed else 4) as pool:
        out = list(pool.map(one, FRED_SERIES))

    table = ["### FRED series", "", "| Series | First | Last | Last value | Info | Flag |", "|---|---|---|---|---|---|"]
    found, missing = {}, []
    for s, info, err in out:
        if not info:
            missing.append(f"{s}{f' ({err})' if err else ''}")
            table.append(f"| {s} | ❌ {err or 'not found'} | | | | |")
            continue
        found[s] = info
        flag = "ICE 3y window" if s in ICE_BOFA else ""
        table.append(f"| {s} | {info.first} | {info.last} | {info.last_value:g} | {info.detail} | {flag} |")

    details = ["\n".join(table)]
    notes = [f"{len(found)}/{len(FRED_SERIES)} series via {'API' if keyed else 'keyless graph CSV'}"]
    if missing:
        notes.append("missing: " + ", ".join(missing))
    if keyed:
        hits = []
        for text in REGIONAL_SEARCH:
            rs = fred_api("series/search", search_text=text, limit=5)
            hits += [f"{s['id']} ({s['title'][:60]})" for s in rs.json().get("seriess", [])] if rs.status_code == 200 else []
        notes.append("Richmond/KC composites: " + ("candidates " + "; ".join(hits) if hits else "not on FRED, scrape Fed sites"))
    else:
        notes.append("Richmond/KC composites: unmapped (search needs FRED_API_KEY)")

    sp = found.get("SP500")
    ice_first = min((found[s].first for s in ICE_BOFA if s in found), default=None)
    return Result(
        OK if not missing else WARN,
        latest=(f"newest {max(f.last for f in found.values())}" if found else "—") + (f"; SP500 {sp.last}" if sp else ""),
        history=(f"SP500 {sp.first}; " if sp else "") + f"ICE BofA {ice_first}",
        notes="; ".join(notes),
        details=details,
    )


def probe_alfred() -> Result:
    vintage, series = "2025-06-02", "BAMLH0A0HYM2"
    if env("FRED_API_KEY"):
        rv = fred_api("series/vintagedates", series_id=series, limit=1)
        first_vintage = (rv.json().get("vintage_dates") or [None])[0] if rv.status_code == 200 else None
        ro = fred_api("series/observations", series_id=series, realtime_start=vintage, realtime_end=vintage,
                      observation_start="1996-12-31", limit=1)
        obs = ro.json().get("observations", []) if ro.status_code == 200 else []
        first_obs = obs[0]["date"] if obs else None
        detail = f"earliest vintage {first_vintage}; vintage {vintage} first obs {first_obs}"
        if first_obs and first_obs < "2023-01-01":
            return Result(OK, latest=f"vintage {vintage}", history=first_obs,
                          notes=f"pre-cutoff history available; {detail}")
        return Result(FAIL, latest=f"earliest vintage {first_vintage}", history=first_obs or "—",
                      notes=f"confirmed: ALFRED holds no pre-2023-10-16 history for ICE BofA, backfill impossible; {detail}")

    r = get("https://alfred.stlouisfed.org/graph/alfredgraph.csv",
            params={"id": series, "vintage_date": vintage, "cosd": "1996-12-31"})
    rows = [l for l in r.text.splitlines()[1:] if re.match(r"^\d{4}-\d{2}-\d{2},[\d.]+", l)]
    first_obs = rows[0].split(",")[0] if rows else None
    deep = bool(first_obs) and first_obs < "2023-01-01"
    return Result(OK if deep else WARN, latest=f"vintage {vintage}", history=first_obs or "—",
                  notes=("pre-cutoff history available" if deep else "at risk: keyless vintage truncated")
                  + "; confirm with FRED_API_KEY")


# --------------------------------------------------------------------------- #
# Cboe
# --------------------------------------------------------------------------- #
CBOE_SYMBOLS = ["VIX", "VIX3M", "VIX9D", "VVIX", "SKEW", "COR1M", "COR3M", "COR30D", "VXN"]


def _cboe_one(sym: str) -> tuple[str, dict[date, float] | None, str]:
    r = get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv")
    if r.status_code != 200 or not r.text.upper().startswith("DATE"):
        return sym, None, f"HTTP {r.status_code}"
    points = {}
    for row in csv.reader(r.text.splitlines()[1:]):
        if row and row[0].strip():
            points[parse_mdy(row[0])] = float(row[-1])
    return sym, points, r.url.host


def probe_cboe_csv() -> Result:
    with ThreadPoolExecutor(4) as pool:
        out = list(pool.map(_cboe_one, CBOE_SYMBOLS))
    table = ["### Cboe index CSVs", "", "| Symbol | First | Last | Last close | Rows |", "|---|---|---|---|---|"]
    got, missing, host = {}, [], ""
    for sym, p, info in out:
        if not p:
            missing.append(f"{sym} ({info})")
            table.append(f"| {sym} | ❌ {info} | | | |")
            continue
        got[sym], host = p, info
        table.append(f"| {sym} | {min(p)} | {max(p)} | {p[max(p)]:g} | {len(p)} |")
    if "VIX" in got:
        remember("cboe:VIX", got["VIX"])
    return Result(
        OK if not missing else WARN,
        latest=str(max(max(p) for p in got.values())) if got else "—",
        history=", ".join(f"{s} {min(p).year}" for s, p in got.items()),
        notes=f"{len(got)}/{len(CBOE_SYMBOLS)} symbols; served from {host} (follow redirects)"
        + (f"; missing {', '.join(missing)}" if missing else ""),
        details=["\n".join(table)],
    )


PC_JSON = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{d}_daily_options"
PC_ARCHIVE = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{f}.csv"
PC_JSON_START = "2019-10-07"


def probe_putcall() -> Result:
    latest = None
    for back in range(8):
        d = TODAY - timedelta(days=back)
        if d.weekday() >= 5:
            continue
        r = get(PC_JSON.format(d=d.isoformat()))
        if r.status_code != 200:
            continue
        try:
            ratios = {x["name"]: x["value"] for x in r.json().get("ratios", [])}
        except ValueError:
            continue
        if ratios:
            latest = (d, ratios)
            break
    json_start_ok = get(PC_JSON.format(d=PC_JSON_START)).status_code == 200

    archive = {}
    for f in ("totalpc", "equitypc", "indexpc"):
        r = get(PC_ARCHIVE.format(f=f))
        dates = [parse_mdy(l.split(",")[0]) for l in r.text.splitlines() if re.match(r"^\d{1,2}/\d{1,2}/\d{4},", l)]
        archive[f] = (min(dates), max(dates)) if dates else None

    if not latest:
        return Result(FAIL, notes="no daily JSON found in the last 7 days")
    d, ratios = latest
    arch = archive.get("totalpc")
    complete = json_start_ok and all(archive.values())
    return Result(
        OK if complete else WARN,
        latest=f"{d}: total {ratios.get('TOTAL PUT/CALL RATIO')}, equity {ratios.get('EQUITY PUT/CALL RATIO')}, "
               f"index {ratios.get('INDEX PUT/CALL RATIO')}",
        history=(f"archive CSV {arch[0]}→{arch[1]}; " if arch else "archive CSV missing; ")
                + (f"daily JSON {PC_JSON_START}→" if json_start_ok else "daily JSON start not reachable"),
        notes="post-2019 gap closed by the daily JSON" if complete else "history incomplete",
    )


# --------------------------------------------------------------------------- #
# Prices: Yahoo via yfinance, Nasdaq fallback
# --------------------------------------------------------------------------- #
YF_SYMBOLS = ["^GSPC", "^MOVE", "SPY", "RSP", "SPHB", "SPLV", "IWM", "IEF", "HG=F", "GC=F", "SMH", "SOXX",
              "IGV", "NVDA", "000660.KS", "005930.KS", "CRWV", "NBIS", "SOXL", "NVDL"]


def probe_yahoo() -> Result:
    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    df = yf.download(YF_SYMBOLS, period="10d", interval="1d", progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        return Result(FAIL, notes="yfinance download returned no data (likely HTTP 429 throttling)")
    close = df["Close"]
    got = [s for s in YF_SYMBOLS if s in close.columns and close[s].notna().any()]
    missing = [s for s in YF_SYMBOLS if s not in got]
    last = max(close[s].last_valid_index() for s in got).date()
    if "^GSPC" in got:
        remember("yf:^GSPC", {ts.date(): float(v) for ts, v in close["^GSPC"].dropna().items()})

    notes, history = [f"{len(got)}/{len(YF_SYMBOLS)} symbols"], []
    if missing:
        notes.append("missing " + ", ".join(missing))
    for sym in ("^GSPC", "^MOVE"):
        try:
            h = yf.Ticker(sym).history(period="max", interval="1d")
            history.append(f"{sym} {h.index.min().date()}")
        except Exception as e:  # noqa: BLE001 - report, don't abort the probe
            history.append(f"{sym} ? ({type(e).__name__})")
    options_ok = False
    try:
        nvda = yf.Ticker("NVDA")
        expiries = nvda.options
        chain = nvda.option_chain(expiries[min(2, len(expiries) - 1)])
        options_ok = bool(len(chain.calls)) and chain.calls["impliedVolatility"].gt(0).any()
        notes.append(f"NVDA options: {len(expiries)} expiries, IV field {'present' if options_ok else 'missing'}")
    except Exception as e:  # noqa: BLE001
        notes.append(f"NVDA options failed ({type(e).__name__})")
    try:
        notes.append(f"NVDA forwardPE {yf.Ticker('NVDA').info.get('forwardPE')}")
    except Exception as e:  # noqa: BLE001
        notes.append(f"forwardPE failed ({type(e).__name__})")
    history.append("option IV: none, snapshot from day 1")
    return Result(
        OK if not missing and options_ok else WARN,
        latest=str(last),
        history="; ".join(history),
        notes="; ".join(notes),
    )


NASDAQ_HEADERS = {"Accept": "application/json", "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}


def probe_nasdaq() -> Result:
    r = get("https://api.nasdaq.com/api/quote/SPY/historical", headers=NASDAQ_HEADERS, params={
        "assetclass": "etf", "fromdate": (TODAY - timedelta(days=3660)).isoformat(),
        "todate": TODAY.isoformat(), "limit": 9999})
    rows = (((r.json().get("data") or {}).get("tradesTable") or {}).get("rows") or []) if r.status_code == 200 else []
    oc = get("https://api.nasdaq.com/api/quote/NVDA/option-chain", headers=NASDAQ_HEADERS,
             params={"assetclass": "stocks", "limit": 5})
    chain = (oc.json().get("data") or {}) if oc.status_code == 200 else {}
    if not rows:
        return Result(FAIL, notes=f"SPY history HTTP {r.status_code}")
    return Result(
        OK,
        latest=f"SPY {parse_mdy(rows[0]['date'])} close {rows[0]['close']}",
        history=f"SPY {parse_mdy(rows[-1]['date'])} ({len(rows)} rows)",
        notes=f"NVDA option chain: {chain.get('totalRecord', 'n/a')} rows, no IV field (compute from mids). "
              "No ^MOVE, futures or KRX coverage",
    )


# --------------------------------------------------------------------------- #
# SEC EDGAR
# --------------------------------------------------------------------------- #
SEC_CIKS = {"MSFT": 789019, "GOOGL": 1652044, "AMZN": 1018724, "META": 1326801, "ORCL": 1341439, "NVDA": 1045810}
SEC_TAGS = {
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForProceedsFromProductiveAssets"],
    "rpo": ["RevenueRemainingPerformanceObligation"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "d&a": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
            "DepreciationAmortizationAndAccretionNet", "Depreciation"],  # MSFT/GOOGL/ORCL tag depreciation only
}
SEC_STALE_DAYS = 200


def _latest_fact(gaap: dict, tag: str) -> dict | None:
    units = gaap.get(tag, {}).get("units", {}).get("USD", [])
    facts = [f for f in units if f.get("form") in ("10-Q", "10-K")]
    return max(facts, key=lambda f: (f["end"], f.get("filed", ""))) if facts else None


def probe_sec() -> Result:
    ua = env("SEC_USER_AGENT")
    if not ua:
        return key_missing("sec", extra="SEC fair-access policy requires a declared User-Agent with contact email.")
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
    table = ["### SEC XBRL tags (freshest candidate per metric)", "",
             "| Ticker | " + " | ".join(SEC_TAGS) + " |", "|---|" + "---|" * len(SEC_TAGS)]
    flags, newest, limits = [], "", ""
    for ticker, cik in SEC_CIKS.items():
        r = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=headers)
        limits = limits or rate_headers(r)
        if r.status_code != 200:
            flags.append(f"{ticker} HTTP {r.status_code}")
            continue
        gaap = r.json()["facts"].get("us-gaap", {})
        cells = []
        for metric, tags in SEC_TAGS.items():
            best = max(((t, f) for t in tags if (f := _latest_fact(gaap, t))), key=lambda tf: tf[1]["end"], default=None)
            if not best:
                cells.append("none")
                flags.append(f"{ticker} {metric}: no tag")
                continue
            tag, fact = best
            stale = (TODAY - date.fromisoformat(fact["end"])).days > SEC_STALE_DAYS
            if stale:
                flags.append(f"{ticker} {metric}: stale ({fact['end']})")
            newest = max(newest, fact["end"])
            cells.append(f"{'⚠️ ' if stale else ''}{tag} → {fact['end']} ({fact['form']})")
        table.append(f"| {ticker} | " + " | ".join(cells) + " |")
        time.sleep(0.15)  # stay well under 10 req/s
    core = [f for f in flags if " capex" in f or " ocf" in f or "HTTP" in f]
    return Result(
        OK if not core else WARN,
        latest=f"newest period end {newest}" if newest else "—",
        history="full XBRL history",
        limits=limits,
        notes=("flags: " + "; ".join(flags) if flags else "all tags fresh") + ". 10-Q values are YTD: de-cumulate",
        details=["\n".join(table)],
    )


# --------------------------------------------------------------------------- #
# Keyed market/data APIs
# --------------------------------------------------------------------------- #
def probe_finnhub() -> Result:
    base, key = "https://finnhub.io/api/v1/", env("FINNHUB_API_KEY")
    if not key:
        return key_missing("finnhub", get(base + "quote", params={"symbol": "NVDA"}))
    q = get(base + "quote", params={"symbol": "NVDA", "token": key})
    if q.status_code != 200:
        return Result(FAIL, notes=f"quote HTTP {q.status_code}: {q.text[:100]}")
    cal = get(base + "calendar/earnings", params={
        "from": TODAY.isoformat(), "to": (TODAY + timedelta(days=30)).isoformat(), "token": key})
    surprises = get(base + "stock/earnings", params={"symbol": "NVDA", "token": key})
    n_cal = len((cal.json() or {}).get("earningsCalendar", [])) if cal.status_code == 200 else f"HTTP {cal.status_code}"
    periods = [s.get("period") for s in surprises.json()] if surprises.status_code == 200 else []
    return Result(
        OK,
        latest=f"NVDA quote {datetime.fromtimestamp(q.json().get('t', 0), ET):%Y-%m-%d %H:%M} ET",
        history=f"EPS surprises from {min(periods)}" if periods else "—",
        limits=rate_headers(q),
        notes=f"earnings calendar next 30d: {n_cal} rows; surprises: {len(periods)} quarters",
    )


def _rows(payload) -> list:
    data = payload.get("data") if isinstance(payload, dict) else payload
    return data if isinstance(data, list) else []


def probe_openrouter() -> Result:
    models = get("https://openrouter.ai/api/v1/models")
    listing = _rows(models.json()) if models.status_code == 200 else []
    newest = max((m.get("created", 0) for m in listing), default=0)
    models_note = f"/models keyless: {len(listing)} models, newest added {datetime.fromtimestamp(newest, ET):%Y-%m-%d}"
    url, key = "https://openrouter.ai/api/v1/datasets/rankings-daily", env("OPENROUTER_API_KEY")
    if not key:
        return key_missing("openrouter", get(url), extra=models_note)

    auth = {"Authorization": f"Bearer {key}"}
    r = get(url, headers=auth, params={"start_date": (TODAY - timedelta(days=7)).isoformat(), "end_date": TODAY.isoformat()})
    if r.status_code != 200:
        return Result(FAIL, notes=f"rankings-daily HTTP {r.status_code}: {r.text[:120]}; {models_note}")
    body = r.json()
    rows = _rows(body)
    dates = sorted({str(x.get("date")) for x in rows if isinstance(x, dict) and x.get("date")})
    # The API floor is 2025-01-01 ("Data is only available from 2025-01-01 onward").
    deep = get(url, headers=auth, params={"start_date": "2025-01-01", "end_date": "2025-01-02"})
    deep_rows = _rows(deep.json()) if deep.status_code == 200 else []
    deep_note = f"{len(deep_rows)} rows on 2025-01-01..02" if deep_rows else f"HTTP {deep.status_code}: {deep.text[:80]}"
    return Result(
        OK,
        latest=f"{dates[-1] if dates else '?'} (meta.as_of {body.get('meta', {}).get('as_of')})",
        history=f"2025-01-01 (API floor; {deep_note})",
        limits=rate_headers(r),
        notes=f"last 7d: {len(rows)} rows over {len(dates)} days; {models_note}",
    )


def probe_eia() -> Result:
    url, key = "https://api.eia.gov/v2/electricity/retail-sales/data/", env("EIA_API_KEY")
    params = {"frequency": "monthly", "data[0]": "sales", "facets[sectorid][]": "COM", "facets[stateid][]": "US",
              "sort[0][column]": "period", "sort[0][direction]": "desc", "length": 1}
    if not key:
        return key_missing("eia", get(url, params=params), extra="DEMO_KEY works for testing.")
    r = get(url, params={**params, "api_key": key})
    data = ((r.json().get("response") or {}).get("data") or []) if r.status_code == 200 else []
    if not data:
        return Result(FAIL, notes=f"HTTP {r.status_code}: {r.text[:120]}")
    return Result(OK, latest=f"{data[0]['period']} US commercial sales {data[0].get('sales')}",
                  history="2001-01 (retail-sales)", limits=rate_headers(r))


def probe_massive() -> Result:
    key = env("MASSIVE_API_KEY")
    start, end = (TODAY - timedelta(days=800)).isoformat(), TODAY.isoformat()
    url = f"https://api.massive.com/v2/aggs/ticker/SPY/range/1/day/{start}/{end}"
    if not key:
        return key_missing("massive", get(url), extra="Optional (cross-check only).")
    r = get(url, params={"apiKey": key, "limit": 5000, "adjusted": "true"})
    res = r.json().get("results", []) if r.status_code == 200 else []
    if not res:
        return Result(FAIL, notes=f"HTTP {r.status_code}: {r.text[:120]}")
    day = lambda ms: datetime.fromtimestamp(ms / 1000, ET).date()  # noqa: E731
    return Result(OK, latest=f"SPY {day(res[-1]['t'])}", history=f"SPY {day(res[0]['t'])}", limits=rate_headers(r))


def probe_fmp() -> Result:
    url, key = "https://financialmodelingprep.com/stable/quote", env("FMP_API_KEY")
    if not key:
        return key_missing("fmp", get(url, params={"symbol": "AAPL"}), extra="Optional (cross-check only).")
    r = get(url, params={"symbol": "AAPL", "apikey": key})
    rows = r.json() if r.status_code == 200 else []
    if not isinstance(rows, list) or not rows:
        return Result(FAIL, notes=f"HTTP {r.status_code}: {r.text[:120]}")
    ts = rows[0].get("timestamp")
    return Result(OK, latest=f"AAPL {datetime.fromtimestamp(ts, ET):%Y-%m-%d}" if ts else "AAPL quote ok",
                  limits=rate_headers(r))


def probe_alphavantage() -> Result:
    key = env("ALPHAVANTAGE_API_KEY") or "demo"
    symbol = "IBM" if key == "demo" else "SPY"
    r = get("https://www.alphavantage.co/query", params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key})
    quote = r.json().get("Global Quote", {}) if r.status_code == 200 else {}
    if not quote:
        return Result(FAIL, notes=f"HTTP {r.status_code}: {r.text[:120]}")
    return Result(OK, latest=f"{symbol} {quote.get('07. latest trading day')}",
                  notes=f"{'demo key (IBM only)' if key == 'demo' else 'own key'}; skip per spec unless a gap needs it")


# --------------------------------------------------------------------------- #
# Positioning and surveys
# --------------------------------------------------------------------------- #
CFTC_CODES = {"13874A": "ES (E-mini S&P 500)", "209742": "NQ (E-mini Nasdaq-100)"}


def probe_cftc() -> Result:
    codes = ",".join(f"'{c}'" for c in CFTC_CODES)
    r = get("https://publicreporting.cftc.gov/resource/gpe5-46if.json", params={
        "$select": "cftc_contract_market_code, min(report_date_as_yyyy_mm_dd) as first, "
                   "max(report_date_as_yyyy_mm_dd) as last, count(*) as n",
        "$where": f"cftc_contract_market_code in ({codes})",
        "$group": "cftc_contract_market_code"})
    rows = {x["cftc_contract_market_code"]: x for x in r.json()} if r.status_code == 200 else {}
    if len(rows) < len(CFTC_CODES):
        return Result(FAIL, notes=f"HTTP {r.status_code}; contracts found: {list(rows)}")
    last = min(x["last"][:10] for x in rows.values())
    stale = (TODAY - date.fromisoformat(last)).days > 14
    return Result(
        WARN if stale else OK,
        latest=f"report {last} (Tuesday positions, released Friday)",
        history=", ".join(f"{CFTC_CODES[c]} {x['first'][:10]}" for c, x in rows.items()),
        notes="keyed on contract code (names changed in 2022)" + ("; STALE" if stale else ""),
    )


def probe_btos() -> Result:
    r = get("https://www.census.gov/hfp/btos/downloads/National.xlsx")
    if r.status_code != 200:
        return Result(FAIL, notes=f"National.xlsx HTTP {r.status_code}")
    wb = load_workbook(io.BytesIO(r.content), read_only=True, data_only=True)
    rows = wb["Response Estimates"].iter_rows(values_only=True)
    header = next(rows)
    periods = [str(h) for h in header[4:] if h]
    ai = None
    for row in rows:
        if "Artificial Intelligence" in str(row[1] or "") and str(row[3] or "").strip() == "Yes":
            ai = (str(row[1])[:70], row[4])
            break
    return Result(
        OK if ai else WARN,
        latest=f"period {periods[0]}" + (f": AI use 'Yes' {ai[1]}" if ai else ""),
        history=f"period {periods[-1]} ({len(periods)} periods)",
        notes=(f"question: '{ai[0]}…'" if ai else "AI question row not found")
              + "; series break 2025-11-17 (question broadened)",
    )


def probe_naaim() -> Result:
    r = get("https://naaim.org/wp-json/wp/v2/media", params={
        "search": "USE_Data", "per_page": 1, "orderby": "date", "order": "desc"})
    media = r.json() if r.status_code == 200 else []
    if not media:
        return Result(FAIL, notes=f"WP media search HTTP {r.status_code}")
    url, uploaded = media[0]["source_url"], media[0]["date"][:10]
    x = get(url)
    wb = load_workbook(io.BytesIO(x.content), read_only=True, data_only=True)
    dates = [c for (c, *_) in wb.active.iter_rows(min_row=2, values_only=True) if isinstance(c, datetime)]
    last = max(dates).date() if dates else None
    age = (TODAY - (last or date.fromisoformat(uploaded))).days
    return Result(
        OK if age <= 10 else WARN,
        latest=f"{last} ({age} days old)",
        history=str(min(dates).date()) if dates else "—",
        notes=f"newest xlsx uploaded {uploaded}: {url.rsplit('/', 1)[-1]}"
              + ("; feed STALE, current value is JS-rendered on the page" if age > 10 else ""),
    )


def probe_aaii() -> Result:
    r = get("https://www.aaii.com/sentimentsurvey/sent_results")
    table = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", r.text, re.S):
        cells = [strip_tags(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        if len(cells) >= 4 and cells[1].endswith("%"):
            table.append(cells)
    if not table:
        blocked = "Incapsula" in r.text
        return Result(FAIL, notes=f"HTTP {r.status_code}; no results table" + (" (Incapsula challenge)" if blocked else ""))
    d, bull, neutral, bear = table[0][:4]
    return Result(
        OK,
        latest=f"{d}: bull {bull} / neutral {neutral} / bear {bear}",
        history=f"{len(table)} weeks visible (oldest {table[-1][0]})",
        notes="accumulate weekly; full history is members-only",
    )


def probe_finra() -> Result:
    page = get("https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics")
    m = re.search(r"/sites/default/files/[^\"']+margin-statistics[^\"']*\.xlsx", page.text)
    path = m.group(0) if m else "/sites/default/files/2021-03/margin-statistics.xlsx"
    r = get("https://www.finra.org" + path)
    if r.status_code != 200:
        return Result(FAIL, notes=f"xlsx HTTP {r.status_code}")
    wb = load_workbook(io.BytesIO(r.content), read_only=True, data_only=True)
    rows = [(str(a), b) for a, b, *_ in wb.active.iter_rows(min_row=2, values_only=True)
            if a and re.match(r"^\d{4}-\d{2}$", str(a))]
    rows.sort()
    return Result(
        OK,
        latest=f"{rows[-1][0]}: debit balances ${rows[-1][1] / 1000:,.1f}B",
        history=rows[0][0],
        notes=f"xlsx at {path}",
    )


# --------------------------------------------------------------------------- #
# Earnings / valuation files
# --------------------------------------------------------------------------- #
def probe_spdji() -> Result:
    r = get("https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx",
            headers={"Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"})
    if r.status_code == 200 and "spreadsheet" in r.headers.get("content-type", ""):
        wb = load_workbook(io.BytesIO(r.content), read_only=True)
        return Result(OK, notes=f"sheets: {', '.join(wb.sheetnames[:5])}")
    return Result(FAIL, notes=f"HTTP {r.status_code} ({r.headers.get('server', '?')}): bot protection")


def probe_factset() -> Result:
    base = "https://advantage.factset.com/hubfs/Website/Resources%20Section/Research%20Desk/Earnings%20Insight/"
    for back in range(15):
        d = TODAY - timedelta(days=back)
        r = get(f"{base}EarningsInsight_{d:%m%d%y}.pdf")
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
            return Result(OK, latest=str(d), history="weekly PDFs (archive by URL date)",
                          notes=f"{len(r.content) // 1024} KB PDF: forward P/E, NTM EPS, % beating; needs PDF parsing")
    return Result(FAIL, notes="no Earnings Insight PDF found in the last 15 days")


def probe_shiller() -> Result:
    page = get("https://shillerdata.com/")
    m = re.search(r"(?:https?:)?//img1\.wsimg\.com/[^\"']+/ie_data\.xls[^\"']*", page.text)
    yale = head("http://www.econ.yale.edu/~shiller/data/ie_data.xls").headers.get("last-modified", "?")
    if not m:
        return Result(FAIL, notes=f"ie_data.xls link not found on shillerdata.com; yale.edu copy last modified {yale}")
    url = m.group(0) if m.group(0).startswith("http") else "https:" + m.group(0)
    lm = head(url).headers.get("last-modified")
    age = (NOW - datetime.strptime(lm, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=ZoneInfo("UTC"))).days if lm else None
    return Result(
        OK if age is not None and age <= 45 else WARN,
        latest=f"file updated {lm}",
        history="1871-01",
        notes=f"link has a GUID: scrape shillerdata.com each run; old yale.edu copy frozen ({yale}); .xls needs xlrd",
    )


# --------------------------------------------------------------------------- #
# Taiwan supply chain
# --------------------------------------------------------------------------- #
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def probe_tsmc() -> Result:
    # Plain curl gets a 403 here; httpx's TLS/HTTP fingerprint is let through.
    r = get(f"https://investor.tsmc.com/english/monthly-revenue/{TODAY.year}")
    if r.status_code != 200:
        return Result(FAIL, notes=f"HTTP {r.status_code}: bot protection")
    months = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", r.text, re.S):
        cells = [strip_tags(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        if len(cells) >= 3 and cells[0][:3] in MONTHS and re.match(r"^[\d,]+$", cells[1]):
            months.append(cells)
    if not months:
        return Result(WARN, notes="page reachable but no revenue table parsed")
    month, revenue, yoy = months[-1][:3]
    return Result(
        OK,
        latest=f"{TODAY.year}-{MONTHS.index(month[:3]) + 1:02d}: NT${revenue}M ({yoy} YoY)",
        history="yearly pages (URL per year)",
        notes=f"{len(months)} months on the {TODAY.year} page; cross-check vs MOPS 2330",
    )


MOPS_TICKERS = {"2330": "TSMC", "2382": "Quanta", "3231": "Wistron", "6669": "Wiwynn", "3661": "Alchip"}


def _mops_table(roc_year: int, month: int, suffix: int) -> dict[str, list[str]]:
    r = get(f"https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{roc_year}_{month}_{suffix}.html")
    if r.status_code != 200:
        return {}
    html = r.content.decode("cp950", "replace")
    out = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = [strip_tags(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if cells and cells[0] in MOPS_TICKERS:
            out[cells[0]] = cells
    return out


def probe_mops() -> Result:
    y, m, found = TODAY.year, TODAY.month, {}
    for _ in range(3):
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
        found = {**_mops_table(y - 1911, m, 0), **_mops_table(y - 1911, m, 1)}  # _1 = foreign (KY) issuers
        if "2330" in found:
            break
    if not found:
        return Result(FAIL, notes="no monthly revenue table in the last 3 months")
    table = ["### TWSE MOPS monthly revenue", "", "| Ticker | Name | Revenue (k TWD) |", "|---|---|---|"]
    table += [f"| {t} | {MOPS_TICKERS[t]} | {found[t][2]} |" for t in MOPS_TICKERS if t in found]
    missing = [t for t in MOPS_TICKERS if t not in found]
    return Result(
        OK if not missing else WARN,
        latest=f"{y}-{m:02d}",
        history="monthly files back to 2013 (ROC-year URLs)",
        notes=f"{len(found)}/{len(MOPS_TICKERS)} tickers (Alchip in the _1 KY table); cp950 encoded"
              + (f"; missing {missing}" if missing else ""),
        details=["\n".join(table)],
    )


# --------------------------------------------------------------------------- #
# Compute economics
# --------------------------------------------------------------------------- #
def probe_uvik() -> Result:
    y, m, months, latest = TODAY.year, TODAY.month, [], None
    for _ in range(4):
        r = get(f"https://uvik.net/wp-content/uploads/{y}/{m:02d}/uvik-software-gpu-price-index-{y}-{m:02d}.csv")
        if r.status_code == 200 and r.text.startswith("provider"):
            months.append(f"{y}-{m:02d}")
            latest = latest or r.text
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    if not latest:
        return Result(FAIL, notes="no monthly CSV in the last 4 months")
    rows = list(csv.DictReader(io.StringIO(latest)))
    h100 = [float(x["usd_per_gpu_hour"]) for x in rows if x["gpu"].startswith("H100") and x["usd_per_gpu_hour"]]
    return Result(
        OK,
        latest=f"captured {max(x['captured'] for x in rows)}; H100 median ${statistics.median(h100):.2f}/GPU-hr",
        history=f"files found: {', '.join(months)}",
        notes=f"{len(rows)} rows, {len(h100)} H100; no back history, accumulate monthly",
    )


def probe_semianalysis() -> Result:
    return Result(FAIL, notes="no public machine-readable H100 1y rental index found (checked 2026-10-15)")


def probe_gpuperhour() -> Result:
    r = get("https://api.gpuperhour.com/api/price-index/export.csv", params={"days": 365})
    rows = list(csv.DictReader(io.StringIO(r.text))) if r.status_code == 200 else []
    if not rows:
        return Result(FAIL, notes=f"HTTP {r.status_code}")
    dates = sorted({x["snapshot_date"] for x in rows})
    h100 = [x for x in rows if x["snapshot_date"] == dates[-1] and "h100" in x["gpu_slug"]]
    return Result(
        OK,
        latest=f"{dates[-1]}" + (f"; {h100[0]['gpu_name']} median ${float(h100[0]['median_price_per_gpu']):.2f}" if h100 else ""),
        history=dates[0],
        notes=f"{len(dates)} daily snapshots; on-demand marketplace prices, not 1y contracts",
    )


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
def probe_gdelt() -> Result:
    r = get("https://api.gdeltproject.org/api/v2/doc/doc", params={
        "query": '"artificial intelligence"', "mode": "timelinetone", "format": "json", "timespan": "7d"})
    try:
        data = r.json()["timeline"][0]["data"]
    except (ValueError, KeyError, IndexError):
        return Result(FAIL, notes=f"HTTP {r.status_code}: {r.text[:120]}")
    ts = [datetime.strptime(x["date"], "%Y%m%dT%H%M%SZ") for x in data]
    gap = max((b - a for a, b in zip(ts, ts[1:])), default=timedelta(0))
    return Result(
        OK if gap <= timedelta(hours=24) else WARN,
        latest=f"{ts[-1]:%Y-%m-%d %H:%M} UTC" if ts else "—",
        history="rolling window (timelines up to ~3 months)",
        notes=f"{len(ts)} buckets in 7d; largest gap {gap}",
    )


def probe_wikimedia() -> Result:
    out = {}
    for article in ("Nvidia", "AI_bubble"):
        r = get(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
                f"{article}/daily/{TODAY - timedelta(days=14):%Y%m%d}/{TODAY:%Y%m%d}", headers={"User-Agent": PROJECT_UA})
        items = r.json().get("items", []) if r.status_code == 200 else []
        out[article] = items[-1] if items else None
    if not any(out.values()):
        return Result(FAIL, notes="no pageview items returned")
    latest = "; ".join(f"{a} {i['timestamp'][:8]} {i['views']:,}" for a, i in out.items() if i)
    missing = [a for a, i in out.items() if not i]
    return Result(OK if not missing else WARN, latest=latest, history="2015-07-01",
                  notes=f"missing {missing}" if missing else "")


# --------------------------------------------------------------------------- #
# Cross-check source
# --------------------------------------------------------------------------- #
def probe_treasury() -> Result:
    r = get(f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
            f"{TODAY.year}/all", params={"type": "daily_treasury_yield_curve",
                                         "field_tdr_date_value": TODAY.year, "_format": "csv"})
    rows = list(csv.DictReader(io.StringIO(r.text))) if r.status_code == 200 else []
    points = {parse_mdy(x["Date"]): float(x["10 Yr"]) for x in rows if x.get("10 Yr")}
    if not points:
        return Result(FAIL, notes=f"HTTP {r.status_code}")
    remember("treasury:10y", points)
    last = max(points)
    return Result(OK, latest=f"{last}: 10y {points[last]}", history="1990 (yearly CSVs)",
                  notes="used for the DGS10 cross-check")


# --------------------------------------------------------------------------- #
# Registry, cross-checks, report
# --------------------------------------------------------------------------- #
PROBES: list[tuple[str, str, Callable[[], Result]]] = [
    ("fred", "FRED (series)", probe_fred),
    ("alfred", "ALFRED (pre-cutoff ICE BofA)", probe_alfred),
    ("cboe_csv", "Cboe index CSVs", probe_cboe_csv),
    ("putcall", "Cboe daily market stats (put/call)", probe_putcall),
    ("yahoo", "yfinance / Yahoo", probe_yahoo),
    ("nasdaq", "Nasdaq API (Yahoo fallback)", probe_nasdaq),
    ("sec", "SEC EDGAR", probe_sec),
    ("finnhub", "Finnhub", probe_finnhub),
    ("openrouter", "OpenRouter", probe_openrouter),
    ("cftc", "CFTC TFF (Socrata gpe5-46if)", probe_cftc),
    ("btos", "Census BTOS", probe_btos),
    ("naaim", "NAAIM Exposure Index", probe_naaim),
    ("aaii", "AAII Sentiment Survey", probe_aaii),
    ("finra", "FINRA margin statistics", probe_finra),
    ("spdji", "S&P DJI sp-500-eps-est.xlsx", probe_spdji),
    ("factset", "FactSet Earnings Insight (S&P DJI fallback)", probe_factset),
    ("shiller", "Shiller CAPE", probe_shiller),
    ("tsmc", "TSMC IR monthly revenue", probe_tsmc),
    ("mops", "TWSE MOPS monthly revenue", probe_mops),
    ("uvik", "Uvik GPU Price Index", probe_uvik),
    ("semianalysis", "SemiAnalysis H100 1y rental index", probe_semianalysis),
    ("gpuperhour", "GPUPerHour index (GPU fallback)", probe_gpuperhour),
    ("gdelt", "GDELT DOC 2.0", probe_gdelt),
    ("wikimedia", "Wikimedia Pageviews", probe_wikimedia),
    ("eia", "EIA API v2", probe_eia),
    ("massive", "Massive (ex-Polygon) Basic", probe_massive),
    ("fmp", "FMP Basic", probe_fmp),
    ("alphavantage", "Alpha Vantage", probe_alphavantage),
    ("treasury", "US Treasury yield curve (cross-check)", probe_treasury),
]

LIMITS = {
    "fred": "API key, 120 req/min; graph CSV keyless",
    "alfred": "FRED API key",
    "cboe_csv": "none",
    "putcall": "none",
    "yahoo": "unofficial; plain HTTP gets 429, yfinance (curl_cffi) passes",
    "nasdaq": "unofficial, undocumented",
    "sec": "declared User-Agent with email; <10 req/s",
    "finnhub": "free key; ~60 req/min",
    "openrouter": "/models keyless; Data API key, 30 req/min, 500/day",
    "cftc": "none (app token optional)",
    "btos": "none",
    "naaim": "none",
    "aaii": "none",
    "finra": "none",
    "spdji": "none (bot-protected)",
    "factset": "none",
    "shiller": "none",
    "tsmc": "none (bot-protected)",
    "mops": "none",
    "uvik": "none",
    "semianalysis": "—",
    "gpuperhour": "none",
    "gdelt": "none; ~1 req/5s",
    "wikimedia": "descriptive User-Agent",
    "eia": "free key",
    "massive": "free key; 5 req/min, EOD, 2y",
    "fmp": "free key; 250/day, fixed symbol list",
    "alphavantage": "free key; 25/day",
    "treasury": "none",
}

TERMS = {
    "fred": "ICE BofA series: 3y window (licensing)",
    "alfred": "ICE licensing",
    "cboe_csv": "Cboe site terms",
    "putcall": "Cboe site terms",
    "yahoo": "Yahoo ToS gray area for automation",
    "nasdaq": "unofficial; ToS gray area",
    "sec": "public; fair-access policy",
    "finnhub": "personal, non-commercial",
    "openrouter": "CC BY 4.0; cite 'Source: OpenRouter (openrouter.ai/rankings), as of {as_of}'",
    "cftc": "public",
    "btos": "public (Census)",
    "naaim": "public page",
    "aaii": "recent weeks public; Incapsula bot protection",
    "finra": "public",
    "spdji": "S&P DJI terms; Akamai bot protection",
    "factset": "FactSet terms; check reuse",
    "shiller": "public; cite Shiller",
    "tsmc": "bot-protected",
    "mops": "public (TWSE)",
    "uvik": "CC BY 4.0; credit Uvik Software",
    "semianalysis": "robots crawl-delay 100; check terms",
    "gpuperhour": "terms unverified",
    "gdelt": "open; cite GDELT",
    "wikimedia": "CC0 data; UA policy",
    "eia": "public domain",
    "massive": "free-tier ToS",
    "fmp": "free-tier ToS",
    "alphavantage": "free-tier ToS",
    "treasury": "public domain",
}

SUBSTITUTES = {
    "fred": "Look up Richmond and KC composite IDs via keyed FRED search, or scrape each Fed's site",
    "alfred": "Moody's BAA10Y/AAA10Y on FRED (1986+) for IG; HYG/LQD vs IEF ETF proxy for HY (2007+); "
              "any pre-April-2026 CSVs on hand",
    "yahoo": "Nasdaq API for EOD and option chains; no free fallback for ^MOVE, futures or KRX",
    "nasdaq": "Yahoo via yfinance",
    "spdji": "FactSet Earnings Insight weekly PDF, or a manual weekly xlsx drop into data/inbox/",
    "factset": "manual weekly entry in events.yaml",
    "shiller": "scrape the shillerdata.com link each run",
    "tsmc": "TWSE MOPS (includes 2330)",
    "naaim": "reverse-engineer the page's admin-ajax call, or a weekly manual events.yaml entry",
    "semianalysis": "monthly manual events.yaml entry; GPUPerHour daily CSV or Silicon Data SDH100RT as alternates",
    "uvik": "accumulate monthly from the first run",
    "gdelt": "keep at low weight; Wikimedia attention as primary",
    "btos": "check the AI question wording in the Data Dictionary sheet",
}


def run_probe(key: str, label: str, fn: Callable[[], Result]) -> Result:
    try:
        res = fn()
    except Exception as e:  # noqa: BLE001 - one broken source must not sink the report
        res = Result(FAIL, notes=f"{type(e).__name__}: {str(e)[:160]}")
    res.source, res.key, res.notes = label, key, redact(res.notes)
    res.limits = "; ".join(x for x in (LIMITS.get(key, ""), res.limits) if x)
    res.terms = res.terms or TERMS.get(key, "—")
    return res


def crosschecks() -> list[str]:
    checks = [
        ("SPX close", "yf:^GSPC", "fred:SP500", "±0.15%",
         lambda a, b: abs(a / b - 1) <= 0.0015, lambda a, b: f"{(a / b - 1) * 100:+.3f}%"),
        ("VIX close", "cboe:VIX", "fred:VIXCLS", "exact",
         lambda a, b: round(a, 2) == round(b, 2), lambda a, b: f"{a - b:+.2f}"),
        ("10y yield", "fred:DGS10", "treasury:10y", "±2bp",
         lambda a, b: abs(a - b) <= 0.02 + 1e-9, lambda a, b: f"{(a - b) * 100:+.1f}bp"),
    ]
    lines = ["| Check | Primary | Secondary | Date | Primary value | Secondary value | Diff | Tolerance | Result |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, ka, kb, tol, passes, diff in checks:
        a, b = _obs.get(ka), _obs.get(kb)
        common = sorted(set(a) & set(b)) if a and b else []
        if not common:
            why = "source unavailable" if not (a and b) else "no common date"
            lines.append(f"| {name} | {ka} | {kb} | — | — | — | — | {tol} | skipped ({why}) |")
            continue
        d = common[-1]
        va, vb = a[d], b[d]
        lines.append(f"| {name} | {ka} | {kb} | {d} | {va:g} | {vb:g} | {diff(va, vb)} | {tol} | "
                     f"{'✅ pass' if passes(va, vb) else '❌ FLAG'} |")
    return lines


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render(results: list[Result]) -> str:
    keys = " · ".join(f"{name} {'✓' if env(name) else '✗'}" for name in KEY_ENVS.values())
    counts = {s: sum(r.status == s for r in results) for s in (OK, WARN, KEY, FAIL)}
    out = [
        "# Source status (Phase 0 probe)",
        "",
        f"Run {NOW.astimezone(PT):%Y-%m-%d %H:%M} PT ({NOW:%H:%M} ET) · Python {platform.python_version()} · "
        f"`scripts/probe_sources.py`",
        "",
        f"Keys: {keys}",
        "",
        "Legend: ✅ works · ⚠️ works with a problem · 🔑 needs a free key · ❌ blocked or no feed",
        "",
        "| # | Source | Status | Latest obs | History start | Auth / rate limits | Terms flags | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        out.append(f"| {i} | {md(r.source)} | {r.status} | {md(r.latest)} | {md(r.history)} | "
                   f"{md(r.limits)} | {md(r.terms)} | {md(r.notes)} |")
    out += ["", f"**Tally:** {counts[OK]} ✅ · {counts[WARN]} ⚠️ · {counts[KEY]} 🔑 · {counts[FAIL]} ❌", "",
            "## Failures, warnings and substitutes", ""]
    for r in results:
        if r.status == KEY:
            out.append(f"- {KEY} **{r.source}**: set `{KEY_ENVS[r.key]}` in `.env`.")
        elif r.status in (WARN, FAIL):
            out.append(f"- {r.status} **{r.source}**: {r.notes}. Substitute: {SUBSTITUTES.get(r.key, 'n/a')}.")
    out += ["", "## Cross-checks (Section 3.5)", "", *crosschecks()]
    blocks = [block for r in results for block in r.details]
    if blocks:
        out += ["", "## Details"]
        for block in blocks:
            out += ["", block]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma-separated probe keys: " + ",".join(k for k, _, _ in PROBES))
    parser.add_argument("--no-write", action="store_true", help="print only; do not write the report")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    selected = PROBES
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        unknown = wanted - {k for k, _, _ in PROBES}
        if unknown:
            parser.error(f"unknown probe keys: {', '.join(sorted(unknown))}")
        selected = [p for p in PROBES if p[0] in wanted]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda p: run_probe(*p), selected))

    report = redact(render(results))
    print(report)
    if not args.no_write:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
