"""Data access. Free sources only; no API key is required for the scores.

Yahoo (yfinance)  index, ETF and member prices, VIX family history
AAII              weekly sentiment and monthly asset-allocation surveys (context only)
Cboe              COR1M history CSV, daily put/call ratios
FRED (keyless)    Fed Z.1 series for the equity-allocation model
shillerdata.com   CAPE
SSGA              S&P 500 member list (SPY holdings)
Nebius            GPU spot prices (context only)
Finnhub           AI company earnings dates and surprises (optional, needs FINNHUB_API_KEY)
"""
from __future__ import annotations

import io
import os
import re
import time as _time
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "data" / "state"
ET = ZoneInfo("America/New_York")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")

CBOE_INDEX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
CBOE_PC = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{day}_daily_options"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
SHILLER_PAGE = "https://shillerdata.com/"
SSGA_SPY = ("https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/"
            "holdings-daily-us-en-spy.xlsx")
AAII_SENTIMENT = "https://www.aaii.com/files/surveys/sentiment.xls"
AAII_ALLOCATION = "https://www.aaii.com/files/surveys/asset.xls"
NEBIUS = "https://api.nebius.cloud/billing/v1alpha1/spot-price/list-series"
NEBIUS_START = date(2026, 10, 8)   # prices were fixed, not market-driven, before this
FINNHUB = "https://finnhub.io/api/v1/"

PC_NAMES = {"TOTAL PUT/CALL RATIO": "pc_total", "EQUITY PUT/CALL RATIO": "pc_equity",
            "INDEX PUT/CALL RATIO": "pc_index"}


class SourceLog:
    """What each source returned this run; written into signals.json."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, ok: bool, latest=None, note: str = "") -> None:
        self.rows.append({"name": name, "ok": bool(ok), "latest": str(latest) if latest is not None else None,
                          "note": note})


def completed_session(now: datetime | None = None) -> date:
    """The most recent weekday whose US cash session has closed (holidays simply have no bar)."""
    now = now or datetime.now(ET)
    d = now.date()
    if now.weekday() >= 5 or now.time() < time(16, 30):
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _get(url: str, params: dict | None = None, tries: int = 3, timeout: float = 45) -> httpx.Response:
    last: Exception | None = None
    for i in range(tries):
        try:
            r = httpx.get(url, params=params, headers={"User-Agent": UA}, timeout=timeout, follow_redirects=True)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} {url.split('?')[0]}")
            if r.status_code in (400, 401, 403, 404):
                break
        except httpx.HTTPError as e:
            last = e
        _time.sleep(1.5 * (i + 1))
    raise last or RuntimeError(url)


def cached(name: str, fetch, log: SourceLog, label: str | None = None, index_col: int | None = 0) -> pd.DataFrame:
    """Run `fetch`; on success save it to data/state/<name>.csv, on failure fall back to that file."""
    path = STATE / f"{name}.csv"
    label = label or name
    try:
        df = fetch()
        if df is None or df.empty:
            raise RuntimeError("empty result")
        STATE.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index_col is not None)
        log.add(label, True, df.index.max().date() if index_col is not None and hasattr(df.index.max(), "date") else None)
        return df
    except Exception as e:  # noqa: BLE001 - any source failure falls back to the last good copy
        if not path.exists():
            log.add(label, False, note=f"{type(e).__name__}: {e}"[:200])
            raise
        df = pd.read_csv(path, index_col=index_col, parse_dates=index_col is not None)
        log.add(label, False, df.index.max().date() if index_col is not None else None,
                note=f"using last good copy ({type(e).__name__}: {e})"[:200])
        return df


# ------------------------------------------------------------------ Yahoo
def yahoo_close(symbols: list[str], session: date, start: str = "1990-01-01", chunk: int = 100) -> pd.DataFrame:
    """Adjusted closes through `session` (today's unfinished bar is never included)."""
    import yfinance as yf

    end = (session + timedelta(days=1)).isoformat()
    parts = []
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False, threads=True)
                if df is not None and not df.empty:
                    break
            except Exception:  # noqa: BLE001
                df = None
            _time.sleep(3 * (attempt + 1))
        if df is None or df.empty:
            continue
        close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].set_axis(batch, axis=1)
        parts.append(close)
    if not parts:
        raise RuntimeError("Yahoo returned no prices")
    out = pd.concat(parts, axis=1, sort=True).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.loc[:pd.Timestamp(session)]
    return out.loc[:, ~out.columns.duplicated()].dropna(axis=1, how="all")


# ------------------------------------------------------------------- Cboe
def cboe_index(sym: str) -> pd.DataFrame:
    """Official daily close history for a Cboe index (DATE,...,CLOSE)."""
    text = _get(CBOE_INDEX.format(sym=sym)).text
    df = pd.read_csv(io.StringIO(text))
    out = pd.DataFrame({sym: pd.to_numeric(df.iloc[:, -1], errors="coerce").values},
                       index=pd.to_datetime(df.iloc[:, 0], format="%m/%d/%Y"))
    out.index.name = "date"
    return out.dropna()


def putcall(session: date, log: SourceLog, max_days: int = 400) -> pd.DataFrame:
    """Cboe daily put/call ratios. History lives in data/state/putcall.csv; each run appends the missing days."""
    path = STATE / "putcall.csv"
    hist = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()
    last = hist.index.max().date() if not hist.empty else session - timedelta(days=max_days)
    days = [d.date() for d in pd.bdate_range(last + timedelta(days=1), session)][-max_days:]
    rows, failed = {}, 0
    for d in days:
        try:
            r = httpx.get(CBOE_PC.format(day=d.isoformat()), headers={"User-Agent": UA}, timeout=30)
            if r.status_code != 200:      # holidays and not-yet-posted days return 403
                continue
            vals = {PC_NAMES[i["name"].strip()]: float(i["value"]) for i in r.json().get("ratios", [])
                    if i.get("name", "").strip() in PC_NAMES and i.get("value") not in (None, "")}
            if vals:
                rows[pd.Timestamp(d)] = vals
        except Exception:  # noqa: BLE001
            failed += 1
    if rows:
        new = pd.DataFrame.from_dict(rows, orient="index")
        hist = pd.concat([hist, new]).sort_index()
        hist = hist[~hist.index.duplicated(keep="last")]
        hist.index.name = "date"
        STATE.mkdir(parents=True, exist_ok=True)
        hist.to_csv(path)
    if hist.empty:
        log.add("Cboe put/call", False, note="no history in data/state/putcall.csv")
        return hist
    stale = (pd.Timestamp(session) - hist.index.max()).days > 4
    log.add("Cboe put/call", not stale and not failed, hist.index.max().date(),
            f"{len(rows)} new day(s)" + (f", {failed} request(s) failed" if failed else ""))
    return hist


# ------------------------------------------------------------------- FRED
def fred(series_id: str) -> pd.DataFrame:
    """One FRED series: the JSON API when FRED_API_KEY is set (fast), else the keyless CSV (slow, sometimes throttled)."""
    key = os.environ.get("FRED_API_KEY")
    out = None
    if key:
        try:
            obs = _get(FRED_API, {"series_id": series_id, "api_key": key, "file_type": "json"},
                       tries=2, timeout=30).json()["observations"]
            out = pd.DataFrame({series_id: [float(o["value"]) if o["value"] != "." else float("nan") for o in obs]},
                               index=pd.to_datetime([o["date"] for o in obs]))
        except Exception:  # noqa: BLE001 - fall through to the keyless endpoint
            out = None
    if out is None:
        df = pd.read_csv(io.StringIO(_get(FRED_CSV, {"id": series_id}, tries=2, timeout=60).text))
        out = pd.DataFrame({series_id: pd.to_numeric(df.iloc[:, 1], errors="coerce").values},
                           index=pd.to_datetime(df.iloc[:, 0]))
    out.index.name = "date"
    return out.dropna()


def fred_many(ids: list[str]) -> pd.DataFrame:
    return pd.concat([fred(i) for i in ids], axis=1, sort=True).sort_index()


# ---------------------------------------------------------------- Shiller
def shiller() -> pd.DataFrame:
    """Monthly price, dividend, earnings, CPI and CAPE from Robert Shiller's ie_data.xls."""
    import xlrd

    m = re.search(r"(?:https?:)?//img1\.wsimg\.com/[^\"']+/ie_data\.xls[^\"']*", _get(SHILLER_PAGE).text)
    if not m:
        raise RuntimeError("ie_data.xls link not found on shillerdata.com")
    url = m.group(0) if m.group(0).startswith("http") else "https:" + m.group(0)
    sheet = xlrd.open_workbook(file_contents=_get(url).content).sheet_by_name("Data")
    head = next(i for i in range(min(20, sheet.nrows)) if str(sheet.cell_value(i, 0)).strip() == "Date")
    header = [str(v).strip() for v in sheet.row_values(head)]
    cols = {"P": "price", "D": "dividend", "E": "earnings", "CPI": "cpi", "CAPE": "cape"}
    idx = {name: header.index(c) for c, name in cols.items() if c in header}
    rows = {}
    for i in range(head + 1, sheet.nrows):
        r = sheet.row_values(i)
        if not isinstance(r[0], float):
            continue
        year = int(r[0]); month = int(round((r[0] - year) * 100))   # 2026.1 is October, 2026.01 January
        if 1 <= month <= 12:
            rows[pd.Timestamp(year, month, 1)] = {n: r[j] for n, j in idx.items() if isinstance(r[j], float)}
    out = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out.index.name = "date"
    return out


# ------------------------------------------------------------------- AAII
def _aaii_xls(url: str, names: list[str]) -> pd.DataFrame:
    """AAII publishes each survey's full history as one .xls: a date column, then shares that sum to 1."""
    import xlrd

    wb = xlrd.open_workbook(file_contents=_get(url).content)
    sh = wb.sheet_by_index(0)
    n, rows = len(names) + 1, {}
    for i in range(sh.nrows):
        v = sh.row_values(i)
        if isinstance(v[0], float) and v[0] > 20000 and all(isinstance(x, float) for x in v[1:n]):
            rows[pd.Timestamp(xlrd.xldate_as_datetime(v[0], wb.datemode).date())] = [x * 100 for x in v[1:n]]
    out = pd.DataFrame.from_dict(rows, orient="index", columns=names).sort_index().round(2)
    out.index.name = "date"
    if len(out) < 300:
        raise RuntimeError(f"AAII file parsed to only {len(out)} rows")
    return out


def aaii_sentiment() -> pd.DataFrame:
    """Weekly bullish / neutral / bearish shares (percent) since 1987, stamped with the Thursday they were reported."""
    return _aaii_xls(AAII_SENTIMENT, ["bull", "neutral", "bear"])


def aaii_allocation() -> pd.DataFrame:
    """Monthly asset allocation survey (percent) since 1987, stamped at month end."""
    return _aaii_xls(AAII_ALLOCATION, ["stock_funds", "stocks", "bond_funds", "bonds", "cash"])


# ------------------------------------------------------------------- SSGA
def spy_members() -> pd.DataFrame:
    """Current S&P 500 members as Yahoo symbols (from SPDR's daily SPY holdings file)."""
    from openpyxl import load_workbook

    rows = list(load_workbook(io.BytesIO(_get(SSGA_SPY).content), read_only=True, data_only=True)
                .worksheets[0].iter_rows(values_only=True))
    head = next(i for i, r in enumerate(rows) if r and r[0] == "Name" and len(r) > 1 and r[1] == "Ticker")
    col = {n: i for i, n in enumerate(rows[head]) if n}
    out = []
    for r in rows[head + 1:]:
        if not r or all(v is None for v in r):
            break
        t = str(r[col["Ticker"]] or "").strip()
        if re.fullmatch(r"[A-Z][A-Z0-9]*(\.[A-Z])?", t) and r[col["Weight"]] is not None:
            out.append(t.replace(".", "-") if re.fullmatch(r"[A-Z]+\.[A-Z]", t) else t)
    if len(out) < 400:
        raise RuntimeError(f"SPY holdings: only {len(out)} tickers parsed")
    return pd.DataFrame({"ticker": sorted(set(out))})


# ----------------------------------------------------------------- Nebius
def nebius(log: SourceLog) -> pd.DataFrame:
    """Daily mean GPU spot price per platform.region. The API serves 30 days, so history accrues in state."""
    path = STATE / "nebius_spot.csv"
    hist = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()
    try:
        today = datetime.now(timezone.utc).date()
        params = {"filters.startTime": f"{(today - timedelta(days=29)).isoformat()}T00:00:00Z",
                  "aggregation.granularity": "GRANULARITY_HOUR_1", "aggregation.method": "METHOD_AVG_PRICE",
                  "pageSize": 100}
        days: dict[tuple, list[float]] = {}
        token = None
        while True:
            payload = _get(NEBIUS, {**params, **({"pageToken": token} if token else {})}).json()
            for item in payload.get("items", []):
                series = f"{item['computeInstance']['platform']}.{item['region']}"
                for p in item.get("points", []):
                    hour = datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00")) - timedelta(hours=1)
                    if NEBIUS_START <= hour.date() < today:
                        days.setdefault((pd.Timestamp(hour.date()), series), []).append(float(p["value"]))
            token = payload.get("nextPageToken")
            if not token:
                break
        if days:
            new = pd.Series({k: sum(v) / len(v) for k, v in days.items()}).unstack()
            hist = new.combine_first(hist).sort_index() if not hist.empty else new.sort_index()
            hist.index.name = "date"
            STATE.mkdir(parents=True, exist_ok=True)
            hist.round(5).to_csv(path)
        log.add("Nebius spot", True, hist.index.max().date() if not hist.empty else None)
    except Exception as e:  # noqa: BLE001
        log.add("Nebius spot", False, hist.index.max().date() if not hist.empty else None,
                f"{type(e).__name__}: {e}"[:200])
    return hist


# ---------------------------------------------------------------- Finnhub
def finnhub(path: str, **params) -> dict | list:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY not set")
    r = httpx.get(FINNHUB + path, params={**params, "token": key}, timeout=30)
    if r.status_code == 429:
        _time.sleep(20)
        r = httpx.get(FINNHUB + path, params={**params, "token": key}, timeout=30)
    r.raise_for_status()
    return r.json()
