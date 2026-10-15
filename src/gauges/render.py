"""Write data/latest.json and data/history.json for the dashboard (BUILD_PROMPT.md Section 9).

Usage: python -m gauges.render [--db PATH]   (run gauges.score first; make refresh does both)

Everything shown on the dashboard comes from these two files, and every number
in them traces to the raw store or to an events.yaml / calendar.yaml entry with
a source URL.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from gauges.config import ROOT, load_baskets, load_calendar, load_events, load_series_config
from gauges.features import Features
from gauges.registry import Indicator, Registry, load_registry
from gauges.store import DEFAULT_DB, Store
from gauges.validate import CROSSCHECKS

DATA_DIR = ROOT / "data"
ET, PT = ZoneInfo("America/New_York"), ZoneInfo("America/Los_Angeles")
REFRESH_UTC_HOUR, REFRESH_WEEKDAYS = 2, {1, 2, 3, 4, 5}   # cron "0 2 * * 2-6" (Tue-Sat 02:00 UTC)
SPARK_DAYS, SPARK_STEP = 126, 2
CATALYST_DAYS = 30
VERIFIED_REFS = {f"{s}:{k}" for c in CROSSCHECKS for s, k in (c.primary, c.secondary)}
SOURCE_PAGES = {
    "cboe": "https://www.cboe.com/us/indices/dashboard/{series}/",
    "yfinance": "https://finance.yahoo.com/quote/{ticker}",
    "finra": "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics",
    "factset": "https://insight.factset.com/topic/earnings",
    "shiller": "https://shillerdata.com/",
    "tsmc": "https://investor.tsmc.com/english/monthly-revenue",
    "mops": "https://mopsov.twse.com.tw/mops/web/t21sc04_ifrs",
    "aaii": "https://www.aaii.com/sentimentsurvey",
    "naaim": "https://naaim.org/programs/naaim-exposure-index/",
    "cftc": "https://publicreporting.cftc.gov/stories/s/r4w3-av2u",
    "btos": "https://www.census.gov/hfp/btos",
    "openrouter": "https://openrouter.ai/rankings",
    "gpuperhour": "https://gpuperhour.com/gpu-price-index",
    "uvik": "https://uvik.net/blog/gpu-price-index/",
    "wikimedia": "https://pageviews.wmcloud.org/?project=en.wikipedia.org&pages={series}",
    "gdelt": "https://api.gdeltproject.org/api/v2/doc/doc",
    "eia": "https://www.eia.gov/electricity/data/browser/",
    "treasury": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates",
}
SOURCE_NAMES = {
    "cboe": "Cboe", "finra": "FINRA", "factset": "FactSet", "shiller": "Shiller", "tsmc": "TSMC IR", "mops": "TWSE MOPS",
    "aaii": "AAII", "naaim": "NAAIM", "cftc": "CFTC", "btos": "Census BTOS", "openrouter": "OpenRouter",
    "gpuperhour": "GPUPerHour", "uvik": "Uvik", "wikimedia": "Wikipedia pageviews", "gdelt": "GDELT", "eia": "EIA",
    "treasury": "US Treasury",
}
PANEL_SOURCES = {   # panel -> (label, url)
    "net_highs_lows": ("Yahoo prices, SPY members (SSGA)", "https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy"),
    "mcclellan_summation": ("Yahoo prices and volume, SPY members (SSGA)", "https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy"),
    "pct_above_ma": ("Yahoo prices, SPY members (SSGA)", "https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy"),
    "cftc_net_pct": ("CFTC Traders in Financial Futures (ES, NQ)", "https://publicreporting.cftc.gov/stories/s/r4w3-av2u"),
    "top10_weight": ("SSGA SPY holdings", "https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy"),
    "basket_trend": ("Yahoo prices, AI basket", "https://finance.yahoo.com/quote/SMH"),
    "basket_relative_strength": ("Yahoo prices: SMH, SPY, RSP, AI basket", "https://finance.yahoo.com/quote/SMH"),
    "basket_breadth": ("Yahoo prices, AI basket", "https://finance.yahoo.com/quote/SMH"),
    "basket_drawdown": ("Yahoo prices, AI basket", "https://finance.yahoo.com/quote/SMH"),
    "basket_correlation": ("Yahoo prices, AI basket", "https://finance.yahoo.com/quote/SMH"),
    "basket_index_weight": ("SSGA SPY holdings", "https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy"),
    "basket_forward_pe": ("Yahoo quote snapshots, AI basket", "https://finance.yahoo.com/quote/NVDA"),
    "leveraged_etf_growth": ("Yahoo quote snapshots: SOXL, NVDL", "https://finance.yahoo.com/quote/SOXL"),
    "credit_proxy": ("Yahoo prices: ORCL, CRWV", "https://finance.yahoo.com/quote/ORCL"),
    "atm_iv": ("Yahoo option chains (daily snapshots)", "https://finance.yahoo.com/quote/NVDA/options"),
    "iv_spread": ("Yahoo option chains + Cboe VIX", "https://finance.yahoo.com/quote/NVDA/options"),
    "risk_reversal": ("Yahoo option chains: NVDA, SMH", "https://finance.yahoo.com/quote/NVDA/options"),
    "call_put_volume": ("Yahoo option chains, AI basket", "https://finance.yahoo.com/quote/NVDA/options"),
    "beat_fade": ("Finnhub EPS surprises + EDGAR 8-K 2.02 dates + Yahoo prices", "https://www.sec.gov/edgar/search/"),
    "xbrl_aggregate": ("SEC EDGAR XBRL company facts", "https://www.sec.gov/edgar/search/"),
    "token_growth": ("OpenRouter rankings (CC BY 4.0)", "https://openrouter.ai/rankings"),
    "token_price_index": ("OpenRouter model catalog", "https://openrouter.ai/models"),
    "revenue_proxy": ("OpenRouter rankings and model catalog", "https://openrouter.ai/rankings"),
}


def _clean(v):
    """JSON-safe scalar: NaN -> None, numpy -> python, dates -> ISO."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) or math.isinf(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.date().isoformat()
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


def _round_list(values, nd=1) -> list:
    return [None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd) for v in values]


def next_refresh(now: datetime) -> datetime:
    t = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for _ in range(24 * 8):
        t += timedelta(hours=1)
        if t.hour == REFRESH_UTC_HOUR and t.weekday() in REFRESH_WEEKDAYS:
            return t
    raise RuntimeError("no refresh slot within a week")


def drawdown_periods(price: pd.Series, threshold: float = -10.0) -> list[list[str]]:
    """[peak date, recovery date] for every decline whose trough reached `threshold` percent."""
    p = price.dropna()
    out, peak_val, peak_date, trough = [], -np.inf, None, 0.0
    for d, v in p.items():
        if v >= peak_val:
            if peak_date is not None and trough <= threshold:
                out.append([peak_date.date().isoformat(), d.date().isoformat()])
            peak_val, peak_date, trough = v, d, 0.0
        else:
            trough = min(trough, (v / peak_val - 1) * 100)
    if peak_date is not None and trough <= threshold:
        out.append([peak_date.date().isoformat(), p.index[-1].date().isoformat()])
    return out


def source_of(ind: Indicator) -> tuple[str, str | None]:
    if ind.kind == "panel":
        return PANEL_SOURCES.get(ind.panel, (ind.panel, None))
    if ind.kind == "event":
        return ("events.yaml (manual, sourced)", None)
    if ind.kind == "none":
        return ("not available on a free source", None)
    labels, url = [], None
    for inp in ind.inputs:
        src, series = inp.source, inp.series
        if src in ("fred", "alfred"):
            labels.append(f"FRED {series}" + (" (ALFRED vintages)" if src == "alfred" else ""))
            url = url or f"https://fred.stlouisfed.org/series/{series}"
        elif src == "yfinance":
            ticker = series.rsplit(".", 1)[0]
            labels.append(f"Yahoo {ticker}")
            url = url or SOURCE_PAGES["yfinance"].format(ticker=ticker)
        else:
            labels.append(f"{SOURCE_NAMES.get(src, src)} {series}")
            page = SOURCE_PAGES.get(src)
            url = url or (page.format(series=series.split(".")[-1]) if page else None)
    return " + ".join(labels), url


def badges(ind: Indicator, row: pd.Series, flagged: set[str]) -> list[str]:
    out = []
    if ind.kind == "event":
        out.append("manual")
    elif ind.inputs and all(i.ref in VERIFIED_REFS and i.ref not in flagged for i in ind.inputs):
        out.append("verified")
    else:
        out.append("single-source")
    if row is not None and not pd.isna(row.get("stale", np.nan)) and row["stale"] < 1:
        out.append("stale")
    if row is not None and bool(row.get("short_history", False)):
        out.append("short-history")
    return out


def build(store: Store) -> tuple[dict, dict]:
    reg = load_registry()
    events, event_errors = load_events()
    con = store.con
    summary = json.loads(con.execute("SELECT summary FROM score_latest").fetchone()[0])
    ind = con.execute("SELECT * FROM indicator_daily").df()
    ind["date"] = pd.to_datetime(ind["date"])
    gauges = con.execute("SELECT * FROM gauge_daily ORDER BY date").df()
    gauges["date"] = pd.to_datetime(gauges["date"])
    overlay = con.execute("SELECT * FROM overlay_daily ORDER BY date").df()
    overlay["date"] = pd.to_datetime(overlay["date"])
    as_of = pd.Timestamp(summary["as_of"])
    now = datetime.now(timezone.utc)
    features = Features(store, load_series_config(), load_baskets(), events)

    # Sources: the latest run each source appeared in.
    log = con.execute(
        "SELECT * FROM fetch_log QUALIFY row_number() OVER (PARTITION BY source ORDER BY finished_at DESC) = 1 "
        "ORDER BY source").df()
    sources = [{"source": r["source"], "status": r["status"], "latest_obs": _clean(r["latest_obs"]),
                "fetched_at": _clean(pd.Timestamp(r["finished_at"]).tz_localize("UTC").isoformat()),
                "message": (r["message"] or "")[:300]} for _, r in log.iterrows()]
    last_fetch = log["finished_at"].max()
    run_id = con.execute("SELECT max(run_id) FROM fetch_log").fetchone()[0]
    flagged = {f"{s}:{k}" for s, k in con.execute(
        "SELECT source, series FROM validation_flags WHERE run_id = ? AND check_name = 'crosscheck'",
        [run_id]).fetchall() if s}

    # Components per gauge.
    latest_rows = ind[ind["date"] == as_of].set_index("id")
    by_id = {k: g.set_index("date") for k, g in ind.groupby("id")}
    gauge_out = {}
    for gid, gspec in reg.gauges.items():
        s = summary["gauges"][gid]
        hist = gauges[gauges["gauge"] == gid].set_index("date")["display"].dropna()
        ghosts = {}
        for label, n in (("prev_close", 1), ("1w", 5), ("1m", 21), ("1y", 252)):
            ghosts[label] = round(float(hist.iloc[-1 - n]), 1) if len(hist) > n else None
        target_total = sum(gspec.pillars.values())
        components = []
        for i in [x for x in reg.indicators if x.gauge == gid]:
            members = [x for x in reg.indicators if x.gauge == gid and x.pillar == i.pillar and x.active]
            target = (gspec.pillars[i.pillar] * i.weight / sum(m.weight for m in members) / target_total * 100
                      if i.active and members else 0.0)
            row = latest_rows.loc[i.id] if i.id in latest_rows.index else None
            series = by_id.get(i.id)
            spark, ch1w, ch1m = [], None, None
            if series is not None:
                sc = series["score"].dropna()
                spark = _round_list(sc.iloc[-SPARK_DAYS::SPARK_STEP].tolist())
                ch1w = round(float(sc.iloc[-1] - sc.iloc[-6]), 1) if len(sc) > 5 else None
                ch1m = round(float(sc.iloc[-1] - sc.iloc[-22]), 1) if len(sc) > 21 else None
            label, url = source_of(i)
            mult = float(row["mult"]) if row is not None and not pd.isna(row["mult"]) else 0.0
            components.append({
                "id": i.id, "name": i.name, "pillar": i.pillar, "role": i.role, "direction": i.direction,
                "kind": i.kind, "active": i.active, "cnn": i.cnn, "fragility": i.fragility,
                "value": _clean(row["value"]) if row is not None else None,
                "pct": _clean(row["pct"]) if row is not None else None,
                "score": _clean(row["score"]) if row is not None else None,
                "as_of": _clean(row["as_of"]) if row is not None else None,
                "published": _clean(row["avail"]) if row is not None else None,
                "target_weight": round(target, 2), "live_weight": round(target * mult, 2),
                "contribution": s.get("contributions", {}).get(i.id),
                "change_1w": ch1w, "change_1m": ch1m, "spark": spark,
                "source": label, "source_url": url, "badges": badges(i, row, flagged) if i.kind != "none" else ["unavailable"],
                "notes": i.notes, "validation": i.validation,
            })
        gauge_out[gid] = {**{k: s[k] for k in ("name", "score", "display", "band", "coverage", "low_coverage",
                                                "pillars", "change_1w", "change_1m", "drivers_1w", "drivers_1m")},
                          "as_of": summary["as_of"], "scale": gspec.scale, "smoothing": gspec.smoothing,
                          "pillar_weights": gspec.pillars, "ghosts": ghosts, "components": components}
        if gid == "G2":
            gauge_out[gid]["infl_penalty"] = s.get("infl_penalty")

    # Watch list: latest entry per kpi/entity/metric/basis.
    watch = {}
    for e in sorted(events, key=lambda e: e.as_of):
        watch[(e.kpi, e.entity, e.metric, e.basis)] = e
    watchlist = [{"kpi": e.kpi, "entity": e.entity, "metric": e.metric, "value": e.value, "period": e.period,
                  "as_of": e.as_of.isoformat(), "basis": e.basis, "status": e.status, "source_url": str(e.source_url),
                  "note": e.note} for e in sorted(watch.values(), key=lambda e: e.as_of, reverse=True)]

    # Catalysts: tracked earnings (Finnhub) and macro dates for the next 30 days.
    today = now.astimezone(ET).date()
    horizon = today + timedelta(days=CATALYST_DAYS)
    earn = con.execute(
        "SELECT ticker, report_date, hour, fiscal_period, eps_estimate FROM earnings_events WHERE source = "
        "'finnhub_calendar' AND report_date BETWEEN ? AND ? QUALIFY row_number() OVER (PARTITION BY ticker, "
        "report_date ORDER BY fetched_at DESC) = 1 ORDER BY report_date, ticker", [today, horizon]).df()
    catalysts = [{"date": _clean(r["report_date"]), "label": f"{r['ticker']} earnings", "kind": "earnings",
                  "detail": " ".join(str(x) for x in (_clean(r["hour"]), _clean(r["fiscal_period"])) if x),
                  "source_url": "https://finnhub.io/docs/api/earnings-calendar"} for _, r in earn.iterrows()]
    catalysts += [{"date": c.date.isoformat(), "label": c.event, "kind": c.kind, "detail": "",
                   "source_url": str(c.source_url)} for c in load_calendar() if today <= c.date <= horizon]
    catalysts.sort(key=lambda c: c["date"])

    # What changed: largest week-over-week score moves among active indicators.
    movers = []
    for g in gauge_out.values():
        for c in g["components"]:
            if c["active"] and c["change_1w"] is not None:
                movers.append({"id": c["id"], "name": c["name"], "gauge": next(k for k, v in gauge_out.items() if v is g),
                               "change_1w": c["change_1w"], "score": c["score"]})
    movers.sort(key=lambda m: abs(m["change_1w"]), reverse=True)

    stale = [c for g in gauge_out.values() for c in g["components"] if "stale" in c["badges"] and c["active"]]
    latest = {
        "generated_at": now.isoformat(timespec="seconds"),
        "as_of": summary["as_of"],
        "last_fetch_utc": _clean(pd.Timestamp(last_fetch).tz_localize("UTC").isoformat()) if not pd.isna(last_fetch) else None,
        "next_refresh_utc": next_refresh(now).isoformat(),
        "global_coverage": round(float(np.mean([g["coverage"] for g in gauge_out.values()])), 1),
        "sources": sources,
        "source_issues": [s for s in sources if s["status"] != "ok"],
        "stale_indicators": [{"id": c["id"], "as_of": c["as_of"]} for c in stale],
        "gauges": gauge_out,
        "fragility": {**summary["fragility"], "members": [i.id for i in reg.indicators if i.fragility],
                      "divergence_rule": reg.fragility.divergence},
        "cnn_replica": summary["cnn_replica"],
        "cnn_rows": [i.id for i in reg.indicators if i.cnn],
        "banners": summary["banners"],
        "ai_fragility": summary["ai_fragility"],
        "unwind": summary["unwind"],
        "open_weight_token_share": summary.get("open_weight_token_share"),
        "watchlist": watchlist,
        "events_rejected": event_errors,
        "catalysts": catalysts,
        "movers": movers[:8],
        "gaps": summary["gaps"],
        "methodology": methodology(reg),
        "valuation": valuation_panel(features),
        "ai_companies": ai_companies(features),
        "attribution": [
            "Source: OpenRouter (openrouter.ai/rankings), as of " + (sources_latest(sources, "openrouter") or "n/a")
            + ". Licensed under CC BY 4.0.",
            "GPU Price Index: Uvik Software (CC BY 4.0).",
            "FRED and ALFRED: Federal Reserve Bank of St. Louis. ICE BofA indices on FRED are licensed by ICE.",
            "Cboe, CFTC, SEC EDGAR, Census BTOS, FINRA, NAAIM, AAII, FactSet Earnings Insight, Shiller (Yale), "
            "TSMC IR, TWSE MOPS, Wikimedia, GDELT, EIA, US Treasury, SSGA. Yahoo Finance via yfinance is unofficial.",
        ],
    }
    history = build_history(store, gauges, overlay)
    return latest, history


VALUATION_WINDOWS = (5, 10, 20)


def stretch(s: pd.Series, label: str, unit: str, source: str, url: str, expensive_when: str = "high",
            extra: dict | None = None) -> dict | None:
    """Current value vs its 5/10/20-year and full-history averages: mean, z-score (SDs) and percentile."""
    s = s.dropna().sort_index()
    if len(s) < 10:
        return None
    last_date, value = s.index[-1], float(s.iloc[-1])
    out = {"label": label, "unit": unit, "value": round(value, 2), "as_of": last_date.date().isoformat(),
           "since": s.index[0].date().isoformat(), "source": source, "source_url": url,
           "expensive_when": expensive_when, "windows": {}}
    for years in (*VALUATION_WINDOWS, None):
        w = s if years is None else s[s.index > last_date - pd.DateOffset(years=years)]
        span = (w.index[-1] - w.index[0]).days / 365.25
        if len(w) < 8 or (years is not None and span < years * 0.8) or w.std() == 0:
            continue
        out["windows"]["all" if years is None else f"{years}y"] = {
            "mean": round(float(w.mean()), 2), "sd": round(float(w.std()), 2),
            "z": round(float((value - w.mean()) / w.std()), 2),
            "pct": round(float(((w < value).sum() + 0.5 * (w == value).sum()) / len(w) * 100), 1)}
    out.update(extra or {})
    return out


def valuation_panel(F: Features) -> list[dict]:
    """S&P 500 valuation vs history, in standard deviations (display only: the backtest keeps these out of G2)."""
    factset = "https://insight.factset.com/topic/earnings"
    pe = F.obs("factset", "fwd_pe")
    fs_avgs = {"factset_5y_avg": _last(F.obs("factset", "fwd_pe_5y_avg")),
               "factset_10y_avg": _last(F.obs("factset", "fwd_pe_10y_avg"))}
    price, earnings, dividend = (F.obs("shiller", k) for k in ("price", "earnings", "dividend"))
    real10 = F.obs("fred", "DFII10")
    erp = (100 / pe - real10.reindex(pe.index.union(real10.index)).ffill().reindex(pe.index)).dropna()
    eq = (F.obs("alfred", "NCBEILQ027S") / F.obs("alfred", "GDP") * 0.1).dropna()
    top10 = F.panel_top10_weight()["value"].dropna()
    top10 = top10[~top10.index.duplicated()]
    rows = [
        stretch(pe, "Forward P/E", "x", "FactSet Earnings Insight (weekly, from 2017)", factset, extra=fs_avgs),
        stretch(F.obs("shiller", "cape"), "Shiller CAPE", "x", "Shiller (shillerdata.com), monthly from 1881",
                "https://shillerdata.com/"),
        stretch((price / earnings).dropna(), "Trailing P/E", "x", "Shiller price / 12-month earnings",
                "https://shillerdata.com/"),
        stretch((dividend / price * 100).dropna(), "Dividend yield", "%", "Shiller dividends / price",
                "https://shillerdata.com/", expensive_when="low"),
        stretch(erp, "Equity risk premium (forward earnings yield − 10y real)", "pts", "FactSet + FRED DFII10",
                "https://fred.stlouisfed.org/series/DFII10", expensive_when="low"),
        stretch(eq, "Corporate equities / GDP", "%", "FRED NCBEILQ027S / GDP (latest vintage)",
                "https://fred.stlouisfed.org/series/NCBEILQ027S"),
        stretch(top10, "Top-10 weight in the S&P 500", "%", "SSGA SPY holdings (daily, from first run)",
                "https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy"),
    ]
    return [r for r in rows if r]


def _last(s: pd.Series):
    s = s.dropna()
    return None if s.empty else round(float(s.iloc[-1]), 2)


def ai_companies(F: Features) -> list[dict]:
    """Per AI-basket company: recent returns, latest earnings release, guidance, surprise and revenue growth."""
    closes = F.basket()["closes"]
    layer_of = {t: name for name, layer in F.baskets.layers.items() for t in layer.tickers}
    events = {m: F.release_events(m) for m in ("quote_positivity", "release_tone", "ai_mentions", "rev_guide")}
    guide_growth = F.guidance_growth_events()
    surprises = F.surprise_events()
    upcoming = F.con.execute(
        "SELECT ticker, min(report_date) FROM earnings_events WHERE source = 'finnhub_calendar' AND report_date >= "
        "current_date GROUP BY 1").fetchall()
    next_report = {t: d.isoformat() for t, d in upcoming}
    rows = []
    for ticker in F.baskets.tickers:
        px = closes[ticker].dropna() if ticker in closes else pd.Series(dtype=float)
        if px.empty:
            continue
        last = px.index[-1]
        ret = lambda n: round(float(px.iloc[-1] / px.iloc[-1 - n] - 1) * 100, 1) if len(px) > n else None  # noqa: E731
        ytd_base = px[px.index < pd.Timestamp(last.year, 1, 1)]
        row = {"ticker": ticker, "layer": layer_of.get(ticker), "as_of": last.date().isoformat(),
               "ret_1w": ret(5), "ret_1m": ret(21), "ret_3m": ret(63),
               "ret_ytd": round(float(px.iloc[-1] / ytd_base.iloc[-1] - 1) * 100, 1) if len(ytd_base) else None,
               "drawdown": round(float(px.iloc[-1] / px.iloc[-252:].max() - 1) * 100, 1),
               "next_report": next_report.get(ticker)}
        for metric, ev in events.items():
            mine = ev[ev["ticker"] == ticker] if not ev.empty else ev
            if not mine.empty:
                row[metric] = round(float(mine["value"].iloc[-1]), 1)
                row["release_date"] = max(row.get("release_date", ""), (mine["date"].iloc[-1] - pd.Timedelta(days=1)).date().isoformat())
        gg = guide_growth[guide_growth["ticker"] == ticker]
        row["guide_growth"] = round(float(gg["value"].iloc[-1]), 1) if not gg.empty else None
        sp = surprises[surprises["ticker"] == ticker].sort_values("date")
        row["eps_surprise"] = round(float(sp["value"].iloc[-1]), 1) if not sp.empty else None
        rev = F.company_revenue(ticker) if "." not in ticker else pd.DataFrame()
        if len(rev) >= 8:
            ttm = rev["value"].rolling(4).sum()
            row["revenue_ttm_yoy"] = round(float(ttm.iloc[-1] / ttm.iloc[-5] - 1) * 100, 1) if ttm.iloc[-5] else None
        rows.append(row)
    return rows


def sources_latest(sources: list[dict], name: str) -> str | None:
    return next((s["latest_obs"] for s in sources if s["source"] == name), None)


def methodology(reg: Registry) -> dict:
    return {
        "rules": {
            "percentile": "Each indicator is ranked against its own trailing history (5 years for daily series, 20 "
                          "years for slower ones, all available history for AI series) using only data public on "
                          "each day; revised macro series are ranked within each data vintage.",
            "orientation": "Ranks are oriented so 100 means greed (G1, G3) or healthy (G2, G4).",
            "staleness": "An input's weight is full until it is 1.5x its expected age since publication, then "
                         "falls linearly to zero at 3x.",
            "events": "Manual entries carry status weights: primary 1.0, guidance 0.8, secondhand 0.6, estimate 0.3.",
            "composite": "Pillar = weighted mean of live indicators; gauge = weighted mean of live pillars, "
                         "renormalized for missing weight. Coverage is the live share of target weight; below 70% "
                         "the needle is hatched.",
            "display": "G1-G3 needles use a 3-day EMA; G4 is unsmoothed.",
            "validation": "Phase 3 backtest demoted indicators whose sign was wrong or unstable across stress "
                          "regimes; see reports/validation.md.",
        },
        "gauges": {g: {"name": s.name, "pillars": s.pillars, "scale": s.scale} for g, s in reg.gauges.items()},
        "indicators": [{
            "id": i.id, "gauge": i.gauge, "pillar": i.pillar, "name": i.name, "kind": i.kind, "role": i.role,
            "direction": i.direction, "weight": i.weight, "expected_freq": i.expected_freq,
            "lookback": i.lookback_pct or ("5y" if i.expected_freq == "1d" else "20y"),
            "inputs": [x.ref + (f" ({x.op})" if x.op != "identity" else "") for x in i.inputs],
            "transform": i.panel or i.transform, "fragility": i.fragility, "cnn": i.cnn, "notes": i.notes,
            "validation": i.validation} for i in reg.indicators],
    }


def build_history(store: Store, gauges: pd.DataFrame, overlay: pd.DataFrame) -> dict:
    dates = pd.DatetimeIndex(sorted(set(gauges["date"])))
    prices = {}
    for key, series in (("spx", "^GSPC.close"), ("smh", "SMH.close")):
        rows = store.con.execute(
            "SELECT obs_date, arg_max(value, fetched_at) FROM raw_observations WHERE source = 'yfinance' AND "
            "series = ? GROUP BY 1 ORDER BY 1", [series]).df()
        prices[key] = pd.Series(rows.iloc[:, 1].to_numpy(), index=pd.to_datetime(rows["obs_date"])).reindex(dates).ffill()
    out = {"dates": [d.date().isoformat() for d in dates], "gauges": {}}
    for gid, g in gauges.groupby("gauge"):
        g = g.set_index("date").reindex(dates)
        out["gauges"][gid] = {"display": _round_list(g["display"]), "score": _round_list(g["score"]),
                              "coverage": _round_list(g["coverage"], 0)}
    ov = overlay.set_index("date").reindex(dates)
    out["fragility"] = _round_list(ov["fragility"])
    out["cnn_replica"] = _round_list(ov["cnn_replica"])
    out["prices"] = {k: _round_list(v, 2) for k, v in prices.items()}
    out["drawdowns"] = {k: drawdown_periods(v) for k, v in prices.items()}
    return out


def write(latest: dict, history: dict, out_dir: Path = DATA_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(latest, default=_clean, separators=(",", ":")), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, default=_clean, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write data/latest.json and data/history.json")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    store = Store(args.db)
    try:
        latest, history = build(store)
        write(latest, history)
    finally:
        store.close()
    sizes = {p.name: f"{p.stat().st_size / 1024:.0f} KB" for p in (DATA_DIR / "latest.json", DATA_DIR / "history.json")}
    print(f"Wrote data/latest.json and data/history.json {sizes}")


if __name__ == "__main__":
    main()
