"""Yahoo Finance via yfinance: EOD prices, quote snapshots and option chains.

Unofficial and rate-limited: one batched price pull per run, small sleeps
between chain requests, and graceful degradation (failures become notes).
Partial bars for an unfinished session are dropped; point-in-time snapshots
(quotes, option chains) are only taken when no session is in progress.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd

from gauges.schema import OBS_COLUMNS
from gauges.sources import ssga
from gauges.sources.base import Context, FetchResult

SOURCE = "yfinance"
CHUNK = 80
REFETCH_DAYS = 7
FIELDS = {"Close": "close", "Volume": "volume", "Dividends": "dividend", "Stock Splits": "split"}

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def frame_to_obs(df: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """yf.download output with (field, ticker) columns -> observation rows."""
    parts = []
    fields = set(df.columns.get_level_values(0))
    for field, suffix in FIELDS.items():
        if field not in fields:
            continue
        long = (df[field].rename_axis("obs_date").reset_index()
                .melt(id_vars="obs_date", var_name="ticker", value_name="value").dropna(subset=["value"]))
        if suffix != "close":
            long = long[long["value"] != 0]
        long["series"] = long["ticker"].astype(str) + "." + suffix
        parts.append(long)
    if not parts:
        return pd.DataFrame(columns=OBS_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    out["obs_date"] = pd.to_datetime(out["obs_date"])
    if out["obs_date"].dt.tz is not None:
        out["obs_date"] = out["obs_date"].dt.tz_localize(None)
    out = out[out["obs_date"].dt.date <= cutoff]
    return out.assign(source=SOURCE, vintage=None, endpoint="yfinance:download")[OBS_COLUMNS]


def _download(symbols: list[str], start: date, end: date, notes: list[str]) -> list[pd.DataFrame]:
    import yfinance as yf

    frames = []
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        try:
            df = yf.download(chunk, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                             interval="1d", auto_adjust=False, actions=True, progress=False, threads=True,
                             group_by="column", multi_level_index=True)
        except Exception as e:  # noqa: BLE001
            notes.append(f"download {chunk[0]}..{chunk[-1]} failed: {type(e).__name__}: {e}")
            continue
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(1)
    return frames


def _prices(ctx: Context, symbols: list[str], new_start: date, notes: list[str]) -> pd.DataFrame:
    """New symbols get full history from new_start; known ones re-read a short window."""
    last = {s: ctx.last_date(SOURCE, f"{s}.close") for s in symbols}
    new = [s for s, d in last.items() if d is None]
    known = [s for s, d in last.items() if d is not None]
    frames = []
    if new:
        frames += _download(new, new_start, ctx.session, notes)
    if known:
        frames += _download(known, min(last[s] for s in known) - timedelta(days=REFETCH_DAYS), ctx.session, notes)
    if not frames:
        notes.append(f"no price data for any of {len(symbols)} symbols")
        return pd.DataFrame(columns=OBS_COLUMNS)
    obs = pd.concat([frame_to_obs(f, ctx.session) for f in frames], ignore_index=True)
    got = set(obs.loc[obs["series"].str.endswith(".close"), "series"].str[:-6])
    missing = sorted(set(symbols) - got)
    if missing:
        notes.append(f"no price data for {len(missing)}: {', '.join(missing[:15])}{' ...' if len(missing) > 15 else ''}")
    return obs


def _quote_snapshots(symbols: list[str], session: date) -> tuple[list[tuple], list[str]]:
    import yfinance as yf

    rows, notes = [], []
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
        except Exception as e:  # noqa: BLE001
            notes.append(f"{sym} info: {type(e).__name__}")
            continue
        # ETFs (SOXL, NVDL) report no share count; totalAssets / navPrice gives it.
        for key, suffix in (("forwardPE", "forward_pe"), ("marketCap", "market_cap"),
                            ("sharesOutstanding", "shares_out"), ("totalAssets", "total_assets"),
                            ("navPrice", "nav")):
            if isinstance(info.get(key), (int, float)):
                rows.append((SOURCE, f"{sym}.{suffix}", session, float(info[key]), None, "yfinance:Ticker.info"))
        time.sleep(0.4)
    return rows, notes


def _chain(sym: str, session: date, max_days: int, underlying_price: float | None) -> pd.DataFrame:
    import yfinance as yf

    tk = yf.Ticker(sym)
    expiries = [e for e in tk.options if 0 <= (date.fromisoformat(e) - session).days <= max_days]
    frames = []
    for e in expiries:
        chain = tk.option_chain(e)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            if df.empty:
                continue
            frames.append(pd.DataFrame({
                "snapshot_date": session, "underlying": sym, "expiry": date.fromisoformat(e), "option_type": kind,
                "strike": df["strike"], "bid": df.get("bid"), "ask": df.get("ask"), "last": df.get("lastPrice"),
                "volume": df.get("volume"), "open_interest": df.get("openInterest"),
                "implied_vol": df.get("impliedVolatility"), "underlying_price": underlying_price}))
        time.sleep(0.3)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def option_volume_obs(chain: pd.DataFrame, sym: str, session: date, max_days: int) -> list[tuple]:
    near = chain[(pd.to_datetime(chain["expiry"]) - pd.Timestamp(session)).dt.days <= max_days]
    rows = []
    for kind in ("call", "put"):
        total = float(near.loc[near["option_type"] == kind, "volume"].fillna(0).sum())
        rows.append((SOURCE, f"{sym}.opt_{kind}_volume", session, total, None, "yfinance:Ticker.option_chain"))
    return rows


def fetch(ctx: Context) -> FetchResult:
    cfg, notes = ctx.cfg.fetch.yfinance, []
    basket = ctx.baskets.tickers
    base = list(dict.fromkeys(cfg.symbols + basket + ctx.baskets.reference_etfs))
    try:
        members = [s for s in ssga.constituents(ctx) if s not in base]
    except Exception as e:  # noqa: BLE001
        members = []
        notes.append(f"SPY constituents unavailable ({type(e).__name__}); breadth universe skipped")

    obs = _prices(ctx, base, cfg.history_start, notes)
    if members:
        years_back = ctx.today.replace(year=ctx.today.year - cfg.constituent_years)
        obs = pd.concat([obs, _prices(ctx, members, years_back, notes)], ignore_index=True)

    result = FetchResult(obs, notes=notes)
    spy = obs[obs["series"] == "SPY.close"]
    if spy.empty:
        notes.append("SPY close missing: snapshots skipped")
        return result
    session = spy["obs_date"].max().date()
    closes = {s[:-6]: v for s, v in obs[(obs["obs_date"].dt.date == session)
                                        & obs["series"].str.endswith(".close")][["series", "value"]].values}

    if not ctx.snapshots_allowed:
        notes.append("session in progress: quote and option snapshots deferred")
        return result

    extra: list[tuple] = []
    quote_syms = [s for s in dict.fromkeys(basket + cfg.info_symbols)
                  if session not in (ctx.last.get((SOURCE, f"{s}.market_cap")), ctx.last.get((SOURCE, f"{s}.nav")))]
    rows, quote_notes = _quote_snapshots(quote_syms, session)
    extra += rows
    notes += quote_notes

    have = ctx.snapshot_keys.get("option_snapshots", set())
    volume_syms = [s for s in basket if "." not in s and ctx.last.get((SOURCE, f"{s}.opt_call_volume")) != session]
    for sym in dict.fromkeys(cfg.option_chains + volume_syms):
        full_chain = sym in cfg.option_chains and (session, sym) not in have
        max_days = cfg.option_chain_max_days if full_chain else cfg.option_volume_max_days
        try:
            chain = _chain(sym, session, max_days, closes.get(sym))
        except Exception as e:  # noqa: BLE001
            notes.append(f"{sym} options: {type(e).__name__}: {e}")
            continue
        if chain.empty:
            notes.append(f"{sym} options: no expiries within {max_days}d")
            continue
        if full_chain:
            result.add_table("option_snapshots", chain)
        if sym in volume_syms:
            extra += option_volume_obs(chain, sym, session, cfg.option_volume_max_days)

    if extra:
        extra_df = pd.DataFrame(extra, columns=OBS_COLUMNS)
        result.observations = pd.concat([obs, extra_df], ignore_index=True)
    notes.append(f"session {session}; {len(quote_syms)} quote snapshots; "
                 f"{len(result.tables.get('option_snapshots', []))} option rows")
    return result
