"""Robustness study: what the gauges showed through bull and bear markets, crises
and named events, what followed peak fear and greed, and whether simple rules
built on the gauges survive stress tests.

Usage: python -m gauges.robustness [--paths 2000] [--reps 1000] [--seed 20261015]

Reads the scored history (run `make score` first; nothing is re-scored or
written to the store) and writes reports/robustness.md and data/robustness.json
(drawn by dashboard/backtest.html).

1. Market cycles: bull and bear markets (20% moves) and corrections (10%)
   found in S&P 500 closes, with the gauges' readings at each turn.
2. Peak fear and greed: runs of Extreme Fear (<= 20) and Extreme Greed (>= 80)
   in G1 (vs the S&P 500) and G3 (vs SMH), with returns after entry, the
   extreme and exit, against random entry dates; returns by band.
3. Named events (config/market_events.yaml): readings and the returns after.
4. Strategies: simple rules on the gauges, traded at the next close on SPY
   total return with idle cash at the fed funds rate.
5. Robustness: parameter grids, sub-periods, execution delay, costs, signal
   variants, leave-one-pillar-out, other assets and a circular-shift placebo.
6. Correlations of each composite with forward returns and drawdowns, with
   moving-block bootstrap confidence intervals.
7. Scenarios: stress-window replays, today's nearest historical analogs, and
   forward returns by G1 band x G2 band.
8. Monte Carlo: a stationary block bootstrap of strategy vs buy-and-hold
   returns, and a state-conditional block bootstrap of the next 12 months.

Readings are point in time (gauges.score), but the registry's Phase 3
demotions used this same history, and early readings have low coverage; both
are flagged. Event dates come from the public record; peaks and troughs are
checked against the closes. Nothing is fitted here: rules use the dashboard's
band edges, and the grids show how much results depend on that choice.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from gauges import transforms as T
from gauges.backtest import forward_max_drawdown, forward_return
from gauges.config import CONFIG_DIR, ROOT, load_baskets, load_events, load_series_config
from gauges.features import Features
from gauges.registry import load_registry
from gauges.store import DEFAULT_DB, Store

REPORT = ROOT / "reports" / "robustness.md"
OUT = ROOT / "data" / "robustness.json"
EVENTS_FILE = CONFIG_DIR / "market_events.yaml"

H = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
EDGES = np.array([20.0, 40.0, 60.0, 80.0])     # the dashboard's band edges
FEAR, GREED = 20.0, 80.0
GAP = 21              # zone visits closer than this many trading days form one episode
BEAR, CORRECTION = 20.0, 10.0
SNAP = 5              # a peak/trough date moves to the closing extreme within this many trading days
NEAR = 10             # trading days either side of a low to look for the gauge's low
LOOKBACK = 63         # trading days before a top to look for a warning
DELAY = 1             # trading days from signal to trade: signal at close t, trade at close t+1
RELIABLE = 50.0       # G1 coverage (%) from which history counts as reliable
YEAR = 252
ANCHORS = ("start", "extreme", "exit")
ASSET_LABEL = {"^GSPC": "S&P 500", "SMH": "SMH", "SPY": "SPY"}
PCTS = (5, 25, 50, 75, 95)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Data:
    cal: pd.DatetimeIndex
    display: pd.DataFrame        # gauge -> dashboard needle (3-day EMA where the gauge smooths)
    raw: pd.DataFrame            # gauge -> unsmoothed score
    coverage: pd.DataFrame
    overlay: pd.DataFrame        # fragility, cnn_replica
    pillars: dict[str, pd.DataFrame]
    weights: dict[str, dict[str, float]]
    smoothing: dict[str, str]
    scales: dict[str, list[str]]
    close: pd.DataFrame          # ^GSPC, SPY, RSP, IWM, SMH closes
    tr: pd.DataFrame             # daily total returns (dividends reinvested; ^GSPC is price only)
    cash: pd.Series              # daily return on cash at the fed funds rate
    vix: pd.Series

    @property
    def reliable(self) -> pd.Series:
        """Days from which G1 coverage stays at or above RELIABLE."""
        below = (self.coverage["G1"] < RELIABLE) | self.coverage["G1"].isna()
        last_bad = below[below].index.max()
        return pd.Series(self.cal > last_bad if pd.notna(last_bad) else True, index=self.cal)


def total_return(close: pd.Series, dividend: pd.Series) -> pd.Series:
    """Daily total return, dividends reinvested at the ex-date close (yfinance closes are split-adjusted only)."""
    return (close + dividend.fillna(0.0)) / close.shift(1) - 1


def cash_return(rate_pct: pd.Series) -> pd.Series:
    """Daily return on cash at an annual rate in percent, using the rate known the day before."""
    return (1 + rate_pct.shift(1) / 100) ** (1 / YEAR) - 1


def load(store: Store) -> Data:
    reg = load_registry()
    F = Features(store, load_series_config(), load_baskets(), load_events()[0])
    cal, con = F.calendar, store.con

    def frame(sql: str) -> pd.DataFrame:
        df = con.execute(sql).df()
        df["date"] = pd.to_datetime(df["date"])
        return df

    g = frame("SELECT date, gauge, score, display, coverage FROM gauge_daily")
    wide = {c: g.pivot(index="date", columns="gauge", values=c).reindex(cal) for c in ("score", "display", "coverage")}
    overlay = frame("SELECT date, fragility, cnn_replica FROM overlay_daily").set_index("date").reindex(cal)
    p = frame("SELECT date, gauge, pillar, score FROM pillar_daily")
    pillars = {gid: grp.pivot(index="date", columns="pillar", values="score").reindex(cal)
               for gid, grp in p.groupby("gauge")}
    symbols = ["^GSPC", "SPY", "RSP", "IWM", "SMH"]
    close = pd.DataFrame({s: F.obs("yfinance", f"{s}.close") for s in symbols}).reindex(cal).ffill(limit=5)
    div = pd.DataFrame({s: F.obs("yfinance", f"{s}.dividend") for s in symbols}).reindex(cal)
    tr = pd.DataFrame({s: total_return(close[s], div[s]) for s in symbols})
    dff = F.obs("fred", "DFF")
    rate = dff.reindex(dff.index.union(cal)).ffill().reindex(cal)
    vix = F.obs("cboe", "VIX").reindex(cal)
    return Data(cal, wide["display"], wide["score"], wide["coverage"], overlay, pillars,
                {gid: dict(s.pillars) for gid, s in reg.gauges.items()},
                {gid: s.smoothing for gid, s in reg.gauges.items()},
                {gid: list(s.scale) for gid, s in reg.gauges.items()},
                close, tr, cash_return(rate), vix)


# --------------------------------------------------------------------------- #
# Small helpers (pure)
# --------------------------------------------------------------------------- #
def band_index(v: float) -> int | None:
    return None if v is None or not np.isfinite(v) else int(np.searchsorted(EDGES, v, side="right"))


def band_of(v: float, scale: list[str]) -> str | None:
    b = band_index(v)
    return None if b is None else scale[b]


def bands(s: pd.Series) -> np.ndarray:
    """Band index per day (0-4), -1 where the value is missing."""
    x = s.to_numpy(dtype=float)
    out = np.searchsorted(EDGES, np.nan_to_num(x, nan=-1.0), side="right")
    return np.where(np.isfinite(x), out, -1)


def fwd_at(price: pd.Series, i: int, h: int) -> float:
    j = i + h
    if i < 0 or j >= len(price):
        return math.nan
    a, b = price.iloc[i], price.iloc[j]
    return float((b / a - 1) * 100) if np.isfinite(a) and np.isfinite(b) and a else math.nan


def first_at_or_after(cal: pd.DatetimeIndex, d) -> int:
    return int(cal.searchsorted(pd.Timestamp(d)))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def ranks(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


def pct_summary(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)), "hit": float((v > 0).mean() * 100)}


# --------------------------------------------------------------------------- #
# 1. Market cycles
# --------------------------------------------------------------------------- #
def turning_points(price: pd.Series, pct: float) -> list[tuple[pd.Timestamp, str]]:
    """Alternating peaks and troughs in closes. A peak is confirmed once the price falls `pct`% below it,
    a trough once it rises `pct`% above it (with pct=20, the usual bull/bear market rule)."""
    p = price.dropna()
    if p.empty:
        return []
    points: list[tuple[pd.Timestamp, str]] = []
    mode = None
    hi_d = lo_d = p.index[0]
    hi = lo = float(p.iloc[0])
    for d, v in p.items():
        if v > hi:
            hi, hi_d = v, d
        if v < lo:
            lo, lo_d = v, d
        if mode != "down" and v <= hi * (1 - pct / 100):
            points.append((hi_d, "peak"))
            mode, lo, lo_d = "down", v, d
        elif mode != "up" and v >= lo * (1 + pct / 100):
            points.append((lo_d, "trough"))
            mode, hi, hi_d = "up", v, d
    return points


def swings(price: pd.Series, pct: float) -> list[dict]:
    """Declines (peak -> trough) and advances (trough -> peak) between turning points.
    The last swing is still open: a decline ends at its low so far, an advance at the latest close."""
    p = price.dropna()
    pts = turning_points(p, pct)
    out = []
    for (d0, kind), nxt in zip(pts, pts[1:] + [None]):
        if nxt is not None:
            d1 = nxt[0]
        else:
            d1 = p.loc[d0:].idxmin() if kind == "peak" else p.index[-1]
        out.append({"kind": "decline" if kind == "peak" else "advance", "start": d0, "end": d1,
                    "change": float((p[d1] / p[d0] - 1) * 100),
                    "days": int(p.index.get_loc(d1) - p.index.get_loc(d0)), "ongoing": nxt is None})
    return out


def recovery_date(price: pd.Series, peak, trough):
    after = price.loc[trough:]
    hit = after[after >= price[peak]]
    return hit.index[0] if len(hit) else None


def overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def decline_rows(D: Data, declines: list[dict]) -> list[dict]:
    cal, spx = D.cal, D.close["^GSPC"]
    g1, g2, fr, cov = D.display["G1"], D.display["G2"], D.overlay["fragility"], D.coverage["G1"]
    rows = []
    for c in declines:
        i0, i1 = cal.get_loc(c["start"]), cal.get_loc(c["end"])
        pre = g1.iloc[max(0, i0 - LOOKBACK):i0 + 1]
        span = g1.iloc[i0:min(len(cal), i1 + NEAR + 1)]
        low_d = span.idxmin() if span.notna().any() else None
        during = g1.iloc[i0:i1 + 1]
        fear = during[during <= FEAR]
        first_fear = fear.index[0] if len(fear) else None
        row = {**c, "recovered": recovery_date(spx, c["start"], c["end"]), "g1_cov": cov.iloc[i0],
               "g1_peak": g1.iloc[i0], "g1_pre_max": pre.max(), "frag_peak": fr.iloc[i0],
               "frag_pre_max": fr.iloc[max(0, i0 - LOOKBACK):i0 + 1].max(),
               "g2_peak": g2.iloc[i0], "g2_trough": g2.iloc[i1], "g1_trough": g1.iloc[i1],
               "g1_low": span.min(), "g1_low_date": low_d,
               "g1_low_lead": int(i1 - cal.get_loc(low_d)) if low_d is not None else None,
               "first_fear": first_fear,
               "further_fall": float((spx[c["end"]] / spx[first_fear] - 1) * 100) if first_fear is not None else math.nan}
        for k in ("3m", "6m", "12m"):
            row[f"trough_{k}"] = fwd_at(spx, i1, H[k])
            row[f"fear_{k}"] = fwd_at(spx, cal.get_loc(first_fear), H[k]) if first_fear is not None else math.nan
        rows.append(row)
    return rows


def advance_rows(D: Data, advances: list[dict]) -> list[dict]:
    cal, spx = D.cal, D.close["^GSPC"]
    g1, fr, cov = D.display["G1"], D.overlay["fragility"], D.coverage["G1"]
    rows = []
    for c in advances:
        i0, i1 = cal.get_loc(c["start"]), cal.get_loc(c["end"])
        during = g1.iloc[i0:i1 + 1]
        known = during.dropna()
        greed = known[known >= GREED]
        first_greed = greed.index[0] if len(greed) else None
        rows.append({**c, "g1_cov": cov.iloc[i0], "g1_start": g1.iloc[i0], "g1_end": g1.iloc[i1],
                     "g1_max": known.max() if len(known) else math.nan,
                     "g1_max_date": known.idxmax() if len(known) else None,
                     "greed_share": float((known >= 60).mean() * 100) if len(known) else math.nan,
                     "extreme_greed_days": int(len(greed)), "first_greed": first_greed,
                     "gain_after_greed": float((spx[c["end"]] / spx[first_greed] - 1) * 100) if first_greed is not None else math.nan,
                     "frag_end": fr.iloc[i1]})
    return rows


def market_cycles(D: Data) -> dict:
    spx = D.close["^GSPC"]
    big = swings(spx, BEAR)
    bears = [s for s in big if s["kind"] == "decline"]
    bulls = [s for s in big if s["kind"] == "advance"]
    small = [s for s in swings(spx, CORRECTION) if s["kind"] == "decline"]
    corrections = [s for s in small if s["change"] > -BEAR and not any(overlaps(s, b) for b in bears)]
    return {"bears": decline_rows(D, bears), "corrections": decline_rows(D, corrections), "bulls": advance_rows(D, bulls)}


# --------------------------------------------------------------------------- #
# 2. Peak fear and greed
# --------------------------------------------------------------------------- #
def episodes(s: pd.Series, zone: str, level: float, gap: int = GAP) -> pd.DataFrame:
    """Runs of days in the zone (fear: s <= level; greed: s >= level), merged when visits are closer than `gap`
    trading days. Positions: start, end (last day in the zone), extreme, exit (first day out, -1 if still in)."""
    x = s.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        inside = (x <= level) if zone == "fear" else (x >= level)
    cols = ["start", "end", "extreme", "exit", "days", "extreme_value"]
    hits = np.flatnonzero(inside)
    if not len(hits):
        return pd.DataFrame(columns=cols)
    cut = np.flatnonzero(np.diff(hits) > gap)
    rows = []
    for a, b in zip(np.r_[hits[0], hits[cut + 1]], np.r_[hits[cut], hits[-1]]):
        seg = x[a:b + 1]
        k = int(a + (np.nanargmin(seg) if zone == "fear" else np.nanargmax(seg)))
        rows.append({"start": int(a), "end": int(b), "extreme": k, "exit": int(b + 1) if b + 1 < len(x) else -1,
                     "days": int(inside[a:b + 1].sum()), "extreme_value": float(x[k])})
    return pd.DataFrame(rows, columns=cols)


def random_entry_p(values: np.ndarray, picks: np.ndarray, eligible: np.ndarray, higher: bool,
                   rng: np.random.Generator, reps: int) -> tuple[float, float]:
    """Mean of `values` at `picks`, and the one-sided p-value against the means of as many random eligible days
    (higher=True: share of random means at or above the observed one)."""
    picks = picks[(picks >= 0) & np.isfinite(values[np.clip(picks, 0, None)])]
    pool = np.flatnonzero(eligible & np.isfinite(values))
    n = len(picks)
    if n < 3 or len(pool) < 10 * n:
        return (float(values[picks].mean()) if n else math.nan), math.nan
    obs = float(values[picks].mean())
    draws = values[rng.choice(pool, size=(reps, n), replace=True)].mean(axis=1)
    hits = (draws >= obs).sum() if higher else (draws <= obs).sum()
    return obs, float((hits + 1) / (reps + 1))


def zone_study(s: pd.Series, price: pd.Series, zone: str, level: float, rng: np.random.Generator, reps: int) -> dict:
    ep = episodes(s, zone, level)
    cal = s.index
    fwd = {k: forward_return(price, h).to_numpy() for k, h in H.items()}
    mdd = forward_max_drawdown(price, 63).to_numpy()
    eligible = s.notna().to_numpy()
    rows = []
    for e in ep.itertuples():
        row = {"start": cal[e.start], "end": cal[e.end], "extreme": cal[e.extreme],
               "exit": cal[e.exit] if e.exit >= 0 else None, "days": e.days, "start_value": float(s.iloc[e.start]),
               "extreme_value": e.extreme_value, "mdd3m": float(mdd[e.start])}
        for a in ANCHORS:
            i = getattr(e, a)
            for k in H:
                row[f"{a}_{k}"] = float(fwd[k][i]) if i >= 0 else math.nan
        rows.append(row)
    summary = []
    for a in ANCHORS:
        idx = ep[a].to_numpy(dtype=int) if len(ep) else np.array([], dtype=int)
        idx = idx[idx >= 0]
        for k in H:
            v = fwd[k][idx] if len(idx) else np.array([])
            base = fwd[k][eligible & np.isfinite(fwd[k])]
            _, p = random_entry_p(fwd[k], idx, eligible, zone == "fear", rng, reps)
            st = pct_summary(v)
            summary.append({"anchor": a, "horizon": k, **st, "base": float(base.mean()) if len(base) else math.nan,
                            "base_hit": float((base > 0).mean() * 100) if len(base) else math.nan,
                            "excess": st.get("mean", math.nan) - (float(base.mean()) if len(base) else math.nan),
                            "p": p})
    return {"zone": zone, "level": level, "episodes": rows, "summary": summary,
            "days_in_zone": int(np.nansum((s <= level) if zone == "fear" else (s >= level)))}


def episode_correlation(z: dict) -> dict:
    """Across every fear and greed episode: the gauge at entry vs the return that followed."""
    eps = z["fear"]["episodes"] + z["greed"]["episodes"]
    out = {}
    for k in ("3m", "12m"):
        pairs = np.array([(e["start_value"], e[f"start_{k}"]) for e in eps if np.isfinite(e[f"start_{k}"])])
        if len(pairs) >= 5:
            out[k] = {"n": len(pairs), "pearson": pearson(pairs[:, 0], pairs[:, 1]),
                      "spearman": pearson(ranks(pairs[:, 0]), ranks(pairs[:, 1]))}
    return out


def threshold_sensitivity(s: pd.Series, price: pd.Series, rng, reps) -> list[dict]:
    out = []
    for zone, levels in (("fear", (10, 15, 20, 25, 30)), ("greed", (70, 75, 80, 85, 90))):
        for lv in levels:
            st = zone_study(s, price, zone, lv, rng, reps)
            pick = {r["horizon"]: r for r in st["summary"] if r["anchor"] == "start"}
            out.append({"zone": zone, "level": lv, "episodes": len(st["episodes"]),
                        **{f"excess_{k}": pick[k].get("excess", math.nan) for k in ("3m", "12m")},
                        **{f"p_{k}": pick[k].get("p", math.nan) for k in ("3m", "12m")}})
    return out


def band_table(s: pd.Series, price: pd.Series, scale: list[str], mask: pd.Series | None = None) -> list[dict]:
    b = bands(s)
    m = np.ones(len(s), bool) if mask is None else mask.to_numpy()
    fwd = {k: forward_return(price, h).to_numpy() for k, h in H.items()}
    mdd = forward_max_drawdown(price, 63).to_numpy()
    rows = []
    for i, name in enumerate(scale):
        sel = (b == i) & m
        row = {"band": name, "range": f"{int(([0.0] + list(EDGES))[i])}–{int((list(EDGES) + [100.0])[i])}",
               "days": int(sel.sum())}
        for k in H:
            v = fwd[k][sel]
            v = v[np.isfinite(v)]
            row[k] = float(v.mean()) if len(v) else math.nan
        v12 = fwd["12m"][sel]
        v12 = v12[np.isfinite(v12)]
        row["hit12m"] = float((v12 > 0).mean() * 100) if len(v12) else math.nan
        dd = mdd[sel]
        dd = dd[np.isfinite(dd)]
        row["dd10"] = float((dd <= -10).mean() * 100) if len(dd) else math.nan
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# 3. Named events
# --------------------------------------------------------------------------- #
def load_market_events(path=EVENTS_FILE) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    out = []
    for e in raw:
        if e.get("kind") not in ("peak", "trough", "shock") or e.get("market", "equity") not in ("equity", "ai"):
            raise ValueError(f"market_events.yaml: bad kind/market in {e}")
        out.append({"date": pd.Timestamp(e["date"]), "name": str(e["name"]), "kind": e["kind"],
                    "market": e.get("market", "equity")})
    return sorted(out, key=lambda e: e["date"])


def event_row(D: Data, e: dict) -> dict | None:
    gauge, asset = ("G1", "^GSPC") if e["market"] == "equity" else ("G3", "SMH")
    cal, price, s = D.cal, D.close[asset], D.display[gauge]
    i = first_at_or_after(cal, e["date"])
    if i >= len(cal):
        return None
    moved_from = None
    if e["kind"] in ("peak", "trough"):
        w = price.iloc[max(0, i - SNAP):min(len(cal), i + SNAP + 1)].dropna()
        if len(w):
            d2 = w.idxmax() if e["kind"] == "peak" else w.idxmin()
            if d2 != cal[i]:
                moved_from, i = cal[i], cal.get_loc(d2)
    d = cal[i]
    reading = s.iloc[i]
    row = {**e, "trade_date": d, "moved_from": moved_from, "gauge": gauge, "asset": asset, "reading": reading,
           "band": band_of(reading, D.scales[gauge]), "coverage": D.coverage[gauge].iloc[i],
           "change_1m": reading - s.iloc[i - 21] if i >= 21 else math.nan,
           "readings": {g: D.display[g].iloc[i] for g in D.display.columns},
           "fragility": D.overlay["fragility"].iloc[i], "mdd3m": float(forward_max_drawdown(price.iloc[i:i + 64], 63).iloc[0])
           if i + 63 < len(cal) else math.nan}
    for k, h in H.items():
        row[f"ret_{k}"] = fwd_at(price, i, h)
    if not np.isfinite(reading):
        row.update(signal="no reading", hit=None)
        return row
    near = s.iloc[max(0, i - NEAR):i + NEAR + 1]
    after = s.iloc[i:i + 22]
    fear_after = after[after <= FEAR]
    row["fear_signal"] = fear_after.index[0] if len(fear_after) else None
    if row["fear_signal"] is not None:
        j = cal.get_loc(row["fear_signal"])
        row.update({f"signal_{k}": fwd_at(price, j, H[k]) for k in ("3m", "12m")})
    if e["kind"] == "trough":
        low = near.min()
        row["gauge_low"] = low
        row["hit"] = "yes" if low <= FEAR else "partial" if low <= EDGES[1] else "no"
        row["signal"] = f"{gauge} low {low:.0f} within {NEAR} days ({band_of(low, D.scales[gauge])})"
    elif e["kind"] == "peak":
        g_max = s.iloc[max(0, i - LOOKBACK):i + 1].max()
        f_max = D.overlay["fragility"].iloc[max(0, i - LOOKBACK):i + 1].max() if gauge == "G1" else math.nan
        row.update(gauge_high=g_max, frag_high=f_max)
        hit = g_max >= GREED or (np.isfinite(f_max) and f_max >= 80)
        partial = g_max >= EDGES[2] or (np.isfinite(f_max) and f_max >= 60)
        row["hit"] = "yes" if hit else "partial" if partial else "no"
        bits = [f"{gauge} high {g_max:.0f}"] + ([f"fragility high {f_max:.0f}"] if np.isfinite(f_max) else [])
        row["signal"] = ", ".join(bits) + f" in the {LOOKBACK} days before"
    else:
        low = after.min()
        row["gauge_low"] = low
        row["hit"] = None
        row["signal"] = (f"Extreme Fear on {row['fear_signal'].date()}" if row["fear_signal"] is not None
                         else f"{gauge} low {low:.0f} in the next month ({band_of(low, D.scales[gauge])})")
    return row


def event_study(D: Data) -> dict:
    rows = [r for r in (event_row(D, e) for e in load_market_events()) if r is not None]
    graded = {}
    for kind in ("trough", "peak"):
        rs = [r for r in rows if r["kind"] == kind and r.get("hit") is not None]
        graded[kind] = {"n": len(rs), "yes": sum(r["hit"] == "yes" for r in rs),
                        "partial": sum(r["hit"] == "partial" for r in rs), "no": sum(r["hit"] == "no" for r in rs)}
    corr = {}
    for market in ("equity", "ai"):
        rs = [r for r in rows if r["market"] == market and np.isfinite(r["reading"])]
        for k in ("3m", "6m", "12m"):
            pairs = np.array([(r["reading"], r[f"ret_{k}"]) for r in rs if np.isfinite(r[f"ret_{k}"])])
            if len(pairs) >= 5:
                corr[f"{market}_{k}"] = {"n": len(pairs), "pearson": pearson(pairs[:, 0], pairs[:, 1]),
                                         "spearman": pearson(ranks(pairs[:, 0]), ranks(pairs[:, 1]))}
    return {"rows": rows, "graded": graded, "correlation": corr}


# --------------------------------------------------------------------------- #
# 4. Strategies
# --------------------------------------------------------------------------- #
def greed_trim(g: pd.Series, hi: float = GREED, back: float = 60.0, trim: float = 0.5) -> pd.Series:
    """Fully invested; cut to `trim` once the gauge reaches `hi`; back to fully invested once it falls to `back`."""
    x = g.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    state = None
    for i, v in enumerate(x):
        if np.isnan(v):
            if state is not None:
                out[i] = state
            continue
        if state is None:
            state = 1.0
        if state == 1.0 and v >= hi:
            state = trim
        elif state != 1.0 and v <= back:
            state = 1.0
        out[i] = state
    return pd.Series(out, index=g.index)


def fear_add(g: pd.Series, lo: float = FEAR, hold: int = 126, base: float = 0.6) -> pd.Series:
    """`base` exposure; fully invested for `hold` trading days after any reading at or below `lo`."""
    fear = (g <= lo).astype(float).where(g.notna())
    recent = fear.rolling(hold, min_periods=1).max().fillna(0.0)
    return (base + (1 - base) * recent).where(g.notna().cummax())


def tilt(g: pd.Series, lower: float = FEAR, upper: float = GREED, floor: float = 0.5) -> pd.Series:
    """Fully invested at or below `lower`, `floor` at or above `upper`, linear between."""
    w = ((upper - g) / (upper - lower)).clip(0, 1)
    return (floor + (1 - floor) * w).ffill().where(g.notna().cummax())


def level_filter(s: pd.Series, level: float, below: bool, low: float = 0.5) -> pd.Series:
    """Fully invested; `low` while s is below `level` (below=True) or at/above it (below=False)."""
    trip = (s < level) if below else (s >= level)
    return pd.Series(np.where(trip, low, 1.0), index=s.index).where(s.notna()).ffill().where(s.notna().cummax())


def run_strategy(expo: pd.Series, r: pd.Series, cash: pd.Series, delay: int = DELAY, cost_bps: float = 0.0) -> pd.DataFrame:
    """Daily returns of holding `expo` of the asset (decided at each close, traded `delay` days later), the
    rest in cash, less `cost_bps` per 100% of the portfolio traded."""
    held = expo.shift(delay + 1)
    turnover = held.diff().abs().fillna(0.0)
    ret = held * r + (1 - held) * cash - turnover * cost_bps / 1e4
    ok = held.notna() & r.notna() & cash.notna()
    return pd.DataFrame({"ret": ret[ok], "held": held[ok]})


def metrics(ret: pd.Series, cash: pd.Series) -> dict:
    if len(ret) < YEAR:
        return {}
    eq = (1 + ret).cumprod()
    years = len(ret) / YEAR
    cagr = eq.iloc[-1] ** (1 / years) - 1
    ex = ret - cash.reindex(ret.index)
    mdd = float((eq / eq.cummax().clip(lower=1.0) - 1).min())
    return {"cagr": float(cagr * 100), "vol": float(ret.std() * math.sqrt(YEAR) * 100),
            "sharpe": float(ex.mean() / ex.std() * math.sqrt(YEAR)) if ex.std() > 0 else math.nan,
            "max_dd": mdd * 100, "calmar": float(cagr / abs(mdd)) if mdd < 0 else math.nan,
            "years": years, "start": ret.index[0], "end": ret.index[-1]}


def evaluate(expo: pd.Series, r: pd.Series, cash: pd.Series, delay: int = DELAY, cost_bps: float = 0.0,
             start=None, end=None) -> dict | None:
    run = run_strategy(expo, r, cash, delay, cost_bps).loc[start:end]
    if len(run) < 2 * YEAR:
        return None
    ret, held = run["ret"], run["held"]
    c, bh = cash.reindex(ret.index), r.reindex(ret.index)
    m = float(held.mean())
    matched = m * bh + (1 - m) * c
    out = {"ret": ret, "bh": bh, "matched": matched, "cash": c, "held": held,
           "m": metrics(ret, c), "m_bh": metrics(bh, c), "m_matched": metrics(matched, c),
           "exposure": m * 100, "turnover": float(held.diff().abs().sum() / (len(held) / YEAR) * 100)}
    out["d_sharpe"] = out["m"]["sharpe"] - out["m_bh"]["sharpe"]
    out["d_cagr_matched"] = out["m"]["cagr"] - out["m_matched"]["cagr"]
    out["d_cagr_bh"] = out["m"]["cagr"] - out["m_bh"]["cagr"]
    out["d_mdd"] = out["m"]["max_dd"] - out["m_bh"]["max_dd"]
    years = pd.DataFrame({"s": ret, "b": bh, "m": matched}).groupby(ret.index.year).apply(lambda d: (1 + d).prod() - 1)
    years = years[ret.groupby(ret.index.year).size() >= 120]
    out["win_years"] = win_share(years["s"], years["b"])
    out["win_years_matched"] = win_share(years["s"], years["m"])
    return out


def win_share(a: pd.Series, b: pd.Series, tol: float = 1e-4) -> float:
    """Share (%) of years `a` beat `b`, among years where they differ (a rule that sat at 100% all year ties)."""
    diff = (a - b)[(a - b).abs() > tol]
    return float((diff > 0).mean() * 100) if len(diff) else math.nan


def strategy_specs(D: Data) -> list[dict]:
    g1, g2, g3, fr = D.display["G1"], D.display["G2"], D.display["G3"], D.overlay["fragility"]
    t, health = tilt(g1), level_filter(g2, 40.0, below=True)
    return [
        {"key": "greed_trim", "name": "Greed trim", "asset": "SPY", "expo": greed_trim(g1),
         "rule": "100% SPY; 50% once G1 reaches 80 (Extreme Greed); back to 100% when G1 falls to 60"},
        {"key": "fear_add", "name": "Fear add", "asset": "SPY", "expo": fear_add(g1),
         "rule": "60% SPY; 100% for 6 months after any G1 reading at or below 20 (Extreme Fear)"},
        {"key": "tilt", "name": "Contrarian tilt", "asset": "SPY", "expo": t,
         "rule": "100% SPY at G1 ≤ 20, 50% at G1 ≥ 80, linear between"},
        {"key": "health", "name": "Health filter", "asset": "SPY", "expo": health,
         "rule": "100% SPY; 50% while G2 is below 40 (Weakening or Breaking down)"},
        {"key": "fragility", "name": "Fragility de-risk", "asset": "SPY", "expo": level_filter(fr, 80.0, below=False),
         "rule": "100% SPY; 50% while the fragility overlay is at or above 80"},
        {"key": "tilt_health", "name": "Tilt + health", "asset": "SPY", "expo": np.minimum(t, health).where(t.notna() & health.notna()),
         "rule": "Contrarian tilt, capped at 50% while G2 is below 40"},
        {"key": "ai_tilt", "name": "AI contrarian tilt", "asset": "SMH", "expo": tilt(g3),
         "rule": "100% SMH at G3 ≤ 20, 50% at G3 ≥ 80, linear between (G3 has hindsight and low coverage)"},
    ]


def weekly_curve(ret: pd.Series) -> pd.Series:
    return (1 + ret).cumprod().resample("W-FRI").last()


def run_strategies(D: Data) -> dict:
    out = {}
    for spec in strategy_specs(D):
        ev = evaluate(spec["expo"], D.tr[spec["asset"]], D.cash)
        if ev is not None:
            out[spec["key"]] = {**spec, **ev}
    return out


# --------------------------------------------------------------------------- #
# 5. Robustness checks
# --------------------------------------------------------------------------- #
GRIDS = {
    "greed_trim": ("G1", greed_trim, [dict(hi=h, back=b, trim=t) for h in (65, 70, 75, 80, 85, 90)
                                      for b in (40, 50, 60) for t in (0.0, 0.5)]),
    "fear_add": ("G1", fear_add, [dict(lo=lo, hold=d, base=0.6) for lo in (10, 15, 20, 25, 30, 35)
                                  for d in (21, 63, 126, 252)]),
    "tilt": ("G1", tilt, [dict(lower=a, upper=b, floor=f) for a in (10, 20, 30) for b in (70, 80, 90)
                          for f in (0.0, 0.5)]),
}


def grid_check(D: Data) -> dict:
    out = {}
    r = D.tr["SPY"]
    for key, (gid, fn, params) in GRIDS.items():
        rows = []
        for p in params:
            ev = evaluate(fn(D.display[gid], **p), r, D.cash)
            if ev is None:
                continue
            rows.append({**p, "d_sharpe": ev["d_sharpe"], "d_cagr_matched": ev["d_cagr_matched"],
                         "d_cagr_bh": ev["d_cagr_bh"], "d_mdd": ev["d_mdd"], "sharpe": ev["m"]["sharpe"],
                         "exposure": ev["exposure"]})
        ds = np.array([x["d_sharpe"] for x in rows])
        dc = np.array([x["d_cagr_matched"] for x in rows])
        dd = np.array([x["d_mdd"] for x in rows])
        out[key] = {"rows": rows, "n": len(rows),
                    "share_sharpe_up": float((ds > 0).mean() * 100) if len(ds) else math.nan,
                    "share_beat_matched": float((dc > 0).mean() * 100) if len(dc) else math.nan,
                    "share_shallower_dd": float((dd > 0).mean() * 100) if len(dd) else math.nan,
                    "median_d_sharpe": float(np.median(ds)) if len(ds) else math.nan,
                    "min_d_sharpe": float(ds.min()) if len(ds) else math.nan,
                    "max_d_sharpe": float(ds.max()) if len(ds) else math.nan,
                    "median_d_cagr_matched": float(np.median(dc)) if len(dc) else math.nan}
    return out


def periods(D: Data) -> dict[str, tuple]:
    last = D.cal[-1].year
    rel = D.reliable
    rel_start = rel[rel].index.min()
    cov70 = (D.coverage["G1"] < 70) | D.coverage["G1"].isna()
    start70 = D.cal[D.cal > cov70[cov70].index.max()].min() if cov70.any() else D.cal[0]
    return {"2000–2008": ("2000-01-01", "2008-12-31"), "2009–2016": ("2009-01-01", "2016-12-31"),
            f"2017–{last}": ("2017-01-01", f"{last}-12-31"),
            f"G1 coverage ≥ {RELIABLE:.0f}% (from {rel_start.year})": (rel_start, None),
            f"G1 coverage ≥ 70% (from {start70.year})": (start70, None)}


def subperiod_check(D: Data, specs: list[dict]) -> list[dict]:
    rows = []
    for label, (a, b) in periods(D).items():
        for spec in specs:
            ev = evaluate(spec["expo"], D.tr[spec["asset"]], D.cash, start=a, end=b)
            if ev is None:
                continue
            rows.append({"period": label, "key": spec["key"], "cagr": ev["m"]["cagr"], "cagr_bh": ev["m_bh"]["cagr"],
                         "sharpe": ev["m"]["sharpe"], "sharpe_bh": ev["m_bh"]["sharpe"], "d_sharpe": ev["d_sharpe"],
                         "d_cagr_matched": ev["d_cagr_matched"], "max_dd": ev["m"]["max_dd"],
                         "max_dd_bh": ev["m_bh"]["max_dd"], "win_years": ev["win_years"]})
    return rows


def sensitivity(D: Data, specs: list[dict], **kw) -> list[dict]:
    rows = []
    name, values = next(iter(kw.items()))
    for v in values:
        for spec in specs:
            args = {"delay": DELAY, "cost_bps": 0.0}
            args["delay" if name == "delay" else "cost_bps"] = v
            ev = evaluate(spec["expo"], D.tr[spec["asset"]], D.cash, **args)
            if ev is not None:
                rows.append({name: v, "key": spec["key"], "d_sharpe": ev["d_sharpe"],
                             "d_cagr_matched": ev["d_cagr_matched"], "cagr": ev["m"]["cagr"], "sharpe": ev["m"]["sharpe"]})
    return rows


def composite_without(D: Data, gid: str, drop: str | None) -> pd.Series:
    """Rebuild a gauge's display from its stored pillar scores, leaving one pillar out."""
    P = D.pillars[gid]
    if drop is not None:
        P = P.drop(columns=[drop], errors="ignore")
    w = pd.Series({k: v for k, v in D.weights[gid].items() if k in P.columns}, dtype=float)
    P = P[w.index]
    denom = P.notna().mul(w, axis=1).sum(axis=1).replace(0, np.nan)
    g = (P.fillna(0) * w).sum(axis=1) / denom
    return (T.ema(g, 3) if D.smoothing[gid] == "ema3" else g).where(g.notna())


def variant_check(D: Data) -> dict:
    g1 = D.display["G1"]
    spx, r = D.close["^GSPC"], D.tr["SPY"]
    rebuilt = composite_without(D, "G1", None)
    recon = float((rebuilt - g1).abs().max())
    variants = {"G1 display (3-day EMA)": g1, "G1 raw score": D.raw["G1"], "CNN-method replica": D.overlay["cnn_replica"],
                "G1 sampled weekly (Fridays)": g1.where(g1.index.dayofweek == 4).ffill(limit=6)}
    for p in D.weights["G1"]:
        variants[f"G1 without {p}"] = composite_without(D, "G1", p)
    fwd63 = forward_return(spx, 63)
    rows = []
    for label, s in variants.items():
        st = zone_study(s, spx, "fear", FEAR, np.random.default_rng(0), 200)
        start3m = next(x for x in st["summary"] if x["anchor"] == "start" and x["horizon"] == "3m")
        m = s.notna() & fwd63.notna()
        row = {"variant": label, "corr_with_g1": pearson(s[m & g1.notna()].to_numpy(), g1[m & g1.notna()].to_numpy()),
               "ic_fwd3m": pearson(ranks(s[m].to_numpy()), ranks(fwd63[m].to_numpy())),
               "fear_episodes": len(st["episodes"]), "fear_excess_3m": start3m.get("excess", math.nan)}
        for key, fn in (("tilt", tilt), ("greed_trim", greed_trim)):
            ev = evaluate(fn(s), r, D.cash)
            row[f"{key}_d_sharpe"] = ev["d_sharpe"] if ev else math.nan
            row[f"{key}_d_cagr_matched"] = ev["d_cagr_matched"] if ev else math.nan
        rows.append(row)
    return {"rows": rows, "reconstruction_error": recon}


def asset_check(D: Data) -> list[dict]:
    rows = []
    g1 = D.display["G1"]
    labels = {"SPY": "SPY (total return)", "^GSPC": "S&P 500 (price only)", "RSP": "RSP equal weight (TR, from 2003)",
              "IWM": "IWM small caps (TR)"}
    for sym, label in labels.items():
        for key, expo in (("greed_trim", greed_trim(g1)), ("fear_add", fear_add(g1)), ("tilt", tilt(g1))):
            ev = evaluate(expo, D.tr[sym], D.cash)
            if ev is not None:
                rows.append({"asset": label, "key": key, "d_sharpe": ev["d_sharpe"], "d_cagr_matched": ev["d_cagr_matched"],
                             "d_mdd": ev["d_mdd"], "sharpe_bh": ev["m_bh"]["sharpe"]})
    return rows


def placebo(held: np.ndarray, r: np.ndarray, cash: np.ndarray, rng: np.random.Generator, n: int = 1000) -> dict:
    """Circular-shift test: the same exposure path laid over shifted dates keeps its level, persistence and
    turnover but loses any timing. Returns the actual statistics and one-sided p-values."""
    T_ = len(held)
    if T_ < 3 * YEAR:
        return {}

    def stats(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ret = h * r + (1 - h) * cash
        ex = ret - cash
        sharpe = ex.mean(axis=-1) / ex.std(axis=-1, ddof=1) * math.sqrt(YEAR)
        cagr = (np.exp(np.log1p(ret).sum(axis=-1) / (T_ / YEAR)) - 1) * 100
        return sharpe, cagr

    shifts = rng.integers(YEAR, T_ - YEAR, size=n)
    idx = (np.arange(T_)[None, :] - shifts[:, None]) % T_
    s_null, c_null = stats(held[idx])
    s_act, c_act = stats(held)
    return {"sharpe": float(s_act), "cagr": float(c_act), "null_sharpe_median": float(np.median(s_null)),
            "null_sharpe_p95": float(np.percentile(s_null, 95)), "null_cagr_median": float(np.median(c_null)),
            "p_sharpe": float(((s_null >= s_act).sum() + 1) / (n + 1)),
            "p_cagr": float(((c_null >= c_act).sum() + 1) / (n + 1)), "shifts": n}


def placebo_check(D: Data, runs: dict, rng: np.random.Generator, n: int) -> dict:
    out = {}
    for key, run in runs.items():
        out[key] = placebo(run["held"].to_numpy(), run["bh"].to_numpy(), run["cash"].to_numpy(), rng, n)
    return out


# --------------------------------------------------------------------------- #
# 6. Correlations
# --------------------------------------------------------------------------- #
def block_bootstrap_ci(x: np.ndarray, y: np.ndarray, block: int, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    """95% interval for Pearson's r by moving-block bootstrap (blocks of `block` consecutive pairs)."""
    n = len(x)
    block = max(1, min(block, n // 4))
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
    X, Y = x[idx], y[idx]
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)
    r = (X * Y).sum(1) / np.sqrt((X ** 2).sum(1) * (Y ** 2).sum(1))
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return float(lo), float(hi)


def corr_stats(x: pd.Series, y: pd.Series, h: int, reps: int, rng: np.random.Generator) -> dict:
    m = x.notna() & y.notna()
    xv, yv = x[m].to_numpy(dtype=float), y[m].to_numpy(dtype=float)
    n = len(xv)
    if n < max(100, 3 * h):
        return {"n": n}
    xr, yr = ranks(xv), ranks(yv)
    xs, ys = xv[::h], yv[::h]
    r_ind = pearson(xs, ys)
    t = r_ind * math.sqrt((len(xs) - 2) / (1 - r_ind ** 2)) if np.isfinite(r_ind) and abs(r_ind) < 1 else math.nan
    return {"n": n, "pearson": pearson(xv, yv), "pearson_ci": block_bootstrap_ci(xv, yv, h, reps, rng),
            "spearman": pearson(xr, yr), "spearman_ci": block_bootstrap_ci(xr, yr, h, reps, rng),
            "n_indep": len(xs), "pearson_indep": r_ind, "t_indep": t, "start": x[m].index[0]}


def correlations(D: Data, reps: int, rng: np.random.Generator) -> dict:
    signals = {"G1": (D.display["G1"], "^GSPC"), "G2": (D.display["G2"], "^GSPC"),
               "Fragility": (D.overlay["fragility"], "^GSPC"), "CNN replica": (D.overlay["cnn_replica"], "^GSPC"),
               "G3": (D.display["G3"], "SMH"), "G4": (D.display["G4"], "SMH")}
    rows = []
    for name, (s, asset) in signals.items():
        price = D.close[asset]
        targets = {f"{k} return": (forward_return(price, h), h) for k, h in H.items()}
        targets["3m max drawdown"] = (forward_max_drawdown(price, 63), 63)
        for tname, (y, h) in targets.items():
            for sample, mask in (("all", None), ("reliable", D.reliable)):
                xs = s if mask is None else s.where(mask)
                st = corr_stats(xs, y, h, reps, rng)
                if st.get("n", 0) and "pearson" in st:
                    rows.append({"signal": name, "asset": asset, "target": tname, "sample": sample, **st})
    level = pd.DataFrame({"G1": D.display["G1"], "G2": D.display["G2"], "G3": D.display["G3"], "G4": D.display["G4"],
                          "Fragility": D.overlay["fragility"], "CNN replica": D.overlay["cnn_replica"], "VIX": D.vix})
    matrix = level.corr().round(3)
    return {"rows": rows, "matrix": {c: matrix[c].to_dict() for c in matrix}}


# --------------------------------------------------------------------------- #
# 7. Scenarios
# --------------------------------------------------------------------------- #
def stress_replay(D: Data, cycles: dict, runs: dict) -> list[dict]:
    windows = [("Bear", c) for c in cycles["bears"]] + [("Correction", c) for c in cycles["corrections"]]
    keys = [k for k in runs if runs[k]["asset"] == "SPY"]
    bh = D.tr["SPY"].dropna()
    rows = []
    for label, c in sorted(windows, key=lambda w: w[1]["start"]):
        a, b = c["start"], c["end"]
        after_end = D.cal[min(len(D.cal) - 1, D.cal.get_loc(b) + YEAR)]
        row = {"type": label, "start": a, "end": b, "depth": c["change"]}
        for k in ["bh"] + keys:
            ret = bh if k == "bh" else runs[k]["ret"]
            w = ret.loc[a:b].iloc[1:]
            if len(w) < 5 or ret.index[0] > a:
                continue
            eq = (1 + w).cumprod()
            row[f"{k}_ret"] = float((eq.iloc[-1] - 1) * 100)
            row[f"{k}_dd"] = float((eq / eq.cummax().clip(lower=1.0) - 1).min() * 100)
            nxt = ret.loc[b:after_end].iloc[1:]
            row[f"{k}_next12m"] = float(((1 + nxt).prod() - 1) * 100) if len(nxt) >= YEAR * 0.9 else math.nan
        rows.append(row)
    return rows


def current_state(D: Data) -> dict:
    X = pd.DataFrame({"G1": D.display["G1"], "G2": D.display["G2"], "fragility": D.overlay["fragility"]})
    last = X.dropna().index[-1]
    return {"date": last, **{k: float(v) for k, v in X.loc[last].items()},
            "g1_band": band_of(X.loc[last, "G1"], D.scales["G1"]), "g2_band": band_of(X.loc[last, "G2"], D.scales["G2"])}


def analogs(D: Data, k: int = 25, spacing: int = 21) -> dict:
    X = pd.DataFrame({"G1": D.display["G1"], "G2": D.display["G2"], "fragility": D.overlay["fragility"]})
    spx = D.close["^GSPC"]
    now = current_state(D)
    pos = np.arange(len(X))
    cand = X.notna().all(axis=1).to_numpy() & D.reliable.to_numpy() & (pos < len(X) - YEAR)
    Xc = X[cand]
    z = (X - Xc.mean()) / Xc.std()
    target = z.loc[now["date"]]
    dist = np.sqrt(((z[cand] - target) ** 2).sum(axis=1)).sort_values()
    chosen: list[int] = []
    for d in dist.index:
        p = D.cal.get_loc(d)
        if all(abs(p - c) >= spacing for c in chosen):
            chosen.append(p)
        if len(chosen) == k:
            break
    fwd = {kk: forward_return(spx, h).to_numpy() for kk, h in H.items()}
    mdd = forward_max_drawdown(spx, 63).to_numpy()
    base_idx = np.flatnonzero(cand)
    summary = {}
    for kk in ("3m", "6m", "12m"):
        summary[kk] = {"analogs": pct_summary(fwd[kk][chosen]), "all": pct_summary(fwd[kk][base_idx])}
    dd = mdd[chosen]
    rows = [{"date": D.cal[p], "G1": X["G1"].iloc[p], "G2": X["G2"].iloc[p], "fragility": X["fragility"].iloc[p],
             "distance": float(dist[D.cal[p]]), "ret_3m": fwd["3m"][p], "ret_12m": fwd["12m"][p], "mdd3m": mdd[p]}
            for p in chosen]
    return {"now": now, "k": len(chosen), "summary": summary, "rows": rows,
            "dd10_analogs": float((dd <= -10).mean() * 100) if len(dd) else math.nan,
            "dd10_all": float((mdd[base_idx][np.isfinite(mdd[base_idx])] <= -10).mean() * 100)}


def band_grid(D: Data) -> dict:
    spx = D.close["^GSPC"]
    b1, b2 = bands(D.display["G1"]), bands(D.display["G2"])
    rel = D.reliable.to_numpy()
    f3, f12 = forward_return(spx, 63).to_numpy(), forward_return(spx, 252).to_numpy()
    cells = []
    for i in range(5):
        for j in range(5):
            sel = (b1 == i) & (b2 == j) & rel
            v3, v12 = f3[sel], f12[sel]
            v3, v12 = v3[np.isfinite(v3)], v12[np.isfinite(v12)]
            cells.append({"g1": i, "g2": j, "days": int(sel.sum()), "ret_3m": float(v3.mean()) if len(v3) else math.nan,
                          "ret_12m": float(v12.mean()) if len(v12) else math.nan,
                          "hit_12m": float((v12 > 0).mean() * 100) if len(v12) else math.nan})
    now = current_state(D)
    return {"cells": cells, "now": [band_index(now["G1"]), band_index(now["G2"])],
            "g1_scale": D.scales["G1"], "g2_scale": D.scales["G2"]}


# --------------------------------------------------------------------------- #
# 8. Monte Carlo
# --------------------------------------------------------------------------- #
def stationary_bootstrap(n: int, length: int, paths: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices: blocks of geometric length (mean `block`), wrapping."""
    idx = np.empty((paths, length), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, paths)
    jump = rng.random((paths, length)) < 1.0 / block
    fresh = rng.integers(0, n, (paths, length))
    for t in range(1, length):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def path_stats(R: np.ndarray, C: np.ndarray) -> dict[str, np.ndarray]:
    """Per path: CAGR %, Sharpe (excess over cash) and max drawdown %."""
    years = R.shape[1] / YEAR
    logeq = np.cumsum(np.log1p(R), axis=1)
    eq = np.exp(logeq)
    peak = np.maximum.accumulate(np.maximum(eq, 1.0), axis=1)
    ex = R - C
    return {"cagr": (np.exp(logeq[:, -1] / years) - 1) * 100,
            "sharpe": ex.mean(axis=1) / ex.std(axis=1, ddof=1) * math.sqrt(YEAR),
            "mdd": (eq / peak - 1).min(axis=1) * 100}


def pctl(v: np.ndarray) -> dict:
    return {f"p{q}": float(np.percentile(v, q)) for q in PCTS}


def mc_strategies(runs: dict, paths: int, years: int, block: float, rng: np.random.Generator) -> dict:
    keys = [k for k in runs if runs[k]["asset"] == "SPY"]
    common = runs[keys[0]]["ret"].index
    for k in keys[1:]:
        common = common.intersection(runs[k]["ret"].index)
    M = pd.DataFrame({"bh": runs[keys[0]]["bh"].reindex(common), **{k: runs[k]["ret"].reindex(common) for k in keys}})
    cash = runs[keys[0]]["cash"].reindex(common).to_numpy()
    idx = stationary_bootstrap(len(M), years * YEAR, paths, block, rng)
    C = cash[idx]
    stats = {k: path_stats(M[k].to_numpy()[idx], C) for k in M}
    b = stats["bh"]
    rows = []
    for k, s in stats.items():
        rows.append({"key": k, "cagr": pctl(s["cagr"]), "mdd": pctl(s["mdd"]), "sharpe": pctl(s["sharpe"]),
                     "p_loss": float((s["cagr"] < 0).mean() * 100),
                     "p_beat_cagr": float((s["cagr"] > b["cagr"]).mean() * 100) if k != "bh" else math.nan,
                     "p_beat_sharpe": float((s["sharpe"] > b["sharpe"]).mean() * 100) if k != "bh" else math.nan,
                     "p_shallower_dd": float((s["mdd"] > b["mdd"]).mean() * 100) if k != "bh" else math.nan})
    return {"rows": rows, "paths": paths, "years": years, "block": block, "start": common[0], "end": common[-1]}


def state_fan(r: np.ndarray, state: np.ndarray, start_state: int, rng: np.random.Generator,
              paths: int = 10000, steps: int = 12, block: int = 21) -> dict:
    """Simulate `steps` blocks of `block` days. Each block is a real stretch of history that began in the
    current state; the next state is the one history was in at that block's end."""
    n = len(r)
    starts = np.arange(n - block - 1)
    windows = np.lib.stride_tricks.sliding_window_view(r[1:], block)
    ok = (state[starts] >= 0) & (state[starts + block] >= 0) & np.isfinite(windows[starts]).all(axis=1)
    pools = {s: starts[ok & (state[starts] == s)] for s in np.unique(state[starts][ok])}
    if start_state not in pools or len(pools[start_state]) < 21:
        return {}
    cur = np.full(paths, start_state)
    sim = np.empty((paths, steps * block))
    for t in range(steps):
        pick = np.empty(paths, dtype=np.int64)
        for s in np.unique(cur):
            mask = cur == s
            pool = pools.get(s)
            if pool is None or not len(pool):
                pool = np.concatenate(list(pools.values()))
            pick[mask] = rng.choice(pool, mask.sum())
        sim[:, t * block:(t + 1) * block] = windows[pick]
        cur = state[pick + block]
    eq = np.cumprod(1 + sim, axis=1)
    peak = np.maximum.accumulate(np.maximum(eq, 1.0), axis=1)
    mdd = (eq / peak - 1).min(axis=1) * 100
    final = (eq[:, -1] - 1) * 100
    months = [{"month": m, **pctl((eq[:, m * block - 1] - 1) * 100)} for m in range(1, steps + 1)]
    return {"paths": paths, "pool_days": int(sum(len(p) for p in pools.values())), "start_pool": int(len(pools[start_state])),
            "final": pctl(final), "mean": float(final.mean()), "p_loss": float((final < 0).mean() * 100),
            "p_dd10": float((mdd <= -10).mean() * 100), "p_dd20": float((mdd <= -20).mean() * 100),
            "months": months}


def mc_fans(D: Data, rng: np.random.Generator, paths: int) -> dict:
    out = {}
    r = D.tr["SPY"].to_numpy()
    rel = D.reliable.to_numpy()
    b1 = np.where(rel, bands(D.display["G1"]), -1)
    now = band_index(current_state(D)["G1"])
    out["spy_g1"] = {**state_fan(r, b1, now, rng, paths), "state": D.scales["G1"][now], "gauge": "G1", "asset": "SPY"}
    out["spy_all"] = {**state_fan(r, np.where(rel & np.isfinite(r), 0, -1), 0, rng, paths), "state": "any",
                      "gauge": "none", "asset": "SPY"}
    g3 = D.display["G3"]
    if g3.notna().sum() > 3 * YEAR:
        r3 = D.tr["SMH"].to_numpy()
        b3 = bands(g3)
        now3 = band_index(g3.dropna().iloc[-1])
        out["smh_g3"] = {**state_fan(r3, b3, now3, rng, paths), "state": D.scales["G3"][now3], "gauge": "G3", "asset": "SMH"}
        out["smh_all"] = {**state_fan(r3, np.where(np.isfinite(g3.to_numpy()), 0, -1), 0, rng, paths), "state": "any",
                          "gauge": "none", "asset": "SMH"}
    return out


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def study(store: Store, paths: int = 2000, reps: int = 1000, seed: int = 20261015) -> dict:
    rng = np.random.default_rng(seed)
    D = load(store)
    spx = D.close["^GSPC"]
    cycles = market_cycles(D)
    zones = {"G1": {"fear": zone_study(D.display["G1"], spx, "fear", FEAR, rng, reps * 10),
                    "greed": zone_study(D.display["G1"], spx, "greed", GREED, rng, reps * 10)},
             "G3": {"fear": zone_study(D.display["G3"], D.close["SMH"], "fear", FEAR, rng, reps * 10),
                    "greed": zone_study(D.display["G3"], D.close["SMH"], "greed", GREED, rng, reps * 10)}}
    thresholds = threshold_sensitivity(D.display["G1"], spx, rng, reps)
    zones["G1"]["correlation"] = episode_correlation(zones["G1"])
    band_tables = {"G1 all": band_table(D.display["G1"], spx, D.scales["G1"]),
                   "G1 reliable": band_table(D.display["G1"], spx, D.scales["G1"], D.reliable),
                   "G2 reliable": band_table(D.display["G2"], spx, D.scales["G2"], D.reliable),
                   "G3 vs SMH": band_table(D.display["G3"], D.close["SMH"], D.scales["G3"])}
    events = event_study(D)
    runs = run_strategies(D)
    specs = [s for s in strategy_specs(D) if s["key"] in runs]
    spy_specs = [s for s in specs if s["asset"] == "SPY"]
    checks = {"grids": grid_check(D), "subperiods": subperiod_check(D, specs),
              "delay": sensitivity(D, spy_specs, delay=(0, 1, 2, 5, 10)),
              "costs": sensitivity(D, spy_specs, cost=(0, 5, 10, 25, 50)),
              "variants": variant_check(D), "assets": asset_check(D),
              "placebo": placebo_check(D, runs, rng, reps)}
    corr = correlations(D, reps, rng)
    scenarios = {"stress": stress_replay(D, cycles, runs), "analogs": analogs(D), "grid": band_grid(D)}
    mc = {"strategies": mc_strategies(runs, paths, 10, 63, rng), "fans": mc_fans(D, rng, paths * 5)}
    return {"D": D, "cycles": cycles, "zones": zones, "thresholds": thresholds, "band_tables": band_tables,
            "events": events, "runs": runs, "checks": checks, "corr": corr, "scenarios": scenarios, "mc": mc,
            "seed": seed, "paths": paths, "reps": reps}


# --------------------------------------------------------------------------- #
# Output: JSON for the dashboard page
# --------------------------------------------------------------------------- #
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.date().isoformat() if not pd.isna(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, np.bool_):
        return bool(x)
    if x is pd.NaT:
        return None
    return x


def to_json(res: dict) -> dict:
    D = res["D"]
    wk = lambda s: s.resample("W-FRI").last()
    frame = pd.DataFrame({"G1": wk(D.display["G1"]), "G2": wk(D.display["G2"]), "G3": wk(D.display["G3"]),
                          "G4": wk(D.display["G4"]), "fragility": wk(D.overlay["fragility"]),
                          "spx": wk(D.close["^GSPC"]), "smh": wk(D.close["SMH"])})
    series = {"dates": [d.date().isoformat() for d in frame.index], **{c: [jsonable(v) for v in frame[c]] for c in frame}}
    strategies = []
    for key, run in res["runs"].items():
        curve = pd.DataFrame({"strategy": weekly_curve(run["ret"]), "buy_hold": weekly_curve(run["bh"]),
                              "matched": weekly_curve(run["matched"])})
        strategies.append({k: run[k] for k in ("key", "name", "rule", "asset", "exposure", "turnover", "d_sharpe",
                                                "d_cagr_matched", "d_cagr_bh", "d_mdd", "win_years", "win_years_matched")}
                          | {"metrics": run["m"], "buy_hold": run["m_bh"], "matched": run["m_matched"],
                             "curve": {"dates": [d.date().isoformat() for d in curve.index],
                                       **{c: [jsonable(v) for v in curve[c]] for c in curve}}})
    events = [{k: v for k, v in r.items() if k not in ("date",)} | {"date": r["date"]} for r in res["events"]["rows"]]
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": D.cal[-1], "seed": res["seed"], "paths": res["paths"], "reps": res["reps"],
        "reliable_from": D.reliable[D.reliable].index.min(), "scales": D.scales,
        "series": series, "cycles": res["cycles"], "zones": res["zones"], "thresholds": res["thresholds"],
        "band_tables": res["band_tables"], "events": {**res["events"], "rows": events},
        "strategies": strategies, "checks": res["checks"], "correlations": res["corr"],
        "scenarios": res["scenarios"], "monte_carlo": res["mc"], "scorecard": scorecard(res),
    }
    return jsonable(payload)


# --------------------------------------------------------------------------- #
# Output: markdown report
# --------------------------------------------------------------------------- #
def n_(x, nd: int = 1, sign: bool = False, pct: bool = False) -> str:
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "—"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s + ("%" if pct else "")


def d_(x) -> str:
    return "—" if x is None or pd.isna(x) else pd.Timestamp(x).date().isoformat()


def p_(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def headline_corr(res: dict) -> dict:
    return next((r for r in res["corr"]["rows"] if r["signal"] == "G1" and r["target"] == "3m return"
                 and r["sample"] == "all"), {})


def scorecard(res: dict) -> dict:
    z = res["zones"]["G1"]
    pick = lambda zone, a, k: next(r for r in z[zone]["summary"] if r["anchor"] == a and r["horizon"] == k)
    runs = res["runs"]
    best = max((k for k in runs if runs[k]["asset"] == "SPY"), key=lambda k: runs[k]["m"]["sharpe"])
    g = res["events"]["graded"]
    return {"headline_corr": headline_corr(res), "event_corr": res["events"]["correlation"],
            "fear_3m": pick("fear", "start", "3m"), "fear_12m": pick("fear", "start", "12m"),
            "greed_3m": pick("greed", "start", "3m"), "greed_12m": pick("greed", "start", "12m"),
            "fear_exit_1m": pick("fear", "exit", "1m"), "fear_exit_3m": pick("fear", "exit", "3m"),
            "fear_exit_12m": pick("fear", "exit", "12m"),
            "band_fear": res["band_tables"]["G1 reliable"][0], "band_greed": res["band_tables"]["G1 reliable"][4],
            "episode_corr": z.get("correlation", {}),
            "troughs": g.get("trough"), "peaks": g.get("peak"), "best_sharpe": best,
            "grids": {k: {kk: v[kk] for kk in ("n", "share_sharpe_up", "share_beat_matched", "median_d_sharpe")}
                      for k, v in res["checks"]["grids"].items()},
            "placebo": {k: {kk: v.get(kk) for kk in ("p_sharpe", "p_cagr")} for k, v in res["checks"]["placebo"].items()},
            "fan": {k: {kk: v.get(kk) for kk in ("state", "mean", "p_loss", "p_dd10")} | {"median": v.get("final", {}).get("p50")}
                    for k, v in res["mc"]["fans"].items() if v}}


def render(res: dict) -> str:
    D = res["D"]
    sc = scorecard(res)
    runs = res["runs"]
    names = {k: v["name"] for k, v in runs.items()} | {"bh": "Buy and hold"}
    rel_from = D.reliable[D.reliable].index.min()
    hc = sc["headline_corr"]
    out = ["# Robustness study", "",
           f"Generated by `make robustness` (`python -m gauges.robustness`) on {datetime.now(timezone.utc):%Y-%m-%d} from "
           f"the scored history through {d_(D.cal[-1])}. Seed {res['seed']}; {res['paths']:,} Monte Carlo paths for "
           f"strategies, {res['paths'] * 5:,} for the 12-month fans; {res['reps']:,} bootstrap draws for intervals.", "",
           "**How to read this.** Every reading is the gauge as it stood that day, computed from data public by then "
           "(the same point-in-time history the dashboard draws). Three things flatter or weaken it, and each table "
           "says when they matter:", "",
           f"- **Coverage.** G1's live weight is under 40% before 2007, so early readings rest on few inputs. History "
           f"counts as reliable from {d_(rel_from)}, when G1 coverage stays at or above {RELIABLE:.0f}%. "
           "G4 has almost no history before 2019 and is left out of the event study.",
           "- **In-sample choices.** The Phase 3 backtest removed indicators using this same history "
           "(reports/validation.md), which flatters every result below to some degree. Rules here use the "
           "dashboard's band edges, not fitted thresholds; the grids show how much the choice matters.",
           "- **Hindsight in the AI basket.** G3 and G4 use today's AI companies applied backwards, so their history "
           "is optimistic about which names mattered.", "",
           "Returns are S&P 500 price returns for events and episodes, and SPY total return (dividends reinvested) "
           "for strategies, with idle cash earning the fed funds rate. Strategies act on the close after the signal.", "",
           "## Scorecard", ""]
    fear3, fear12, greed3, greed12 = sc["fear_3m"], sc["fear_12m"], sc["greed_3m"], sc["greed_12m"]
    tr, pk = sc["troughs"] or {}, sc["peaks"] or {}
    lines = [
        ["Correlation of G1 with the next 3 months' S&P 500 return (all days)",
         f"Pearson r = {n_(hc.get('pearson'), 3)} (95% CI {n_(hc.get('pearson_ci', [math.nan])[0], 2)} to "
         f"{n_(hc.get('pearson_ci', [math.nan, math.nan])[1], 2)}); Spearman ρ = {n_(hc.get('spearman'), 3)}; "
         f"{hc.get('n_indep', 0)} non-overlapping windows, t = {n_(hc.get('t_indep'), 1)}"],
        [f"After Extreme Fear (G1 ≤ {FEAR:.0f}), {fear3.get('n', 0)} episodes",
         f"S&P 500 next 3m {n_(fear3.get('mean'), 1, True, True)} (base {n_(fear3.get('base'), 1, True, True)}, "
         f"p = {p_(fear3.get('p'))}); next 12m {n_(fear12.get('mean'), 1, True, True)} "
         f"(base {n_(fear12.get('base'), 1, True, True)}, p = {p_(fear12.get('p'))}); up after 12m in {n_(fear12.get('hit'), 0)}%"],
        [f"Leaving Extreme Fear (first day back above {FEAR:.0f}), same episodes",
         f"next 1m {n_(sc['fear_exit_1m'].get('mean'), 1, True, True)} (p = {p_(sc['fear_exit_1m'].get('p'))}); next 3m "
         f"{n_(sc['fear_exit_3m'].get('mean'), 1, True, True)} (p = {p_(sc['fear_exit_3m'].get('p'))}); next 12m "
         f"{n_(sc['fear_exit_12m'].get('mean'), 1, True, True)} (p = {p_(sc['fear_exit_12m'].get('p'))})"],
        [f"Every Extreme Fear day since {d_(rel_from)} ({sc['band_fear']['days']} days)",
         f"next 3m {n_(sc['band_fear']['3m'], 1, True, True)}, next 12m {n_(sc['band_fear']['12m'], 1, True, True)}, up after "
         f"12m {n_(sc['band_fear']['hit12m'], 0)}% (all reliable days: see 2, band table)"],
        [f"After Extreme Greed (G1 ≥ {GREED:.0f}), {greed3.get('n', 0)} episodes (none since "
         f"{d_(max((e['start'] for e in res['zones']['G1']['greed']['episodes']), default=None))})",
         f"next 3m {n_(greed3.get('mean'), 1, True, True)} (p = {p_(greed3.get('p'))}); next 12m "
         f"{n_(greed12.get('mean'), 1, True, True)} (p = {p_(greed12.get('p'))}); up after 12m in {n_(greed12.get('hit'), 0)}%"],
        ["Named market lows: G1 at Extreme Fear within 10 days",
         f"{tr.get('yes', 0)} of {tr.get('n', 0)} (Fear but not extreme: {tr.get('partial', 0)}; missed: {tr.get('no', 0)})"],
        ["Named market tops: Extreme Greed or fragility ≥ 80 in the 3 months before",
         f"{pk.get('yes', 0)} of {pk.get('n', 0)} (weaker warning: {pk.get('partial', 0)}; none: {pk.get('no', 0)})"],
    ]
    for key, label in (("equity_3m", "named equity events, G1 vs next 3m"), ("equity_12m", "named equity events, G1 vs next 12m")):
        c = sc["event_corr"].get(key)
        if c:
            lines.append([f"Correlation across {label}", f"r = {n_(c['pearson'], 2)}, ρ = {n_(c['spearman'], 2)} (n = {c['n']})"])
    for key in ("greed_trim", "fear_add", "tilt", "health", "fragility", "tilt_health"):
        if key in runs:
            r = runs[key]
            pl = res["checks"]["placebo"].get(key, {})
            lines.append([f"Strategy: {r['name']}",
                          f"Sharpe {n_(r['m']['sharpe'], 2)} vs {n_(r['m_bh']['sharpe'], 2)} buy-and-hold; max drawdown "
                          f"{n_(r['m']['max_dd'], 1, pct=True)} vs {n_(r['m_bh']['max_dd'], 1, pct=True)}; CAGR "
                          f"{n_(r['d_cagr_matched'], 2, True)} pts vs a constant mix at the same average exposure; "
                          f"placebo p (Sharpe) = {p_(pl.get('p_sharpe'))}"])
    for key, g in sc["grids"].items():
        lines.append([f"Parameter grid: {names.get(key, key)}",
                      f"{n_(g['share_sharpe_up'], 0)}% of {g['n']} settings beat buy-and-hold on Sharpe; "
                      f"{n_(g['share_beat_matched'], 0)}% beat the exposure-matched mix on CAGR"])
    fan, fan0 = sc["fan"].get("spy_g1", {}), sc["fan"].get("spy_all", {})
    if fan:
        lines.append([f"Next 12 months from today's G1 band ({fan.get('state')}), Monte Carlo",
                      f"median SPY {n_(fan.get('median'), 1, True, True)}, P(loss) {n_(fan.get('p_loss'), 0)}%, "
                      f"P(10%+ drawdown) {n_(fan.get('p_dd10'), 0)}% vs unconditional median "
                      f"{n_(fan0.get('median'), 1, True, True)}, P(loss) {n_(fan0.get('p_loss'), 0)}%, "
                      f"P(10%+ drawdown) {n_(fan0.get('p_dd10'), 0)}%"])
    out += table(["Check", "Result"], lines)

    # 1. Cycles
    out += ["## 1. Bull and bear markets", "",
            f"Turning points in S&P 500 closes: a bear market is a fall of {BEAR:.0f}% or more from a peak, ending at "
            f"the low before a {BEAR:.0f}% rise; corrections are {CORRECTION:.0f}–{BEAR:.0f}% falls outside bear "
            "markets. Dates come from the data, not from a list.", "", "### Bear markets and corrections", ""]
    rows = []
    for label, items in (("Bear", res["cycles"]["bears"]), ("Correction", res["cycles"]["corrections"])):
        for c in items:
            rows.append([label, d_(c["start"]), d_(c["end"]) + (" (so far)" if c["ongoing"] else ""), n_(c["change"], 1, pct=True),
                         c["days"], d_(c["recovered"]), n_(c["g1_cov"], 0), n_(c["g1_pre_max"], 0), n_(c["frag_pre_max"], 0),
                         n_(c["g1_low"], 0), d_(c["g1_low_date"]),
                         ("—" if c["g1_low_lead"] is None else f"{c['g1_low_lead']:+d}"), d_(c["first_fear"]),
                         n_(c["further_fall"], 1, pct=True), n_(c["fear_12m"], 1, True, True), n_(c["trough_12m"], 1, True, True)])
    rows.sort(key=lambda r: r[1])
    out += table(["Type", "Peak", "Low", "Fall", "Days", "Recovered", "G1 cov %", "G1 max 3m before", "Fragility max 3m before",
                  "G1 low", "G1 low date", "Days G1 low led price low", "First Extreme Fear", "Further fall after it",
                  "S&P 12m after first Extreme Fear", "S&P 12m after the low"], rows)
    out += ["_Days G1 low led price low_: positive means the gauge bottomed before the index did. "
            "_Further fall_: how much lower the index went after G1 first read Extreme Fear in that decline.", "",
            "### Bull markets", ""]
    rows = [[d_(c["start"]), d_(c["end"]) + (" (so far)" if c["ongoing"] else ""), n_(c["change"], 1, pct=True), c["days"],
             n_(c["g1_cov"], 0), n_(c["g1_start"], 0), n_(c["g1_max"], 0), d_(c["g1_max_date"]), n_(c["greed_share"], 0),
             c["extreme_greed_days"], d_(c["first_greed"]), n_(c["gain_after_greed"], 1, True, True), n_(c["g1_end"], 0),
             n_(c["frag_end"], 0)] for c in res["cycles"]["bulls"]]
    out += table(["Start (low)", "End (peak)", "Gain", "Days", "G1 cov %", "G1 at start", "G1 max", "G1 max date",
                  "% days G1 ≥ 60", "Extreme Greed days", "First Extreme Greed", "S&P gain after it", "G1 at peak",
                  "Fragility at peak"], rows)

    # 2. Episodes
    out += ["## 2. Peak fear and peak greed", "",
            f"An episode is a run of days in the zone, with visits less than {GAP} trading days apart merged. Returns "
            "are measured from the first day in the zone (tradeable), the most extreme day (hindsight: you only know it "
            "afterwards) and the first day out (tradeable). _p_ is the share of 10× more random-entry draws (the same "
            "number of random days with a reading) whose average beat the episodes': small p after fear means "
            "better-than-random returns; small p after greed means worse-than-random.", ""]
    for gid, asset in (("G1", "S&P 500"), ("G3", "SMH")):
        for zone in ("fear", "greed"):
            st = res["zones"][gid][zone]
            title = f"{gid} {'Extreme Fear (≤ ' + str(int(FEAR)) if zone == 'fear' else 'Extreme Greed (≥ ' + str(int(GREED))}) vs {asset}"
            out += [f"### {title}", "", f"{len(st['episodes'])} episodes, {st['days_in_zone']} days in the zone."
                    + (" G3 is low confidence (hindsight basket, about 40% coverage)." if gid == "G3" else ""), ""]
            rows = [[a, k, s.get("n", 0), n_(s.get("mean"), 1, True, True), n_(s.get("median"), 1, True, True),
                     n_(s.get("hit"), 0), n_(s.get("base"), 1, True, True), n_(s.get("excess"), 1, True), p_(s.get("p"))]
                    for s in st["summary"] for a, k in [(s["anchor"], s["horizon"])]]
            out += table(["From", "Horizon", "n", "Mean", "Median", "% up", "All-days mean", "Excess (pts)", "p"], rows)
            if gid == "G1":
                rows = [[d_(e["start"]), d_(e["extreme"]), d_(e["exit"]), e["days"], n_(e["extreme_value"], 0),
                         n_(e["start_3m"], 1, True, True), n_(e["start_12m"], 1, True, True), n_(e["extreme_12m"], 1, True, True),
                         n_(e["exit_12m"], 1, True, True), n_(e["mdd3m"], 1, pct=True)] for e in st["episodes"]]
                out += ["<details><summary>Episode list</summary>", ""]
                out += table(["Start", "Extreme", "Exit", "Days", "Extreme value", "3m from start", "12m from start",
                              "12m from extreme", "12m from exit", "Worst drawdown in 3m from start"], rows)
                out += ["</details>", ""]
    out += ["### Sensitivity to the zone threshold (G1, from episode start)", ""]
    rows = [[t["zone"], t["level"], t["episodes"], n_(t["excess_3m"], 1, True), p_(t["p_3m"]), n_(t["excess_12m"], 1, True),
             p_(t["p_12m"])] for t in res["thresholds"]]
    out += table(["Zone", "Threshold", "Episodes", "Excess 3m (pts)", "p 3m", "Excess 12m (pts)", "p 12m"], rows)
    out += ["### Average S&P 500 return by band (every day, not just episodes)", ""]
    for label, rows_ in res["band_tables"].items():
        out += [f"**{label}**" + (f" (from {d_(rel_from)})" if "reliable" in label else ""), ""]
        out += table(["Band", "Range", "Days", "Next 1m", "Next 3m", "Next 6m", "Next 12m", "% up 12m", "10%+ drawdown in 3m %"],
                     [[r["band"], r["range"], r["days"], n_(r["1m"], 2, True, True), n_(r["3m"], 2, True, True),
                       n_(r["6m"], 2, True, True), n_(r["12m"], 2, True, True), n_(r["hit12m"], 0), n_(r["dd10"], 0)] for r in rows_])

    # 3. Events
    ev = res["events"]
    out += ["## 3. Named events", "",
            "Dates are from the public record (config/market_events.yaml); peaks and troughs are moved to the closing "
            f"extreme within {SNAP} trading days when the data disagree (shown). Equity events read G1 against the "
            "S&P 500; AI events read G3 against SMH. Lows count as a hit when the gauge reached Extreme Fear within "
            f"{NEAR} days; tops when it reached Extreme Greed, or the fragility overlay reached 80, in the "
            f"{LOOKBACK} trading days before. Shocks are not graded: the column shows whether and when the gauge hit "
            "Extreme Fear, and what buying then returned.", ""]
    rows = []
    for r in ev["rows"]:
        rd = r["readings"]
        rows.append([d_(r["trade_date"]) + (f" (list: {d_(r['moved_from'])})" if r["moved_from"] is not None else ""), r["name"],
                     r["kind"], f"{r['gauge']} {n_(r['reading'], 0)} {r['band'] or ''}".strip(), n_(r["coverage"], 0),
                     n_(r["change_1m"], 0, True), n_(rd.get("G2"), 0), n_(r["fragility"], 0), r.get("signal", "—"),
                     {"yes": "✅", "partial": "◐", "no": "❌"}.get(r.get("hit"), "—"),
                     n_(r["ret_1m"], 1, True, True), n_(r["ret_3m"], 1, True, True), n_(r["ret_6m"], 1, True, True),
                     n_(r["ret_12m"], 1, True, True), n_(r["mdd3m"], 1, pct=True),
                     n_(r.get("signal_3m"), 1, True, True), n_(r.get("signal_12m"), 1, True, True)])
    out += table(["Date", "Event", "Kind", "Reading", "Cov %", "1m change", "G2", "Fragility", "What the gauge did", "Hit",
                  "Next 1m", "Next 3m", "Next 6m", "Next 12m", "Worst DD in 3m", "3m after fear signal",
                  "12m after fear signal"], rows)
    if ev["correlation"]:
        out += ["Correlation of the gauge reading on the event day with the return that followed, across events:", ""]
        out += table(["Market · horizon", "n", "Pearson r", "Spearman ρ"],
                     [[k.replace("_", " · "), c["n"], n_(c["pearson"], 2), n_(c["spearman"], 2)] for k, c in ev["correlation"].items()])

    # 4. Strategies
    out += ["## 4. Strategies", "",
            "Each rule sets the share of the portfolio in the asset from the gauge at a close and trades at the next "
            "close; the rest earns the fed funds rate. No costs here (see 5.4). _Matched_ is a constant mix of the asset "
            "and cash at the strategy's own average exposure, rebalanced daily: beating it means the timing added "
            "return, not just less or more risk.", ""]
    rows = []
    for key, r in runs.items():
        m, b, mm = r["m"], r["m_bh"], r["m_matched"]
        rows.append([r["name"], r["rule"], f"{d_(m['start'])} → {d_(m['end'])}", n_(r["exposure"], 0), n_(m["cagr"], 2, pct=True),
                     n_(b["cagr"], 2, pct=True), n_(mm["cagr"], 2, pct=True), n_(m["sharpe"], 2), n_(b["sharpe"], 2),
                     n_(m["max_dd"], 1, pct=True), n_(b["max_dd"], 1, pct=True), n_(m["vol"], 1, pct=True),
                     n_(r["turnover"], 0), n_(r["win_years"], 0), n_(r["win_years_matched"], 0)])
    out += table(["Strategy", "Rule", "Sample", "Avg exposure %", "CAGR", "Buy-hold CAGR", "Matched CAGR", "Sharpe",
                  "Buy-hold Sharpe", "Max DD", "Buy-hold max DD", "Vol", "Turnover %/yr", "Years beating buy-hold %",
                  "Years beating matched %"], rows)

    # 5. Robustness
    ch = res["checks"]
    out += ["## 5. Robustness checks", "", "### 5.1 Parameter grids", "",
            "Every combination below was run; a rule that only works at one setting is a warning sign.", ""]
    for key, g in ch["grids"].items():
        out += [f"**{names.get(key, key)}**: {g['n']} settings; Sharpe above buy-and-hold in {n_(g['share_sharpe_up'], 0)}%; "
                f"CAGR above the matched mix in {n_(g['share_beat_matched'], 0)}%; shallower max drawdown in "
                f"{n_(g['share_shallower_dd'], 0)}%. Sharpe difference: median {n_(g['median_d_sharpe'], 3, True)}, range "
                f"{n_(g['min_d_sharpe'], 3, True)} to {n_(g['max_d_sharpe'], 3, True)}.", ""]
        params = [k for k in g["rows"][0] if k in ("hi", "back", "trim", "lo", "hold", "base", "lower", "upper", "floor")] if g["rows"] else []
        out += ["<details><summary>All settings</summary>", ""]
        out += table(params + ["Avg exposure %", "Sharpe", "Δ Sharpe vs buy-hold", "Δ CAGR vs matched (pts)", "Δ max DD (pts)"],
                     [[x[p] for p in params] + [n_(x["exposure"], 0), n_(x["sharpe"], 2), n_(x["d_sharpe"], 3, True),
                                                n_(x["d_cagr_matched"], 2, True), n_(x["d_mdd"], 1, True)] for x in g["rows"]])
        out += ["</details>", ""]
    out += ["### 5.2 Sub-periods", "", "Each period is evaluated on its own (buy-and-hold over the same days).", ""]
    out += table(["Period", "Strategy", "CAGR", "Buy-hold CAGR", "Sharpe", "Buy-hold Sharpe", "Δ CAGR vs matched", "Max DD",
                  "Buy-hold max DD", "Years beating buy-hold %"],
                 [[r["period"], names.get(r["key"], r["key"]), n_(r["cagr"], 2, pct=True), n_(r["cagr_bh"], 2, pct=True),
                   n_(r["sharpe"], 2), n_(r["sharpe_bh"], 2), n_(r["d_cagr_matched"], 2, True), n_(r["max_dd"], 1, pct=True),
                   n_(r["max_dd_bh"], 1, pct=True), n_(r["win_years"], 0)] for r in ch["subperiods"]])
    for title, key, unit in (("5.3 Execution delay", "delay", "trading days from signal to trade"),
                             ("5.4 Trading costs", "cost", "bps per 100% of the portfolio traded")):
        out += [f"### {title}", "", f"Δ Sharpe vs buy-and-hold / Δ CAGR vs the matched mix, by {unit}.", ""]
        vals = sorted({r[key] for r in ch[key if key == "delay" else "costs"]})
        rows_ = ch[key if key == "delay" else "costs"]
        keys = list(dict.fromkeys(r["key"] for r in rows_))
        out += table(["Strategy"] + [str(v) for v in vals],
                     [[names.get(k, k)] + [next((f"{n_(r['d_sharpe'], 3, True)} / {n_(r['d_cagr_matched'], 2, True)}"
                                                 for r in rows_ if r["key"] == k and r[key] == v), "—") for v in vals] for k in keys])
    vc = ch["variants"]
    out += ["### 5.5 Signal variants and leave-one-pillar-out", "",
            f"G1 rebuilt from its stored pillars matches the dashboard series to within {n_(vc['reconstruction_error'], 4)} "
            "points, so the leave-one-pillar-out rows differ only by the pillar removed.", ""]
    out += table(["Variant", "Corr with G1", "IC with next 3m", "Extreme Fear episodes", "Excess 3m after fear (pts)",
                  "Tilt Δ Sharpe", "Tilt Δ CAGR vs matched", "Greed trim Δ Sharpe", "Greed trim Δ CAGR vs matched"],
                 [[r["variant"], n_(r["corr_with_g1"], 3), n_(r["ic_fwd3m"], 3), r["fear_episodes"], n_(r["fear_excess_3m"], 1, True),
                   n_(r["tilt_d_sharpe"], 3, True), n_(r["tilt_d_cagr_matched"], 2, True), n_(r["greed_trim_d_sharpe"], 3, True),
                   n_(r["greed_trim_d_cagr_matched"], 2, True)] for r in vc["rows"]])
    out += ["### 5.6 Other assets", "", "The G1 rules applied to other US equity exposures.", ""]
    out += table(["Asset", "Strategy", "Buy-hold Sharpe", "Δ Sharpe", "Δ CAGR vs matched", "Δ max DD (pts)"],
                 [[r["asset"], names.get(r["key"], r["key"]), n_(r["sharpe_bh"], 2), n_(r["d_sharpe"], 3, True),
                   n_(r["d_cagr_matched"], 2, True), n_(r["d_mdd"], 1, True)] for r in ch["assets"]])
    out += ["### 5.7 Placebo: circular shift", "",
            f"The strategy's own exposure path is slid {YEAR}+ days along the calendar ({res['reps']:,} random shifts, "
            "wrapping at the ends), which keeps its average level, persistence and turnover but breaks its timing. "
            "_p_ is the share of shifted paths that did at least as well. A small p means the timing, not the exposure "
            "level, earned the result.", ""]
    out += table(["Strategy", "Sharpe", "Placebo median", "Placebo 95th pct", "p (Sharpe)", "CAGR", "Placebo median CAGR",
                  "p (CAGR)"],
                 [[names.get(k, k), n_(p.get("sharpe"), 2), n_(p.get("null_sharpe_median"), 2), n_(p.get("null_sharpe_p95"), 2),
                   p_(p.get("p_sharpe")), n_(p.get("cagr"), 2, pct=True), n_(p.get("null_cagr_median"), 2, pct=True),
                   p_(p.get("p_cagr"))] for k, p in ch["placebo"].items() if p])

    # 6. Correlations
    out += ["## 6. Correlations", "",
            "Pearson r and Spearman ρ between each composite's daily reading and what followed. 95% intervals come "
            "from a moving-block bootstrap with blocks as long as the horizon (overlapping windows make naive intervals "
            "far too narrow); _t_ uses non-overlapping windows only. Sentiment gauges (G1, G3, CNN replica) are "
            "expected to correlate negatively with later returns if contrarian; health gauges (G2, G4) positively; "
            "fragility negatively with later drawdowns (drawdowns are negative numbers, so a deeper one is lower).", ""]
    rows = [[r["signal"], ASSET_LABEL.get(r["asset"], r["asset"]), r["target"], r["sample"], r["n"], n_(r["pearson"], 3),
             f"{n_(r['pearson_ci'][0], 2)} to {n_(r['pearson_ci'][1], 2)}", n_(r["spearman"], 3),
             f"{n_(r['spearman_ci'][0], 2)} to {n_(r['spearman_ci'][1], 2)}", r["n_indep"], n_(r["t_indep"], 1)]
            for r in res["corr"]["rows"]]
    out += table(["Signal", "Asset", "Target", "Sample", "Days", "Pearson r", "95% CI", "Spearman ρ", "95% CI",
                  "Independent windows", "t"], rows)
    mx = res["corr"]["matrix"]
    cols = list(mx)
    out += ["Correlation between the composites' levels (and VIX):", ""]
    out += table([""] + cols, [[a] + [n_(mx[a].get(b), 2) for b in cols] for a in cols])

    # 7. Scenarios
    sc_ = res["scenarios"]
    out += ["## 7. Scenarios", "", "### 7.1 Stress replays", "",
            "Each strategy through every bear market and correction above, and over the 12 months after the low.", ""]
    keys = ["bh"] + [k for k in runs if runs[k]["asset"] == "SPY"]
    out += table(["Window", "Fall"] + [f"{names[k]} (return / max DD / next 12m)" for k in keys],
                 [[f"{r['type']} {d_(r['start'])} → {d_(r['end'])}", n_(r["depth"], 1, pct=True)]
                  + [f"{n_(r.get(k + '_ret'), 1, True, True)} / {n_(r.get(k + '_dd'), 1, pct=True)} / "
                     f"{n_(r.get(k + '_next12m'), 1, True, True)}" for k in keys] for r in sc_["stress"]])
    an = sc_["analogs"]
    now = an["now"]
    out += ["### 7.2 Today's historical analogs", "",
            f"Today ({d_(now['date'])}): G1 {n_(now['G1'], 1)} ({now['g1_band']}), G2 {n_(now['G2'], 1)} ({now['g2_band']}), "
            f"fragility {n_(now['fragility'], 1)}. The {an['k']} closest reliable days (standardized distance on those "
            "three readings, at least a month apart, at least a year old) and what the S&P 500 did next:", ""]
    out += table(["Horizon", "Analogs: mean", "median", "10th pct", "90th pct", "% up", "All reliable days: mean", "% up"],
                 [[k, n_(v["analogs"].get("mean"), 1, True, True), n_(v["analogs"].get("median"), 1, True, True),
                   n_(v["analogs"].get("p10"), 1, True, True), n_(v["analogs"].get("p90"), 1, True, True),
                   n_(v["analogs"].get("hit"), 0), n_(v["all"].get("mean"), 1, True, True), n_(v["all"].get("hit"), 0)]
                  for k, v in an["summary"].items()])
    out += [f"A 10%+ drawdown within 3 months followed {n_(an['dd10_analogs'], 0)}% of analogs vs {n_(an['dd10_all'], 0)}% of all reliable days.", "",
            "<details><summary>Closest analogs</summary>", ""]
    out += table(["Date", "G1", "G2", "Fragility", "Distance", "Next 3m", "Next 12m", "Worst DD in 3m"],
                 [[d_(r["date"]), n_(r["G1"], 0), n_(r["G2"], 0), n_(r["fragility"], 0), n_(r["distance"], 2),
                   n_(r["ret_3m"], 1, True, True), n_(r["ret_12m"], 1, True, True), n_(r["mdd3m"], 1, pct=True)] for r in an["rows"]])
    out += ["</details>", ""]
    gr = sc_["grid"]
    out += ["### 7.3 What-if: G1 band × G2 band", "",
            "Mean S&P 500 return over the next 12 months (next 3 months and days in the cell in brackets), on reliable "
            "days. Today's cell is marked ★. Cells with fewer than 60 days are blank.", ""]
    rows = []
    for i, g1name in enumerate(gr["g1_scale"]):
        row = [f"G1 {g1name}"]
        for j in range(5):
            c = next(c for c in gr["cells"] if c["g1"] == i and c["g2"] == j)
            mark = " ★" if gr["now"] == [i, j] else ""
            row.append("—" + mark if c["days"] < 60 else f"{n_(c['ret_12m'], 1, True, True)} ({n_(c['ret_3m'], 1, True, True)}; {c['days']}){mark}")
        rows.append(row)
    out += table([""] + [f"G2 {s}" for s in gr["g2_scale"]], rows)

    # 8. Monte Carlo
    mc = res["mc"]
    ms = mc["strategies"]
    out += ["## 8. Monte Carlo", "", "### 8.1 Strategies over 10-year resampled histories", "",
            f"{ms['paths']:,} stationary block bootstrap paths (mean block {ms['block']} trading days) of {ms['years']} years, "
            f"resampling whole days so each strategy and buy-and-hold see the same days. Sample {d_(ms['start'])} → {d_(ms['end'])}.", ""]
    out += table(["Strategy", "CAGR 5th / 50th / 95th", "Max DD 5th / 50th / 95th", "Sharpe 50th", "P(10y loss)",
                  "P(beats buy-hold CAGR)", "P(beats buy-hold Sharpe)", "P(shallower max DD)"],
                 [[names.get(r["key"], r["key"]), f"{n_(r['cagr']['p5'], 1)} / {n_(r['cagr']['p50'], 1)} / {n_(r['cagr']['p95'], 1)}%",
                   f"{n_(r['mdd']['p5'], 1)} / {n_(r['mdd']['p50'], 1)} / {n_(r['mdd']['p95'], 1)}%", n_(r["sharpe"]["p50"], 2),
                   n_(r["p_loss"], 1, pct=True), n_(r["p_beat_cagr"], 0, pct=True), n_(r["p_beat_sharpe"], 0, pct=True),
                   n_(r["p_shallower_dd"], 0, pct=True)] for r in ms["rows"]])
    out += ["### 8.2 The next 12 months from today's reading", "",
            "Twelve one-month blocks of real history. Each block starts on a day whose gauge band matched the simulated "
            "state; the next state is the band history reached at the block's end. The unconditional version draws "
            "from every day. This is a description of past paths from similar readings, not a forecast.", ""]
    rows = []
    for key, f in mc["fans"].items():
        if not f or "final" not in f:
            continue
        rows.append([f"{f['asset']} · {'from ' + f['gauge'] + ' ' + f['state'] if f['gauge'] != 'none' else 'unconditional'}",
                     f"{f['paths']:,}", f["start_pool"], n_(f["final"]["p5"], 1, True, True), n_(f["final"]["p25"], 1, True, True),
                     n_(f["final"]["p50"], 1, True, True), n_(f["final"]["p75"], 1, True, True), n_(f["final"]["p95"], 1, True, True),
                     n_(f["p_loss"], 0, pct=True), n_(f["p_dd10"], 0, pct=True), n_(f["p_dd20"], 0, pct=True)])
    out += table(["Simulation", "Paths", "Starting pool (days)", "5th", "25th", "Median", "75th", "95th", "P(loss)",
                  "P(10%+ DD)", "P(20%+ DD)"], rows)
    out += ["The SMH rows rest on G3, which carries hindsight bias and low coverage.", ""]
    out += correlation_summary(res)
    return "\n".join(out) + "\n"


def correlation_summary(res: dict) -> list[str]:
    rows_ = res["corr"]["rows"]

    def daily(signal: str, target: str, sample: str = "all") -> list:
        r = next((x for x in rows_ if x["signal"] == signal and x["target"] == target and x["sample"] == sample), None)
        if r is None:
            return [f"{signal} vs next {target} ({ASSET_LABEL.get('SMH' if signal in ('G3', 'G4') else '^GSPC')}), daily",
                    "—", "—", "—", "—"]
        asset = ASSET_LABEL.get(r["asset"], r["asset"])
        return [f"{signal} vs next {target} ({asset}), every day from {d_(r['start'])}", f"{r['n']:,}", n_(r["pearson"], 3),
                f"{n_(r['pearson_ci'][0], 2)} to {n_(r['pearson_ci'][1], 2)}", n_(r["spearman"], 3)]

    lines = [daily("G1", f"{k} return") for k in H] + [daily("G1", "3m max drawdown")]
    lines += [daily("G2", "3m return"), daily("G2", "12m return"), daily("G2", "3m max drawdown"),
              daily("Fragility", "3m max drawdown"), daily("CNN replica", "3m return"), daily("G3", "3m return"),
              daily("G4", "3m max drawdown")]
    for k, c in res["events"]["correlation"].items():
        market, h = k.split("_")
        lines.append([f"{'G1' if market == 'equity' else 'G3'} on named {market} event days vs next {h}", c["n"],
                      n_(c["pearson"], 3), "—", n_(c["spearman"], 3)])
    for k, c in res["zones"]["G1"].get("correlation", {}).items():
        lines.append([f"G1 at the start of each Extreme Fear / Extreme Greed episode vs next {k}", c["n"],
                      n_(c["pearson"], 3), "—", n_(c["spearman"], 3)])
    hc = headline_corr(res)
    out = ["## 9. Correlation coefficients", "",
           "The coefficients from sections 2, 3 and 6 in one place. Negative for G1 means contrarian (greed before "
           "weaker returns). For drawdown targets, positive means a higher reading came before a shallower drawdown.", ""]
    out += table(["Pair", "n", "Pearson r", "95% CI", "Spearman ρ"], lines)
    out += [f"**Headline:** across every day since {d_(hc.get('start'))}, G1's correlation with the S&P 500's next "
            f"three months is r = {n_(hc.get('pearson'), 3)} (95% CI {n_(hc.get('pearson_ci', [math.nan] * 2)[0], 2)} to "
            f"{n_(hc.get('pearson_ci', [math.nan] * 2)[1], 2)}).", ""]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness study of the gauges; writes reports/robustness.md")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--paths", type=int, default=2000, help="Monte Carlo paths for the strategy bootstrap")
    parser.add_argument("--reps", type=int, default=1000, help="bootstrap draws for intervals and placebo shifts")
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    store = Store(args.db)
    try:
        res = study(store, args.paths, args.reps, args.seed)
    finally:
        store.close()
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(render(res), encoding="utf-8")
    OUT.write_text(json.dumps(to_json(res), separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(ROOT)} and {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
