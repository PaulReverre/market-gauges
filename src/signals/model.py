"""Scoring and evidence.

A gauge is a handful of indicators, each ranked against its own history using
only data available on the day (expanding percentile), averaged within pillars,
averaged across pillars, and ranked again. 0 = extreme fear, 100 = extreme
complacency. Everything the dashboard says about "what happened next" is
recomputed here on every run from the full history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = {21: "1m", 63: "3m", 126: "6m", 252: "12m"}
MIN_HISTORY = 504            # trading days before an indicator (or the composite) gets a percentile
EPISODE_GAP = 21             # zone visits closer than this many trading days are one episode
SPLIT = "2010-01-01"         # split-half stability check
ZONES = [                    # key, label, lower bound (inclusive), upper bound (exclusive)
    ("buy", "Extreme fear", 0, 10),
    ("fear", "Fear", 10, 25),
    ("neutral", "Neutral", 25, 75),
    ("calm", "Calm", 75, 90),
    ("caution", "Complacent", 90, 100.001),
]
SEED = 20261109


# ------------------------------------------------------------ primitives
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def iv_percentile(s: pd.Series, n: int = 252) -> pd.Series:
    """Share of the prior n sessions that closed below today's value, in percent."""
    return s.rolling(n + 1).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True) * 100


def iv_rank(s: pd.Series, n: int = 252) -> pd.Series:
    lo, hi = s.rolling(n).min(), s.rolling(n).max()
    return (s - lo) / (hi - lo) * 100


def realized_vol(px: pd.Series, n: int = 20) -> pd.Series:
    return np.log(px).diff().rolling(n).std() * np.sqrt(252) * 100


def expanding_pct(s: pd.Series, minp: int = MIN_HISTORY) -> pd.Series:
    """Point-in-time percentile (0-1): today's rank among all values known up to today."""
    return s.dropna().expanding(minp).rank(pct=True)


def forward_returns(px: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({h: (px.shift(-h) / px - 1) * 100 for h in HORIZONS})


def zone_of(score: float) -> dict:
    for key, label, lo, hi in ZONES:
        if lo <= score < hi:
            return {"key": key, "label": label, "lo": lo, "hi": min(hi, 100)}
    return {"key": "neutral", "label": "Neutral", "lo": 25, "hi": 75}


def count_episodes(mask: np.ndarray, gap: int = EPISODE_GAP) -> int:
    idx = np.flatnonzero(mask)
    return 0 if len(idx) == 0 else 1 + int((np.diff(idx) > gap).sum())


# -------------------------------------------------------------- composite
def composite(fear_pct: pd.DataFrame, pillars: dict[str, list[str]], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """fear_pct: per-indicator point-in-time fear percentiles (0-1, high = fear).

    Returns score (0-100, low = fear), raw (mean fear percentile) and n (indicators live).
    """
    p = fear_pct.reindex(calendar).ffill(limit=5)
    pm = pd.DataFrame({k: p[[c for c in cols if c in p]].mean(axis=1, skipna=True) for k, cols in pillars.items()})
    raw = pm.mean(axis=1, skipna=True)
    used = [c for cols in pillars.values() for c in cols if c in p]
    n = p[used].notna().sum(axis=1)
    raw = raw.where(n >= 2)
    score = 100 - expanding_pct(raw) * 100
    out = pd.DataFrame({"raw": raw, "n": n, "score": score.reindex(calendar)})
    for k in pm:
        out[f"pillar_{k}"] = 100 - pm[k] * 100
    return out


# --------------------------------------------------------------- evidence
def _stats(y: pd.Series) -> dict | None:
    y = y.dropna()
    if len(y) < 5:
        return None
    return {"n": int(len(y)), "mean": round(float(y.mean()), 1), "median": round(float(y.median()), 1),
            "hit": round(float((y > 0).mean() * 100)), "p10": round(float(y.quantile(0.1)), 1),
            "p90": round(float(y.quantile(0.9)), 1)}


def odds_table(score: pd.Series, px: pd.Series) -> dict:
    """What the benchmark did after every day the score sat in each zone."""
    fw = forward_returns(px)
    d = pd.concat([score.rename("s"), fw], axis=1, sort=True).dropna(subset=["s"])
    zones = []
    for key, label, lo, hi in ZONES:
        m = (d.s >= lo) & (d.s < hi)
        zones.append({"key": key, "label": label, "lo": lo, "hi": min(hi, 100), "days": int(m.sum()),
                      "episodes": count_episodes(m.values),
                      "h": {HORIZONS[h]: _stats(d.loc[m, h]) for h in HORIZONS}})
    return {"since": d.index[0].date().isoformat(), "days": int(len(d)),
            "zones": zones, "base": {HORIZONS[h]: _stats(d[h]) for h in HORIZONS}}


def _shift_p(x: np.ndarray, y: np.ndarray, mask_fn, sign: float, n: int = 400, minshift: int = 252) -> float | None:
    """One-sided p-value for a tail's excess forward return, from circular shifts of the signal.

    Shifting keeps both series' autocorrelation (overlapping windows, clustered
    signals) and only breaks their alignment, so it does not overstate significance.
    """
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 1500 or mask_fn(x).sum() < 20:
        return None
    base = y.mean()
    obs = sign * (y[mask_fn(x)].mean() - base)
    rng = np.random.default_rng(SEED)
    sims = np.array([sign * (y[mask_fn(np.roll(x, k))].mean() - base)
                     for k in rng.integers(minshift, len(x) - minshift, n)])
    return round(float((sims >= obs).mean()), 3)


def tail_evidence(level: pd.Series, px: pd.Series, lo_cut: float, hi_cut: float, horizons=(21, 63, 126, 252)) -> dict:
    """Excess forward return after low readings (<= lo_cut) and high readings (>= hi_cut) of `level`.

    `level` is always oriented like a gauge score: 0 = fear, 100 = complacency.
    """
    fw = forward_returns(px)
    d = pd.concat([level.rename("x"), fw], axis=1, sort=True).dropna(subset=["x"])
    out = {}
    for side, mask_fn, sign in (("low", lambda v: v <= lo_cut, 1.0), ("high", lambda v: v >= hi_cut, 1.0)):
        m = mask_fn(d.x.values)
        res = {"days": int(m.sum()), "episodes": count_episodes(m), "h": {}}
        for h in horizons:
            y = d[h]
            sub = y[m].dropna()
            if len(sub) < 5:
                continue
            pre, post = d.loc[:SPLIT], d.loc[SPLIT:]
            def ex(part):
                mm = mask_fn(part.x.values)
                yy = part[h]
                return None if mm.sum() < 5 or yy[mm].dropna().empty else round(float(yy[mm].mean() - yy.mean()), 1)
            res["h"][HORIZONS[h]] = {
                "mean": round(float(sub.mean()), 1), "hit": round(float((sub > 0).mean() * 100)),
                "excess": round(float(sub.mean() - y.mean()), 1), "base_mean": round(float(y.mean()), 1),
                "base_hit": round(float((y.dropna() > 0).mean() * 100)),
                "pre": ex(pre), "post": ex(post)}
        out[side] = res
    # significance at 3 months; "low" is tested for better-than-average returns, "high" for worse
    out["low"]["p"] = _shift_p(d.x.values, d[63].values, lambda v: v <= lo_cut, 1.0)
    out["high"]["p"] = _shift_p(d.x.values, d[63].values, lambda v: v >= hi_cut, -1.0)
    return out


def grade(side: dict, want: str) -> str:
    """strong | moderate | weak | none, from the 3-month p-value and split-half agreement.

    want = "above": the tail should beat the average; "below": it should lag it.
    """
    h = side.get("h", {}).get("3m")
    p = side.get("p")
    if not h or p is None:
        return "none"
    sgn = 1 if want == "above" else -1
    halves = [v for v in (h.get("pre"), h.get("post")) if v is not None]
    agree = bool(halves) and all(sgn * v > 0 for v in halves)
    if sgn * h["excess"] <= 0:
        return "none"
    if p <= 0.02 and agree:
        return "strong"
    if p <= 0.10 and agree:
        return "moderate"
    return "weak"


def split_odds(score: pd.Series, px: pd.Series, flag: pd.Series, cut: float = 10) -> list[dict]:
    """Buy-zone days split by a yes/no condition (`flag`), with what the benchmark did next in each half."""
    fw = forward_returns(px)
    d = pd.concat([score.rename("s"), flag.rename("f"), fw], axis=1, sort=True).dropna(subset=["s", "f"])
    out = []
    for key, m in (("yes", (d.s <= cut) & d.f.astype(bool)), ("no", (d.s <= cut) & ~d.f.astype(bool))):
        row = {"key": key, "days": int(m.sum()), "episodes": count_episodes(m.values), "h": {}}
        for h, lab in HORIZONS.items():
            y = d.loc[m, h].dropna()
            row["h"][lab] = None if len(y) < 5 else {"mean": round(float(y.mean()), 1), "hit": round(float((y > 0).mean() * 100))}
        out.append(row)
    return out


def episodes(score: pd.Series, px: pd.Series, cut: float, below: bool = True) -> list[dict]:
    """Every visit to a zone, with what the benchmark did from the first day in it."""
    s = score.dropna()
    m = (s <= cut) if below else (s >= cut)
    idx = np.flatnonzero(m.values)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > EPISODE_GAP)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    pos = {d: i for i, d in enumerate(px.index)}
    out = []
    for a, b in zip(starts, ends):
        seg = s.iloc[a:b + 1]
        start, end = s.index[a], s.index[b]
        ext = seg.idxmin() if below else seg.idxmax()
        i0 = pos.get(start)
        row = {"start": start.date().isoformat(), "end": end.date().isoformat(),
               "days": int(((seg <= cut) if below else (seg >= cut)).sum()),
               "extreme": round(float(seg.min() if below else seg.max()), 1), "extreme_date": ext.date().isoformat()}
        if i0 is not None:
            p0 = px.iloc[i0]
            for h, lab in HORIZONS.items():
                row[lab] = round(float(px.iloc[i0 + h] / p0 - 1) * 100, 1) if i0 + h < len(px) else None
            win = px.iloc[i0:i0 + 127]
            row["worst"] = round(float(win.min() / p0 - 1) * 100, 1)
            row["open"] = bool(b == len(s) - 1)
        out.append(row)
    return out


# ------------------------------------------------------------- utilities
def thin(index: pd.DatetimeIndex, keep_daily: int = 520) -> pd.DatetimeIndex:
    """Chart axis: daily for the last ~2 years, weekly (Fridays' last bar) before that."""
    if len(index) <= keep_daily:
        return index
    old = pd.Series(index[:-keep_daily], index=index[:-keep_daily]).resample("W-FRI").last().dropna()
    return pd.DatetimeIndex(old.values).append(index[-keep_daily:])


def clean(v):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.date().isoformat()
    return v


def series_list(s: pd.Series, axis: pd.DatetimeIndex, nd: int = 2) -> list:
    vals = s.reindex(axis)
    return [None if not np.isfinite(v) else round(float(v), nd) for v in vals.values]
