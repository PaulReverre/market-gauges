"""Pure transforms: no I/O, unit-tested against hand-computed fixtures.

Three families:
- OPS act on one series in observation time (its own frequency), e.g. a
  12-month change of a monthly series.
- COMBINES act on several inputs aligned on the daily trading calendar.
- Panel and scoring helpers (breadth, McClellan, basket index, options,
  XBRL quarters, percentiles, staleness).
Percent outputs are in percent units (5.0 means 5%).
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Single-series ops (observation time)
# --------------------------------------------------------------------------- #


def identity(s: pd.Series) -> pd.Series:
    return s


def rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def pct_from_ma(s: pd.Series, window: int) -> pd.Series:
    return (s / rolling_mean(s, window) - 1) * 100


def change(s: pd.Series, periods: int, scale: float = 1.0) -> pd.Series:
    return s.diff(periods) * scale


def pct_change(s: pd.Series, periods: int) -> pd.Series:
    return (s / s.shift(periods) - 1) * 100


def avg_pct_change(s: pd.Series, window: int, periods: int) -> pd.Series:
    return pct_change(rolling_mean(s, window), periods)


def zscore(s: pd.Series, window: int, smooth: int = 1, clip: float = 3.0) -> pd.Series:
    """z of the (optionally smoothed) value vs its trailing window, winsorized at ±clip."""
    x = rolling_mean(s, smooth) if smooth > 1 else s
    mean = x.rolling(window, min_periods=max(window // 2, 2)).mean()
    std = x.rolling(window, min_periods=max(window // 2, 2)).std()
    return ((x - mean) / std.replace(0, np.nan)).clip(-clip, clip)


OPS: dict[str, Callable[..., pd.Series]] = {
    "identity": identity, "rolling_mean": rolling_mean, "pct_from_ma": pct_from_ma, "change": change,
    "pct_change": pct_change, "avg_pct_change": avg_pct_change, "zscore": zscore,
}

# --------------------------------------------------------------------------- #
# Combines (inputs aligned on the trading calendar, columns in input order)
# --------------------------------------------------------------------------- #


def _cols(df: pd.DataFrame, n: int) -> list[pd.Series]:
    if df.shape[1] != n:
        raise ValueError(f"expected {n} inputs, got {df.shape[1]}")
    return [df.iloc[:, i] for i in range(n)]


def c_identity(df: pd.DataFrame) -> pd.Series:
    return _cols(df, 1)[0]


def c_ratio(df: pd.DataFrame, scale: float = 1.0) -> pd.Series:
    a, b = _cols(df, 2)
    return a / b.replace(0, np.nan) * scale


def c_diff(df: pd.DataFrame) -> pd.Series:
    a, b = _cols(df, 2)
    return a - b


def c_mean(df: pd.DataFrame) -> pd.Series:
    return df.mean(axis=1, skipna=True)


def c_share(df: pd.DataFrame) -> pd.Series:
    """First input as a percent of the two inputs' sum (e.g. negative / (negative + positive) guidance)."""
    a, b = _cols(df, 2)
    return a / (a + b).replace(0, np.nan) * 100


def c_pct_diff(df: pd.DataFrame) -> pd.Series:
    """Percent premium of the first input over the second (e.g. P/E vs its 10-year average)."""
    a, b = _cols(df, 2)
    return (a / b.replace(0, np.nan) - 1) * 100


def c_ratio_change(df: pd.DataFrame, periods: int) -> pd.Series:
    return pct_change(c_ratio(df), periods)


def c_diff_max(df: pd.DataFrame, window: int, scale: float = 1.0) -> pd.Series:
    return c_diff(df).rolling(window, min_periods=window).max() * scale


def c_diff_mean(df: pd.DataFrame, window: int) -> pd.Series:
    return rolling_mean(c_diff(df), window)


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Annualized close-to-close volatility in percent."""
    returns = np.log(close / close.shift(1))
    return returns.rolling(window, min_periods=window).std() * math.sqrt(252) * 100


def c_vrp(df: pd.DataFrame, window: int) -> pd.Series:
    vix, spx = _cols(df, 2)
    return vix - realized_vol(spx, window)


def c_return_spread(df: pd.DataFrame, periods: int) -> pd.Series:
    a, b = _cols(df, 2)
    return pct_change(a, periods) - pct_change(b, periods)


def c_earnings_yield_gap(df: pd.DataFrame) -> pd.Series:
    """Forward earnings yield (100 / forward P/E) minus the 10y real yield."""
    pe, real = _cols(df, 2)
    return 100 / pe.replace(0, np.nan) - real


def c_net_liquidity_change(df: pd.DataFrame, periods: int) -> pd.Series:
    """(WALCL - WTREGEN) in $mn minus RRP in $bn, as a change in $bn."""
    walcl, tga, rrp = _cols(df, 3)
    net_bn = (walcl - tga) / 1000 - rrp
    return net_bn.diff(periods)


def c_sum_avg_pct_change(df: pd.DataFrame, window: int, periods: int) -> pd.Series:
    """Sum the inputs (rows where all are present), then a window-average percent change."""
    total = df.dropna().sum(axis=1)
    return avg_pct_change(total, window, periods)


COMBINES: dict[str, Callable[..., pd.Series]] = {
    "identity": c_identity, "ratio": c_ratio, "diff": c_diff, "mean": c_mean, "ratio_change": c_ratio_change,
    "diff_max": c_diff_max, "diff_mean": c_diff_mean, "vrp": c_vrp, "return_spread": c_return_spread,
    "earnings_yield_gap": c_earnings_yield_gap, "net_liquidity_change": c_net_liquidity_change,
    "sum_avg_pct_change": c_sum_avg_pct_change, "share": c_share, "pct_diff": c_pct_diff,
}
# These combine monthly inputs on their shared observation dates, not daily.
OBS_TIME_COMBINES = {"sum_avg_pct_change"}

# --------------------------------------------------------------------------- #
# Breadth panels (closes/volumes: dates x tickers)
# --------------------------------------------------------------------------- #


def pct_above_ma(closes: pd.DataFrame, window: int) -> pd.Series:
    ma = closes.rolling(window, min_periods=window).mean()
    valid = ma.notna() & closes.notna()
    above = (closes > ma) & valid
    return (above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)) * 100


def net_highs_lows(closes: pd.DataFrame, window: int, smooth: int) -> pd.Series:
    hi = closes.rolling(window, min_periods=window).max()
    lo = closes.rolling(window, min_periods=window).min()
    valid = hi.notna() & closes.notna()
    net = ((closes >= hi) & valid).sum(axis=1) - ((closes <= lo) & valid).sum(axis=1)
    net = net.where(valid.sum(axis=1) > 0)
    return rolling_mean(net.astype(float), smooth)


def mcclellan_summation(closes: pd.DataFrame, volumes: pd.DataFrame) -> pd.Series:
    """Ratio-adjusted (RANA) McClellan summation on advancing vs declining volume."""
    moves = closes.diff()
    adv = volumes.where(moves > 0).sum(axis=1)
    dec = volumes.where(moves < 0).sum(axis=1)
    total = (adv + dec).replace(0, np.nan)
    rana = ((adv - dec) / total * 1000).iloc[1:]
    oscillator = rana.ewm(alpha=0.10, adjust=False).mean() - rana.ewm(alpha=0.05, adjust=False).mean()
    return oscillator.cumsum().reindex(closes.index)


# --------------------------------------------------------------------------- #
# Basket helpers
# --------------------------------------------------------------------------- #


def usd_returns(closes: pd.DataFrame, fx_per_usd: dict[str, pd.Series]) -> pd.DataFrame:
    """Daily returns in USD; fx_per_usd maps ticker -> local currency per USD (e.g. KRW=X)."""
    r = closes.pct_change(fill_method=None)
    for ticker, fx in fx_per_usd.items():
        if ticker in r:
            fx = fx.reindex(closes.index).ffill()
            r[ticker] = (1 + r[ticker]) * (fx.shift(1) / fx) - 1
    return r


def basket_index(returns: pd.DataFrame, layers: dict[str, tuple[float, list[str]]]) -> pd.Series:
    """Equal weight within a layer, fixed layer weights; layers missing a day are renormalized."""
    layer_r, layer_w = {}, {}
    for name, (weight, tickers) in layers.items():
        cols = [t for t in tickers if t in returns]
        layer_r[name] = returns[cols].mean(axis=1, skipna=True)
        layer_w[name] = weight
    lr = pd.DataFrame(layer_r)
    w = pd.Series(layer_w)
    live = lr.notna()
    daily = (lr.fillna(0) * w).sum(axis=1) / (live * w).sum(axis=1).replace(0, np.nan)
    daily = daily.dropna()
    return (1 + daily).cumprod()


def event_window_mean(events: pd.DataFrame, calendar: pd.DatetimeIndex, days: int, min_events: int) -> pd.Series:
    """Mean of event values (columns date, value) over a trailing calendar-day window, on each calendar date.

    Used for quarterly company events (earnings releases): a value counts from its
    date until `days` later; days with fewer than `min_events` events are NaN.
    """
    if events.empty:
        return pd.Series(np.nan, index=calendar)
    daily = events.groupby("date")["value"].agg(["sum", "count"])
    daily = daily.reindex(calendar.union(daily.index), fill_value=0.0)
    window = daily.rolling(f"{days}D").sum()
    return (window["sum"] / window["count"]).where(window["count"] >= min_events).reindex(calendar)


def drawdown(s: pd.Series, window: int = 252) -> pd.Series:
    return (s / s.rolling(window, min_periods=min(window, 20)).max() - 1) * 100


def avg_pairwise_corr(returns: pd.DataFrame, window: int, step: int = 5) -> pd.Series:
    """Mean off-diagonal correlation of the trailing window, sampled every `step` rows."""
    out = {}
    for i in range(window, len(returns) + 1, step):
        block = returns.iloc[i - window:i].dropna(axis=1, thresh=int(window * 0.8))
        if block.shape[1] < 3:
            continue
        c = block.corr().to_numpy()
        n = c.shape[0]
        out[returns.index[i - 1]] = (np.nansum(c) - n) / (n * (n - 1))
    return pd.Series(out, dtype=float)


# --------------------------------------------------------------------------- #
# Options (one snapshot: columns expiry, option_type, strike, implied_vol, ...)
# --------------------------------------------------------------------------- #


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def bs_delta(spot: float, strike: np.ndarray, years: float, vol: np.ndarray, kind: str) -> np.ndarray:
    """Black-Scholes delta with zero rates and dividends."""
    d1 = (np.log(spot / strike) + 0.5 * vol ** 2 * years) / (vol * math.sqrt(years))
    nd1 = _norm_cdf(d1)
    return nd1 if kind == "call" else nd1 - 1


def _clean_chain(chain: pd.DataFrame) -> pd.DataFrame:
    c = chain.dropna(subset=["implied_vol", "strike"])
    return c[(c["implied_vol"] > 0.01) & (c["implied_vol"] < 3.0)]


def _interp_variance(points: list[tuple[float, float]], target_days: float) -> float | None:
    """Interpolate implied vol in total variance across expiries (days, vol)."""
    if not points:
        return None
    points = sorted(points)
    below = [p for p in points if p[0] <= target_days]
    above = [p for p in points if p[0] >= target_days]
    if not below or not above:
        return min(points, key=lambda p: abs(p[0] - target_days))[1]
    (d0, v0), (d1, v1) = below[-1], above[0]
    if d1 == d0:
        return v0
    w0 = v0 ** 2 * d0
    w1 = v1 ** 2 * d1
    w = w0 + (w1 - w0) * (target_days - d0) / (d1 - d0)
    return math.sqrt(w / target_days)


def atm_iv(chain: pd.DataFrame, spot: float, snapshot: pd.Timestamp, target_days: int = 30) -> float | None:
    """30-day ATM IV in vol points: mean call/put IV at the strike nearest spot, per expiry."""
    c = _clean_chain(chain)
    points = []
    for expiry, g in c.groupby("expiry"):
        days = (pd.Timestamp(expiry) - snapshot).days
        if days < 7:
            continue
        strike = g.loc[(g["strike"] - spot).abs().idxmin(), "strike"]
        iv = g.loc[g["strike"] == strike, "implied_vol"].mean()
        points.append((days, iv))
    v = _interp_variance(points, target_days)
    return None if v is None else v * 100


def risk_reversal(chain: pd.DataFrame, spot: float, snapshot: pd.Timestamp, target_days: int = 30,
                  delta: float = 0.25) -> float | None:
    """25-delta call IV minus 25-delta put IV (vol points) at the expiry nearest target_days."""
    c = _clean_chain(chain)
    expiries = sorted({(pd.Timestamp(e) - snapshot).days for e in c["expiry"]} - set(range(0, 7)))
    if not expiries:
        return None
    days = min(expiries, key=lambda d: abs(d - target_days))
    g = c[(pd.to_datetime(c["expiry"]) - snapshot).dt.days == days]
    years = days / 365
    ivs = {}
    for kind, target in (("call", delta), ("put", -delta)):
        side = g[g["option_type"] == kind].sort_values("strike")
        if len(side) < 2:
            return None
        d = bs_delta(spot, side["strike"].to_numpy(), years, side["implied_vol"].to_numpy(), kind)
        order = np.argsort(d)
        ivs[kind] = float(np.interp(target, d[order], side["implied_vol"].to_numpy()[order]))
    return (ivs["call"] - ivs["put"]) * 100


# --------------------------------------------------------------------------- #
# XBRL: quarterly values from year-to-date filings
# --------------------------------------------------------------------------- #


def quarters_from_ytd(facts: pd.DataFrame) -> pd.DataFrame:
    """Turn duration facts (period_start, period_end, value, filed) into discrete quarters.

    Uses the first-filed value for each period (point in time). A quarter is a
    ~3-month fact, or a year-to-date fact minus the previous year-to-date fact
    with the same start. Returns period_end, value, filed (when the quarter was known).
    """
    f = facts.dropna(subset=["period_start"]).sort_values("filed")
    f = f.drop_duplicates(subset=["period_start", "period_end"], keep="first")
    f = f.assign(days=(pd.to_datetime(f["period_end"]) - pd.to_datetime(f["period_start"])).dt.days)
    out = {}
    for _, row in f[(f["days"] > 75) & (f["days"] < 105)].iterrows():
        out[row["period_end"]] = (row["value"], row["filed"])
    for start, chain in f.groupby("period_start"):
        chain = chain.sort_values("period_end")
        prev = None
        for _, row in chain.iterrows():
            if prev is not None and row["period_end"] not in out:
                gap = (pd.Timestamp(row["period_end"]) - pd.Timestamp(prev["period_end"])).days
                if 75 < gap < 105:
                    out[row["period_end"]] = (row["value"] - prev["value"], max(row["filed"], prev["filed"]))
            prev = row
    q = pd.DataFrame([(e, v, d) for e, (v, d) in out.items()], columns=["period_end", "value", "filed"])
    return q.sort_values("period_end").reset_index(drop=True)


def calendar_quarter(d) -> pd.Timestamp:
    """Map a fiscal period end to the nearest calendar quarter end (ORCL's May 31 -> Jun 30)."""
    d = pd.Timestamp(d)
    ends = [pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) for y in (d.year - 1, d.year, d.year + 1)
            for m in (3, 6, 9, 12)]
    return min(ends, key=lambda e: abs((e - d).days))


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #


def rolling_percentile(s: pd.Series, window: int | None, min_periods: int) -> pd.Series:
    """Percentile (0-100) of each value within its trailing window, ties averaged, centered.

    window=None uses all history to date (expanding). Values are ranked only
    against earlier and same-day values, so the result is point-in-time.
    """
    x = s.dropna()
    roller = x.expanding(min_periods=min_periods) if window is None else x.rolling(window, min_periods=min_periods)
    rank = roller.rank(method="average")
    count = roller.count()
    return ((rank - 0.5) / count * 100).reindex(s.index)


def staleness_multiplier(age_bdays: np.ndarray | pd.Series, freq_bdays: int) -> np.ndarray:
    """1 up to 1.5x the expected age, falling linearly to 0 at 3x."""
    age = np.asarray(age_bdays, dtype=float)
    lo, hi = 1.5 * freq_bdays, 3.0 * freq_bdays
    return np.clip((hi - age) / (hi - lo), 0.0, 1.0)


def piecewise(x: float, points: list[tuple[float, float]]) -> float:
    """Linear interpolation through (x, score) breakpoints, flat beyond the ends."""
    xs, ys = zip(*sorted(points))
    return float(np.interp(x, xs, ys))


def band(score: float, labels: list[str]) -> str:
    """0-20, 20-40, 40-60, 60-80, 80-100."""
    return labels[min(int(score // 20), 4)]


def ema(s: pd.Series, span: int = 3) -> pd.Series:
    return s.ewm(span=span, adjust=False, ignore_na=True).mean()
