"""Long-horizon layer: what valuation says about the next 10 years.

Two models, each a one-variable regression of the S&P 500's subsequent 10-year
annualized nominal total return (Shiller monthly data) on a valuation measure:

  equity allocation  Aggregate investor allocation to equities, from the Fed's Z.1
                     accounts: market value of US equities / (equities + all debt
                     owed by households, non-financial business, governments and
                     the rest of the world). Quarterly since 1951.
  CAPE               Shiller's cyclically adjusted P/E.

Both are slow, both are useless for timing the next year, and both are fitted
on overlapping windows, so their error bars are wider than the fit suggests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUITY = ["NCBEILQ027S", "FBCELLQ027S"]
DEBT = ["TCMILBSNNCB", "TCMILBSHNO", "FGTCMDODNS", "SLGTCMDODNS", "WCMITCMFODNS"]
Z1 = EQUITY + DEBT
FIT_FROM = "1951-01-01"
YEARS = 10


def total_return_index(sh: pd.DataFrame) -> pd.Series:
    p, d = sh["price"], sh["dividend"].ffill()
    return ((p + d / 12) / p.shift(1)).fillna(1).cumprod()


def forward_ann(ix: pd.Series, years: int = YEARS) -> pd.Series:
    return ((ix.shift(-12 * years) / ix) ** (1 / years) - 1) * 100


def _fit(x: pd.Series, y: pd.Series) -> dict:
    d = pd.concat([x, y], axis=1, sort=True).dropna()
    b, a = np.polyfit(d.iloc[:, 0], d.iloc[:, 1], 1)
    resid = d.iloc[:, 1] - (a + b * d.iloc[:, 0])
    r = float(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1])
    return {"a": float(a), "b": float(b), "r2": round(r * r, 2), "se": round(float(resid.std()), 1), "n": int(len(d)),
            "x_min": float(d.iloc[:, 0].min()), "x_max": float(d.iloc[:, 0].max())}


def _out_of_sample(x: pd.Series, y: pd.Series, min_obs: int = 240) -> dict:
    """Refit at each date using only 10-year windows that had finished by then."""
    preds = {}
    for t in x.dropna().index[::3]:
        cut = t - pd.DateOffset(years=YEARS)
        d = pd.concat([x.loc[:cut], y.loc[:cut]], axis=1, sort=True).dropna()
        if len(d) >= min_obs:
            b, a = np.polyfit(d.iloc[:, 0], d.iloc[:, 1], 1)
            preds[t] = a + b * x[t]
    f = pd.Series(preds)
    d = pd.concat([f.rename("f"), y.rename("y")], axis=1, sort=True).dropna()
    if len(d) < 20:
        return {}
    naive = y.expanding().mean().shift(12 * YEARS).reindex(d.index)
    return {"corr": round(float(d.f.corr(d.y)), 2), "mae": round(float((d.f - d.y).abs().mean()), 1),
            "mae_naive": round(float((naive - d.y).abs().mean()), 1), "from": d.index[0].date().isoformat(),
            "n": int(len(d))}


def build(sh: pd.DataFrame, z1: pd.DataFrame, spx: pd.Series, dgs10: pd.Series | None) -> dict:
    sh = sh.resample("MS").last()
    tr = total_return_index(sh)
    y = forward_ann(tr)
    hist_avg = round(float(y.loc[FIT_FROM:].mean()), 1)
    spx_m = spx.resample("MS").last()
    models, charts = [], {}

    # ---- equity allocation (quarter stamped at its first month; public about 10 weeks after it ends)
    z = z1[Z1].dropna()
    eq, debt = z[EQUITY].sum(axis=1), z[DEBT].sum(axis=1)
    alloc = eq / (eq + debt) * 100
    q_last = alloc.index[-1]
    q_end = q_last + pd.DateOffset(months=3) - pd.DateOffset(days=1)
    spx_qe = spx.loc[:q_end].iloc[-1]
    eq_now = eq.iloc[-1] * float(spx.iloc[-1] / spx_qe)
    alloc_now = float(eq_now / (eq_now + debt.iloc[-1]) * 100)
    x_alloc = alloc.shift(freq=pd.DateOffset(months=6)).reindex(sh.index, method="ffill", limit=4).loc[FIT_FROM:]
    fit = _fit(x_alloc, y)
    pred = fit["a"] + fit["b"] * alloc_now
    models.append({
        "id": "alloc", "name": "Investor equity allocation", "unit": "%", "nd": 1,
        "value": round(alloc_now, 1), "official": round(float(alloc.iloc[-1]), 1),
        "official_as_of": f"Q{(q_last.month - 1) // 3 + 1} {q_last.year}",
        "nowcast": True, "pct": round(float((alloc <= alloc_now).mean() * 100), 1),
        "hist_max": round(float(alloc.iloc[:-1].max()), 1), "hist_max_date": alloc.iloc[:-1].idxmax().date().isoformat(),
        "extrapolated": bool(alloc_now > fit["x_max"]),
        "pred": round(float(pred), 1), "r2": fit["r2"], "se": fit["se"], "since": 1951,
        "oos": _out_of_sample(x_alloc, y),
        "what": "Share of all US financial assets (stocks plus every dollar of debt) held as stocks. From the Fed's "
                "quarterly Z.1 accounts; moved to today with the S&P 500's change since quarter end.",
        "why": "The best long-horizon predictor tested: fitted out of sample it tracked subsequent 10-year returns "
               "far better than the historical average did. High allocation means little new buying power is left.",
        "source": "FRED (Fed Z.1)"})
    q = alloc.loc[FIT_FROM:]
    fitted = (fit["a"] + fit["b"] * q)
    realized = y.reindex(q.index + pd.DateOffset(months=6), method="nearest", tolerance=pd.Timedelta(days=20))
    charts["alloc"] = {"dates": [d.date().isoformat() for d in q.index],
                       "value": [round(float(v), 1) for v in q.values],
                       "pred": [round(float(v), 1) for v in fitted.values],
                       "realized": [None if not np.isfinite(v) else round(float(v), 1) for v in realized.values]}

    # ---- CAPE (monthly; latest month scaled to today's index level)
    cape = sh["cape"].dropna()
    cape_now = float(cape.iloc[-1] * spx.iloc[-1] / sh["price"].reindex(cape.index).iloc[-1])
    x_cape = np.log(cape).loc[FIT_FROM:]
    fit = _fit(x_cape, y)
    pred_c = fit["a"] + fit["b"] * np.log(cape_now)
    models.append({
        "id": "cape", "name": "Shiller CAPE", "unit": "x", "nd": 1,
        "value": round(cape_now, 1), "official": round(float(cape.iloc[-1]), 1),
        "official_as_of": cape.index[-1].strftime("%b %Y"), "nowcast": True,
        "pct": round(float((cape <= cape_now).mean() * 100), 1),
        "hist_max": round(float(cape.max()), 1), "hist_max_date": cape.idxmax().date().isoformat(),
        "extrapolated": bool(np.log(cape_now) > fit["x_max"]),
        "pred": round(float(pred_c), 1), "r2": fit["r2"], "se": fit["se"], "since": 1951,
        "oos": _out_of_sample(np.log(cape).loc["1881":], y),
        "what": "Price divided by 10 years of inflation-adjusted earnings. Monthly since 1881; scaled to today's "
                "index level.",
        "why": "The standard valuation yardstick. A weaker predictor than equity allocation (it explains about "
               "half as much), kept as a second opinion.",
        "source": "shillerdata.com"})
    c = cape.loc[FIT_FROM:].iloc[::3]
    charts["cape"] = {"dates": [d.date().isoformat() for d in c.index], "value": [round(float(v), 1) for v in c.values],
                      "pred": [round(float(fit["a"] + fit["b"] * np.log(v)), 1) for v in c.values],
                      "realized": [None if not np.isfinite(v) else round(float(v), 1) for v in y.reindex(c.index).values]}

    preds = [m["pred"] for m in models]
    out = {
        "as_of": spx.index[-1].date().isoformat(), "horizon_years": YEARS,
        "expected": {"mid": round(float(np.mean(preds)), 1), "low": round(min(preds), 1), "high": round(max(preds), 1)},
        "hist_avg": hist_avg,
        "treasury10": None if dgs10 is None or dgs10.dropna().empty else round(float(dgs10.dropna().iloc[-1]), 2),
        "models": models, "charts": charts,
    }
    mid = out["expected"]["mid"]
    out["zone"] = ("Very poor" if mid < 2 else "Poor" if mid < 5 else "Below average" if mid < 8
                   else "Average" if mid < 12 else "Good" if mid < 15 else "Very good")
    return out
