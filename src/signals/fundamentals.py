"""Fundamentals gauge: what valuation says about the next 5 to 10 years.

Every candidate is judged the same way, on monthly data since 1951:

  fit           one-variable regression of the S&P 500's subsequent 10-year annualized
                nominal total return (Shiller data) on the measure
  out of sample the same regression refitted at each date using only 10-year windows
                that had already finished; its average miss is compared with simply
                assuming the historical average return
  split half    the relation must have the same sign before and after 1988

Measures that beat the average out of sample with the same sign in both halves are
scored. The gauge's 0-100 score is the mean of their percentiles (0 = cheapest
reading to date, 100 = most expensive); the expected return is the mean of their
five separate forecasts, which out of sample was as accurate as the best single
measure and does not depend on any one series.

Valuation says almost nothing about the next year. Ten-year windows overlap, so
about seven independent decades stand behind every number here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from signals import model as M

EQUITY = ["NCBEILQ027S", "FBCELLQ027S"]
DEBT = ["TCMILBSNNCB", "TCMILBSHNO", "FGTCMDODNS", "SLGTCMDODNS", "WCMITCMFODNS"]
Z1 = EQUITY + DEBT
# gross value added, net worth, net equity issuance, net dividends, profits after tax (all nonfinancial
# corporate business), economy-wide profits after tax, GDP, 10-year Treasury (monthly)
FRED = ["A455RC1Q027SBEA", "TNWMVBSNNCB", "NCBCEBQ027S", "BOGZ1FA106121075Q", "NFCPATAX", "CP", "GDP", "GS10"]
FIT_FROM = "1951-01-01"
SPLIT = "1988-01-01"
HZ = {1: "1y", 3: "3y", 5: "5y", 10: "10y"}
MIN_RANK = 240          # months of history before a measure gets a percentile
ZONE_LABEL = {"buy": "Very cheap", "fear": "Cheap", "neutral": "Fair", "calm": "Expensive", "caution": "Very expensive"}
UNTESTED = [
    {"name": "Forward P/E", "why": "Analyst estimates are proprietary (FactSet, LSEG). No free source has more than a few "
                                   "years of history, and a 10-year test needs decades."},
    {"name": "Free cash flow yield", "why": "No free index-level series exists. Building one from 500 companies' filings "
                                            "would only reach back to about 2009."},
    {"name": "Forward EPS growth and earnings revisions", "why": "Same problem as forward P/E: estimate history is "
                                                                 "proprietary. Trailing earnings growth was tested "
                                                                 "instead and has no relation to later returns."},
]


@dataclass
class Metric:
    id: str
    name: str
    unit: str            # "", "%", "x", "pts"
    nd: int
    exp: str             # "high": a high reading means expensive; "low": a low reading does
    role: str            # scored | context
    what: str
    why: str
    source: str
    series: pd.Series = field(repr=False, default=None)   # monthly, dated when it was public
    now: float | None = None                               # today's estimate
    official: float | None = None
    official_as_of: str = ""
    log: bool = False


def total_return_index(sh: pd.DataFrame) -> pd.Series:
    p, d = sh["price"], sh["dividend"].ffill()
    return ((p + d / 12) / p.shift(1)).fillna(1).cumprod()


def forward_returns(tr: pd.Series) -> pd.DataFrame:
    """Subsequent annualized nominal total return over each horizon, in percent."""
    return pd.DataFrame({lab: ((tr.shift(-12 * n) / tr) ** (1 / n) - 1) * 100 for n, lab in HZ.items()})


def _fit(x: pd.Series, y: pd.Series) -> dict | None:
    d = pd.concat([x, y], axis=1, sort=True).dropna()
    if len(d) < 120 or d.iloc[:, 0].std() == 0:
        return None
    b, a = np.polyfit(d.iloc[:, 0], d.iloc[:, 1], 1)
    r = float(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1])
    resid = d.iloc[:, 1] - (a + b * d.iloc[:, 0])
    return {"a": float(a), "b": float(b), "r2": r * r, "se": float(resid.std()), "n": int(len(d)),
            "x_min": float(d.iloc[:, 0].min()), "x_max": float(d.iloc[:, 0].max())}


def out_of_sample(x: pd.Series, y: pd.Series, years: int = 10, min_obs: int = 240) -> pd.Series:
    """Forecast at each date from a fit on windows that had finished by then (every third month)."""
    preds = {}
    for t in x.dropna().index[::3]:
        cut = t - pd.DateOffset(years=years)
        d = pd.concat([x.loc[:cut], y.loc[:cut]], axis=1, sort=True).dropna()
        if len(d) >= min_obs:
            b, a = np.polyfit(d.iloc[:, 0], d.iloc[:, 1], 1)
            preds[t] = a + b * x[t]
    return pd.Series(preds, dtype=float)


def record(forecast: pd.Series, y: pd.Series, years: int = 10) -> dict | None:
    """How a forecast series did against the outcome, and against assuming the average return to date."""
    naive = y.expanding().mean().shift(12 * years)
    d = pd.concat([forecast.rename("f"), y.rename("y"), naive.rename("n")], axis=1, sort=True).dropna()
    if len(d) < 30:
        return None
    return {"corr": round(float(d.f.corr(d.y)), 2), "mae": round(float((d.f - d.y).abs().mean()), 1),
            "mae_naive": round(float((d.n - d.y).abs().mean()), 1), "from": d.index[0].date().isoformat(),
            "n": int(len(d))}


def grade(signed_r2: float | None, pre: float | None, post: float | None, rec: dict | None) -> str:
    """strong | moderate | weak | none. signed_r2 > 0 means expensive readings preceded lower returns."""
    if signed_r2 is None or rec is None or signed_r2 <= 0:
        return "none"
    agree = pre is not None and post is not None and pre > 0 and post > 0
    if agree and rec["mae"] <= 0.8 * rec["mae_naive"] and rec["corr"] >= 0.5:
        return "strong"
    if agree and rec["mae"] < rec["mae_naive"]:
        return "moderate"
    return "weak" if signed_r2 >= 0.1 else "none"


def metrics(sh: pd.DataFrame, z1: pd.DataFrame, fr: pd.DataFrame, spx: pd.Series, dgs10: pd.Series | None) -> list[Metric]:
    """Every candidate as a monthly series plus today's estimate. `sh` is monthly Shiller data."""
    idx = pd.date_range(sh.index[0], spx.index[-1].to_period("M").to_timestamp(), freq="MS")
    px_now = float(spx.iloc[-1])

    def q(s: pd.Series) -> pd.Series:      # quarterly accounts: public about 10 weeks after the quarter ends
        return s.dropna().shift(freq=pd.DateOffset(months=6)).reindex(idx, method="ffill", limit=5)

    def q_label(s: pd.Series) -> str:
        t = s.dropna().index[-1]
        return f"Q{(t.month - 1) // 3 + 1} {t.year}"

    def since_quarter(s: pd.Series) -> float:   # S&P 500 change since the end of the latest official quarter
        end = s.dropna().index[-1] + pd.DateOffset(months=3) - pd.DateOffset(days=1)
        return px_now / float(spx.loc[:end].iloc[-1])

    mv = z1["NCBEILQ027S"].dropna()
    eq, debt = z1[EQUITY].dropna().sum(axis=1), z1[DEBT].dropna().sum(axis=1)
    f = lambda c: fr[c].dropna() if c in fr else pd.Series(dtype=float)  # noqa: E731
    gva, nw, issue, divs, nfc_pat, cp, gdp = (f(c) for c in FRED[:7])
    gs10 = f("GS10").reindex(idx).ffill(limit=3)
    gs10_now = float(dgs10.dropna().iloc[-1]) if dgs10 is not None and not dgs10.dropna().empty else float(gs10.dropna().iloc[-1])
    e_lag = sh["earnings"].reindex(idx).shift(4)        # reported earnings reach the file a few months late
    e_last, d_last = float(sh["earnings"].dropna().iloc[-1]), float(sh["dividend"].dropna().iloc[-1])
    price = sh["price"].reindex(idx)
    cape = sh["cape"].reindex(idx)
    cape_m = sh["cape"].dropna()
    cape_now = float(cape_m.iloc[-1] * px_now / sh["price"].reindex(cape_m.index).iloc[-1])
    cpi = sh["cpi"].reindex(idx).ffill(limit=3)
    infl10 = ((cpi / cpi.shift(120)) ** (1 / 10) - 1) * 100
    k = since_quarter(mv)
    alloc = eq / (eq + debt) * 100
    eq_now = float(eq.iloc[-1]) * k
    ps, buff, tq = mv / (gva * 1000), mv / (gdp * 1000) * 100, mv / nw
    payout = ((divs - issue).rolling(4).mean() / mv * 100)
    buyback = (-issue.rolling(4).mean() / mv * 100)
    margin, prof = nfc_pat / gva * 100, cp / gdp * 100
    eps_g = (e_lag / cpi.shift(4)).pct_change(12) * 100
    last = lambda s: float(s.dropna().iloc[-1])  # noqa: E731
    ql = q_label(mv)
    sh_month = sh["cape"].dropna().index[-1].strftime("%b %Y")
    z1src, bea = "FRED (Fed Z.1 accounts)", "FRED (Fed Z.1 and BEA)"
    return [
        Metric("alloc", "Investor equity allocation", "%", 1, "high", "scored", source=z1src,
               what="Share of all US financial assets (stocks plus every dollar of debt) that is held as stocks.",
               why="The best single predictor tested. Out of sample it missed the next decade's return by a little "
                   "over half as much as assuming the average. High allocation means little buying power is left.",
               series=q(alloc), now=eq_now / (eq_now + float(debt.iloc[-1])) * 100, official=last(alloc), official_as_of=ql),
        Metric("tobin_q", "Tobin's Q", "x", 2, "high", "scored", source=z1src, log=True,
               what="Market value of nonfinancial companies' shares divided by their net worth at replacement cost.",
               why="Prices relative to what the businesses would cost to rebuild. The second-smallest out-of-sample "
                   "miss after equity allocation, though its fit is much weaker since 1988 than before.",
               series=q(tq), now=last(tq) * k, official=last(tq), official_as_of=ql),
        Metric("ecy", "Equity risk premium (excess CAPE yield)", "pts", 1, "low", "scored",
               source="shillerdata.com, FRED",
               what="Shiller's measure: the CAPE earnings yield minus the real 10-year Treasury yield. What stocks "
                    "offer over bonds.",
               why="The equity risk premium on your list, in the one form that works: it held in both halves and "
                   "beats the average out of sample. The simpler Fed-model version (below) does not.",
               series=100 / cape - (gs10 - infl10), now=100 / cape_now - (gs10_now - last(infl10)),
               official=last(100 / cape - (gs10 - infl10)), official_as_of=sh_month),
        Metric("cape", "Shiller CAPE", "x", 1, "high", "scored", source="shillerdata.com", log=True,
               what="Price divided by ten years of inflation-adjusted earnings.",
               why="The standard yardstick, and the steadiest: almost the same strength before and after 1988.",
               series=cape, now=cape_now, official=last(sh["cape"]), official_as_of=sh_month),
        Metric("payout", "Shareholder yield", "%", 1, "low", "scored", source=z1src,
               what="Dividends plus net buybacks of nonfinancial companies over four quarters, as a percent of their "
                    "market value. Net issuance of new shares counts against it.",
               why="Passes, more modestly than the others. Dividend yield alone fails (companies switched to "
                   "buybacks) and buybacks alone fail; the sum is what holds up.",
               series=q(payout), now=last(payout) / k, official=last(payout), official_as_of=ql),
        # ---- tested, not scored
        Metric("pe", "Trailing P/E", "x", 1, "high", "context", source="shillerdata.com", log=True,
               what="Price divided by the last twelve months of reported earnings.",
               why="Barely better than assuming the average, and weak since 1988: one year of earnings is too noisy "
                   "(it collapses in recessions, making stocks look expensive at the bottom). CAPE fixes that.",
               series=price / e_lag, now=px_now / e_last, official=last(price / e_lag), official_as_of=sh_month),
        Metric("ps", "Price to sales", "x", 2, "high", "context", source=bea, log=True,
               what="Market value of nonfinancial companies divided by their gross value added, the economy-wide "
                    "analogue of price-to-sales.",
               why="Ranks valuations well but its level has drifted upward for decades, so its forecasts missed by "
                   "more than the plain average did.",
               series=q(ps), now=last(ps) * k, official=last(ps), official_as_of=ql),
        Metric("buffett", "Buffett indicator", "%", 0, "high", "context", source=bea, log=True,
               what="Market value of nonfinancial companies' shares as a percent of GDP.",
               why="Same story as price to sales (the two move together almost one for one): right direction, but "
                   "the level keeps ratcheting higher, so out of sample it did worse than the average.",
               series=q(buff), now=last(buff) * k, official=last(buff), official_as_of=ql),
        Metric("dy", "Dividend yield", "%", 2, "low", "context", source="shillerdata.com",
               what="Twelve months of S&P 500 dividends divided by the price.",
               why="Ranked valuations well, but its forecasts missed by more than the average: payouts moved from "
                   "dividends to buybacks. Shareholder yield replaces it.",
               series=sh["dividend"].reindex(idx).ffill(limit=6) / price * 100, now=d_last / px_now * 100,
               official=last(sh["dividend"].reindex(idx).ffill(limit=6) / price * 100), official_as_of=sh_month),
        Metric("fed", "Earnings yield minus Treasury yield", "pts", 1, "low", "context", source="shillerdata.com, FRED",
               what="The Fed model: trailing earnings yield minus the 10-year Treasury yield, both nominal.",
               why="Rejected. No relation to 10-year returns, and out of sample it pointed the wrong way: it compares "
                   "a real yield with a nominal one.",
               series=e_lag / price * 100 - gs10, now=e_last / px_now * 100 - gs10_now,
               official=last(e_lag / price * 100 - gs10), official_as_of=sh_month),
        Metric("buyback", "Net buyback yield", "%", 2, "low", "context", source=z1src,
               what="Net share repurchases of nonfinancial companies over four quarters, as a percent of market value.",
               why="Rejected on its own: no usable relation to later returns. It earns its keep only inside "
                   "shareholder yield.",
               series=q(buyback), now=last(buyback) / k, official=last(buyback), official_as_of=ql),
        Metric("margin", "Net profit margin", "%", 1, "high", "context", source="FRED (BEA)",
               what="After-tax profits of nonfinancial companies as a percent of their gross value added.",
               why="Rejected. The idea is that fat margins revert and drag returns down; in the data, margins had no "
                   "relation to later returns at any horizon.",
               series=q(margin), now=last(margin), official=last(margin), official_as_of=q_label(margin)),
        Metric("profits", "Corporate profits to GDP", "%", 1, "high", "context", source="FRED (BEA)",
               what="Economy-wide after-tax corporate profits as a percent of GDP.",
               why="Rejected for the same reason as margins: no relation to later returns.",
               series=q(prof), now=last(prof), official=last(prof), official_as_of=q_label(prof)),
        Metric("eps_g", "Trailing EPS growth (real)", "%", 0, "high", "context", source="shillerdata.com",
               what="Change in inflation-adjusted S&P 500 earnings over the past year. The free stand-in for forward "
                    "growth and revisions, which have no free history.",
               why="Rejected. Past earnings growth had no relation to returns over the next 1, 3, 5 or 10 years.",
               series=eps_g, now=last(eps_g), official=last(eps_g), official_as_of=sh_month),
    ]


def _x(m: Metric, s: pd.Series) -> pd.Series:
    return np.log(s.where(s > 0)) if m.log else s


def evaluate(m: Metric, Y: pd.DataFrame) -> tuple[dict, pd.Series, dict | None]:
    """Everything the dashboard shows for one measure; also its out-of-sample forecasts and its fit."""
    s = m.series.loc[FIT_FROM:]
    x = _x(m, s)
    want = -1 if m.exp == "high" else 1                       # expensive -> lower returns
    def signed(part_x, h="10y"):
        f = _fit(part_x, Y[h])
        return None if f is None else round(f["r2"] * (1 if np.sign(f["b"]) == want else -1), 2)
    fit = _fit(x, Y["10y"])
    fc = out_of_sample(x, Y["10y"])
    rec = record(fc, Y["10y"])
    pre, post = signed(x.loc[:SPLIT]), signed(x.loc[SPLIT:])
    hist = s.dropna()
    x_now = float(np.log(m.now)) if m.log else float(m.now)
    exp_pct = float((hist <= m.now).mean() * 100) if m.exp == "high" else float((hist >= m.now).mean() * 100)
    row = {
        "id": m.id, "name": m.name, "unit": m.unit, "nd": m.nd, "exp": m.exp, "role": m.role,
        "what": m.what, "why": m.why, "source": m.source,
        "value": round(float(m.now), m.nd + 1), "official": round(float(m.official), m.nd + 1),
        "official_as_of": m.official_as_of, "pct": round(exp_pct, 1), "since": hist.index[0].year,
        "pred": None if fit is None else round(fit["a"] + fit["b"] * x_now, 1),
        "se": None if fit is None else round(fit["se"], 1),
        "r2": {h: signed(x, h) for h in HZ.values()}, "pre": pre, "post": post, "oos": rec,
        "extrapolated": bool(fit is not None and not fit["x_min"] <= x_now <= fit["x_max"]),
        "grade": grade(signed(x), pre, post, rec),
    }
    return row, fc, fit


def build(sh: pd.DataFrame, z1: pd.DataFrame, fr: pd.DataFrame, spx: pd.Series, dgs10: pd.Series | None) -> tuple[dict, dict]:
    sh = sh.resample("MS").last()
    tr = total_return_index(sh)
    Y = forward_returns(tr)
    ms = metrics(sh, z1, fr, spx, dgs10)
    idx = ms[0].series.index
    rows, fcs, fits, charts = [], {}, {}, {}
    for m in ms:
        row, fc, fit = evaluate(m, Y)
        rows.append(row)
        fcs[m.id], fits[m.id] = fc, fit
        s = m.series.loc[FIT_FROM:].dropna().iloc[::3]
        if fit is not None:
            charts[m.id] = {"dates": [d.date().isoformat() for d in s.index],
                            "value": [round(float(v), m.nd + 1) for v in s.values],
                            "pred": [round(float(fit["a"] + fit["b"] * v), 1) for v in _x(m, s).values],
                            "realized": [None if not np.isfinite(v) else round(float(v), 1) for v in Y["10y"].reindex(s.index).values]}
    scored = [m for m in ms if m.role == "scored"]
    srow = {r["id"]: r for r in rows}

    # ---- 0-100 score: mean percentile of the scored measures, 0 = cheapest reading to date
    P = {}
    for m in scored:
        s = (m.series if m.exp == "high" else -m.series).loc[FIT_FROM:].copy()
        s.iloc[-1] = m.now if m.exp == "high" else -m.now          # the current month carries today's estimate
        P[m.id] = s.dropna().expanding(MIN_RANK).rank(pct=True)
    P = pd.DataFrame(P).reindex(idx)
    score = (P.mean(axis=1, skipna=True).where(P.notna().sum(axis=1) >= 3) * 100).dropna()
    now = float(score.iloc[-1])
    zone = M.zone_of(now)
    zone["label"] = ZONE_LABEL[zone["key"]]

    # ---- what followed each valuation zone
    d = pd.concat([score.rename("s"), Y], axis=1, sort=True).dropna(subset=["s"])

    def stats(y: pd.Series) -> dict | None:
        y = y.dropna()
        if len(y) < 6:
            return None
        return {"n": int(len(y)), "median": round(float(y.median()), 1), "mean": round(float(y.mean()), 1),
                "hit": round(float((y > 0).mean() * 100)), "p10": round(float(y.quantile(.1)), 1),
                "p90": round(float(y.quantile(.9)), 1)}
    zones = []
    for key, _, lo, hi in M.ZONES:
        mask = (d.s >= lo) & (d.s < hi)
        zones.append({"key": key, "label": ZONE_LABEL[key], "lo": lo, "hi": min(hi, 100), "months": int(mask.sum()),
                      "h": {h: stats(d.loc[mask, h]) for h in HZ.values()}})
    odds = {"since": d.index[0].date().isoformat(), "months": int(len(d)), "zones": zones,
            "base": {h: stats(d[h]) for h in HZ.values()}}

    # ---- expected return: mean of the scored measures' separate forecasts
    preds = [srow[m.id]["pred"] for m in scored if srow[m.id]["pred"] is not None]
    blend_fc = pd.DataFrame({m.id: fcs[m.id].reindex(idx).ffill(limit=2) for m in scored}).mean(axis=1, skipna=False).dropna()
    blend_fit = pd.DataFrame({m.id: fits[m.id]["a"] + fits[m.id]["b"] * _x(m, m.series) for m in scored if fits[m.id]}
                             ).mean(axis=1, skipna=False).loc[FIT_FROM:].dropna()
    axis = score.index[::3]
    charts["_gauge"] = {
        "dates": [t.date().isoformat() for t in axis], "score": M.series_list(score, axis, 1),
        "estimate": M.series_list(blend_fc, axis, 1), "fitted": M.series_list(blend_fit, axis, 1),
        "realized": M.series_list(Y["10y"], axis, 1)}

    def delta(n):
        return round(now - float(score.iloc[-1 - n]), 1) if len(score) > n else None

    hist_avg = round(float(Y["10y"].loc[FIT_FROM:].mean()), 1)
    out = {
        "id": "fund", "name": "Fundamentals", "benchmark": "S&P 500", "as_of": spx.index[-1].date().isoformat(),
        "score": round(now, 1), "zone": zone, "chg": {"m1": delta(1), "y1": delta(12), "y5": delta(60)},
        "expected": {"mid": round(float(np.mean(preds)), 1), "median": round(float(np.median(preds)), 1),
                     "low": round(min(preds), 1), "high": round(max(preds), 1), "models": len(preds)},
        "record": record(blend_fc, Y["10y"]), "hist_avg": hist_avg,
        "treasury10": None if dgs10 is None or dgs10.dropna().empty else round(float(dgs10.dropna().iloc[-1]), 2),
        "now": {"zone": next(z for z in zones if z["key"] == zone["key"]), "base": odds["base"]},
        "odds": odds, "inputs": rows, "untested": UNTESTED,
        "spark": [round(float(v), 1) for v in score.iloc[-180:].values],
    }
    return out, charts
