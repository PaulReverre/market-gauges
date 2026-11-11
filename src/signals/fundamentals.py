"""Fundamentals gauge: what valuation says about the next 5 to 10 years.

Every candidate is judged the same way, on monthly data since 1951:

  fit           one-variable regression of the S&P 500's subsequent 10-year annualized
                nominal total return (Shiller data) on the measure
  out of sample the same regression refitted at each date using only 10-year windows
                that had already finished; its average miss is compared with simply
                assuming the historical average return
  split half    the relation must have the same sign before and after 1988

Measures that beat the average out of sample with the same sign in both halves are
scored. Each scored measure is read against two yardsticks:

  all history   its level, ranked against every month since 1951
  own era       its distance from its own trailing 30-year average

The second exists because every valuation model has under-predicted returns since
2010: profit margins, buybacks and index composition moved the "normal" level, and
a measure pinned at its all-time high carries no more information than "highest
ever". Out of sample the two yardsticks are about equally accurate (the era one
slightly better on the years both cover), so the gauge uses both: its 0-100 score
is the mean of the ten ranks and its expected return the mean of the ten forecasts.

Tested and rejected as ways to make the gauge less bearish (see the README):
crediting strong earnings growth, margins or calm credit; correcting forecasts by
their own recent error; Damodaran's implied equity risk premium.

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
ERA = 360               # months in a measure's "own era" average
ERA_MIN = 180           # months needed before that average exists
ERA_RANK = 120          # months of era readings before they are ranked
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


def metrics(sh: pd.DataFrame, z1: pd.DataFrame, fr: pd.DataFrame, spx: pd.Series, dgs10: pd.Series | None,
            dam: pd.DataFrame | None = None) -> list[Metric]:
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
    out = [
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
    if dam is not None and not dam.empty and {"erp", "tbond"} <= set(dam.columns):
        ier = (dam["erp"] + dam["tbond"]).dropna()
        ier_m = ier.reindex(idx.union(ier.index)).ffill(limit=13).reindex(idx)     # annual until 2008, then monthly
        out.append(Metric(
            "dam", "Implied expected return (Damodaran)", "%", 1, "low", "context", source="Aswath Damodaran, NYU Stern",
            what="The return the market is priced to deliver: Damodaran's implied equity risk premium plus the "
                 "10-year Treasury yield. It is forward-looking, built on analysts' earnings growth estimates and "
                 "current cash returned to shareholders.",
            why="The forward-looking measure with real history (annual from 1961). It fits the full sample, but "
                "out of sample since 1992 it has not beaten assuming the average return, so it is shown rather "
                "than scored. It is the main number behind the view that stocks are fairly priced.",
            series=ier_m, now=float(dam["erp"].dropna().iloc[-1]) + gs10_now, official=float(ier.iloc[-1]),
            official_as_of=ier.index[-1].strftime("%b %Y")))
    return out


def _x(m: Metric, s: pd.Series) -> pd.Series:
    return np.log(s.where(s > 0)) if m.log else s


def _era(x: pd.Series) -> pd.Series:
    """A measure's own-era average: its trailing 30-year mean (15 years needed)."""
    return x.rolling(ERA, min_periods=ERA_MIN).mean()


def evaluate(m: Metric, Y: pd.DataFrame, era: bool = False) -> tuple[dict, dict, dict]:
    """Everything the dashboard shows for one measure, its out-of-sample forecasts and its fits.

    With era=True the measure is also tested as a distance from its own trailing 30-year average.
    """
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
        "era": None,
    }
    fcs, fits = {"hist": fc}, {"hist": fit}
    if era:
        mu = _era(x)
        dev = (x - mu).dropna()
        fit_e = _fit(dev, Y["10y"])
        if fit_e is not None and len(dev) > ERA_RANK:
            fc_e = out_of_sample(dev, Y["10y"])
            dev_now = x_now - float(mu.dropna().iloc[-1])
            rank = float((dev <= dev_now).mean() * 100) if m.exp == "high" else float((dev >= dev_now).mean() * 100)
            avg = float(np.exp(mu.dropna().iloc[-1])) if m.log else float(mu.dropna().iloc[-1])
            row["era"] = {"avg": round(avg, m.nd + 1), "pct": round(rank, 1), "pred": round(fit_e["a"] + fit_e["b"] * dev_now, 1),
                          "r2": signed(dev), "oos": record(fc_e, Y["10y"])}
            fcs["era"], fits["era"] = fc_e, fit_e
    return row, fcs, fits


def support_check(score: pd.Series, Y: pd.DataFrame, eps_g: pd.Series, baa: pd.Series | None, cut: float = 75) -> dict | None:
    """Did sound fundamentals change what followed expensive markets?

    'Supported' = real earnings higher than a year earlier AND the Baa credit spread not up more
    than 0.3 points in three months. Splits every expensive month (score >= cut) on that.
    """
    if baa is None or baa.dropna().empty:
        return None
    widen = baa.dropna().resample("MS").last().diff(3).reindex(score.index).ffill(limit=2)
    g = eps_g.reindex(score.index)
    ok = (g > 0) & (widen <= 0.3)
    known = g.notna() & widen.notna()
    d = pd.concat([score.rename("s"), ok.rename("ok"), known.rename("k"), Y], axis=1, sort=True).dropna(subset=["s"])
    rows = []
    for key, mask in (("yes", (d.s >= cut) & d.k & d.ok), ("no", (d.s >= cut) & d.k & ~d.ok)):
        i = np.flatnonzero(mask.values)
        row = {"key": key, "months": int(mask.sum()), "episodes": 0 if len(i) == 0 else 1 + int((np.diff(i) > 12).sum()), "h": {}}
        for h in HZ.values():
            y = d.loc[mask, h].dropna()
            row["h"][h] = None if len(y) < 6 else {"mean": round(float(y.mean()), 1), "hit": round(float((y > 0).mean() * 100))}
        rows.append(row)
    dot = d.loc["2000-01":"2000-12"]
    return {"cut": cut, "rows": rows,
            "now": {"eps_growth": None if g.dropna().empty else round(float(g.dropna().iloc[-1]), 1),
                    "spread_chg": None if widen.dropna().empty else round(float(widen.dropna().iloc[-1]), 2),
                    "supported": bool(ok.iloc[-1]) if bool(known.iloc[-1]) else None},
            "dotcom": None if dot.empty or dot.k.sum() == 0 else {"supported_share": round(float(dot.ok[dot.k].mean() * 100)),
                                                                   "next_1y": None if dot["1y"].dropna().empty else round(float(dot["1y"].mean()), 1)}}


def build(sh: pd.DataFrame, z1: pd.DataFrame, fr: pd.DataFrame, spx: pd.Series, dgs10: pd.Series | None,
          baa: pd.Series | None = None, dam: pd.DataFrame | None = None) -> tuple[dict, dict]:
    sh = sh.resample("MS").last()
    tr = total_return_index(sh)
    Y = forward_returns(tr)
    ms = metrics(sh, z1, fr, spx, dgs10, dam)
    idx = ms[0].series.index
    rows, fcs, fits, charts = [], {}, {}, {}
    for m in ms:
        row, fc, fit = evaluate(m, Y, era=m.role == "scored")
        rows.append(row)
        fcs[m.id], fits[m.id] = fc, fit
        s = m.series.loc[FIT_FROM:].dropna().iloc[::3]
        if fit["hist"] is not None:
            f = fit["hist"]
            charts[m.id] = {"dates": [d.date().isoformat() for d in s.index],
                            "value": [round(float(v), m.nd + 1) for v in s.values],
                            "pred": [round(float(f["a"] + f["b"] * v), 1) for v in _x(m, s).values],
                            "realized": [None if not np.isfinite(v) else round(float(v), 1) for v in Y["10y"].reindex(s.index).values]}
    scored = [m for m in ms if m.role == "scored"]
    srow = {r["id"]: r for r in rows}

    # ---- 0-100 score: mean rank across the scored measures and both yardsticks (0 = cheapest to date)
    PA, PB = {}, {}
    for m in scored:
        s = (m.series if m.exp == "high" else -m.series).loc[FIT_FROM:].copy()
        s.iloc[-1] = m.now if m.exp == "high" else -m.now          # the current month carries today's estimate
        PA[m.id] = s.dropna().expanding(MIN_RANK).rank(pct=True)
        x = m.series.loc[FIT_FROM:].copy()
        x.iloc[-1] = m.now
        x = _x(m, x)
        dev = (x - _era(x)).dropna()
        PB[m.id] = (dev if m.exp == "high" else -dev).expanding(ERA_RANK).rank(pct=True)
    PA, PB = pd.DataFrame(PA).reindex(idx), pd.DataFrame(PB).reindex(idx)
    view_a = PA.mean(axis=1, skipna=True).where(PA.notna().sum(axis=1) >= 3) * 100
    view_b = PB.mean(axis=1, skipna=True).where(PB.notna().sum(axis=1) >= 3) * 100
    score = pd.concat([view_a, view_b], axis=1).mean(axis=1, skipna=True).dropna()     # all history alone until the era view exists
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

    # ---- expected return: the two yardsticks, each the mean of the scored measures' forecasts, then averaged
    pa = [srow[m.id]["pred"] for m in scored if srow[m.id]["pred"] is not None]
    pb = [srow[m.id]["era"]["pred"] for m in scored if srow[m.id]["era"]]
    views = {"history": round(float(np.mean(pa)), 1), "era": round(float(np.mean(pb)), 1) if pb else None}
    mids = [v for v in views.values() if v is not None]
    al = lambda k: pd.DataFrame({m.id: fcs[m.id][k].reindex(idx).ffill(limit=2) for m in scored if k in fcs[m.id]}).mean(axis=1, skipna=False).dropna()  # noqa: E731
    fc_a, fc_b = al("hist"), al("era")
    fc_blend = pd.concat([fc_a, fc_b], axis=1).mean(axis=1, skipna=True)
    rec_blend = record(fc_blend, Y["10y"])
    err = (Y["10y"] - fc_blend).dropna()
    recent = err.iloc[-84:]                                    # the last seven years of forecasts whose decade has finished
    axis = score.index[::3]
    charts["_gauge"] = {
        "dates": [t.date().isoformat() for t in axis], "score": M.series_list(score, axis, 1),
        "score_history": M.series_list(view_a, axis, 1), "score_era": M.series_list(view_b, axis, 1),
        "estimate": M.series_list(fc_blend, axis, 1), "realized": M.series_list(Y["10y"], axis, 1)}

    def delta(n):
        return round(now - float(score.iloc[-1 - n]), 1) if len(score) > n else None

    hist_avg = round(float(Y["10y"].loc[FIT_FROM:].mean()), 1)
    eps_g = next((m.series for m in ms if m.id == "eps_g"), None)
    peak = score.loc["1999-06":"2000-12"]
    out = {
        "id": "fund", "name": "Fundamentals", "benchmark": "S&P 500", "as_of": spx.index[-1].date().isoformat(),
        "score": round(now, 1), "zone": zone, "chg": {"m1": delta(1), "y1": delta(12), "y5": delta(60)},
        "views": {"history": {"score": round(float(view_a.dropna().iloc[-1]), 1), "pred": views["history"], "record": record(fc_a, Y["10y"])},
                  "era": {"score": round(float(view_b.dropna().iloc[-1]), 1), "pred": views["era"], "record": record(fc_b, Y["10y"]),
                          "years": ERA // 12}},
        "expected": {"mid": round(float(np.mean(mids)), 1), "low": round(min(pa + pb), 1), "high": round(max(pa + pb), 1),
                     "models": len(pa) + len(pb)},
        "record": rec_blend,
        "recent_bias": None if len(recent) < 24 else {"mean": round(float(recent.mean()), 1), "from": recent.index[0].year,
                                                      "to": recent.index[-1].year},
        "peak_2000": None if peak.empty else {"score": round(float(peak.max()), 1), "date": peak.idxmax().date().isoformat()},
        "hist_avg": hist_avg,
        "treasury10": None if dgs10 is None or dgs10.dropna().empty else round(float(dgs10.dropna().iloc[-1]), 2),
        "now": {"zone": next(z for z in zones if z["key"] == zone["key"]), "base": odds["base"]},
        "odds": odds, "support": None if eps_g is None else support_check(score, Y, eps_g, baa),
        "inputs": rows, "untested": UNTESTED,
        "spark": [round(float(v), 1) for v in score.iloc[-180:].values],
    }
    return out, charts
