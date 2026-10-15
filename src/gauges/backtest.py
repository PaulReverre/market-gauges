"""Validation and backtest (BUILD_PROMPT.md Section 8): gates composite membership.

Usage: python -m gauges.backtest [--apply]

Scores are rebuilt point in time (gauges.score), then each indicator and
composite is compared with forward outcomes on the S&P 500 (G1, G2, fragility)
or SMH (G3, G4; the AI basket is reported as a hindsight-biased secondary):
forward 21/63/126-day returns, 63-day max drawdown, and whether a 10%+
drawdown happens within 63 days.

Expected relation by gauge family and role (`EXPECTED`): sentiment readings
(greed = high) should precede weaker returns (contrarian, coincident) or
shallower drawdowns (leading); health readings should precede shallower
drawdowns (leading, coincident) or better 126-day returns (context).
An indicator with at least MIN_YEARS of history fails if its full-sample IC
has the wrong sign or the sign disagrees in most stress regimes. A failing
indicator is relabeled when another role fits with |t| >= 2, otherwise demoted
to display. Weights are never fitted to the sample. --apply writes the changes
to config/indicators.yaml and re-scores; the report shows before and after.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from gauges import score as score_mod
from gauges.config import CONFIG_DIR, ROOT, load_baskets, load_events, load_series_config
from gauges.features import Features
from gauges.registry import Indicator, Registry, load_registry
from gauges.store import DEFAULT_DB, Store

REPORT = ROOT / "reports" / "validation.md"
HORIZONS = (21, 63, 126)
DD_WINDOW, DD_THRESHOLD = 63, -10.0
MIN_YEARS = 5
MIN_REGIME_OBS = 60
REGIMES = {
    "2008": ("2007-10-01", "2009-06-30"), "2011": ("2011-04-01", "2011-12-30"),
    "2015-16": ("2015-06-01", "2016-03-31"), "2018 Q4": ("2018-09-04", "2019-03-29"),
    "2020": ("2020-01-02", "2020-08-31"), "2022": ("2022-01-03", "2022-12-30"),
    "Apr 2025": ("2025-01-15", "2025-07-31"), "2026": ("2026-01-02", "2026-12-31"),
}
EXPECTED = {   # (family, role) -> (target, expected sign of Spearman IC)
    ("greed", "contrarian"): ("fwd63", -1), ("greed", "coincident"): ("fwd63", -1),
    ("greed", "context"): ("fwd63", -1), ("greed", "leading"): ("mdd63", 1),
    ("health", "leading"): ("mdd63", 1), ("health", "coincident"): ("mdd63", 1),
    ("health", "context"): ("fwd126", 1),
}
HORIZON_OF = {"fwd21": 21, "fwd63": 63, "fwd126": 126, "mdd63": 63, "dd10": 63}
# Fragility validation: top decile must clear the base rate by both margins, with a rising profile.
FRAG_RATIO, FRAG_EXCESS_PP, FRAG_MONOTONIC = 1.5, 5.0, 0.6


# --------------------------------------------------------------------------- #
# Targets and statistics (pure)
# --------------------------------------------------------------------------- #
def forward_return(price: pd.Series, h: int) -> pd.Series:
    return (price.shift(-h) / price - 1) * 100


def forward_max_drawdown(price: pd.Series, h: int) -> pd.Series:
    """Worst peak-to-trough decline (%) within [t, t+h]; NaN where the window runs past the data."""
    p = price.to_numpy(dtype=float)
    out = np.full(len(p), np.nan)
    for i in range(len(p) - h):
        w = p[i:i + h + 1]
        if np.isnan(w).any():
            continue
        out[i] = (w / np.maximum.accumulate(w) - 1).min() * 100
    return pd.Series(out, index=price.index)


def targets(price: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({f"fwd{h}": forward_return(price, h) for h in HORIZONS})
    df["mdd63"] = forward_max_drawdown(price, DD_WINDOW)
    df["dd10"] = (df["mdd63"] <= DD_THRESHOLD).astype(float).where(df["mdd63"].notna())
    return df


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 30 or x[m].nunique() < 3:
        return math.nan, n
    return float(x[m].corr(y[m], method="spearman")), n


def nonoverlap_t(x: pd.Series, y: pd.Series, h: int) -> tuple[float, int, float]:
    """Spearman on every h-th joint observation (independent windows) and its t-stat."""
    m = x.notna() & y.notna()
    xs, ys = x[m].iloc[::h], y[m].iloc[::h]
    rho, n = spearman(xs, ys) if len(xs) >= 30 else (math.nan, len(xs))
    if math.isnan(rho) or n < 4 or abs(rho) >= 1:
        return rho, n, math.nan
    return rho, n, rho * math.sqrt((n - 2) / (1 - rho ** 2))


def decile_table(score: pd.Series, tg: pd.DataFrame) -> pd.DataFrame:
    df = tg.assign(score=score).dropna(subset=["score", "fwd63"])
    if len(df) < 100:
        return pd.DataFrame()
    df["decile"] = pd.qcut(df["score"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    g = df.groupby("decile")
    return pd.DataFrame({
        "days": g.size(), "score": g["score"].mean(), "fwd21": g["fwd21"].mean(), "fwd63": g["fwd63"].mean(),
        "fwd126": g["fwd126"].mean(), "hit63": g["fwd63"].apply(lambda s: (s > 0).mean() * 100),
        "mdd63": g["mdd63"].mean(), "dd10": g["dd10"].mean() * 100,
    })


def monotonicity(table: pd.DataFrame, col: str) -> float:
    if table.empty:
        return math.nan
    return float(pd.Series(table.index, index=table.index).corr(table[col], method="spearman"))


def regime_ics(x: pd.Series, y: pd.Series) -> dict[str, float]:
    out = {}
    for name, (start, end) in REGIMES.items():
        rho, n = spearman(x.loc[start:end], y.loc[start:end])
        if n >= MIN_REGIME_OBS and not math.isnan(rho):
            out[name] = rho
    return out


# --------------------------------------------------------------------------- #
# Indicator verdicts
# --------------------------------------------------------------------------- #
@dataclass
class IndicatorResult:
    ind: Indicator
    target: str
    sign: int
    ic: float
    n: int
    ic_nonoverlap: float
    t: float
    ic_ret63: float
    t_ret63: float
    ic_mdd63: float
    t_mdd63: float
    ic_fwd126: float
    t_fwd126: float
    regimes: dict[str, float]
    years: float
    verdict: str = "keep"
    new_role: str | None = None
    reason: str = ""
    spread63: float = math.nan     # top-decile minus bottom-decile mean 63d return

    @property
    def regimes_agree(self) -> int:
        return sum(1 for v in self.regimes.values() if v * self.sign > 0)


def family(gauge: str) -> str:
    return "greed" if gauge in ("G1", "G3") else "health"


def evaluate_indicator(ind: Indicator, x: pd.Series, tg: pd.DataFrame) -> IndicatorResult:
    fam = family(ind.gauge)
    target, sign = EXPECTED.get((fam, ind.role), ("fwd63", -1 if fam == "greed" else 1))
    ic, n = spearman(x, tg[target])
    _, _, t = nonoverlap_t(x, tg[target], HORIZON_OF[target])
    ic_no = nonoverlap_t(x, tg[target], HORIZON_OF[target])[0]
    stats = {}
    for col in ("fwd63", "mdd63", "fwd126"):
        stats[col] = (spearman(x, tg[col])[0], nonoverlap_t(x, tg[col], HORIZON_OF[col])[2])
    table = decile_table(x, tg)
    r = IndicatorResult(ind, target, sign, ic, n, ic_no, t, stats["fwd63"][0], stats["fwd63"][1], stats["mdd63"][0],
                        stats["mdd63"][1], stats["fwd126"][0], stats["fwd126"][1], regime_ics(x, tg[target]),
                        years=n / 252,
                        spread63=(table.loc[10, "fwd63"] - table.loc[1, "fwd63"]) if not table.empty else math.nan)
    decide(r, fam)
    return r


def decide(r: IndicatorResult, fam: str) -> None:
    role = r.ind.role
    if role == "display":
        r.verdict, r.reason = "display", "display only; not tested"
        return
    if r.years < MIN_YEARS or math.isnan(r.ic):
        r.verdict, r.reason = "keep (low confidence)", f"{r.years:.1f} years of history, rules need {MIN_YEARS}"
        return
    wrong_sign = r.ic * r.sign <= 0
    flips = len(r.regimes) > 0 and r.regimes_agree < len(r.regimes) - r.regimes_agree
    if not wrong_sign and not flips:
        r.verdict, r.reason = "keep", "sign as expected" + (" in most regimes" if r.regimes else "")
        return
    why = ("wrong sign in full sample" if wrong_sign else "") + (" and " if wrong_sign and flips else "") + \
          (f"sign holds in only {r.regimes_agree}/{len(r.regimes)} regimes" if flips else "")
    # Does another role fit clearly (|t| >= 2 on non-overlapping windows)?
    alternatives = []
    if fam == "greed":
        alternatives = [("contrarian", r.ic_ret63 < 0, r.t_ret63), ("leading", r.ic_mdd63 > 0, r.t_mdd63)]
    else:
        alternatives = [("leading", r.ic_mdd63 > 0, r.t_mdd63), ("context", r.ic_fwd126 > 0, r.t_fwd126)]
    for alt, ok, t in alternatives:
        if alt != role and ok and not math.isnan(t) and abs(t) >= 2:
            r.verdict, r.new_role, r.reason = "relabel", alt, f"{why}; fits {alt} (t={t:.1f})"
            return
    r.verdict, r.new_role, r.reason = "demote", "display", why


@dataclass
class FragilityMember:
    id: str
    ic_mdd63: float
    t: float
    years: float
    regimes: dict[str, float]
    keep: bool
    reason: str


def evaluate_fragility_member(ind: Indicator, pct: pd.Series, tg: pd.DataFrame) -> FragilityMember:
    fragile = pct if ind.fragility == "higher_is_fragile" else 100 - pct
    ic, n = spearman(fragile, tg["mdd63"])
    _, _, t = nonoverlap_t(fragile, tg["mdd63"], 63)
    regimes = regime_ics(fragile, tg["mdd63"])
    agree = sum(1 for v in regimes.values() if v < 0)
    years = n / 252
    if years < MIN_YEARS or math.isnan(ic):
        return FragilityMember(ind.id, ic, t, years, regimes, True, "low confidence: short history")
    if ic >= 0 or agree < len(regimes) - agree:
        return FragilityMember(ind.id, ic, t, years, regimes, False,
                               "more fragile did not precede deeper drawdowns" if ic >= 0 else
                               f"holds in only {agree}/{len(regimes)} regimes")
    return FragilityMember(ind.id, ic, t, years, regimes, True, "more fragile preceded deeper drawdowns")


# --------------------------------------------------------------------------- #
# Composites
# --------------------------------------------------------------------------- #
@dataclass
class CompositeResult:
    name: str
    target_name: str
    table: pd.DataFrame
    ics: dict[str, tuple[float, float]]
    regimes: dict[str, float]
    base_dd10: float
    extra: dict = field(default_factory=dict)


def evaluate_composite(name: str, s: pd.Series, tg: pd.DataFrame, target_name: str) -> CompositeResult:
    table = decile_table(s, tg)
    ics = {col: (spearman(s, tg[col])[0], nonoverlap_t(s, tg[col], HORIZON_OF[col])[2])
           for col in ("fwd21", "fwd63", "fwd126", "mdd63")}
    base = float(tg["dd10"].loc[s.dropna().index].mean() * 100) if s.notna().any() else math.nan
    return CompositeResult(name, target_name, table, ics, regime_ics(s, tg["fwd63"]), base)


def validate_fragility(res: CompositeResult) -> tuple[bool, str]:
    t = res.table
    if t.empty:
        return False, "no history"
    top, base = t.loc[10, "dd10"], res.base_dd10
    mono = monotonicity(t, "dd10")
    ok = top >= FRAG_RATIO * base and top >= base + FRAG_EXCESS_PP and mono >= FRAG_MONOTONIC
    detail = (f"top-decile 10%+ drawdown rate {top:.1f}% vs base {base:.1f}% "
              f"({top / base if base else math.nan:.2f}x); decile monotonicity {mono:.2f}")
    return ok, detail


# --------------------------------------------------------------------------- #
# Registry edits
# --------------------------------------------------------------------------- #
def edit_registry(text: str, ind_id: str, changes: dict[str, str | None]) -> str:
    """Set (or with None, remove) top-level keys inside one indicator block, keeping the rest."""
    m = re.search(rf"(?m)^  - id: {re.escape(ind_id)}\n((?:    .*\n)*)", text)
    if not m:
        raise KeyError(ind_id)
    block = m.group(1)
    for key, value in changes.items():
        pattern = rf"(?m)^    {key}: .*\n"
        if value is None:
            block = re.sub(pattern, "", block)
        elif re.search(pattern, block):
            block = re.sub(pattern, f"    {key}: {value}\n", block)
        else:
            block += f"    {key}: {value}\n"
    return text[:m.start(1)] + block + text[m.end(1):]


def apply_changes(results: list[IndicatorResult], members: list[FragilityMember], frag_ok: bool) -> list[dict]:
    path = CONFIG_DIR / "indicators.yaml"
    text = path.read_text(encoding="utf-8")
    changes, edits, notes = [], {}, {}
    for r in results:
        if r.verdict in ("relabel", "demote"):
            edits.setdefault(r.ind.id, {})["role"] = r.new_role
            notes.setdefault(r.ind.id, []).append(f"{r.verdict} {r.ind.role} -> {r.new_role} ({r.reason})")
            changes.append({"id": r.ind.id, "field": "role", "before": r.ind.role, "after": r.new_role, "why": r.reason})
    for m in members:
        if not m.keep:
            edits.setdefault(m.id, {})["fragility"] = None
            notes.setdefault(m.id, []).append(f"removed from fragility overlay ({m.reason})")
            changes.append({"id": m.id, "field": "fragility", "before": "member", "after": "removed", "why": m.reason})
    for ind_id, fields in edits.items():
        fields["validation"] = json.dumps("Phase 3 backtest: " + "; ".join(notes[ind_id]))
        text = edit_registry(text, ind_id, fields)
    text = re.sub(r"(?m)^  validated: (true|false)", f"  validated: {'true' if frag_ok else 'false'}", text)
    path.write_text(text, encoding="utf-8")
    load_registry(path)   # fail loudly if the edit broke the schema
    return changes


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def load_wide(store: Store, column: str) -> pd.DataFrame:
    df = store.con.execute(f"SELECT date, id, {column} AS v FROM indicator_daily").df()
    return df.pivot(index="date", columns="id", values="v").sort_index()


def composites(store: Store) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = store.con.execute("SELECT date, gauge, display FROM gauge_daily").df()
    gauges = g.pivot(index="date", columns="gauge", values="display").sort_index()
    overlay = store.con.execute("SELECT date, fragility, cnn_replica FROM overlay_daily").df().set_index("date")
    return gauges, overlay.sort_index()


def price_targets(F: Features) -> dict[str, pd.DataFrame]:
    cal = F.calendar
    spx = F.obs("yfinance", "^GSPC.close").reindex(cal).ffill()
    smh = F.obs("yfinance", "SMH.close").reindex(cal).ffill()
    basket = F.basket()["index"].reindex(cal).ffill()
    return {"S&P 500": targets(spx), "SMH": targets(smh), "AI basket (hindsight)": targets(basket)}


def run(store: Store, apply: bool) -> str:
    reg = load_registry()
    events, _ = load_events()
    score_mod.run(store)
    F = Features(store, load_series_config(), load_baskets(), events)
    tgs = price_targets(F)
    tg_for = {"G1": tgs["S&P 500"], "G2": tgs["S&P 500"], "G3": tgs["SMH"], "G4": tgs["SMH"]}
    scores, pcts = load_wide(store, "score"), load_wide(store, "pct")
    cal = F.calendar

    results = []
    for ind in reg.indicators:
        if ind.id not in scores or ind.kind in ("none", "event"):
            continue
        results.append(evaluate_indicator(ind, scores[ind.id].reindex(cal), tg_for[ind.gauge]))
    members = [evaluate_fragility_member(i, pcts[i.id].reindex(cal), tgs["S&P 500"])
               for i in reg.indicators if i.fragility and i.id in pcts]

    before = composite_results(store, tgs, cal)
    frag_ok_before, frag_detail_before = validate_fragility(before["Fragility overlay"])
    changes, after, frag_ok, frag_detail = [], None, frag_ok_before, frag_detail_before
    if apply:
        changes = apply_changes(results, members, frag_ok_before)
        score_mod.run(store)
        after = composite_results(store, tgs, cal)
        frag_ok, frag_detail = validate_fragility(after["Fragility overlay"])
        if frag_ok != frag_ok_before:
            apply_changes([], [], frag_ok)
        save_fragility_deciles(store, after["Fragility overlay"], frag_ok)
        score_mod.run(store)
    survivorship = survivorship_check(scores, cal, tgs["S&P 500"])
    report = render(reg, results, members, before, after, changes, (frag_ok_before, frag_detail_before),
                    (frag_ok, frag_detail), survivorship, apply)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    return report


def composite_results(store: Store, tgs: dict, cal: pd.DatetimeIndex) -> dict[str, CompositeResult]:
    gauges, overlay = composites(store)
    gauges, overlay = gauges.reindex(cal), overlay.reindex(cal)
    out = {}
    for g, target in (("G1", "S&P 500"), ("G2", "S&P 500"), ("G3", "SMH"), ("G4", "SMH")):
        out[g] = evaluate_composite(g, gauges[g], tgs[target], target)
    out["G3 vs AI basket"] = evaluate_composite("G3", gauges["G3"], tgs["AI basket (hindsight)"], "AI basket (hindsight)")
    out["Fragility overlay"] = evaluate_composite("Fragility", overlay["fragility"], tgs["S&P 500"], "S&P 500")
    out["CNN replica"] = evaluate_composite("CNN replica", overlay["cnn_replica"], tgs["S&P 500"], "S&P 500")
    return out


def save_fragility_deciles(store: Store, res: CompositeResult, validated: bool) -> None:
    """Decile edges and empirical 10%+ drawdown rates, read by score.py when validated."""
    s = store.con.execute("SELECT fragility FROM overlay_daily WHERE fragility IS NOT NULL").df()["fragility"]
    edges = np.nanpercentile(s, np.arange(10, 100, 10))
    t = res.table
    df = pd.DataFrame({"decile": t.index, "upper_edge": list(edges) + [100.0], "dd10_rate": t["dd10"].to_numpy(),
                       "days": t["days"].to_numpy(), "validated": validated})
    store.con.register("staged_deciles", df)
    store.con.execute("CREATE OR REPLACE TABLE fragility_deciles AS SELECT * FROM staged_deciles")
    store.con.unregister("staged_deciles")


def survivorship_check(scores: pd.DataFrame, cal, tg: pd.DataFrame) -> dict:
    """Constituent breadth (current members, survivorship-biased) vs RSP/SPY breadth."""
    out = {}
    if "br_200" in scores and "br_ew" in scores:
        a, b = scores["br_200"].reindex(cal), scores["br_ew"].reindex(cal)
        out["score_correlation"] = spearman(a, b)[0]
        out["br_200_ic_mdd63"] = spearman(a, tg["mdd63"])[0]
        out["br_ew_ic_mdd63"] = spearman(b, tg["mdd63"])[0]
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def f(x, nd=2) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def table_md(t: pd.DataFrame) -> list[str]:
    if t.empty:
        return ["_Not enough history for deciles._"]
    lines = ["| Decile | Days | Mean score | Fwd 21d % | Fwd 63d % | Fwd 126d % | 63d hit % | Mean 63d max DD % | 10%+ DD rate % |",
             "|---|---|---|---|---|---|---|---|---|"]
    for d, row in t.iterrows():
        lines.append(f"| {d} | {int(row['days'])} | {f(row['score'], 1)} | {f(row['fwd21'])} | {f(row['fwd63'])} | "
                     f"{f(row['fwd126'])} | {f(row['hit63'], 1)} | {f(row['mdd63'])} | {f(row['dd10'], 1)} |")
    return lines


def composite_md(title: str, r: CompositeResult) -> list[str]:
    ics = " · ".join(f"{k}: IC {f(v[0])} (t {f(v[1], 1)})" for k, v in r.ics.items())
    regimes = ", ".join(f"{k} {f(v)}" for k, v in r.regimes.items()) or "—"
    return [f"#### {title} vs {r.target_name}", "", f"Spearman IC (t from non-overlapping windows): {ics}", "",
            f"IC with 63d return by regime: {regimes}", "", f"Base 10%+ drawdown rate: {f(r.base_dd10, 1)}%", "",
            *table_md(r.table), ""]


def verdict_paragraph(name: str, r: CompositeResult, low_confidence: bool) -> str:
    t = r.table
    if t.empty:
        return f"**{name}:** not enough history to evaluate."
    top, bottom = t.loc[10], t.loc[1]
    ic63, t63 = r.ics["fwd63"]
    icdd, tdd = r.ics["mdd63"]
    if name in ("G1", "G3"):
        shape = ("behaves as a contrarian signal: readings in the greediest decile were followed by weaker 63-day "
                 "returns than the most fearful decile" if top["fwd63"] < bottom["fwd63"]
                 else "does not behave as a contrarian signal at the extremes in this sample")
        text = (f"**{name}:** {shape} ({f(top['fwd63'])}% vs {f(bottom['fwd63'])}%; IC {f(ic63)}, t {f(t63, 1)}). "
                f"Extreme fear (decile 1) saw a {f(bottom['dd10'], 1)}% rate of 10%+ drawdowns versus "
                f"{f(top['dd10'], 1)}% in extreme greed, so the needle is coincident with stress rather than a "
                f"predictor of it; downside prediction is left to the fragility overlay, as Section 2 intends.")
    else:
        text = (f"**{name}:** the strongest decile was followed by a mean 63-day max drawdown of {f(top['mdd63'])}% "
                f"versus {f(bottom['mdd63'])}% for the weakest (IC with drawdown {f(icdd)}, t {f(tdd, 1)}), and "
                f"63-day returns of {f(top['fwd63'])}% vs {f(bottom['fwd63'])}% (IC {f(ic63)}, t {f(t63, 1)}).")
    if low_confidence:
        text += (" Low confidence: AI inputs mostly start in 2023 or later, the basket is defined with hindsight, "
                 "and no AI drawdown probabilities are shown.")
    return text


def render(reg: Registry, results, members, before, after, changes, frag_before, frag_after, survivorship, applied) -> str:
    final = after or before
    out = ["# Validation report (Phase 3)", "",
           "Generated by `make backtest` (`python -m gauges.backtest --apply`). All scores are point in time: each "
           "day uses only data public by then, ranked against its own trailing history. Weights are not fitted to "
           "the sample, so no walk-forward tuning was needed.", "",
           "**Caveats.** Forward windows overlap, so daily ICs overstate significance; t-stats use non-overlapping "
           "windows. Constituent breadth (net highs/lows, McClellan, % above 200d) uses today's S&P 500 members and "
           "carries survivorship bias. The AI basket is today's selection applied backwards, so AI trend and breadth "
           "inputs carry hindsight bias; G3/G4 are scored against SMH. ICE BofA spreads only exist from 2023-10-16 "
           "and several AI and earnings series start in 2023-2025, so those indicators sit below the "
           f"{MIN_YEARS}-year threshold and are kept with a low-confidence label rather than judged.", "",
           "## Verdicts", ""]
    for name, low in (("G1", False), ("G2", False), ("G3", True), ("G4", True)):
        text = verdict_paragraph(name, final[name], low)
        if after is not None:
            b, a = before[name].ics, after[name].ics
            text += (f" Before the registry changes the IC with 63-day drawdown was {f(b['mdd63'][0])} and with 63-day "
                     f"return {f(b['fwd63'][0])}; after, {f(a['mdd63'][0])} and {f(a['fwd63'][0])} (in-sample).")
        out += [text, ""]
    ok, detail = frag_after
    out += [f"**Fragility overlay:** {'validated' if ok else 'Not validated'}: {detail}. "
            + ("The dashboard shows the empirical 10%+ drawdown probability for the current decile with its sample "
               "count." if ok else "The dashboard shows 'Not validated' instead of a probability."), ""]
    cnn = final["CNN replica"]
    if not cnn.table.empty:
        out += [f"**CNN-method replica:** greediest decile 63-day return {f(cnn.table.loc[10, 'fwd63'])}% vs "
                f"{f(cnn.table.loc[1, 'fwd63'])}% for the most fearful (IC {f(cnn.ics['fwd63'][0])}), the same "
                "contrarian shape. Its spot checks against CNN's published value are manual.", ""]

    out += ["## Registry changes", ""]
    if not applied:
        out += ["_Dry run: no changes written. Run with --apply._", ""]
    if changes:
        out += ["Made in this run:", "", "| Indicator | Field | Before | After | Reason |", "|---|---|---|---|---|"]
        out += [f"| {c['id']} | {c['field']} | {c['before']} | {c['after']} | {c['why']} |" for c in changes]
    else:
        out += ["No indicator failed validation in this run."]
    recorded = [i for i in load_registry().indicators if i.validation]
    if recorded:
        out += ["", "Recorded in config/indicators.yaml (all runs):", "", "| Indicator | Gauge | Role now | Change |",
                "|---|---|---|---|"]
        out += [f"| {i.id} | {i.gauge} | {i.role} | {i.validation.removeprefix('Phase 3 backtest: ')} |" for i in recorded]
    out += ["", f"Fragility overlay validated: before {frag_before[0]}, after {frag_after[0]}.", ""]

    out += ["## Composites", ""]
    for label, res in (("Before registry changes", before), ("After registry changes", after)):
        if res is None:
            continue
        out += [f"### {label}", ""]
        if res is after:
            out += ["_In-sample: indicators were removed using this same history, which flatters these numbers. "
                    "They show what the gauges now measure, not out-of-sample skill._", ""]
        for name, r in res.items():
            out += composite_md(name, r)

    out += ["## Indicators", "",
            f"Target and expected sign by role (`EXPECTED`). Rules apply with at least {MIN_YEARS} years of scored "
            "history. Regimes: " + ", ".join(REGIMES) + ".", "",
            "| Indicator | Gauge | Role | Target (sign) | Years | IC | t | IC 63d ret | IC 63d DD | IC 126d ret | "
            "Regimes agree | Top-bottom 63d % | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        verdict = r.verdict + (f" → {r.new_role}" if r.new_role and r.verdict == "relabel" else "")
        out.append(f"| {r.ind.id} | {r.ind.gauge} | {r.ind.role} | {r.target} ({'+' if r.sign > 0 else '−'}) | "
                   f"{f(r.years, 1)} | {f(r.ic)} | {f(r.t, 1)} | {f(r.ic_ret63)} | {f(r.ic_mdd63)} | {f(r.ic_fwd126)} | "
                   f"{r.regimes_agree}/{len(r.regimes)} | {f(r.spread63)} | {verdict}: {r.reason} |")
    out += ["", "## Fragility members", "",
            "Expected: a more fragile reading precedes a deeper 63-day drawdown (negative IC).", "",
            "| Member | Years | IC with 63d DD | t | Regimes agree | Kept | Reason |", "|---|---|---|---|---|---|---|"]
    for m in members:
        agree = sum(1 for v in m.regimes.values() if v < 0)
        out.append(f"| {m.id} | {f(m.years, 1)} | {f(m.ic_mdd63)} | {f(m.t, 1)} | {agree}/{len(m.regimes)} | "
                   f"{'yes' if m.keep else 'no'} | {m.reason} |")
    if survivorship:
        out += ["", "## Survivorship cross-check", "",
                f"Constituent breadth (br_200, current members) vs RSP/SPY breadth (br_ew): score correlation "
                f"{f(survivorship.get('score_correlation'))}; IC with 63d drawdown {f(survivorship.get('br_200_ic_mdd63'))} "
                f"vs {f(survivorship.get('br_ew_ic_mdd63'))}. A gap between them is the bias the constituent version carries."]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest indicators and composites; write reports/validation.md")
    parser.add_argument("--apply", action="store_true", help="write demotions and role changes to indicators.yaml")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    store = Store(args.db)
    try:
        run(store, args.apply)
        print(f"Wrote {REPORT.relative_to(ROOT)}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
