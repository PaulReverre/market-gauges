"""Scores, pillars, gauges, coverage and overlays (BUILD_PROMPT.md Sections 6.2, 6.5, 6.7, 7).

Usage: python -m gauges.score [--db PATH]

Per indicator and trading day: percentile of the as-known value within its
trailing window (point in time), oriented to 0-100, times a staleness
multiplier (1 up to 1.5x the expected age since publication, 0 at 3x) and the
events status multiplier. Pillar = weighted mean of live components; gauge =
weighted mean of live pillars; coverage = live share of target gauge weight.
Results are written to derived tables (replaced each run); raw data is untouched.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from gauges import transforms as T
from gauges.config import ROOT, load_baskets, load_events, load_series_config
from gauges.features import Features
from gauges.registry import Indicator, Registry, load_registry
from gauges.store import DEFAULT_DB, Store

MIN_HISTORY = 126          # trading days of known values before an indicator is scored
SHORT_HISTORY = 504        # less than 2 years of history -> "short-history" badge
LOW_COVERAGE = 70.0        # % of target gauge weight
INFL_THRESHOLD, INFL_POINTS, INFL_MAX = 3.0, 10.0, 20.0   # core PCE % above which rates & liquidity is penalized
HYPERSCALERS = {"microsoft": "MSFT", "alphabet": "GOOGL", "google": "GOOGL", "amazon": "AMZN", "meta": "META"}


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def business_age(avail: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    a = pd.to_datetime(avail)
    mask = a.notna().to_numpy()
    out = np.full(len(dates), np.nan)
    out[mask] = np.busday_count(a[mask].to_numpy().astype("datetime64[D]"),
                                dates[mask].to_numpy().astype("datetime64[D]"))
    return out


def score_indicator(ind: Indicator, known: pd.DataFrame | None, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(index=calendar)
    if known is None:
        known = pd.DataFrame(index=calendar, columns=["value", "as_of", "avail"], dtype=float)
    if ind.kind == "event":
        frame["value"] = known["score"].astype(float)
        frame["pct"] = np.nan
        frame["score"] = known["score"].astype(float)
        frame["status"] = known["status_mult"].astype(float).fillna(0.0)
        frame["rejected"] = False
    else:
        value = known["value"].astype(float)
        rejected = pd.Series(False, index=calendar)
        if ind.bounds:
            rejected = value.notna() & ~value.between(*ind.bounds)
            value = value.where(~rejected)
        if "pct" in known:   # revised series ranked within each vintage (features._known_vintaged)
            pct = known["pct"].astype(float).where(~rejected)
        else:
            pct = T.rolling_percentile(value, ind.lookback_days, MIN_HISTORY)
        frame["value"], frame["pct"] = value, pct
        frame["score"] = pct if ind.sign > 0 else 100 - pct
        frame["status"] = 1.0
        frame["rejected"] = rejected
    frame["as_of"] = pd.to_datetime(known["as_of"])
    frame["avail"] = pd.to_datetime(known["avail"])
    age = business_age(frame["avail"], calendar)
    frame["stale"] = np.where(np.isnan(age), 0.0, T.staleness_multiplier(np.nan_to_num(age), ind.freq_bdays))
    frame["mult"] = frame["stale"] * frame["status"] * frame["score"].notna()
    history = frame["value"].notna().cumsum()
    frame["short_history"] = history < min(ind.lookback_days or SHORT_HISTORY, SHORT_HISTORY)
    if "history_years" in known:
        frame["short_history"] = known["history_years"].astype(float).fillna(0) < SHORT_HISTORY / 252
    return frame


# --------------------------------------------------------------------------- #
# Composites
# --------------------------------------------------------------------------- #
@dataclass
class GaugeResult:
    gauge: pd.DataFrame                     # score, display, coverage, low_coverage
    pillars: pd.DataFrame                   # one column per pillar
    contributions: pd.DataFrame             # one column per component; rows sum to the gauge score
    extra: dict = field(default_factory=dict)


def composite(reg: Registry, gauge_id: str, scored: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex,
              penalty: pd.Series | None = None) -> GaugeResult:
    spec = reg.gauges[gauge_id]
    inds = [i for i in reg.indicators if i.gauge == gauge_id and i.active]
    pillar_scores, shares, target = {}, {}, {}
    for pillar, pw in spec.pillars.items():
        members = [i for i in inds if i.pillar == pillar]
        total_w = sum(i.weight for i in members)
        W = pd.DataFrame({i.id: i.weight * scored[i.id]["mult"] for i in members}, index=calendar)
        S = pd.DataFrame({i.id: scored[i.id]["score"] for i in members}, index=calendar)
        wsum = W.sum(axis=1).replace(0, np.nan)
        ps = (W * S.fillna(0)).sum(axis=1) / wsum
        if penalty is not None and gauge_id == "G2" and pillar == "rates_liquidity":
            ps = (ps - penalty.reindex(calendar).fillna(0)).clip(0, 100)
        pillar_scores[pillar] = ps
        shares[pillar] = W.div(wsum, axis=0)
        for i in members:
            target[i.id] = pw * i.weight / total_w
    P = pd.DataFrame(pillar_scores)
    pw = pd.Series(spec.pillars)
    live_pw = P.notna().mul(pw, axis=1)
    denom = live_pw.sum(axis=1).replace(0, np.nan)
    gauge = (P.fillna(0) * pw).sum(axis=1) / denom
    coverage = sum(target[i.id] * scored[i.id]["mult"] for i in inds) / sum(spec.pillars.values()) * 100

    contrib = {}
    for i in inds:
        contrib[i.id] = live_pw[i.pillar] / denom * shares[i.pillar][i.id].fillna(0) * scored[i.id]["score"].fillna(0)
    if penalty is not None and gauge_id == "G2":
        contrib["infl_penalty"] = -(live_pw["rates_liquidity"] / denom) * penalty.reindex(calendar).fillna(0)
    display = T.ema(gauge, 3) if spec.smoothing == "ema3" else gauge
    frame = pd.DataFrame({"score": gauge, "display": display.where(gauge.notna()), "coverage": coverage,
                          "low_coverage": coverage < LOW_COVERAGE})
    return GaugeResult(frame, P, pd.DataFrame(contrib, index=calendar))


def infl_penalty(scored: dict[str, pd.DataFrame]) -> pd.Series:
    core = scored["infl"]["value"]
    return ((core - INFL_THRESHOLD) * INFL_POINTS).clip(0, INFL_MAX).where(core.notna())


# --------------------------------------------------------------------------- #
# Overlays
# --------------------------------------------------------------------------- #
def fragility(reg: Registry, scored: dict[str, pd.DataFrame], F: Features) -> pd.DataFrame:
    """Equal-weight mean of members oriented so 100 = fragile, plus the breadth-divergence flag."""
    cal = F.calendar
    parts, mults = {}, {}
    for ind in [i for i in reg.indicators if i.fragility]:
        pct = scored[ind.id]["pct"]
        parts[ind.id] = pct if ind.fragility == "higher_is_fragile" else 100 - pct
        mults[ind.id] = scored[ind.id]["stale"] * pct.notna()
    div = reg.fragility.divergence
    spx = F.obs("yfinance", "^GSPC.close").reindex(cal).ffill()
    near_high = spx >= (1 - div["near_high_pct"] / 100) * spx.rolling(252, min_periods=200).max()
    breadth = scored["br_200"]["value"]
    flag = (near_high & (breadth < div["breadth_below"])).astype(float) * 100
    known_flag = breadth.notna() & spx.rolling(252, min_periods=200).max().notna()
    parts["breadth_divergence"] = flag.where(known_flag)
    mults["breadth_divergence"] = known_flag.astype(float)
    Pt, M = pd.DataFrame(parts), pd.DataFrame(mults)
    score = (Pt.fillna(0) * M).sum(axis=1) / M.sum(axis=1).replace(0, np.nan)
    return pd.DataFrame({"fragility": score, "divergence": flag.where(known_flag),
                         "members_live": (M > 0).sum(axis=1)})


def cnn_replica(reg: Registry, scored: dict[str, pd.DataFrame]) -> pd.Series:
    S = pd.DataFrame({i.id: scored[i.id]["score"].where(scored[i.id]["mult"] > 0) for i in reg.indicators if i.cnn})
    return S.mean(axis=1)


def banners(g1: pd.Series, g2: pd.Series, frag: pd.Series) -> pd.DataFrame:
    falling = (g2 < g2.shift(5)) & (g2.shift(5) < g2.shift(10)) & (g2.shift(10) < g2.shift(15))
    return pd.DataFrame({
        "complacent_and_fragile": (g1 >= 70) & (frag >= 70),
        "fear_without_fundamental_damage": (g1 <= 25) & (g2 >= 50),
        "sentiment_ahead_of_fundamentals": (g1 >= 70) & falling,
    })


def ai_fragility(scored: dict[str, pd.DataFrame], F: Features) -> dict[str, bool | None]:
    """Latest state of the AI fragility conditions (None = not enough data)."""
    def last(s: pd.Series):
        s = s.dropna()
        return None if s.empty else s.iloc[-1]

    smh = F.obs("yfinance", "SMH.close")
    breadth = last(scored["ai_breadth"]["value"])
    near = smh.iloc[-1] >= 0.95 * smh.iloc[-252:].max() if len(smh) > 252 else None
    fade = scored["ai_beatfade"]["value"].dropna()
    fade_rising = (bool(fade.iloc[-1] > fade.iloc[-64] > fade.iloc[-127]) if len(fade) > 127 else None)
    ivs = scored["ai_iv_spread"]["value"].dropna()
    iv_rising = (bool(ivs.iloc[-1] > ivs.iloc[-22] and smh.iloc[-1] > smh.iloc[-22]) if len(ivs) > 22 else None)
    corr_pct = last(scored["ai_corr"]["pct"])
    mem = last(F.memory_drawdown())
    return {
        "smh_near_high_with_weak_breadth": None if near is None or breadth is None else bool(near and breadth < 50),
        "basket_correlation_top_decile": None if corr_pct is None else bool(corr_pct >= 90),
        "beat_fade_rising_two_seasons": fade_rising,
        "iv_spread_rising_with_price": iv_rising,
        "memory_drawdown_over_25pct": None if mem is None else bool(mem < -25),
    }


def _light(red: bool, amber: bool, known: bool) -> str:
    return "grey" if not known else "red" if red else "amber" if amber else "green"


def unwind_signature(scored: dict[str, pd.DataFrame], F: Features) -> dict:
    """The four Unwind lights (Section 6.7), latest state only, with the rule and inputs behind each."""
    def last(ind_id: str, col: str = "value"):
        s = scored[ind_id][col].dropna()
        return None if s.empty else float(s.iloc[-1])

    # 1. Capacity loosens but growth stays flat.
    gpu = last("ce_gpu")
    tokens = last("mo_tokens")
    labs_add_down = None
    labs = sorted((e for e in F.events if e.kpi == "mo_labs" and isinstance(e.value, float)),
                  key=lambda e: (e.entity, e.period))
    for entity in {e.entity for e in labs}:
        vals = [e.value for e in labs if e.entity == entity]
        if len(vals) >= 3:
            adds = np.diff(vals)
            labs_add_down = bool(adds[-1] < adds[-2]) or bool(labs_add_down)
    leg_a = None if gpu is None else gpu < -10
    leg_b = None if (labs_add_down is None and tokens is None) else bool(labs_add_down) or (tokens is not None and tokens < 0)
    legs = [x for x in (leg_a, leg_b) if x is not None]
    light1 = _light(all(legs) and len(legs) == 2, any(legs), bool(legs))

    # 2. Labs cut prices: same-model flagship price change over ~30 days.
    flag = F.flagships()
    change = None
    if not flag.empty:
        latest_snap = flag["snapshot_date"].max()
        cat = F.con.execute("SELECT snapshot_date, model_id, prompt_price, completion_price FROM model_catalog").df()
        cat["snapshot_date"] = pd.to_datetime(cat["snapshot_date"])
        old = cat[cat["snapshot_date"] <= latest_snap - pd.Timedelta(days=30)]
        if not old.empty:
            old = old[old["snapshot_date"] == old["snapshot_date"].max()].set_index("model_id")
            now = flag[flag["snapshot_date"] == latest_snap].set_index("model_id")
            common = now.index.intersection(old.index)
            if len(common):
                blended_old = (3 * old.loc[common, "prompt_price"] + old.loc[common, "completion_price"]) / 4 * 1e6
                change = float(((now.loc[common, "blended"] / blended_old - 1) * 100).median())
    cut_event = any(e.kpi == "price_cut" and (pd.Timestamp.today() - pd.Timestamp(e.as_of)).days <= 30 for e in F.events)
    light2 = _light(cut_event or (change is not None and change <= -20), change is not None and change <= -10,
                    cut_event or change is not None)

    # 3. Hyperscalers trim 2027 capex (sp_guide_next events).
    framing = {}
    for e in sorted((e for e in F.events if e.kpi == "sp_guide_next"), key=lambda e: e.as_of):
        ticker = next((t for k, t in HYPERSCALERS.items() if k in e.entity.lower()), None)
        if ticker and ticker != "ORCL":
            framing[ticker] = e.value
    values = list(framing.values())
    light3 = _light("cut" in values, values.count("hold") >= 2, bool(values))

    # 4. AI credit spreads widen.
    bbb = last("fc_bbb")
    proxy_pct = last("fc_proxy", "pct")
    fin = sorted((e for e in F.events if e.kpi == "fc_events"), key=lambda e: e.as_of)
    recent = fin[-1].value if fin and (pd.Timestamp.today() - pd.Timestamp(fin[-1].as_of)).days <= 90 else None
    red4 = (bbb is not None and bbb >= 25) or (proxy_pct is not None and proxy_pct <= 10) or recent == "strong_negative"
    amber4 = (bbb is not None and bbb >= 12.5) or (proxy_pct is not None and proxy_pct <= 20) or recent == "negative"
    light4 = _light(red4, amber4, bbb is not None or proxy_pct is not None or recent is not None)

    lights = {
        "capacity_loosens_growth_flat": {"state": light1, "gpu_rental_63d_pct": gpu, "tokens_28d_growth_pct": tokens,
                                         "lab_dollar_add_declining": labs_add_down},
        "labs_cut_prices": {"state": light2, "flagship_price_30d_pct": change, "price_cut_event": cut_event},
        "hyperscalers_trim_2027_capex": {"state": light3, "framing": framing},
        "ai_credit_spreads_widen": {"state": light4, "bbb_oas_21d_bp": bbb, "orcl_crwv_drawdown_pct_rank": proxy_pct,
                                    "latest_financing_event": recent},
    }
    reds = sum(v["state"] == "red" for v in lights.values())
    return {"lights": lights, "red_count": reds, "alert": reds >= 3}


def attribution(contrib: pd.DataFrame, periods: int, top: int = 3) -> list[dict]:
    if len(contrib) <= periods:
        return []
    delta = (contrib.iloc[-1] - contrib.iloc[-1 - periods]).dropna()
    delta = delta[delta.abs() > 1e-9]
    return [{"id": k, "change": round(float(v), 2)} for k, v in delta.reindex(delta.abs().sort_values(ascending=False).index)[:top].items()]


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(store: Store) -> dict:
    reg = load_registry()
    events, _ = load_events()
    F = Features(store, load_series_config(), load_baskets(), events)
    cal = F.calendar
    scored = {}
    for ind in reg.indicators:
        known = F.event_scores(ind) if ind.kind == "event" else F.build(ind)
        scored[ind.id] = score_indicator(ind, known, cal)

    penalty = infl_penalty(scored)
    results = {g: composite(reg, g, scored, cal, penalty if g == "G2" else None) for g in reg.gauges}
    frag = fragility(reg, scored, F)
    cnn = cnn_replica(reg, scored)
    ban = banners(results["G1"].gauge["display"], results["G2"].gauge["display"], frag["fragility"])

    persist(store, reg, scored, results, frag, cnn, ban)
    summary = latest_summary(reg, scored, results, frag, cnn, ban, F, penalty)
    store.con.execute("CREATE OR REPLACE TABLE score_latest AS SELECT ? AS as_of, ? AS summary",
                      [cal[-1].date(), json.dumps(summary, default=str)])
    return summary


def persist(store, reg, scored, results, frag, cnn, ban) -> None:
    ind_rows = []
    for ind in reg.indicators:
        f = scored[ind.id].reset_index(names="date").assign(id=ind.id, gauge=ind.gauge, pillar=ind.pillar)
        ind_rows.append(f[f["value"].notna() | f["score"].notna()])
    indicators = pd.concat(ind_rows, ignore_index=True)
    gauges = pd.concat([r.gauge.reset_index(names="date").assign(gauge=g) for g, r in results.items()], ignore_index=True)
    pillars = pd.concat([r.pillars.reset_index(names="date").melt(id_vars="date", var_name="pillar", value_name="score")
                         .assign(gauge=g) for g, r in results.items()], ignore_index=True)
    overlay = frag.join(cnn.rename("cnn_replica")).join(ban).reset_index(names="date")
    for name, df in (("indicator_daily", indicators), ("gauge_daily", gauges.dropna(subset=["score"])),
                     ("pillar_daily", pillars.dropna(subset=["score"])), ("overlay_daily", overlay)):
        store.con.register("staged_scores", df)
        store.con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM staged_scores")
        store.con.unregister("staged_scores")


def latest_summary(reg, scored, results, frag, cnn, ban, F, penalty) -> dict:
    cal = F.calendar
    as_of = cal[-1]
    gauges = {}
    for g, r in results.items():
        row = r.gauge.iloc[-1]
        display = None if pd.isna(row["display"]) else round(float(row["display"]), 1)
        gauges[g] = {
            "name": reg.gauges[g].name, "as_of": as_of.date(), "score": None if pd.isna(row["score"]) else round(float(row["score"]), 1),
            "display": display, "band": None if display is None else T.band(display, reg.gauges[g].scale),
            "coverage": round(float(row["coverage"]), 1), "low_coverage": bool(row["low_coverage"]),
            "pillars": {p: (None if pd.isna(v) else round(float(v), 1)) for p, v in r.pillars.iloc[-1].items()},
            "change_1w": _delta(r.gauge["display"], 5), "change_1m": _delta(r.gauge["display"], 21),
            "drivers_1w": attribution(r.contributions, 5), "drivers_1m": attribution(r.contributions, 21),
            "contributions": {k: round(float(v), 2) for k, v in r.contributions.iloc[-1].items() if abs(v) > 1e-9},
        }
    gauges["G2"]["infl_penalty"] = None if pd.isna(penalty.iloc[-1]) else round(float(penalty.iloc[-1]), 1)
    gaps = {"missing": [], "stale": [], "short_history": []}
    for ind in reg.indicators:
        f = scored[ind.id].iloc[-1]
        if not ind.active:
            continue
        if pd.isna(f["score"]):
            gaps["missing"].append(ind.id)
        elif f["stale"] < 1:
            gaps["stale"].append(f"{ind.id} (as of {pd.Timestamp(f['as_of']).date()}, weight x{f['stale']:.2f})")
        if not pd.isna(f["score"]) and f["short_history"]:
            gaps["short_history"].append(ind.id)
    frag_row = frag.iloc[-1]
    return {
        "as_of": as_of.date(),
        "gauges": gauges,
        "fragility": {"score": None if pd.isna(frag_row["fragility"]) else round(float(frag_row["fragility"]), 1),
                      "breadth_divergence": None if pd.isna(frag_row["divergence"]) else bool(frag_row["divergence"]),
                      "validated": reg.fragility.validated,
                      "drawdown_probability": drawdown_probability(F, reg, frag_row["fragility"])},
        "cnn_replica": None if pd.isna(cnn.iloc[-1]) else round(float(cnn.iloc[-1]), 1),
        "banners": [k for k, v in ban.iloc[-1].items() if bool(v)],
        "ai_fragility": ai_fragility(scored, F),
        "unwind": unwind_signature(scored, F),
        "open_weight_token_share": _last_round(F.open_weight_share()),
        "gaps": gaps,
    }


def drawdown_probability(F: Features, reg: Registry, fragility_score: float):
    """Empirical 10%+ S&P drawdown rate within 63 days for the current fragility decile (Phase 3 backtest)."""
    if not reg.fragility.validated or pd.isna(fragility_score):
        return "Not validated"
    try:
        deciles = F.con.execute("SELECT decile, upper_edge, dd10_rate, days FROM fragility_deciles ORDER BY decile").df()
    except Exception:  # noqa: BLE001 - table appears after the first backtest
        return "Not validated"
    row = deciles[deciles["upper_edge"] >= fragility_score].iloc[0]
    return {"decile": int(row["decile"]), "probability_pct": round(float(row["dd10_rate"]), 1),
            "sample_days": int(row["days"])}


def _delta(s: pd.Series, n: int):
    s = s.dropna()
    return None if len(s) <= n else round(float(s.iloc[-1] - s.iloc[-1 - n]), 1)


def _last_round(s: pd.Series):
    s = s.dropna()
    return None if s.empty else round(float(s.iloc[-1]), 1)


def render_text(summary: dict) -> str:
    lines = [f"Gauges as of {summary['as_of']}", ""]
    for g, v in summary["gauges"].items():
        cov = f"coverage {v['coverage']:.0f}%" + (" LOW COVERAGE" if v["low_coverage"] else "")
        lines.append(f"{g} {v['name']}: {v['display']} ({v['band']}) | 1W {v['change_1w']:+} 1M {v['change_1m']:+} | {cov}"
                     if v["display"] is not None else f"{g} {v['name']}: no score | {cov}")
        lines.append("   pillars: " + ", ".join(f"{p} {s}" for p, s in v["pillars"].items()))
        lines.append("   1W drivers: " + ", ".join(f"{d['id']} {d['change']:+}" for d in v["drivers_1w"]))
        lines.append("   1M drivers: " + ", ".join(f"{d['id']} {d['change']:+}" for d in v["drivers_1m"]))
    f = summary["fragility"]
    prob = f["drawdown_probability"]
    prob = prob if isinstance(prob, str) else (f"{prob['probability_pct']}% (decile {prob['decile']}, "
                                               f"{prob['sample_days']} days)")
    lines += ["", f"Fragility overlay: {f['score']} (breadth divergence: {f['breadth_divergence']}; "
                  f"10%+ drawdown probability in 63 days: {prob})",
              f"CNN-method replica: {summary['cnn_replica']}",
              f"Banners: {', '.join(summary['banners']) or 'none'}",
              "AI fragility flags: " + ", ".join(f"{k}={v}" for k, v in summary["ai_fragility"].items()),
              f"Unwind signature ({summary['unwind']['red_count']} red{' - ALERT' if summary['unwind']['alert'] else ''}):"]
    for name, light in summary["unwind"]["lights"].items():
        detail = ", ".join(f"{k}={v}" for k, v in light.items() if k != "state")
        lines.append(f"   {light['state']:<6} {name}: {detail}")
    gaps = summary["gaps"]
    lines += ["", f"Missing ({len(gaps['missing'])}): {', '.join(gaps['missing'])}",
              f"Stale ({len(gaps['stale'])}): {'; '.join(gaps['stale'])}",
              f"Short history ({len(gaps['short_history'])}): {', '.join(gaps['short_history'])}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score indicators and gauges from the raw store")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    store = Store(args.db)
    try:
        print(render_text(run(store)))
    finally:
        store.close()


if __name__ == "__main__":
    main()
