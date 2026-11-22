"""The two tactical gauges: which indicators they use, and how each is computed.

Every indicator carries `fear`: "high" when a high reading means fear (VIX) and
"low" when a low reading does (share of stocks above their 50-day average).
`role` is "scored" (in the gauge) or "context" (shown with its own evidence,
kept out of the score; `why` says what the research found).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from signals import model as M


@dataclass
class Ind:
    id: str
    name: str
    unit: str            # "", "%", "pts", "x"
    fear: str            # high | low
    role: str            # scored | context
    what: str            # one line: what it measures
    why: str             # one line: why it is in (or out of) the score
    nd: int = 1
    pillar: str | None = None
    source: str = ""
    series: pd.Series | None = field(default=None, repr=False)
    last: pd.Timestamp | None = None   # date of the newest real observation behind `series`
    gap: int = 5                       # calendar days between updates before the reading counts as stale


AI_BASKET = {
    "Compute & networking": ["NVDA", "AMD", "AVGO", "MRVL", "ANET", "ARM"],
    "Foundry & equipment": ["TSM", "ASML", "LRCX", "AMAT"],
    "Memory": ["MU", "000660.KS", "005930.KS"],
    "Hyperscalers": ["MSFT", "GOOGL", "AMZN", "META", "ORCL"],
    "Neocloud & infrastructure": ["CRWV", "NBIS", "VRT"],
    "Power": ["GEV", "CEG", "VST", "BE"],
}
INDEX_SYMBOLS = ["^GSPC", "^VIX", "^VIX3M", "^VXN", "^SOX", "^NDX", "SMH", "^VIX9D", "HYG", "IEF", "SPLV", "SPHB", "SPY"]

SPX_PILLARS = {"Volatility": ["vix", "vix_ivp", "vix_ts", "vrp"], "Options positioning": ["pc_equity", "cor1m"]}
AI_PILLARS = {"Semis & Nasdaq volatility": ["sox_rv", "vxn"], "Market stress": ["vix", "vix_ts"]}


def _al(s: pd.Series | None, cal: pd.DatetimeIndex, limit: int = 5) -> tuple[pd.Series, pd.Timestamp | None]:
    """Align a source series to the gauge calendar; also return the date of its last real observation."""
    if s is None or s.dropna().empty:
        return pd.Series(np.nan, cal), None
    s = s.dropna()
    return s.reindex(cal).ffill(limit=limit), s.index[-1]


def _oldest(*dates) -> pd.Timestamp | None:
    known = [d for d in dates if d is not None]
    return min(known) if known else None


def _breadth(members: pd.DataFrame, cal: pd.DatetimeIndex, min_names: int) -> dict[str, pd.Series]:
    mem = members.reindex(cal)
    ma = mem.rolling(50).mean()
    valid = ma.notna()
    n = valid.sum(axis=1)
    above = ((mem > ma) & valid).sum(axis=1) / n.where(n >= min_names) * 100
    r = mem.apply(M.rsi)
    live = mem.notna().sum(axis=1)
    oversold = ((r < 30) & mem.notna()).sum(axis=1) / live.where(live >= min_names) * 100
    chg = mem.pct_change(fill_method=None)
    priced = chg.notna().sum(axis=1)
    down = (chg < 0).sum(axis=1) / priced.where(priced >= min_names) * 100
    return {"above50": above, "oversold": oversold, "down10": down.rolling(10).mean()}


def fund_flow_pct(flows: pd.DataFrame, z1: pd.DataFrame) -> pd.Series | None:
    """Net equity purchases by mutual funds and ETFs (Fed Z.1, annual rate), 4-quarter average, as a
    percent of the value of all US equities. Dated when it became public: about 6 months after the
    quarter's first day."""
    need = ["BOGZ1FA653064100Q", "BOGZ1FA563064100Q"]
    if flows is None or z1 is None or any(c not in flows for c in need) or "NCBEILQ027S" not in z1:
        return None
    buy = flows[need[0]].fillna(0) + flows[need[1]].fillna(0)
    value = z1["NCBEILQ027S"] + z1["FBCELLQ027S"]
    out = (buy.rolling(4).mean() / value * 100).dropna()
    out.index = out.index + pd.DateOffset(months=6)
    return out


def spx_indicators(px: pd.DataFrame, cboe: pd.DataFrame, pc: pd.DataFrame, members: pd.DataFrame,
                   baa: pd.Series | None, extra: dict | None = None) -> tuple[list[Ind], pd.Series]:
    """`extra` carries the slower context series: aaii (weekly frame), alloc (monthly frame),
    hy (daily series), flows (quarterly series from fund_flow_pct)."""
    extra = extra or {}
    spx = px["^GSPC"].dropna()
    cal = spx.index
    col = lambda df, c: df[c] if df is not None and c in df else None  # noqa: E731
    vix, d_vix = _al(col(px, "^VIX"), cal, 3)
    vix3m, d_v3 = _al(col(px, "^VIX3M"), cal, 3)
    pce, d_pc = _al(col(pc, "pc_equity"), cal)
    cor, d_cor = _al(col(cboe, "COR1M"), cal)
    vvix, d_vvix = _al(col(cboe, "VVIX"), cal)
    skew, d_skew = _al(col(cboe, "SKEW"), cal)
    baa_s, d_baa = _al(baa, cal)
    has_members = members is not None and not members.empty
    br = _breadth(members, cal, 200) if has_members else {}
    d_mem = members.index[-1] if has_members else None
    I = [
        Ind("vix", "VIX", "", "high", "scored", pillar="Volatility", source="Cboe, via Yahoo",
            what="30-day implied volatility of the S&P 500, ranked against every day since 1990.",
            why="The most dependable fear reading on its own: its highest readings were followed by above-average "
                "returns at every horizon, before and after 2010.", series=vix, last=d_vix),
        Ind("vix_ivp", "VIX IV percentile (1y)", "%", "high", "scored", nd=0, pillar="Volatility",
            source="Cboe, via Yahoo",
            what="Share of the past year's sessions that closed with a lower VIX than today.",
            why="Fear relative to the recent regime rather than all of history, which is what lets the gauge register "
                "complacency: adding it is what made the calm side of the score hold up in both halves of the sample.",
            series=M.iv_percentile(vix), last=d_vix),
        Ind("vix_ts", "VIX / 3-month VIX", "x", "high", "scored", nd=2, pillar="Volatility", source="Cboe, via Yahoo",
            what="Near-term implied volatility relative to 3-month. Above 1.00 (backwardation) means panic now.",
            why="Backwardation is the cleanest picture of panic, and steep contango of complacency. History starts in "
                "2006, so 2008 weighs heavily: the signal was early that autumn and has rarely missed since.",
            series=vix / vix3m, last=_oldest(d_vix, d_v3)),
        Ind("vrp", "VIX minus realized volatility", "pts", "high", "scored", pillar="Volatility",
            source="Cboe and S&P 500, via Yahoo",
            what="Implied volatility minus the S&P 500's actual 20-day volatility: the premium paid for protection.",
            why="A fat premium means hedging demand is outrunning actual risk. One of the two inputs that pass on "
                "their own in both halves of the sample.", series=vix - M.realized_vol(spx), last=d_vix),
        Ind("pc_equity", "Equity put/call ratio, 10-day", "", "high", "scored", nd=2, pillar="Options positioning",
            source="Cboe daily market statistics",
            what="Puts traded per call on single stocks at Cboe, averaged over 10 sessions.",
            why="The one input that measures what investors are doing rather than what options cost. History "
                "starts in 2006, so its own evidence is modest.",
            series=pce.rolling(10).mean(), last=d_pc),
        Ind("cor1m", "Implied correlation (COR1M)", "", "high", "scored", pillar="Options positioning",
            source="Cboe COR1M",
            what="How strongly options price S&P 500 stocks to move together over the next month.",
            why="Spikes when everything is sold at once and collapses when investors are picking stocks without "
                "fear. Few episodes since 2006, so its own evidence is modest.", series=cor, last=d_cor),
        # ---- context: tested, shown with their own evidence, kept out of the score
        Ind("vix_ivr", "VIX IV rank (1y)", "%", "high", "context", nd=0, pillar="Volatility", source="Cboe, via Yahoo",
            what="Where today's VIX sits between its 1-year low and 1-year high.",
            why="Same idea as IV percentile, but set by two extreme days, so one spike distorts it for a year. "
                "IV percentile is used instead.", series=M.iv_rank(vix), last=d_vix),
        Ind("above50", "S&P 500 stocks above their 50-day average", "%", "low", "context", nd=0, pillar="Breadth",
            source="Yahoo, current SPY members",
            what="Share of today's index members trading above their own 50-day average.",
            why="Breadth washouts pass the test on their own, mostly as a 1-month bounce signal. Left out because "
                "adding breadth made the score worse at 3 to 12 months; use it as confirmation. Built from today's "
                "members, so the past looks slightly healthier than it was.", series=br.get("above50"), last=d_mem),
        Ind("oversold", "S&P 500 stocks with RSI below 30", "%", "high", "context", nd=0, pillar="Breadth",
            source="Yahoo, current SPY members",
            what="Share of index members that are oversold on 14-day RSI.",
            why="Passes on its own but fires constantly (80+ episodes) for a small edge. Confirmation, not a signal.",
            series=br.get("oversold"), last=d_mem),
        Ind("dd", "S&P 500 below its 52-week high", "%", "high", "context", pillar="Price", source="Yahoo",
            what="How far the index is below its highest close of the past year.",
            why="Rejected. Deep drawdowns were followed by worse-than-average returns before 2010 (2000-02, 2008) "
                "and better ones since: price being down is not, by itself, a buy signal.",
            series=-(spx / spx.rolling(252).max() - 1) * 100, last=spx.index[-1]),
        Ind("ma200", "S&P 500 vs its 200-day average", "%", "low", "context", pillar="Price", source="Yahoo",
            what="Distance of the index from its 200-day average.",
            why="Rejected as a contrarian input for the same reason as drawdown. Useful only as trend context.",
            series=(spx / spx.rolling(200).mean() - 1) * 100, last=spx.index[-1]),
        Ind("vvix", "VVIX", "", "high", "context", pillar="Volatility", source="Cboe VVIX",
            what="Implied volatility of VIX options (the volatility of volatility).",
            why="Left out: it rises with VIX and its history only starts in 2006, so it could not be checked across "
                "both halves of the sample.", series=vvix, last=d_vvix),
        Ind("skew", "Cboe SKEW", "", "low", "context", pillar="Volatility", source="Cboe SKEW",
            what="Price of far out-of-the-money S&P 500 puts relative to at-the-money options.",
            why="Left out. High SKEW is popularly read as a crash warning, but high readings had no relation to "
                "forward returns. Low readings did precede better returns, yet the index has drifted upward for "
                "15 years, so its rank against its own history is not comparable over time.", series=skew, last=d_skew),
        Ind("baa", "Baa corporate bond spread", "pts", "high", "context", nd=2, pillar="Credit", source="FRED BAA10Y",
            what="Moody's Baa corporate yield minus the 10-year Treasury yield.",
            why="Rejected. Wide credit spreads flagged the 2001 and 2008 recessions, when buying early lost money. "
                "A stress thermometer, not a buy signal.", series=baa_s, last=d_baa),
    ]
    # ---- slower context: surveys, flows, high-yield spread (tested 2026-11-10; none improved the score)
    aaii, alloc, hy, flows = (extra.get(k) for k in ("aaii", "alloc", "hy", "flows"))
    if aaii is not None and not aaii.empty:
        spread4, d_aaii = _al((aaii["bull"] - aaii["bear"]).rolling(4).mean(), cal, 10)
        I.append(Ind("aaii", "AAII bulls minus bears, 4-week", "pts", "low", "context", pillar="Surveys & flows",
                     source="AAII Sentiment Survey", gap=12,
                     what="Share of individual investors who are bullish minus the share who are bearish in AAII's "
                          "weekly survey, averaged over four weeks.",
                     why="Has a modest edge on its own (deep pessimism was followed by slightly better returns in "
                         "both halves of the sample), but adding it to the score cut the buy zone's 3-month edge "
                         "from 3.0 to 1.8 points. Survey mood is noisier than what investors pay for options.",
                     series=spread4, last=d_aaii))
    if alloc is not None and not alloc.empty:
        avail = alloc["cash"].copy()
        avail.index = avail.index + pd.offsets.BDay(3)          # month-end survey, published early the next month
        cash, d_cash = _al(avail, cal, 45)
        I.append(Ind("aaii_cash", "AAII cash allocation", "%", "high", "context", pillar="Surveys & flows",
                     source="AAII Asset Allocation Survey", gap=50,
                     what="Share of individual investors' portfolios held in cash, from AAII's monthly survey. The "
                          "free stand-in for BofA's fund-manager cash level, which is not published as data.",
                     why="No edge found: neither high nor low cash levels were followed by returns different from "
                         "average at 1 to 3 months. It moves over years, not weeks.", series=cash, last=d_cash))
    if flows is not None and not flows.dropna().empty:
        fl, d_fl = _al(flows, cal, 120)
        I.append(Ind("flows", "Equity fund flows, 4 quarters", "%", "low", "context", nd=2, pillar="Surveys & flows",
                     source="FRED (Fed Z.1 accounts)", gap=200,
                     what="Net buying of stocks by mutual funds and ETFs over the past four quarters, as a percent "
                          "of the value of all US equities. Negative means investors were pulling money out.",
                     why="Heavy outflows were followed by better returns on their own, but the data is quarterly and arrives about "
                         "10 weeks late, so it confirms a low months afterwards rather than flagging it; adding it "
                         "weakened the score. Weekly flow data (ICI, EPFR) is not freely available to a script.",
                     series=fl, last=d_fl))
    if hy is not None and not hy.dropna().empty:
        hy_s, d_hy = _al(hy, cal)
        I.append(Ind("hy_oas", "High-yield credit spread", "pts", "high", "context", nd=2, pillar="Credit",
                     source="FRED BAMLH0A0HYM2 (ICE BofA)",
                     what="Extra yield junk bonds pay over Treasuries (ICE BofA option-adjusted spread).",
                     why="FRED now serves only three years of this series, too little to test. Its long-history "
                         "cousin, the Baa spread below, fails as a buy signal. Credit earns its place as a check on "
                         "the buy zone instead: see the credit check above.", series=hy_s, last=d_hy))
    # ---- tested 2026-11-21: strong or moderate on their own, but none improved the score (see the README)
    hyg, ief = _al(col(px, "HYG"), cal, 3)[0], _al(col(px, "IEF"), cal, 3)[0]
    credit3m = -((hyg / ief) / (hyg / ief).shift(63) - 1) * 100
    I.append(Ind("hy_tsy", "High-yield bonds vs Treasuries, 3 months", "%", "high", "context", pillar="Credit",
                 source="HYG and IEF, via Yahoo",
                 what="How far junk bonds (HYG) have lagged Treasuries (IEF) over 63 sessions. The free stand-in for "
                      "credit default swap indices, which are licensed data.",
                 why="Strong on its own since 2009: big credit sell-offs were followed by above-average stock returns. "
                     "Left out because with 2008 included it said buy months too early, and adding it made the "
                     "buy zone's 3-month record worse. Shown as confirmation, alongside the credit check.",
                 series=credit3m, last=_oldest(_al(col(px, "HYG"), cal)[1], _al(col(px, "IEF"), cal)[1])))
    v9, d9 = _al(col(px, "^VIX9D"), cal, 3)
    I.append(Ind("vix9d", "9-day VIX / VIX", "x", "high", "context", nd=2, pillar="Volatility", source="Cboe, via Yahoo",
                 what="Next-nine-days implied volatility relative to 30-day: panic about the coming week.",
                 why="Passes on its own on both sides since 2013, but adds nothing once VIX and its 3-month term "
                     "structure are in the score.", series=v9 / vix, last=_oldest(d9, d_vix)))
    lv, hb = _al(col(px, "SPLV"), cal, 3)[0], _al(col(px, "SPHB"), cal, 3)[0]
    I.append(Ind("lowvol", "Low-volatility vs high-beta stocks, 3 months", "%", "high", "context", pillar="Rotation",
                 source="SPLV and SPHB, via Yahoo",
                 what="How much the S&P 500's least volatile stocks have beaten its most volatile ones over 63 sessions.",
                 why="Strong on its own since 2013, but it duplicates the volatility inputs: adding it left the buy "
                     "zone unchanged and weakened the 12-month record.", series=((lv / hb) / (lv / hb).shift(63) - 1) * 100,
                 last=spx.index[-1]))
    spy_vol = extra.get("spy_vol")
    if spy_vol is not None and not spy_vol.dropna().empty:
        sv, d_sv = _al(spy_vol.where(spy_vol > 0), cal, 3)
        I.append(Ind("spy_vol", "SPY volume vs its 50-day average, 5 days", "x", "high", "context", nd=2,
                     pillar="Selling pressure", source="SPY, via Yahoo",
                     what="Trading volume in the largest S&P 500 fund relative to its recent norm: capitulation selling.",
                     why="Passes on its own in both halves of the sample, but adding it made the buy zone fire more "
                         "often with weaker results.", series=(sv / sv.rolling(50).mean()).rolling(5).mean(), last=d_sv))
    if br.get("down10") is not None:
        I.append(Ind("down10", "S&P 500 stocks falling each day, 10-day average", "%", "high", "context", nd=0,
                     pillar="Selling pressure", source="Yahoo, current SPY members",
                     what="Share of index members that closed lower each day, averaged over two weeks.",
                     why="A moderate edge on its own in both halves, but adding it weakened the score at 12 months.",
                     series=br["down10"], last=d_mem))
    return [i for i in I if i.series is not None and not i.series.dropna().empty], spx


def ai_indicators(px: pd.DataFrame, basket: pd.DataFrame) -> tuple[list[Ind], pd.Series]:
    sox = px["^SOX"].dropna()
    cal = sox.index
    col = lambda c: px[c] if c in px else None  # noqa: E731
    spx, _ = _al(col("^GSPC"), cal)
    ndx, d_ndx = _al(col("^NDX"), cal)
    vix, d_vix = _al(col("^VIX"), cal, 3)
    vix3m, d_v3 = _al(col("^VIX3M"), cal, 3)
    vxn, d_vxn = _al(col("^VXN"), cal, 3)
    rs = sox / spx
    b = basket.reindex(cal).ffill(limit=3)
    ma = b.rolling(50).mean()
    valid = ma.notna()
    n = valid.sum(axis=1)
    bk50 = ((b > ma) & valid).sum(axis=1) / n.where(n >= 8) * 100
    d_sox = sox.index[-1]
    I = [
        Ind("sox_rv", "Semiconductor realized volatility (20d)", "%", "high", "scored", nd=0,
            pillar="Semis & Nasdaq volatility", source="PHLX SOX, via Yahoo",
            what="How violently the semiconductor index has actually been moving, annualized.",
            why="The sector-specific stress reading. On its own, violent semis were not a buy signal (they kept "
                "falling in 2000-02), but it is what makes the combination work across the whole sample: without "
                "it, buy-zone readings before 2010 lost money.", series=M.realized_vol(sox), last=d_sox),
        Ind("vxn", "VXN (Nasdaq-100 volatility)", "", "high", "scored", pillar="Semis & Nasdaq volatility",
            source="Cboe, via Yahoo",
            what="30-day implied volatility of the Nasdaq-100, the closest listed fear gauge to AI stocks.",
            why="History starts in the 2001 crash, when it sat at 60 to 80, so its highest decile has almost never "
                "been reached since and its own evidence is thin.",
            series=vxn, last=d_vxn),
        Ind("vix", "VIX", "", "high", "scored", pillar="Market stress", source="Cboe, via Yahoo",
            what="30-day implied volatility of the S&P 500.",
            why="Semis bottom when the whole market panics, not when semis alone are weak. The strongest single "
                "input: its highest readings beat the average at every horizon, before and after 2010.",
            series=vix, last=d_vix),
        Ind("vix_ts", "VIX / 3-month VIX", "x", "high", "scored", nd=2, pillar="Market stress",
            source="Cboe, via Yahoo",
            what="Near-term implied volatility relative to 3-month. Above 1.00 (backwardation) means panic now.",
            why="Marks market-wide panic. History starts in 2006, so 2008 weighs heavily: early that autumn, and it "
                "has rarely missed since.", series=vix / vix3m, last=_oldest(d_vix, d_v3)),
        # ---- context
        Ind("vxn_vrp", "VXN minus Nasdaq realized volatility", "pts", "high", "context", pillar="Volatility",
            source="Cboe and Nasdaq-100, via Yahoo",
            what="Implied minus actual 20-day Nasdaq-100 volatility: the premium paid for protection.",
            why="Passes on its own at 1 to 6 months. Left out because adding it weakened the score at 6 and 12 "
                "months. A useful short-term confirmation.",
            series=vxn - M.realized_vol(ndx), last=_oldest(d_vxn, d_ndx)),
        Ind("sox_dd", "Semis below their 52-week high", "%", "high", "context", pillar="Price", source="PHLX SOX, via Yahoo",
            what="How far the semiconductor index is below its highest close of the past year.",
            why="Rejected. It looks like a great buy signal since 2010, but once 2000-02 (an 80% fall) is in the "
                "sample the edge is not statistically distinguishable from chance.",
            series=-(sox / sox.rolling(252).max() - 1) * 100, last=d_sox),
        Ind("sox_ma200", "Semis vs their 200-day average", "%", "low", "context", pillar="Price", source="PHLX SOX, via Yahoo",
            what="Distance of the semiconductor index from its 200-day average.",
            why="Rejected for the same reason as drawdown. Trend context only.",
            series=(sox / sox.rolling(200).mean() - 1) * 100, last=d_sox),
        Ind("rs63", "Semis vs S&P 500, 3-month", "%", "low", "context", pillar="Price", source="Yahoo",
            what="Semiconductor index return minus S&P 500 return over the past 63 sessions.",
            why="Rejected. Relative weakness did not reliably predict a rebound, and relative strength was followed "
                "by more strength, not a fall.", series=(rs / rs.shift(63) - 1) * 100, last=d_sox),
        Ind("bk50", "AI basket above 50-day average", "%", "low", "context", nd=0, pillar="Breadth", source="Yahoo, 25-name AI basket",
            what="Share of 25 AI names (chips, memory, hyperscalers, neoclouds, power) above their 50-day average.",
            why="Rejected as a signal: no reliable relation to forward returns, and the basket is today's winners "
                "applied backwards. Shown because it describes the damage.",
            series=bk50, last=b.dropna(how="all").index[-1] if not b.dropna(how="all").empty else None),
    ]
    hyg, ief = (_al(col(c), cal, 3) for c in ("HYG", "IEF"))
    if hyg[1] is not None and ief[1] is not None:
        I.append(Ind("hy_tsy", "High-yield bonds vs Treasuries, 3 months", "%", "high", "context", pillar="Credit",
                     source="HYG and IEF, via Yahoo",
                     what="How far junk bonds (HYG) have lagged Treasuries (IEF) over 63 sessions. The free stand-in "
                          "for credit default swap indices, which are licensed data.",
                     why="Strong on its own since 2009, but with 2008 included it said buy too early, and as part of "
                         "the score it halved the number of buy signals without improving them.",
                     series=-((hyg[0] / ief[0]) / (hyg[0] / ief[0]).shift(63) - 1) * 100, last=_oldest(hyg[1], ief[1])))
    if "NVDA" in b:
        I.append(Ind("nvda_rv", "NVDA realized volatility (20d)", "%", "high", "context", nd=0, pillar="Volatility", source="Yahoo",
                     what="Nvidia's actual 20-day volatility, annualized.",
                     why="Left out: one stock is too narrow to score, and it showed no edge of its own.",
                     series=M.realized_vol(b["NVDA"]), last=d_sox))
    return [i for i in I if i.series is not None and not i.series.dropna().empty], sox


def build_gauge(gid: str, name: str, benchmark: str, inds: list[Ind], px: pd.Series, pillars: dict[str, list[str]],
                with_history: bool = True) -> tuple[dict, dict]:
    """Score one gauge and assemble everything the dashboard shows for it."""
    cal = px.index
    fear_pct = {}
    for i in inds:
        s = i.series.reindex(cal)
        fear_pct[i.id] = M.expanding_pct(s if i.fear == "high" else -s)
    fear_pct = pd.DataFrame(fear_pct).reindex(cal)
    comp = M.composite(fear_pct, pillars, cal)
    score = comp["score"].dropna()
    last = score.index[-1]
    now = float(score.iloc[-1])
    zone = M.zone_of(now)
    odds = M.odds_table(score, px)
    ev = M.tail_evidence(score, px, 10, 90)
    zrow = next(z for z in odds["zones"] if z["key"] == zone["key"])

    def delta(n):
        return round(now - float(score.iloc[-1 - n]), 1) if len(score) > n else None

    in_zone = (score >= zone["lo"]) & (score < (zone["hi"] if zone["hi"] < 100 else 100.001))
    run = 0
    for v in in_zone.values[::-1]:
        if not v:
            break
        run += 1
    scored_ids = [c for cols in pillars.values() for c in cols]
    live = int(fear_pct[scored_ids].ffill(limit=5).loc[last].notna().sum())
    rows = []
    for i in inds:
        s = i.series.reindex(cal)
        v = s.dropna()
        fp = fear_pct[i.id].dropna()
        if v.empty:
            continue
        level = (100 - fp * 100)                       # 0 = fear, 100 = complacency, like the gauge
        e = M.tail_evidence(level, px, 10, 90) if len(fp) > 1500 else None
        rows.append({
            "id": i.id, "name": i.name, "unit": i.unit, "nd": i.nd, "fear": i.fear, "role": i.role,
            "pillar": i.pillar, "what": i.what, "why": i.why, "source": i.source,
            "value": round(float(v.iloc[-1]), i.nd + 1), "as_of": (i.last or v.index[-1]).date().isoformat(),
            "stale": bool((last - (i.last or v.index[-1])).days > i.gap),
            "score": round(float(level.iloc[-1]), 1) if len(level) else None,
            "since": v.index[0].date().isoformat(),
            "chg_1w": round(float(v.iloc[-1] - v.iloc[-6]), i.nd + 1) if len(v) > 6 else None,
            "evidence": None if e is None else {
                "fear": {**e["low"], "grade": M.grade(e["low"], "above")},
                "calm": {**e["high"], "grade": M.grade(e["high"], "below")}},
        })
    checks = []
    baa = next((i for i in inds if i.id == "baa"), None)
    if baa is not None:
        # Credit check: was the Baa spread widening unusually fast (top 10% of its 3-month changes to date)?
        widening = M.expanding_pct(baa.series.reindex(cal).diff(63))
        w = widening.dropna()
        if len(w) > 1500:
            rows_c = M.split_odds(score, px, widening >= 0.9)
            checks.append({"id": "credit", "name": "Credit check",
                           "now": round(float(w.iloc[-1]) * 100), "flag": bool(w.iloc[-1] >= 0.9),
                           "yes": "Credit spreads blowing out", "no": "Credit spreads not blowing out",
                           "rows": rows_c,
                           "note": "Blowing out = the Baa corporate spread's 3-month widening is in the top 10% of "
                                   "its history to date."})
    gauge = {
        "id": gid, "name": name, "benchmark": benchmark, "as_of": last.date().isoformat(),
        "score": round(now, 1), "zone": zone, "zone_days": run,
        "chg": {"d1": delta(1), "w1": delta(5), "m1": delta(21)},
        "coverage": {"live": live, "of": len(scored_ids)},
        "price": round(float(px.iloc[-1]), 2),
        "price_chg": {"d1": round(float(px.iloc[-1] / px.iloc[-2] - 1) * 100, 2),
                      "m1": round(float(px.iloc[-1] / px.iloc[-22] - 1) * 100, 1),
                      "from_high": round(float(px.iloc[-1] / px.iloc[-252:].max() - 1) * 100, 1)},
        "now": {"zone": zrow, "base": odds["base"]},
        "odds": odds,
        "evidence": {"fear": {**ev["low"], "grade": M.grade(ev["low"], "above")},
                     "calm": {**ev["high"], "grade": M.grade(ev["high"], "below")}},
        "pillars": [{"name": k, "score": round(float(comp[f"pillar_{k}"].dropna().iloc[-1]), 1), "ids": v}
                    for k, v in pillars.items() if not comp[f"pillar_{k}"].dropna().empty],
        "checks": checks,
        "indicators": rows,
        "episodes": {"buy": M.episodes(score, px, 10, True)[::-1], "caution": M.episodes(score, px, 90, False)[::-1]},
        "spark": [round(float(v), 1) for v in score.iloc[-126:].values],
    }
    hist = {}
    if with_history:
        axis = M.thin(score.index)
        hist = {"dates": [d.date().isoformat() for d in axis], "score": M.series_list(score, axis, 1),
                "price": M.series_list(px, axis, 2), "ind": {}}
        for i in inds:
            hist["ind"][i.id] = M.series_list(i.series.reindex(cal), axis, i.nd + 1)
    return gauge, hist
