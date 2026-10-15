"""Daily as-known indicator values, point in time.

For every trading day t, an indicator's value is what could have been computed
from data public by t:
- revised series (source alfred) are rebuilt vintage by vintage; a value is
  known from its vintage date;
- other series are known at obs_date + lag_days (config/series.yaml);
- panels compute from many series, then carry their own availability.
Each result is a frame on the trading calendar with columns value, as_of (the
observation date behind the value) and avail (when it became public).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

from gauges import transforms as T
from gauges.config import Baskets, Event, SeriesConfig
from gauges.registry import Indicator, InputRef
from gauges.sources.ssga import yahoo_symbol
from gauges.store import Store

KNOWN = ["value", "as_of", "avail"]
MIN_VINTAGE_OBS = 12   # observations a vintage needs before its latest value is ranked
CAPEX_TAGS = {   # per-company XBRL tag for cash capex (EDGAR probe, Phase 1)
    "MSFT": "PaymentsToAcquirePropertyPlantAndEquipment",
    "GOOGL": "PaymentsToAcquirePropertyPlantAndEquipment",
    "AMZN": "PaymentsToAcquireProductiveAssets",
    "META": "PaymentsToAcquirePropertyPlantAndEquipment",
    "ORCL": "PaymentsToAcquirePropertyPlantAndEquipment",
}
OCF_TAG = "NetCashProvidedByUsedInOperatingActivities"
RPO_TAG = "RevenueRemainingPerformanceObligation"
RPO_COMPANIES = ["MSFT", "GOOGL", "ORCL"]   # AMZN's tag stopped in 2020; META does not tag RPO
MEMORY_LAYER = "memory"
FX = {".KS": "KRW=X"}
FLAGSHIP_LABS = ["anthropic", "openai", "google"]


class Features:
    def __init__(self, store: Store, cfg: SeriesConfig, baskets: Baskets, events: list[Event], start: date = date(2000, 1, 1)):
        self.con, self.cfg, self.baskets, self.events = store.con, cfg, baskets, events
        rows = self.con.execute(
            "SELECT DISTINCT obs_date FROM raw_observations WHERE source = 'yfinance' AND series = 'SPY.close' "
            "AND obs_date >= ? ORDER BY 1", [start]).fetchall()
        self.calendar = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows], name="date")
        self._obs: dict[tuple[str, str], pd.Series] = {}
        self._panels: dict[str, object] = {}

    # ---------------------------------------------------------------- loading
    def latest_many(self, source: str, series: list[str]) -> pd.DataFrame:
        """Wide frame (obs_date x series) of the latest-vintage value per obs_date."""
        if not series:
            return pd.DataFrame()
        df = self.con.execute(
            """
            SELECT series, obs_date,
                   arg_max(value, {'v': coalesce(vintage, DATE '1900-01-01'), 'f': fetched_at}) AS value
            FROM raw_observations
            WHERE source = ? AND list_contains(?, series)
            GROUP BY series, obs_date
            """, [source, series]).df()
        if df.empty:
            return pd.DataFrame(columns=series)
        df["obs_date"] = pd.to_datetime(df["obs_date"])
        return df.pivot(index="obs_date", columns="series", values="value").sort_index()

    def obs(self, source: str, series: str) -> pd.Series:
        key = (source, series)
        if key not in self._obs:
            wide = self.latest_many(source, [series])
            self._obs[key] = (wide[series].dropna() if series in wide
                              else pd.Series(dtype=float, index=pd.DatetimeIndex([])))
        return self._obs[key]

    def to_calendar(self, values: pd.Series, as_of: pd.Series | None = None) -> pd.DataFrame:
        """Carry values (indexed by availability date) forward onto the trading calendar.

        A row whose as_of is older than one already known (a late, stale print) is ignored.
        """
        values = values.dropna()
        if values.empty:
            return pd.DataFrame(index=self.calendar, columns=KNOWN, dtype=float)
        as_of = (as_of if as_of is not None else pd.Series(values.index, index=values.index)).reindex(values.index)
        df = pd.DataFrame({"value": values.to_numpy(), "as_of": pd.to_datetime(as_of.to_numpy()),
                           "avail": values.index}, index=pd.DatetimeIndex(values.index))
        df = df.sort_index(kind="stable")
        df = df[df["as_of"] >= df["as_of"].cummax()]
        df = df[~df.index.duplicated(keep="last")]
        return df.reindex(df.index.union(self.calendar)).ffill().reindex(self.calendar)

    def lag(self, source: str, series: str) -> pd.Timedelta:
        spec = self.cfg.spec_for(source, series)
        return pd.Timedelta(days=spec.lag_days if spec else 0)

    # ------------------------------------------------------------ series kind
    def known(self, inp: InputRef, lookback_days: int | None = None) -> pd.DataFrame:
        op = T.OPS[inp.op]
        if inp.source == "alfred":
            return self._known_vintaged(inp.source, inp.series, lambda s: op(s, **inp.args), lookback_days)
        y = op(self.obs(inp.source, inp.series), **inp.args).dropna()
        if y.empty:
            return self.to_calendar(y)
        avail = y.index + self.lag(inp.source, inp.series)
        return self.to_calendar(pd.Series(y.to_numpy(), index=avail), pd.Series(y.index, index=avail))

    def _known_vintaged(self, source: str, series: str, fn, lookback_days: int | None) -> pd.DataFrame:
        """Rebuild the series vintage by vintage. Besides the latest value, each vintage ranks
        that value within its own (revised) history over the lookback: real-time ranking that
        stays consistent with revisions and works for series that only recently joined ALFRED."""
        rows = self.con.execute(
            """
            SELECT obs_date, vintage, arg_max(value, fetched_at) AS value FROM raw_observations
            WHERE source = ? AND series = ? GROUP BY obs_date, vintage ORDER BY vintage, obs_date
            """, [source, series]).df()
        if rows.empty:
            return self.to_calendar(pd.Series(dtype=float))
        rows["obs_date"] = pd.to_datetime(rows["obs_date"])
        rows["vintage"] = pd.to_datetime(rows["vintage"])
        span = None if lookback_days is None else pd.Timedelta(days=lookback_days / 252 * 365.25)
        current: dict[pd.Timestamp, float] = {}
        out_v, out_val, out_asof, out_pct, out_years = [], [], [], [], []
        for vintage, grp in rows.groupby("vintage", sort=True):
            current.update(zip(grp["obs_date"], grp["value"]))
            y = fn(pd.Series(current).sort_index()).dropna()
            if not len(y):
                continue
            last = y.iloc[-1]
            hist = y if span is None else y[y.index > y.index[-1] - span]
            out_v.append(vintage)
            out_val.append(last)
            out_asof.append(y.index[-1])
            out_pct.append(((hist < last).sum() + 0.5 * (hist == last).sum()) / len(hist) * 100
                           if len(hist) >= MIN_VINTAGE_OBS else np.nan)
            out_years.append((hist.index[-1] - hist.index[0]).days / 365.25)
        frame = self.to_calendar(pd.Series(out_val, index=out_v), pd.Series(out_asof, index=out_v))
        extra = self.to_calendar(pd.Series(out_pct, index=out_v), pd.Series(out_asof, index=out_v))
        years = self.to_calendar(pd.Series(out_years, index=out_v), pd.Series(out_asof, index=out_v))
        return frame.assign(pct=extra["value"], history_years=years["value"])

    def build_series(self, ind: Indicator) -> pd.DataFrame:
        if ind.transform in T.OBS_TIME_COMBINES:
            cols = {i.ref: T.OPS[i.op](self.obs(i.source, i.series), **i.args) for i in ind.inputs}
            df = pd.DataFrame(cols)
            y = T.COMBINES[ind.transform](df, **ind.targs).dropna()
            lag = max(self.lag(i.source, i.series) for i in ind.inputs)
            avail = y.index + lag
            return self.to_calendar(pd.Series(y.to_numpy(), index=avail), pd.Series(y.index, index=avail))
        parts = [self.known(i, ind.lookback_days) for i in ind.inputs]
        if len(parts) == 1 and ind.transform == "identity":
            return parts[0]   # keeps the vintage-consistent percentile of revised series
        values = pd.concat([p["value"] for p in parts], axis=1)
        y = T.COMBINES[ind.transform](values.astype(float), **ind.targs)
        as_of = pd.concat([p["as_of"] for p in parts], axis=1).min(axis=1)
        avail = pd.concat([p["avail"] for p in parts], axis=1).min(axis=1)
        out = pd.DataFrame({"value": y, "as_of": as_of, "avail": avail})
        return out.where(out["value"].notna())

    # ------------------------------------------------------------- dispatcher
    def build(self, ind: Indicator) -> pd.DataFrame | None:
        if ind.kind == "series":
            return self.build_series(ind)
        if ind.kind == "panel":
            return getattr(self, f"panel_{ind.panel}")(**ind.targs)
        return None

    def _daily(self, s: pd.Series, lag_days: int = 0) -> pd.DataFrame:
        """A series computed on observation dates, known lag_days later."""
        s = s.dropna()
        if s.empty:
            return self.to_calendar(s)
        avail = s.index + pd.Timedelta(days=lag_days)
        return self.to_calendar(pd.Series(s.to_numpy(), index=avail), pd.Series(s.index, index=avail))

    # ---------------------------------------------------------- equity breadth
    def member_tickers(self) -> list[str]:
        rows = self.con.execute(
            "SELECT ticker FROM index_holdings WHERE fund = 'SPY' AND as_of = (SELECT max(as_of) FROM index_holdings "
            "WHERE fund = 'SPY')").fetchall()
        return sorted({yahoo_symbol(r[0]) for r in rows})

    def members(self, field: str) -> pd.DataFrame:
        key = f"members_{field}"
        if key not in self._panels:
            wide = self.latest_many("yfinance", [f"{t}.{field}" for t in self.member_tickers()])
            wide.columns = [c.rsplit(".", 1)[0] for c in wide.columns]
            self._panels[key] = wide.reindex(self.calendar)
        return self._panels[key]

    def panel_net_highs_lows(self, window: int, smooth: int) -> pd.DataFrame:
        return self._daily(T.net_highs_lows(self.members("close"), window, smooth))

    def panel_mcclellan_summation(self) -> pd.DataFrame:
        closes = self.members("close")
        start = closes.dropna(how="all").index.min()
        closes = closes.loc[start:]
        return self._daily(T.mcclellan_summation(closes, self.members("volume").loc[start:]))

    def panel_pct_above_ma(self, window: int) -> pd.DataFrame:
        return self._daily(T.pct_above_ma(self.members("close"), window))

    def panel_cftc_net_pct(self) -> pd.DataFrame:
        legs = []
        for alias in self.cfg.fetch.cftc.contracts.values():
            f = {k: self.obs("cftc", f"{alias}.{k}") for k in (
                "asset_mgr_positions_long", "asset_mgr_positions_short", "lev_money_positions_long",
                "lev_money_positions_short", "open_interest_all")}
            net = (f["asset_mgr_positions_long"] - f["asset_mgr_positions_short"] + f["lev_money_positions_long"]
                   - f["lev_money_positions_short"])
            legs.append(net / f["open_interest_all"] * 100)
        return self._daily(pd.concat(legs, axis=1).dropna().mean(axis=1), lag_days=3)

    def panel_top10_weight(self) -> pd.DataFrame:
        df = self.con.execute(
            "SELECT as_of, sum(weight) FROM (SELECT as_of, weight, row_number() OVER (PARTITION BY as_of ORDER BY "
            "weight DESC) AS rk FROM index_holdings WHERE fund = 'SPY') WHERE rk <= 10 GROUP BY as_of").df()
        s = pd.Series(df.iloc[:, 1].to_numpy(), index=pd.to_datetime(df["as_of"])).sort_index()
        return self._daily(s)   # end-of-day holdings: known at that close

    # -------------------------------------------------------------- AI basket
    def basket(self) -> dict:
        if "basket" not in self._panels:
            tickers = self.baskets.tickers
            wide = self.latest_many("yfinance", [f"{t}.close" for t in tickers + ["SMH", "SPY", "RSP", "KRW=X"]])
            wide.columns = [c[:-6] for c in wide.columns]
            wide = wide.ffill(limit=3).reindex(self.calendar)    # KRX and US holidays differ
            fx = {t: wide["KRW=X"] for t in tickers if t.endswith(".KS")}
            returns = T.usd_returns(wide[[t for t in tickers if t in wide]], fx)
            layers = {n: (layer.weight, layer.tickers) for n, layer in self.baskets.layers.items()}
            memory = {MEMORY_LAYER: (1.0, self.baskets.layers[MEMORY_LAYER].tickers)}
            self._panels["basket"] = {"closes": wide, "returns": returns, "index": T.basket_index(returns, layers),
                                      "memory": T.basket_index(returns, memory)}
        return self._panels["basket"]

    def panel_basket_trend(self) -> pd.DataFrame:
        idx = self.basket()["index"]
        return self._daily((T.pct_from_ma(idx, 50) + T.pct_from_ma(idx, 200)) / 2)

    def panel_basket_relative_strength(self, periods: int) -> pd.DataFrame:
        b = self.basket()
        smh = T.pct_change(b["closes"]["SMH"] / b["closes"]["SPY"], periods)
        basket = T.pct_change(b["index"] / b["closes"]["RSP"].reindex(b["index"].index), periods)
        return self._daily(pd.concat([smh, basket], axis=1).mean(axis=1, skipna=False))

    def panel_basket_breadth(self, window: int) -> pd.DataFrame:
        closes = self.basket()["closes"][[t for t in self.baskets.tickers if t in self.basket()["closes"]]]
        return self._daily(T.pct_above_ma(closes, window))

    def panel_basket_drawdown(self) -> pd.DataFrame:
        b = self.basket()
        return self._daily((T.drawdown(b["index"]) + T.drawdown(b["memory"])) / 2)

    def panel_basket_correlation(self, window: int) -> pd.DataFrame:
        return self._daily(T.avg_pairwise_corr(self.basket()["returns"], window))

    def memory_drawdown(self) -> pd.Series:
        return T.drawdown(self.basket()["memory"])

    def panel_basket_index_weight(self) -> pd.DataFrame:
        names = [t for t in self.baskets.tickers if "." not in t]
        df = self.con.execute(
            "SELECT as_of, sum(weight) FROM index_holdings WHERE fund = 'SPY' AND list_contains(?, ticker) "
            "GROUP BY as_of", [names]).df()
        s = pd.Series(df.iloc[:, 1].to_numpy(), index=pd.to_datetime(df["as_of"])).sort_index()
        return self._daily(s)

    def panel_basket_forward_pe(self) -> pd.DataFrame:
        wide = self.latest_many("yfinance", [f"{t}.forward_pe" for t in self.baskets.tickers])
        return self._daily(wide.where(wide > 0).median(axis=1))

    def panel_leveraged_etf_growth(self, symbols: list[str], periods: int) -> pd.DataFrame:
        growth = []
        for sym in symbols:
            assets, nav = self.obs("yfinance", f"{sym}.total_assets"), self.obs("yfinance", f"{sym}.nav")
            growth.append(T.pct_change((assets / nav).dropna(), periods))
        return self._daily(pd.concat(growth, axis=1).mean(axis=1))

    def panel_credit_proxy(self, symbols: list[str], window: int) -> pd.DataFrame:
        wide = self.latest_many("yfinance", [f"{s}.close" for s in symbols])
        dd = pd.concat([T.drawdown(wide[c].dropna(), window) for c in wide], axis=1)
        return self._daily(dd.mean(axis=1))

    # ---------------------------------------------------------------- options
    def chains(self, underlying: str) -> pd.DataFrame:
        df = self.con.execute("SELECT * FROM option_snapshots WHERE underlying = ?", [underlying]).df()
        if not df.empty:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
            df["expiry"] = pd.to_datetime(df["expiry"])
        return df

    def _per_snapshot(self, underlying: str, fn) -> pd.Series:
        out = {}
        for snap, chain in self.chains(underlying).groupby("snapshot_date"):
            spot = chain["underlying_price"].dropna()
            if spot.empty:
                continue
            v = fn(chain, float(spot.iloc[0]), snap)
            if v is not None:
                out[snap] = v
        return pd.Series(out, dtype=float)

    def panel_atm_iv(self, underlying: str, days: int) -> pd.DataFrame:
        return self._daily(self._per_snapshot(underlying, lambda c, s, d: T.atm_iv(c, s, d, days)))

    def panel_iv_spread(self, underlying: str, days: int) -> pd.DataFrame:
        iv = self._per_snapshot(underlying, lambda c, s, d: T.atm_iv(c, s, d, days))
        return self._daily(iv - self.obs("cboe", "VIX").reindex(iv.index))

    def panel_risk_reversal(self, underlyings: list[str], days: int) -> pd.DataFrame:
        parts = [self._per_snapshot(u, lambda c, s, d: T.risk_reversal(c, s, d, days)) for u in underlyings]
        return self._daily(pd.concat(parts, axis=1).mean(axis=1))

    def panel_call_put_volume(self, window: int) -> pd.DataFrame:
        names = [t for t in self.baskets.tickers if "." not in t]
        calls = self.latest_many("yfinance", [f"{t}.opt_call_volume" for t in names]).sum(axis=1)
        puts = self.latest_many("yfinance", [f"{t}.opt_put_volume" for t in names]).sum(axis=1)
        ratio = calls.rolling(window, min_periods=window).sum() / puts.rolling(window, min_periods=window).sum()
        return self._daily(ratio)

    # --------------------------------------------------------------- earnings
    def panel_beat_fade(self, days: int) -> pd.DataFrame:
        """Share of EPS beats whose 2-day reaction (close before report to close after) was negative."""
        closes = self.basket()["closes"]
        reports = self.con.execute(
            "SELECT ticker, report_date FROM earnings_events WHERE source = 'edgar_8k_202'").df()
        reports["report_date"] = pd.to_datetime(reports["report_date"])
        events = []
        for ticker in [t for t in self.baskets.tickers if "." not in t and t in closes]:
            actual, est = self.obs("finnhub", f"{ticker}.eps_actual"), self.obs("finnhub", f"{ticker}.eps_estimate")
            dates = sorted(reports.loc[reports["ticker"] == ticker, "report_date"])
            px = closes[ticker].dropna()
            for period in actual.index.intersection(est.index):
                if actual[period] <= est[period]:
                    continue
                report = next((d for d in dates if period < d <= period + pd.Timedelta(days=80)), None)
                if report is None:
                    continue
                before, after = px[px.index < report], px[px.index > report]
                if before.empty or after.empty:
                    continue
                events.append((after.index[0], float(after.iloc[0] / before.iloc[-1] - 1 < 0)))
        if not events:
            return self.to_calendar(pd.Series(dtype=float))
        daily = pd.DataFrame(events, columns=["date", "fade"]).groupby("date")["fade"].agg(["sum", "count"])
        daily = daily.reindex(self.calendar.union(daily.index), fill_value=0.0)
        window = daily.rolling(f"{days}D").sum()
        share = (window["sum"] / window["count"]).where(window["count"] >= 3) * 100
        return self._daily(share.reindex(self.calendar))

    # ------------------------------------------------------------------- XBRL
    def _quarters(self, ticker: str, tag: str) -> pd.DataFrame:
        facts = self.con.execute(
            "SELECT period_start, period_end, value, filed FROM xbrl_facts WHERE ticker = ? AND tag = ? AND unit = 'USD' "
            "AND taxonomy = 'us-gaap'", [ticker, tag]).df()
        if facts.empty:
            return pd.DataFrame(columns=["cq", "value", "filed"])
        q = T.quarters_from_ytd(facts)
        q["cq"] = q["period_end"].map(T.calendar_quarter)
        q["filed"] = pd.to_datetime(q["filed"])
        return q.drop_duplicates("cq", keep="last").set_index("cq")[["value", "filed"]]

    def _ttm(self, tags: dict[str, str]) -> pd.DataFrame:
        """Aggregate trailing-four-quarter sum across companies; avail = last filing needed."""
        parts = []
        for ticker, tag in tags.items():
            q = self._quarters(ticker, tag).sort_index()
            consecutive = (q.index.to_series().diff().dt.days.rolling(3).max() < 100)
            ttm = q["value"].rolling(4).sum().where(consecutive)
            # The current quarter's filing is the last one the TTM sum needs.
            parts.append(pd.DataFrame({f"{ticker}_v": ttm, f"{ticker}_f": q["filed"]}))
        df = pd.concat(parts, axis=1).dropna()
        value = df[[c for c in df if c.endswith("_v")]].sum(axis=1)
        avail = df[[c for c in df if c.endswith("_f")]].max(axis=1)
        return pd.DataFrame({"value": value, "avail": avail})

    def panel_xbrl_aggregate(self, metric: str) -> pd.DataFrame:
        if metric == "rpo_yoy":
            parts = []
            for ticker in RPO_COMPANIES:
                facts = self.con.execute(
                    "SELECT period_end, value, min(filed) AS filed FROM xbrl_facts WHERE ticker = ? AND tag = ? AND "
                    "period_start IS NULL GROUP BY period_end, value", [ticker, RPO_TAG]).df()
                facts = facts.sort_values("filed").drop_duplicates("period_end", keep="first")
                facts["cq"] = facts["period_end"].map(T.calendar_quarter)
                parts.append(facts.set_index("cq")[["value", "filed"]].add_prefix(f"{ticker}_"))
            df = pd.concat(parts, axis=1).dropna()
            total = df[[c for c in df if c.endswith("_value")]].sum(axis=1)
            frame = pd.DataFrame({"value": T.pct_change(total, 4),
                                  "avail": pd.to_datetime(df[[c for c in df if c.endswith("_filed")]].max(axis=1))})
        else:
            capex = self._ttm(CAPEX_TAGS)
            if metric == "capex_yoy":
                frame = pd.DataFrame({"value": T.pct_change(capex["value"], 4), "avail": capex["avail"]})
            else:
                ocf = self._ttm({t: OCF_TAG for t in CAPEX_TAGS})
                joined = capex.join(ocf, rsuffix="_ocf", how="inner")
                value = (joined["value"] / joined["value_ocf"] * 100 if metric == "capex_to_ocf"
                         else (joined["value_ocf"] - joined["value"]) / 1e9)
                frame = pd.DataFrame({"value": value, "avail": joined[["avail", "avail_ocf"]].max(axis=1)})
        frame = frame.dropna()
        return self.to_calendar(pd.Series(frame["value"].to_numpy(), index=pd.to_datetime(frame["avail"])),
                                pd.Series(frame.index, index=pd.to_datetime(frame["avail"])))

    # ------------------------------------------------------------- OpenRouter
    def daily_tokens(self) -> pd.Series:
        df = self.con.execute(
            "SELECT obs_date, sum(value) FROM (SELECT series, obs_date, arg_max(value, fetched_at) AS value "
            "FROM raw_observations WHERE source = 'openrouter' GROUP BY 1, 2) GROUP BY 1 ORDER BY 1").df()
        return pd.Series(df.iloc[:, 1].to_numpy(), index=pd.to_datetime(df["obs_date"]))

    def token_growth(self, window: int) -> pd.Series:
        tokens = self.daily_tokens()
        recent = tokens.rolling(window, min_periods=window).sum()
        return (recent / recent.shift(window) - 1) * 100

    def panel_token_growth(self, window: int) -> pd.DataFrame:
        return self._daily(self.token_growth(window), lag_days=1)

    def flagships(self) -> pd.DataFrame:
        """Per catalog snapshot: each lab's most expensive non-variant model from the last year."""
        cat = self.con.execute(
            "SELECT snapshot_date, model_id, author, created, prompt_price, completion_price FROM model_catalog "
            "WHERE list_contains(?, author) AND position(':' IN model_id) = 0 AND prompt_price > 0 "
            "AND completion_price > 0", [FLAGSHIP_LABS]).df()
        if cat.empty:
            return cat
        cat["snapshot_date"] = pd.to_datetime(cat["snapshot_date"])
        cat = cat[cat["created"] >= cat["snapshot_date"] - pd.Timedelta(days=365)]
        cat = cat.assign(blended=(3 * cat["prompt_price"] + cat["completion_price"]) / 4 * 1e6)
        return cat.loc[cat.groupby(["snapshot_date", "author"])["blended"].idxmax()]

    def token_price_index(self) -> pd.Series:
        f = self.flagships()
        return f.groupby("snapshot_date")["blended"].median() if not f.empty else pd.Series(dtype=float)

    def panel_token_price_index(self) -> pd.DataFrame:
        return self._daily(self.token_price_index())

    def panel_revenue_proxy(self, window: int) -> pd.DataFrame:
        price = self.token_price_index()
        if len(price) < 2:
            return self.to_calendar(pd.Series(dtype=float))
        price_change = T.pct_change(price.asfreq("D").ffill(), window)
        growth = self.token_growth(window)
        return self._daily((growth + price_change.reindex(growth.index)).dropna(), lag_days=1)

    def open_weight_share(self) -> pd.Series:
        """Share of tokens (models matched to the catalog) from models with published weights."""
        df = self.con.execute(
            """
            WITH t AS (SELECT series, obs_date, arg_max(value, fetched_at) AS value FROM raw_observations
                       WHERE source = 'openrouter' GROUP BY 1, 2),
                 m AS (SELECT DISTINCT canonical_slug, hugging_face_id IS NOT NULL AS open FROM model_catalog)
            SELECT t.obs_date, sum(t.value) FILTER (WHERE m.open) / sum(t.value) * 100 AS share
            FROM t JOIN m ON t.series = 'tokens:' || m.canonical_slug GROUP BY 1 ORDER BY 1
            """).df()
        return pd.Series(df["share"].to_numpy(), index=pd.to_datetime(df["obs_date"]))

    # ----------------------------------------------------------------- events
    def event_scores(self, ind: Indicator) -> pd.DataFrame:
        """Score (0-100) and status multiplier from events.yaml, as known on each day."""
        spec = ind.event
        entries = sorted((e for e in self.events if e.kpi == spec.kpi), key=lambda e: e.as_of)
        empty = pd.DataFrame(index=self.calendar, columns=["score", "as_of", "avail", "status_mult"], dtype=float)
        if not entries or (spec.categories is None and spec.score_map is None):
            return empty
        latest: dict[tuple, Event] = {}
        history: dict[tuple, list[Event]] = defaultdict(list)
        steps = {}
        for e in entries:
            key = (e.entity, e.metric, e.basis)
            history[key].append(e)
            latest[key] = e
            scores, weights = [], []
            for k, ev in latest.items():
                if spec.categories is not None:
                    if not isinstance(ev.value, str) or ev.value not in spec.categories:
                        continue
                    scores.append(spec.categories[ev.value])
                else:
                    value = ev.value
                    if spec.growth:
                        prior = [p for p in history[k] if p.period < ev.period and isinstance(p.value, float)]
                        if not prior or not isinstance(value, float):
                            continue
                        value = (value / prior[-1].value - 1) * 100
                    if not isinstance(value, float):
                        continue
                    scores.append(T.piecewise(value, spec.score_map))
                weights.append(ev.weight)
            if scores:
                steps[pd.Timestamp(e.as_of)] = (float(np.average(scores, weights=weights)), float(np.mean(weights)))
        if not steps:
            return empty
        idx = pd.DatetimeIndex(sorted(steps))
        frame = self.to_calendar(pd.Series([steps[d][0] for d in idx], index=idx))
        mult = self.to_calendar(pd.Series([steps[d][1] for d in idx], index=idx))["value"]
        return pd.DataFrame({"score": frame["value"], "as_of": frame["as_of"], "avail": frame["avail"],
                             "status_mult": mult})
