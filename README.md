# Market Gauges

Contrarian buy/sell gauges for the S&P 500 and for AI and semiconductor stocks,
plus a slow valuation read on the next decade. Live at
<https://market-gauges.vercel.app/dashboard/>, rebuilt after every US session.

The dashboard shows three tiles. Select one for the detail behind it; select any
row in the detail for that input's own chart and track record.

| Tile | What it is | Horizon |
|---|---|---|
| **S&P 500** | 0–100 score from six volatility and options-positioning inputs | 1 to 12 months |
| **AI & semiconductors** | 0–100 score from four volatility inputs, judged against the PHLX Semiconductor index | 1 to 12 months |
| **S&P 500 · next 10 years** | Expected annual return implied by investor equity allocation and CAPE | 10 years |

0 is extreme fear (historically the time to buy), 100 is complacency. A score of
10 or lower is the **buy zone**; 90 or higher is the **caution zone**.

## What "predictive" means here

No market metric determines future returns. What this project does instead:

- **Point in time.** Each input is ranked against its own history using only
  data available on that day (expanding percentile, 2 years minimum). The score
  is the average of those ranks within each group, then across groups, ranked
  the same way.
- **Odds, recomputed daily.** For every score zone the build measures what the
  benchmark did 1, 3, 6 and 12 months later on every past day in that zone:
  median, share positive, the middle 80% of outcomes, and the number of separate
  episodes (the honest sample size, since daily windows overlap).
- **Significance that respects clustering.** The p-value for a zone's excess
  return comes from circularly shifting the signal against returns 400 times,
  which keeps both series' autocorrelation and breaks only their alignment.
- **Split-half check.** An edge has to have the same sign before and after 2010
  to be graded "strong" or "moderate".

What that found (figures from the 2026-11-09 build; the dashboard always shows
current ones):

- **Buy side is real.** S&P 500 score ≤ 10: 51 episodes since 1994, +5.5% mean
  over the next 3 months against +2.5% on any day, positive 79% of the time
  against 70%, same sign in both halves, p < 0.01. Semis: +17.3% against +4.9%.
- **Sell side is weak.** S&P 500 score ≥ 90: +0.7% over 3 months against +2.5%
  (p ≈ 0.04). Returns after complacency were below average, not negative. For
  semis it is not statistically distinguishable from chance at 3 months.
- **Valuation is the long-horizon signal.** Fitted out of sample, investor
  equity allocation missed the subsequent 10-year return by 2.6 points a year on
  average, against 4.9 for the historical mean.

Caveat that applies to all of it: inputs and zone edges were chosen on this same
history, which flatters every number to some degree.

## Every metric, and what happened to it

**In the S&P 500 score**

| Input | Why |
|---|---|
| VIX | Most dependable single fear reading; passes alone in both halves |
| VIX IV percentile (1 year) | Fear relative to the recent regime; makes the calm side of the score work |
| VIX / 3-month VIX | Backwardation = panic. History from 2006, so 2008 weighs on it |
| VIX minus 20-day realized volatility | Price of protection; passes alone in both halves |
| Equity put/call ratio, 10-day | What investors are doing, not what options cost |
| Implied correlation (COR1M) | Everything being sold at once |

**In the AI & semiconductors score**: semiconductor 20-day realized volatility,
VXN, VIX, VIX / 3-month VIX. Semis bottom when the whole market panics; the
sector's own realized volatility is what keeps the combination working before
2010.

**Tested and shown, not scored** (each has its own evidence on the dashboard):
VIX IV rank, share of S&P 500 members above their 50-day average and with RSI
below 30 (good 1-month confirmation, but they weakened the score at 3–12 months),
VVIX, SKEW, VXN minus realized volatility, NVDA realized volatility, AI basket
breadth, Nebius GPU spot prices (too little history to test).

**Rejected as signals**: drawdown from the 52-week high and distance from the
200-day average (buying them lost money in 2000–02 and 2008), credit spreads,
semis' relative strength.

**Dropped from the old four-gauge dashboard**

| Old metric(s) | Reason |
|---|---|
| Fundamental-state gauges (growth, earnings, rates, consumer; AI spend, supply chain, monetization, financing) | Coincident, not contrarian; most inputs had already failed the old backtest; 30–56% data coverage |
| Fragility overlay | The old robustness study found no link to later drawdowns or returns |
| CNN-method replica, AI unwind lights, AI fragility flags | Display-only extras with nothing behind them to test |
| AAII, NAAIM | No free history (AAII) or a source that went stale (NAAIM) |
| Margin debt, MOVE, NFCI, credit spreads, copper/gold, dollar | Wrong sign or no edge against forward returns, here or in the old backtest |
| SOFR−IORB, HY/IG and CCC/BB spreads | Too little history to test (2021 and 2023 onward) |
| CFTC positioning | Weekly and lagged; not retested |
| RSP/SPY, IWM/SPY, SPHB/SPLV, S&P vs 125-day average | Trend and relative-strength measures: coincident, not contrarian |
| McClellan summation, net highs−lows | Breadth, now covered by the two breadth context metrics |
| NVDA option IV and skew, option-volume ratios, leveraged-ETF flows | Weeks of history only |
| Wikipedia and GDELT attention, earnings-release tone | Failed the old backtest, or sources that kept breaking |
| S&P 500 valuation panel (forward P/E, ERP, dividend yield, equities/GDP) | Replaced by the 10-year layer, which keeps the two measures that predict |

The AI company table and the catalyst calendar are kept, behind the AI gauge.

## How it runs

```
.github/workflows/refresh.yml   02:00 UTC Tue–Sat (+ 10:00 UTC retry)
  └─ python -m signals.build
       fetch    Yahoo (indices, 503 members, AI basket) · Cboe (COR1M, VVIX, SKEW, put/call)
                FRED (Z.1, Baa, 10y) · shillerdata.com · SSGA · Nebius · Finnhub
       score    src/signals/gauges.py, model.py, longterm.py
       write    data/signals.json, data/signals_history.json, data/state/*
  └─ commit → Vercel redeploys dashboard/index.html
```

The build re-downloads full history every run, so there is no database and a
missed day heals itself. `data/state/` holds the two series that cannot be
re-downloaded (Cboe put/call, Nebius spot) and a last-good copy of the slow
sources; if a source fails, the build uses that copy and the dashboard's
"Sources" menu says so. Only a Yahoo failure stops the build, leaving the
previous day's file in place (GitHub emails you about the failed run).

Cost: nothing. No key is needed for the scores. `FRED_API_KEY` makes FRED faster
and `FINNHUB_API_KEY` fills the earnings columns of the AI table; both are free
and already set as repository secrets.

## Local use

```bash
make setup      # Python 3.11 venv + dependencies
make signals    # rebuild data/signals.json (about 3 minutes)
make serve      # http://127.0.0.1:8765/dashboard/
make test       # unit tests
```

The bot commits `data/` every weekday: `git pull` before working, and do not
commit a locally built `data/signals.json` unless you mean to publish it.

## Layout

```
src/signals/    fetch.py · gauges.py (indicator definitions) · model.py (scoring, odds, evidence)
                longterm.py · companies.py · build.py
dashboard/      index.html (static; ECharts from cdnjs)
config/         calendar.yaml (macro dates for the catalyst list)
data/           signals.json · signals_history.json · state/
tests/          test_signals.py
```

The earlier four-gauge pipeline (v1) was removed on 2026-11-10. Its code, tests,
configuration and the `data/archive` parquet history are preserved at the git tag
`v1-final` (`git checkout v1-final` to look, `git checkout main` to come back).

Not investment advice.
