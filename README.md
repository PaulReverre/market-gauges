# Market Gauges

Contrarian buy/sell gauges for the S&P 500 and for AI and semiconductor stocks,
plus a fundamentals gauge for the next decade. Live at
<https://market-gauges.vercel.app/dashboard/>, rebuilt after every US session.

The dashboard shows three tiles. Select one for the detail behind it; select any
row in the detail for that input's own chart and track record.

| Tile | What it is | Horizon |
|---|---|---|
| **S&P 500** | 0–100 score from six volatility and options-positioning inputs | 1 to 12 months |
| **AI & semiconductors** | 0–100 score from four volatility inputs, judged against the PHLX Semiconductor index | 1 to 12 months |
| **Fundamentals** | 0–100 score (cheap to expensive) from five valuation measures, each read against all history and against its own last 30 years, with the 10-year return they imply | 5 to 10 years |

0 is extreme fear (historically the time to buy), 100 is complacency. A score of
10 or lower is the **buy zone**; 90 or higher is the **caution zone**.

Inside those two zones the scale is stretched (since 2026-11-21) so its ends stay
rare: a 5 is more extreme than 97.5% of past days and a 1 than 99.9%. Before,
the score was a plain percentile, so 1 meant "more fearful than 99% of days",
which every sharp sell-off reached. The zone edges and everything between them
are unchanged, so the same days count as buy and caution days and every
evidence table is the same; only how far the needle travels inside a zone
changed. On the 2026-11-20 close this moved the AI gauge from 1.3 to 3.6 and
the S&P 500 from 7.4 to 8.6. For the AI gauge, nothing since 1998 has reached 1;
March 2020 came closest at 1.4.

## Alerts and refreshing

- **Email alerts.** When a gauge's score first closes at or below its lower
  threshold, or at or above its upper one, the nightly job opens a GitHub issue
  assigned to the repository owner, and GitHub emails it. The issue closes
  itself (a second email) when the score leaves the range. Thresholds live in
  [`config/alerts.yaml`](config/alerts.yaml), default 10 and 90 for both gauges;
  the dashboard's status menu links straight to the editor. The dashboard also
  shows a banner while an alert is active and marks the thresholds on each dial.
- **Daily refresh.** GitHub Actions rebuilds the data after every US session
  (scheduled 02:00 UTC Tuesday to Saturday; GitHub often starts it some hours
  late) and retries later the same day if anything was missing.
- **Refresh button.** Re-reads the latest published data without reloading the
  page; an open tab also checks by itself every 10 minutes. It cannot make newer
  data exist: for that, the status menu links to "Run a rebuild now on GitHub".

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
- **Valuation is the long-horizon signal.** The blend of five valuation
  measures, fitted out of sample, missed the subsequent 10-year return by 2.5
  points a year on average, against 4.7 for the historical mean. It says almost
  nothing about the next year, and it has run too pessimistic since 2009.

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

**Checked on request (2026-11-10)**, none of which improved the score:

| Metric | Verdict |
|---|---|
| AAII sentiment (bulls minus bears, 4-week) | Shown as context. Modest edge on its own, same sign in both halves, but adding it cut the buy zone's 3-month edge from 3.0 to 1.8 points |
| BofA fund-manager cash | No free data exists. The stand-in, AAII's monthly cash allocation, is shown as context and has no edge at 1 to 3 months |
| NAAIM exposure | Not added. NAAIM's public feed now runs about three months behind, and with 2008 in the sample its low readings were not followed by better returns |
| High-yield credit spreads | Shown as context (FRED serves only three years). Its long-history cousin, the Baa spread, fails as a signal but works as a **credit check**: buy-zone days with credit blowing out returned about half as much over 12 months as the rest |
| Equity fund flows | Shown as context from the Fed's quarterly accounts. Heavy outflows preceded better returns, but the data arrives 10 weeks late; weekly ICI data is closed to scripts |

**Tested on request (2026-11-21)**: about 30 more candidates for the S&P 500
gauge and 17 for the AI gauge, each judged alone, added to the gauge, and at 18
named market lows since 2002. None made the scores reliably better, so none was
added to a score:

| Candidate | Result |
|---|---|
| Credit default swaps (CDX) | Licensed data; no free source. Tested the free proxies instead |
| High-yield bonds vs Treasuries (HYG/IEF), 1 and 3 months, and drawdown | Strong alone since 2009, and it looked like an improvement on that sample. Ranked early enough to include 2008, it said buy months too early and made the buy zone worse. **Shown as context on both gauges** |
| Baa spread change, Baa minus Aaa, investment grade vs Treasuries, St. Louis Fed and Chicago Fed stress indexes, TED spread | No edge alone; each made the gauge worse |
| MOVE (Treasury volatility), 10-year yield moves, Treasuries vs stocks | Weak or no edge alone; made the gauge worse |
| 9-day VIX / VIX | Strong alone on both sides since 2013, but adds nothing to VIX and its term structure. **Shown as context** |
| Low-volatility vs high-beta stocks | Strong alone since 2013, duplicates the volatility inputs. **Shown as context** |
| Capitulation: SPY volume surge, share of stocks falling | Pass alone in both halves, but made the buy zone fire more often with weaker results. **Shown as context** |
| Down-volume share, 90% down days, new 52-week lows, S&P 500 volume | Weak or no edge alone; made the gauge worse |
| Sector rotation: utilities or staples vs discretionary, small vs large, gold vs stocks | Weak or no edge alone; made the gauge worse |
| Jobless claims | Made the gauge worse |
| Yield curve (10-year minus 3-month) | Improved both gauges' statistics, especially the caution side. Rejected: the S&P gain rested on two episodes (2002, 2021), and on the AI gauge it moved buy signals into the 2023–25 inverted-curve bull market, a coincidence of one macro cycle rather than a fear signal |
| AI: SMH, SOXX and NVDA volume surges | SOXX helped, the near-identical SMH hurt badly: the "gain" is noise |
| AI: semis vs S&P volatility, VXN / VIX, VVIX, 9-day VIX, tech or Nasdaq-100 relative strength, market-wide put/call and implied correlation | Each made the AI gauge worse |
| Margin debt | FINRA now blocks automated downloads of its statistics |

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
| S&P 500 valuation panel (forward P/E, ERP, dividend yield, equities/GDP) | Replaced by the fundamentals gauge, which scores the measures that predict and shows the rest with the reason |

The AI company table and the catalyst calendar are kept, behind the AI gauge.

## The fundamentals gauge

Every candidate was judged the same way on monthly data since 1951: a
one-variable regression against the S&P 500's subsequent 10-year total return,
refitted out of sample using only decades that had already finished, and
required to hold the same sign before and after 1988. A measure is scored only
if its out-of-sample miss beats simply assuming the historical average.

| Measure | Verdict | Out-of-sample miss (average: 4.9 pts a year) |
|---|---|---|
| Investor equity allocation (Fed Z.1) | **Scored.** Best single predictor | 2.6 |
| Tobin's Q | **Scored.** Weaker since 1988 | 3.5 |
| Equity risk premium, as Shiller's excess CAPE yield | **Scored** | 3.9 (average 4.7) |
| Shiller CAPE | **Scored.** Same strength in both halves | 4.3 |
| Shareholder yield (dividends + net buybacks) | **Scored**, modestly | 4.1 |
| Trailing P/E | Not scored: barely beats the average, weak since 1988 | 4.7 |
| Dividend yield | Not scored: payouts moved to buybacks | 5.0 |
| Buffett indicator (market value / GDP) | Not scored: right direction, level keeps drifting up | 5.5 |
| Price to sales (market value / gross value added) | Not scored: same as Buffett | 5.7 |
| ERP as earnings yield minus Treasury yield (Fed model) | Rejected: no relation, wrong way out of sample | 6.2 |
| Net buyback yield alone | Rejected | 9.9 |
| Net profit margin; corporate profits / GDP | Rejected: no relation to later returns at any horizon | 6.2; 5.7 |
| Trailing EPS growth | Rejected: no relation | 5.4 |
| Forward P/E; free cash flow yield; forward EPS growth and revisions | Not testable: no free history | — |

### Two yardsticks, and what was tried to make the gauge less bearish

Each scored measure is read twice: against **all history since 1951**, and
against **its own trailing 30-year average**. The second was added on
2026-11-11 after checking the gauge's recent record: for decades starting
2009–2016, every one of the five models under-predicted, by 4.4 points a year
on average. Out of sample the two yardsticks are about equally accurate (2.7
points a year since 1983 for all history, 2.6 since 1998 for the 30-year one;
on the years both cover the 30-year one is better, 2.6 against 3.2), and the
blend is better than either (2.5 against 4.7 for the average). The 0–100 score
is the mean of the ten ranks; the expected return is the mean of the two
yardsticks' forecasts.

What that changed, on the 2026-11-10 close: score 96 → 91, expected return
+3.7% → +5.4% a year (+3.7% against all history, +7.1% against the last 30
years), and today now reads below the July 2000 peak (98) rather than above it.

Tested at the same time and **not** adopted, because the data did not support
them:

| Idea | Result |
|---|---|
| Credit the market for strong earnings growth and wide margins | Neither has any relation to later returns alone, and adding either to a valuation model made its forecasts worse (margins helped one model slightly and hurt another) |
| Credit calm credit spreads | No improvement to the forecast |
| "Expensive but supported": split expensive months by earnings growing and credit calm | A small edge (about 2 points a year over the next year). It did not prevent 2000, when two thirds of months met both conditions. Shown on the dashboard as a check, not scored |
| Add back the models' recent error | Made past forecasts worse: errors reverse rather than persist (at five years, the 1990s over-shoot was repaid in the 2000s) |
| Damodaran's implied equity risk premium + Treasury yield (forward-looking; currently implies about 9% a year) | Fits the full sample since 1961, but out of sample since 1992 it missed by 4.0 points a year against 3.8 for the average. Shown as context |
| Forward P/E | Still untestable: no free history |

Also tested on 2026-11-21 and rejected with the same checks, all listed on the
dashboard: equity share of new financing (Baker-Wurgler issuance), net equity
issuance (dilution), corporate debt / GDP, corporate leverage, interest
coverage, money-market funds / equities, earnings yield minus Baa yield, the
10-year Treasury yield and real yield, the Baa spread, bank credit and M2
growth. None predicted 10-year returns, and each made the five-measure blend's
forecast worse.

Ranking the inputs as percentiles before fitting was also tried and rejected: it
discards how far beyond its history a reading is. Index-level price to sales,
margins and Tobin's Q use the Fed's and BEA's nonfinancial-corporate accounts,
the free equivalents of the S&P 500 figures. This gauge sends no alerts and
shows no buy or caution signal: it moves over years and says nothing about
timing.

## How it runs

```
.github/workflows/refresh.yml   02:00 UTC Tue–Sat (+ 10:00 UTC retry)
  └─ python -m signals.build
       fetch    Yahoo (indices, 503 members, AI basket) · Cboe (COR1M, VVIX, SKEW, put/call)
                FRED (Z.1, corporate accounts, Baa, HY, 10y) · HYG, IEF, SPLV, SPHB, 9-day VIX, SPY volume (Yahoo) · shillerdata.com · Damodaran (NYU) · AAII · SSGA · Nebius · Finnhub
       score    src/signals/gauges.py, model.py, fundamentals.py
       write    data/signals.json, data/signals_history.json, data/state/*
  └─ python -m signals.alerts --send   (GitHub issue when a threshold is crossed)
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
                fundamentals.py · companies.py · alerts.py · build.py
dashboard/      index.html (static; ECharts from cdnjs)
config/         alerts.yaml (thresholds) · calendar.yaml (macro dates for the catalyst list)
data/           signals.json · signals_history.json · state/
tests/          test_signals.py
```

The earlier four-gauge pipeline (v1) was removed on 2026-11-10. Its code, tests,
configuration and the `data/archive` parquet history are preserved at the git tag
`v1-final` (`git checkout v1-final` to look, `git checkout main` to come back).

Not investment advice.
