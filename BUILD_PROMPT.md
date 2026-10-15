# Build Spec: Equity and AI Sentiment / State Gauges

**How to use:** save this file as `BUILD_PROMPT.md` at the root of an empty repo, open Claude Code there, and say:
"Read BUILD_PROMPT.md. Enter plan mode. Execute Phase 0 only, then stop and report."

Prepared 2026-10-15. Seed values in Section 12 come from prior research and must be re-verified by the pipeline, never trusted as-is.

---

## 1. Role and operating standard

Act as two people at once:

1. A senior cross-asset volatility and sentiment strategist who knows which indicators lead drawdowns, which are coincident, and which only work as contrarian signals at extremes.
2. A careful data engineer whose first loyalty is data integrity.

Operating rules:

- No number appears on the dashboard unless it came from a fetched source, or from an `events.yaml` entry that carries a source URL. Never type a market value from memory.
- Every value carries: source, endpoint or series id, source as-of date, fetch timestamp, expected frequency, staleness status.
- When a source fails, show the gap. Never interpolate silently, never backfill with a guess.
- Distinguish reported figures from guidance, and primary sources from secondhand reporting.
- If an indicator fails validation (Section 8), demote it to display-only. Do not keep it in a composite because it sounds right.
- Ask before adding any paid data source.

---

## 2. What to build

A daily-refreshing dashboard: two rows, two gauges per row, each scored 0 to 100.

| Row | Left: Sentiment & Volatility | Right: Fundamental State |
|---|---|---|
| US Equity (S&P 500) | **G1.** Vol structure, trend and breadth, credit appetite, positioning. 0 = extreme fear, 100 = extreme greed. Includes a CNN-method replica sub-score and a separate **Fragility overlay**. | **G2.** Growth, earnings, valuation, rates and liquidity, consumer credit. 0 = breaking down, 100 = very strong. |
| AI Sector | **G3.** AI basket trend, implied vol and skew, crowding, attention. 0 = extreme fear, 100 = euphoria. | **G4.** AI health: spend, supply chain throughput, monetization, compute economics, financing and credit, valuation. Plus the 4-light **Unwind Signature** panel. |

**Design principle the build must respect.** Sentiment composites, CNN Fear & Greed included, are mostly coincident and work as contrarian signals at the extremes: extreme fear has tended to precede above-average forward returns. Downside prediction lives in a narrower set of leading fragility indicators (credit spread changes, vol term structure, breadth divergence, funding stress, leverage growth). So the "correlates with downside" requirement is met by the Fragility overlay and the G1/G2 divergence logic, validated by backtest. Do not make the headline needle bearish to force it.

---

## 3. Non-negotiable data rules

1. **Point-in-time.** Backtests use only data available on each date. Use ALFRED vintages for revised macro series. Lag monthly and quarterly releases to their actual release date, not period end.
2. **Accumulate or lose.** Several free sources only expose recent data (Section 5). The daily job persists every observation from day one. Option-implied vol history is not free anywhere, so snapshot it daily from the first run.
3. **Staleness.** Each indicator has `expected_freq`. Weight decays linearly from 100% at 1.5x expected age to 0% at 3x. Stale components render grey with their age shown.
4. **Coverage.** If a gauge's live weight falls below 70% of target, the needle renders hatched with "low coverage" and any narrative must say so.
5. **Cross-checks.** SPX close (yfinance vs FRED `SP500`) within 0.15%. VIX close (Cboe CSV vs FRED `VIXCLS`) exact. 10y yield (FRED `DGS10` vs Treasury) within 2bp. Mismatch: keep the primary, log both, flag.
6. **Sanity bounds** per indicator in the registry (e.g., VIX 5 to 150, HY OAS 1 to 25, forward P/E 5 to 60). Out of bounds: reject and flag.
7. **Manual inputs** live only in `events.yaml` and must include `source_url`, `as_of`, and `status` (`primary` | `guidance` | `secondhand` | `estimate`). Schema validation rejects anything else. Weight multipliers: primary 1.0, guidance 0.8, secondhand 0.6, estimate 0.3.
8. **No LLM-generated numbers** anywhere in the pipeline. The Claude skill may propose `events.yaml` entries with sources for human approval. It never writes values directly into the store.

---

## 4. Architecture

```
market-gauges/
  config/
    indicators.yaml        # registry: source, transform, direction, weight, bounds, freq, role
    baskets.yaml           # AI basket tickers by layer
    events.yaml            # manual / event KPIs with sources
    calendar.yaml          # FOMC and other fixed macro dates
  src/gauges/
    sources/               # one client per source: fred, alfred, cboe, yf, edgar, finnhub,
                           # openrouter, cftc, census, naaim, aaii, finra, spdji, shiller,
                           # tsmc, mops, gpu_index, gdelt, wikimedia, eia
    ingest.py              # pulls and persists raw observations
    transforms.py          # pure functions, unit tested
    score.py               # percentiles, orientation, staleness, weights, composites
    validate.py            # cross-checks, bounds, schema
    backtest.py            # Section 8
    render.py              # writes data/latest.json and data/history.json
  data/
    gauges.duckdb          # raw (append-only) + derived
    latest.json
    history.json
  dashboard/index.html     # static, reads the JSON files
  reports/validation.md
  scripts/probe_sources.py # Phase 0
  tests/
  .claude/skills/market-gauges/SKILL.md
  .github/workflows/daily.yml
  .env.example
  Makefile                 # probe | ingest | score | render | refresh | backtest | test
```

**Stack:** Python 3.11+, httpx + tenacity (retries, backoff), requests-cache, pandas, numpy, scipy, duckdb, pydantic v2, pyyaml, yfinance, openpyxl, jinja2, pytest. Dashboard: one HTML file, ECharts from cdnjs, no build step, dark mode via `prefers-color-scheme`, usable at phone width.

**Registry entry schema** (`config/indicators.yaml`):

```yaml
- id: vol_ts
  gauge: G1
  pillar: volatility
  name: VIX / VIX3M term structure
  source: cboe_csv
  series: [VIX, VIX3M]
  transform: ratio              # VIX / VIX3M
  lookback_pct: 5y              # percentile window
  direction: lower_is_greed     # orientation onto 0-100
  weight: 6
  expected_freq: 1d
  bounds: [0.5, 1.6]
  role: leading                 # leading | coincident | contrarian | display
  fragility: true               # member of the fragility overlay
  notes: Ratio above 1.0 is backwardation, historically tied to acute stress.
```

---

## 5. Data sources (free tiers only)

| Source | Key | Limits and caveats | Feeds |
|---|---|---|---|
| FRED API + ALFRED | free key | ICE BofA spread series on FRED carry only 3 years of history since April 2026. Pull pre-cutoff history from an ALFRED vintage dated before late April 2026, then append daily. S&P 500 series on FRED carry 10 years of daily history. | Credit spreads, rates, macro, NFCI, STLFSI4, VIXCLS, VXNCLS, Ramp AI Index series |
| Cboe index CSVs, pattern `cdn.cboe.com/api/global/us_indices/daily_prices/{SYMBOL}_History.csv` | none | Confirm each symbol in Phase 0: VIX, VIX3M, VIX9D, VVIX, SKEW, and implied correlation (COR1M) if offered. | Vol structure |
| Cboe Daily Market Statistics page | none | Free put/call archive files stop in 2019. Scrape the daily page every day and store. Backfill post-2019 only from a verified free source (Alphacast lists CBOE equity P/C from 2006 behind a free account; verify terms). | Total, equity, index put/call |
| yfinance | none | Unofficial, rate-limited, can break without notice. One batched pull per day, cache, back off, cross-check. | EOD prices (S&P constituents, ETFs, AI basket, futures, ^MOVE), option chains for IV and skew, forward P/E |
| SEC EDGAR (companyfacts, companyconcept, submissions) | none; declared User-Agent with email required, stay under 10 req/s | XBRL tags vary by company. Map per company, unit test, document. | Capex, OCF, RPO, D&A, Form 4 |
| Finnhub | free key | About 60 calls/min, personal non-commercial use | Earnings calendar, EPS surprises, quote cross-check, news |
| OpenRouter Data API | free key | 30 req/min, 500/day. CC BY 4.0, must cite "Source: OpenRouter (openrouter.ai/rankings), as of {as_of}". Token counts are not comparable across providers' tokenizers. Skews to cost-sensitive developers. | Daily tokens for top 50 models, share by author, open-weight share; `/models` for a price index |
| CFTC Public Reporting (Socrata) | none (app token optional) | Weekly; Tuesday positions released Friday | Traders in Financial Futures, ES and NQ |
| Census BTOS | none | Biweekly. AI-use question was broadened on 2025-11-17: treat as a series break. | US business AI use rate |
| NAAIM Exposure Index | none | Weekly xlsx | Active manager exposure |
| AAII Sentiment Survey | none for recent weeks | Full history is members-only. Scrape the public recent-results table weekly and accumulate. | Bull minus bear |
| FINRA margin statistics | none | Monthly, ~3-week lag | Margin debt YoY |
| S&P DJI `sp-500-eps-est.xlsx` | none | Snapshot every release to build your own revisions history | Forward and trailing operating EPS, P/E, margins |
| Shiller data | none | Monthly | CAPE |
| TSMC IR monthly revenue; TWSE MOPS monthly revenue | none | Scrape; released around the 10th | TSMC, Alchip, Wiwynn, Quanta, Wistron |
| Uvik GPU Price Index (CSV / JSON feed) | none | Verify feed URL and terms in Phase 0 | On-demand GPU rental medians |
| SemiAnalysis public H100 1-year rental index | none | Public dashboard; check terms before scraping, else monthly manual entry | Contract rental trend |
| GDELT 2.0 DOC API | none | Noisy; low weight | AI and chip news tone |
| Wikimedia Pageviews API | none | Official and stable | Attention proxies |
| EIA API | free key | Monthly | Power demand (optional) |
| Massive (formerly Polygon) Basic | free key | 5 calls/min, end-of-day, 2 years of history | Price cross-check only |
| FMP Basic | free key | 250 calls/day; free keys limited to a fixed symbol list | Cross-check only |
| Alpha Vantage | free key | 25 calls/day | Skip unless a specific gap needs it |

**Not free. Do not fake; log as gaps:** single-name CDS (ORCL, CRWV), TrendForce HBM/DRAM contract prices, ISM PMI, EPS revision breadth, official MOVE history, historical option IV, full AAII history.

---

## 6. Indicator registry (starting point)

Orientation: for G1 and G3, "high = greed" means a higher raw value maps toward 100. For G2 and G4, "high = healthy".

### 6.1 G1: Equity Sentiment & Volatility

Pillar weights: Volatility & options 30, Trend & breadth 30, Credit & funding 20, Positioning & surveys 15, Safe haven & cross-asset 5.

| ID | Indicator | Source | Transform | Orientation | Role | CNN |
|---|---|---|---|---|---|---|
| eq_mom125 | S&P 500 vs 125-day MA | yfinance ^GSPC, FRED SP500 | % distance | high = greed | coincident | yes |
| eq_nh_nl | Net 52-week highs minus lows, S&P 500 constituents | yfinance | count, 10d avg | high = greed | coincident | analog |
| eq_mcclellan | McClellan Summation analog on S&P 500 advancing vs declining volume | yfinance | standard McClellan | high = greed | coincident | analog |
| eq_pc5 | Total put/call, 5-day avg | Cboe daily stats | avg | low = greed | contrarian | yes |
| eq_junk | HY minus IG effective yield | FRED BAMLH0A0HYM2EY, BAMLC0A0CMEY | spread | low = greed | coincident | yes |
| eq_vix50 | VIX vs its 50-day MA | Cboe, FRED VIXCLS | % distance | low = greed | coincident | yes |
| eq_safehaven | 20d S&P return minus 20d Treasury return | yfinance ^GSPC, IEF | diff | high = greed | coincident | yes |
| vol_ts | VIX / VIX3M | Cboe | ratio | low = greed | leading | |
| vol_vrp | VIX minus 20d realized SPX vol | Cboe + yfinance | diff | low VRP = greed (complacency) | leading | |
| vol_vvix | VVIX and VVIX/VIX | Cboe | level, ratio | low = greed | leading | |
| vol_9d | VIX9D / VIX | Cboe | ratio | low = greed | coincident | |
| vol_move | MOVE | yfinance ^MOVE (unofficial) | percentile | low = greed | leading | |
| vol_cor | 1M implied correlation (if Cboe CSV verified) | Cboe | percentile | display + fragility | leading | |
| eq_pc_equity | Equity-only put/call, 10d avg | Cboe daily stats | avg | low = greed | contrarian | |
| br_200 | % of S&P 500 above 200-DMA | yfinance | % | high = greed | leading when diverging | |
| br_ew | RSP / SPY, 63d change | yfinance | % change | high = greed | leading | |
| br_beta | SPHB / SPLV, 63d change | yfinance | % change | high = greed | coincident | |
| br_small | IWM / SPY, 63d change | yfinance | % change | high = greed | coincident | |
| cr_hy_chg | HY OAS 21d change (bp) | FRED BAMLH0A0HYM2 + ALFRED | diff | low = greed | leading | |
| cr_ccc_bb | CCC OAS minus BB OAS | FRED BAMLH0A3HYC, BAMLH0A1HYBB | spread | low = greed | leading | |
| cr_nfci | Chicago Fed NFCI, 4w change | FRED NFCI | diff | low = greed | leading | |
| fund_sofr | SOFR minus IORB, 5d max | FRED SOFR, IORB | diff | low = greed | leading | |
| pos_aaii | AAII bull minus bear, 4w avg | AAII | spread | high = greed | contrarian | |
| pos_naaim | NAAIM exposure | NAAIM | level | high = greed | contrarian | |
| pos_cftc | Asset manager + leveraged fund net ES+NQ, % of OI | CFTC TFF | z-score | high = greed | contrarian | |
| pos_margin | FINRA margin debt YoY | FINRA | % | high = greed | leading (slow) | |
| xa_dxy | Broad dollar 21d change | FRED DTWEXBGS | % | low = greed | coincident | |
| xa_cugold | Copper/gold ratio 63d change | yfinance HG=F, GC=F | % | high = greed | coincident | |

Also output `cnn_replica`: equal-weight of the seven CNN-flagged rows, for spot-checking against CNN's published value. Do not scrape CNN (robots disallowed). Comparison is manual.

### 6.2 Equity Fragility overlay

Separate from G1. Candidate members: `vol_ts`, `vol_vrp`, `vol_vvix`, `vol_cor`, `cr_hy_chg`, `cr_ccc_bb`, `cr_nfci`, `fund_sofr`, `br_ew`, `pos_margin`, plus a breadth-divergence flag (S&P within 3% of its 52-week high while `br_200` is below 55%).

Output: fragility score 0 to 100, and from the backtest, the empirical probability of a 10%+ S&P drawdown within the next 63 trading days for the current fragility decile, with sample count. If Section 8 shows no monotonic relationship, render "Not validated" instead of a probability.

Banners:
- G1 >= 70 and fragility >= 70: **"Complacent and fragile"** (highest-risk combination).
- G1 <= 25 and G2 >= 50: **"Fear without fundamental damage"** (historically favorable).
- G1 >= 70 and G2 down three consecutive weeks: **"Sentiment ahead of fundamentals."**

### 6.3 G2: Equity Fundamental State

Pillar weights: Growth & cycle 30, Earnings 25, Valuation 20, Rates & liquidity 15, Consumer & credit 10.

| ID | Indicator | Source | Transform | Orientation |
|---|---|---|---|---|
| gr_gdpnow | Atlanta Fed GDPNow | FRED GDPNOW (verify id) | level | high = healthy |
| gr_cfnai | CFNAI 3-month avg | FRED CFNAIMA3 | level | high |
| gr_regional | Regional Fed manufacturing composite (Philly, NY, Richmond, KC, Dallas) | FRED (verify ids) | mean of z-scores | high |
| gr_claims | Initial claims 4w avg, YoY | FRED ICSA | % | low = healthy |
| gr_sahm | Sahm rule, real-time | FRED SAHMREALTIME | level | low = healthy |
| gr_permits | Building permits YoY | FRED PERMIT | % | high |
| ea_fwd_rev | Next-year S&P operating EPS estimate, 13w change | S&P DJI xlsx snapshots | % | high |
| ea_growth | NTM EPS growth | S&P DJI xlsx | % | high |
| ea_margin | Operating margin, 4-quarter change | S&P DJI xlsx | diff | high |
| ea_beats | % beating, latest season | events.yaml from FactSet Earnings Insight (free PDF) | level | high |
| va_fpe | Forward P/E percentile, 20y | S&P DJI xlsx + price | percentile | low = healthy |
| va_erp | Forward earnings yield minus 10y real yield | xlsx + FRED DFII10 | diff | high = healthy |
| va_cape | Shiller CAPE percentile | Shiller | percentile | low = healthy |
| va_eq_gdp | Corporate equities / GDP | FRED NCBEILQ027S / GDP (do not rely on Wilshire series on FRED; verify) | ratio percentile | low = healthy |
| va_conc | Top-10 weight in S&P 500 | yfinance market caps | % | low = healthy |
| rl_real_rate | Fed funds minus core PCE YoY | FRED DFF, PCEPILFE | diff | low = healthy |
| rl_real10 | 10y real yield, 63d change | FRED DFII10 | diff | low = healthy |
| rl_curve | 10y minus 3m, with re-steepening-after-inversion flag | FRED T10Y3M | level + flag | context |
| rl_liq | Net liquidity (WALCL minus WTREGEN minus RRPONTSYD), 13w change | FRED | diff | high = healthy (low weight; correlation is debated) |
| rl_sloos | Net % of banks tightening C&I standards | FRED DRTSCILM | level | low = healthy |
| cc_retail | Real retail sales YoY | FRED RRSFS | % | high |
| cc_delinq | Credit card delinquency rate, YoY change | FRED DRCCLACBS | diff | low = healthy |
| cc_umich | UMich consumer sentiment | FRED UMCSENT | level | high |
| infl | Core PCE YoY and 10y breakeven | FRED PCEPILFE, T10YIE | display; penalty on rl pillar when core PCE > 3% | |

### 6.4 AI basket (`config/baskets.yaml`)

Equal-weight within layer; layer weights in brackets (sum 100):

- Compute & networking [25]: NVDA, AMD, AVGO, MRVL, ANET, ARM
- Foundry & equipment [15]: TSM, ASML, LRCX, AMAT
- Memory [15]: MU, 000660.KS (SK Hynix), 005930.KS (Samsung)
- Hyperscalers [25]: MSFT, GOOGL, AMZN, META, ORCL
- Neocloud & infra [10]: CRWV, NBIS, VRT
- Power [10]: GEV, CEG, VST, BE

Reference ETFs: SMH, SOXX, IGV. Add Anthropic and OpenAI when and if they list; that is a config change, logged. Use local-currency returns for KRX momentum, USD for aggregation.

### 6.5 G3: AI Sentiment & Volatility

Pillar weights: Trend & relative strength 30, Implied vol & skew 30, Crowding & reaction 25, Attention 15.

| ID | Indicator | Source | Transform | Orientation | Role |
|---|---|---|---|---|---|
| ai_trend | Basket vs 50 and 200-DMA | yfinance | % distance | high = greed | coincident |
| ai_rs | SMH/SPY and basket/RSP, 63d change | yfinance | % | high = greed | coincident |
| ai_breadth | % of basket above 50-DMA | yfinance | % | high = greed | leading when diverging |
| ai_dd | Basket and memory-layer drawdown from 52w high | yfinance | % | small drawdown = greed | coincident |
| ai_vxn | VXN | FRED VXNCLS | percentile | low = greed | coincident |
| ai_nvda_iv | NVDA 30d ATM IV interpolated from chain, percentile vs own stored history | yfinance chains, daily snapshot | percentile | low = greed | coincident |
| ai_iv_spread | NVDA 30d IV minus VIX | derived | diff | low = greed | leading |
| ai_skew | NVDA and SMH 25-delta risk reversal (call IV minus put IV) | chains | diff | call skew = greed / euphoria | contrarian |
| ai_cpvol | Call/put volume across basket, 5d | chains | ratio | high = greed | contrarian |
| ai_beatfade | Share of basket earnings beats followed by a negative next-day return, trailing two seasons | Finnhub surprises + yfinance | % | low = greed; rising = exhaustion | leading |
| ai_conc | AI basket market cap as % of S&P 500 | yfinance | % | high = greed | contrarian |
| ai_lev | Shares outstanding growth in SOXL, NVDL (if retrievable) | yfinance or issuer | % | high = greed | optional |
| ai_wiki | Wikipedia pageviews: "Nvidia" (greed attention) and "AI bubble" (fear attention), 4w z | Wikimedia | z | net of the two | low weight |
| ai_gdelt | GDELT tone, AI and semiconductor queries, 7d | GDELT | z | high tone = greed | low weight |
| ai_corr | Average pairwise 63d correlation across basket | yfinance | level | display + AI fragility | leading |

**AI fragility flag** (display beside G3): SMH within 5% of its high while `ai_breadth` < 50%; `ai_corr` in top decile; `ai_beatfade` rising for two seasons; `ai_iv_spread` rising while price rises; memory-layer drawdown > 25%.

### 6.6 G4: AI Fundamental Health

Pillar weights: Spend 15, Supply chain throughput 20, Monetization 25, Compute economics & pricing 10, Financing & credit 20, Valuation 10.

| ID | Indicator | Source | Freq | Orientation |
|---|---|---|---|---|
| sp_capex_yoy | Aggregate cash capex (MSFT, GOOGL, AMZN, META, ORCL), YoY | EDGAR XBRL `PaymentsToAcquirePropertyPlantAndEquipment`; add finance-lease additions where the company's headline capex includes them. Document per company. | Q | high = healthy demand |
| sp_guide_cur, sp_guide_next | Capex guidance direction for current and next year per hyperscaler (raise / hold / cut) | events.yaml | Q | raise = healthy |
| sp_commit | Non-cancelable purchase commitments + leases not yet commenced | events.yaml from 10-Q notes | Q | display |
| sc_tsmc | TSMC monthly revenue, 3m avg YoY | TSMC IR | M | high |
| sc_odm | Alchip, Wiwynn, Quanta, Wistron monthly revenue, 3m avg YoY | TWSE MOPS | M | high |
| sc_nvda_dc | NVDA data center revenue YoY, and guide vs prior guide | EDGAR / Finnhub + events | Q | high |
| sc_sia | Global semiconductor sales, 3m avg YoY | SIA monthly release | M | high |
| sc_dram | DRAM spot trend | DRAMeXchange public spot table (check terms) or events | W | context: rising = tight |
| mo_cloud | Azure, AWS, Google Cloud growth and change vs prior quarter | events.yaml from earnings releases | Q | high and accelerating |
| mo_rpo | RPO / backlog growth (MSFT commercial RPO, Google Cloud backlog, ORCL RPO, AWS backlog) | EDGAR `RevenueRemainingPerformanceObligation` + events for segment figures | Q | high |
| mo_labs | AI lab revenue: Anthropic quarterly revenue and Q/Q dollar add; OpenAI run rate. OpenAI reports net of partner revenue share, Anthropic gross: label or normalize, never compare raw. | events.yaml | event | high |
| mo_ramp | Ramp AI Index overall and by provider | FRED RAMPAIALL, RAMPAIANTHROPIC, RAMPAIOPENAI, RAMPAIXAI | M | high |
| mo_btos | Census BTOS AI use rate | Census | 2w | high (series break 2025-11-17) |
| mo_tokens | OpenRouter total daily tokens, 28d growth; open-weight share; share by author | OpenRouter | D | growth high = healthy; open-weight share rising = pricing pressure flag |
| ce_gpu | H100 rental: SemiAnalysis 1y contract index; Uvik on-demand median | scrape or events | M | rising or stable = healthy; >15% fall in 3m = overcapacity warning |
| ce_tokprice | Frontier token price index: median blended $/1M tokens across flagship Anthropic, OpenAI, Google models | OpenRouter `/models` | D | context; feeds Unwind light 2 |
| ce_rev_proxy | Token volume growth minus token price decline | derived | D | high = healthy |
| ce_deprec | Useful-life changes, impairments | events.yaml | event | shortening lives while raising capex = negative; "impairment" = strong negative |
| fc_capex_ocf | Capex / operating cash flow, aggregate | EDGAR | Q | low = healthy |
| fc_fcf | Aggregate FCF after capex, TTM | EDGAR | Q | high |
| fc_bbb | BBB OAS 21d change (hyperscaler and AI issuance sits mostly in IG) | FRED BAMLC0A4CBBB + ALFRED | D | low = healthy |
| fc_real10 | 10y real yield level and 63d change (discount rate on AI duration) | FRED DFII10 | D | low = healthy |
| fc_proxy | ORCL and CRWV equity drawdown + 30d IV percentile, as credit proxies | yfinance | D | low = healthy |
| fc_events | AI debt issuance, CDS quotes, vendor-financing and circular deals | events.yaml | event | widening or new circular deals = negative |
| va_ai_pe | Basket forward P/E (cap-weighted and median), percentile vs own history | yfinance (unofficial), cross-checked | D | low = healthy |
| va_nvda | NVDA forward P/E vs NTM growth | yfinance + events | D | low = healthy |
| pw_eia | Commercial electricity sales growth, data-center-heavy states | EIA | M | display (optional) |

### 6.7 Unwind Signature panel (from the 2026-10-14 AI trade review)

Four lights, green / amber / red, rule shown on hover. Alert banner when 3 or more are red. One weak print is noise; the signal is these moving together.

1. **Capacity loosens but growth stays flat.** Red if `ce_gpu` falls >10% over 3 months AND (`mo_labs` Q/Q dollar add declines OR `mo_tokens` 28d growth < 0). Amber if one leg.
2. **Labs cut prices.** Red if `ce_tokprice` for flagship models falls >20% in 30 days not explained by a new model launch, or an events.yaml price-cut entry from Anthropic, OpenAI or Google. Amber at >10%.
3. **Hyperscalers trim 2027 capex.** Red if any of MSFT, GOOGL, AMZN, META frames 2027 capex below its prior framing. Amber if two or more hold flat. Green if raised. Source: events.yaml after the late-October prints.
4. **AI credit spreads widen.** Red if `fc_bbb` +25bp in 21 days, OR `fc_proxy` in its top stress decile, OR an events.yaml CDS entry showing material widening. Amber at half thresholds.

Starting state to verify on first run: as of 2026-10-14, only the rates and financing piece (light 4) was clearly flashing.

**Watch list** (events.yaml, shown as a table with last value, date, status, source):

- Anthropic Q4 2026 dollar revenue add vs Q3's $5.2B; exit run rate vs investor expectations (WSJ: ~$110B at year-end) and FT reporting it lands below $100B.
- Anthropic public S-1: GAAP margins, gross vs net revenue, customer concentration. Must be public at least 15 days before roadshow.
- Rate limits vs growth: capacity loosening with flat growth = demand problem; growth re-accelerating = supply story confirmed.
- OpenAI run rate (net basis) vs its $70B+ year-end expectation.
- Ramp AI Index provider shares; Menlo Ventures enterprise LLM spend share (semiannual).
- Open-weight token share (OpenRouter; Vercel AI Gateway if a public feed exists).
- Customer insourcing: Microsoft, Meta, Cursor routing away from third-party labs.
- HBM and DRAM contract pricing (TrendForce press releases).
- SemiAnalysis compute-supply commentary (note only, no score).

### 6.8 `events.yaml` schema

```yaml
- id: anthropic_rev_2026q3
  kpi: mo_labs
  entity: Anthropic
  metric: quarterly_revenue_usd_bn
  value: 16.8
  period: 2026Q3
  as_of: 2026-10-14
  basis: gross
  status: secondhand          # primary | guidance | secondhand | estimate
  source_url: https://www.investing.com/news/stock-market-news/anthropics-q3-revenue-surges-45-to-nearly-17-billion-4950208
  note: Reuters reading of a non-public prospectus. Replace with the S-1 figure when filed.

- id: msft_capex_2027_framing_2026q3
  kpi: sp_guide_next
  entity: Microsoft
  value: raise                # raise | hold | cut
  period: FY2027
  as_of: 2026-10-27
  status: primary
  source_url: <earnings call transcript or release URL>
```

---

## 7. Scoring engine

1. Raw value, then transform per registry.
2. Percentile rank vs trailing window (`lookback_pct`). Defaults: daily market data 5y; macro 20y; AI-specific series all available history with a 2-year minimum, otherwise scored but badged "short history". Use percentiles rather than z-scores for fat-tailed series; winsorize at 1/99.
3. Orient to 0-100 per `direction`.
4. Apply staleness decay and event status multipliers.
5. Pillar = weighted mean of live components. Gauge = weighted mean of pillars, renormalized for missing weight. Record coverage %.
6. Display needle uses a 3-day EMA for G1-G3 (store raw too). No smoothing on G4.
7. Bands. G1, G3: 0-20 Extreme Fear, 20-40 Fear, 40-60 Neutral, 60-80 Greed, 80-100 Extreme Greed. G2, G4: 0-20 Breaking down, 20-40 Weakening, 40-60 Stable, 60-80 Strong, 80-100 Very strong.
8. Attribution: top 3 contributors to each gauge's 1-week and 1-month change.
9. Every output field carries `as_of` and `coverage`.

---

## 8. Validation and backtest (gates composite membership)

- **Targets:** S&P 500 forward 21/63/126-day returns; max drawdown over next 63 days; binary "10%+ drawdown within 63 days". For AI gauges, the same on SMH and on the equal-weight basket.
- **Tests** per indicator and per composite: decile tables of forward outcomes, Spearman IC, hit rates, monotonicity, stability across regimes (2008, 2011, 2015-16, 2018 Q4, 2020, 2022, April 2025, any 2026 drawdowns).
- **Point-in-time only.** Constituent-based breadth uses current members, so it carries survivorship bias: label it and cross-check with RSP/SPY-based breadth.
- **Rules:** an indicator with the wrong sign in the full sample, or flipping sign across most regimes, is demoted to display. Do not optimize weights on the full sample. If you tune, use walk-forward and report out-of-sample.
- **Expected result:** G1 headline is contrarian at extremes. That is fine. The Fragility overlay must show a materially higher 10%+ drawdown hit rate in its top decile than the unconditional base rate, or it is labeled "Not validated".
- **AI gauges** have short histories (most series start 2023). Report with explicit low-confidence labeling and do not display AI drawdown probabilities.
- Write `reports/validation.md`: tables plus a one-paragraph verdict per gauge.

---

## 9. Dashboard spec

- **Header:** as-of time in PT and ET, next refresh time, global coverage %, list of stale or failed sources.
- **Row 1, Equity:** G1 and G2 semicircle gauges side by side. Needle plus ghost markers for previous close, 1W, 1M, 1Y. Big number, band label, delta chips. Fragility badge and any banner under G1. CNN-replica value in small text.
- **Row 2, AI:** G3 and G4 the same way. AI fragility flag under G3. Unwind Signature panel beside G4. Watch list table below.
- **Component tables** under each gauge (expandable): indicator, raw value, percentile, score, live/target weight, contribution, 1W and 1M change, sparkline, source link, as-of, badge (`verified` | `single-source` | `manual` | `stale` | `short-history`).
- **History chart:** each composite vs S&P 500 (or SMH) with drawdown shading.
- **Catalyst strip:** next 30 days of earnings (Finnhub calendar) and macro dates (`calendar.yaml`).
- **What changed:** top movers week over week.
- **Methodology tab** rendered from `indicators.yaml`.

---

## 10. Claude Code skill: `.claude/skills/market-gauges/SKILL.md`

Description: "Refresh, validate and explain the Equity and AI sentiment / state gauges. Use when asked for the gauge read, fear and greed, AI health, the unwind signature, or to log a new AI KPI after earnings."

Commands:

- **refresh:** run `make refresh`; report gauge levels and 1W changes from `latest.json` only, plus coverage, failures, stale sources.
- **explain:** read `latest.json`; write a read of 200 words or fewer: level and band per gauge, top 3 contributors, fragility and divergence flags, unwind lights, data gaps. No number that is not in `latest.json`.
- **log-event `<kpi>`:** web search for the specific KPI, propose an `events.yaml` entry (value, period, status, source_url), wait for approval, then append and rescore.
- **post-earnings `<ticker>`:** checklist for MSFT, GOOGL, META, AMZN, ORCL, NVDA prints: cloud growth, quarter capex, current-year and next-year capex framing, RPO / backlog, useful-life or impairment language, AI revenue run-rate disclosures. Output proposed `events.yaml` entries for approval.
- **validate:** `make test && make backtest`; summarize changes in `reports/validation.md`.

---

## 11. Scheduling and deployment

- **GitHub Actions `daily.yml`:** cron `0 2 * * 2-6` (UTC). That is 7pm PDT / 6pm PST Monday to Friday, after the close and Cboe CSV updates. Weekly job Saturday for CFTC, NAAIM, AAII, S&P DJI snapshot. Monthly job for FINRA, TSMC, MOPS, SIA, Shiller.
- **Secrets:** `FRED_API_KEY`, `FINNHUB_API_KEY`, `OPENROUTER_API_KEY`, `EIA_API_KEY`, `SEC_USER_AGENT`; optional `MASSIVE_API_KEY`, `FMP_API_KEY`.
- Commit `data/latest.json`, `data/history.json` and the DuckDB file (or a parquet export once size grows) back to the repo, so history accumulates when the laptop is off.
- Host `dashboard/` on Vercel or GitHub Pages.
- Expose `latest.json` at a stable URL so other workflows (e.g., the daily portfolio brief) can pull gauge readings.

---

## 12. Seed context as of 2026-10-15 (prior research; verify before use)

Use only to sanity-check first-run outputs and to pre-populate `events.yaml` with `status: secondhand`. Never hardcode into scoring.

- **Anthropic quarterly revenue:** Q1 2026 $4.7B, Q2 $11.6B, Q3 $16.8B (Reuters via Investing.com, 2026-10-14, from a prospectus not yet public). Q3 dollar add $5.2B.
- **OpenAI:** told investors its run rate was about $20B below earlier indications (Irish Times, 2026-10-09). About $50B at end of September, expects $70B+ by year-end. Net-basis reporting.
- **Cloud growth baseline:** Azure +40%, AWS +28% (Q1 2026).
- **2026 hyperscaler capex:** roughly $700-725B for MSFT, GOOGL, AMZN, META per spring guidance (AMZN ~$200B; GOOGL raised to $180-190B in April; META $115-135B; MSFT ~$190B calendar 2026). Moody's projected $785B for 2026 including ORCL and CoreWeave, approaching $1T in 2027. Alphabet disclosed a $467.6B backlog and $232.7B of non-cancelable commitments at 2026-03-31.
- **NVDA:** FQ2 FY27 revenue $96.2B; FQ3 guide $108B (2026-08-27).
- **GPU rental:** SemiAnalysis H100 1-year contract $1.70/hr (Oct 2025) to $2.35/hr (Mar 2026). Nebius raised on-demand H100 from $3.85 to $4.50 on 2026-10-01 (Uvik index).
- **Depreciation lives:** Meta 5.5 years, Amazon 5 years; Amazon took about $920M of accelerated depreciation moving from 6 to 5 years.
- **Memory:** Micron, SK Hynix and Samsung reached ~$1T valuations over the summer; October rout tied to CXMT DRAM news, rates and AI-debt concerns. A Bloomberg headline on 2026-10-15 cited a 57% SK Hynix decline (verify the basis).
- **Equity valuation (Aug 2026):** S&P forward P/E ~20.4-21.5x; top-10 weight 38.7% (record); CAPE 42.2.
- **Rates (late Aug 2026):** 10y ~4.66%; core PCE 3.3%; HY OAS ~2.6-2.7%. A hotter US inflation print landed this week.
- **Ramp (Aug 2026, prior research):** 43.8% of US businesses paid Anthropic vs 39.8% for OpenAI. Check against FRED `RAMPAIANTHROPIC` / `RAMPAIOPENAI` definitions: the FRED series measure the share of all sampled businesses and may not match this framing.
- **Menlo Ventures:** Anthropic 40% of enterprise LLM spend vs OpenAI 27%.
- **Open-weight models:** 11% to 29% of tokens on Vercel's gateway over two months.
- **Catalysts:** MSFT and GOOGL report Oct 27; FOMC decision Oct 28; META Oct 28 (unconfirmed); AMZN Oct 29.
- **Bias note:** Claude is built by Anthropic. Treat any Claude-written narrative about Anthropic or its competitors with extra skepticism; the numbers must come from sources.

---

## 13. Build phases and acceptance criteria

**Phase 0: Probe. Stop after this and report.**
`scripts/probe_sources.py` hits every source in Section 5 and prints: status, latest observation date, history start, rate-limit headers, terms flags. Output a markdown table. List failures with a proposed substitute. Write no scoring code until I approve.

**Phase 1: Ingestion and storage.** Append-only raw table (source, series, date, value, fetched_at, vintage). Pydantic schemas. Cross-checks and bounds. Tests.

**Phase 2: Registry and scoring.** Complete `indicators.yaml`; transforms unit-tested against hand-computed fixtures; composites with coverage.

**Phase 3: Backtest and validation report.** Adjust roles from results; show me a before/after of the registry.

**Phase 4: Dashboard.**

**Phase 5: Skill, GitHub Actions, deployment.**

Acceptance:
- All four gauges render with as-of and coverage; G1 and G2 coverage at least 80% on first run.
- Every displayed number has a source tooltip and as-of date.
- `cnn_replica` within ±10 points of CNN's published value on 5 spot-check days I enter by hand; explain systematic gaps (CNN uses NYSE breadth, we use S&P 500 constituents).
- pytest green; cross-checks pass or are flagged.
- `reports/validation.md` exists with verdicts.
- `make refresh` run twice in a row is idempotent.

---

## 14. Known gaps (show them, do not paper over them)

- No free single-name CDS or AI bond spreads: equity-based proxies plus manual events.
- No free historical option IV: history starts the day snapshotting starts.
- ICE BofA spreads on FRED: 3-year window; ALFRED backfill required.
- Put/call: free history gap after 2019 unless a verified source is found.
- AAII full history is paywalled.
- EPS revision breadth and ISM are not free.
- AI fundamentals are mostly quarterly or event-driven. G4 moving in steps around earnings is correct behavior, not a bug.
- yfinance is unofficial and can break. The pipeline degrades gracefully and says so.
