# Source status (Phase 0 probe)

Run 2026-10-15 11:25 PT (14:25 ET) · Python 3.11.14 · `scripts/probe_sources.py`

Keys: FRED_API_KEY ✓ · FINNHUB_API_KEY ✓ · OPENROUTER_API_KEY ✓ · EIA_API_KEY ✓ · SEC_USER_AGENT ✓ · MASSIVE_API_KEY ✗ · FMP_API_KEY ✗ · ALPHAVANTAGE_API_KEY ✗

Legend: ✅ works · ⚠️ works with a problem · 🔑 needs a free key · ❌ blocked or no feed

| # | Source | Status | Latest obs | History start | Auth / rate limits | Terms flags | Notes |
|---|---|---|---|---|---|---|---|
| 1 | FRED (series) | ✅ | newest 2026-10-15; SP500 2026-10-14 | SP500 2016-10-17; ICE BofA 2023-10-16 | API key, 120 req/min; graph CSV keyless | ICE BofA series: 3y window (licensing) | 41/41 series via API; Richmond/KC composites: not on FRED, scrape Fed sites |
| 2 | ALFRED (pre-cutoff ICE BofA) | ❌ | earliest vintage 2023-10-16 | 2023-10-16 | FRED API key | ICE licensing | confirmed: ALFRED holds no pre-2023-10-16 history for ICE BofA, backfill impossible; earliest vintage 2023-10-16; vintage 2025-06-02 first obs 2023-10-16 |
| 3 | Cboe index CSVs | ✅ | 2026-10-14 | VIX 1990, VIX3M 2009, VIX9D 2011, VVIX 2006, SKEW 1990, COR1M 2006, COR3M 2006, COR30D 2006, VXN 2009 | none | Cboe site terms | 9/9 symbols; served from cdn-api.cboe.com (follow redirects) |
| 4 | Cboe daily market stats (put/call) | ✅ | 2026-10-14: total 0.88, equity 0.61, index 1.01 | archive CSV 2006-11-01→2019-10-04; daily JSON 2019-10-07→ | none | Cboe site terms | post-2019 gap closed by the daily JSON |
| 5 | yfinance / Yahoo | ✅ | 2026-10-15 | ^GSPC 1927-12-30; ^MOVE 2002-11-12; option IV: none, snapshot from day 1 | unofficial; plain HTTP gets 429, yfinance (curl_cffi) passes | Yahoo ToS gray area for automation | 20/20 symbols; NVDA options: 25 expiries, IV field present; NVDA forwardPE 13.640933 |
| 6 | Nasdaq API (Yahoo fallback) | ✅ | SPY 2026-10-14 close 769.60 | SPY 2016-10-14 (2513 rows) | unofficial, undocumented | unofficial; ToS gray area | NVDA option chain: 503 rows, no IV field (compute from mids). No ^MOVE, futures or KRX coverage |
| 7 | SEC EDGAR | ✅ | newest period end 2026-08-31 | full XBRL history | declared User-Agent with email; <10 req/s | public; fair-access policy | flags: AMZN rpo: stale (2020-06-30); META rpo: no tag. 10-Q values are YTD: de-cumulate |
| 8 | Finnhub | ✅ | NVDA quote 2026-10-15 14:24 ET | EPS surprises from 2025-12-31 | free key; ~60 req/min; x-ratelimit-limit: 60; x-ratelimit-remaining: 59; x-ratelimit-reset: 1792088780 | personal, non-commercial | earnings calendar next 30d: 1500 rows; surprises: 4 quarters |
| 9 | OpenRouter | ✅ | 2026-10-14 (meta.as_of 2026-10-15T18:25:21.798Z) | 2025-01-01 (API floor; 102 rows on 2025-01-01..02) | /models keyless; Data API key, 30 req/min, 500/day | CC BY 4.0; cite 'Source: OpenRouter (openrouter.ai/rankings), as of {as_of}' | last 7d: 357 rows over 7 days; /models keyless: 463 models, newest added 2026-10-15 |
| 10 | CFTC TFF (Socrata gpe5-46if) | ✅ | report 2026-10-06 (Tuesday positions, released Friday) | ES (E-mini S&P 500) 2006-06-13, NQ (E-mini Nasdaq-100) 2006-06-13 | none (app token optional) | public | keyed on contract code (names changed in 2022) |
| 11 | Census BTOS | ✅ | period 202620: AI use 'Yes' 23.8% | period 202319 (80 periods) | none | public (Census) | question: 'In the last two weeks, did this business use Artificial Intelligence (…'; series break 2025-11-17 (question broadened) |
| 12 | NAAIM Exposure Index | ⚠️ | 2026-08-19 (57 days old) | 2006-07-05 | none | public page | newest xlsx uploaded 2026-08-20: USE_Data-since-Inception_2026-08-19.xlsx; feed STALE, current value is JS-rendered on the page |
| 13 | AAII Sentiment Survey | ✅ | Oct 14: bull 38.7% / neutral 19.7% / bear 41.6% | 22 weeks visible (oldest May 20) | none | recent weeks public; Incapsula bot protection | accumulate weekly; full history is members-only |
| 14 | FINRA margin statistics | ✅ | 2026-08: debit balances $1,453.8B | 1997-01 | none | public | xlsx at /sites/default/files/2021-03/margin-statistics.xlsx |
| 15 | S&P DJI sp-500-eps-est.xlsx | ❌ | — | — | none (bot-protected) | S&P DJI terms; Akamai bot protection | HTTP 403 (AkamaiGHost): bot protection |
| 16 | FactSet Earnings Insight (S&P DJI fallback) | ✅ | 2026-10-09 | weekly PDFs (archive by URL date) | none | FactSet terms; check reuse | 1236 KB PDF: forward P/E, NTM EPS, % beating; needs PDF parsing |
| 17 | Shiller CAPE | ✅ | file updated Tue, 06 Oct 2026 14:33:57 GMT | 1871-01 | none | public; cite Shiller | link has a GUID: scrape shillerdata.com each run; old yale.edu copy frozen (Tue, 17 Oct 2023 03:27:29 GMT); .xls needs xlrd |
| 18 | TSMC IR monthly revenue | ✅ | 2026-09: NT$511,857M (54.6% YoY) | yearly pages (URL per year) | none (bot-protected) | bot-protected | 9 months on the 2026 page; cross-check vs MOPS 2330 |
| 19 | TWSE MOPS monthly revenue | ✅ | 2026-09 | monthly files back to 2013 (ROC-year URLs) | none | public (TWSE) | 5/5 tickers (Alchip in the _1 KY table); cp950 encoded |
| 20 | Uvik GPU Price Index | ✅ | captured 2026-10-07; H100 median $3.99/GPU-hr | files found: 2026-10 | none | CC BY 4.0; credit Uvik Software | 87 rows, 21 H100; no back history, accumulate monthly |
| 21 | SemiAnalysis H100 1y rental index | ❌ | — | — | — | robots crawl-delay 100; check terms | no public machine-readable H100 1y rental index found (checked 2026-10-15) |
| 22 | GPUPerHour index (GPU fallback) | ✅ | 2026-10-15; NVIDIA H100 NVL median $3.11 | 2026-09-14 | none | terms unverified | 31 daily snapshots; on-demand marketplace prices, not 1y contracts |
| 23 | GDELT DOC 2.0 | ⚠️ | 2026-10-15 16:00 UTC | rolling window (timelines up to ~3 months) | none; ~1 req/5s | open; cite GDELT | 8 buckets in 7d; largest gap 5 days, 4:00:00 |
| 24 | Wikimedia Pageviews | ✅ | Nvidia 20261014 5,297; AI_bubble 20261014 1,644 | 2015-07-01 | descriptive User-Agent | CC0 data; UA policy |  |
| 25 | EIA API v2 | ✅ | 2026-07 US commercial sales 150578.72802 | 2001-01 (retail-sales) | free key | public domain |  |
| 26 | Massive (ex-Polygon) Basic | 🔑 | — | — | free key; 5 req/min, EOD, 2y | free-tier ToS | unauthenticated request → HTTP 401. Set MASSIVE_API_KEY in .env. Optional (cross-check only). |
| 27 | FMP Basic | 🔑 | — | — | free key; 250/day, fixed symbol list | free-tier ToS | unauthenticated request → HTTP 401. Set FMP_API_KEY in .env. Optional (cross-check only). |
| 28 | Alpha Vantage | ✅ | IBM 2026-10-14 | — | free key; 25/day | free-tier ToS | demo key (IBM only); skip per spec unless a gap needs it |
| 29 | US Treasury yield curve (cross-check) | ✅ | 2026-10-14: 10y 5.19 | 1990 (yearly CSVs) | none | public domain | used for the DGS10 cross-check |

**Tally:** 22 ✅ · 2 ⚠️ · 2 🔑 · 3 ❌

## Failures, warnings and substitutes

- ❌ **ALFRED (pre-cutoff ICE BofA)**: confirmed: ALFRED holds no pre-2023-10-16 history for ICE BofA, backfill impossible; earliest vintage 2023-10-16; vintage 2025-06-02 first obs 2023-10-16. Substitute: Moody's BAA10Y/AAA10Y on FRED (1986+) for IG; HYG/LQD vs IEF ETF proxy for HY (2007+); any pre-April-2026 CSVs on hand.
- ⚠️ **NAAIM Exposure Index**: newest xlsx uploaded 2026-08-20: USE_Data-since-Inception_2026-08-19.xlsx; feed STALE, current value is JS-rendered on the page. Substitute: reverse-engineer the page's admin-ajax call, or a weekly manual events.yaml entry.
- ❌ **S&P DJI sp-500-eps-est.xlsx**: HTTP 403 (AkamaiGHost): bot protection. Substitute: FactSet Earnings Insight weekly PDF, or a manual weekly xlsx drop into data/inbox/.
- ❌ **SemiAnalysis H100 1y rental index**: no public machine-readable H100 1y rental index found (checked 2026-10-15). Substitute: monthly manual events.yaml entry; GPUPerHour daily CSV or Silicon Data SDH100RT as alternates.
- ⚠️ **GDELT DOC 2.0**: 8 buckets in 7d; largest gap 5 days, 4:00:00. Substitute: keep at low weight; Wikimedia attention as primary.
- 🔑 **Massive (ex-Polygon) Basic**: set `MASSIVE_API_KEY` in `.env`.
- 🔑 **FMP Basic**: set `FMP_API_KEY` in `.env`.

## Cross-checks (Section 3.5)

| Check | Primary | Secondary | Date | Primary value | Secondary value | Diff | Tolerance | Result |
|---|---|---|---|---|---|---|---|---|
| SPX close | yf:^GSPC | fred:SP500 | 2026-10-14 | 7718.94 | 7718.94 | -0.000% | ±0.15% | ✅ pass |
| VIX close | cboe:VIX | fred:VIXCLS | 2026-10-14 | 16.87 | 16.87 | +0.00 | exact | ✅ pass |
| 10y yield | fred:DGS10 | treasury:10y | 2026-10-13 | 5.21 | 5.21 | +0.0bp | ±2bp | ✅ pass |

## Details

### FRED series

| Series | First | Last | Last value | Info | Flag |
|---|---|---|---|---|---|
| SP500 | 2016-10-17 | 2026-10-14 | 7718.94 | D, updated 2026-10-14 |  |
| VIXCLS | 1990-01-02 | 2026-10-14 | 16.87 | D, updated 2026-10-15 |  |
| VXNCLS | 2001-02-02 | 2026-10-14 | 22.7 | D, updated 2026-10-15 |  |
| DGS10 | 1962-01-02 | 2026-10-13 | 5.21 | D, updated 2026-10-14 |  |
| BAMLH0A0HYM2EY | 2023-10-16 | 2026-10-14 | 8.2 | D, updated 2026-10-15 | ICE 3y window |
| BAMLC0A0CMEY | 2023-10-16 | 2026-10-14 | 5.94 | D, updated 2026-10-15 | ICE 3y window |
| BAMLH0A0HYM2 | 2023-10-16 | 2026-10-14 | 3.25 | D, updated 2026-10-15 | ICE 3y window |
| BAMLH0A3HYC | 2023-10-16 | 2026-10-14 | 12.82 | D, updated 2026-10-15 | ICE 3y window |
| BAMLH0A1HYBB | 2023-10-16 | 2026-10-14 | 1.99 | D, updated 2026-10-15 | ICE 3y window |
| BAMLC0A4CBBB | 2023-10-16 | 2026-10-14 | 1.04 | D, updated 2026-10-15 | ICE 3y window |
| NFCI | 1971-01-08 | 2026-10-09 | -0.499 | W, updated 2026-10-15 |  |
| STLFSI4 | 1993-12-31 | 2026-10-09 | -0.7206 | W, updated 2026-10-14 |  |
| SOFR | 2018-04-03 | 2026-10-14 | 3.88 | D, updated 2026-10-15 |  |
| IORB | 2021-07-29 | 2026-10-15 | 3.9 | D, updated 2026-10-14 |  |
| DTWEXBGS | 2006-01-02 | 2026-10-09 | 121.602 | D, updated 2026-10-13 |  |
| GDPNOW | 2011-07-01 | 2026-07-01 | 3.6429 | Q, updated 2026-10-15 |  |
| CFNAIMA3 | 1967-05-01 | 2026-08-01 | 0.01 | M, updated 2026-09-21 |  |
| ICSA | 1967-01-07 | 2026-10-10 | 199000 | W, updated 2026-10-15 |  |
| SAHMREALTIME | 1959-12-01 | 2026-09-01 | 0 | M, updated 2026-10-02 |  |
| PERMIT | 1960-01-01 | 2026-08-01 | 1403 | M, updated 2026-09-24 |  |
| DFII10 | 2003-01-02 | 2026-10-13 | 2.88 | D, updated 2026-10-14 |  |
| NCBEILQ027S | 1945-10-01 | 2026-04-01 | 8.30508e+07 | Q, updated 2026-09-11 |  |
| GDP | 1947-01-01 | 2026-04-01 | 32563 | Q, updated 2026-09-30 |  |
| DFF | 1954-07-01 | 2026-10-13 | 3.88 | D, updated 2026-10-14 |  |
| PCEPILFE | 1959-01-01 | 2026-08-01 | 130.455 | M, updated 2026-09-30 |  |
| T10Y3M | 1982-01-04 | 2026-10-14 | 0.91 | D, updated 2026-10-14 |  |
| WALCL | 2002-12-18 | 2026-10-07 | 6.74756e+06 | W, updated 2026-10-08 |  |
| WTREGEN | 2002-12-18 | 2026-10-07 | 880253 | W, updated 2026-10-08 |  |
| RRPONTSYD | 2003-02-07 | 2026-10-15 | 1.201 | D, updated 2026-10-15 |  |
| DRTSCILM | 1990-04-01 | 2026-07-01 | 0 | Q, updated 2026-08-03 |  |
| RRSFS | 1992-01-01 | 2026-09-01 | 221536 | M, updated 2026-10-15 |  |
| DRCCLACBS | 1991-01-01 | 2026-04-01 | 2.85 | Q, updated 2026-08-25 |  |
| UMCSENT | 1952-11-01 | 2026-08-01 | 51.7 | M, updated 2026-09-25 |  |
| T10YIE | 2003-01-02 | 2026-10-14 | 2.35 | D, updated 2026-10-14 |  |
| RAMPAIALL | 2023-01-01 | 2026-09-01 | 55.8599 | M, updated 2026-10-14 |  |
| RAMPAIANTHROPIC | 2023-01-01 | 2026-09-01 | 43.7743 | M, updated 2026-10-14 |  |
| RAMPAIOPENAI | 2023-01-01 | 2026-09-01 | 40.6324 | M, updated 2026-10-14 |  |
| RAMPAIXAI | 2023-01-01 | 2026-09-01 | 4.1367 | M, updated 2026-10-14 |  |
| GACDFSA066MSFRBPHI | 1968-05-01 | 2026-10-01 | 20.9 | M, updated 2026-10-15 |  |
| GACDISA066MSFRBNY | 2001-07-01 | 2026-10-01 | 25.5 | M, updated 2026-10-15 |  |
| BACTSAMFRBDAL | 2004-06-01 | 2026-09-01 | 9.8 | M, updated 2026-09-28 |  |

### Cboe index CSVs

| Symbol | First | Last | Last close | Rows |
|---|---|---|---|---|
| VIX | 1990-01-02 | 2026-10-14 | 16.87 | 9294 |
| VIX3M | 2009-09-18 | 2026-10-14 | 18.79 | 4294 |
| VIX9D | 2011-01-04 | 2026-10-14 | 14.59 | 3968 |
| VVIX | 2006-03-06 | 2026-10-14 | 94.45 | 5125 |
| SKEW | 1990-01-02 | 2026-10-14 | 157.3 | 9248 |
| COR1M | 2006-01-03 | 2026-10-14 | 9.02 | 5228 |
| COR3M | 2006-01-03 | 2026-10-14 | 12.8 | 5213 |
| COR30D | 2006-01-03 | 2026-10-14 | 9.6 | 5228 |
| VXN | 2009-09-14 | 2026-10-14 | 22.7 | 4300 |

### SEC XBRL tags (freshest candidate per metric)

| Ticker | capex | rpo | ocf | d&a |
|---|---|---|---|---|
| MSFT | PaymentsToAcquirePropertyPlantAndEquipment → 2026-06-30 (10-K) | RevenueRemainingPerformanceObligation → 2026-06-30 (10-K) | NetCashProvidedByUsedInOperatingActivities → 2026-06-30 (10-K) | Depreciation → 2026-06-30 (10-K) |
| GOOGL | PaymentsToAcquirePropertyPlantAndEquipment → 2026-06-30 (10-Q) | RevenueRemainingPerformanceObligation → 2026-06-30 (10-Q) | NetCashProvidedByUsedInOperatingActivities → 2026-06-30 (10-Q) | Depreciation → 2026-06-30 (10-Q) |
| AMZN | PaymentsToAcquireProductiveAssets → 2026-06-30 (10-Q) | ⚠️ RevenueRemainingPerformanceObligation → 2020-06-30 (10-Q) | NetCashProvidedByUsedInOperatingActivities → 2026-06-30 (10-Q) | DepreciationDepletionAndAmortization → 2026-06-30 (10-Q) |
| META | PaymentsToAcquirePropertyPlantAndEquipment → 2026-06-30 (10-Q) | none | NetCashProvidedByUsedInOperatingActivities → 2026-06-30 (10-Q) | DepreciationDepletionAndAmortization → 2026-06-30 (10-Q) |
| ORCL | PaymentsToAcquirePropertyPlantAndEquipment → 2026-08-31 (10-Q) | RevenueRemainingPerformanceObligation → 2026-08-31 (10-Q) | NetCashProvidedByUsedInOperatingActivities → 2026-08-31 (10-Q) | Depreciation → 2026-08-31 (10-Q) |
| NVDA | PaymentsToAcquireProductiveAssets → 2026-07-26 (10-Q) | RevenueRemainingPerformanceObligation → 2026-07-26 (10-Q) | NetCashProvidedByUsedInOperatingActivities → 2026-07-26 (10-Q) | DepreciationDepletionAndAmortization → 2026-07-26 (10-Q) |

### TWSE MOPS monthly revenue

| Ticker | Name | Revenue (k TWD) |
|---|---|---|
| 2330 | TSMC | 511,857,399 |
| 2382 | Quanta | 325,299,509 |
| 3231 | Wistron | 380,806,168 |
| 6669 | Wiwynn | 172,620,352 |
| 3661 | Alchip | 9,978,701 |
