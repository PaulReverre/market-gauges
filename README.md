# Market Gauges

A daily-refreshing dashboard of four 0–100 gauges, built to the spec in
[`BUILD_PROMPT.md`](BUILD_PROMPT.md):

| | Sentiment & volatility | Fundamental state |
|---|---|---|
| **US equity (S&P 500)** | G1 · plus the fragility overlay and a CNN-method replica | G2 |
| **AI sector** | G3 · plus the AI fragility flags | G4 · plus the four-light Unwind Signature |

Every number on the dashboard comes from a fetched source or from a
`config/events.yaml` entry with a source URL. Scores are point in time: each day
uses only data public by then (revised macro series via ALFRED vintages).

## Quick start

```bash
make setup                 # Python 3.11 venv + dependencies
cp .env.example .env       # then fill in the keys below
make restore               # rebuild the local store from data/archive (if present)
make refresh               # ingest -> score -> render (a few minutes; full backfill on an empty store)
make serve                 # dashboard at http://127.0.0.1:8765/dashboard/
```

Keys (`.env`, free tiers): `FRED_API_KEY`, `FINNHUB_API_KEY`, `OPENROUTER_API_KEY`,
`EIA_API_KEY`, and `SEC_USER_AGENT` (a name and contact email, per SEC fair-access
policy). `MASSIVE_API_KEY` and `FMP_API_KEY` are optional. A missing key makes
that source report `missing_key`; everything else still runs.

## Make targets

| Target | What it does |
|---|---|
| `probe` | Phase 0 source probe → `reports/source_status.md` |
| `ingest` | Pull every source into the append-only store (`data/gauges.duckdb`) and archive the run |
| `score` | Percentiles, staleness, pillars, gauges, coverage, overlays (derived tables) |
| `render` | Write `data/latest.json` and `data/history.json` |
| `refresh` | `ingest` + `score` + `render` |
| `backtest` | Phase 3 validation; applies demotions to `config/indicators.yaml`, writes `reports/validation.md` |
| `restore` | Rebuild the store from `data/archive` |
| `serve` | Static server for the dashboard |
| `test` | pytest (no network) |

## How it fits together

```
config/series.yaml      raw series: what each source pulls, frequency, release lag, sanity bounds, cadence
config/indicators.yaml  indicator registry: inputs, transforms, direction, weight, role, fragility, CNN rows
config/events.yaml      manual KPIs (source_url + status required)    config/baskets.yaml  AI basket
config/calendar.yaml    FOMC and BEA dates for the catalyst strip

src/gauges/sources/*    one client per source  ->  ingest.py  ->  store.py (DuckDB, append-only)
                                                                 archive.py (parquet per run, committed)
features.py (as-known daily values) -> score.py (gauges, overlays) -> render.py (JSON) -> dashboard/index.html
backtest.py (Section 8 validation, gates composite membership)
```

The DuckDB file is git-ignored. Each ingest run writes the rows it added to
`data/archive/<table>/<run_id>.parquet` (never rewritten), so history survives
in git and a fresh machine or CI runner rebuilds the store with `make restore`.

## Scheduled refresh (GitHub Actions)

`.github/workflows/refresh.yml` runs on the spec's schedule: daily at 02:00 UTC
Tue–Sat (after Mon–Fri closes), weekly on Saturday (CFTC, NAAIM, AAII, FactSet,
Census BTOS), and monthly on the 12th (FINRA, TSMC, MOPS, Shiller, Uvik, EIA).
It restores the store, ingests that group, scores, renders, and commits
`data/` back. Run it by hand from the Actions tab (`workflow_dispatch`).

Add the keys as repository secrets (Settings → Secrets and variables → Actions),
or from a terminal with the GitHub CLI, one per key:

```bash
gh secret set FRED_API_KEY --repo PaulReverre/market-gauges
```

(`gh secret set` prompts for the value, so it never lands in your shell history.)

## Hosting

Pick one:

- **GitHub Pages**: Settings → Pages → Source "GitHub Actions", then add the
  repository variable `PAGES_ENABLED=true`. The `pages` job publishes the
  dashboard, `data/latest.json` and `data/history.json` after each refresh.
  Pages on a private repository needs a paid GitHub plan.
- **Vercel**: import the repository in Vercel. `vercel.json` serves it as a
  static site (`/` → `/dashboard/`); every data commit redeploys.

Either way `data/latest.json` sits at a stable URL (e.g.
`https://<project>.vercel.app/data/latest.json`) for other workflows to read.

## Claude Code skill

`.claude/skills/market-gauges/SKILL.md` adds `refresh`, `explain` (≤200 words,
numbers only from `python -m gauges.digest`), `log-event <kpi>` and
`post-earnings <ticker>` (propose sourced `events.yaml` entries, write only
after approval), and `validate`.

## Acceptance checks (Section 13)

- All four gauges render with as-of and coverage; G1 and G2 above 80% coverage.
- Every dashboard number has a source tooltip/link and an as-of date.
- `cnn_replica` within ±10 of CNN's published value on 5 days you enter by
  hand: compare `data/latest.json` → `cnn_replica` with CNN's page. Expect
  systematic gaps: CNN uses NYSE breadth; this uses S&P 500 members.
- `make test` green; cross-checks pass or are flagged.
- `reports/validation.md` exists with verdicts.
- `make ingest` twice in a row: the second run inserts nothing.

## Known gaps

Shown on the dashboard rather than papered over: no free single-name CDS or AI
bond spreads (equity proxies + manual events); no free historical option IV
(snapshots accumulate from the first post-close run); ICE BofA spreads only
from 2023-10-16 (ALFRED holds no older vintages); S&P DJI EPS file blocked
(FactSet weekly PDF instead, from 2025); AAII full history is paywalled; NAAIM's
weekly file stopped updating on 2026-08-19; SemiAnalysis H100 index has no
feed; GDELT rate-limits heavily; AI fundamentals move in steps around earnings.
