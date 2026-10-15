---
name: market-gauges
description: Refresh, validate and explain the Equity and AI sentiment / state gauges. Use when asked for the gauge read, fear and greed, AI health, the unwind signature, or to log a new AI KPI after earnings.
---

# Market gauges

Work from the repository root (`market-gauges/`). The pipeline is described in
`BUILD_PROMPT.md`; the registry is `config/indicators.yaml`; manual KPIs live in
`config/events.yaml`; the dashboard reads `data/latest.json`.

Ground rules (these override any urge to be helpful with numbers):

- Never state a market value from memory. Every number you report comes from
  `python -m gauges.digest`, `data/latest.json`, a command's output, or a
  source page you just fetched and cite.
- Never write values into the store or `data/`. New KPIs go through
  `config/events.yaml`, only after the user approves the exact entry.
- Claude is built by Anthropic. Treat any narrative about Anthropic or its
  competitors with extra skepticism, and let the sourced numbers speak.

## refresh

1. Run `make refresh` (ingest, score, render). It takes a few minutes; run it in
   the background if your harness allows.
2. Run `python -m gauges.digest`.
3. Report, using only the digest: each gauge's level, band and 1W change; global
   and per-gauge coverage (flag any LOW); sources that failed or were partial;
   stale inputs. Mention that option and quote snapshots only happen after the
   US close (16:30 ET).

## explain

1. Run `python -m gauges.digest` (do not refresh unless asked).
2. Write a read of 200 words or fewer: level and band per gauge, the top 3 1W
   drivers, the fragility overlay and breadth divergence, banners, AI fragility
   flags that are on, the Unwind lights, and data gaps (low coverage, failed or
   stale sources). Sentiment gauges are coincident and contrarian at extremes;
   say so instead of reading G1/G3 as a forecast. If the fragility overlay says
   "Not validated", do not quote a drawdown probability.
3. Use no number that is not in the digest.

## log-event `<kpi>`

1. Find `<kpi>` in `config/indicators.yaml` (an `event` indicator, or a watch
   list item) and note the expected `entity`, `metric`, `value` type and `basis`.
   Categorical values must be one of raise, hold, cut, positive, neutral,
   negative, strong_negative.
2. Web search for the specific figure. Prefer the primary source (company IR
   release, SEC filing, earnings call transcript); fall back to reputable
   secondhand reporting only when no primary source exists.
3. Propose one YAML entry with `id` (lowercase, unique), `kpi`, `entity`,
   `metric`, `value`, `period`, `as_of`, `basis` (labs: gross or net, never mix),
   `status` (primary | guidance | secondhand | estimate), `source_url`, and a
   short `note`. Show it to the user with the quote or table it came from.
4. Stop and wait for explicit approval. Do not edit any file before that.
5. After approval: append the entry to `config/events.yaml`, run
   `python -m gauges.validate events` (fix and re-propose if rejected), then
   `make score render`, and report what moved using `python -m gauges.digest`.

## post-earnings `<ticker>`

For MSFT, GOOGL, META, AMZN, ORCL or NVDA after a print, read the earnings
release, the 10-Q/10-K (EDGAR) and the call transcript, and check:

- Cloud growth and its change vs the prior quarter (Azure, Google Cloud, AWS,
  OCI) → `mo_cloud`.
- Quarter capex (EDGAR flows in automatically on the next refresh; just confirm).
- Current-year and next-year capex framing vs the prior framing → raise, hold
  or cut for `sp_guide_cur` and `sp_guide_next`.
- RPO / backlog, including segment figures not in XBRL → `mo_rpo` notes.
- Useful-life changes, accelerated depreciation or impairment language →
  `ce_deprec` (impairment = strong_negative).
- AI revenue run-rate disclosures → watch list.
- NVDA: data-center revenue year over year and guide vs prior guide →
  `sc_nvda_dc`.

Output the proposed `events.yaml` entries (same format as log-event) with
sources, and wait for approval before writing any of them.

## validate

1. Run `make test`. Stop and report failures if any.
2. Run `make backtest`. It re-scores, applies its demotion rules to
   `config/indicators.yaml`, and rewrites `reports/validation.md`.
3. Summarize: per-gauge verdicts, whether the fragility overlay validated, and
   every registry change (`git diff config/indicators.yaml`). Do not commit;
   ask the user first.
