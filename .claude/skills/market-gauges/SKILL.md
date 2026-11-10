---
name: market-gauges
description: Refresh and explain the contrarian market gauges (S&P 500, AI & semiconductors, 10-year outlook). Use when asked for the gauge read, fear/complacency, buy or caution zones, or what an input says.
---

# Market gauges

Work from the repository root (`market-gauges/`). The pipeline is `src/signals`
(see `README.md`); the dashboard reads `data/signals.json` and
`data/signals_history.json`. The earlier four-gauge pipeline (v1) was removed;
it survives only at the git tag `v1-final`.

Ground rules:

- Never state a market value from memory. Every number comes from
  `data/signals.json`, a command's output, or a source page you just fetched.
- Report odds as what happened after similar readings, with the episode count,
  never as a forecast. Say when the evidence grade is weak. The buy side is far
  better supported than the sell side.
- Indicator definitions and the reason each is in or out of a score live in
  `src/signals/gauges.py`. Changing the inputs of a score means re-running the
  evidence (`make signals`) and checking both halves still agree.
- The bot commits `data/` after each US session: `git pull` first, and do not
  commit a locally built `data/signals.json` unless the user wants it published.

## refresh

1. `git pull --ff-only`, then `make signals` (about 3 minutes; run in the background).
2. Read `data/signals.json` and report: each gauge's score, zone and 1-week
   change; the 3-month odds for the current zone against any day; the 10-year
   outlook; any entry in `sources` with `ok: false`.

## explain

For "why is the score X": list the scored inputs with value and rank
(`gauges.<id>.indicators`, `role == "scored"`), the pillar scores, and each
input's own evidence grade.
