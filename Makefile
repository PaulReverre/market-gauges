PY := .venv/bin/python

.PHONY: setup signals serve test probe ingest score render refresh backtest robustness restore sync legacy-test

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install -e '.[dev]'

# The dashboard's data: full rebuild of data/signals.json (about 3 minutes).
signals:
	$(PY) -m signals.build

# ---- retired v1 pipeline (see LEGACY.md); not scheduled any more ----
probe:
	$(PY) scripts/probe_sources.py

ingest:
	$(PY) -m gauges.ingest

score:
	$(PY) -m gauges.score

render:
	$(PY) -m gauges.render

refresh: ingest score render

serve:
	@echo "Dashboard at http://127.0.0.1:8765/dashboard/"
	$(PY) -m http.server 8765 --bind 127.0.0.1

restore:
	$(PY) -m gauges.archive restore

# Pull the bot's data commits, then rebuild the local store from the archive.
sync:
	git pull --ff-only
	$(PY) -m gauges.archive restore --force

backtest:
	$(PY) -m gauges.backtest --apply

# Event, episode, strategy and Monte Carlo study of the scored history (run `make score` first).
robustness:
	$(PY) -m gauges.robustness

test:
	$(PY) -m pytest tests/test_signals.py

legacy-test:
	$(PY) -m pytest --ignore=tests/test_signals.py
