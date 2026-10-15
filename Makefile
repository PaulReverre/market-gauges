PY := .venv/bin/python

.PHONY: setup probe ingest score render refresh backtest restore sync serve test

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install -e '.[dev]'

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

test:
	$(PY) -m pytest
