PY := .venv/bin/python

.PHONY: setup probe ingest score refresh backtest test

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install -e '.[dev]'

probe:
	$(PY) scripts/probe_sources.py

ingest:
	$(PY) -m gauges.ingest

score:
	$(PY) -m gauges.score

refresh: ingest score

backtest:
	$(PY) -m gauges.backtest --apply

test:
	$(PY) -m pytest
