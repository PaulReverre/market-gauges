PY := .venv/bin/python

.PHONY: setup probe ingest score refresh test

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

test:
	$(PY) -m pytest
