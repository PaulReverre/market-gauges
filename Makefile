PY := .venv/bin/python

.PHONY: setup probe ingest test

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install -e '.[dev]'

probe:
	$(PY) scripts/probe_sources.py

ingest:
	$(PY) -m gauges.ingest

test:
	$(PY) -m pytest
