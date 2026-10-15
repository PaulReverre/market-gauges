PY := .venv/bin/python

.PHONY: setup probe

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install -r requirements.txt

probe:
	$(PY) scripts/probe_sources.py
