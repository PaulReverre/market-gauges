PY := .venv/bin/python

.PHONY: setup signals serve test

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install -e '.[dev]'

# Full rebuild of data/signals.json and data/signals_history.json (about 3 minutes).
signals:
	$(PY) -m signals.build

serve:
	@echo "Dashboard at http://127.0.0.1:8765/dashboard/"
	$(PY) -m http.server 8765 --bind 127.0.0.1

test:
	$(PY) -m pytest
