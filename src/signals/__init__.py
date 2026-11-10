"""Contrarian market signals: a lean, mostly stateless daily pipeline.

Every run re-downloads full price and volatility history, recomputes each
indicator, score and odds table from scratch, and writes data/signals.json and
data/signals_history.json for the static dashboard. The only state kept between
runs lives in data/state/ (series that cannot be re-downloaded in full, plus a
last-good cache of slow sources).
"""
