"""Source clients, keyed by the `source` name they write into raw_observations."""
from __future__ import annotations

from collections.abc import Callable

from gauges.sources import (aaii, alfred, cboe, census, cftc, edgar, eia, factset, finnhub, finra, fred, gdelt,
                            gpu_index, mops, naaim, openrouter, releases, shiller, ssga, treasury, tsmc, wikimedia, yf)
from gauges.sources.base import Context, FetchResult

SOURCES: dict[str, Callable[[Context], FetchResult]] = {
    "fred": fred.fetch,
    "alfred": alfred.fetch,
    "cboe": cboe.fetch,
    "ssga": ssga.fetch,
    "yfinance": yf.fetch,
    "edgar": edgar.fetch,
    "releases": releases.fetch,
    "finnhub": finnhub.fetch,
    "openrouter": openrouter.fetch,
    "cftc": cftc.fetch,
    "btos": census.fetch,
    "naaim": naaim.fetch,
    "aaii": aaii.fetch,
    "finra": finra.fetch,
    "factset": factset.fetch,
    "shiller": shiller.fetch,
    "tsmc": tsmc.fetch,
    "mops": mops.fetch,
    "uvik": gpu_index.fetch_uvik,
    "gpuperhour": gpu_index.fetch_gpuperhour,
    "gdelt": gdelt.fetch,
    "wikimedia": wikimedia.fetch,
    "eia": eia.fetch,
    "treasury": treasury.fetch,
}
