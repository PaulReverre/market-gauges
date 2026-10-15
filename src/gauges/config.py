"""Pydantic schemas and loaders for config/*.yaml and the manual events file."""
from __future__ import annotations

from datetime import date
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

Freq = Literal["1d", "1w", "2w", "1mo", "3mo"]
Status = Literal["primary", "guidance", "secondhand", "estimate"]
STATUS_WEIGHT: dict[str, float] = {"primary": 1.0, "guidance": 0.8, "secondhand": 0.6, "estimate": 0.3}
# raise/hold/cut: capex guidance; positive..strong_negative: impairments, financing events.
CATEGORICAL_VALUES = {"raise", "hold", "cut", "positive", "neutral", "negative", "strong_negative"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# config/series.yaml
# --------------------------------------------------------------------------- #
class SeriesSpec(Strict):
    match: str
    freq: Freq
    bounds: tuple[float, float] | None = None
    lag_days: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _bounds_ordered(self) -> SeriesSpec:
        if self.bounds and self.bounds[0] >= self.bounds[1]:
            raise ValueError(f"{self.match}: lower bound must be below upper bound")
        return self


class SeriesList(Strict):
    series: list[str]


class CboeFetch(Strict):
    indices: list[str]
    putcall_json_start: date


class YFinanceFetch(Strict):
    symbols: list[str]
    history_start: date
    constituent_years: int = Field(gt=0)
    info_symbols: list[str]
    option_chains: list[str]
    option_chain_max_days: int = Field(gt=0)
    option_volume_max_days: int = Field(gt=0)


class EdgarFetch(Strict):
    tickers: list[str]
    tags: list[str]


class FinnhubFetch(Strict):
    calendar_days_ahead: int = Field(gt=0)


class OpenRouterFetch(Strict):
    start: date


class CftcFetch(Strict):
    contracts: dict[str, str]
    fields: list[str]


class FactsetFetch(Strict):
    backfill_since: date


class TsmcFetch(Strict):
    start_year: int


class MopsFetch(Strict):
    start: date
    tickers: dict[str, str]


class GdeltFetch(Strict):
    queries: dict[str, str]


class WikimediaFetch(Strict):
    articles: list[str]
    start: date


class EiaFetch(Strict):
    states: list[str]


class TreasuryFetch(Strict):
    start_year: int


class FetchConfig(Strict):
    fred: SeriesList
    alfred: SeriesList
    cboe: CboeFetch
    yfinance: YFinanceFetch
    edgar: EdgarFetch
    finnhub: FinnhubFetch
    openrouter: OpenRouterFetch
    cftc: CftcFetch
    factset: FactsetFetch
    tsmc: TsmcFetch
    mops: MopsFetch
    gdelt: GdeltFetch
    wikimedia: WikimediaFetch
    eia: EiaFetch
    treasury: TreasuryFetch


class SeriesConfig(Strict):
    fetch: FetchConfig
    cadence: dict[Literal["weekly", "monthly"], list[str]] = Field(default_factory=dict)
    specs: list[SeriesSpec]

    def spec_for(self, source: str, series: str) -> SeriesSpec | None:
        """First spec whose pattern matches "source:series"."""
        return _first_match(self.specs, f"{source}:{series}")


@lru_cache(maxsize=65536)
def _match_cached(patterns: tuple[str, ...], key: str) -> int | None:
    return next((i for i, p in enumerate(patterns) if fnmatchcase(key, p)), None)


def _first_match(specs: list[SeriesSpec], key: str) -> SeriesSpec | None:
    i = _match_cached(tuple(s.match for s in specs), key)
    return None if i is None else specs[i]


# --------------------------------------------------------------------------- #
# config/baskets.yaml
# --------------------------------------------------------------------------- #
class Layer(Strict):
    weight: float = Field(gt=0)
    tickers: list[str] = Field(min_length=1)


class Baskets(Strict):
    layers: dict[str, Layer]
    reference_etfs: list[str]

    @model_validator(mode="after")
    def _check(self) -> Baskets:
        total = sum(layer.weight for layer in self.layers.values())
        if abs(total - 100) > 1e-9:
            raise ValueError(f"layer weights sum to {total}, expected 100")
        tickers = [t for layer in self.layers.values() for t in layer.tickers]
        dupes = {t for t in tickers if tickers.count(t) > 1}
        if dupes:
            raise ValueError(f"tickers in more than one layer: {sorted(dupes)}")
        return self

    @property
    def tickers(self) -> list[str]:
        return [t for layer in self.layers.values() for t in layer.tickers]


# --------------------------------------------------------------------------- #
# config/events.yaml (manual inputs, Section 3.7)
# --------------------------------------------------------------------------- #
class Event(Strict):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    kpi: str
    entity: str
    metric: str | None = None
    value: float | str
    period: str
    as_of: date
    basis: Literal["gross", "net"] | None = None
    status: Status
    source_url: HttpUrl
    note: str | None = None

    @field_validator("value")
    @classmethod
    def _value_vocabulary(cls, v: float | str) -> float | str:
        if isinstance(v, str) and v not in CATEGORICAL_VALUES:
            raise ValueError(f"categorical value must be one of {sorted(CATEGORICAL_VALUES)}")
        return v

    @field_validator("as_of")
    @classmethod
    def _not_future(cls, v: date) -> date:
        if v > date.today():
            raise ValueError("as_of is in the future")
        return v

    @property
    def weight(self) -> float:
        return STATUS_WEIGHT[self.status]


class CalendarEvent(Strict):
    date: date
    event: str
    kind: Literal["fomc", "macro", "other"]
    source_url: HttpUrl


def load_calendar(path: Path = CONFIG_DIR / "calendar.yaml") -> list[CalendarEvent]:
    return [CalendarEvent.model_validate(e) for e in (_read_yaml(path) or [])]


def _read_yaml(path: Path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_series_config(path: Path = CONFIG_DIR / "series.yaml") -> SeriesConfig:
    return SeriesConfig.model_validate(_read_yaml(path))


def load_baskets(path: Path = CONFIG_DIR / "baskets.yaml") -> Baskets:
    return Baskets.model_validate(_read_yaml(path))


def load_events(path: Path = CONFIG_DIR / "events.yaml") -> tuple[list[Event], list[str]]:
    """Validate each entry independently: valid events load, invalid ones are reported."""
    raw = _read_yaml(path) or []
    if not isinstance(raw, list):
        return [], [f"{path.name}: top level must be a list"]
    events, errors, seen = [], [], set()
    for i, entry in enumerate(raw):
        label = entry.get("id", f"#{i}") if isinstance(entry, dict) else f"#{i}"
        try:
            event = Event.model_validate(entry)
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            errors.append(f"{label}: {problems}")
            continue
        if event.id in seen:
            errors.append(f"{label}: duplicate id")
            continue
        seen.add(event.id)
        events.append(event)
    return events, errors
