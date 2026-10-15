"""Pydantic schema for config/indicators.yaml."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gauges.config import CONFIG_DIR, Freq

FREQ_BDAYS = {"1d": 1, "1w": 5, "2w": 10, "1mo": 21, "3mo": 63}
Direction = Literal["higher_is_greed", "lower_is_greed", "higher_is_healthy", "lower_is_healthy"]
Role = Literal["leading", "coincident", "contrarian", "context", "display"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Gauge(Strict):
    name: str
    scale: list[str] = Field(min_length=5, max_length=5)
    smoothing: Literal["ema3", "none"]
    pillars: dict[str, float]


class InputRef(Strict):
    ref: str = Field(pattern=r"^[a-z_]+:.+$")
    op: str = "identity"
    args: dict = Field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.ref.split(":", 1)[0]

    @property
    def series(self) -> str:
        return self.ref.split(":", 1)[1]


class EventSpec(Strict):
    kpi: str
    categories: dict[str, float] | None = None
    score_map: list[tuple[float, float]] | None = None
    growth: bool = False   # score period-over-period growth of a numeric metric


class Indicator(Strict):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    gauge: Literal["G1", "G2", "G3", "G4"]
    pillar: str
    name: str
    kind: Literal["series", "panel", "event", "none"] = "series"
    inputs: list[InputRef] = Field(default_factory=list)
    transform: str = "identity"
    targs: dict = Field(default_factory=dict)
    panel: str | None = None
    event: EventSpec | None = None
    lookback_pct: str | None = None          # "5y", "20y", "all"; default by frequency
    direction: Direction
    weight: float = Field(ge=0)
    expected_freq: Freq
    bounds: tuple[float, float] | None = None
    role: Role
    fragility: Literal["higher_is_fragile", "lower_is_fragile"] | None = None
    cnn: bool = False
    notes: str | None = None

    @field_validator("inputs", mode="before")
    @classmethod
    def _plain_refs(cls, v):
        return [{"ref": x} if isinstance(x, str) else x for x in (v or [])]

    @field_validator("lookback_pct")
    @classmethod
    def _lookback(cls, v):
        if v is not None and v != "all" and not re.fullmatch(r"\d+y", v):
            raise ValueError("lookback_pct must look like '5y' or be 'all'")
        return v

    @model_validator(mode="after")
    def _kind_fields(self) -> Indicator:
        if self.kind == "series" and not self.inputs:
            raise ValueError(f"{self.id}: series indicators need inputs")
        if self.kind == "panel" and not self.panel:
            raise ValueError(f"{self.id}: panel indicators need `panel`")
        if self.kind == "event" and not self.event:
            raise ValueError(f"{self.id}: event indicators need `event`")
        return self

    @property
    def sign(self) -> int:
        return 1 if self.direction.startswith("higher") else -1

    @property
    def active(self) -> bool:
        """Counts toward its gauge (display-only indicators do not)."""
        return self.weight > 0 and self.role != "display"

    @property
    def lookback_days(self) -> int | None:
        """Trading days in the percentile window; None means all history."""
        lookback = self.lookback_pct or ("5y" if self.expected_freq == "1d" else "20y")
        return None if lookback == "all" else int(lookback[:-1]) * 252

    @property
    def freq_bdays(self) -> int:
        return FREQ_BDAYS[self.expected_freq]


class Fragility(Strict):
    divergence: dict[str, float]
    validated: bool = False


class Registry(Strict):
    gauges: dict[str, Gauge]
    indicators: list[Indicator]
    fragility: Fragility

    @model_validator(mode="after")
    def _consistent(self) -> Registry:
        ids = [i.id for i in self.indicators]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate indicator ids: {sorted(dupes)}")
        for ind in self.indicators:
            if ind.pillar not in self.gauges[ind.gauge].pillars:
                raise ValueError(f"{ind.id}: pillar {ind.pillar!r} is not a pillar of {ind.gauge}")
            greed = ind.direction.endswith("greed")
            if greed != (ind.gauge in ("G1", "G3")):
                raise ValueError(f"{ind.id}: {ind.direction} does not fit {ind.gauge}")
        for gauge, spec in self.gauges.items():
            for pillar in spec.pillars:
                if not any(i.gauge == gauge and i.pillar == pillar and i.active for i in self.indicators):
                    raise ValueError(f"{gauge}.{pillar} has no active indicator")
        return self

    def by_id(self, ind_id: str) -> Indicator:
        return next(i for i in self.indicators if i.id == ind_id)


def load_registry(path: Path = CONFIG_DIR / "indicators.yaml") -> Registry:
    with path.open(encoding="utf-8") as f:
        return Registry.model_validate(yaml.safe_load(f))
