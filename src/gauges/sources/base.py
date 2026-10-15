"""What a source client receives (Context) and returns (FetchResult)."""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import TypeVar
from zoneinfo import ZoneInfo

import pandas as pd

from gauges.config import Baskets, SeriesConfig
from gauges.schema import Obs

ET = ZoneInfo("America/New_York")
CLOSE_CUTOFF = time(16, 30)  # Cboe/Yahoo end-of-day data is settled by then
FAILED = "FAILED "           # note prefix that marks a source run as partial
T = TypeVar("T")


def completed_session(now_et: datetime) -> date:
    """Last weekday whose US session has closed (holidays are filtered by the data itself)."""
    d = now_et.date()
    if d.weekday() < 5 and now_et.time() >= CLOSE_CUTOFF:
        return d
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def market_open(now_et: datetime) -> bool:
    return now_et.weekday() < 5 and time(9, 30) <= now_et.time() < CLOSE_CUTOFF


def business_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


@dataclass
class Context:
    cfg: SeriesConfig
    baskets: Baskets
    last: dict[tuple[str, str], date]          # (source, series) -> latest stored obs_date
    snapshot_keys: dict[str, set[tuple]]       # table -> existing group keys
    now_et: datetime
    vintages: dict[tuple[str, str], date] = field(default_factory=dict)   # (source, series) -> newest vintage
    full: bool = False
    cache: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def today(self) -> date:
        return self.now_et.date()

    @property
    def session(self) -> date:
        return completed_session(self.now_et)

    @property
    def snapshots_allowed(self) -> bool:
        """Point-in-time snapshots (options, quotes) only when no session is in progress."""
        return not market_open(self.now_et)

    def last_date(self, source: str, series: str) -> date | None:
        return None if self.full else self.last.get((source, series))

    def last_vintage(self, source: str, series: str) -> date | None:
        return None if self.full else self.vintages.get((source, series))

    def last_date_prefix(self, source: str, prefix: str) -> date | None:
        if self.full:
            return None
        dates = [d for (s, k), d in self.last.items() if s == source and k.startswith(prefix)]
        return max(dates) if dates else None

    def cached(self, key: str, make: Callable[[], T]) -> T:
        """Compute once per run, shared across source threads."""
        with self.lock:
            if key not in self.cache:
                self.cache[key] = make()
            return self.cache[key]


@dataclass
class FetchResult:
    observations: list[Obs] | pd.DataFrame = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add_table(self, name: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        self.tables[name] = pd.concat([self.tables[name], df], ignore_index=True) if name in self.tables else df


def each(items: Iterable[T], fn: Callable[[T], list[Obs]], notes: list[str], label: Callable[[T], str] = str) -> list[Obs]:
    """Run fn per item; a failing item becomes a note instead of sinking the source."""
    out: list[Obs] = []
    failures = 0
    items = list(items)
    for item in items:
        try:
            out += fn(item)
        except Exception as e:  # noqa: BLE001
            failures += 1
            notes.append(f"{FAILED}{label(item)}: {type(e).__name__}: {e}")
    if items and failures == len(items):
        raise RuntimeError(f"all {failures} items failed; first: {notes[-failures]}")
    return out
