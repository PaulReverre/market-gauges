"""Sanity bounds, cross-source checks and the events.yaml schema check (Section 3)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from gauges.config import SeriesConfig, load_events
from gauges.store import Store


def apply_bounds(df: pd.DataFrame, cfg: SeriesConfig) -> tuple[pd.DataFrame, pd.DataFrame, set[tuple[str, str]]]:
    """Split rows into (kept, rejected-with-reason) and report series with no spec."""
    if df.empty:
        return df, df.assign(reason=pd.Series(dtype=str)), set()
    pairs = df[["source", "series"]].drop_duplicates()
    lo, hi, unmatched = {}, {}, set()
    for source, series in pairs.itertuples(index=False):
        spec = cfg.spec_for(source, series)
        if spec is None:
            unmatched.add((source, series))
        elif spec.bounds:
            lo[(source, series)], hi[(source, series)] = spec.bounds
    keys = list(zip(df["source"], df["series"]))
    low = np.array([lo.get(k, -np.inf) for k in keys])
    high = np.array([hi.get(k, np.inf) for k in keys])
    values = df["value"].to_numpy(dtype=float)
    bad = (values < low) | (values > high)
    rejected = df[bad].assign(reason=[f"outside bounds [{l:g}, {h:g}]" for l, h in zip(low[bad], high[bad])])
    return df[~bad].reset_index(drop=True), rejected.reset_index(drop=True), unmatched


@dataclass
class CrossCheck:
    name: str
    primary: tuple[str, str]
    secondary: tuple[str, str]
    tolerance: float
    relative: bool
    secondary_scale: float = 1.0
    lookback: int = 5          # compare the latest N common dates
    window_days: int = 30      # ...within this many days of today

    def compare(self, a: float, b: float) -> tuple[float, bool]:
        b = b * self.secondary_scale
        diff = (a / b - 1) if self.relative else (a - b)
        return diff, abs(diff) <= self.tolerance + 1e-12


# Section 3.5, plus TSMC IR vs MOPS (NT$ millions vs thousands).
CROSSCHECKS = [
    CrossCheck("SPX close", ("yfinance", "^GSPC.close"), ("fred", "SP500"), 0.0015, relative=True),
    CrossCheck("VIX close", ("cboe", "VIX"), ("fred", "VIXCLS"), 0.005, relative=False),
    CrossCheck("10y yield", ("fred", "DGS10"), ("treasury", "10y"), 0.02, relative=False),
    CrossCheck("TSMC revenue", ("tsmc", "revenue_ntd_m"), ("mops", "2330.revenue_ntd_k"), 0.001, relative=True,
               secondary_scale=0.001, lookback=3, window_days=150),
]


@dataclass
class CheckResult:
    name: str
    obs_date: date | None
    primary: float | None
    secondary: float | None
    diff: float | None
    ok: bool | None          # None = skipped
    detail: str


def run_crosschecks(store: Store, today: date) -> list[CheckResult]:
    results = []
    for check in CROSSCHECKS:
        since = today - timedelta(days=check.window_days)
        a = store.series_points(*check.primary, since=since)
        b = store.series_points(*check.secondary, since=since)
        common = sorted(set(a) & set(b))[-check.lookback:]
        if not common:
            why = "no data" if not (a and b) else "no common dates"
            results.append(CheckResult(check.name, None, None, None, None, None, why))
            continue
        failures = []
        for d in common:
            diff, ok = check.compare(a[d], b[d])
            if not ok:
                failures.append((d, a[d], b[d], diff))
        latest = common[-1]
        diff, ok = check.compare(a[latest], b[latest])
        unit = "rel" if check.relative else "abs"
        detail = (f"{len(common)} dates checked, all within ±{check.tolerance:g} {unit}" if not failures else
                  f"{len(failures)}/{len(common)} dates outside ±{check.tolerance:g} {unit}: "
                  + "; ".join(f"{d} {x:g} vs {y:g}" for d, x, y, _ in failures[:3]))
        results.append(CheckResult(check.name, latest, a[latest], b[latest] * check.secondary_scale, diff,
                                   not failures, detail))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate config/events.yaml against its schema")
    parser.add_argument("what", choices=["events"])
    parser.parse_args()
    events, errors = load_events()
    print(f"{len(events)} valid events, {len(errors)} rejected")
    for e in errors:
        print(f"  REJECT {e}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
