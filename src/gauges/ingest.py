"""Pull every source, validate, and append to the raw store.

Usage: python -m gauges.ingest [--only fred,cboe] [--skip yfinance] [--full] [--db PATH]

Fetches run in parallel threads; all writes happen on the main thread. Each
source's outcome is recorded in fetch_log, and every rejected row, out-of-bounds
value, cross-check mismatch and invalid events.yaml entry in validation_flags.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

from dotenv import load_dotenv

from gauges.config import ROOT, load_baskets, load_events, load_series_config
from gauges.http import MissingKey, redact
from gauges.schema import coerce
from gauges.sources import SOURCES
from gauges.sources.base import ET, FAILED, Context, FetchResult
from gauges.store import DEFAULT_DB, SNAPSHOT_TABLES, Store
from gauges.validate import apply_bounds, run_crosschecks


@dataclass
class Outcome:
    source: str
    status: str
    seconds: float
    fetched: int = 0
    inserted: int = 0
    rejected: int = 0
    latest: object = None
    tables: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _run(name: str, ctx: Context) -> tuple[FetchResult | None, str | None, str, float]:
    started = time.monotonic()
    try:
        return SOURCES[name](ctx), None, "ok", time.monotonic() - started
    except MissingKey as e:
        return None, str(e), "missing_key", time.monotonic() - started
    except Exception as e:  # noqa: BLE001 - a failed source is reported, never fatal
        return None, redact(f"{type(e).__name__}: {e}"), "failed", time.monotonic() - started


def write(store: Store, ctx: Context, name: str, result: FetchResult, run_id: str, fetched_at: datetime) -> Outcome:
    valid, bad_schema = coerce(result.observations, ctx.today)
    kept, bad_bounds, unmatched = apply_bounds(valid, ctx.cfg)
    out = Outcome(name, "ok", 0.0, fetched=len(valid) + len(bad_schema))
    out.inserted = store.append_observations(kept, run_id, fetched_at)
    out.rejected = len(bad_schema) + len(bad_bounds)
    store.flag_frame(run_id, fetched_at, "schema", "reject", bad_schema)
    store.flag_frame(run_id, fetched_at, "bounds", "reject", bad_bounds)
    for source, series in sorted(unmatched):
        store.flag(run_id, fetched_at, "no_spec", "warn", source=source, series=series,
                   detail="no entry in config/series.yaml specs; stored without bounds")
    for table, df in result.tables.items():
        out.tables[table] = store.append_snapshot(table, df, run_id, fetched_at)
    out.latest = kept["obs_date"].max().date() if not kept.empty else None
    out.notes = [redact(n) for n in result.notes]
    if unmatched:
        out.notes.append(f"{len(unmatched)} series without a spec")
    if any(n.startswith(FAILED) for n in out.notes) or out.rejected:
        out.status = "partial"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma-separated sources: " + ",".join(SOURCES))
    parser.add_argument("--skip", help="comma-separated sources to leave out")
    parser.add_argument("--full", action="store_true", help="ignore stored state and re-pull full history")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    names = list(SOURCES)
    for flag, keep in ((args.only, True), (args.skip, False)):
        if flag:
            chosen = {s.strip() for s in flag.split(",")}
            unknown = chosen - set(SOURCES)
            if unknown:
                parser.error(f"unknown sources: {', '.join(sorted(unknown))}")
            names = [n for n in names if (n in chosen) == keep]

    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    run_id = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    store = Store(args.db)
    events, event_errors = load_events()
    for err in event_errors:
        store.flag(run_id, fetched_at, "events", "reject", detail=err)
    ctx = Context(
        cfg=load_series_config(), baskets=load_baskets(), last=store.last_dates(),
        snapshot_keys={t: store.snapshot_keys(t, cols) for t, (mode, cols) in SNAPSHOT_TABLES.items() if mode == "group"},
        now_et=datetime.now(ET), full=args.full, vintages=store.last_vintages())

    print(f"run {run_id} · session {ctx.session} · {len(names)} sources · db {args.db}", flush=True)
    outcomes: list[Outcome] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_run, name, ctx): name for name in names}
        for fut in as_completed(futures):
            name = futures[fut]
            result, error, status, seconds = fut.result()
            if result is None:
                out = Outcome(name, status, seconds, notes=[error or ""])
            else:
                try:
                    out = write(store, ctx, name, result, run_id, fetched_at)
                except Exception as e:  # noqa: BLE001
                    out = Outcome(name, "failed", 0.0, notes=[redact(f"write: {type(e).__name__}: {e}")])
                out.seconds = seconds
            store.log_fetch(run_id=run_id, source=name, started_at=fetched_at,
                            finished_at=datetime.now(timezone.utc).replace(tzinfo=None), status=out.status,
                            rows_fetched=out.fetched, rows_inserted=out.inserted, rows_rejected=out.rejected,
                            latest_obs=out.latest, message=" | ".join(out.notes)[:2000])
            outcomes.append(out)
            print(f"  {out.status:<11} {name:<11} {out.seconds:6.1f}s  +{out.inserted:<8} latest {out.latest}", flush=True)

    checks = run_crosschecks(store, ctx.today)
    for c in checks:
        if c.ok is False:
            store.flag(run_id, fetched_at, "crosscheck", "flag", obs_date=c.obs_date, value=c.primary,
                       detail=f"{c.name}: {c.detail}")
    print(report(outcomes, checks, len(events), event_errors))
    store.close()
    if outcomes and all(o.status in ("failed", "missing_key") for o in outcomes):
        raise SystemExit(1)


def _num(v) -> str:
    return "—" if v is None else f"{v:,.4f}".rstrip("0").rstrip(".")


def report(outcomes: list[Outcome], checks, n_events: int, event_errors: list[str]) -> str:
    order = {name: i for i, name in enumerate(SOURCES)}
    lines = ["", "| Source | Status | Fetched | Inserted | Rejected | Latest obs | Tables | Notes |",
             "|---|---|---|---|---|---|---|---|"]
    for o in sorted(outcomes, key=lambda o: order[o.source]):
        tables = ", ".join(f"{t} +{n}" for t, n in o.tables.items()) or "—"
        notes = "; ".join(o.notes)[:220].replace("|", "/")
        lines.append(f"| {o.source} | {o.status} | {o.fetched:,} | {o.inserted:,} | {o.rejected:,} | "
                     f"{o.latest or '—'} | {tables} | {notes} |")
    lines += ["", "| Cross-check | Date | Primary | Secondary | Result |", "|---|---|---|---|---|"]
    for c in checks:
        result = "skipped" if c.ok is None else ("✅ pass" if c.ok else "❌ FLAG")
        lines.append(f"| {c.name} | {c.obs_date or '—'} | {_num(c.primary)} | {_num(c.secondary)} | {result}: {c.detail} |")
    lines += ["", f"events.yaml: {n_events} valid, {len(event_errors)} rejected"]
    lines += [f"  REJECT {e}" for e in event_errors]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
