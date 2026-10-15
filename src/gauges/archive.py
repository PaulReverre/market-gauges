"""Git-friendly archive of the raw store: one parquet file per ingest run and table.

The DuckDB file stays local (git-ignored). Each ingest run writes the rows it
added as data/archive/<table>/<run_id>.parquet; files are written once and never
rewritten, so a daily commit only adds small files. `restore` rebuilds an empty
store from the archive, e.g. on a fresh GitHub Actions runner.

Usage: python -m gauges.archive export [--all] | restore [--db PATH] [--force]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from gauges.config import ROOT
from gauges.store import DEFAULT_DB, Store

ARCHIVE = ROOT / "data" / "archive"
TABLES = ["raw_observations", "option_snapshots", "xbrl_facts", "earnings_events", "index_holdings",
          "model_catalog", "fetch_log", "validation_flags"]
RUN_ID = re.compile(r"^\d{8}T\d{6}Z$")


def export_run(store: Store, run_id: str, root: Path = ARCHIVE) -> dict[str, int]:
    """Write this run's rows for every table that has some; existing files are left alone."""
    if not RUN_ID.match(run_id):
        raise ValueError(f"unexpected run id {run_id!r}")
    written = {}
    for table in TABLES:
        n = store.con.execute(f"SELECT count(*) FROM {table} WHERE run_id = ?", [run_id]).fetchone()[0]
        path = root / table / f"{run_id}.parquet"
        if n == 0 or path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        store.con.execute(f"COPY (SELECT * FROM {table} WHERE run_id = '{run_id}') TO '{path}' "
                          "(FORMAT parquet, COMPRESSION zstd)")
        written[table] = n
    return written


def export_all(store: Store, root: Path = ARCHIVE) -> dict[str, dict[str, int]]:
    runs = sorted({r[0] for t in TABLES for r in store.con.execute(f"SELECT DISTINCT run_id FROM {t}").fetchall()})
    return {run: export_run(store, run, root) for run in runs}


def restore(store: Store, root: Path = ARCHIVE, force: bool = False) -> dict[str, int]:
    """Load every archived file into an empty store (or replace its contents with force)."""
    existing = store.con.execute("SELECT count(*) FROM raw_observations").fetchone()[0]
    if existing and not force:
        raise RuntimeError(f"store already holds {existing:,} raw rows; pass --force to rebuild it from the archive")
    counts = {}
    for table in TABLES:
        files = sorted((root / table).glob("*.parquet"))
        if force:
            store.con.execute(f"DELETE FROM {table}")
        if not files:
            continue
        paths = ", ".join(f"'{f}'" for f in files)
        store.con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM read_parquet([{paths}], union_by_name = true)")
        counts[table] = store.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write archive files for runs that do not have them yet")
    ex.add_argument("--all", action="store_true", help="every run in the store (default: the latest run)")
    rs = sub.add_parser("restore", help="rebuild the store from data/archive")
    rs.add_argument("--force", action="store_true", help="replace the store's raw tables")
    for p in (ex, rs):
        p.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()
    store = Store(args.db)
    try:
        if args.cmd == "restore":
            for table, n in restore(store, force=args.force).items():
                print(f"{table:<18} {n:>10,} rows")
        elif args.all:
            for run, written in export_all(store).items():
                print(run, written or "already archived")
        else:
            run = store.con.execute("SELECT max(run_id) FROM fetch_log").fetchone()[0]
            print(run, export_run(store, run) or "already archived")
    finally:
        store.close()


if __name__ == "__main__":
    main()
