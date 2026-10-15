"""DuckDB store: an append-only raw observation table plus snapshot tables.

Appends are idempotent. An observation is written only when its key
(source, series, obs_date, vintage) is new or its value differs from the latest
stored value for that key, so re-running a fetch never duplicates rows and a
source revision is kept as a new row with a later fetched_at.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from gauges.config import ROOT

DEFAULT_DB = ROOT / "data" / "gauges.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_observations (
    source     VARCHAR   NOT NULL,
    series     VARCHAR   NOT NULL,
    obs_date   DATE      NOT NULL,
    value      DOUBLE    NOT NULL,
    vintage    DATE,                -- date the value became public; NULL = known at the obs_date close
    endpoint   VARCHAR   NOT NULL,
    fetched_at TIMESTAMP NOT NULL,
    run_id     VARCHAR   NOT NULL
);

CREATE OR REPLACE VIEW latest_observations AS
SELECT * EXCLUDE (rn) FROM (
    SELECT *, row_number() OVER (
        PARTITION BY source, series, obs_date, vintage ORDER BY fetched_at DESC) AS rn
    FROM raw_observations
) WHERE rn = 1;

CREATE TABLE IF NOT EXISTS fetch_log (
    run_id        VARCHAR NOT NULL,
    source        VARCHAR NOT NULL,
    started_at    TIMESTAMP NOT NULL,
    finished_at   TIMESTAMP NOT NULL,
    status        VARCHAR NOT NULL,     -- ok | partial | failed | missing_key
    rows_fetched  BIGINT,
    rows_inserted BIGINT,
    rows_rejected BIGINT,
    latest_obs    DATE,
    message       VARCHAR
);

CREATE TABLE IF NOT EXISTS validation_flags (
    run_id     VARCHAR NOT NULL,
    flagged_at TIMESTAMP NOT NULL,
    check_name VARCHAR NOT NULL,        -- bounds | schema | no_spec | crosscheck | events
    severity   VARCHAR NOT NULL,        -- reject | flag | warn
    source     VARCHAR,
    series     VARCHAR,
    obs_date   DATE,
    value      DOUBLE,
    detail     VARCHAR
);

CREATE TABLE IF NOT EXISTS option_snapshots (
    snapshot_date    DATE NOT NULL,
    underlying       VARCHAR NOT NULL,
    expiry           DATE NOT NULL,
    option_type      VARCHAR NOT NULL,   -- call | put
    strike           DOUBLE NOT NULL,
    bid              DOUBLE,
    ask              DOUBLE,
    last             DOUBLE,
    volume           DOUBLE,
    open_interest    DOUBLE,
    implied_vol      DOUBLE,
    underlying_price DOUBLE,
    fetched_at       TIMESTAMP NOT NULL,
    run_id           VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS xbrl_facts (
    cik          BIGINT NOT NULL,
    ticker       VARCHAR NOT NULL,
    taxonomy     VARCHAR NOT NULL,
    tag          VARCHAR NOT NULL,
    unit         VARCHAR NOT NULL,
    period_start DATE,
    period_end   DATE NOT NULL,
    value        DOUBLE NOT NULL,
    form         VARCHAR,
    fy           BIGINT,
    fp           VARCHAR,
    filed        DATE NOT NULL,          -- point-in-time availability
    accn         VARCHAR NOT NULL,
    fetched_at   TIMESTAMP NOT NULL,
    run_id       VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS earnings_events (
    ticker           VARCHAR NOT NULL,
    report_date      DATE NOT NULL,
    source           VARCHAR NOT NULL,   -- finnhub_calendar | edgar_8k_202
    hour             VARCHAR,            -- bmo | amc | dmh
    fiscal_period    VARCHAR,
    eps_actual       DOUBLE,
    eps_estimate     DOUBLE,
    revenue_actual   DOUBLE,
    revenue_estimate DOUBLE,
    accepted_at      VARCHAR,
    accn             VARCHAR,
    fetched_at       TIMESTAMP NOT NULL,
    run_id           VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS index_holdings (
    as_of      DATE NOT NULL,
    fund       VARCHAR NOT NULL,
    ticker     VARCHAR NOT NULL,
    name       VARCHAR,
    weight     DOUBLE,
    shares     DOUBLE,
    identifier VARCHAR,
    fetched_at TIMESTAMP NOT NULL,
    run_id     VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS model_catalog (
    snapshot_date    DATE NOT NULL,
    model_id         VARCHAR NOT NULL,
    canonical_slug   VARCHAR,
    author           VARCHAR,
    created          TIMESTAMP,
    prompt_price     DOUBLE,
    completion_price DOUBLE,
    hugging_face_id  VARCHAR,
    context_length   BIGINT,
    fetched_at       TIMESTAMP NOT NULL,
    run_id           VARCHAR NOT NULL
);
"""

# How each snapshot table de-duplicates. "group": a snapshot is written once per
# key (re-running the same day never adds a second copy). "row": a row is
# written unless an identical row is already stored, so changed values append.
SNAPSHOT_TABLES: dict[str, tuple[str, list[str]]] = {
    "option_snapshots": ("group", ["snapshot_date", "underlying"]),
    "index_holdings": ("group", ["as_of", "fund"]),
    "model_catalog": ("group", ["snapshot_date"]),
    "xbrl_facts": ("row", []),
    "earnings_events": ("row", []),
}


class Store:
    def __init__(self, path: Path | str = DEFAULT_DB):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute(SCHEMA)

    def close(self) -> None:
        self.con.close()

    # ----------------------------------------------------------------- state
    def last_dates(self) -> dict[tuple[str, str], date]:
        rows = self.con.execute(
            "SELECT source, series, max(obs_date) FROM raw_observations GROUP BY 1, 2").fetchall()
        return {(s, k): d for s, k, d in rows}

    def last_vintages(self) -> dict[tuple[str, str], date]:
        rows = self.con.execute(
            "SELECT source, series, max(vintage) FROM raw_observations WHERE vintage IS NOT NULL GROUP BY 1, 2").fetchall()
        return {(s, k): d for s, k, d in rows}

    def snapshot_keys(self, table: str, columns: list[str]) -> set[tuple]:
        cols = ", ".join(columns)
        return set(self.con.execute(f"SELECT DISTINCT {cols} FROM {table}").fetchall())

    # ---------------------------------------------------------------- writes
    def append_observations(self, df: pd.DataFrame, run_id: str, fetched_at: datetime) -> int:
        """Insert rows that are new, or whose value changed versus the latest stored row."""
        if df.empty:
            return 0
        staged = df.copy()
        staged["obs_date"] = pd.to_datetime(staged["obs_date"]).dt.date
        staged["vintage"] = pd.to_datetime(staged["vintage"]).dt.date
        staged["vintage"] = staged["vintage"].astype(object).where(staged["vintage"].notna(), None)
        self.con.register("staged_obs", staged)
        try:
            before = self._count("raw_observations")
            self.con.execute(
                """
                INSERT INTO raw_observations
                SELECT s.source, s.series, CAST(s.obs_date AS DATE), s.value, CAST(s.vintage AS DATE),
                       s.endpoint, ?, ?
                FROM staged_obs s
                LEFT JOIN latest_observations l
                  ON l.source = s.source AND l.series = s.series
                 AND l.obs_date = CAST(s.obs_date AS DATE)
                 AND l.vintage IS NOT DISTINCT FROM CAST(s.vintage AS DATE)
                WHERE l.value IS NULL
                   OR abs(l.value - s.value) > 1e-9 * greatest(1.0, abs(s.value))
                """,
                [fetched_at, run_id],
            )
            return self._count("raw_observations") - before
        finally:
            self.con.unregister("staged_obs")

    def append_snapshot(self, table: str, df: pd.DataFrame, run_id: str, fetched_at: datetime) -> int:
        """Insert snapshot rows per the table's de-duplication mode (see SNAPSHOT_TABLES)."""
        if df.empty:
            return 0
        mode, group_cols = SNAPSHOT_TABLES[table]
        cols = list(df.columns)
        match_cols = group_cols if mode == "group" else cols
        match = " AND ".join(f"t.{c} IS NOT DISTINCT FROM s.{c}" for c in match_cols)
        self.con.register("staged_snap", df)
        try:
            before = self._count(table)
            self.con.execute(
                f"""
                INSERT INTO {table} ({", ".join(cols)}, fetched_at, run_id)
                SELECT DISTINCT s.*, ?, ? FROM staged_snap s
                WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE {match})
                """,
                [fetched_at, run_id],
            )
            return self._count(table) - before
        finally:
            self.con.unregister("staged_snap")

    def log_fetch(self, **row) -> None:
        cols = list(row)
        self.con.execute(f"INSERT INTO fetch_log ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                         list(row.values()))

    def flag(self, run_id: str, flagged_at: datetime, check_name: str, severity: str, *, source=None,
             series=None, obs_date=None, value=None, detail: str = "") -> None:
        self.con.execute(
            "INSERT INTO validation_flags VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [run_id, flagged_at, check_name, severity, source, series, obs_date, value, detail])

    def flag_frame(self, run_id: str, flagged_at: datetime, check_name: str, severity: str,
                   df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = df.assign(run_id=run_id, flagged_at=flagged_at, check_name=check_name, severity=severity)
        rows["obs_date"] = pd.to_datetime(rows["obs_date"], errors="coerce").dt.date
        rows = rows[["run_id", "flagged_at", "check_name", "severity", "source", "series", "obs_date", "value",
                     "reason"]]
        self.con.register("staged_flags", rows)
        try:
            self.con.execute("INSERT INTO validation_flags SELECT * FROM staged_flags")
        finally:
            self.con.unregister("staged_flags")

    # ----------------------------------------------------------------- reads
    def latest_value(self, source: str, series: str, on: date | None = None) -> tuple[date, float] | None:
        sql = ("SELECT obs_date, value FROM latest_observations WHERE source = ? AND series = ?"
               + (" AND obs_date = ?" if on else "") + " ORDER BY obs_date DESC, vintage DESC NULLS LAST LIMIT 1")
        row = self.con.execute(sql, [source, series, on] if on else [source, series]).fetchone()
        return (row[0], row[1]) if row else None

    def series_points(self, source: str, series: str, since: date | None = None) -> dict[date, float]:
        """Latest-vintage value per obs_date."""
        rows = self.con.execute(
            """
            SELECT obs_date, value FROM (
                SELECT obs_date, value, row_number() OVER (
                    PARTITION BY obs_date ORDER BY vintage DESC NULLS LAST, fetched_at DESC) AS rn
                FROM raw_observations WHERE source = ? AND series = ? AND obs_date >= ?)
            WHERE rn = 1
            """,
            [source, series, since or date(1900, 1, 1)],
        ).fetchall()
        return dict(rows)

    def _count(self, table: str) -> int:
        return self.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
