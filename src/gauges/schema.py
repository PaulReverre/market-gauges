"""Row schema for raw observations, enforced on every write.

Every stored value carries its source, series id, observation date, the date it
became public when the source says so (`vintage`; NULL means known at the
obs_date close), and the endpoint it came from. fetched_at and run_id are added
by the store.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd

OBS_COLUMNS = ["source", "series", "obs_date", "value", "vintage", "endpoint"]
KEY_COLUMNS = ["source", "series", "obs_date", "vintage"]


class Obs(NamedTuple):
    source: str
    series: str
    obs_date: date
    value: float
    vintage: date | None
    endpoint: str


def to_frame(rows: Iterable[Obs] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.loc[:, OBS_COLUMNS].copy()
    return pd.DataFrame(list(rows), columns=OBS_COLUMNS)


def coerce(rows: Iterable[Obs] | pd.DataFrame, today: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valid, rejected). Rejected rows carry a `reason` column."""
    df = to_frame(rows)
    if df.empty:
        return df, df.assign(reason=pd.Series(dtype=str))
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
    df["obs_date"] = pd.to_datetime(df["obs_date"], errors="coerce")
    df["vintage"] = pd.to_datetime(df["vintage"], errors="coerce")
    for col in ("source", "series", "endpoint"):
        df[col] = df[col].astype("string")

    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    reason[df["endpoint"].isna() | (df["endpoint"] == "")] = "missing endpoint"
    reason[df["series"].isna() | (df["series"] == "")] = "missing series"
    reason[df["source"].isna() | (df["source"] == "")] = "missing source"
    reason[df["obs_date"].isna()] = "bad obs_date"
    reason[~np.isfinite(df["value"].to_numpy())] = "non-finite value"
    reason[df["obs_date"] > pd.Timestamp(today + timedelta(days=1))] = "obs_date in the future"

    bad = reason.notna()
    rejected = df[bad].assign(reason=reason[bad])
    valid = df[~bad].drop_duplicates(subset=KEY_COLUMNS, keep="last").reset_index(drop=True)
    return valid, rejected.reset_index(drop=True)
