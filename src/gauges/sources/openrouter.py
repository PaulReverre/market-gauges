"""OpenRouter: daily tokens for the top 50 models (Data API) and a daily model catalog.

Licensed CC BY 4.0. Cite "Source: OpenRouter (openrouter.ai/rankings), as of
{as_of}". Token counts use each provider's tokenizer and are not comparable
across providers. The API serves data from 2025-01-01 onward. Today's partial
day is excluded so re-runs are idempotent.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from gauges.http import get_ok, require
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult

SOURCE = "openrouter"
RANKINGS = "https://openrouter.ai/api/v1/datasets/rankings-daily"
MODELS = "https://openrouter.ai/api/v1/models"
RESTATE_DAYS = 3   # figures may be restated as snapshots refresh


def parse_rankings(payload: dict) -> list[Obs]:
    out = []
    for row in payload.get("data", []):
        if row.get("total_tokens") in (None, ""):
            continue
        out.append(Obs(SOURCE, f"tokens:{row['model_permaslug']}", date.fromisoformat(row["date"]),
                       float(row["total_tokens"]), None, RANKINGS))
    return out


def parse_models(payload: dict, snapshot: date) -> pd.DataFrame:
    rows = []
    for m in payload.get("data", []):
        pricing = m.get("pricing") or {}
        rows.append({
            "snapshot_date": snapshot, "model_id": m["id"], "canonical_slug": m.get("canonical_slug"),
            "author": m["id"].split("/")[0],
            "created": datetime.fromtimestamp(m["created"], timezone.utc).replace(tzinfo=None) if m.get("created") else None,
            "prompt_price": _price(pricing.get("prompt")), "completion_price": _price(pricing.get("completion")),
            "hugging_face_id": m.get("hugging_face_id") or None, "context_length": m.get("context_length")})
    return pd.DataFrame(rows)


def _price(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch(ctx: Context) -> FetchResult:
    key = require("OPENROUTER_API_KEY")
    utc_today = datetime.now(timezone.utc).date()
    end = utc_today - timedelta(days=1)
    last = ctx.last_date_prefix(SOURCE, "tokens:")
    start = max(ctx.cfg.fetch.openrouter.start, last - timedelta(days=RESTATE_DAYS)) if last else ctx.cfg.fetch.openrouter.start
    rows: list[Obs] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(end, date(chunk_start.year, 12, 31))
        payload = get_ok(RANKINGS, params={"start_date": chunk_start.isoformat(), "end_date": chunk_end.isoformat()},
                         headers={"Authorization": f"Bearer {key}"}, what="openrouter rankings-daily").json()
        rows += parse_rankings(payload)
        chunk_start = chunk_end + timedelta(days=1)

    result = FetchResult(rows, notes=[f"rankings {start}..{end}: {len(rows)} rows"])
    if (utc_today,) not in ctx.snapshot_keys.get("model_catalog", set()):
        result.add_table("model_catalog", parse_models(get_ok(MODELS, what="openrouter models").json(), utc_today))
    return result
