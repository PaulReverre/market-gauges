"""GPU rental prices: Uvik GPU Price Index (monthly, CC BY 4.0), GPUPerHour (daily) and
Nebius spot prices for pre-emptible VMs (hourly, public API).

The SemiAnalysis H100 1-year contract index has no public feed; it is a
monthly manual events.yaml entry. Uvik publishes one CSV per month with no back
history, so history accumulates from the first run. Nebius keeps 30 days of
spot history, so each run re-pulls the window and the store keeps the rest.
"""
from __future__ import annotations

import csv
import io
from datetime import date, datetime, timedelta, timezone

from gauges.http import get, get_ok
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult

UVIK = "https://uvik.net/wp-content/uploads/{y}/{m:02d}/uvik-software-gpu-price-index-{y}-{m:02d}.csv"
GPUPERHOUR = "https://api.gpuperhour.com/api/price-index/export.csv"
GPUPERHOUR_STATS = ["min", "p25", "median", "p75", "max"]
# docs.nebius.com/signup-billing/spot-prices#rest-api (no authentication)
NEBIUS = "https://api.nebius.cloud/billing/v1alpha1/spot-price/list-series"
NEBIUS_HISTORY_DAYS = 29   # the API accepts start times within the last 30 days


def parse_uvik(text: str, endpoint: str) -> list[Obs]:
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        if not row.get("usd_per_gpu_hour") or not row.get("captured"):
            continue
        series = "|".join(row[k].strip() for k in ("gpu", "provider", "product", "pricing"))
        out.append(Obs("uvik", series, date.fromisoformat(row["captured"]), float(row["usd_per_gpu_hour"]), None,
                       endpoint))
    return out


def parse_gpuperhour(text: str, before: date) -> list[Obs]:
    """Daily stats per GPU; today's snapshot can still change, so it is left for the next run."""
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        d = date.fromisoformat(row["snapshot_date"])
        if d >= before:
            continue
        for stat in GPUPERHOUR_STATS:
            v = row.get(f"{stat}_price_per_gpu")
            if v not in (None, ""):
                out.append(Obs("gpuperhour", f"{row['gpu_slug']}.{stat}", d, float(v), None, GPUPERHOUR))
    return out


def parse_nebius(payload: dict, start: date, before: date) -> list[Obs]:
    """UTC-daily mean of hourly average spot prices per platform and region.

    A point's timestamp is the end of its hour. Days before `start` (Nebius moved
    pre-emptible VMs from fixed to dynamic prices on 2026-10-08) and from
    `before` on (today is incomplete) are left out.
    """
    out = []
    for item in payload.get("items", []):
        series = f"{item['computeInstance']['platform']}.{item['region']}"
        days: dict[date, list[float]] = {}
        for p in item.get("points", []):
            hour = datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00")) - timedelta(hours=1)
            days.setdefault(hour.date(), []).append(float(p["value"]))
        for d, values in sorted(days.items()):
            if start <= d < before:
                out.append(Obs("nebius", series, d, sum(values) / len(values), None, NEBIUS))
    return out


def fetch_nebius(ctx: Context) -> FetchResult:
    utc_today = datetime.now(timezone.utc).date()
    since = utc_today - timedelta(days=NEBIUS_HISTORY_DAYS)
    params = {"filters.startTime": f"{since.isoformat()}T00:00:00Z", "aggregation.granularity": "GRANULARITY_HOUR_1",
              "aggregation.method": "METHOD_AVG_PRICE", "pageSize": 100}
    rows, token = [], None
    while True:
        payload = get_ok(NEBIUS, params={**params, **({"pageToken": token} if token else {})}, what="nebius").json()
        rows += parse_nebius(payload, ctx.cfg.fetch.nebius.start, utc_today)
        token = payload.get("nextPageToken")
        if not token:
            break
    return FetchResult(rows, notes=["pre-emptible spot prices, daily mean of hourly averages (UTC)"])


def fetch_uvik(ctx: Context) -> FetchResult:
    rows, months = [], []
    y, m = ctx.today.year, ctx.today.month
    for _ in range(3):
        url = UVIK.format(y=y, m=m)
        r = get(url)
        if r.status_code == 200 and r.text.startswith("provider"):
            rows += parse_uvik(r.text, url)
            months.append(f"{y}-{m:02d}")
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    if not rows:
        raise RuntimeError("no Uvik CSV in the last 3 months")
    return FetchResult(rows, notes=[f"months found: {', '.join(months)}"])


def fetch_gpuperhour(ctx: Context) -> FetchResult:
    utc_today = datetime.now(timezone.utc).date()
    rows = parse_gpuperhour(get_ok(GPUPERHOUR, params={"days": 365}, what="gpuperhour").text, utc_today)
    return FetchResult(rows, notes=["on-demand marketplace prices, not 1-year contracts"])
