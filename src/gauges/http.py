"""Shared HTTP plumbing: per-thread httpx clients, retries, throttles, redaction."""
from __future__ import annotations

import os
import re
import threading
import time
from datetime import date, datetime

import httpx
from tenacity import retry, retry_if_exception_type, retry_if_result, stop_after_attempt, wait_exponential

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0 Safari/537.36"
)
PROJECT_UA = "market-gauges/0.1 (data pipeline)"
RETRY_STATUS = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524}   # 52x: Cloudflare origin errors
SECRET_ENVS = ("FRED_API_KEY", "FINNHUB_API_KEY", "OPENROUTER_API_KEY", "EIA_API_KEY", "SEC_USER_AGENT",
               "MASSIVE_API_KEY", "FMP_API_KEY", "ALPHAVANTAGE_API_KEY")


class SourceError(RuntimeError):
    """A source answered, but not with usable data."""


class MissingKey(SourceError):
    def __init__(self, name: str):
        super().__init__(f"{name} is not set in .env")
        self.name = name


def env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def require(name: str) -> str:
    value = env(name)
    if not value:
        raise MissingKey(name)
    return value


_local = threading.local()


def client() -> httpx.Client:
    if not hasattr(_local, "client"):
        _local.client = httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": BROWSER_UA})
    return _local.client


@retry(
    retry=retry_if_exception_type(httpx.TransportError) | retry_if_result(lambda r: r.status_code in RETRY_STATUS),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry_error_callback=lambda state: state.outcome.result(),
)
def get(url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
    return client().get(url, params=params, headers=headers)


def get_ok(url: str, params: dict | None = None, headers: dict | None = None, what: str = "") -> httpx.Response:
    """GET and raise SourceError (without the query string, which may hold a key) on non-200."""
    r = get(url, params=params, headers=headers)
    if r.status_code != 200:
        raise SourceError(f"{what or url.split('?')[0]}: HTTP {r.status_code} {redact(r.text[:160])}")
    return r


class Throttle:
    """Spaces calls to one API across threads."""

    def __init__(self, interval: float):
        self.interval, self.lock, self.next_at = interval, threading.Lock(), 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next_at - now
            self.next_at = max(now, self.next_at) + self.interval
        if delay > 0:
            time.sleep(delay)


def redact(text: str) -> str:
    for name in SECRET_ENVS:
        secret = env(name)
        if secret and len(secret) >= 8:
            text = text.replace(secret, f"<{name}>")
    return text


def strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>|&nbsp;", " ", s)).strip()


def parse_mdy(s: str) -> date:
    return datetime.strptime(s.strip(), "%m/%d/%Y").date()


def month_start(year: int, month: int) -> date:
    """Monthly and quarterly observations are dated the first day of the period (FRED convention)."""
    return date(year, month, 1)
