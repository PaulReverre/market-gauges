"""Wikimedia Pageviews API: daily human pageviews for attention proxies."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gauges.http import PROJECT_UA, SourceError, get
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult, each

SOURCE = "wikimedia"
URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
       "{article}/daily/{start:%Y%m%d}/{end:%Y%m%d}")


def parse_pageviews(payload: dict, article: str, endpoint: str) -> list[Obs]:
    return [Obs(SOURCE, f"pageviews.{article}", datetime.strptime(i["timestamp"][:8], "%Y%m%d").date(),
                float(i["views"]), None, endpoint) for i in payload.get("items", [])]


def fetch(ctx: Context) -> FetchResult:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    notes: list[str] = []

    def one(article: str) -> list[Obs]:
        last = ctx.last_date(SOURCE, f"pageviews.{article}")
        start = (last + timedelta(days=1)) if last else ctx.cfg.fetch.wikimedia.start
        if start > end:
            return []
        url = URL.format(article=article, start=start, end=end)
        r = get(url, headers={"User-Agent": PROJECT_UA})
        if r.status_code == 404:   # no views recorded in the range
            return []
        if r.status_code != 200:
            raise SourceError(f"HTTP {r.status_code}")
        return parse_pageviews(r.json(), article, url.split("/daily/")[0])

    return FetchResult(each(ctx.cfg.fetch.wikimedia.articles, one, notes), notes=notes)
