"""AI company earnings releases (SEC 8-K Item 2.02, exhibit 99.1): leader tone and guidance.

For every earnings press release from the tracked AI companies since `start`:
- quote_positivity: executives' quoted remarks ("..." said <name>), net
  positive words per 100 words using the Loughran-McDonald finance sentiment
  lists (a positive-minus-negative ratio is pinned near +100: leaders rarely
  use negative finance words, so intensity is the informative part);
- release_tone: net tone per 1,000 words of the narrative before the
  forward-looking-statements boilerplate (which is full of risk words);
- ai_mentions: mentions of AI per 1,000 words (narrative intensity);
- rev_guide: midpoint of next-quarter revenue guidance in $ millions, where the
  release states it ("revenue is expected to be $X billion, plus or minus 2%").
Scores are word counts, never model output (spec rule 8). Values are known on
the filing date (lag in series.yaml covers after-close filings).

Loughran-McDonald dictionary (sraf.nd.edu): free for academic research; a
commercial license is required otherwise. It is downloaded at run time into
data/cache (git-ignored) and never redistributed.
"""
from __future__ import annotations

import csv
import html
import io
import re
from datetime import date

from gauges.config import ROOT
from gauges.http import SourceError, get_ok, require
from gauges.schema import Obs
from gauges.sources.base import Context, FetchResult
from gauges.sources.edgar import SUBMISSIONS, THROTTLE, cik_map, sec_json

SOURCE = "releases"
LM_URL = "https://drive.usercontent.google.com/download?id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF&export=download&confirm=t"
LM_CACHE = ROOT / "data" / "cache" / "lm_master_dictionary.csv"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}"
QUOTE_VERBS = r"(?:said|says|added|stated|commented|noted)"
QUOTE_AFTER = re.compile(r'"([^"]{40,1500})"\s*,?\s*' + QUOTE_VERBS + r'\s+([^,"]{3,90})')   # "Dr. Lisa Su, ..."
QUOTE_BEFORE = re.compile(r'([A-Z][A-Za-z.\'\- ]{3,60}),[^"]{0,160}?' + QUOTE_VERBS + r'[,:]?\s*"([^"]{40,1500})"')
BOILERPLATE = re.compile(r"forward-looking statements|safe harbor|cautionary statement", re.I)
MIN_QUOTE_WORDS = 15
AI_TERMS = re.compile(r"\bAI\b|\bartificial intelligence\b|\bgenerative\b|\baccelerated computing\b|\bLLMs?\b", re.I)
MONEY = r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million)"
GUIDE_PATTERNS = [   # (pattern, kind)
    (re.compile(r"revenue (?:is|are) expected to be " + MONEY, re.I), "point"),
    (re.compile(r"expects? (?:net |total )?revenues? (?:for [^,.]{0,60})?to be approximately " + MONEY, re.I), "point"),
    (re.compile(r"(?:net |total )?revenues? (?:is expected to be |of )?(?:in the range of|between) " + MONEY
                + r"\s*(?:and|to)\s*" + MONEY, re.I), "range"),
    (re.compile(r"(?:guidance|outlook)[^.]{0,80}revenues? of approximately " + MONEY, re.I), "point"),
]


# ---------------------------------------------------------------- dictionary
def load_lexicon(ctx: Context | None = None) -> tuple[frozenset[str], frozenset[str]]:
    def load() -> tuple[frozenset[str], frozenset[str]]:
        if not LM_CACHE.exists():
            LM_CACHE.parent.mkdir(parents=True, exist_ok=True)
            LM_CACHE.write_bytes(get_ok(LM_URL, what="Loughran-McDonald dictionary").content)
        return parse_lexicon(LM_CACHE.read_text(encoding="utf-8", errors="replace"))
    return ctx.cached("lm_lexicon", load) if ctx else load()


def parse_lexicon(text: str) -> tuple[frozenset[str], frozenset[str]]:
    pos, neg = set(), set()
    for row in csv.DictReader(io.StringIO(text)):
        word = row["Word"].strip().upper()
        if row.get("Positive", "0") not in ("0", ""):
            pos.add(word)
        if row.get("Negative", "0") not in ("0", ""):
            neg.add(word)
    if len(pos) < 100 or len(neg) < 1000:
        raise SourceError(f"Loughran-McDonald lists look truncated ({len(pos)} positive, {len(neg)} negative)")
    return frozenset(pos), frozenset(neg)


# ------------------------------------------------------------------ parsing
def clean_text(raw_html: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw_html))
    text = text.translate({0x201C: '"', 0x201D: '"', 0x2018: "'", 0x2019: "'", 0xA0: " "})
    return re.sub(r"\s+", " ", text).strip()


def executive_quotes(text: str) -> list[tuple[str, str]]:
    """(speaker, quote) pairs for quotes attributed with said/says/added..."""
    out = [(who.strip(), q) for q, who in QUOTE_AFTER.findall(text)]
    out += [(who.strip(), q) for who, q in QUOTE_BEFORE.findall(text)]
    seen, unique = set(), []
    for who, q in out:
        if q not in seen:
            seen.add(q)
            unique.append((who, q))
    return unique


def tone_counts(text: str, lexicon: tuple[frozenset[str], frozenset[str]]) -> tuple[int, int, int]:
    words = re.findall(r"[A-Za-z]+", text.upper())
    pos, neg = lexicon
    return sum(w in pos for w in words), sum(w in neg for w in words), len(words)


def _usd_millions(amount: str, unit: str) -> float:
    value = float(amount.replace(",", ""))
    return value * 1000 if unit.lower() == "billion" else value


def revenue_guidance(text: str) -> float | None:
    """Midpoint of next-quarter revenue guidance in $ millions, if stated in a recognised form."""
    for pattern, kind in GUIDE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        if kind == "point":
            return _usd_millions(m.group(1), m.group(2))
        low, high = _usd_millions(m.group(1), m.group(2)), _usd_millions(m.group(3), m.group(4))
        return (low + high) / 2
    return None


def narrative(text: str) -> str:
    """The release up to its forward-looking-statements section (if that is not in the opening lines)."""
    m = BOILERPLATE.search(text)
    return text[:m.start()] if m and m.start() > 1500 else text


def analyse(text: str, lexicon) -> dict[str, float]:
    out: dict[str, float] = {}
    quotes = executive_quotes(text)
    if quotes:
        pos, neg, n = tone_counts(" ".join(q for _, q in quotes), lexicon)
        out["quote_words"] = float(n)
        if n >= MIN_QUOTE_WORDS:
            out["quote_positivity"] = (pos - neg) / n * 100
    body = narrative(text)
    pos, neg, n = tone_counts(body, lexicon)
    if n >= 200:
        out["release_tone"] = (pos - neg) / n * 1000
        out["ai_mentions"] = len(AI_TERMS.findall(body)) / n * 1000
    guide = revenue_guidance(text)
    if guide is not None:
        out["rev_guide"] = guide
    return out


# ------------------------------------------------------------------ fetching
def sec_text(url: str) -> str:
    THROTTLE.wait()
    return get_ok(url, headers={"User-Agent": require("SEC_USER_AGENT"), "Accept-Encoding": "gzip, deflate"},
                  what="sec archive").text


def earnings_filings(cik: int, since: date) -> list[tuple[str, date]]:
    """(accession, filing date) of 8-K Item 2.02 filings since `since`, including older submission pages."""
    payload = sec_json(SUBMISSIONS.format(cik=cik))
    pages = [payload["filings"]["recent"]]
    for f in payload["filings"].get("files", []):
        if date.fromisoformat(f["filingTo"]) >= since:
            pages.append(sec_json(f"https://data.sec.gov/submissions/{f['name']}"))
    out = []
    for page in pages:
        for form, items, accn, filed in zip(page["form"], page["items"], page["accessionNumber"], page["filingDate"]):
            d = date.fromisoformat(filed)
            if form == "8-K" and "2.02" in (items or "").split(",") and d >= since:
                out.append((accn, d))
    return sorted(set(out), key=lambda x: x[1])


def exhibit_url(cik: int, accn: str) -> str | None:
    folder = accn.replace("-", "")
    index = sec_text(f"{ARCHIVES.format(cik=cik, folder=folder)}/{accn}-index.htm")
    candidates = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", index, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(cells) < 4:
            continue
        kind = re.sub(r"<[^>]+>", "", cells[3]).strip().upper()
        link = re.search(r'href="([^"]+\.html?)"', cells[2])
        if link and re.fullmatch(r"EX-99(\.0?1)?", kind):
            candidates.append((kind != "EX-99.1" and kind != "EX-99.01", link.group(1)))
    return "https://www.sec.gov" + sorted(candidates)[0][1] if candidates else None


def fetch(ctx: Context) -> FetchResult:
    lexicon = load_lexicon(ctx)
    since = ctx.cfg.fetch.releases.start
    ciks = cik_map(ctx)
    tickers = [t for t in dict.fromkeys(ctx.cfg.fetch.edgar.tickers + ctx.baskets.tickers) if "." not in t]
    rows: list[Obs] = []
    notes: list[str] = []
    for ticker in tickers:
        if ticker not in ciks:
            notes.append(f"{ticker}: no SEC filer (foreign issuer)")
            continue
        last = ctx.last_date(SOURCE, f"{ticker}.release_tone")
        try:
            filings = [(a, d) for a, d in earnings_filings(ciks[ticker], since) if last is None or d > last]
        except SourceError as e:
            notes.append(f"FAILED {ticker} submissions: {e}")
            continue
        for accn, filed in filings:
            try:
                url = exhibit_url(ciks[ticker], accn)
                if not url:
                    continue
                values = analyse(clean_text(sec_text(url)), lexicon)
            except SourceError as e:
                notes.append(f"FAILED {ticker} {accn}: {e}")
                continue
            rows += [Obs(SOURCE, f"{ticker}.{name}", filed, v, None, url) for name, v in values.items()]
    found = {r.series.split(".")[0] for r in rows}
    notes.append(f"{len({(r.series.split('.')[0], r.obs_date) for r in rows})} releases from {len(found)} companies")
    return FetchResult(rows, notes=notes)

