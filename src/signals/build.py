"""Daily build: fetch, score, write data/signals.json and data/signals_history.json.

Usage: python -m signals.build [--no-companies] [--if-stale]

Exits non-zero (leaving the previous JSON in place) only when the core price
and volatility history cannot be fetched. Every other source falls back to its
last good copy in data/state and is reported in the "sources" block.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from signals import alerts as A
from signals import companies as C
from signals import fetch as F
from signals import gauges as G
from signals import longterm as L
from signals import model as M

DATA = F.ROOT / "data"
CORE = ["^GSPC", "^VIX", "^SOX"]


def next_refresh(now: datetime) -> datetime:
    """The workflow runs at 02:00 UTC Tuesday to Saturday (after each Monday to Friday close)."""
    t = now.replace(hour=2, minute=0, second=0, microsecond=0)
    while t <= now or t.weekday() not in (1, 2, 3, 4, 5):
        t += timedelta(days=1)
    return t


def _repo_slug() -> str | None:
    """owner/name of the GitHub remote, for the dashboard's "rebuild" and "edit thresholds" links."""
    try:
        url = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True, text=True,
                             cwd=F.ROOT, timeout=10).stdout.strip()
        m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
        return m.group(1) if m else None
    except Exception:  # noqa: BLE001
        return None


def is_current(session) -> bool:
    """True when the published file is for `session`, no source failed and no scored input lags it."""
    path = DATA / "signals.json"
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
        scored = [i for g in d["gauges"].values() for i in g["indicators"] if i["role"] == "scored"]
        return (d["as_of"] == session.isoformat() and all(s["ok"] for s in d["sources"])
                and all(i["as_of"] == d["as_of"] for i in scored) and d.get("longterm") is not None)
    except Exception:  # noqa: BLE001 - an unreadable file is simply rebuilt
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-companies", action="store_true", help="skip the Finnhub earnings columns")
    ap.add_argument("--if-stale", action="store_true",
                    help="do nothing when data/signals.json already covers the last session with every input current")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(F.ROOT / ".env")
    except ImportError:
        pass

    log = F.SourceLog()
    session = F.completed_session()
    print(f"session {session}", flush=True)
    if args.if_stale and is_current(session):
        print("data/signals.json is already complete for this session; nothing to do")
        return

    # ---- core prices: without these there is nothing to publish
    basket_syms = [t for ts in G.AI_BASKET.values() for t in ts]
    px = F.yahoo_close(G.INDEX_SYMBOLS + basket_syms, session)
    missing = [s for s in CORE if s not in px or px[s].dropna().empty]
    if missing:
        log.add("Yahoo indices", False, note=f"missing {missing}")
        print(f"FATAL: Yahoo returned no data for {missing}", file=sys.stderr)
        raise SystemExit(1)
    as_of = px["^GSPC"].dropna().index[-1].date()
    log.add("Yahoo indices", (session - as_of).days <= 4, as_of,
            "" if as_of == session else f"latest bar is {as_of}, expected {session}")
    print(f"prices through {as_of}", flush=True)

    # ---- S&P 500 members for breadth (context only): list from SSGA, prices from Yahoo
    members = pd.DataFrame()
    try:
        tick = F.cached("sp500_members", F.spy_members, log, "SSGA SPY holdings", index_col=None)["ticker"].tolist()
        members = F.yahoo_close(tick, session)
        log.add("Yahoo members", len(members.columns) >= 400, members.index[-1].date(),
                f"{len(members.columns)} of {len(tick)} members")
    except Exception as e:  # noqa: BLE001
        log.add("Yahoo members", False, note=f"breadth unavailable: {type(e).__name__}: {e}"[:200])
    print(f"members {members.shape}", flush=True)

    def soft(name, fn, label):
        try:
            return F.cached(name, fn, log, label)
        except Exception:  # noqa: BLE001 - already logged; the indicator shows as missing
            return pd.DataFrame()

    def slow(name, fn, label, max_age_days):
        """Like soft(), but skips the request while the saved copy is newer than the source's own cadence."""
        path = F.STATE / f"{name}.csv"
        if path.exists():
            saved = pd.read_csv(path, index_col=0, parse_dates=True)
            if not saved.empty and (pd.Timestamp(session) - saved.index.max()).days <= max_age_days:
                log.add(label, True, saved.index.max().date())
                return saved
        return soft(name, fn, label)

    cboe = pd.concat([soft(f"cboe_{s.lower()}", lambda s=s: F.cboe_index(s), f"Cboe {s}") for s in ("COR1M", "VVIX", "SKEW")],
                     axis=1, sort=True)
    pc = F.putcall(session, log)
    baa = soft("fred_baa10y", lambda: F.fred("BAA10Y"), "FRED BAA10Y")
    dgs10 = soft("fred_dgs10", lambda: F.fred("DGS10"), "FRED DGS10")
    z1 = soft("fred_z1", lambda: F.fred_many(L.Z1), "FRED Z.1 accounts")
    sh = soft("shiller", F.shiller, "Shiller CAPE")
    neb = F.nebius(log)
    # AAII's site sits behind a bot filter that sometimes answers a script with a web page instead of
    # the file. Ask only when the saved copy is due an update, and fall back to that copy when refused.
    aaii = slow("aaii_sentiment", F.aaii_sentiment, "AAII sentiment", 6)
    alloc = slow("aaii_allocation", F.aaii_allocation, "AAII allocation", 31)
    hy = soft("fred_hy_oas", lambda: F.fred("BAMLH0A0HYM2"), "FRED high-yield spread")
    flows = soft("fred_fund_flows", lambda: F.fred_many(["BOGZ1FA653064100Q", "BOGZ1FA563064100Q"]), "FRED fund flows")

    # ---- gauges
    extra = {"aaii": aaii, "alloc": alloc, "hy": hy["BAMLH0A0HYM2"] if "BAMLH0A0HYM2" in hy else None,
             "flows": G.fund_flow_pct(flows, z1) if not flows.empty and not z1.empty else None}
    spx_inds, spx = G.spx_indicators(px, cboe, pc, members, baa["BAA10Y"] if "BAA10Y" in baa else None, extra)
    g_spx, h_spx = G.build_gauge("spx", "S&P 500", "S&P 500", spx_inds, spx, G.SPX_PILLARS)
    basket = px[[c for c in basket_syms if c in px]]
    ai_inds, sox = G.ai_indicators(px, basket)
    g_ai, h_ai = G.build_gauge("ai", "AI & semiconductors", "PHLX Semiconductor index (SOX)", ai_inds, sox, G.AI_PILLARS)
    print(f"S&P {g_spx['score']} {g_spx['zone']['label']} · AI {g_ai['score']} {g_ai['zone']['label']}", flush=True)

    # ---- long-term layer
    longterm = None
    if not z1.empty and not sh.empty:
        try:
            longterm = L.build(sh, z1, spx, dgs10["DGS10"] if "DGS10" in dgs10 else None)
            print(f"10y outlook {longterm['expected']}", flush=True)
        except Exception as e:  # noqa: BLE001
            log.add("Long-term model", False, note=f"{type(e).__name__}: {e}"[:200])

    # ---- AI context
    if args.no_companies:
        prev = json.loads((DATA / "signals.json").read_text()).get("ai_context", {}) if (DATA / "signals.json").exists() else {}
        rows, earnings = prev.get("companies", []), [c for c in prev.get("catalysts", []) if c.get("kind") == "earnings"]
    else:
        rows, earnings = C.table(basket, session, log)
    ai_context = {"companies": rows, "catalysts": C.catalysts(earnings, session, F.ROOT / "config" / "calendar.yaml"),
                  "gpu_spot": C.gpu_spot(neb)}

    now = datetime.now(timezone.utc)
    out = {
        "generated_at": now.isoformat(timespec="seconds"), "as_of": as_of.isoformat(),
        "next_refresh_utc": next_refresh(now).isoformat(timespec="seconds"),
        "horizons": list(M.HORIZONS.values()),
        "zones": [{"key": k, "label": lab, "lo": lo, "hi": min(hi, 100)} for k, lab, lo, hi in M.ZONES],
        "gauges": {"spx": g_spx, "ai": g_ai}, "longterm": longterm, "ai_context": ai_context,
        "alerts": A.evaluate({"spx": g_spx, "ai": g_ai}, A.load_config()),
        "repo": os.environ.get("GITHUB_REPOSITORY") or _repo_slug(),
        "sources": log.rows,
    }
    charts = longterm.pop("charts") if longterm else {}
    history = {"as_of": as_of.isoformat(), "spx": h_spx, "ai": h_ai, "longterm": charts}
    DATA.mkdir(exist_ok=True)
    (DATA / "signals.json").write_text(json.dumps(M.clean(out), separators=(",", ":")))
    (DATA / "signals_history.json").write_text(json.dumps(M.clean(history), separators=(",", ":")))
    bad = [r for r in log.rows if not r["ok"]]
    print(f"wrote data/signals.json ({(DATA / 'signals.json').stat().st_size // 1024} KB), "
          f"data/signals_history.json ({(DATA / 'signals_history.json').stat().st_size // 1024} KB)")
    for r in bad:
        print(f"  source issue: {r['name']}: {r['note']}")


if __name__ == "__main__":
    main()
