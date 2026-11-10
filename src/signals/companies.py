"""AI context that sits behind the AI gauge: the company table, upcoming catalysts, GPU spot prices.

None of this feeds a score. Price columns come from the same Yahoo download as
everything else; earnings columns need a free Finnhub key and are simply left
blank without one.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from signals import fetch as F
from signals.gauges import AI_BASKET


def _ret(s: pd.Series, n: int) -> float | None:
    s = s.dropna()
    return None if len(s) <= n else round(float(s.iloc[-1] / s.iloc[-1 - n] - 1) * 100, 1)


def table(basket: pd.DataFrame, session: date, log: F.SourceLog) -> tuple[list[dict], list[dict]]:
    rows, earnings = [], []
    have_key = True
    errors = 0
    for layer, tickers in AI_BASKET.items():
        for t in tickers:
            if t not in basket:
                continue
            s = basket[t].dropna()
            if len(s) < 30:
                continue
            ytd = s[s.index.year == s.index[-1].year]
            prev = s[s.index.year < s.index[-1].year]
            row = {"ticker": t, "layer": layer, "ret_1w": _ret(s, 5), "ret_1m": _ret(s, 21), "ret_3m": _ret(s, 63),
                   "ret_ytd": None if prev.empty or ytd.empty else round(float(ytd.iloc[-1] / prev.iloc[-1] - 1) * 100, 1),
                   "from_high": round(float(s.iloc[-1] / s.iloc[-252:].max() - 1) * 100, 1),
                   "eps_surprise": None, "rev_growth": None, "next_report": None}
            if have_key and "." not in t:       # Finnhub's free tier covers US listings only
                try:
                    surprises = F.finnhub("stock/earnings", symbol=t)
                    if surprises and surprises[0].get("surprisePercent") is not None:
                        row["eps_surprise"] = round(float(surprises[0]["surprisePercent"]), 1)
                    metric = F.finnhub("stock/metric", symbol=t, metric="all").get("metric", {})
                    if metric.get("revenueGrowthTTMYoy") is not None:
                        row["rev_growth"] = round(float(metric["revenueGrowthTTMYoy"]), 1)
                    cal = F.finnhub("calendar/earnings", symbol=t, **{"from": session.isoformat(),
                                    "to": (session + timedelta(days=60)).isoformat()}).get("earningsCalendar", [])
                    dates = sorted(e["date"] for e in cal if e.get("symbol") == t and e.get("date"))
                    if dates:
                        row["next_report"] = dates[0]
                        earnings.append({"date": dates[0], "label": f"{t} earnings", "kind": "earnings"})
                    time.sleep(3.2)             # 3 calls per ticker, 60 calls a minute allowed
                except RuntimeError as e:
                    if "FINNHUB_API_KEY" in str(e):
                        have_key = False
                    else:
                        errors += 1
                except Exception:  # noqa: BLE001
                    errors += 1
            rows.append(row)
    if not have_key:
        log.add("Finnhub", False, note="FINNHUB_API_KEY not set: earnings columns left blank")
    else:
        log.add("Finnhub", errors == 0, session, f"{errors} ticker(s) failed" if errors else "")
    return rows, earnings


def catalysts(earnings: list[dict], session: date, calendar_path: Path, days: int = 30) -> list[dict]:
    out = [e for e in earnings if session.isoformat() <= e["date"] <= (session + timedelta(days=days)).isoformat()]
    if calendar_path.exists():
        for e in yaml.safe_load(calendar_path.read_text()) or []:
            d = e["date"] if isinstance(e["date"], date) else date.fromisoformat(str(e["date"]))
            if session <= d <= session + timedelta(days=days):
                out.append({"date": d.isoformat(), "label": e["event"], "kind": e.get("kind", "macro")})
    return sorted(out, key=lambda e: (e["date"], e["label"]))


def gpu_spot(nebius: pd.DataFrame) -> dict | None:
    """Chain-linked equal-weight index of Nebius GPU spot prices (100 = first day)."""
    if nebius is None or nebius.empty or len(nebius) < 3:
        return None
    step = np.log(nebius.where(nebius > 0)).diff().mean(axis=1, skipna=True)
    idx = 100 * np.exp(step.fillna(0).cumsum())
    return {"as_of": idx.index[-1].date().isoformat(), "since": idx.index[0].date().isoformat(),
            "value": round(float(idx.iloc[-1]), 1),
            "chg_1w": round(float(idx.iloc[-1] / idx.iloc[-8] - 1) * 100, 1) if len(idx) > 8 else None,
            "chg_all": round(float(idx.iloc[-1] / idx.iloc[0] - 1) * 100, 1),
            "series": len(nebius.columns), "days": int(len(idx)),
            "dates": [d.date().isoformat() for d in idx.index], "index": [round(float(v), 1) for v in idx.values]}
