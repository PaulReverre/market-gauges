"""Compact plain-text read of data/latest.json, for the market-gauges skill.

Usage: python -m gauges.digest [--path data/latest.json]

Every number printed here is copied from latest.json, so a summary written from
this output cannot introduce figures that the pipeline did not produce.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gauges.config import ROOT


def _f(v, nd=1) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def digest(d: dict) -> str:
    lines = [f"as_of {d['as_of']} | built {d['generated_at']} | next refresh {d['next_refresh_utc']} | "
             f"global coverage {_f(d['global_coverage'])}%"]
    for gid, g in d["gauges"].items():
        names = {c["id"]: c["name"] for c in g["components"]}
        drivers = "; ".join(f"{names.get(x['id'], x['id'])} {x['change']:+.1f}" for x in g["drivers_1w"])
        lines.append(f"{gid} {g['name']}: {_f(g['display'])} {g['band']} | 1W {_f(g['change_1w'])} 1M {_f(g['change_1m'])} "
                     f"| coverage {_f(g['coverage'])}%{' LOW' if g['low_coverage'] else ''} | 1W drivers: {drivers or 'none'}")
    f = d["fragility"]
    prob = f["drawdown_probability"] if isinstance(f["drawdown_probability"], str) else \
        f"{f['drawdown_probability']['probability_pct']}% (decile {f['drawdown_probability']['decile']})"
    lines.append(f"fragility {_f(f['score'])} | 10%+ drawdown in 63d: {prob} | breadth divergence {f['breadth_divergence']}")
    lines.append(f"cnn_replica {_f(d['cnn_replica'])} | banners: {', '.join(d['banners']) or 'none'}")
    flags = [k for k, v in d["ai_fragility"].items() if v]
    unknown = [k for k, v in d["ai_fragility"].items() if v is None]
    lines.append(f"ai_fragility on: {', '.join(flags) or 'none'} | not enough data: {', '.join(unknown) or 'none'}")
    lights = "; ".join(f"{k} {v['state']}" for k, v in d["unwind"]["lights"].items())
    lines.append(f"unwind ({d['unwind']['red_count']} red{', ALERT' if d['unwind']['alert'] else ''}): {lights}")
    issues = "; ".join(f"{s['source']} {s['status']}" for s in d["source_issues"]) or "none"
    stale = ", ".join(s["id"] for s in d["stale_indicators"]) or "none"
    lines.append(f"source issues: {issues} | stale inputs: {stale} | missing: {len(d['gaps']['missing'])} indicators")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=str(ROOT / "data" / "latest.json"))
    args = parser.parse_args()
    print(digest(json.loads(Path(args.path).read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
