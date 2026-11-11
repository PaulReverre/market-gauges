"""Threshold alerts.

`evaluate` runs inside the build and writes the "alerts" block of signals.json.
`python -m signals.alerts --send` runs after the build in the scheduled job: it
compares each gauge's alert state with the one it last told you about
(data/state/alerts.json) and, when that changed, opens or closes a GitHub issue
assigned to the repository owner, which GitHub emails. Without `--send` it only
prints what it would do.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "alerts.yaml"
STATE = ROOT / "data" / "state" / "alerts.json"
SIGNALS = ROOT / "data" / "signals.json"
LABEL = "gauge-alert"
DEFAULTS = {"below": 10, "above": 90}
KIND = {"below": "Buy alert", "above": "Caution alert"}


def load_config(path: Path = CONFIG) -> dict:
    cfg = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    out = {"dashboard_url": cfg.get("dashboard_url", "")}
    for gid in ("spx", "ai"):
        t = {**DEFAULTS, **(cfg.get(gid) or {})}
        lo, hi = float(t["below"]), float(t["above"])
        if not 0 <= lo < hi <= 100:
            raise ValueError(f"config/alerts.yaml: {gid} needs 0 <= below < above <= 100, got {lo} and {hi}")
        out[gid] = {"below": lo, "above": hi}
    return out


def state_of(score: float, t: dict) -> str:
    return "below" if score <= t["below"] else "above" if score >= t["above"] else "none"


def evaluate(gauges: dict, cfg: dict) -> dict:
    """The alerts block for signals.json: thresholds, and which gauges are inside an alert range now."""
    active = []
    for gid, g in gauges.items():
        t = cfg[gid]
        st = state_of(g["score"], t)
        if st == "none":
            continue
        days = 0
        for v in reversed(g.get("spark") or [g["score"]]):
            if state_of(v, t) != st:
                break
            days += 1
        active.append({"gauge": gid, "name": g["name"], "kind": st, "score": g["score"], "threshold": t[st], "days": days})
    return {"thresholds": {gid: cfg[gid] for gid in gauges}, "active": active}


def transitions(prev: dict, gauges: dict, cfg: dict) -> list[dict]:
    """What changed since the last notification: one event per gauge whose alert state moved."""
    out = []
    for gid, g in gauges.items():
        now, was = state_of(g["score"], cfg[gid]), prev.get(gid, "none")
        if now != was:
            out.append({"gauge": gid, "was": was, "now": now})
    return out


def issue_text(g: dict, kind: str, threshold: float, url: str) -> tuple[str, str]:
    word = "at or below" if kind == "below" else "at or above"
    title = f"{KIND[kind]}: {g['name']} gauge at {g['score']:.0f} ({word} {threshold:g})"
    z, base = g["now"]["zone"], g["now"]["base"]
    lines = [f"**{g['name']}: {g['score']:.1f} / 100, {g['zone']['label']}**, at the close of {g['as_of']}. "
             f"Your alert fires {word} {threshold:g}.", ""]
    if z["h"].get("3m") and base.get("3m"):
        a, b = z["h"]["3m"], base["3m"]
        lines += [f"After past readings in this zone ({z['episodes']} episodes), the benchmark was higher 3 months "
                  f"later {a['hit']}% of the time, median {a['median']:+.1f}%. On any day: {b['hit']}%, "
                  f"{b['median']:+.1f}%. These are past frequencies, not a forecast.", ""]
    if url:
        lines += [f"[Open the dashboard]({url}#{g['id']})", ""]
    lines += ["Thresholds are set in `config/alerts.yaml`. This issue closes itself when the score leaves the range."]
    return title, "\n".join(lines)


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def send(events: list[dict], gauges: dict, cfg: dict, assignee: str | None) -> None:
    _gh("label", "create", LABEL, "--color", "5319e7", "--description", "Market gauge threshold alert", "--force")
    for e in events:
        g = gauges[e["gauge"]]
        if e["was"] != "none":                           # close whatever is open for this gauge
            found = json.loads(_gh("issue", "list", "--label", LABEL, "--state", "open", "--json", "number,title", "--limit", "50"))
            for it in found:
                if f": {g['name']} gauge" in it["title"]:
                    _gh("issue", "close", str(it["number"]), "--comment",
                        f"Cleared: the score is {g['score']:.1f} at the close of {g['as_of']}.")
        if e["now"] != "none":
            title, body = issue_text(g, e["now"], cfg[e["gauge"]][e["now"]], cfg["dashboard_url"])
            cmd = ["issue", "create", "--title", title, "--body", body, "--label", LABEL]
            _gh(*cmd, *(["--assignee", assignee] if assignee else []))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--send", action="store_true", help="open and close GitHub issues (needs the gh CLI and GH_TOKEN)")
    ap.add_argument("--test", action="store_true", help="also open one clearly labelled test issue, to check delivery")
    args = ap.parse_args()
    cfg = load_config()
    gauges = json.loads(SIGNALS.read_text())["gauges"]
    prev = json.loads(STATE.read_text()) if STATE.exists() else {}
    events = transitions(prev, gauges, cfg)
    for e in events:
        print(f"{gauges[e['gauge']]['name']}: {e['was']} -> {e['now']} (score {gauges[e['gauge']]['score']})")
    if not events:
        print("no alert changes")
    assignee = os.environ.get("ALERT_ASSIGNEE")
    if args.send:
        if events:
            send(events, gauges, cfg, assignee)
        if args.test:
            g = gauges["spx"]
            _gh("label", "create", LABEL, "--color", "5319e7", "--description", "Market gauge threshold alert", "--force")
            _gh("issue", "create", "--title", "Test alert: market gauges can reach you", "--label", LABEL,
                "--body", f"This is a delivery test, not a signal. {g['name']} is at {g['score']:.1f}. "
                          f"Real alerts fire at or below {cfg['spx']['below']:g} or at or above {cfg['spx']['above']:g}. "
                          "You can close this issue.", *(["--assignee", assignee] if assignee else []))
            print("test issue opened")
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({gid: state_of(g["score"], cfg[gid]) for gid, g in gauges.items()}, indent=1) + "\n")


if __name__ == "__main__":
    main()
