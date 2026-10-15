from __future__ import annotations

from gauges.ingest import scheduled
from gauges.sources import SOURCES


def test_cadence_and_ci_skip_name_known_sources(cfg):
    named = {s for group in cfg.cadence.values() for s in group} | set(cfg.ci_skip)
    assert named <= set(SOURCES), named - set(SOURCES)


def test_local_run_pulls_every_source(cfg):
    assert scheduled(cfg, None) == list(SOURCES)


def test_scheduled_runs_leave_out_ci_skip(cfg):
    assert "tsmc" in cfg.ci_skip
    monthly = scheduled(cfg, "monthly")
    assert "tsmc" not in monthly and "mops" in monthly
    assert all("tsmc" not in scheduled(cfg, c) for c in ("daily", "weekly", "all"))
    assert set(scheduled(cfg, "all")) == set(SOURCES) - set(cfg.ci_skip)


def test_ci_skip_applies_to_the_daily_group(cfg):
    cfg = cfg.model_copy(update={"ci_skip": ["gdelt"]})
    daily = scheduled(cfg, "daily")
    assert "gdelt" not in daily and "fred" in daily
    assert not set(daily) & {s for group in cfg.cadence.values() for s in group}
