from __future__ import annotations

from datetime import date, timedelta

import pytest
import yaml
from pydantic import ValidationError

from gauges.config import Baskets, load_events

GOOD = {
    "id": "anthropic_rev_2026q3", "kpi": "mo_labs", "entity": "Anthropic", "metric": "quarterly_revenue_usd_bn",
    "value": 16.8, "period": "2026Q3", "as_of": "2026-10-14", "basis": "gross", "status": "secondhand",
    "source_url": "https://www.investing.com/news/x",
}


def write_events(tmp_path, entries):
    path = tmp_path / "events.yaml"
    path.write_text(yaml.safe_dump(entries))
    return load_events(path)


def test_repo_configs_load(cfg, baskets):
    assert len(baskets.tickers) == 25
    assert cfg.fetch.cboe.putcall_json_start == date(2019, 10, 7)
    events, errors = load_events()
    assert errors == [] and len(events) >= 1


def test_spec_first_match_wins(cfg):
    assert cfg.spec_for("yfinance", "^MOVE.close").bounds == (30, 300)
    assert cfg.spec_for("yfinance", "NVDA.close").match == "yfinance:*.close"
    assert cfg.spec_for("alfred", "GACDISA066MSFRBNY").bounds == (-100, 100)
    assert cfg.spec_for("nowhere", "x") is None


def test_every_fetched_fred_series_has_a_spec(cfg):
    for source, series in [("fred", s) for s in cfg.fetch.fred.series] + [("alfred", s) for s in cfg.fetch.alfred.series]:
        assert cfg.spec_for(source, series) is not None, f"{source}:{series}"


def test_basket_weights_must_sum_to_100():
    with pytest.raises(ValidationError, match="sum to 90"):
        Baskets.model_validate({"layers": {"a": {"weight": 90, "tickers": ["X"]}}, "reference_etfs": []})


def test_basket_ticker_in_two_layers_rejected():
    with pytest.raises(ValidationError, match="more than one layer"):
        Baskets.model_validate({"layers": {"a": {"weight": 50, "tickers": ["X"]}, "b": {"weight": 50, "tickers": ["X"]}},
                                "reference_etfs": []})


def test_valid_event_loads_with_status_weight(tmp_path):
    events, errors = write_events(tmp_path, [GOOD])
    assert errors == [] and events[0].weight == 0.6


@pytest.mark.parametrize("change, message", [
    ({"source_url": None}, "source_url"),
    ({"source_url": "<earnings call transcript or release URL>"}, "source_url"),
    ({"status": "rumour"}, "status"),
    ({"as_of": (date.today() + timedelta(days=5)).isoformat()}, "future"),
    ({"value": "maybe"}, "categorical"),
    ({"id": "Has Spaces"}, "id"),
    ({"surprise": 1}, "surprise"),
])
def test_invalid_events_are_rejected(tmp_path, change, message):
    entry = {**GOOD, **change}
    if change.get("source_url", "") is None:
        del entry["source_url"]
    events, errors = write_events(tmp_path, [entry])
    assert events == [] and len(errors) == 1 and message in errors[0]


def test_categorical_guidance_value_accepted(tmp_path):
    events, errors = write_events(tmp_path, [{**GOOD, "id": "msft_capex", "kpi": "sp_guide_next", "value": "raise"}])
    assert errors == [] and events[0].value == "raise"


def test_duplicate_ids_rejected(tmp_path):
    events, errors = write_events(tmp_path, [GOOD, GOOD])
    assert len(events) == 1 and "duplicate" in errors[0]
