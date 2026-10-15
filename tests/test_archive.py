"""Archive round trip and the skill digest."""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from gauges.archive import export_all, export_run, restore
from gauges.config import ROOT
from gauges.digest import digest
from gauges.schema import Obs, coerce
from gauges.store import Store


def test_archive_round_trip(tmp_path, fetched_at):
    src = Store(tmp_path / "a.duckdb")
    rows = [Obs("cboe", "VIX", date(2026, 10, 14), 16.87, None, "e"), Obs("cboe", "VIX", date(2026, 10, 13), 17.1, None, "e")]
    src.append_observations(coerce(rows, date(2026, 10, 15))[0], "20261015T000000Z", fetched_at)
    src.append_snapshot("model_catalog", pd.DataFrame({"snapshot_date": [date(2026, 10, 15)], "model_id": ["a/b"]}),
                        "20261015T000000Z", fetched_at)
    src.append_observations(coerce([Obs("cboe", "VIX", date(2026, 10, 15), 18.0, None, "e")], date(2026, 10, 16))[0],
                            "20261016T000000Z", fetched_at)
    root = tmp_path / "archive"
    out = export_all(src, root)
    assert out["20261015T000000Z"] == {"raw_observations": 2, "model_catalog": 1}
    assert export_run(src, "20261015T000000Z", root) == {}            # files are immutable
    with pytest.raises(ValueError):
        export_run(src, "../../etc", root)

    dst = Store(tmp_path / "b.duckdb")
    counts = restore(dst, root)
    assert counts == {"raw_observations": 3, "model_catalog": 1}
    a = src.con.execute("SELECT * FROM raw_observations ORDER BY obs_date").fetchall()
    b = dst.con.execute("SELECT * FROM raw_observations ORDER BY obs_date").fetchall()
    assert a == b
    with pytest.raises(RuntimeError, match="--force"):
        restore(dst, root)
    assert restore(dst, root, force=True)["raw_observations"] == 3
    src.close()
    dst.close()


def test_digest_reads_the_repo_latest_json():
    path = ROOT / "data" / "latest.json"
    if not path.exists():
        pytest.skip("run make render first")
    text = digest(json.loads(path.read_text()))
    assert text.splitlines()[0].startswith("as_of ")
    assert all(f"G{i} " in text for i in range(1, 5)) and "unwind (" in text
