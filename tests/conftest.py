from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pytest
from openpyxl import Workbook

from gauges.config import load_baskets, load_series_config
from gauges.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture(scope="session")
def cfg():
    return load_series_config()


@pytest.fixture(scope="session")
def baskets():
    return load_baskets()


@pytest.fixture
def fetched_at():
    return datetime(2026, 10, 15, 23, 0, 0)


def xlsx(sheets: dict[str, list[list]]) -> bytes:
    """Build an in-memory workbook: {sheet name: rows}."""
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
