"""UX-01: un fallo de SQLite se explica sin traza y ofrece reintentar."""

import os
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import core.db
from core.operations import record_instance

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"


@pytest.fixture
def runtime_dir(tmp_path: Path) -> Iterator[Path]:
    previous = os.environ.get("GOAL_SETTINGS_RUNTIME_DIR")
    os.environ["GOAL_SETTINGS_RUNTIME_DIR"] = str(tmp_path)
    try:
        yield tmp_path
    finally:
        if previous is None:
            os.environ.pop("GOAL_SETTINGS_RUNTIME_DIR", None)
        else:
            os.environ["GOAL_SETTINGS_RUNTIME_DIR"] = previous


def test_disk_io_error_shows_plain_message_and_retry(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_connection(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(core.db, "get_connection", broken_connection)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    errors = " ".join(item.value for item in app.error)
    assert "No pudimos leer o guardar tus datos" in errors
    assert "No sabemos si tu última acción quedó guardada" in errors
    assert any(button.label == "Reintentar" for button in app.button)


def test_recent_other_instance_is_warned(runtime_dir: Path) -> None:
    record_instance(runtime_dir, instance_id="docker-container:1", now=datetime.now(UTC))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert any("otra copia de Goal Settings" in item.value for item in app.warning)
