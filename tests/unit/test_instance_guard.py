"""UX-01: aviso de otra instancia usando la misma carpeta de datos."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.operations import record_instance

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def test_first_instance_sees_no_other(tmp_path: Path) -> None:
    assert record_instance(tmp_path, instance_id="windows:1", now=NOW) is None


def test_same_instance_is_not_reported(tmp_path: Path) -> None:
    record_instance(tmp_path, instance_id="windows:1", now=NOW)
    later = NOW + timedelta(seconds=30)
    assert record_instance(tmp_path, instance_id="windows:1", now=later) is None


def test_recent_different_instance_is_reported(tmp_path: Path) -> None:
    record_instance(tmp_path, instance_id="docker:7", now=NOW)
    other = record_instance(tmp_path, instance_id="windows:1", now=NOW + timedelta(seconds=30))
    assert other is not None
    assert other.instance_id == "docker:7"
    assert other.seen_at == NOW


def test_stale_different_instance_is_ignored(tmp_path: Path) -> None:
    record_instance(tmp_path, instance_id="docker:7", now=NOW)
    later = NOW + timedelta(minutes=10)
    assert record_instance(tmp_path, instance_id="windows:1", now=later) is None


def test_corrupt_marker_never_breaks_the_app(tmp_path: Path) -> None:
    (tmp_path / ".goal_settings-instance.json").write_text("{no es json", encoding="utf-8")
    assert record_instance(tmp_path, instance_id="windows:1", now=NOW) is None
    assert record_instance(tmp_path, instance_id="docker:7", now=NOW) is not None
