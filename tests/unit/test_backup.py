"""Seguridad básica del respaldo offline de runtime."""

from pathlib import Path

import pytest

from core.db import get_connection, migrate
from core.operations import BackupError, create_runtime_backup, restore_runtime_backup


def _runtime(tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    connection = get_connection(runtime / "goal_settings.sqlite3")
    migrate(connection)
    connection.close()
    (runtime / "reports").mkdir()
    (runtime / "reports" / "fixture.txt").write_text("fixture sintético", encoding="utf-8")
    return runtime


def test_backup_and_restore_never_overwrite_existing_destinations(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    existing_backup = tmp_path / "existing-backup"
    existing_backup.mkdir()
    with pytest.raises(BackupError, match="no se sobrescribe"):
        create_runtime_backup(runtime, existing_backup)

    backup = create_runtime_backup(runtime, tmp_path / "backup")
    existing_runtime = tmp_path / "existing-runtime"
    existing_runtime.mkdir()
    with pytest.raises(BackupError, match="no se sobrescribe"):
        restore_runtime_backup(backup, existing_runtime)


def test_restore_rejects_tampered_backup(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    backup = create_runtime_backup(runtime, tmp_path / "backup")
    (backup / "runtime/reports/fixture.txt").write_text("alterado", encoding="utf-8")

    with pytest.raises(BackupError, match="integridad"):
        restore_runtime_backup(backup, tmp_path / "restored")
