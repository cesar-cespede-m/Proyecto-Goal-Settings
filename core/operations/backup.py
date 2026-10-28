"""Backup verificable del runtime local.

La copia es deliberadamente offline: el operador debe detener Streamlit antes de copiar
datasets y artefactos. SQLite se captura con su API de backup para obtener una base coherente.
"""

import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATABASE_NAME = "goal_settings.sqlite3"
MANIFEST_NAME = "backup-manifest.json"


class BackupError(RuntimeError):
    """El respaldo o la restauración no cumplen las precondiciones seguras."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_new_directory(path: Path, label: str) -> None:
    if path.exists():
        raise BackupError(f"{label} ya existe; no se sobrescribe: {path}")


def _inventory(content: Path) -> list[dict[str, Any]]:
    entries = list(content.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise BackupError("El runtime/respaldo no puede contener enlaces simbólicos")
    return [
        {
            "path": file.relative_to(content).as_posix(),
            "bytes": file.stat().st_size,
            "sha256": _sha256(file),
        }
        for file in sorted(item for item in entries if item.is_file())
    ]


def _verify_inventory(content: Path, files: list[dict[str, Any]]) -> None:
    expected = {str(item["path"]): item for item in files}
    entries = list(content.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise BackupError("El respaldo no puede contener enlaces simbólicos")
    observed = {file.relative_to(content).as_posix(): file for file in entries if file.is_file()}
    if set(expected) != set(observed):
        raise BackupError("El contenido del respaldo no coincide con su manifiesto")
    for relative, path in observed.items():
        item = expected[relative]
        if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise BackupError(f"Falló la integridad del respaldo: {relative}")


def create_runtime_backup(runtime: str | Path, destination: str | Path) -> Path:
    """Crea un backup nuevo sin modificar ni bloquear permanentemente el runtime fuente."""

    source = Path(runtime).resolve()
    target = Path(destination).resolve()
    database = source / DATABASE_NAME
    if not source.is_dir() or not database.is_file():
        raise BackupError(f"Runtime inválido o sin {DATABASE_NAME}: {source}")
    _require_new_directory(target, "El destino del respaldo")
    if source == target or source in target.parents:
        raise BackupError("El respaldo debe quedar fuera del runtime para evitar copias recursivas")

    content = target / "runtime"
    content.mkdir(parents=True)
    try:
        for item in source.rglob("*"):
            if item.is_symlink():
                raise BackupError("El runtime no puede contener enlaces simbólicos")
            relative = item.relative_to(source)
            if relative.as_posix() in {
                DATABASE_NAME,
                f"{DATABASE_NAME}-wal",
                f"{DATABASE_NAME}-shm",
            }:
                continue
            destination_item = content / relative
            if item.is_dir():
                destination_item.mkdir(parents=True, exist_ok=True)
            elif item.is_file():
                destination_item.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination_item)

        source_db = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
        backup_db = sqlite3.connect(content / DATABASE_NAME)
        try:
            source_db.backup(backup_db)
        finally:
            backup_db.close()
            source_db.close()

        files = _inventory(content)
        manifest = {
            "schema_version": "goal_settings-runtime-backup.v1",
            "created_at": datetime.now(UTC).isoformat(),
            "database": DATABASE_NAME,
            "offline_copy_required": True,
            "files": files,
        }
        (target / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target


def restore_runtime_backup(backup: str | Path, destination: str | Path) -> Path:
    """Verifica y restaura hacia un directorio nuevo; nunca pisa un runtime existente."""

    source = Path(backup).resolve()
    target = Path(destination).resolve()
    content = source / "runtime"
    manifest_path = source / MANIFEST_NAME
    _require_new_directory(target, "El runtime de restauración")
    if not content.is_dir() or not manifest_path.is_file():
        raise BackupError("La ruta no contiene un respaldo Goal Settings completo")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "goal_settings-runtime-backup.v1":
            raise BackupError("Versión de manifiesto de respaldo no soportada")
        _verify_inventory(content, manifest["files"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise BackupError("Manifiesto de respaldo inválido") from error
    shutil.copytree(content, target)
    return target
