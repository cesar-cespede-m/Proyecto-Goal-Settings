"""Operaciones locales de continuidad para la demo de una sola instancia."""

from core.operations.backup import BackupError, create_runtime_backup, restore_runtime_backup
from core.operations.instance_guard import OtherInstance, record_instance

__all__ = [
    "BackupError",
    "OtherInstance",
    "create_runtime_backup",
    "record_instance",
    "restore_runtime_backup",
]
