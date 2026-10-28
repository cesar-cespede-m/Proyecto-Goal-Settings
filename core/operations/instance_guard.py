"""Aviso de otra instancia usando la misma carpeta de datos.

SQLite en modo WAL no admite escrituras desde dos sistemas distintos sobre el mismo archivo
(por ejemplo, el contenedor Docker que monta ``./runtime`` y un Streamlit local en Windows).
Este registro no bloquea nada: sólo deja una marca con la última instancia vista para que la
interfaz pueda advertirlo antes de que aparezca un ``disk I/O error``.
"""

import json
import os
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

MARKER_NAME = ".goal_settings-instance.json"
DEFAULT_WINDOW = timedelta(minutes=2)


class OtherInstance(NamedTuple):
    instance_id: str
    seen_at: datetime


def current_instance_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def record_instance(
    runtime_root: Path,
    *,
    instance_id: str | None = None,
    now: datetime | None = None,
    window: timedelta = DEFAULT_WINDOW,
) -> OtherInstance | None:
    """Registra esta instancia y devuelve otra distinta vista dentro de ``window``.

    Los fallos de lectura o escritura de la marca nunca interrumpen la aplicación.
    """

    instance_id = instance_id or current_instance_id()
    now = now or datetime.now(UTC)
    marker = runtime_root / MARKER_NAME
    other: OtherInstance | None = None
    try:
        previous = json.loads(marker.read_text(encoding="utf-8"))
        seen_at = datetime.fromisoformat(str(previous["seen_at"]))
        if previous["instance_id"] != instance_id and now - seen_at <= window:
            other = OtherInstance(str(previous["instance_id"]), seen_at)
    except (OSError, ValueError, KeyError, TypeError):
        other = None
    try:
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"instance_id": instance_id, "seen_at": now.isoformat()}),
            encoding="utf-8",
        )
        temporary.replace(marker)
    except OSError:
        pass
    return other
