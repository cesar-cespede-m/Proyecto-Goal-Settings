"""Creación centralizada de conexiones SQLite."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def get_connection(database: str | Path, *, use_wal: bool = False) -> sqlite3.Connection:
    """Abre una conexión con integridad referencial activa.

    WAL es opt-in porque solo es seguro para el volumen local de una instancia.
    """

    connection = sqlite3.connect(str(database), isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if use_wal:
        connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Ejecuta una transacción explícita, con rollback ante cualquier error."""

    if connection.in_transaction:
        raise RuntimeError("No se permiten transacciones SQLite anidadas")
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()
