"""Ejecutor de migraciones SQL numeradas."""

import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent


def migrate(connection: sqlite3.Connection) -> None:
    """Aplica en orden las migraciones pendientes una sola vez."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    applied = {
        row["version"]
        for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
    }
    for migration_path in sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9][0-9]_*.sql")):
        version = int(migration_path.name.split("_", maxsplit=1)[0])
        if version in applied:
            continue
        sql = migration_path.read_text(encoding="utf-8")
        migration_script = (
            "BEGIN IMMEDIATE;\n"
            f"{sql}\n"
            f"INSERT INTO schema_migrations(version) VALUES ({version});\n"
            "COMMIT;"
        )
        try:
            connection.executescript(migration_script)
        except sqlite3.Error:
            if connection.in_transaction:
                connection.rollback()
            raise
