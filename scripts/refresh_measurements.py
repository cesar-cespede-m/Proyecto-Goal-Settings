"""Actualiza archivos y seguimiento desde eventos observados en el runtime local."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import get_connection, migrate  # noqa: E402
from core.db.repositories import WorkflowRepository  # noqa: E402
from core.integration.measurement import MeasurementService  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="ID del proyecto")
    parser.add_argument("--actor", required=True, help="Responsable declarado del refresco")
    parser.add_argument(
        "--classification",
        required=True,
        choices=("synthetic", "sandbox"),
        help="Clasificación permitida para los archivos creados",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = Path(os.environ.get("GOAL_SETTINGS_RUNTIME_DIR", ROOT / "runtime"))
    runtime.mkdir(parents=True, exist_ok=True)
    connection = get_connection(runtime / "goal_settings.sqlite3", use_wal=True)
    try:
        migrate(connection)
        report = MeasurementService(WorkflowRepository(connection), runtime).refresh(
            args.project,
            actor_label=args.actor,
            classification=args.classification,
        )
    finally:
        connection.close()
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
