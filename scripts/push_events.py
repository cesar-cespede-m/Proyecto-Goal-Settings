"""Ingiere un archivo JSONL de ObservationEventV1 en el runtime local."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import get_connection, migrate  # noqa: E402
from core.db.repositories import WorkflowRepository  # noqa: E402
from core.integration.events import EventIngestionService  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = {}
        events.append(value if isinstance(value, dict) else {})
    return events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--source-label", "--source", dest="source_label", required=True)
    parser.add_argument("--actor-label", "--actor", dest="actor_label", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = Path(os.environ.get("GOAL_SETTINGS_RUNTIME_DIR", "runtime"))
    runtime.mkdir(parents=True, exist_ok=True)
    connection = get_connection(runtime / "goal_settings.sqlite3", use_wal=True)
    try:
        migrate(connection)
        result = EventIngestionService(WorkflowRepository(connection), runtime).ingest(
            args.project_id,
            _read_jsonl(args.jsonl),
            source_label=args.source_label,
            actor_label=args.actor_label,
        )
    finally:
        connection.close()
    print(
        json.dumps(
            {
                "accepted": result.accepted,
                "duplicates": result.duplicates,
                "rejected": len(result.rejected),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
