"""Exporta release-evidence.v1 para un reporte publicado y un commit consumidor exacto."""

import argparse
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import get_connection, migrate  # noqa: E402
from core.db.repositories import WorkflowRepository  # noqa: E402
from core.integration import ReleaseEvidenceService, load_private_key  # noqa: E402

_PENDING_ACTORS = {"", "pendiente", "pending", "todo"}


def _goal_settings_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime")
    parser.add_argument("--report", required=True)
    parser.add_argument("--consumer-repo", required=True)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.actor.strip().casefold() in _PENDING_ACTORS:
        raise SystemExit("--actor requiere el nombre o rol de la persona que exporta.")
    password_value = os.environ.get("GOAL_SETTINGS_SIGNING_KEY_PASSWORD")
    password = password_value.encode("utf-8") if password_value else None
    private_key = load_private_key(args.private_key, password)

    connection = get_connection(args.runtime / "goal_settings.sqlite3", use_wal=True)
    try:
        migrate(connection)
        evidence = ReleaseEvidenceService(WorkflowRepository(connection), args.runtime).export(
            report_id=args.report,
            consumer_repository=args.consumer_repo,
            consumer_commit=args.consumer_commit,
            goal_settings_version=version("proyecto-goal-settings"),
            goal_settings_commit=_goal_settings_commit(),
            actor_label=args.actor.strip(),
            private_key=private_key,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    finally:
        connection.close()
    print(
        f"Evidencia firmada para {evidence.consumer.repository}@"
        f"{evidence.consumer.commit_sha}; vence {evidence.validity.expires_at.isoformat()}"
    )


if __name__ == "__main__":
    main()
