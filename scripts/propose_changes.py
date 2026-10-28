"""Genera borradores locales de cambios C2-05; no llama a Git ni a GitHub."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ai_engine import LLMSettings, build_provider  # noqa: E402
from core.ai_engine.change_proposals import (  # noqa: E402
    ChangeProposalService,
    collect_evidence,
)
from core.db.repositories import WorkflowRepository  # noqa: E402
from core.integration.change_drafts import write_drafts  # noqa: E402


def _open_repository(runtime: Path) -> tuple[sqlite3.Connection, WorkflowRepository]:
    database = (runtime / "goal_settings.sqlite3").resolve()
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    repository = WorkflowRepository(connection)
    return connection, repository


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, metavar="ID")
    parser.add_argument("--out", required=True, type=Path, metavar="DIR")
    parser.add_argument(
        "--json", action="store_true", help="Emite un resumen JSON para automatización"
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path(os.environ.get("GOAL_SETTINGS_RUNTIME_DIR", ROOT / "runtime")),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    connection, repository = _open_repository(args.runtime)
    try:
        evidence = collect_evidence(repository, args.project, storage_root=args.runtime)
        env_file = Path(os.environ.get("GOAL_SETTINGS_ENV_FILE", ROOT / ".env"))
        settings = LLMSettings.from_env(env_file=env_file)
        answer = ChangeProposalService(build_provider(settings)).propose(evidence)
        paths = write_drafts(answer.proposals, args.out)
    finally:
        connection.close()

    if args.json:
        print(
            json.dumps(
                {
                    "evidence": len(evidence),
                    "proposals": len(answer.proposals),
                    "drafts": [str(path) for path in paths],
                },
                ensure_ascii=False,
            )
        )
    else:
        for path in paths:
            print(path)
        print(f"evidence={len(evidence)} proposals={len(answer.proposals)} drafts={len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
