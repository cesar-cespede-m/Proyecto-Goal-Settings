"""Publica evidencia válida en un secreto Actions fuera del commit consumidor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.integration import (  # noqa: E402
    DEFAULT_EVIDENCE_VARIABLE,
    EvidenceTransportError,
    prepare_evidence_variable,
    publish_evidence_variable,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--variable", default=DEFAULT_EVIDENCE_VARIABLE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verifica y prepara el transporte sin modificar GitHub.",
    )
    args = parser.parse_args()

    try:
        payload = prepare_evidence_variable(
            evidence_path=args.evidence,
            public_key_path=args.public_key,
            expected_repository=args.repository,
            expected_commit=args.commit,
            variable_name=args.variable,
        )
        if not args.dry_run:
            publish_evidence_variable(payload)
    except EvidenceTransportError as error:
        print(f"{error.code}: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    action = "preparada" if args.dry_run else "publicada"
    print(
        f"Evidencia {action} fuera del commit para "
        f"{payload.repository}@{payload.commit_sha}; secreto={payload.variable_name}"
    )


if __name__ == "__main__":
    main()
