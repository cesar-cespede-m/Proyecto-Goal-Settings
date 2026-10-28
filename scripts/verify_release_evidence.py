"""Verifica release-evidence.v1 offline y termina con el código contractual 0/10–15."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.integration import (  # noqa: E402
    VerificationCode,
    load_public_key,
    verify_release_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--expected-repo", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    try:
        public_key = load_public_key(args.public_key)
        raw_evidence = args.evidence.read_bytes()
    except (OSError, ValueError) as error:
        print(f"Evidencia no verificable: {error}", file=sys.stderr)
        raise SystemExit(VerificationCode.SCHEMA_INVALID) from error

    code, message = verify_release_evidence(
        raw_evidence,
        public_key,
        expected_repository=args.expected_repo,
        expected_commit=args.expected_commit,
    )
    stream = sys.stdout if code is VerificationCode.VALID else sys.stderr
    print(f"{int(code)}: {message}", file=stream)
    raise SystemExit(int(code))


if __name__ == "__main__":
    main()
