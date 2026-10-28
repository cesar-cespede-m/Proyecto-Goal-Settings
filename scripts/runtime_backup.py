"""Crea o restaura copias offline del runtime privado de Goal Settings."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.operations import create_runtime_backup, restore_runtime_backup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--runtime", type=Path, default=ROOT / "runtime")
    create.add_argument("--destination", type=Path, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--backup", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        result = create_runtime_backup(args.runtime, args.destination)
        print(f"Respaldo creado y manifestado en {result}")
    else:
        result = restore_runtime_backup(args.backup, args.destination)
        print(f"Runtime restaurado y verificado en {result}")


if __name__ == "__main__":
    main()
