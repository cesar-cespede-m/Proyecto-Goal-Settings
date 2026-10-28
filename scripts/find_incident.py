"""Busca una falla por su código de incidente (p. ej. GSA-261020-7F3A) y muestra su detalle.

Uso:
    python scripts/find_incident.py GSA-261020-7F3A
    python scripts/find_incident.py --last 5
    python scripts/find_incident.py GSA-261020-7F3A --runtime otra/carpeta
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _entries(log: Path) -> list[dict]:
    if not log.is_file():
        return []
    entries = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _show(entry: dict) -> None:
    print(f"{entry['incident_id']} · {entry['created_at']} · {entry['where']}")
    print(f"Código: {entry['code']} ({entry['error_type']})")
    print(f"Detalle: {entry['detail']}")
    print("Traza:")
    print(entry["traceback"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("incident", nargs="?", help="Código de incidente mostrado en pantalla")
    parser.add_argument("--last", type=int, help="Muestra los últimos N incidentes")
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime")
    args = parser.parse_args()

    entries = _entries(args.runtime / "logs" / "incidents.jsonl")
    if args.last:
        for entry in entries[-args.last :]:
            print(
                f"{entry['incident_id']}  {entry['created_at']}  {entry['where']}  {entry['code']}"
            )
        return
    if not args.incident:
        parser.error("indica un código de incidente o --last N")
    wanted = args.incident.strip().upper()
    for entry in reversed(entries):
        if entry.get("incident_id") == wanted:
            _show(entry)
            return
    raise SystemExit(f"No se encontró {wanted} en {args.runtime / 'logs' / 'incidents.jsonl'}")


if __name__ == "__main__":
    main()
