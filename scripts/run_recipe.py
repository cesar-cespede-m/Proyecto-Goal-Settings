"""Reentrena un modelo con su receta desde la terminal (ciclo 2, C2-09).

Para que otro sistema prepare el reentrenamiento con archivos nuevos sin abrir la interfaz:

    python scripts/run_recipe.py --run RUN_ID --file casos=nuevos_casos.csv \\
        --file grupos=nuevos_grupos.csv --actor sistema-x --authorized

Crea el proyecto de reentrenamiento, rehace la tabla con los mismos pasos y copia las
decisiones. NO entrena ni publica: la estrategia, el entrenamiento y la publicación siguen
esperando la aprobación de una persona en la app. Sin receta (modelo entrenado con un archivo
directo) usa `--file archivo=...` con un único archivo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import get_connection, migrate  # noqa: E402
from core.db.repositories import WorkflowRepository  # noqa: E402
from core.domain.retraining_service import RetrainingService  # noqa: E402


def _files(pairs: list[str]) -> dict[str, tuple[str, bytes]]:
    files = {}
    for pair in pairs:
        name, _, path = pair.partition("=")
        if not name or not path:
            raise SystemExit(f"Formato inválido: {pair!r}; usa nombre=ruta.csv")
        files[name] = (Path(path).name, Path(path).read_bytes())
    return files


def run(repository: WorkflowRepository, runtime: Path, args: argparse.Namespace) -> dict:
    service = RetrainingService(repository, runtime)
    files = _files(args.file)
    needed = service.recipe_sources(args.run)
    if needed:
        started = service.start_from_recipe(
            source_run_id=args.run,
            files=files,
            actor_label=args.actor,
            data_authorized=args.authorized,
        )
    else:
        if len(files) != 1:
            raise SystemExit("Este modelo no tiene receta: pasa exactamente un archivo")
        filename, content = next(iter(files.values()))
        started = service.start(
            source_run_id=args.run,
            filename=filename,
            content=content,
            actor_label=args.actor,
            data_authorized=args.authorized,
        )
    return {
        "project_id": started.project_id,
        "dataset_id": started.dataset_id,
        "context_version_id": started.context_version_id,
        "next": "Revisar la estrategia y aprobar el entrenamiento en la app.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--file", action="append", default=[], help="nombre=ruta.csv")
    parser.add_argument("--actor", required=True)
    parser.add_argument(
        "--authorized",
        action="store_true",
        help="Confirma que los archivos son sintéticos, de sandbox o anonimizados y autorizados",
    )
    args = parser.parse_args(argv)
    runtime = Path(os.environ.get("GOAL_SETTINGS_RUNTIME_DIR", ROOT / "runtime"))
    connection = get_connection(runtime / "goal_settings.sqlite3")
    migrate(connection)
    try:
        output = run(WorkflowRepository(connection), runtime, args)
    finally:
        connection.close()
    print(json.dumps(output, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
