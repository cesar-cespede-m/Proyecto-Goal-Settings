"""Ciclo de aprendizaje en un solo paso, para correrlo a mano o programado (V2-25).

    python scripts/learning_cycle.py --actor ciclo-semanal --classification synthetic

Para cada proyecto activo: convierte en archivo los eventos nuevos y mide los modelos en uso
(sin contarlo como uso), y luego revisa la salud de cada modelo vigente. No entrena, no aprueba y
no publica: imprime en JSON qué necesita una persona (por ejemplo, «reentrenar») y la Guía de la
app lo muestra como siguiente paso.

Programarlo en Windows (cada lunes a las 7:00):
    schtasks /Create /SC WEEKLY /D MON /ST 07:00 /TN Goal Settings-ciclo ^
      /TR "<ruta>\\.venv\\Scripts\\python.exe <ruta>\\scripts\\learning_cycle.py --actor ciclo"
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
from core.domain.cycle_summary import cycle_summary  # noqa: E402


def run_cycle(
    repository: WorkflowRepository, runtime: Path, *, actor: str, classification: str
) -> list[dict]:
    from core.integration.measurement import MeasurementService

    results = []
    for project in repository.list_active_projects():
        project_id = str(project["id"])
        has_events = repository.connection.execute(
            "SELECT 1 FROM observation_events WHERE project_id = ? LIMIT 1", (project_id,)
        ).fetchone()
        measured = None
        if has_events:
            report = MeasurementService(repository, runtime).refresh(
                project_id, actor_label=actor, classification=classification
            )
            measured = {
                "archivos_creados": len(report.created_files),
                "modelos_medidos": len(report.measured_models),
                "avisos": report.warnings,
            }
        summary = cycle_summary(repository, runtime, project_id)
        needs_person = [
            f"Reentrenar «{model.selected_model}»: perdió calidad"
            for model in summary.models_in_use
            if model.recommendation == "PROPOSE_RETRAINING"
        ]
        results.append(
            {
                "proyecto": str(project["name"]),
                "medicion": measured,
                "modelos_en_uso": len(summary.models_in_use),
                "necesita_una_persona": needs_person,
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--actor", required=True, help="Responsable declarado del ciclo")
    parser.add_argument("--classification", choices=("synthetic", "sandbox"), default="synthetic")
    args = parser.parse_args(argv)
    runtime = Path(os.environ.get("GOAL_SETTINGS_RUNTIME_DIR", ROOT / "runtime"))
    connection = get_connection(runtime / "goal_settings.sqlite3")
    migrate(connection)
    try:
        output = run_cycle(
            WorkflowRepository(connection),
            runtime,
            actor=args.actor,
            classification=args.classification,
        )
    finally:
        connection.close()
    print(json.dumps(output, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
