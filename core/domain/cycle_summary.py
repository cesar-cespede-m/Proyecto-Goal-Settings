"""C2-07 (ciclo 2, B3): cómo va el ciclo dato → modelo → decisión → resultado en un proyecto.

Sólo lee lo ya registrado y devuelve conteos; sirve para decidir qué hacer después (medir,
reentrenar, anotar resultados) sin revisar pantalla por pantalla.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core.domain.decisions import DecisionService


class ModelInUse(BaseModel):
    run_id: str
    selected_model: str
    health_status: str
    recommendation: str | None


class CycleSummaryV1(BaseModel):
    schema_version: str = "cycle-summary.v1"
    accepted_files: int
    models_in_use: list[ModelInUse]
    decisions: int
    decisions_with_outcome: int
    event_entities: int
    events_received: int
    next_steps: list[str]


def _published_runs(repository: Any, project_id: str) -> list[Any]:
    return repository.connection.execute(
        """
        SELECT DISTINCT run.id, run.selected_model
        FROM runs AS run
        JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
        JOIN context_versions AS context ON context.id = strategy.context_version_id
        JOIN reports AS report ON report.run_id = run.id
        WHERE context.project_id = ? AND run.status = 'COMPLETED'
          AND report.status IN ('APPROVED', 'PUBLISHED')
        ORDER BY run.finished_at DESC
        """,
        (project_id,),
    ).fetchall()


def cycle_summary(repository: Any, storage_root: str | Path, project_id: str) -> CycleSummaryV1:
    from core.ml_engine.execution import RunBlockedError
    from core.ml_engine.scoring import ScoringService

    scoring = ScoringService(repository, storage_root)
    models: list[ModelInUse] = []
    for row in _published_runs(repository, project_id):
        try:
            scoring.approved_run(str(row["id"]))  # vigente: reporte aprobado y no superado
            health = scoring.health(str(row["id"]))
        except (RunBlockedError, LookupError, ValueError):
            continue  # reporte aprobado pero superado: ya no es un modelo en uso
        models.append(
            ModelInUse(
                run_id=str(row["id"]),
                selected_model=str(row["selected_model"]),
                health_status=str(health.status),
                recommendation=None
                if health.recommendation is None
                else str(health.recommendation),
            )
        )
    decisions = DecisionService(repository).list(project_id)
    with_outcome = sum(1 for item in decisions if item.outcomes)
    events = repository.connection.execute(
        "SELECT COUNT(DISTINCT entity) AS entities, COUNT(*) AS total "
        "FROM observation_events WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    files = len(repository.list_datasets(project_id, admission_status="ACCEPTED"))

    steps: list[str] = []
    if not files and not events["total"]:
        steps.append("Sube un archivo o recibe eventos desde otro sistema para empezar.")
    if files and not models:
        steps.append("Todavía no hay un modelo en uso: prepara la tabla, entrena y aprueba.")
    if any(model.recommendation == "PROPOSE_RETRAINING" for model in models):
        steps.append("Un modelo en uso perdió calidad: conviene reentrenarlo con datos recientes.")
    if decisions and with_outcome < len(decisions):
        steps.append(
            f"Hay {len(decisions) - with_outcome} decisiones sin resultado anotado: "
            "anótalo para saber si ayudaron."
        )
    if events["total"]:
        steps.append("Convierte los eventos recibidos en archivo para medir con datos nuevos.")
    return CycleSummaryV1(
        accepted_files=files,
        models_in_use=models,
        decisions=len(decisions),
        decisions_with_outcome=with_outcome,
        event_entities=int(events["entities"] or 0),
        events_received=int(events["total"] or 0),
        next_steps=steps,
    )
