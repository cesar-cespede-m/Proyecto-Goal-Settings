"""V2-28: la línea de tiempo del aprendizaje de un modelo a lo largo de sus reentrenamientos.

Une las versiones del proyecto actual con las de los proyectos de los que nació por
reentrenamiento (y los que nacieron de él), en orden: cuándo se entrenó cada una, con qué
medida le fue, si se aprobó para usar y cuántas veces se usó. Sólo lee lo registrado.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from core.domain.retraining_service import RetrainingService


class TimelineEntryV1(BaseModel):
    project_id: str
    project_name: str
    run_id: str
    started_at: str
    selected_model: str | None
    metric: str | None
    test_value: float | None
    approved: bool
    uses: int = Field(ge=0)
    origin: str  # «primera versión» o «reentrenada desde …»


class LearningTimelineV1(BaseModel):
    entries: list[TimelineEntryV1]
    summary: str


def _project_chain(repository: Any, storage_root: Any, project_id: str) -> list[str]:
    service = RetrainingService(repository, storage_root)
    links = list(repository.list_retraining_links())
    chain = [project_id]
    current = project_id
    while (source := service.source_of(current)) is not None:  # hacia atrás
        previous = repository.get_run_project_id(source["source_run_id"])
        if previous in chain:
            break
        chain.insert(0, previous)
        current = previous
    frontier = [project_id]
    while frontier:  # hacia adelante
        current = frontier.pop()
        for link in links:
            if (
                repository.get_run_project_id(link["source_run_id"]) == current
                and link["project_id"] not in chain
            ):
                chain.append(link["project_id"])
                frontier.append(link["project_id"])
    return chain


def _test_value(evaluation: dict[str, Any]) -> tuple[str | None, float | None]:
    if evaluation.get("schema_version") == "task-evaluation.v1":
        metric = evaluation.get("primary_metric")
        return metric, (evaluation.get("test") or {}).get(metric)
    holdout = evaluation.get("holdout") or {}
    metric = (evaluation.get("selection") or {}).get("primary_metric")
    return metric, holdout.get(metric) if metric else None


def learning_timeline(repository: Any, storage_root: Any, project_id: str) -> LearningTimelineV1:
    names = {str(row["id"]): str(row["name"]) for row in repository.list_active_projects()}
    entries: list[TimelineEntryV1] = []
    chain = _project_chain(repository, storage_root, project_id)
    for index, member in enumerate(chain):
        for row in repository.list_project_runs(member):
            metric, value = _test_value(json.loads(row["candidate_metrics_json"]))
            entries.append(
                TimelineEntryV1(
                    project_id=member,
                    project_name=names.get(member, "Proyecto archivado"),
                    run_id=str(row["run_id"]),
                    started_at=str(row["started_at"]),
                    selected_model=row["selected_model"],
                    metric=metric,
                    test_value=value,
                    approved=row["report_status"] in {"APPROVED", "PUBLISHED"},
                    uses=len(repository.list_scorings(str(row["run_id"]))),
                    origin="primera versión"
                    if index == 0
                    else f"reentrenada desde «{names.get(chain[index - 1], 'otro proyecto')}»",
                )
            )
    entries.sort(key=lambda item: item.started_at)
    if not entries:
        summary = "Todavía no hay versiones entrenadas."
    elif len(chain) == 1:
        summary = f"{len(entries)} versión(es) entrenadas en este proyecto, sin reentrenamientos."
    else:
        summary = (
            f"{len(entries)} versiones en {len(chain)} etapas de aprendizaje: cada reentrenamiento "
            "nació del modelo anterior con datos más recientes."
        )
    return LearningTimelineV1(entries=entries, summary=summary)
