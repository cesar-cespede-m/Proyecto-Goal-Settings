"""V2-11: revisión previa a cambiar la tabla usada para entrenar.

Este módulo sólo interpreta metadatos y linaje persistidos. No cambia la tabla activa ni
invalida artefactos: la pantalla decide si continúa y ``set_modeling_table`` conserva su
comportamiento existente.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.db.repositories import WorkflowRepository

DatasetOrigin = Literal["source", "computed", "composition"]
ReviewLevel = Literal["ok", "warning", "block"]
InvalidatedArtifact = Literal["context", "plan", "approval", "report"]

_SUMMARY_EXPLANATION = (
    "Esto parece un resumen por grupos, no una fila por caso: un modelo necesita casos."
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class TableDimensionsV1(_StrictModel):
    dataset_id: str = Field(min_length=1)
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)


class TableChangeReviewV1(_StrictModel):
    schema_version: Literal["table-change-review.v1"] = "table-change-review.v1"
    active: TableDimensionsV1 | None
    candidate: TableDimensionsV1
    looks_like_summary: bool
    summary_explanation: str | None = None
    missing_result_column: str | None = None
    invalidated: list[InvalidatedArtifact] = Field(default_factory=list)
    invalidation_messages: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    level: ReviewLevel

    @model_validator(mode="after")
    def details_match_flags(self) -> TableChangeReviewV1:
        if self.looks_like_summary != (self.summary_explanation is not None):
            raise ValueError("La explicación de resumen debe coincidir con la detección")
        if len(self.invalidated) != len(self.invalidation_messages):
            raise ValueError("Cada elemento sin vigencia debe tener una explicación")
        if len(self.invalidated) != len(set(self.invalidated)):
            raise ValueError("Un elemento sin vigencia no puede repetirse")
        return self

    @property
    def active_rows(self) -> int | None:
        return None if self.active is None else self.active.rows

    @property
    def active_columns(self) -> int | None:
        return None if self.active is None else self.active.columns

    @property
    def candidate_rows(self) -> int:
        return self.candidate.rows

    @property
    def candidate_columns(self) -> int:
        return self.candidate.columns


def dataset_origin(repository: WorkflowRepository, dataset_id: str) -> DatasetOrigin:
    """Distingue una fuente, una tabla calculada y un cruce por su linaje real."""

    if repository.get_analysis_table(dataset_id) is not None:
        return "computed"
    if repository.get_dataset_composition(dataset_id) is not None:
        return "composition"
    return "source"


def _dimensions(dataset) -> TableDimensionsV1:
    profile = json.loads(dataset["profile_json"])
    rows = dataset["n_rows"]
    columns = dataset["n_cols"]
    if rows is None:
        rows = profile.get("row_count", 0)
    if columns is None:
        columns = profile.get("column_count")
    if columns is None:
        raw_columns = profile.get("columns", [])
        columns = len(raw_columns) if isinstance(raw_columns, list) else 0
    return TableDimensionsV1(
        dataset_id=str(dataset["id"]),
        rows=int(rows),
        columns=int(columns),
    )


def _column_names(dataset) -> set[str]:
    profile = json.loads(dataset["profile_json"])
    raw_columns = profile.get("columns", [])
    if not isinstance(raw_columns, list):
        return set()
    return {
        str(column["name"])
        for column in raw_columns
        if isinstance(column, dict) and column.get("name") is not None
    }


def _has_grouped_recipe(repository: WorkflowRepository, dataset_id: str) -> bool:
    table = repository.get_analysis_table(dataset_id)
    if table is None:
        return False
    plan = json.loads(table["plan_json"])
    steps = plan.get("steps", [])
    return isinstance(steps, list) and any(
        isinstance(step, dict) and step.get("op") in {"group_by", "pivot"} for step in steps
    )


def _invalidation_details(
    repository: WorkflowRepository, project_id: str, *, changes_dataset: bool
) -> tuple[list[InvalidatedArtifact], list[str], object | None]:
    if not changes_dataset:
        return [], [], repository.get_latest_context(project_id)

    context = repository.get_latest_context(project_id)
    if context is None:
        return [], [], None

    invalidated: list[InvalidatedArtifact] = ["context"]
    messages = [
        "Las decisiones actuales sobre el resultado y las columnas dejarán de estar vigentes."
    ]
    strategy = repository.get_latest_strategy(context["id"])
    if strategy is None:
        return invalidated, messages, context

    invalidated.append("plan")
    messages.append("El plan de entrenamiento actual dejará de estar vigente.")
    training_decision = repository.get_latest_training_decision(strategy["id"])
    run = repository.get_latest_run(strategy["id"])
    report = repository.get_latest_report(run["id"]) if run is not None else None
    publication_decision = (
        repository.get_latest_publication_decision(report["id"]) if report is not None else None
    )
    approved = any(
        decision is not None and decision["decision"] == "APPROVED"
        for decision in (training_decision, publication_decision)
    )
    if approved:
        invalidated.append("approval")
        messages.append("Las aprobaciones ligadas a la tabla actual dejarán de estar vigentes.")
    if report is not None:
        invalidated.append("report")
        messages.append("El reporte del entrenamiento actual dejará de estar vigente.")
    return invalidated, messages, context


def compare_tables(
    repository: WorkflowRepository, project_id: str, candidate_dataset_id: str
) -> TableChangeReviewV1:
    """Compara sin mutar la tabla activa y una candidata antes de entrenar."""

    candidate = repository.get_dataset(candidate_dataset_id)
    if candidate["project_id"] != project_id:
        raise LookupError("La tabla candidata no pertenece a este proyecto")
    if candidate["admission_status"] != "ACCEPTED":
        raise ValueError("Sólo una tabla aceptada puede revisarse para entrenar")

    active = repository.get_active_dataset(project_id)
    active_dimensions = None if active is None else _dimensions(active)
    candidate_dimensions = _dimensions(candidate)
    changes_dataset = active is not None and active["id"] != candidate_dataset_id

    grouped_recipe = _has_grouped_recipe(repository, candidate_dataset_id)
    much_smaller = (
        active_dimensions is not None and candidate_dimensions.rows * 2 < active_dimensions.rows
    )
    looks_like_summary = candidate_dimensions.rows < 30 or much_smaller or grouped_recipe
    summary_explanation = _SUMMARY_EXPLANATION if looks_like_summary else None

    invalidated, invalidation_messages, context = _invalidation_details(
        repository, project_id, changes_dataset=changes_dataset
    )
    missing_result_column = None
    warnings: list[str] = []
    if summary_explanation is not None:
        warnings.append(summary_explanation)

    if context is not None:
        target = str(context["target_column"])
        if target not in _column_names(candidate):
            missing_result_column = target
            warnings.append(
                f"Tus decisiones usan «{target}» como resultado y esta tabla no la tiene: "
                "tendrás que definirlas de nuevo."
            )

    if candidate_dimensions.rows < 10:
        warnings.append("Esta tabla tiene menos de 10 filas y no alcanza para entrenar un modelo.")
    if candidate_dimensions.columns < 2:
        warnings.append(
            "Esta tabla tiene menos de 2 columnas y no alcanza para separar un resultado "
            "de las variables."
        )

    if candidate_dimensions.rows < 10 or candidate_dimensions.columns < 2:
        level: ReviewLevel = "block"
    elif warnings or invalidated:
        level = "warning"
    else:
        level = "ok"

    return TableChangeReviewV1(
        active=active_dimensions,
        candidate=candidate_dimensions,
        looks_like_summary=looks_like_summary,
        summary_explanation=summary_explanation,
        missing_result_column=missing_result_column,
        invalidated=invalidated,
        invalidation_messages=invalidation_messages,
        warnings=warnings,
        level=level,
    )
