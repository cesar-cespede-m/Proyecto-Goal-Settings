"""Contratos del ciclo 2: Goal Settings dentro del ciclo de desarrollo de software (GA-11).

Sólo definen formatos estables y genéricos para que la Fase B (captura de datos desde otros
sistemas, decisiones de producto, propuestas de cambio) se agregue sin rehacer el ciclo 1. No hay
conectores ni endpoints todavía: `DataSourceV1` sólo habilita la carga de CSV.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256 = r"^[0-9a-f]{64}$"
_SLUG = r"^[a-z][a-z0-9_]{0,63}$"
ENABLED_SOURCE_KINDS = frozenset({"csv_upload"})


class StrictPlatformModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("La fecha debe incluir zona horaria")
    return value


class DataSourceV1(StrictPlatformModel):
    """Cómo entra un dato al catálogo, con la misma validación sea cual sea su origen."""

    schema_version: Literal["data-source.v1"] = "data-source.v1"
    kind: Literal["csv_upload", "api_push", "webhook", "database_query", "event_stream"]
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)

    @property
    def enabled(self) -> bool:
        return self.kind in ENABLED_SOURCE_KINDS


class ObservationEventV1(StrictPlatformModel):
    """Un hecho observado en cualquier sistema; el resultado puede llegar después."""

    schema_version: Literal["observation-event.v1"] = "observation-event.v1"
    source: str = Field(pattern=_SLUG)
    entity_type: str = Field(pattern=_SLUG)
    entity_id: str = Field(min_length=1, max_length=200)
    occurred_at: datetime
    attributes: dict[str, str | int | float | bool | None] = Field(max_length=200)
    outcome: str | int | float | bool | None = None
    outcome_observed_at: datetime | None = None

    _occurred = field_validator("occurred_at")(_require_timezone)

    @model_validator(mode="after")
    def outcome_has_its_moment(self) -> ObservationEventV1:
        if (self.outcome is None) != (self.outcome_observed_at is None):
            raise ValueError("El resultado y su momento se informan juntos")
        if self.outcome_observed_at is not None:
            _require_timezone(self.outcome_observed_at)
            if self.outcome_observed_at < self.occurred_at:
                raise ValueError("El resultado no puede observarse antes que el hecho")
        return self


class VersionRef(StrictPlatformModel):
    id: str = Field(min_length=1, max_length=100)
    sha256: str = Field(pattern=_SHA256)


class BusinessContextRefV2(StrictPlatformModel):
    """Referencia exportable al contexto de negocio confirmado por personas."""

    schema_version: Literal["business-context-ref.v2"] = "business-context-ref.v2"
    context: VersionRef
    source_project_id: str = Field(min_length=1, max_length=100)


class RecipeV1(StrictPlatformModel):
    """Todo lo necesario para reproducir o reentrenar un modelo con datos nuevos."""

    schema_version: Literal["recipe.v1"] = "recipe.v1"
    context: VersionRef
    analysis_plan: VersionRef | None = None  # preparación de datos, si la hubo
    strategy: VersionRef
    task_kind: str = Field(pattern=_SLUG)
    sources: list[VersionRef] = Field(min_length=1, max_length=20)


class DecisionRecordV1(StrictPlatformModel):
    """Una decisión tomada con un análisis o una predicción, y lo que pasó después."""

    schema_version: Literal["decision-record.v1"] = "decision-record.v1"
    decision: str = Field(min_length=3, max_length=2000)
    based_on: Literal["analysis", "prediction", "report"]
    reference_id: str = Field(min_length=1, max_length=100)
    actor_label: str = Field(min_length=1, max_length=200)
    decided_at: datetime
    observed_result: str | None = Field(default=None, max_length=2000)
    observed_at: datetime | None = None

    _decided = field_validator("decided_at")(_require_timezone)

    @model_validator(mode="after")
    def result_comes_after_the_decision(self) -> DecisionRecordV1:
        if (self.observed_result is None) != (self.observed_at is None):
            raise ValueError("El resultado observado y su momento se informan juntos")
        if self.observed_at is not None:
            _require_timezone(self.observed_at)
            if self.observed_at < self.decided_at:
                raise ValueError("El resultado no puede observarse antes de decidir")
        return self


class ChangeProposalV1(StrictPlatformModel):
    """Propuesta de la IA para el ciclo de desarrollo; sólo llega como pull request revisado."""

    schema_version: Literal["change-proposal.v1"] = "change-proposal.v1"
    kind: Literal["analysis_kind", "transform", "task_kind", "code_change"]
    title: str = Field(min_length=5, max_length=200)
    rationale: str = Field(min_length=10, max_length=4000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=50)
    delivery: Literal["pull_request"] = "pull_request"
    requires_human_review: Literal[True] = True
