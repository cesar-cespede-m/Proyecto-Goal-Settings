"""MV-11: predicción privada, asignación y observaciones de una cohorte sintética."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class BatchCaseV1(_Strict):
    case_id: str = Field(min_length=1, max_length=200)
    row_number: int = Field(ge=1)
    rank: int = Field(ge=1)
    score: float = Field(ge=0, le=1)
    assigned: bool


class PredictionBatchV1(_Strict):
    schema_version: Literal["prediction-batch.v1"] = "prediction-batch.v1"
    id: str
    capability_id: str
    cycle_id: str
    policy_version_id: str
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str
    report_id: str
    scoring_event_id: str
    scoring_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    period: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    data_classification: Literal["synthetic", "sandbox", "anonymized"]
    provenance_basis: str = Field(min_length=1, max_length=500)
    cases: list[BatchCaseV1] = Field(min_length=1)
    actor_label: str = Field(min_length=1, max_length=200)
    created_at: datetime

    @model_validator(mode="after")
    def unique_cases(self) -> PredictionBatchV1:
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("El lote exige IDs únicos")
        if len({c.row_number for c in self.cases}) != len(self.cases):
            raise ValueError("El lote exige filas únicas")
        return self


class ActionObservationV1(_Strict):
    schema_version: Literal["action-observation.v1"] = "action-observation.v1"
    id: str
    batch_id: str
    case_id: str
    event_type: Literal["ACTED", "OUTCOME_PENDING", "OUTCOME_OBSERVED"]
    outcome_positive: bool | None = None
    occurred_at: datetime
    actor_label: str = Field(min_length=1, max_length=200)
    created_at: datetime

    @model_validator(mode="after")
    def result_only_if_observed(self) -> ActionObservationV1:
        if (self.event_type == "OUTCOME_OBSERVED") != (self.outcome_positive is not None):
            raise ValueError("Sólo un resultado observado lleva la etiqueta positiva/negativa")
        return self


class BatchSummaryV1(_Strict):
    batch_id: str
    period: str
    cases: int = Field(ge=0)
    assigned: int = Field(ge=0)
    acted: int = Field(ge=0)
    assigned_not_acted: int = Field(ge=0)
    observed: int = Field(ge=0)
    pending: int = Field(ge=0)
    positive_observed: int = Field(ge=0)
