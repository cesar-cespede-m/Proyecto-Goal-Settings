"""MV-13: prueba común sellada y promoción humana del par modelo-política."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.contracts.economics import EconomicEvaluationV1


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PromotionCohortV1(_Strict):
    schema_version: Literal["promotion-cohort.v1"] = "promotion-cohort.v1"
    id: str
    capability_id: str
    current_policy_id: str
    candidate_policy_id: str
    current_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: int = Field(ge=1)
    entity_column: str
    target_column: str
    data_classification: Literal["synthetic", "sandbox", "anonymized"]
    provenance_basis: str = Field(min_length=1, max_length=500)
    coverage_complete: Literal[True] = True
    coverage_basis: str = Field(min_length=1, max_length=500)
    observed_until: datetime
    min_rows: int = Field(ge=1)
    min_f1_improvement: float = Field(ge=0, le=1)
    actor_label: str
    created_at: datetime


class PromotionEvaluationV1(_Strict):
    schema_version: Literal["promotion-evaluation.v1"] = "promotion-evaluation.v1"
    id: str
    cohort_id: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_run_id: str
    candidate_run_id: str
    current_report_id: str
    candidate_report_id: str
    rows: int = Field(ge=1)
    current_f1: float = Field(ge=0, le=1)
    candidate_f1: float = Field(ge=0, le=1)
    current_selected: int = Field(ge=0)
    candidate_selected: int = Field(ge=0)
    current_scenario: EconomicEvaluationV1
    candidate_scenario: EconomicEvaluationV1
    recommendation: Literal["CANDIDATE_BETTER", "KEEP_CURRENT", "INSUFFICIENT"]
    evidence_kind: Literal["synthetic_common_cohort"] = "synthetic_common_cohort"
    actor_label: str
    created_at: datetime


class PromotionReviewV1(_Strict):
    id: str
    evaluation_id: str
    approved_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["APPROVED", "REJECTED"]
    actor_label: str
    created_at: datetime
