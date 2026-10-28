"""Política económica versionada; el test nunca selecciona una política."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.contracts.economics import EconomicEvaluationV1
from core.ml_engine.decision import DecisionPlanV1


class PolicyContentV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["value-policy.v1"] = "value-policy.v1"
    capability_id: str
    cycle_id: str
    run_id: str
    strategy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_model: str
    predictive_version_id: str
    economic_version_id: str
    economic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan: DecisionPlanV1
    economic_evaluation: EconomicEvaluationV1
    evidence_partition: Literal["validation"] = "validation"
    evidence_status: Literal["exploratory"] = "exploratory"
    final_policy_evidence: None = None


class PolicyVersionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    capability_id: str
    version_no: int = Field(gt=0)
    based_on_version_id: str | None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content: PolicyContentV1
    actor_label: str
    created_at: datetime


class PolicyReviewV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    policy_version_id: str
    approved_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["APPROVED", "REJECTED"]
    actor_label: str
    created_at: datetime
