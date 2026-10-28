"""Contrato firmado que enlaza evidencia Goal Settings con un commit consumidor."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_SHA_PATTERN = r"^[0-9a-f]{40}$"
_REPOSITORY_PATTERN = r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
MAX_VALIDITY_DAYS = 30


class StrictEvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ConsumerRef(StrictEvidenceModel):
    repository: str = Field(pattern=_REPOSITORY_PATTERN, max_length=200)
    commit_sha: str = Field(pattern=_GIT_SHA_PATTERN)


class GoalSettingsAnalyticsStudioRef(StrictEvidenceModel):
    package_version: str = Field(min_length=1, max_length=64)
    commit_sha: str = Field(pattern=_GIT_SHA_PATTERN)


class EvidenceRefs(StrictEvidenceModel):
    project_id: str = Field(min_length=1, max_length=100)
    run_id: str = Field(min_length=1, max_length=100)
    report_id: str = Field(min_length=1, max_length=100)
    context_version_id: str = Field(min_length=1, max_length=100)
    context_version_no: int = Field(gt=0)
    strategy_version_id: str = Field(min_length=1, max_length=100)
    strategy_version_no: int = Field(gt=0)


class EvidenceHashes(StrictEvidenceModel):
    dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    context_sha256: str = Field(pattern=_SHA256_PATTERN)
    strategy_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    report_sha256: str = Field(pattern=_SHA256_PATTERN)


class ApprovalEvidence(StrictEvidenceModel):
    approval_id: str = Field(min_length=1, max_length=100)
    stage: Literal["TRAINING", "PUBLICATION"]
    decision: Literal["APPROVED", "REJECTED"]
    actor_label: str = Field(min_length=1, max_length=100)
    approved_at: datetime
    approved_hash: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("approved_at")
    @classmethod
    def approved_at_has_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("approved_at debe incluir zona horaria")
        return value


class EvidenceApprovals(StrictEvidenceModel):
    training: ApprovalEvidence
    publication: ApprovalEvidence


class EvidenceResult(StrictEvidenceModel):
    selected_model: Literal["LogisticRegression", "RandomForestClassifier"]
    primary_metric: Literal[
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "average_precision",
        "roc_auc",
    ]
    holdout_value: float = Field(ge=0, le=1)
    guardrails_passed: bool


class EvidenceValidity(StrictEvidenceModel):
    issued_at: datetime
    expires_at: datetime
    report_status: Literal["DRAFT", "APPROVED", "PUBLISHED", "REJECTED", "SUPERSEDED"]
    context_current_at_issuance: bool
    strategy_current_at_issuance: bool

    @model_validator(mode="after")
    def validity_window_is_bounded(self) -> EvidenceValidity:
        if self.issued_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("issued_at y expires_at deben incluir zona horaria")
        duration = self.expires_at - self.issued_at
        if duration <= timedelta(0) or duration > timedelta(days=MAX_VALIDITY_DAYS):
            raise ValueError(f"La vigencia debe ser positiva y no superar {MAX_VALIDITY_DAYS} días")
        return self


class ReleaseEvidenceUnsignedV1(StrictEvidenceModel):
    schema_version: Literal["release-evidence.v1"] = "release-evidence.v1"
    consumer: ConsumerRef
    goal_settings: GoalSettingsAnalyticsStudioRef
    refs: EvidenceRefs
    hashes: EvidenceHashes
    approvals: EvidenceApprovals
    result: EvidenceResult
    validity: EvidenceValidity


class EvidenceSignature(StrictEvidenceModel):
    algorithm: Literal["Ed25519"] = "Ed25519"
    key_id: str = Field(pattern=_SHA256_PATTERN)
    value: str = Field(min_length=1, max_length=256)


class ReleaseEvidenceV1(ReleaseEvidenceUnsignedV1):
    signature: EvidenceSignature

    def unsigned(self) -> ReleaseEvidenceUnsignedV1:
        return ReleaseEvidenceUnsignedV1.model_validate(
            self.model_dump(mode="json", exclude={"signature"})
        )
