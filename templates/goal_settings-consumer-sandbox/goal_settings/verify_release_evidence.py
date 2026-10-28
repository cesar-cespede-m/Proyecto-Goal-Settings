"""Verificador autónomo de release-evidence.v1 para el CI del repositorio consumidor.

Copia mínima del contrato y del verificador de Goal Settings (`core/contracts/release_evidence.py`
y `core.integration.verify_release_evidence`). No accede a bases, red ni claves privadas.
Goal Settings mantiene una prueba de paridad (`tests/unit/test_consumer_verifier_parity.py`) que
exige el mismo esquema y los mismos códigos de salida 0/10–15 que el original.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

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


class VerificationCode(IntEnum):
    VALID = 0
    SCHEMA_INVALID = 10
    SIGNATURE_INVALID = 11
    CONSUMER_MISMATCH = 12
    EXPIRED = 13
    NOT_APPROVED_OR_PUBLISHED = 14
    NOT_CURRENT_AT_ISSUANCE = 15


_MESSAGES = {
    VerificationCode.VALID: "Evidencia válida",
    VerificationCode.SCHEMA_INVALID: "La evidencia no cumple release-evidence.v1",
    VerificationCode.SIGNATURE_INVALID: "La firma Ed25519 no es válida",
    VerificationCode.CONSUMER_MISMATCH: "La evidencia corresponde a otro repositorio o commit",
    VerificationCode.EXPIRED: "La evidencia está vencida o todavía no era válida",
    VerificationCode.NOT_APPROVED_OR_PUBLISHED: "El reporte no está aprobado y publicado",
    VerificationCode.NOT_CURRENT_AT_ISSUANCE: "El contexto o la estrategia no estaban vigentes",
}


def _canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def public_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def load_public_key(path: Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("La clave pública debe ser Ed25519")
    return key


def verify_release_evidence(
    raw_evidence: str | bytes | dict[str, Any],
    public_key: Ed25519PublicKey,
    *,
    expected_repository: str,
    expected_commit: str,
    now: datetime | None = None,
) -> tuple[VerificationCode, str]:
    try:
        if isinstance(raw_evidence, bytes):
            raw_evidence = raw_evidence.decode("utf-8")
        value = json.loads(raw_evidence) if isinstance(raw_evidence, str) else raw_evidence
        evidence = ReleaseEvidenceV1.model_validate(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError):
        code = VerificationCode.SCHEMA_INVALID
        return code, _MESSAGES[code]

    try:
        signature = base64.b64decode(evidence.signature.value, validate=True)
        if evidence.signature.key_id != public_key_id(public_key):
            raise InvalidSignature
        public_key.verify(signature, _canonical_json(evidence.unsigned()).encode("utf-8"))
    except (InvalidSignature, ValueError, binascii.Error):
        code = VerificationCode.SIGNATURE_INVALID
        return code, _MESSAGES[code]

    if (
        evidence.consumer.repository != expected_repository
        or evidence.consumer.commit_sha != expected_commit
    ):
        code = VerificationCode.CONSUMER_MISMATCH
        return code, _MESSAGES[code]

    checked_at = now or datetime.now(UTC)
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=UTC)
    if checked_at < evidence.validity.issued_at or checked_at >= evidence.validity.expires_at:
        code = VerificationCode.EXPIRED
        return code, _MESSAGES[code]

    if (
        evidence.validity.report_status != "PUBLISHED"
        or evidence.approvals.training.decision != "APPROVED"
        or evidence.approvals.publication.decision != "APPROVED"
        or evidence.approvals.training.approved_hash != evidence.hashes.strategy_sha256
        or evidence.approvals.publication.approved_hash != evidence.hashes.report_sha256
    ):
        code = VerificationCode.NOT_APPROVED_OR_PUBLISHED
        return code, _MESSAGES[code]

    if not (
        evidence.validity.context_current_at_issuance
        and evidence.validity.strategy_current_at_issuance
    ):
        code = VerificationCode.NOT_CURRENT_AT_ISSUANCE
        return code, _MESSAGES[code]

    code = VerificationCode.VALID
    return code, _MESSAGES[code]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--expected-repo", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    try:
        public_key = load_public_key(args.public_key)
        raw_evidence = args.evidence.read_bytes()
    except (OSError, ValueError) as error:
        print(f"Evidencia no verificable: {error}", file=sys.stderr)
        raise SystemExit(VerificationCode.SCHEMA_INVALID) from error

    code, message = verify_release_evidence(
        raw_evidence,
        public_key,
        expected_repository=args.expected_repo,
        expected_commit=args.expected_commit,
    )
    stream = sys.stdout if code is VerificationCode.VALID else sys.stderr
    print(f"{int(code)}: {message}", file=stream)
    raise SystemExit(int(code))


if __name__ == "__main__":
    main()
