"""Adaptador opcional para exportar y verificar evidencia SDLC firmada."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from datetime import UTC, datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import ValidationError

from core.contracts.release_evidence import (
    MAX_VALIDITY_DAYS,
    EvidenceSignature,
    ReleaseEvidenceUnsignedV1,
    ReleaseEvidenceV1,
)
from core.db.repositories import WorkflowRepository
from core.domain.canonical import canonical_hash, canonical_json
from core.domain.gates import has_publication_approval, is_report_current


class ReleaseEvidenceError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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
_PENDING_ACTORS = {"", "pendiente", "pending", "todo"}


def _signed_bytes(evidence: ReleaseEvidenceUnsignedV1) -> bytes:
    return canonical_json(evidence).encode("utf-8")


def public_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def sign_release_evidence(
    evidence: ReleaseEvidenceUnsignedV1,
    private_key: Ed25519PrivateKey,
) -> ReleaseEvidenceV1:
    signature = private_key.sign(_signed_bytes(evidence))
    return ReleaseEvidenceV1.model_validate(
        {
            **evidence.model_dump(mode="json"),
            "signature": EvidenceSignature(
                key_id=public_key_id(private_key.public_key()),
                value=base64.b64encode(signature).decode("ascii"),
            ).model_dump(mode="json"),
        }
    )


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
        public_key.verify(signature, _signed_bytes(evidence.unsigned()))
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


def load_private_key(path: Path, password: bytes | None) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=password)
    if not isinstance(key, Ed25519PrivateKey):
        raise ReleaseEvidenceError("SIGNING_KEY_INVALID", "La clave privada no es Ed25519")
    return key


def load_public_key(path: Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ReleaseEvidenceError("PUBLIC_KEY_INVALID", "La clave pública no es Ed25519")
    return key


def _sqlite_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


class ReleaseEvidenceService:
    def __init__(self, repository: WorkflowRepository, runtime_root: Path) -> None:
        self.repository = repository
        self.runtime_root = runtime_root.resolve()

    def _artifact_hash(self, relative_path: str, expected_hash: str) -> str:
        artifact = (self.runtime_root / Path(relative_path)).resolve()
        if not artifact.is_relative_to(self.runtime_root):
            raise ReleaseEvidenceError("ARTIFACT_PATH_INVALID", "Ruta de artefacto inválida")
        if not artifact.is_file():
            raise ReleaseEvidenceError("ARTIFACT_MISSING", "No existe el artefacto del modelo")
        observed = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if observed != expected_hash:
            raise ReleaseEvidenceError(
                "ARTIFACT_HASH_MISMATCH",
                "El artefacto no coincide con el hash del manifiesto",
            )
        return observed

    def export(
        self,
        *,
        report_id: str,
        consumer_repository: str,
        consumer_commit: str,
        goal_settings_version: str,
        goal_settings_commit: str,
        actor_label: str,
        private_key: Ed25519PrivateKey,
        issued_at: datetime | None = None,
    ) -> ReleaseEvidenceV1:
        if actor_label.strip().casefold() in _PENDING_ACTORS:
            raise ReleaseEvidenceError(
                "EXPORT_ACTOR_REQUIRED", "La exportación exige una persona o rol responsable"
            )
        source = self.repository.get_release_evidence_source(report_id)
        if source["report_status"] != "PUBLISHED":
            raise ReleaseEvidenceError("REPORT_NOT_PUBLISHED", "El reporte no está publicado")
        if not is_report_current(self.repository.connection, report_id):
            raise ReleaseEvidenceError("REPORT_NOT_CURRENT", "El reporte ya no está vigente")
        if not has_publication_approval(self.repository.connection, report_id):
            raise ReleaseEvidenceError(
                "PUBLICATION_NOT_APPROVED", "El reporte no conserva una aprobación vigente"
            )
        if source["run_status"] != "COMPLETED" or not source["selected_model"]:
            raise ReleaseEvidenceError(
                "MODEL_NOT_RELEASEABLE", "La ejecución no contiene un modelo recomendado"
            )

        training = self.repository.get_latest_training_decision(source["strategy_version_id"])
        publication = self.repository.get_latest_publication_decision(report_id)
        if training is None or publication is None:
            raise ReleaseEvidenceError("APPROVAL_MISSING", "Falta una aprobación requerida")

        manifest = json.loads(source["manifest_json"])
        artifact_hash = manifest.get("artifact_sha256")
        artifact_path = source["artifact_path"]
        if not artifact_hash or not artifact_path:
            raise ReleaseEvidenceError("ARTIFACT_MISSING", "El run no registra artefacto y hash")
        self._artifact_hash(str(artifact_path), str(artifact_hash))

        metrics = json.loads(source["candidate_metrics_json"])
        selected_model = str(source["selected_model"])
        primary_metric = str(metrics["selection"]["primary_metric"])
        holdout = metrics.get("holdout")
        if not isinstance(holdout, dict) or primary_metric not in holdout:
            raise ReleaseEvidenceError("HOLDOUT_MISSING", "Falta la medida principal en holdout")
        selected_validation = metrics.get("validation", {}).get(selected_model)
        if not isinstance(selected_validation, dict):
            raise ReleaseEvidenceError("VALIDATION_MISSING", "Falta la evaluación del modelo")

        emitted = issued_at or datetime.now(UTC)
        emitted = emitted.replace(tzinfo=UTC) if emitted.tzinfo is None else emitted.astimezone(UTC)
        unsigned = ReleaseEvidenceUnsignedV1.model_validate(
            {
                "consumer": {
                    "repository": consumer_repository,
                    "commit_sha": consumer_commit,
                },
                "goal_settings": {
                    "package_version": goal_settings_version,
                    "commit_sha": goal_settings_commit,
                },
                "refs": {
                    "project_id": source["project_id"],
                    "run_id": source["run_id"],
                    "report_id": source["report_id"],
                    "context_version_id": source["context_version_id"],
                    "context_version_no": source["context_version_no"],
                    "strategy_version_id": source["strategy_version_id"],
                    "strategy_version_no": source["strategy_version_no"],
                },
                "hashes": {
                    "dataset_sha256": source["dataset_sha256"],
                    "context_sha256": source["context_sha256"],
                    "strategy_sha256": source["strategy_fingerprint"],
                    "model_artifact_sha256": artifact_hash,
                    "report_sha256": source["report_sha256"],
                },
                "approvals": {
                    "training": {
                        "approval_id": training["id"],
                        "stage": training["stage"],
                        "decision": training["decision"],
                        "actor_label": training["actor_label"],
                        "approved_at": _sqlite_utc(str(training["created_at"])),
                        "approved_hash": training["approved_hash"],
                    },
                    "publication": {
                        "approval_id": publication["id"],
                        "stage": publication["stage"],
                        "decision": publication["decision"],
                        "actor_label": publication["actor_label"],
                        "approved_at": _sqlite_utc(str(publication["created_at"])),
                        "approved_hash": publication["approved_hash"],
                    },
                },
                "result": {
                    "selected_model": selected_model,
                    "primary_metric": primary_metric,
                    "holdout_value": holdout[primary_metric],
                    "guardrails_passed": selected_validation["guardrails_passed"],
                },
                "validity": {
                    "issued_at": emitted,
                    "expires_at": emitted + timedelta(days=MAX_VALIDITY_DAYS),
                    "report_status": source["report_status"],
                    "context_current_at_issuance": True,
                    "strategy_current_at_issuance": True,
                },
            }
        )
        evidence = sign_release_evidence(unsigned, private_key)
        self.repository.record_release_evidence_export(
            report_id=report_id,
            actor_label=actor_label.strip(),
            detail={
                "consumer_repository": consumer_repository,
                "consumer_commit": consumer_commit,
                "evidence_sha256": canonical_hash(evidence),
                "expires_at": evidence.validity.expires_at.isoformat(),
                "key_id": evidence.signature.key_id,
            },
        )
        return evidence
