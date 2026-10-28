"""Adaptadores opcionales para integrar Goal Settings con sistemas externos."""

from core.integration.events import EventIngestionResult, EventIngestionService
from core.integration.evidence_transport import (
    DEFAULT_EVIDENCE_VARIABLE,
    EvidenceTransportError,
    EvidenceVariablePayload,
    prepare_evidence_variable,
    publish_evidence_variable,
)
from core.integration.release_evidence import (
    ReleaseEvidenceError,
    ReleaseEvidenceService,
    VerificationCode,
    load_private_key,
    load_public_key,
    public_key_id,
    sign_release_evidence,
    verify_release_evidence,
)

__all__ = [
    "DEFAULT_EVIDENCE_VARIABLE",
    "EventIngestionResult",
    "EventIngestionService",
    "EvidenceTransportError",
    "EvidenceVariablePayload",
    "ReleaseEvidenceError",
    "ReleaseEvidenceService",
    "VerificationCode",
    "load_private_key",
    "load_public_key",
    "public_key_id",
    "prepare_evidence_variable",
    "publish_evidence_variable",
    "sign_release_evidence",
    "verify_release_evidence",
]
