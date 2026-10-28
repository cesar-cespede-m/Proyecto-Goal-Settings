"""Revisión local de todo texto que saldría hacia un proveedor LLM (§5.3 del plan).

Sanear la sintaxis no basta: un nombre de columna o la descripción libre pueden contener
datos personales o secretos. Esta revisión detecta patrones, los redacta y deja la lista de
hallazgos para que la persona vea, edite o excluya antes de confirmar el envío.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from core.domain.assistant import SafeProfileForLLMV1

MAX_MESSAGE_CHARS = 2000

FindingKind = Literal["EMAIL", "URL", "RUT", "PHONE", "LONG_NUMBER", "SECRET"]

# El orden importa: lo más específico primero, para no clasificar un RUT como número largo.
_PATTERNS: list[tuple[FindingKind, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("URL", re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)),
    ("RUT", re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b")),
    (
        "SECRET",
        re.compile(
            r"\b(?:sk|pk|rk|ghp|gho|xox[abp])[-_][A-Za-z0-9_-]{12,}\b"
            r"|\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{32,}\b"
        ),
    ),
    ("PHONE", re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")),
    ("LONG_NUMBER", re.compile(r"\b\d{6,}\b")),
]


class OutgoingFinding(BaseModel):
    kind: FindingKind
    location: str = Field(description="'message' o la referencia de columna, p. ej. C003")


class OutgoingReview(BaseModel):
    """Lo que la persona ve antes de confirmar el envío."""

    message: str
    profile: SafeProfileForLLMV1
    findings: list[OutgoingFinding]
    truncated: bool

    @property
    def is_clean(self) -> bool:
        return not self.findings


def redact(text: str) -> tuple[str, list[FindingKind]]:
    found: list[FindingKind] = []
    for kind, pattern in _PATTERNS:
        text, count = pattern.subn(f"[{kind.lower()}-redactado]", text)
        found.extend([kind] * count)
    return text, found


def review_outgoing(message: str, profile: SafeProfileForLLMV1) -> OutgoingReview:
    """Redacta mensaje y nombres de columna; nunca envía nada por sí misma."""

    truncated = len(message) > MAX_MESSAGE_CHARS
    clean_message, kinds = redact(message[:MAX_MESSAGE_CHARS])
    findings = [OutgoingFinding(kind=kind, location="message") for kind in kinds]
    columns = []
    for column in profile.columns:
        clean_name, column_kinds = redact(column.name)
        findings.extend(OutgoingFinding(kind=kind, location=column.ref) for kind in column_kinds)
        columns.append(column.model_copy(update={"name": clean_name[:64] or column.ref}))
    return OutgoingReview(
        message=clean_message,
        profile=profile.model_copy(update={"columns": columns}),
        findings=findings,
        truncated=truncated,
    )
