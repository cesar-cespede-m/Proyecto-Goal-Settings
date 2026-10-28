"""Transporte fuera de banda de evidencia firmada hacia un repositorio consumidor."""

from __future__ import annotations

import base64
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.integration.release_evidence import (
    VerificationCode,
    load_public_key,
    verify_release_evidence,
)

DEFAULT_EVIDENCE_VARIABLE = "GOAL_SETTINGS_RELEASE_EVIDENCE_B64"


class EvidenceTransportError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EvidenceVariablePayload:
    repository: str
    commit_sha: str
    variable_name: str
    encoded_value: str


def prepare_evidence_variable(
    *,
    evidence_path: Path,
    public_key_path: Path,
    expected_repository: str,
    expected_commit: str,
    variable_name: str = DEFAULT_EVIDENCE_VARIABLE,
    now: datetime | None = None,
) -> EvidenceVariablePayload:
    """Verifica antes de codificar; nunca publica evidencia inválida o de otro commit."""

    try:
        raw_evidence = evidence_path.read_bytes()
        public_key = load_public_key(public_key_path)
    except (OSError, ValueError) as error:
        raise EvidenceTransportError("EVIDENCE_INPUT_INVALID", str(error)) from error
    code, message = verify_release_evidence(
        raw_evidence,
        public_key,
        expected_repository=expected_repository,
        expected_commit=expected_commit,
        now=now,
    )
    if code is not VerificationCode.VALID:
        raise EvidenceTransportError(f"EVIDENCE_VERIFY_{int(code)}", message)
    if not variable_name or not variable_name.replace("_", "").isalnum():
        raise EvidenceTransportError(
            "VARIABLE_NAME_INVALID",
            "El nombre de variable sólo puede contener letras, números y guion bajo",
        )
    return EvidenceVariablePayload(
        repository=expected_repository,
        commit_sha=expected_commit,
        variable_name=variable_name,
        encoded_value=base64.b64encode(raw_evidence).decode("ascii"),
    )


def publish_evidence_variable(
    payload: EvidenceVariablePayload,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Actualiza un secreto Actions sin guardar release-evidence.json en el commit.

    Se usa un secreto y no una variable porque GitHub imprime las variables de cada paso en
    los registros de ejecución, que en un repositorio público son visibles para cualquiera;
    los secretos se ocultan. El valor viaja por la entrada estándar, nunca como argumento.
    """

    command: Sequence[str] = (
        "gh",
        "secret",
        "set",
        payload.variable_name,
        "--repo",
        payload.repository,
    )
    try:
        result = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            input=payload.encoded_value,
        )
    except OSError as error:
        raise EvidenceTransportError(
            "GH_NOT_AVAILABLE", "No se pudo ejecutar GitHub CLI"
        ) from error
    if result.returncode != 0:
        raise EvidenceTransportError(
            "GH_SECRET_UPDATE_FAILED",
            "GitHub CLI rechazó la actualización; revisa autenticación y permisos del repositorio",
        )
