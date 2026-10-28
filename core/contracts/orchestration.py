"""MV-06: tareas persistentes y llamadas declarativas del supervisor local."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AgentRole = Literal["coordinator", "business", "data", "modeling", "value_evidence"]
AgentTaskState = Literal[
    "PENDING", "RUNNING", "WAITING_HUMAN", "WAITING_DATA", "COMPLETED", "FAILED", "CANCELLED"
]
ArtifactKind = Literal[
    "capability",
    "cycle",
    "project",
    "dataset",
    "context",
    "strategy",
    "run",
    "predictive_spec",
    "economic_scenario",
    "decision_value",
    "value_policy",
]
_SHA256 = r"^[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def normalize_dates(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("La fecha debe incluir zona horaria")
            return value.astimezone(UTC)
        return value


class ArtifactRefV1(_Contract):
    kind: ArtifactKind
    id: str = Field(min_length=1, max_length=100)
    sha256: str | None = Field(default=None, pattern=_SHA256)


class AgentBudgetV1(_Contract):
    max_steps: int = Field(default=20, ge=1, le=1000, strict=True)
    max_calls: int = Field(default=8, ge=0, le=1000, strict=True)
    max_seconds: int = Field(default=120, ge=1, le=3600, strict=True)


class AgentTaskV1(_Contract):
    schema_version: Literal["agent-task.v1"] = "agent-task.v1"
    id: str = Field(min_length=1, max_length=100)
    capability_id: str = Field(min_length=1, max_length=100)
    cycle_id: str | None = Field(default=None, min_length=1, max_length=100)
    parent_task_id: str | None = Field(default=None, min_length=1, max_length=100)
    root_task_id: str = Field(min_length=1, max_length=100)
    role: AgentRole
    objective: str | None = Field(default=None, min_length=1, max_length=2000)
    state: AgentTaskState = "PENDING"
    input_refs: list[ArtifactRefV1] = Field(default_factory=list, max_length=100)
    context_sha256: str = Field(pattern=_SHA256)
    revision: int = Field(default=0, ge=0, strict=True)
    budget: AgentBudgetV1
    steps_used: int = Field(default=0, ge=0, strict=True)
    calls_used: int = Field(default=0, ge=0, strict=True)
    elapsed_ms: int = Field(default=0, ge=0, strict=True)
    reserved_ms: int = Field(default=0, ge=0, strict=True)
    lease_owner: str | None = Field(default=None, min_length=1, max_length=200)
    lease_until: datetime | None = None
    last_error_code: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,99}$")
    actor_label: str = Field(min_length=1, max_length=200)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def lease_is_complete(self) -> AgentTaskV1:
        if (self.lease_owner is None) != (self.lease_until is None):
            raise ValueError("La reserva requiere propietario y vencimiento")
        return self


def _json_object(value: dict[str, Any]) -> dict[str, Any]:
    # El registro valida después el esquema particular de cada herramienta.
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > 65536:
        raise ValueError("El contenido estructurado supera 64 KiB")
    return value


class ToolCallV1(_Contract):
    schema_version: Literal["tool-call.v1"] = "tool-call.v1"
    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    arguments: dict[str, Any] = Field(default_factory=dict)

    _validate_arguments = field_validator("arguments")(_json_object)


class ToolResultV1(_Contract):
    schema_version: Literal["tool-result.v1"] = "tool-result.v1"
    status: Literal["COMPLETED", "FAILED"]
    output_refs: list[ArtifactRefV1] = Field(default_factory=list, max_length=100)
    summary: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,99}$")

    _validate_summary = field_validator("summary")(_json_object)

    @model_validator(mode="after")
    def failure_has_code(self) -> ToolResultV1:
        if (self.status == "FAILED") != (self.error_code is not None):
            raise ValueError("Sólo un resultado fallido debe incluir error_code")
        return self


class ToolOperationV1(_Contract):
    schema_version: Literal["tool-operation.v1"] = "tool-operation.v1"
    operation_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=100)
    root_task_id: str = Field(min_length=1, max_length=100)
    call: ToolCallV1
    request_sha256: str = Field(pattern=_SHA256)
    status: Literal["CLAIMED", "COMPLETED", "FAILED"]
    task_revision: int = Field(ge=0, strict=True)
    owner: str = Field(min_length=1, max_length=200)
    lease_until: datetime
    reserved_ms: int = Field(ge=1, strict=True)
    calls_cost: int = Field(ge=0, strict=True)
    result: ToolResultV1 | None = None
    created_at: datetime
    finished_at: datetime | None = None
