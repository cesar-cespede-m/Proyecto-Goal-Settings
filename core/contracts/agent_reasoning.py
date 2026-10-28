"""MV-07: decisiones breves del proveedor, referencias y trazas sin razonamiento privado."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.contracts.orchestration import (
    AgentTaskV1,
    ArtifactRefV1,
    ToolCallV1,
    ToolOperationV1,
    ToolResultV1,
)

_SHA256 = r"^[0-9a-f]{64}$"
_CODE = r"^[A-Z][A-Z0-9_]{0,99}$"
_IDENTIFIER = r"^[A-Za-z0-9_./:-]{1,200}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


class AgentProposalV1(_Contract):
    schema_version: Literal["agent-proposal.v1"] = "agent-proposal.v1"
    action: Literal["tool", "delegate", "wait_human", "wait_data", "finish"]
    message: str = Field(min_length=1, max_length=600)
    evidence_refs: list[ArtifactRefV1] = Field(default_factory=list, max_length=30)
    tool_call: ToolCallV1 | None = None
    delegate_role: Literal["business", "data", "modeling", "value_evidence"] | None = None
    delegate_objective: str | None = Field(default=None, min_length=1, max_length=1000)
    questions: list[str] = Field(default_factory=list, max_length=3)
    reason_code: str | None = Field(default=None, pattern=_CODE)

    @model_validator(mode="after")
    def action_has_only_its_fields(self) -> AgentProposalV1:
        if (self.action == "tool") != (self.tool_call is not None):
            raise ValueError("Sólo tool debe incluir tool_call")
        delegated = self.delegate_role is not None or self.delegate_objective is not None
        if self.action == "delegate":
            if self.delegate_role is None or self.delegate_objective is None:
                raise ValueError("Delegar requiere rol y objetivo explícitos")
        elif delegated:
            raise ValueError("Sólo delegate puede declarar un especialista")
        if self.action in {"wait_human", "wait_data"}:
            if not self.questions:
                raise ValueError("La espera debe indicar qué información hace falta")
        elif self.questions:
            raise ValueError("Sólo una espera puede incluir preguntas")
        if any(not question.strip() or len(question) > 300 for question in self.questions):
            raise ValueError("Cada pregunta debe contener entre 1 y 300 caracteres")
        return self


class AgentTurnResultV1(_Contract):
    schema_version: Literal["agent-turn-result.v1"] = "agent-turn-result.v1"
    operation_id: str = Field(min_length=1, max_length=200)
    task: AgentTaskV1
    proposal: AgentProposalV1 | None = None
    child_task_id: str | None = Field(default=None, min_length=1, max_length=100)
    tool_result: ToolResultV1 | None = None
    error_code: str | None = Field(default=None, pattern=_CODE)
    replayed: bool = False


class AgentTurnV1(_Contract):
    schema_version: Literal["agent-turn.v1"] = "agent-turn.v1"
    operation_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=100)
    operation: ToolOperationV1
    repair_of_operation_id: str | None = Field(default=None, min_length=1, max_length=200)
    context_sha256: str = Field(pattern=_SHA256)
    provider: str = Field(pattern=_IDENTIFIER)
    model: str = Field(pattern=_IDENTIFIER)
    prompt_version: str = Field(pattern=_IDENTIFIER)
    prompt_sha256: str = Field(pattern=_SHA256)
    actual_provider: str | None = Field(default=None, pattern=_IDENTIFIER)
    actual_model: str | None = Field(default=None, pattern=_IDENTIFIER)
    response_id: str | None = Field(default=None, min_length=1, max_length=200)
    response_sha256: str | None = Field(default=None, pattern=_SHA256)
    input_tokens: int | None = Field(default=None, ge=0, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, strict=True)
    latency_ms: int | None = Field(default=None, ge=0, strict=True)
    proposal: AgentProposalV1 | None = None
    proposal_sha256: str | None = Field(default=None, pattern=_SHA256)
    error_code: str | None = Field(default=None, pattern=_CODE)
    dispatch_result: AgentTurnResultV1 | None = None
    dispatched_at: datetime | None = None
