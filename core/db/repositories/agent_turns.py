"""MV-07: llamadas y despacho durables, sin almacenar prompts ni respuestas crudas."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime

from core.contracts.agent_reasoning import AgentProposalV1, AgentTurnResultV1, AgentTurnV1
from core.contracts.orchestration import (
    AgentTaskState,
    AgentTaskV1,
    ToolCallV1,
    ToolOperationV1,
    ToolResultV1,
)
from core.db.connection import transaction
from core.db.repositories.orchestration import OrchestrationError, OrchestrationRepository, _label
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash, canonical_json


class AgentTurnRepository:
    """Los hooks internos escriben únicamente SQLite en la transacción del supervisor."""

    def __init__(self, workflow: WorkflowRepository) -> None:
        self.orchestration = OrchestrationRepository(workflow)
        self.connection = workflow.connection

    def get(self, operation_id: str) -> AgentTurnV1 | None:
        operation_id = self._identifier(operation_id)
        row = self.connection.execute(
            "SELECT * FROM agent_turns WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        operation = self.orchestration.get_operation(operation_id)
        assert operation is not None
        data = dict(row)
        data["operation"] = operation
        data["task_id"] = operation.task_id
        data.pop("response_recorded")
        data.pop("dispatch_sha256")
        proposal_json = data.pop("proposal_json")
        data["proposal"] = None if proposal_json is None else json.loads(proposal_json)
        dispatch_json = data.pop("dispatch_result_json")
        data["dispatch_result"] = None if dispatch_json is None else json.loads(dispatch_json)
        if data["error_code"] is None and operation.result is not None:
            data["error_code"] = operation.result.error_code
        return AgentTurnV1.model_validate(data)

    def get_repair(self, operation_id: str) -> AgentTurnV1 | None:
        operation_id = self._identifier(operation_id)
        row = self.connection.execute(
            "SELECT operation_id FROM agent_turns WHERE repair_of_operation_id=?",
            (operation_id,),
        ).fetchone()
        return None if row is None else self.get(str(row["operation_id"]))

    def list_turns(self, task_id: str) -> list[AgentTurnV1]:
        self.orchestration.get_task(task_id)
        rows = self.connection.execute(
            "SELECT a.operation_id FROM agent_turns a JOIN tool_operations o "
            "ON o.operation_id=a.operation_id WHERE o.task_id=? ORDER BY o.created_at,o.rowid",
            (task_id,),
        ).fetchall()
        turns = [self.get(str(row["operation_id"])) for row in rows]
        return [turn for turn in turns if turn is not None]

    def claim(
        self,
        task_id: str,
        *,
        operation_id: str,
        expected_revision: int,
        actor_label: str,
        owner: str,
        provider: str,
        model: str,
        prompt_version: str,
        prompt_sha256: str,
        lease_seconds: int = 60,
        repair_of_operation_id: str | None = None,
        now: datetime | None = None,
        verify_current: Callable[[], str] | None = None,
    ) -> AgentTurnV1:
        operation_id = self._identifier(operation_id)
        if repair_of_operation_id is not None:
            repair_of_operation_id = self._identifier(repair_of_operation_id)
        call = ToolCallV1(
            tool_name="agent_turn",
            arguments={
                "provider": provider,
                "model": model,
                "prompt_version": prompt_version,
                "prompt_sha256": prompt_sha256,
                "repair_of_operation_id": repair_of_operation_id,
            },
        )

        def persist(operation: ToolOperationV1) -> None:
            task = self.orchestration.get_task(task_id)
            if repair_of_operation_id is not None:
                previous = self.get(repair_of_operation_id)
                if previous is None or previous.task_id != task_id:
                    raise OrchestrationError(
                        "AGENT_REPAIR_SCOPE_MISMATCH",
                        "La reparación requiere el turno de esta tarea",
                    )
                if (
                    previous.repair_of_operation_id is not None
                    or self.get_repair(previous.operation_id) is not None
                ):
                    raise OrchestrationError(
                        "AGENT_REPAIR_LIMIT", "Sólo se admite una reparación por turno"
                    )
                if (
                    previous.operation.status != "FAILED"
                    or previous.proposal is not None
                    or previous.response_sha256 is None
                    or previous.dispatch_result is not None
                ):
                    raise OrchestrationError(
                        "AGENT_REPAIR_NOT_ALLOWED", "El turno no admite reparación de contrato"
                    )
                if previous.context_sha256 != task.context_sha256:
                    raise OrchestrationError("STALE_CONTEXT", "El contexto cambió antes de reparar")
            pending = self.connection.execute(
                "SELECT 1 FROM agent_turns a JOIN tool_operations o "
                "ON o.operation_id=a.operation_id WHERE o.task_id=? "
                "AND a.proposal_json IS NOT NULL AND a.dispatch_result_json IS NULL "
                "AND o.status='COMPLETED'",
                (task_id,),
            ).fetchone()
            if pending is not None:
                raise OrchestrationError(
                    "AGENT_DISPATCH_PENDING", "Primero resuelve la propuesta persistida"
                )
            draft = AgentTurnV1(
                operation_id=operation.operation_id,
                task_id=task_id,
                operation=operation,
                repair_of_operation_id=repair_of_operation_id,
                context_sha256=task.context_sha256,
                provider=provider,
                model=model,
                prompt_version=prompt_version,
                prompt_sha256=prompt_sha256,
            )
            self.connection.execute(
                "INSERT INTO agent_turns(operation_id,repair_of_operation_id,context_sha256,"
                "provider,model,prompt_version,prompt_sha256) VALUES (?,?,?,?,?,?,?)",
                (
                    draft.operation_id,
                    draft.repair_of_operation_id,
                    draft.context_sha256,
                    draft.provider,
                    draft.model,
                    draft.prompt_version,
                    draft.prompt_sha256,
                ),
            )

        self.orchestration.claim_operation(
            task_id,
            call,
            operation_id=operation_id,
            expected_revision=expected_revision,
            actor_label=actor_label,
            owner=owner,
            lease_seconds=lease_seconds,
            calls_cost=1,
            now=now,
            verify_current=verify_current,
            _on_claim=persist,
        )
        stored = self.get(operation_id)
        if stored is None:
            raise OrchestrationError(
                "AGENT_TURN_NOT_FOUND", "La operación no corresponde a un turno IA"
            )
        return stored

    def finish(
        self,
        operation_id: str,
        *,
        expected_revision: int,
        owner: str,
        proposal: AgentProposalV1 | None = None,
        response_sha256: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        response_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: int | None = None,
        error_code: str | None = None,
        next_state: AgentTaskState = "RUNNING",
        now: datetime | None = None,
        verify_current: Callable[[], str] | None = None,
    ) -> AgentTurnV1:
        turn = self._require_turn(operation_id)
        operation_id = turn.operation_id
        if (proposal is None) == (error_code is None):
            raise OrchestrationError(
                "INVALID_AGENT_TURN_OUTCOME", "Se requiere una propuesta o un código de error"
            )
        if proposal is not None:
            proposal = AgentProposalV1.model_validate(
                proposal.model_dump() if isinstance(proposal, AgentProposalV1) else proposal
            )
            if response_sha256 is None:
                raise OrchestrationError(
                    "AGENT_RESPONSE_HASH_REQUIRED", "La propuesta debe referir su respuesta"
                )
        proposal_hash = None if proposal is None else canonical_hash(proposal)
        metadata = {
            "actual_provider": provider,
            "actual_model": model,
            "response_id": response_id,
            "response_sha256": response_sha256,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
        }
        AgentTurnV1.model_validate(
            {
                **turn.model_dump(),
                **metadata,
                "proposal": proposal,
                "proposal_sha256": proposal_hash,
                "error_code": error_code,
            }
        )
        result = ToolResultV1(
            status="COMPLETED" if proposal is not None else "FAILED",
            summary={"proposal_sha256": proposal_hash} if proposal is not None else {},
            error_code=error_code,
        )

        def persist(operation: ToolOperationV1) -> None:
            accepted = proposal if operation.status == "COMPLETED" else None
            accepted_hash = proposal_hash if accepted is not None else None
            if accepted is not None:
                task = self.orchestration.get_task(operation.task_id)
                self.orchestration._validate_refs(task, accepted.evidence_refs)
                if accepted.action == "delegate" and task.role != "coordinator":
                    raise OrchestrationError(
                        "DELEGATION_ROLE_DENIED", "Sólo el coordinador puede delegar"
                    )
            assert operation.result is not None
            self.connection.execute(
                "UPDATE agent_turns SET response_recorded=1,actual_provider=?,actual_model=?,"
                "response_id=?,"
                "response_sha256=?,input_tokens=?,output_tokens=?,latency_ms=?,proposal_json=?,"
                "proposal_sha256=?,error_code=? WHERE operation_id=? AND response_recorded=0",
                (
                    provider,
                    model,
                    response_id,
                    response_sha256,
                    input_tokens,
                    output_tokens,
                    latency_ms,
                    None if accepted is None else canonical_json(accepted),
                    accepted_hash,
                    operation.result.error_code,
                    operation_id,
                ),
            )

        self.orchestration.finish_operation(
            operation_id,
            expected_revision=expected_revision,
            owner=owner,
            result=result,
            next_state=next_state,
            now=now,
            verify_current=verify_current,
            _on_finish=persist,
        )
        return self._require_turn(operation_id)

    def mark_dispatched(
        self,
        operation_id: str,
        *,
        result: AgentTurnResultV1,
        expected_proposal_sha256: str | None,
        actor_label: str,
        now: datetime | None = None,
    ) -> AgentTurnV1:
        instant = self._now(now)
        with transaction(self.connection):
            turn = self._require_turn(operation_id)
            self._persist_dispatch(turn, result, expected_proposal_sha256, instant, actor_label)
        return self._require_turn(operation_id)

    def transition_and_dispatch(
        self,
        operation_id: str,
        *,
        expected_revision: int,
        new_state: AgentTaskState,
        actor_label: str,
        verify_current: Callable[[], str],
        now: datetime | None = None,
    ) -> AgentTurnV1:
        """Aplica una espera/finalización sólo bajo el contexto vigente de aquella respuesta."""

        turn = self._require_turn(operation_id)
        allowed = {
            "wait_human": "WAITING_HUMAN",
            "wait_data": "WAITING_DATA",
            "finish": "COMPLETED",
        }
        if turn.proposal is None or allowed.get(turn.proposal.action) != new_state:
            raise OrchestrationError(
                "AGENT_DISPATCH_MISMATCH", "La transición no corresponde a la propuesta"
            )
        if turn.dispatch_result is not None:
            return turn
        if expected_revision != turn.operation.task_revision + 1:
            raise OrchestrationError(
                "STALE_TASK_REVISION", "La tarea cambió desde la respuesta del proveedor"
            )
        instant = self._now(now)

        def verify() -> str:
            if not callable(verify_current) or verify_current() != turn.context_sha256:
                raise OrchestrationError(
                    "STALE_CONTEXT", "La propuesta requiere revisión con contexto vigente"
                )
            return turn.context_sha256

        def persist(task: AgentTaskV1) -> None:
            self._persist_dispatch(
                turn,
                AgentTurnResultV1(
                    operation_id=turn.repair_of_operation_id or turn.operation_id,
                    task=task,
                    proposal=turn.proposal,
                ),
                turn.proposal_sha256,
                instant,
                actor_label,
            )

        self.orchestration.transition_task(
            turn.task_id,
            expected_revision=expected_revision,
            new_state=new_state,
            actor_label=actor_label,
            now=instant,
            verify_current=verify,
            _on_transition=persist,
        )
        return self._require_turn(operation_id)

    def _persist_dispatch(
        self,
        turn: AgentTurnV1,
        result: AgentTurnResultV1,
        expected_proposal_sha256: str | None,
        instant: datetime,
        actor_label: str,
    ) -> None:
        if not self.connection.in_transaction:
            raise RuntimeError("El despacho requiere una transacción activa")
        if turn.proposal_sha256 != expected_proposal_sha256:
            raise OrchestrationError(
                "AGENT_PROPOSAL_HASH_MISMATCH", "La propuesta cambió desde su lectura"
            )
        if turn.operation.status == "CLAIMED":
            raise OrchestrationError(
                "AGENT_TURN_IN_PROGRESS", "La llamada aún no tiene resultado durable"
            )
        if turn.dispatch_result is not None:
            return
        actor = _label(actor_label, "actor_label")
        result = AgentTurnResultV1.model_validate(result.model_dump())
        if (
            result.operation_id not in {turn.operation_id, turn.repair_of_operation_id}
            or result.task.id != turn.task_id
        ):
            raise OrchestrationError(
                "AGENT_DISPATCH_SCOPE_MISMATCH", "El resultado no corresponde al turno"
            )
        task = self.orchestration.get_task(turn.task_id)
        if (result.task.capability_id, result.task.cycle_id, result.task.root_task_id) != (
            task.capability_id,
            task.cycle_id,
            task.root_task_id,
        ):
            raise OrchestrationError(
                "AGENT_DISPATCH_SCOPE_MISMATCH", "El resultado sale del ámbito de la tarea"
            )
        if result.proposal is not None and canonical_hash(result.proposal) != turn.proposal_sha256:
            raise OrchestrationError(
                "AGENT_PROPOSAL_HASH_MISMATCH", "El resultado refiere otra propuesta"
            )
        if turn.proposal is None and result.error_code is None:
            raise OrchestrationError(
                "INVALID_AGENT_TURN_OUTCOME", "El fallo debe conservar su código"
            )
        if result.child_task_id is not None:
            child = self.orchestration.get_task(result.child_task_id)
            if child.parent_task_id != turn.task_id:
                raise OrchestrationError(
                    "AGENT_DISPATCH_SCOPE_MISMATCH", "La delegación no pertenece a este turno"
                )
        if result.tool_result is not None:
            self.orchestration._validate_refs(task, result.tool_result.output_refs)
        stored = result.model_copy(update={"proposal": turn.proposal, "replayed": False})
        self.connection.execute(
            "UPDATE agent_turns SET dispatch_result_json=?,dispatch_sha256=?,dispatched_at=? "
            "WHERE operation_id=? AND dispatch_result_json IS NULL",
            (
                canonical_json(stored),
                canonical_hash(stored),
                instant.isoformat(),
                turn.operation_id,
            ),
        )
        self.orchestration._event(
            task.id,
            "AGENT_TURN_DISPATCHED",
            actor,
            {
                "operation_id": turn.operation_id,
                "proposal_sha256": turn.proposal_sha256,
                "action": None if turn.proposal is None else turn.proposal.action,
                "child_task_id": stored.child_task_id,
                "error_code": stored.error_code,
            },
            instant,
        )

    def _require_turn(self, operation_id: str) -> AgentTurnV1:
        turn = self.get(operation_id)
        if turn is None:
            raise OrchestrationError("AGENT_TURN_NOT_FOUND", "El turno IA no existe")
        return turn

    @staticmethod
    def _now(value: datetime | None) -> datetime:
        instant = value or datetime.now(UTC)
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise OrchestrationError("INVALID_CLOCK", "La fecha debe incluir zona horaria")
        return instant.astimezone(UTC)

    @staticmethod
    def _identifier(value: str) -> str:
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 200:
            raise OrchestrationError(
                "INVALID_OPERATION", "La clave requiere entre 1 y 200 caracteres"
            )
        return value.strip()
