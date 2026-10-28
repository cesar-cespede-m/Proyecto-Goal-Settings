"""Loop acotado y durable: el proveedor propone, el supervisor ejecuta herramientas."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from core.ai_engine.agents.context import ACTIVE_ROLES, build_context
from core.ai_engine.agents.prompts import PROMPT_VERSION, build_request
from core.ai_engine.agents.validation import validate_proposal
from core.ai_engine.agents.wire import decode_proposal, wire_context
from core.ai_engine.provider import LLMProvider, LLMProviderError, LLMResponse
from core.contracts.agent_reasoning import AgentTurnResultV1, AgentTurnV1
from core.contracts.orchestration import AgentBudgetV1, AgentTaskV1
from core.db.repositories.agent_turns import AgentTurnRepository
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.orchestration.runtime import OrchestrationError, Supervisor

_WAIT = {"WAITING_HUMAN", "WAITING_DATA"}
_ACTIVE = {"PENDING", "RUNNING"}
_REPAIRABLE = {
    "AGENT_INVALID_OUTPUT",
    "AGENT_REFERENCE_NOT_PROVIDED",
    "AGENT_UNSUPPORTED_CLAIM",
    "AGENT_SENSITIVE_OUTPUT",
    "DELEGATION_ROLE_DENIED",
    "TOOL_ROLE_DENIED",
    "INVALID_TOOL_ARGUMENTS",
    "TOOL_HUMAN_FIELD_FORBIDDEN",
    "SPEC_VERSION_CONFLICT",
}


def _key(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 200:
        raise OrchestrationError("INVALID_OPERATION", "Se requiere una clave de operación válida")
    return value.strip()


def _derived(operation: str, action: str) -> str:
    return "agent:" + canonical_hash({"operation_id": operation, "action": action})


class AgentRunner:
    def __init__(
        self,
        workflow: WorkflowRepository,
        provider: LLMProvider | None,
        *,
        clock: Callable[[], datetime] | None = None,
        storage_root: str | Path | None = None,
    ) -> None:
        self.workflow = workflow
        self.provider = provider
        self.supervisor = Supervisor(workflow, clock=clock, storage_root=storage_root)
        self.turns = AgentTurnRepository(workflow)

    def start(
        self,
        *,
        capability_id: str,
        objective: str,
        actor_label: str,
        operation_id: str,
        cycle_id: str | None = None,
        budget: AgentBudgetV1 | None = None,
    ) -> AgentTaskV1:
        operation = _key(operation_id)
        actor = _key(actor_label)
        previous = self.workflow.connection.execute(
            "SELECT id FROM agent_tasks WHERE create_operation_id=?",
            (operation,),
        ).fetchone()
        if previous is not None:
            task = self.supervisor.repository.get_task(str(previous["id"]))
            if (
                task.capability_id,
                task.cycle_id,
                task.role,
                task.parent_task_id,
                task.objective,
                task.actor_label,
                task.budget,
            ) != (
                capability_id,
                cycle_id,
                "coordinator",
                None,
                objective.strip(),
                actor,
                budget or AgentBudgetV1(),
            ):
                raise OrchestrationError(
                    "IDEMPOTENCY_CONFLICT", "La clave ya representa otro trabajo"
                )
            return task
        return self.supervisor.create_task(
            capability_id=capability_id,
            cycle_id=cycle_id,
            role="coordinator",
            objective=objective,
            actor_label=actor,
            operation_id=operation,
            budget=budget,
        )

    def _selected_task(self, task: AgentTaskV1, *, preview: bool = False) -> AgentTaskV1:
        if task.role not in ACTIVE_ROLES:
            raise OrchestrationError(
                "AGENT_ROLE_NOT_ENABLED", "Este especialista aún no está habilitado"
            )
        if not preview and task.state not in _ACTIVE:
            raise OrchestrationError(
                "TASK_NOT_RUNNABLE", "El trabajo requiere revisión o ya terminó"
            )
        children = self.supervisor.repository.list_children(task.id)
        waiting = next((child for child in children if child.state in _WAIT), None)
        if waiting is not None:
            if preview:
                return waiting
            raise OrchestrationError(
                "AGENT_WAITING_INPUT", "El especialista espera información humana"
            )
        return next((child for child in children if child.state in _ACTIVE), task)

    def preview(self, task_id: str) -> dict[str, Any]:
        task = self._selected_task(self.supervisor.repository.get_task(task_id), preview=True)
        return wire_context(build_context(self.workflow, task.id))[0]

    def resume(self, task_id: str, *, expected_revision: int, actor_label: str) -> AgentTaskV1:
        actor = _key(actor_label)
        task = self.supervisor.repository.get_task(task_id)
        self._revision(task, expected_revision)
        waiting = [
            child
            for child in self.supervisor.repository.list_children(task.id)
            if child.state in _WAIT
        ]
        if task.state in _WAIT:
            waiting.insert(0, task)
        if not waiting:
            raise OrchestrationError(
                "TASK_NOT_WAITING", "El trabajo no tiene una revisión pendiente"
            )
        for paused in waiting:
            self.supervisor.transition_task(
                paused.id,
                expected_revision=paused.revision,
                new_state="PENDING",
                actor_label=actor,
                reason_code="HUMAN_REVIEW_RESUMED",
                refresh_context=True,
            )
        return self.supervisor.repository.get_task(task_id)

    def run(
        self,
        task_id: str,
        *,
        expected_revision: int,
        actor_label: str,
        operation_id: str,
        max_turns: int = 8,
    ) -> list[AgentTurnResultV1]:
        if type(max_turns) is not int or not 1 <= max_turns <= 20:
            raise OrchestrationError(
                "INVALID_TURN_LIMIT", "El recorrido admite de uno a veinte turnos"
            )
        operation = _key(operation_id)
        self._revision(self.supervisor.repository.get_task(task_id), expected_revision)
        if self.provider is None:
            raise OrchestrationError(
                "PROVIDER_DISABLED", "Configura un proveedor para ejecutar agentes"
            )
        results: list[AgentTurnResultV1] = []
        for index in range(max_turns):
            task = self.supervisor.repository.get_task(task_id)
            if task.state not in _ACTIVE or any(
                child.state in _WAIT for child in self.supervisor.repository.list_children(task.id)
            ):
                break
            try:
                result = self.step(
                    task_id,
                    expected_revision=task.revision,
                    actor_label=actor_label,
                    operation_id=_derived(operation, f"turn:{index}"),
                )
            except OrchestrationError as error:
                if error.code != "BUDGET_EXHAUSTED":
                    raise
                current = self._selected_task(task)
                self._pause(current, actor_label, error.code)
                break
            results.append(result)
            if result.error_code is not None or result.task.state in _WAIT:
                break
        return results

    def step(
        self,
        task_id: str,
        *,
        expected_revision: int,
        actor_label: str,
        operation_id: str,
    ) -> AgentTurnResultV1:
        operation = _key(operation_id)
        actor = _key(actor_label)
        requested = self.supervisor.repository.get_task(task_id)
        previous = self.turns.get(operation)
        if previous is not None:
            task = self.supervisor.repository.get_task(previous.task_id)
            if requested.id not in {
                task.id,
                task.root_task_id,
            } or previous.operation.request_sha256 != canonical_hash(
                {
                    "task_id": task.id,
                    "call": previous.operation.call.model_dump(mode="json"),
                    "actor_label": actor,
                }
            ):
                raise OrchestrationError(
                    "IDEMPOTENCY_CONFLICT", "La operación pertenece a otro turno"
                )
            return self._complete(previous, actor, replayed=True)
        self._revision(requested, expected_revision)
        pending = next(
            (
                turn
                for candidate in [
                    requested,
                    *self.supervisor.repository.list_children(requested.id),
                ]
                for turn in self.turns.list_turns(candidate.id)
                if turn.proposal is not None
                and turn.dispatch_result is None
                and turn.operation.status == "COMPLETED"
            ),
            None,
        )
        if pending is not None:
            return self._complete(pending, actor, replayed=True)
        task = self._selected_task(requested)
        if self.provider is None:
            raise OrchestrationError(
                "PROVIDER_DISABLED", "Configura un proveedor para ejecutar agentes"
            )
        task = self.supervisor.prepare_turn(
            task.id, expected_revision=task.revision, actor_label=actor
        )
        turn = self._reason(task, operation, actor)
        return self._complete(turn, actor)

    @staticmethod
    def _revision(task: AgentTaskV1, expected: int) -> None:
        if type(expected) is not int or task.revision != expected:
            raise OrchestrationError("STALE_TASK_REVISION", "El trabajo cambió desde su lectura")

    def _capture_request(self, task_id: str, lease_ms: int, repair_error: str | None = None):
        context = build_context(self.workflow, task_id)
        request = build_request(
            context,
            timeout_seconds=max(0.05, min(60.0, lease_ms / 1000 - 0.05)),
            repair_error=repair_error,
        )
        # Un único snapshot produce el payload enviado y su vínculo con referencias locales.
        fingerprint = canonical_hash(
            {
                "request": request.model_dump(mode="json"),
                "local_context_sha256": canonical_hash(context),
            }
        )
        return request, fingerprint, context

    def _verify_request(
        self, task_id: str, request_hash: str, lease_ms: int, repair_error: str | None = None
    ) -> str:
        if self._request_hash(task_id, lease_ms, repair_error) != request_hash:
            raise OrchestrationError(
                "STALE_CONTEXT", "El contexto del agente cambió durante el turno"
            )
        return self.supervisor.context_fingerprint(task_id)

    def _request_hash(self, task_id: str, lease_ms: int, repair_error: str | None) -> str:
        # Vincula aliases a referencias locales sin enviar esas referencias ni sus hashes.
        return self._capture_request(task_id, lease_ms, repair_error)[1]

    def _reason(
        self,
        task: AgentTaskV1,
        operation: str,
        actor: str,
        *,
        repair_of: AgentTurnV1 | None = None,
    ) -> AgentTurnV1:
        if self.provider is None:
            raise OrchestrationError("PROVIDER_DISABLED", "Configura un proveedor para continuar")
        remaining = task.budget.max_seconds * 1000 - task.elapsed_ms - task.reserved_ms
        lease_seconds = min(60, remaining // 1000)
        if lease_seconds < 1:
            raise OrchestrationError("BUDGET_EXHAUSTED", "Se agotó el tiempo de ejecución")
        repair_error = None if repair_of is None else repair_of.error_code
        request, request_hash, sent_context = self._capture_request(
            task.id, lease_seconds * 1000, repair_error
        )

        def verify() -> str:
            return self._verify_request(task.id, request_hash, lease_seconds * 1000, repair_error)

        turn = self.turns.claim(
            task.id,
            operation_id=operation,
            expected_revision=task.revision,
            actor_label=actor,
            owner=self.supervisor.owner,
            provider=self.provider.name,
            model=self.provider.model,
            prompt_version=PROMPT_VERSION,
            prompt_sha256=request_hash,
            lease_seconds=lease_seconds,
            repair_of_operation_id=None if repair_of is None else repair_of.operation_id,
            now=self.supervisor.clock(),
            verify_current=verify,
        )
        if turn.operation.status != "CLAIMED":
            return turn
        response: LLMResponse | None = None
        proposal = None
        error_code = None
        try:
            if self.workflow.connection.in_transaction:
                raise OrchestrationError(
                    "PROVIDER_TRANSACTION_FORBIDDEN", "La red requiere una transacción cerrada"
                )
            raw = self.provider.complete_json(request)
            response = LLMResponse.model_validate(
                raw.model_dump() if isinstance(raw, LLMResponse) else raw
            )
            proposal = validate_proposal(
                decode_proposal(response.content, sent_context), sent_context
            )
        except LLMProviderError as error:
            error_code = error.code
        except OrchestrationError as error:
            error_code = error.code
        except Exception:
            error_code = "AGENT_PROVIDER_FAILURE"
        if error_code is not None and not re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", error_code):
            error_code = "AGENT_PROVIDER_FAILURE"
        metadata: dict[str, Any] = {}
        if response is not None:
            metadata = {
                "response_sha256": hashlib.sha256(response.content.encode("utf-8")).hexdigest(),
                "provider": response.provider,
                "model": response.model,
                "response_id": response.response_id,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
            }
        retry_allowed = repair_of is None and error_code in _REPAIRABLE
        try:
            return self.turns.finish(
                turn.operation_id,
                expected_revision=turn.operation.task_revision,
                owner=self.supervisor.owner,
                proposal=proposal,
                error_code=error_code,
                next_state="RUNNING" if proposal is not None or retry_allowed else "WAITING_HUMAN",
                now=self.supervisor.clock(),
                verify_current=verify,
                **metadata,
            )
        except OrchestrationError as error:
            if error.code != "STALE_CONTEXT":
                raise
            return self.turns.finish(
                turn.operation_id,
                expected_revision=turn.operation.task_revision,
                owner=self.supervisor.owner,
                error_code="STALE_CONTEXT",
                next_state="WAITING_HUMAN",
                now=self.supervisor.clock(),
                **metadata,
            )

    def _complete(
        self, original: AgentTurnV1, actor: str, *, replayed: bool = False
    ) -> AgentTurnResultV1:
        turn = self.turns.get_repair(original.operation_id) or original
        if turn.operation.status == "CLAIMED":
            raise OrchestrationError(
                "AGENT_TURN_IN_PROGRESS", "El turno está reservado; espera o recupera su estado"
            )
        if turn.dispatch_result is not None:
            return turn.dispatch_result.model_copy(update={"replayed": replayed})
        if (
            turn.proposal is None
            and turn.error_code in _REPAIRABLE
            and turn.repair_of_operation_id is None
        ):
            try:
                task = self.supervisor.repository.get_task(turn.task_id)
                turn = self._reason(
                    task, _derived(turn.operation_id, "repair"), actor, repair_of=turn
                )
            except OrchestrationError as error:
                return self._failed(turn, actor, error.code, replayed=replayed)
        if turn.proposal is None:
            return self._failed(
                turn, actor, turn.error_code or "AGENT_INVALID_OUTPUT", replayed=replayed
            )
        try:
            return self._dispatch(turn, actor, replayed=replayed)
        except OrchestrationError as error:
            return self._failed(turn, actor, error.code, replayed=replayed)

    def _verify_turn(self, turn: AgentTurnV1) -> str:
        error = None
        if turn.repair_of_operation_id:
            original = self.turns.get(turn.repair_of_operation_id)
            error = None if original is None else original.error_code
        return self._verify_request(
            turn.task_id, turn.prompt_sha256, turn.operation.reserved_ms, error
        )

    def _dispatch(self, turn: AgentTurnV1, actor: str, *, replayed: bool) -> AgentTurnResultV1:
        proposal = turn.proposal
        assert proposal is not None
        logical = turn.repair_of_operation_id or turn.operation_id
        action_key = _derived(turn.operation_id, "action")
        task = self.supervisor.repository.get_task(turn.task_id)
        child_id = None
        tool_result = None
        child = (
            self.workflow.connection.execute(
                "SELECT id FROM agent_tasks WHERE create_operation_id=?",
                (action_key,),
            ).fetchone()
            if proposal.action == "delegate"
            else None
        )
        tool_operation = (
            self.supervisor.repository.get_operation(action_key)
            if proposal.action == "tool"
            else None
        )
        if child is None and tool_operation is None:
            self._revision(task, turn.operation.task_revision + 1)
            if self._verify_turn(turn) != turn.context_sha256:
                raise OrchestrationError(
                    "STALE_CONTEXT", "La respuesta ya no corresponde al contexto vigente"
                )
        if proposal.action == "tool":
            assert proposal.tool_call is not None
            tool_result = self.supervisor.execute_tool(
                task.id,
                proposal.tool_call,
                operation_id=action_key,
                expected_revision=turn.operation.task_revision + 1,
                actor_label=actor,
            )
        elif proposal.action == "delegate":
            if child is not None:
                delegated = self.supervisor.repository.get_task(str(child["id"]))
                if (delegated.parent_task_id, delegated.role, delegated.objective) != (
                    task.id,
                    proposal.delegate_role,
                    proposal.delegate_objective,
                ):
                    raise OrchestrationError(
                        "IDEMPOTENCY_CONFLICT", "La delegación persistida no coincide"
                    )
            else:
                if any(
                    item.state in _ACTIVE | _WAIT
                    for item in self.supervisor.repository.list_children(task.id)
                ):
                    raise OrchestrationError(
                        "AGENT_CHILD_PENDING", "Resuelve el encargo pendiente antes de delegar otro"
                    )
                delegated = self.supervisor.create_task(
                    capability_id=task.capability_id,
                    cycle_id=task.cycle_id,
                    parent_task_id=task.id,
                    role=proposal.delegate_role,
                    objective=proposal.delegate_objective,
                    input_refs=proposal.evidence_refs,
                    actor_label=actor,
                    operation_id=action_key,
                    parent_expected_revision=turn.operation.task_revision + 1,
                    verify_current=lambda: self._verify_turn(turn),
                )
            child_id = delegated.id
        else:
            state = {
                "wait_human": "WAITING_HUMAN",
                "wait_data": "WAITING_DATA",
                "finish": "COMPLETED",
            }[proposal.action]
            stored = self.turns.transition_and_dispatch(
                turn.operation_id,
                expected_revision=turn.operation.task_revision + 1,
                new_state=state,
                actor_label=actor,
                verify_current=lambda: self._verify_turn(turn),
                now=self.supervisor.clock(),
            )
            assert stored.dispatch_result is not None
            return stored.dispatch_result.model_copy(update={"replayed": replayed})
        result = AgentTurnResultV1(
            operation_id=logical,
            task=self.supervisor.repository.get_task(task.id),
            proposal=proposal,
            child_task_id=child_id,
            tool_result=tool_result,
            error_code=None if tool_result is None else tool_result.error_code,
        )
        stored = self.turns.mark_dispatched(
            turn.operation_id,
            result=result,
            actor_label=actor,
            expected_proposal_sha256=turn.proposal_sha256,
            now=self.supervisor.clock(),
        )
        assert stored.dispatch_result is not None
        return stored.dispatch_result.model_copy(update={"replayed": replayed})

    def _pause(self, task: AgentTaskV1, actor: str, code: str) -> AgentTaskV1:
        if task.state in _ACTIVE and task.lease_owner is None:
            return self.supervisor.transition_task(
                task.id,
                expected_revision=task.revision,
                new_state="WAITING_HUMAN",
                actor_label=actor,
                reason_code=code,
            )
        return task

    def _failed(
        self, turn: AgentTurnV1, actor: str, code: str, *, replayed: bool
    ) -> AgentTurnResultV1:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", code):
            code = "AGENT_EXECUTION_FAILED"
        task = self._pause(self.supervisor.repository.get_task(turn.task_id), actor, code)
        result = AgentTurnResultV1(
            operation_id=turn.repair_of_operation_id or turn.operation_id,
            task=task,
            proposal=turn.proposal,
            error_code=code,
        )
        stored = self.turns.mark_dispatched(
            turn.operation_id,
            result=result,
            actor_label=actor,
            expected_proposal_sha256=turn.proposal_sha256,
            now=self.supervisor.clock(),
        )
        assert stored.dispatch_result is not None
        return stored.dispatch_result.model_copy(update={"replayed": replayed})
