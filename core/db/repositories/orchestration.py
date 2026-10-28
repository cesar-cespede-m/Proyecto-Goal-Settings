"""MV-06: reservas, CAS y resultados durables; sin ejecutar herramientas ni proveedores."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from core.contracts.orchestration import (
    AgentBudgetV1,
    AgentRole,
    AgentTaskState,
    AgentTaskV1,
    ArtifactRefV1,
    ToolCallV1,
    ToolOperationV1,
    ToolResultV1,
)
from core.db.connection import transaction
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash, canonical_json
from core.domain.capabilities import CapabilityService
from core.domain.errors import DomainError

_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
_TRANSITIONS = {
    "PENDING": {"RUNNING", "WAITING_HUMAN", "WAITING_DATA", "FAILED", "CANCELLED", "COMPLETED"},
    "RUNNING": {"WAITING_HUMAN", "WAITING_DATA", "COMPLETED", "FAILED", "CANCELLED"},
    "WAITING_HUMAN": {"PENDING", "RUNNING", "FAILED", "CANCELLED"},
    "WAITING_DATA": {"PENDING", "RUNNING", "FAILED", "CANCELLED"},
    "COMPLETED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}


class OrchestrationError(DomainError, ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 200:
        raise OrchestrationError(
            "INVALID_OPERATION", f"{name} requiere texto de 1 a 200 caracteres"
        )
    return value.strip()


def _now(value: datetime | None) -> datetime:
    instant = value or datetime.now(UTC)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise OrchestrationError("INVALID_CLOCK", "La fecha debe incluir zona horaria")
    return instant.astimezone(UTC)


def _hash(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise OrchestrationError("INVALID_CONTEXT_HASH", "Se requiere una huella SHA-256")
    return value


class OrchestrationRepository:
    def __init__(self, repository: WorkflowRepository) -> None:
        self.repository = repository
        self.connection = repository.connection
        self.capabilities = CapabilityService(repository)

    def create_task(
        self,
        *,
        capability_id: str,
        role: AgentRole,
        context_sha256: str,
        actor_label: str,
        operation_id: str,
        cycle_id: str | None = None,
        parent_task_id: str | None = None,
        input_refs: Sequence[ArtifactRefV1] = (),
        budget: AgentBudgetV1 | None = None,
        now: datetime | None = None,
        objective: str | None = None,
        parent_expected_revision: int | None = None,
        verify_current: Callable[[], str] | None = None,
    ) -> AgentTaskV1:
        actor = _label(actor_label, "actor_label")
        operation = _label(operation_id, "operation_id")
        instant = _now(now)
        task_id = str(uuid4())
        declared_budget = AgentBudgetV1.model_validate(
            (budget.model_dump() if isinstance(budget, AgentBudgetV1) else budget) or {}
        )
        draft = AgentTaskV1(
            id=task_id,
            capability_id=capability_id,
            cycle_id=cycle_id,
            parent_task_id=parent_task_id,
            root_task_id=task_id,
            role=role,
            objective=objective,
            input_refs=list(input_refs),
            context_sha256=context_sha256,
            budget=declared_budget,
            actor_label=actor,
            created_at=instant,
            updated_at=instant,
        )
        request = {
            "capability_id": capability_id,
            "cycle_id": cycle_id,
            "parent_task_id": parent_task_id,
            "role": role,
            "context_sha256": context_sha256,
            "input_refs": [ref.model_dump(mode="json") for ref in draft.input_refs],
            "budget": declared_budget.model_dump(mode="json") if parent_task_id is None else None,
            "actor_label": actor,
            "objective": draft.objective,
        }
        request_hash = canonical_hash(request)
        with transaction(self.connection):
            replay = self.connection.execute(
                "SELECT id,create_request_sha256 FROM agent_tasks WHERE create_operation_id=?",
                (operation,),
            ).fetchone()
            if replay is not None:
                if replay["create_request_sha256"] != request_hash:
                    raise OrchestrationError(
                        "IDEMPOTENCY_CONFLICT", "La clave ya representa otra tarea"
                    )
                return self.get_task(str(replay["id"]))
            if parent_task_id is not None:
                parent = self.get_task(parent_task_id)
                if parent_expected_revision is not None:
                    self._require_revision(parent, parent_expected_revision)
                self._verify_context(parent.context_sha256, verify_current)
                if parent.capability_id != capability_id or (
                    parent.cycle_id is not None and parent.cycle_id != cycle_id
                ):
                    raise OrchestrationError(
                        "TASK_SCOPE_MISMATCH", "La tarea hija sale del ámbito padre"
                    )
                if (
                    parent.state in _TERMINAL
                    or self.get_task(parent.root_task_id).state in _TERMINAL
                ):
                    raise OrchestrationError("TASK_NOT_ACTIVE", "La tarea padre ya terminó")
                if budget is not None and declared_budget != parent.budget:
                    raise OrchestrationError(
                        "CHILD_BUDGET_OVERRIDE", "Las tareas hijas comparten presupuesto"
                    )
                draft = draft.model_copy(
                    update={"root_task_id": parent.root_task_id, "budget": parent.budget}
                )
            else:
                if parent_expected_revision is not None:
                    raise OrchestrationError(
                        "TASK_PARENT_REQUIRED", "La revisión requiere tarea padre"
                    )
                self._verify_context(draft.context_sha256, verify_current)
            self._require_scope(draft, active=True)
            self._validate_refs(draft, draft.input_refs)
            self.connection.execute(
                """INSERT INTO agent_tasks(
                    id,capability_id,cycle_id,parent_task_id,root_task_id,role,objective,state,input_refs_json,
                    context_sha256,revision,actor_label,create_operation_id,create_request_sha256,
                    created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,'PENDING',?,?,0,?,?,?,?,?)""",
                (
                    task_id,
                    capability_id,
                    cycle_id,
                    parent_task_id,
                    draft.root_task_id,
                    role,
                    draft.objective,
                    canonical_json([ref.model_dump(mode="json") for ref in draft.input_refs]),
                    context_sha256,
                    actor,
                    operation,
                    request_hash,
                    instant.isoformat(),
                    instant.isoformat(),
                ),
            )
            if parent_task_id is None:
                self.connection.execute(
                    "INSERT INTO agent_budgets(root_task_id,max_steps,max_calls,max_seconds) "
                    "VALUES (?,?,?,?)",
                    (
                        task_id,
                        declared_budget.max_steps,
                        declared_budget.max_calls,
                        declared_budget.max_seconds,
                    ),
                )
            self._event(
                task_id,
                "TASK_CREATED",
                actor,
                {"role": role, "parent_task_id": parent_task_id},
                instant,
            )
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> AgentTaskV1:
        row = self.connection.execute(
            """SELECT t.*,b.max_steps,b.max_calls,b.max_seconds,b.steps_used,b.calls_used,
                      b.elapsed_ms,b.reserved_ms
               FROM agent_tasks t JOIN agent_budgets b ON b.root_task_id=t.root_task_id
               WHERE t.id=?""",
            (task_id,),
        ).fetchone()
        if row is None:
            raise OrchestrationError("TASK_NOT_FOUND", "La tarea no existe")
        value = dict(row)
        value["input_refs"] = json.loads(value.pop("input_refs_json"))
        value["budget"] = {key: value.pop(key) for key in ("max_steps", "max_calls", "max_seconds")}
        value.pop("create_operation_id")
        value.pop("create_request_sha256")
        return AgentTaskV1.model_validate(value)

    def list_tasks(self, root_task_id: str | None = None) -> list[AgentTaskV1]:
        rows = self.connection.execute(
            "SELECT id FROM agent_tasks WHERE (? IS NULL OR root_task_id=?) "
            "ORDER BY created_at,rowid",
            (root_task_id, root_task_id),
        ).fetchall()
        return [self.get_task(str(row["id"])) for row in rows]

    def list_children(self, task_id: str) -> list[AgentTaskV1]:
        self.get_task(task_id)
        rows = self.connection.execute(
            "SELECT id FROM agent_tasks WHERE parent_task_id=? ORDER BY created_at,rowid",
            (task_id,),
        ).fetchall()
        return [self.get_task(str(row["id"])) for row in rows]

    def get_operation(self, operation_id: str) -> ToolOperationV1 | None:
        row = self.connection.execute(
            "SELECT * FROM tool_operations WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["call"] = json.loads(value.pop("call_json"))
        result = value.pop("result_json")
        value["result"] = None if result is None else json.loads(result)
        return ToolOperationV1.model_validate(value)

    def list_operations(self, task_id: str) -> list[ToolOperationV1]:
        self.get_task(task_id)
        rows = self.connection.execute(
            "SELECT operation_id FROM tool_operations WHERE task_id=? ORDER BY created_at,rowid",
            (task_id,),
        ).fetchall()
        return [self.get_operation(str(row["operation_id"])) for row in rows]  # type: ignore[misc]

    def claim_operation(
        self,
        task_id: str,
        call: ToolCallV1,
        *,
        operation_id: str,
        expected_revision: int,
        owner: str,
        actor_label: str,
        lease_seconds: int = 60,
        calls_cost: int = 0,
        now: datetime | None = None,
        verify_current: Callable[[], str] | None = None,
        _on_claim: Callable[[ToolOperationV1], None] | None = None,
    ) -> ToolOperationV1:
        operation = _label(operation_id, "operation_id")
        owner = _label(owner, "owner")
        actor = _label(actor_label, "actor_label")
        instant = _now(now)
        call = ToolCallV1.model_validate(
            call.model_dump() if isinstance(call, ToolCallV1) else call
        )
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 3600:
            raise OrchestrationError("INVALID_LEASE", "La reserva requiere entre 1 y 3600 segundos")
        if type(calls_cost) is not int or not 0 <= calls_cost <= 1000:
            raise OrchestrationError(
                "INVALID_CALLS_COST", "El costo de llamadas debe ser un entero no negativo"
            )
        request_hash = canonical_hash(
            {"task_id": task_id, "call": call.model_dump(mode="json"), "actor_label": actor}
        )
        with transaction(self.connection):
            task = self.get_task(task_id)
            replay = self.get_operation(operation)
            if replay is not None:
                if replay.request_sha256 != request_hash or replay.task_id != task_id:
                    raise OrchestrationError(
                        "IDEMPOTENCY_CONFLICT", "La clave ya representa otra operación"
                    )
                if replay.status == "CLAIMED":
                    raise OrchestrationError(
                        "OPERATION_IN_PROGRESS",
                        "La operación todavía requiere resultado o recuperación",
                    )
                return replay
            self._require_revision(task, expected_revision)
            self._require_scope(task, active=True)
            if task.state not in {"PENDING", "RUNNING"}:
                raise OrchestrationError(
                    "TASK_NOT_RUNNABLE", "La tarea requiere datos, revisión o ya terminó"
                )
            self._require_active_ancestors(task)
            if (
                self.connection.execute(
                    "SELECT 1 FROM tool_operations WHERE root_task_id=? AND status='CLAIMED'",
                    (task.root_task_id,),
                ).fetchone()
                is not None
            ):
                raise OrchestrationError(
                    "ROOT_BUSY", "El supervisor ya tiene una operación reservada"
                )
            self._verify_context(task.context_sha256, verify_current)
            if (
                task.steps_used >= task.budget.max_steps
                or task.calls_used + calls_cost > task.budget.max_calls
            ):
                raise OrchestrationError(
                    "BUDGET_EXHAUSTED", "Se agotó el presupuesto global de pasos o llamadas"
                )
            remaining_ms = task.budget.max_seconds * 1000 - task.elapsed_ms - task.reserved_ms
            if remaining_ms <= 0:
                raise OrchestrationError("BUDGET_EXHAUSTED", "Se agotó el tiempo activo permitido")
            reserved_ms = min(lease_seconds * 1000, remaining_ms)
            lease_until = instant + timedelta(milliseconds=reserved_ms)
            revision = task.revision + 1
            self.connection.execute(
                "UPDATE agent_budgets SET steps_used=steps_used+1,calls_used=calls_used+?,"
                "reserved_ms=reserved_ms+? WHERE root_task_id=?",
                (calls_cost, reserved_ms, task.root_task_id),
            )
            self.connection.execute(
                "UPDATE agent_tasks SET state='RUNNING',revision=?,lease_owner=?,lease_until=?,"
                "last_error_code=NULL,updated_at=? WHERE id=? AND revision=?",
                (
                    revision,
                    owner,
                    lease_until.isoformat(),
                    instant.isoformat(),
                    task_id,
                    task.revision,
                ),
            )
            self.connection.execute(
                """INSERT INTO tool_operations(operation_id,task_id,root_task_id,call_json,
                    request_sha256,status,task_revision,owner,lease_until,reserved_ms,calls_cost,created_at)
                    VALUES (?,?,?,?,?,'CLAIMED',?,?,?,?,?,?)""",
                (
                    operation,
                    task_id,
                    task.root_task_id,
                    canonical_json(call),
                    request_hash,
                    revision,
                    owner,
                    lease_until.isoformat(),
                    reserved_ms,
                    calls_cost,
                    instant.isoformat(),
                ),
            )
            self._event(
                task_id,
                "TOOL_CLAIMED",
                actor,
                {
                    "operation_id": operation,
                    "tool_name": call.tool_name,
                    "reserved_ms": reserved_ms,
                    "calls_cost": calls_cost,
                    "revision": revision,
                },
                instant,
            )
            if _on_claim is not None:
                claimed = self.get_operation(operation)
                assert claimed is not None
                _on_claim(claimed)
        result = self.get_operation(operation)
        assert result is not None
        return result

    def finish_operation(
        self,
        operation_id: str,
        *,
        expected_revision: int,
        owner: str,
        result: ToolResultV1,
        next_state: AgentTaskState = "RUNNING",
        now: datetime | None = None,
        verify_current: Callable[[], str] | None = None,
        expected_context_sha256: str | None = None,
        _on_finish: Callable[[ToolOperationV1], None] | None = None,
    ) -> ToolOperationV1:
        owner = _label(owner, "owner")
        instant = _now(now)
        result = ToolResultV1.model_validate(
            result.model_dump() if isinstance(result, ToolResultV1) else result
        )
        with transaction(self.connection):
            operation = self.get_operation(operation_id)
            if operation is None:
                raise OrchestrationError("OPERATION_NOT_FOUND", "La operación no existe")
            if operation.status != "CLAIMED":
                if operation.owner == owner and operation.result == result:
                    return operation
                raise OrchestrationError(
                    "OPERATION_ALREADY_FINISHED", "El resultado ya quedó registrado"
                )
            task = self.get_task(operation.task_id)
            self._require_revision(task, expected_revision)
            if (
                operation.task_revision != expected_revision
                or operation.owner != owner
                or task.lease_owner != owner
            ):
                raise OrchestrationError(
                    "LEASE_NOT_OWNED", "El ejecutor no posee la reserva vigente"
                )
            if instant < operation.created_at:
                raise OrchestrationError("INVALID_CLOCK", "El final no puede preceder al comienzo")
            expected_context = _hash(expected_context_sha256 or task.context_sha256)
            self._verify_context(expected_context, verify_current)
            if expected_context != task.context_sha256 and verify_current is None:
                raise OrchestrationError(
                    "CONTEXT_VERIFICATION_REQUIRED", "La nueva huella requiere verificación"
                )
            self._validate_refs(task, result.output_refs)
            if next_state not in _TRANSITIONS["RUNNING"] | {"RUNNING"}:
                raise OrchestrationError(
                    "INVALID_TASK_TRANSITION", "El estado de salida no es válido"
                )
            elapsed_ms = max(1, int((instant - operation.created_at).total_seconds() * 1000))
            if instant >= operation.lease_until:
                result = ToolResultV1(
                    status="FAILED",
                    output_refs=result.output_refs,
                    summary=result.summary,
                    error_code="OPERATION_LEASE_EXPIRED",
                )
                next_state = "WAITING_HUMAN"
            self._finish(operation, task, result, next_state, expected_context, elapsed_ms, instant)
            if _on_finish is not None:
                finished = self.get_operation(operation_id)
                assert finished is not None
                _on_finish(finished)
        stored = self.get_operation(operation_id)
        assert stored is not None
        return stored

    def recover_expired_claims(
        self,
        *,
        now: datetime | None = None,
        recover_result: Callable[[ToolOperationV1], ToolResultV1 | None] | None = None,
        root_task_id: str | None = None,
    ) -> list[str]:
        instant = _now(now)
        recovered: list[str] = []
        with transaction(self.connection):
            if (
                root_task_id is not None
                and self.get_task(root_task_id).root_task_id != root_task_id
            ):
                raise OrchestrationError("INVALID_ROOT_TASK", "La recuperación requiere una raíz")
            rows = self.connection.execute(
                "SELECT operation_id FROM tool_operations "
                "WHERE status='CLAIMED' AND lease_until<=? AND (? IS NULL OR root_task_id=?)",
                (instant.isoformat(), root_task_id, root_task_id),
            ).fetchall()
            for row in rows:
                operation = self.get_operation(str(row["operation_id"]))
                assert operation is not None
                task = self.get_task(operation.task_id)
                result = recover_result(operation) if recover_result is not None else None
                if result is None:
                    result = ToolResultV1(status="FAILED", error_code="OPERATION_OUTCOME_UNKNOWN")
                result = ToolResultV1.model_validate(result.model_dump())
                self._validate_refs(task, result.output_refs)
                self._finish(
                    operation,
                    task,
                    result,
                    "WAITING_HUMAN",
                    task.context_sha256,
                    operation.reserved_ms,
                    instant,
                )
                self._event(
                    task.id,
                    "TOOL_RECOVERED",
                    "supervisor",
                    {
                        "operation_id": operation.operation_id,
                        "outcome_reconstructed": result.status == "COMPLETED",
                    },
                    instant,
                )
                recovered.append(operation.operation_id)
        return recovered

    def transition_task(
        self,
        task_id: str,
        *,
        expected_revision: int,
        new_state: AgentTaskState,
        actor_label: str,
        reason_code: str | None = None,
        context_sha256: str | None = None,
        now: datetime | None = None,
        verify_current: Callable[[], str] | None = None,
        _on_transition: Callable[[AgentTaskV1], None] | None = None,
    ) -> AgentTaskV1:
        actor = _label(actor_label, "actor_label")
        instant = _now(now)
        if reason_code is not None and re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", reason_code) is None:
            raise OrchestrationError("INVALID_REASON_CODE", "El motivo debe ser un código estable")
        with transaction(self.connection):
            task = self.get_task(task_id)
            self._require_revision(task, expected_revision)
            self._verify_context(task.context_sha256, verify_current)
            if new_state not in _TRANSITIONS[task.state]:
                raise OrchestrationError(
                    "INVALID_TASK_TRANSITION", "La transición de tarea no está permitida"
                )
            if task.lease_owner is not None:
                raise OrchestrationError(
                    "TASK_BUSY", "La operación reservada debe terminar o recuperarse"
                )
            if (
                task.id == task.root_task_id
                and self.connection.execute(
                    "SELECT 1 FROM tool_operations WHERE root_task_id=? AND status='CLAIMED'",
                    (task.id,),
                ).fetchone()
                is not None
            ):
                raise OrchestrationError("ROOT_BUSY", "Existe una operación hija reservada")
            if new_state == "COMPLETED" and any(
                child.state not in _TERMINAL for child in self.list_children(task.id)
            ):
                raise OrchestrationError(
                    "TASK_CHILDREN_ACTIVE", "Todavía hay tareas hijas pendientes"
                )
            context = _hash(context_sha256 or task.context_sha256)
            self.connection.execute(
                "UPDATE agent_tasks SET state=?,revision=revision+1,context_sha256=?,"
                "last_error_code=?,"
                "updated_at=? WHERE id=? AND revision=?",
                (new_state, context, reason_code, instant.isoformat(), task.id, task.revision),
            )
            self._event(
                task.id,
                "TASK_STATE_CHANGED",
                actor,
                {
                    "from": task.state,
                    "to": new_state,
                    "reason_code": reason_code,
                },
                instant,
            )
            if _on_transition is not None:
                _on_transition(self.get_task(task.id))
        return self.get_task(task_id)

    def list_events(self, task_id: str) -> list[dict[str, Any]]:
        self.get_task(task_id)
        rows = self.connection.execute(
            "SELECT * FROM orchestration_events WHERE task_id=? ORDER BY created_at,rowid",
            (task_id,),
        ).fetchall()
        return [{**dict(row), "detail": json.loads(row["detail_json"])} for row in rows]

    def refresh_task_context(
        self,
        task_id: str,
        *,
        expected_revision: int,
        context_sha256: str,
        actor_label: str,
        verify_current: Callable[[], str],
        now: datetime | None = None,
    ) -> AgentTaskV1:
        """Captura el contexto antes de un turno; nunca legitima un retorno ya obsoleto."""

        actor = _label(actor_label, "actor_label")
        context = _hash(context_sha256)
        instant = _now(now)
        if not callable(verify_current):
            raise OrchestrationError("CONTEXT_VERIFICATION_REQUIRED", "Se requiere lectura vigente")
        with transaction(self.connection):
            task = self.get_task(task_id)
            self._require_revision(task, expected_revision)
            self._require_scope(task, active=True)
            self._require_active_ancestors(task)
            if task.state not in {"PENDING", "RUNNING"}:
                raise OrchestrationError(
                    "TASK_NOT_RUNNABLE", "La tarea requiere revisión o ya terminó"
                )
            if (
                self.connection.execute(
                    "SELECT 1 FROM tool_operations WHERE root_task_id=? AND status='CLAIMED'",
                    (task.root_task_id,),
                ).fetchone()
                is not None
            ):
                raise OrchestrationError("ROOT_BUSY", "Existe una operación reservada")
            self._verify_context(context, verify_current)
            self.connection.execute(
                "UPDATE agent_tasks SET context_sha256=?,revision=revision+1,updated_at=? "
                "WHERE id=? AND revision=?",
                (context, instant.isoformat(), task.id, task.revision),
            )
            self._event(
                task.id,
                "TASK_CONTEXT_REFRESHED",
                actor,
                {
                    "previous_sha256": task.context_sha256,
                    "context_sha256": context,
                },
                instant,
            )
        return self.get_task(task_id)

    def _finish(
        self,
        operation: ToolOperationV1,
        task: AgentTaskV1,
        result: ToolResultV1,
        next_state: AgentTaskState,
        context_sha256: str,
        elapsed_ms: int,
        instant: datetime,
    ) -> None:
        self.connection.execute(
            "UPDATE tool_operations SET status=?,result_json=?,finished_at=? "
            "WHERE operation_id=? AND status='CLAIMED'",
            (result.status, canonical_json(result), instant.isoformat(), operation.operation_id),
        )
        self.connection.execute(
            "UPDATE agent_budgets SET elapsed_ms=elapsed_ms+?,reserved_ms=reserved_ms-? "
            "WHERE root_task_id=?",
            (elapsed_ms, operation.reserved_ms, task.root_task_id),
        )
        self.connection.execute(
            "UPDATE agent_tasks SET state=?,revision=revision+1,lease_owner=NULL,lease_until=NULL,"
            "last_error_code=?,context_sha256=?,updated_at=? WHERE id=? AND revision=?",
            (
                next_state,
                result.error_code,
                context_sha256,
                instant.isoformat(),
                task.id,
                task.revision,
            ),
        )
        self._event(
            task.id,
            "TOOL_FINISHED",
            operation.owner,
            {
                "operation_id": operation.operation_id,
                "status": result.status,
                "error_code": result.error_code,
                "elapsed_ms": elapsed_ms,
            },
            instant,
        )

    @staticmethod
    def _require_revision(task: AgentTaskV1, expected_revision: int) -> None:
        if type(expected_revision) is not int or task.revision != expected_revision:
            raise OrchestrationError("STALE_TASK_REVISION", "La tarea cambió desde su lectura")

    @staticmethod
    def _verify_context(expected: str, callback: Callable[[], str] | None) -> None:
        if callback is not None and _hash(callback()) != expected:
            raise OrchestrationError("STALE_CONTEXT", "El contexto cambió; requiere revisión")

    def _require_scope(self, task: AgentTaskV1, *, active: bool) -> None:
        capability = self.capabilities.get_capability(task.capability_id)
        product = self.capabilities.get_product(capability.product_id)
        self.capabilities.require_scope(task.capability_id, cycle_id=task.cycle_id)
        if active and (capability.status != "active" or product.status != "active"):
            raise OrchestrationError(
                "CAPABILITY_NOT_ACTIVE", "La capacidad o producto está archivado"
            )
        if active and task.cycle_id is not None:
            cycle = self.capabilities.get_cycle(task.cycle_id)
            if not any(
                row["id"] == cycle.project_id for row in self.repository.list_active_projects()
            ):
                raise OrchestrationError("CYCLE_NOT_ACTIVE", "El ciclo técnico está archivado")

    def _validate_refs(self, task: AgentTaskV1, refs: Sequence[ArtifactRefV1]) -> None:
        for ref in refs:
            expected_hash: str | None = None
            if ref.kind == "capability":
                if ref.id != task.capability_id:
                    raise OrchestrationError(
                        "ARTIFACT_SCOPE_MISMATCH", "La referencia sale de la capacidad"
                    )
                expected_hash = canonical_hash(self.capabilities.get_capability(ref.id))
            elif ref.kind == "cycle":
                self.capabilities.require_scope(task.capability_id, cycle_id=ref.id)
                if task.cycle_id is not None and task.cycle_id != ref.id:
                    raise OrchestrationError(
                        "ARTIFACT_SCOPE_MISMATCH", "La referencia sale del ciclo"
                    )
                expected_hash = canonical_hash(self.capabilities.get_cycle(ref.id))
            elif ref.kind in {"project", "dataset", "run"}:
                self.capabilities.require_scope(
                    task.capability_id, cycle_id=task.cycle_id, **{f"{ref.kind}_id": ref.id}
                )
                if ref.kind == "dataset":
                    expected_hash = str(self.repository.get_dataset(ref.id)["sha256"])
            elif ref.kind in {"context", "strategy"}:
                query = (
                    "SELECT project_id,context_sha256 AS sha256 FROM context_versions WHERE id=?"
                )
                if ref.kind == "strategy":
                    query = (
                        "SELECT c.project_id,s.fingerprint AS sha256 FROM strategy_versions s "
                        "JOIN context_versions c ON c.id=s.context_version_id WHERE s.id=?"
                    )
                row = self.connection.execute(query, (ref.id,)).fetchone()
                if row is None:
                    raise OrchestrationError("ARTIFACT_NOT_FOUND", "La referencia no existe")
                self.capabilities.require_scope(
                    task.capability_id, cycle_id=task.cycle_id, project_id=str(row["project_id"])
                )
                expected_hash = str(row["sha256"])
            else:
                table = {
                    "predictive_spec": "predictive_spec_versions",
                    "economic_scenario": "economic_scenario_versions",
                    "decision_value": "decision_value_spec_versions",
                    "value_policy": "policy_versions",
                }[ref.kind]
                row = self.connection.execute(
                    f"SELECT capability_id,sha256 FROM {table} WHERE id=?", (ref.id,)
                ).fetchone()
                if row is None or row["capability_id"] != task.capability_id:
                    raise OrchestrationError(
                        "ARTIFACT_SCOPE_MISMATCH", "La referencia sale de la capacidad"
                    )
                expected_hash = str(row["sha256"])
            if ref.sha256 is not None:
                if expected_hash is None:
                    raise OrchestrationError(
                        "ARTIFACT_HASH_NOT_SUPPORTED", "Esta referencia no define una huella única"
                    )
                if ref.sha256 != expected_hash:
                    raise OrchestrationError(
                        "ARTIFACT_HASH_MISMATCH", "La referencia no coincide con su huella"
                    )

    def _require_active_ancestors(self, task: AgentTaskV1) -> None:
        parent_id = task.parent_task_id
        while parent_id is not None:
            parent = self.get_task(parent_id)
            if parent.state not in {"PENDING", "RUNNING"}:
                raise OrchestrationError(
                    "ANCESTOR_NOT_RUNNABLE", "Una tarea superior está en espera o ya terminó"
                )
            parent_id = parent.parent_task_id

    def _event(
        self, task_id: str, event_type: str, actor: str, detail: dict[str, Any], instant: datetime
    ) -> None:
        self.connection.execute(
            "INSERT INTO orchestration_events("
            "id,task_id,event_type,actor_label,detail_json,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (str(uuid4()), task_id, event_type, actor, canonical_json(detail), instant.isoformat()),
        )
