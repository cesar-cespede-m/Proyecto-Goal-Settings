"""MV-06: supervisor persistente, sin ejecución libre ni llamadas a proveedores."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

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
from core.db.repositories.orchestration import OrchestrationError, OrchestrationRepository
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.domain.capabilities import CapabilityService
from core.domain.decision_value import DecisionValueService
from core.domain.errors import DomainError
from core.domain.value_policy import ValuePolicyService
from core.orchestration.tools import (
    DecisionDraftArguments,
    ToolContext,
    domain_operation_id,
    registered_tools,
    version_identity,
)


class Supervisor:
    def __init__(
        self,
        workflow: WorkflowRepository,
        *,
        clock: Callable[[], datetime] | None = None,
        storage_root: str | Path | None = None,
    ) -> None:
        self.workflow = workflow
        self.storage_root = storage_root
        self.repository = OrchestrationRepository(workflow)
        self.capabilities = CapabilityService(workflow)
        self.values = DecisionValueService(workflow)
        self.policies = ValuePolicyService(workflow)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.owner = str(uuid4())
        self._tools = registered_tools()

    def create_task(
        self,
        *,
        capability_id: str,
        role: AgentRole,
        actor_label: str,
        operation_id: str,
        cycle_id: str | None = None,
        parent_task_id: str | None = None,
        input_refs: Sequence[ArtifactRefV1 | dict[str, Any]] = (),
        budget: AgentBudgetV1 | None = None,
        objective: str | None = None,
        parent_expected_revision: int | None = None,
        verify_current: Callable[[], str] | None = None,
    ) -> AgentTaskV1:
        refs = [ArtifactRefV1.model_validate(ref) for ref in input_refs]
        try:
            if parent_task_id is not None:
                parent = self.repository.get_task(parent_task_id)
                if parent.role != "coordinator" or role == "coordinator":
                    raise OrchestrationError(
                        "DELEGATION_ROLE_DENIED",
                        "El coordinador puede delegar únicamente en los cuatro especialistas",
                    )
            snapshot = self._snapshot(capability_id, cycle_id, refs)
            return self.repository.create_task(
                capability_id=capability_id,
                cycle_id=cycle_id,
                role=role,
                context_sha256=canonical_hash(snapshot),
                actor_label=actor_label,
                operation_id=operation_id,
                parent_task_id=parent_task_id,
                input_refs=refs,
                budget=budget,
                objective=objective,
                now=self.clock(),
                parent_expected_revision=parent_expected_revision,
                verify_current=verify_current,
            )
        except DomainError as error:
            raise self._domain_error(error) from error

    def execute_tool(
        self,
        task_id: str,
        call: ToolCallV1 | dict[str, Any],
        *,
        operation_id: str,
        expected_revision: int,
        actor_label: str,
        lease_seconds: int | None = None,
    ) -> ToolResultV1:
        """Ejecuta un paso autorizado; un replay exacto devuelve su resultado persistido."""

        task = self.repository.get_task(task_id)
        try:
            parsed = ToolCallV1.model_validate(
                call.model_dump(mode="python") if isinstance(call, ToolCallV1) else call
            )
        except ValidationError as error:
            raise OrchestrationError(
                "INVALID_TOOL_CALL", "La llamada no cumple su contrato"
            ) from error
        tool = self._tools.get(parsed.tool_name)
        if tool is None:
            raise OrchestrationError("TOOL_NOT_ALLOWED", "La herramienta no está habilitada")
        if task.role not in tool.roles:
            raise OrchestrationError(
                "TOOL_ROLE_DENIED", "El rol no puede ejecutar esta herramienta"
            )
        try:
            arguments = tool.arguments.model_validate(parsed.arguments)
        except ValidationError as error:
            raise OrchestrationError(
                "INVALID_TOOL_ARGUMENTS", "Revisa los argumentos permitidos"
            ) from error
        if isinstance(arguments, DecisionDraftArguments) and arguments.spec.owner_label is not None:
            raise OrchestrationError(
                "TOOL_HUMAN_FIELD_FORBIDDEN", "El responsable debe declararlo una persona"
            )
        normalized = ToolCallV1(
            tool_name=parsed.tool_name, arguments=arguments.model_dump(mode="json")
        )
        needed_seconds = 0
        if tool.required_seconds is not None:
            try:
                self._check_arguments_scope(task, arguments)
                needed_seconds = tool.required_seconds(
                    ToolContext(self.workflow, task, actor_label, operation_id, self.storage_root),
                    arguments,
                )
            except DomainError as error:
                raise self._domain_error(error) from error
        selected_lease = max(60, needed_seconds) if lease_seconds is None else lease_seconds
        before: dict[str, Any] = {}

        def verify_before() -> str:
            nonlocal before
            self._check_arguments_scope(task, arguments)
            current = self.repository.get_task(task.id)
            remaining = current.budget.max_seconds * 1000 - current.elapsed_ms - current.reserved_ms
            if needed_seconds and (
                selected_lease < needed_seconds or remaining < needed_seconds * 1000
            ):
                raise OrchestrationError(
                    "BUDGET_INSUFFICIENT_FOR_TRAINING",
                    "El presupuesto no cubre el límite de entrenamiento y su cierre",
                )
            before = self._snapshot(task.capability_id, task.cycle_id, task.input_refs)
            return canonical_hash(before)

        try:
            operation = self.repository.claim_operation(
                task_id,
                normalized,
                operation_id=operation_id,
                expected_revision=expected_revision,
                owner=self.owner,
                actor_label=actor_label,
                lease_seconds=selected_lease,
                calls_cost=0,
                now=self.clock(),
                verify_current=verify_before,
            )
        except DomainError as error:
            raise self._domain_error(error) from error
        if operation.status != "CLAIMED":
            assert operation.result is not None
            return operation.result
        operation_id = operation.operation_id

        def verify_write() -> None:
            current = self._snapshot(task.capability_id, task.cycle_id, task.input_refs)
            if canonical_hash(current) != canonical_hash(before):
                raise OrchestrationError(
                    "STALE_CONTEXT", "El contexto cambió antes de registrar el efecto"
                )

        try:
            outcome = tool.execute(
                ToolContext(
                    self.workflow,
                    task,
                    actor_label,
                    operation_id,
                    self.storage_root,
                    verify_write,
                ),
                arguments,
            )
        except Exception as error:
            # Nunca se reejecuta el efecto: sólo se busca una confirmación ya persistida.
            recovered = self._recover_result(operation)
            result = recovered or ToolResultV1(status="FAILED", error_code=self._error_code(error))
            finished = self.repository.finish_operation(
                operation_id,
                expected_revision=operation.task_revision,
                owner=self.owner,
                result=result,
                next_state="WAITING_HUMAN",
                now=self.clock(),
            )
            assert finished.result is not None
            return finished.result

        expected = {**deepcopy(before), **outcome.context_updates}
        for project in expected["projects"]:
            changes = outcome.project_updates.get(project["project_id"], {})
            if set(changes) - {"strategy", "training_decision", "run"}:
                raise OrchestrationError("INVALID_TOOL_EFFECT", "Cambio de dominio no permitido")
            project.update(changes)
        expected_hash = canonical_hash(expected)

        def verify_after() -> str:
            for reference in outcome.result.output_refs:
                self._reference(task.capability_id, task.cycle_id, reference)
            return canonical_hash(
                self._snapshot(task.capability_id, task.cycle_id, task.input_refs)
            )

        try:
            finished = self.repository.finish_operation(
                operation_id,
                expected_revision=operation.task_revision,
                owner=self.owner,
                result=outcome.result,
                next_state="RUNNING",
                now=self.clock(),
                verify_current=verify_after,
                expected_context_sha256=expected_hash,
            )
        except DomainError as error:
            if getattr(error, "code", None) != "STALE_CONTEXT":
                raise self._domain_error(error) from error
            result = ToolResultV1(
                status="FAILED",
                output_refs=outcome.result.output_refs,
                summary=outcome.result.summary,
                error_code="EFFECT_RECORDED_CONTEXT_CHANGED" if tool.writes else "STALE_CONTEXT",
            )
            finished = self.repository.finish_operation(
                operation_id,
                expected_revision=operation.task_revision,
                owner=self.owner,
                result=result,
                next_state="WAITING_HUMAN",
                now=self.clock(),
            )
        assert finished.result is not None
        return finished.result

    def prepare_turn(
        self, task_id: str, *, expected_revision: int, actor_label: str
    ) -> AgentTaskV1:
        """Captura contexto antes del razonamiento, sin aceptar respuestas anteriores."""

        task = self.repository.get_task(task_id)

        def current() -> str:
            return canonical_hash(
                self._snapshot(task.capability_id, task.cycle_id, task.input_refs)
            )

        return self.repository.refresh_task_context(
            task_id,
            expected_revision=expected_revision,
            context_sha256=current(),
            actor_label=actor_label,
            verify_current=current,
            now=self.clock(),
        )

    def context_fingerprint(self, task_id: str) -> str:
        """Huella vigente de dominio; los agentes añaden su contexto conversacional."""

        task = self.repository.get_task(task_id)
        return canonical_hash(self._snapshot(task.capability_id, task.cycle_id, task.input_refs))

    def transition_task(
        self,
        task_id: str,
        *,
        expected_revision: int,
        new_state: AgentTaskState,
        actor_label: str,
        reason_code: str | None = None,
        refresh_context: bool = False,
    ) -> AgentTaskV1:
        task = self.repository.get_task(task_id)
        context = None
        if refresh_context:
            if task.state not in {"WAITING_HUMAN", "WAITING_DATA"} or new_state != "PENDING":
                raise OrchestrationError(
                    "INVALID_CONTEXT_REFRESH", "Reanuda la tarea pendiente de revisión"
                )
            context = canonical_hash(
                self._snapshot(task.capability_id, task.cycle_id, task.input_refs)
            )
        return self.repository.transition_task(
            task_id,
            expected_revision=expected_revision,
            new_state=new_state,
            actor_label=actor_label,
            reason_code=reason_code,
            context_sha256=context,
            now=self.clock(),
        )

    def recover_expired_operations(
        self, *, now: datetime | None = None, root_task_id: str | None = None
    ) -> list[str]:
        return self.repository.recover_expired_claims(
            now=now or self.clock(),
            recover_result=self._recover_result,
            root_task_id=root_task_id,
        )

    def _recover_result(self, operation: ToolOperationV1) -> ToolResultV1 | None:
        if operation.call.tool_name in {"propose_training_strategy", "train_model"}:
            from core.orchestration.modeling_tools import recover_modeling_result

            try:
                return recover_modeling_result(self.workflow, self.storage_root, operation)
            except Exception:
                # La recuperación sólo acredita efectos íntegros, jamás reintenta el trabajo.
                return None
        if operation.call.tool_name == "propose_value_policy":
            from core.orchestration.value_tools import recover_value_policy_result

            try:
                return recover_value_policy_result(self.workflow, operation)
            except Exception:
                return None
        if operation.call.tool_name != "save_decision_draft":
            return None
        row = self.workflow.connection.execute(
            "SELECT entity_id,request_sha256 FROM value_spec_events WHERE operation_id=? "
            "AND event_type='DECISION_VALUE_SPEC_CREATED'",
            (domain_operation_id(operation.operation_id),),
        ).fetchone()
        if row is None:
            return None
        version = self.values.get_decision_version(str(row["entity_id"]))
        task = self.repository.get_task(operation.task_id)
        arguments = DecisionDraftArguments.model_validate(operation.call.arguments)
        owner = None
        if arguments.based_on_version_id is not None:
            previous = self.values.get_decision_version(arguments.based_on_version_id)
            if previous.capability_id != task.capability_id:
                return None
            owner = previous.spec.owner_label
        expected = {**arguments.spec.model_dump(mode="json"), "owner_label": owner}
        expected_request = {
            "capability_id": task.capability_id,
            "content": expected,
            "based_on_version_id": arguments.based_on_version_id,
            "confirmed_human": False,
            "actor_label": version.actor_label,
        }
        original_request = {
            "task_id": task.id,
            "call": operation.call.model_dump(mode="json"),
            "actor_label": version.actor_label,
        }
        if (
            version.capability_id != task.capability_id
            or version.confirmed_human
            or arguments.spec.owner_label is not None
            or version.sha256 != canonical_hash(expected)
            or row["request_sha256"] != canonical_hash(expected_request)
            or operation.request_sha256 != canonical_hash(original_request)
        ):
            return None
        return ToolResultV1(
            status="COMPLETED",
            output_refs=[
                ArtifactRefV1(kind="decision_value", id=version.id, sha256=version.sha256)
            ],
            summary={
                "version_no": version.version_no,
                "confirmed_human": version.confirmed_human,
                "recovered_from_persisted_effect": True,
            },
        )

    def _check_arguments_scope(self, task: AgentTaskV1, arguments: Any) -> None:
        for name in ("dataset_id", "run_id"):
            identifier = getattr(arguments, name, None)
            if identifier is not None:
                self.capabilities.require_scope(
                    task.capability_id, cycle_id=task.cycle_id, **{name: identifier}
                )
        strategy_id = getattr(arguments, "strategy_id", None)
        if strategy_id is not None:
            self._reference(
                task.capability_id,
                task.cycle_id,
                ArtifactRefV1(kind="strategy", id=strategy_id),
            )
        economic_id = getattr(arguments, "economic_version_id", None)
        if economic_id is not None:
            self._reference(
                task.capability_id,
                task.cycle_id,
                ArtifactRefV1(kind="economic_scenario", id=economic_id),
            )
        if isinstance(arguments, DecisionDraftArguments):
            for kind, identifier in (
                ("decision_value", arguments.based_on_version_id),
                ("predictive_spec", arguments.spec.predictive_version_id),
                ("economic_scenario", arguments.spec.economic_version_id),
            ):
                if identifier is not None:
                    self._reference(
                        task.capability_id, task.cycle_id, ArtifactRefV1(kind=kind, id=identifier)
                    )

    def _snapshot(
        self, capability_id: str, cycle_id: str | None, refs: Sequence[ArtifactRefV1]
    ) -> dict[str, Any]:
        self.capabilities.require_scope(capability_id, cycle_id=cycle_id)
        capability = self.capabilities.get_capability(capability_id)
        product = self.capabilities.get_product(capability.product_id)
        snapshot: dict[str, Any] = {
            "capability": canonical_hash(capability),
            "product": canonical_hash(product),
            "input_refs": [self._reference(capability_id, cycle_id, ref) for ref in refs],
            "projects": [],
        }
        for field_name, versions in (
            ("decision_version", self.values.list_decision_versions(capability_id)),
            ("predictive_version", self.values.list_predictive_versions(capability_id)),
            ("economic_version", self.values.list_economic_versions(capability_id)),
        ):
            snapshot[field_name] = version_identity(versions[-1]) if versions else None
        policies = self.policies.list_versions(capability_id)
        snapshot["policy_version"] = (
            {"id": policies[-1].id, "sha256": policies[-1].sha256} if policies else None
        )
        snapshot["policy_review"] = (
            canonical_hash(self.policies.latest_review(policies[-1].id)) if policies else None
        )
        cycles = (
            [self.capabilities.get_cycle(cycle_id)]
            if cycle_id is not None
            else self.capabilities.list_cycles(capability_id)
        )
        for cycle in cycles:
            project = self.workflow.connection.execute(
                "SELECT status FROM projects WHERE id=?", (cycle.project_id,)
            ).fetchone()
            entry: dict[str, Any] = {
                "cycle_id": cycle.id,
                "project_id": cycle.project_id,
                "status": project["status"],
                "datasets": [
                    {
                        "id": row["id"],
                        "sha256": row["sha256"],
                        "admission_status": row["admission_status"],
                        "is_active": row["is_active"],
                        "metadata_sha256": canonical_hash(
                            {
                                "profile": row["profile_json"],
                                "path": row["stored_path"],
                                "filename": row["original_filename_display"],
                            }
                        ),
                    }
                    for row in self.workflow.connection.execute(
                        "SELECT id,sha256,admission_status,is_active,profile_json,stored_path,"
                        "original_filename_display FROM datasets "
                        "WHERE project_id=? ORDER BY id",
                        (cycle.project_id,),
                    ).fetchall()
                ],
                "snapshot_evidence_sha256": canonical_hash(
                    [
                        dict(row)
                        for row in self.workflow.connection.execute(
                            "SELECT entity_id,event_type,detail_json FROM audit_events "
                            "WHERE project_id=? AND event_type IN "
                            "('SNAPSHOT_TABLE_SAVED','SNAPSHOT_READINESS_RECORDED') "
                            "ORDER BY rowid",
                            (cycle.project_id,),
                        ).fetchall()
                    ]
                ),
                "context": None,
                "strategy": None,
                "training_decision": None,
                "run": None,
                "binding": None,
            }
            binding = self.values.get_cycle_binding(cycle.id)
            if binding is not None:
                entry["binding"] = {
                    "id": binding.id,
                    "predictive_version_id": binding.predictive_version_id,
                }
            context = self.workflow.get_latest_context(cycle.project_id)
            if context is not None:
                entry["context"] = {"id": context["id"], "sha256": context["context_sha256"]}
                strategy = self.workflow.get_latest_strategy(str(context["id"]))
                if strategy is not None:
                    entry["strategy"] = {"id": strategy["id"], "sha256": strategy["fingerprint"]}
                    approval = self.workflow.get_latest_training_decision(str(strategy["id"]))
                    if approval is not None:
                        entry["training_decision"] = {
                            "id": approval["id"],
                            "decision": approval["decision"],
                        }
                    run = self.workflow.get_latest_run(str(strategy["id"]))
                    if run is not None:
                        entry["run"] = {"id": run["id"], "status": run["status"]}
            snapshot["projects"].append(entry)
        return snapshot

    def _reference(
        self, capability_id: str, cycle_id: str | None, reference: ArtifactRefV1
    ) -> dict[str, Any]:
        kind, identifier = reference.kind, reference.id
        if kind == "capability":
            if identifier != capability_id:
                raise OrchestrationError(
                    "ARTIFACT_SCOPE_MISMATCH", "La referencia sale de la capacidad"
                )
            digest = canonical_hash(self.capabilities.get_capability(identifier))
        elif kind == "cycle":
            cycle = self.capabilities.get_cycle(identifier)
            self.capabilities.require_scope(
                capability_id, cycle_id=cycle_id, project_id=cycle.project_id
            )
            digest = canonical_hash(cycle)
        elif kind == "value_policy":
            version = self.policies.get(identifier)
            if version.capability_id != capability_id or (
                cycle_id is not None and version.content.cycle_id != cycle_id
            ):
                raise OrchestrationError("ARTIFACT_SCOPE_MISMATCH", "La política sale del ciclo")
            digest = version.sha256
        elif kind in {"predictive_spec", "economic_scenario", "decision_value"}:
            getter = {
                "predictive_spec": self.values.get_predictive_version,
                "economic_scenario": self.values.get_economic_version,
                "decision_value": self.values.get_decision_version,
            }[kind]
            version = getter(identifier)
            if version.capability_id != capability_id:
                raise OrchestrationError(
                    "ARTIFACT_SCOPE_MISMATCH", "La referencia sale de la capacidad"
                )
            digest = version.sha256
        else:
            queries = {
                "project": "SELECT id AS project_id,status FROM projects WHERE id=?",
                "dataset": "SELECT project_id,sha256 FROM datasets WHERE id=?",
                "context": "SELECT project_id,context_sha256 AS sha256 "
                "FROM context_versions WHERE id=?",
                "strategy": "SELECT c.project_id,s.fingerprint AS sha256 FROM strategy_versions s "
                "JOIN context_versions c ON c.id=s.context_version_id WHERE s.id=?",
                "run": "SELECT c.project_id,r.status,r.manifest_json FROM runs r "
                "JOIN strategy_versions s ON s.id=r.strategy_version_id "
                "JOIN context_versions c ON c.id=s.context_version_id WHERE r.id=?",
            }
            row = self.workflow.connection.execute(queries[kind], (identifier,)).fetchone()
            if row is None:
                raise OrchestrationError("ARTIFACT_NOT_FOUND", "La referencia no existe")
            self.capabilities.require_scope(
                capability_id, cycle_id=cycle_id, project_id=row["project_id"]
            )
            if kind in {"project", "run"} and reference.sha256 is not None:
                raise OrchestrationError(
                    "ARTIFACT_HASH_UNSUPPORTED",
                    "Esta referencia usa ID y no declara una huella única",
                )
            digest = row["sha256"] if "sha256" in row.keys() else canonical_hash(dict(row))
        if reference.sha256 is not None and reference.sha256 != digest:
            raise OrchestrationError(
                "ARTIFACT_HASH_MISMATCH", "La referencia cambió desde su lectura"
            )
        return {"kind": kind, "id": identifier, "sha256": digest}

    @staticmethod
    def _error_code(error: Exception) -> str:
        code = getattr(error, "code", None)
        return (
            code
            if isinstance(code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", code)
            else "TOOL_EXECUTION_FAILED"
        )

    @classmethod
    def _domain_error(cls, error: DomainError) -> OrchestrationError:
        if isinstance(error, OrchestrationError):
            return error
        return OrchestrationError(
            cls._error_code(error), "Revisa la pertenencia y vigencia de las referencias"
        )
