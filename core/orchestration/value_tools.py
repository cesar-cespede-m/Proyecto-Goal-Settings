"""El agente propone una política; una persona decide su aprobación."""

from __future__ import annotations

from pydantic import Field

from core.contracts.orchestration import ArtifactRefV1, ToolOperationV1, ToolResultV1
from core.db.repositories.orchestration import OrchestrationRepository
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.domain.value_policy import ValuePolicyService
from core.orchestration.tools import (
    RegisteredTool,
    ToolContext,
    ToolOutcome,
    _Arguments,
    domain_operation_id,
)


class ValuePolicyArguments(_Arguments):
    run_id: str = Field(min_length=1, max_length=100)
    economic_version_id: str = Field(min_length=1, max_length=100)
    based_on_version_id: str | None = Field(default=None, min_length=1, max_length=100)


def _result(version, *, recovered: bool = False) -> ToolResultV1:
    return ToolResultV1(
        status="COMPLETED",
        output_refs=[ArtifactRefV1(kind="value_policy", id=version.id, sha256=version.sha256)],
        summary={
            "kind": "value_policy",
            "version_no": version.version_no,
            "evidence_partition": "validation",
            "evidence_status": "exploratory",
            "human_review_required": True,
            "published_by_this_tool": False,
            "recovered_from_persisted_effect": recovered,
        },
    )


def _propose(context: ToolContext, arguments: ValuePolicyArguments) -> ToolOutcome:
    if context.task.cycle_id is None:
        raise ValueError("Selecciona un ciclo antes de proponer una política")
    version = ValuePolicyService(context.workflow).propose(
        context.task.capability_id,
        cycle_id=context.task.cycle_id,
        run_id=arguments.run_id,
        economic_version_id=arguments.economic_version_id,
        based_on_version_id=arguments.based_on_version_id,
        actor_label=context.actor_label,
        operation_id=domain_operation_id(context.operation_id),
        verify_current=context.verify_current,
    )
    return ToolOutcome(
        _result(version),
        context_updates={
            "policy_version": {"id": version.id, "sha256": version.sha256},
            "policy_review": canonical_hash(None),
        },
    )


def recover_value_policy_result(
    workflow: WorkflowRepository, operation: ToolOperationV1
) -> ToolResultV1 | None:
    row = workflow.connection.execute(
        "SELECT entity_id,request_sha256,actor_label FROM policy_operations "
        "WHERE operation_id=? AND event_type='POLICY_PROPOSED'",
        (domain_operation_id(operation.operation_id),),
    ).fetchone()
    if row is None:
        return None
    task = OrchestrationRepository(workflow).get_task(operation.task_id)
    args = ValuePolicyArguments.model_validate(operation.call.arguments)
    version = ValuePolicyService(workflow).get(str(row["entity_id"]))
    request = {
        "capability_id": task.capability_id,
        "cycle_id": task.cycle_id,
        "run_id": args.run_id,
        "economic_version_id": args.economic_version_id,
        "based_on_version_id": args.based_on_version_id,
        "actor_label": row["actor_label"],
    }
    original = {
        "task_id": task.id,
        "call": operation.call.model_dump(mode="json"),
        "actor_label": row["actor_label"],
    }
    if (
        version.capability_id != task.capability_id
        or version.content.cycle_id != task.cycle_id
        or version.content.run_id != args.run_id
        or version.content.economic_version_id != args.economic_version_id
        or version.based_on_version_id != args.based_on_version_id
        or row["request_sha256"] != canonical_hash(request)
        or operation.request_sha256 != canonical_hash(original)
    ):
        return None
    return _result(version, recovered=True)


def value_tools() -> tuple[RegisteredTool, ...]:
    return (
        RegisteredTool(
            "propose_value_policy",
            frozenset({"value_evidence"}),
            ValuePolicyArguments,
            _propose,
            writes=True,
        ),
    )
