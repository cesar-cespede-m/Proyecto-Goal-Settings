"""MV-09: estrategia determinista y ML tradicional bajo aprobación humana vigente."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from core.contracts.orchestration import ArtifactRefV1, ToolOperationV1, ToolResultV1
from core.db.repositories.orchestration import OrchestrationError, OrchestrationRepository
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash, canonical_json
from core.domain.capabilities import CapabilityService
from core.domain.data_readiness import DataReadinessService
from core.domain.errors import ApprovalRequiredError, DomainError
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.ml_engine.errors import MLValidationError
from core.ml_engine.execution import MLRunService
from core.ml_engine.splitting import split_dataset
from core.orchestration.data_tools import load_tool_source
from core.orchestration.tools import (
    DatasetArguments,
    RegisteredTool,
    RunArguments,
    ToolContext,
    ToolOutcome,
    _Arguments,
    _run,
    domain_operation_id,
)

MVP_TRAINING_SECONDS = 45
TRAINING_CLOSE_SECONDS = 30


class StrategyArguments(_Arguments):
    strategy_id: str = Field(min_length=1, max_length=100)


@dataclass(frozen=True)
class ModelPreflight:
    summary: dict[str, Any]
    output_refs: list[ArtifactRefV1]
    project_id: str
    proposal: StrategyProposalV1 | None = None


def _storage(context: ToolContext) -> str | Path:
    if context.storage_root is None:
        raise OrchestrationError("MISSING_STORAGE", "El modelado necesita storage configurado")
    return context.storage_root


def _temporal_template(
    workflow: WorkflowRepository, project_id: str, context_id: str, availability: str
) -> StrategyProposalV1:
    template = StrategyService(workflow).build_template(
        project_id=project_id, context_version_id=context_id
    )
    if template.task_type != "binary_classification":
        raise OrchestrationError("UNSUPPORTED_TASK_KIND", "Este agente admite objetivos binarios")
    payload = template.model_dump(mode="python")
    payload["split"] = {
        "method": "temporal",
        "time_column": "snapshot_cutoff",
        "label_available_at_column": availability,
        "train": 0.6,
        "validation": 0.2,
        "test": 0.2,
    }
    payload["caps"] = {"max_training_seconds": MVP_TRAINING_SECONDS, "max_parallel_jobs": 1}
    return StrategyProposalV1.model_validate(payload)


def inspect_model_preflight(context: ToolContext, dataset_id: str) -> ModelPreflight:
    """No ajusta modelos ni publica filas/distribuciones con valores de clase."""

    from core.domain.predictive_guard import require_capability_predictive_context

    storage = _storage(context)
    capabilities = CapabilityService(context.workflow)
    capabilities.require_scope(
        context.task.capability_id, cycle_id=context.task.cycle_id, dataset_id=dataset_id
    )
    dataset = context.workflow.get_dataset(dataset_id)
    project_id = str(dataset["project_id"])
    cycle = capabilities.get_cycle_for_project(project_id)
    assert cycle is not None
    readiness = DataReadinessService(context.workflow, storage).inspect_label_readiness(
        context.task.capability_id, dataset_id=dataset_id, cycle_id=cycle.id
    )
    summary: dict[str, Any] = {
        "kind": "model_viability",
        "status": readiness.status,
        "reason_codes": readiness.reason_codes,
        "measurement_plan": readiness.measurement_plan,
        "readiness_sha256": readiness.evidence_sha256,
        "split": None,
        "grants_training_approval": False,
        "limitations": ["NO_MODEL_FITTED", "HUMAN_APPROVAL_REQUIRED"],
    }
    refs = list(readiness.output_refs)
    if readiness.status != "ready":
        return ModelPreflight(summary, refs, project_id)
    try:
        predictive = require_capability_predictive_context(
            context.workflow, capability_id=context.task.capability_id, cycle_id=cycle.id
        )
        current = context.workflow.get_latest_context(project_id)
        if current is None or current["dataset_id"] != dataset_id:
            raise OrchestrationError(
                "CONTEXT_DATASET_MISMATCH", "Falta contexto vigente de la tabla"
            )
        recipe = predictive.spec.snapshot_recipe
        availability = None if recipe is None else recipe.label_available_at_output_column
        if availability is None:
            raise OrchestrationError(
                "LABEL_AVAILABILITY_NOT_DECLARED", "Falta disponibilidad explícita del resultado"
            )
        proposal = _temporal_template(
            context.workflow, project_id, str(current["id"]), availability
        )
        source = load_tool_source(context, dataset_id)
        splits = split_dataset(source.frame, proposal)
        split: dict[str, Any] = {"method": "temporal", "label_availability_verified": True}
        for name in ("train", "validation", "test"):
            part = getattr(splits.summary, name)
            positive = next(
                item.count
                for item in part.class_distribution
                if type(item.value) is type(proposal.positive_class)
                and item.value == proposal.positive_class
            )
            split[name] = {
                "row_count": part.row_count,
                "positive_support": positive,
                "negative_support": part.row_count - positive,
            }
        temporal = splits.summary.temporal
        assert temporal is not None
        split["purged_train_rows"] = temporal.purged_train_rows
        split["purged_validation_rows"] = temporal.purged_validation_rows
        summary.update(status="ready", reason_codes=[], split=split)
        refs.append(
            ArtifactRefV1(kind="context", id=current["id"], sha256=current["context_sha256"])
        )
        return ModelPreflight(summary, refs, project_id, proposal)
    except (DomainError, MLValidationError, ValueError) as error:
        code = getattr(error, "code", "MODEL_PREFLIGHT_INVALID")
        needs_data = isinstance(error, MLValidationError)
        summary.update(
            status="needs_data" if needs_data else "needs_human",
            reason_codes=[code],
            measurement_plan=[
                "Recolectar más casos maduros y comprobar soporte en cada partición temporal."
                if needs_data
                else "Revisar definición predictiva, contexto y exclusión de metadatos temporales."
            ],
        )
        return ModelPreflight(summary, refs, project_id)


def _viability(context: ToolContext, arguments: DatasetArguments) -> ToolOutcome:
    result = inspect_model_preflight(context, arguments.dataset_id)
    return ToolOutcome(
        ToolResultV1(status="COMPLETED", output_refs=result.output_refs, summary=result.summary)
    )


def _strategy_summary(strategy: Any, *, already_exists: bool = False) -> dict[str, Any]:
    proposal = StrategyProposalV1.model_validate_json(strategy["strategy_json"])
    return {
        "kind": "training_strategy",
        "version_no": int(strategy["version_no"]),
        "already_exists": already_exists,
        "candidates": proposal.candidates,
        "primary_metric": str(proposal.primary_metric),
        "split_method": proposal.split.method,
        "label_availability_required": proposal.split.label_available_at_column is not None,
        "max_training_seconds": proposal.caps.max_training_seconds,
        "grants_training_approval": False,
        "next_step": "REVIEW_CURRENT_HUMAN_APPROVAL",
    }


def _propose(context: ToolContext, arguments: DatasetArguments) -> ToolOutcome:
    preflight = inspect_model_preflight(context, arguments.dataset_id)
    if preflight.proposal is None:
        return ToolOutcome(
            ToolResultV1(
                status="COMPLETED", output_refs=preflight.output_refs, summary=preflight.summary
            )
        )
    proposal = preflight.proposal
    latest = context.workflow.get_latest_strategy(proposal.context_version_id)
    same = latest is not None and latest["strategy_json"] == canonical_json(proposal)
    if not same:
        version = StrategyService(context.workflow).create_version(
            project_id=preflight.project_id,
            proposal=proposal,
            operation_id=domain_operation_id(context.operation_id),
            actor_label=context.actor_label,
            verify_current=context.verify_current,
        )
        latest = context.workflow.get_strategy(version.strategy_version_id)
    assert latest is not None
    ref = ArtifactRefV1(kind="strategy", id=latest["id"], sha256=latest["fingerprint"])
    return ToolOutcome(
        ToolResultV1(
            status="COMPLETED",
            output_refs=[*preflight.output_refs, ref],
            summary=_strategy_summary(latest, already_exists=same),
        ),
        project_updates={}
        if same
        else {
            preflight.project_id: {
                "strategy": {"id": ref.id, "sha256": ref.sha256},
                "training_decision": None,
                "run": None,
            }
        },
    )


def _training_seconds(context: ToolContext, arguments: StrategyArguments) -> int:
    strategy = context.workflow.get_strategy(arguments.strategy_id)
    proposal = StrategyProposalV1.model_validate_json(strategy["strategy_json"])
    return proposal.caps.max_training_seconds + TRAINING_CLOSE_SECONDS


def _training_result(context: ToolContext, run_id: str) -> ToolResultV1:
    result = _run(context, RunArguments(run_id=run_id)).result
    return result.model_copy(
        update={
            "summary": {
                "kind": "model_training",
                **result.summary,
                "published_by_this_tool": False,
                "limitations": [
                    "HOLDOUT_NOT_USED_FOR_POLICY_SELECTION",
                    "PUBLICATION_REQUIRES_HITL",
                ],
            }
        }
    )


def _train(context: ToolContext, arguments: StrategyArguments) -> ToolOutcome:
    existing_operation = context.workflow.connection.execute(
        "SELECT run_id FROM training_operations WHERE operation_id=?",
        (domain_operation_id(context.operation_id),),
    ).fetchone()
    latest = context.workflow.get_latest_run(arguments.strategy_id)
    if (
        existing_operation is None
        and latest is not None
        and latest["status"] in {"COMPLETED", "COMPLETED_WITHOUT_RECOMMENDATION"}
    ):
        raise OrchestrationError(
            "MODEL_ALREADY_TRAINED", "Consulta el resultado existente antes de abrir otro ciclo"
        )
    try:
        run = MLRunService(context.workflow, _storage(context)).execute(
            arguments.strategy_id,
            operation_id=domain_operation_id(context.operation_id),
            verify_start=context.verify_current,
        )
    except ApprovalRequiredError as error:
        raise OrchestrationError(
            "STRATEGY_NOT_APPROVED", "La estrategia requiere aprobación humana vigente"
        ) from error
    strategy = context.workflow.get_strategy(arguments.strategy_id)
    parent = context.workflow.get_context(str(strategy["context_version_id"]))
    return ToolOutcome(
        _training_result(context, run.run_id),
        project_updates={
            str(parent["project_id"]): {"run": {"id": run.run_id, "status": run.status}}
        },
    )


def recover_modeling_result(
    workflow: WorkflowRepository, storage_root: str | Path | None, operation: ToolOperationV1
) -> ToolResultV1 | None:
    """Reconstruye efectos acreditados; nunca vuelve a entrenar ni a crear estrategias."""

    task = OrchestrationRepository(workflow).get_task(operation.task_id)
    domain_op = domain_operation_id(operation.operation_id)
    if operation.call.tool_name == "propose_training_strategy":
        row = workflow.connection.execute(
            "SELECT entity_id,detail_json FROM audit_events "
            "WHERE event_type='STRATEGY_VERSION_CREATED' "
            "AND json_extract(detail_json,'$.operation_id')=?",
            (domain_op,),
        ).fetchone()
        if row is None:
            return None
        detail = json.loads(row["detail_json"])
        strategy = workflow.get_strategy(str(row["entity_id"]))
        arguments = DatasetArguments.model_validate(operation.call.arguments)
        CapabilityService(workflow).require_scope(
            task.capability_id, cycle_id=task.cycle_id, dataset_id=str(strategy["dataset_id"])
        )
        request = {
            "context_version_id": strategy["context_version_id"],
            "dataset_id": strategy["dataset_id"],
            "version_no": None,
            "strategy_json": strategy["strategy_json"],
            "schema_version": strategy["schema_version"],
            "source": strategy["source"],
            "actor_label": detail.get("actor_label"),
            "fingerprint": strategy["fingerprint"],
        }
        original = {
            "task_id": task.id,
            "call": operation.call.model_dump(mode="json"),
            "actor_label": detail.get("actor_label"),
        }
        if (
            arguments.dataset_id != strategy["dataset_id"]
            or detail.get("request_sha256") != canonical_hash(request)
            or operation.request_sha256 != canonical_hash(original)
        ):
            return None
        # La llamada sólo autorizó la plantilla cerrada. Una colisión manual del ID de dominio
        # con otro payload no puede convertirse en un éxito falso durante la recuperación.
        from core.domain.predictive_guard import require_capability_predictive_context

        parent = workflow.get_context(str(strategy["context_version_id"]))
        cycle = CapabilityService(workflow).get_cycle_for_project(str(parent["project_id"]))
        assert cycle is not None
        predictive = require_capability_predictive_context(
            workflow, capability_id=task.capability_id, cycle_id=cycle.id
        )
        recipe = predictive.spec.snapshot_recipe
        if recipe is None or recipe.label_available_at_output_column is None:
            return None
        expected = _temporal_template(
            workflow,
            str(parent["project_id"]),
            str(parent["id"]),
            recipe.label_available_at_output_column,
        )
        if canonical_json(expected) != strategy["strategy_json"]:
            return None
        return ToolResultV1(
            status="COMPLETED",
            output_refs=[
                ArtifactRefV1(kind="strategy", id=strategy["id"], sha256=strategy["fingerprint"])
            ],
            summary={**_strategy_summary(strategy), "recovered_from_persisted_effect": True},
        )
    if operation.call.tool_name != "train_model" or storage_root is None:
        return None
    arguments = StrategyArguments.model_validate(operation.call.arguments)
    mapping = workflow.connection.execute(
        "SELECT strategy_version_id,run_id FROM training_operations WHERE operation_id=?",
        (domain_op,),
    ).fetchone()
    if mapping is None or mapping["strategy_version_id"] != arguments.strategy_id:
        return None
    run_id = str(mapping["run_id"])
    CapabilityService(workflow).require_scope(
        task.capability_id, cycle_id=task.cycle_id, run_id=run_id
    )
    run = workflow.get_run(run_id)
    if run["status"] in {"FAILED", "BLOCKED"}:
        return ToolResultV1(
            status="FAILED",
            output_refs=[ArtifactRefV1(kind="run", id=run_id)],
            summary={"kind": "model_training", "status": run["status"]},
            error_code="TRAINING_OPERATION_TERMINATED",
        )
    if run["status"] not in {"COMPLETED", "COMPLETED_WITHOUT_RECOMMENDATION"}:
        return None
    # Este lector verifica la evidencia/artefacto persistido, sin ejecutar fit ni un nuevo gate.
    MLRunService(workflow, storage_root)._existing_result(run_id)
    context = ToolContext(workflow, task, task.actor_label, operation.operation_id, storage_root)
    result = _training_result(context, run_id)
    return result.model_copy(
        update={"summary": {**result.summary, "recovered_from_persisted_effect": True}}
    )


def modeling_tools() -> tuple[RegisteredTool, ...]:
    roles = frozenset({"modeling"})
    return (
        RegisteredTool("inspect_model_viability", roles, DatasetArguments, _viability),
        RegisteredTool("propose_training_strategy", roles, DatasetArguments, _propose, writes=True),
        RegisteredTool(
            "train_model",
            roles,
            StrategyArguments,
            _train,
            writes=True,
            required_seconds=_training_seconds,
        ),
    )
