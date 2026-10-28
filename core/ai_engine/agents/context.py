"""Contextos por rol: definiciones revisadas y evidencia agregada, sin filas ni secretos."""

from __future__ import annotations

import re
from typing import Any

from core.ai_engine.outgoing_review import redact
from core.data_engine.models import DatasetProfileV1
from core.db.repositories.orchestration import OrchestrationRepository
from core.db.repositories.workflow import WorkflowRepository
from core.domain.assistant import SafeProfileForLLMV1
from core.domain.capabilities import CapabilityService
from core.domain.decision_value import DecisionValueService

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
ACTIVE_ROLES = frozenset({"coordinator", "business", "data", "modeling", "value_evidence"})
ROLE_TOOLS = {
    "coordinator": ("read_capability_context",),
    "business": (
        "read_capability_context",
        "save_decision_draft",
        "evaluate_economic_scenario",
    ),
    "data": (
        "read_capability_context",
        "read_dataset_profile",
        "inspect_data_quality",
        "inspect_label_readiness",
        "aggregate_analysis",
    ),
    "modeling": (
        "read_capability_context",
        "inspect_model_viability",
        "propose_training_strategy",
        "train_model",
        "read_run_summary",
    ),
    "value_evidence": (
        "read_capability_context",
        "read_run_summary",
        "evaluate_economic_scenario",
        "propose_value_policy",
    ),
}


def clean_text(value: str | None, *, limit: int = 1500) -> str | None:
    if value is None:
        return None
    return redact(_CONTROL.sub(" ", value))[0][:limit]


def build_context(workflow: WorkflowRepository, task_id: str) -> dict[str, Any]:
    """Payload exacto de un turno, utilizable también en la revisión previa al envío.

    Los identificadores permiten citar artefactos internos. Se omiten campos de actor,
    responsable, valores de clase, archivos, rutas y contenido tabular. El texto libre
    se redacta por patrones y requiere revisión humana antes de su envío.
    """

    tasks = OrchestrationRepository(workflow)
    task = tasks.get_task(task_id)
    caps = CapabilityService(workflow)
    capability = caps.get_capability(task.capability_id)
    values = DecisionValueService(workflow)
    refs: dict[tuple[str, str], dict[str, Any]] = {}

    def add_reference(kind: str, identifier: str, sha256: str | None = None) -> None:
        refs[(kind, identifier)] = {"kind": kind, "id": identifier, "sha256": sha256}

    add_reference("capability", capability.id)
    if task.cycle_id:
        add_reference("cycle", task.cycle_id)
    context: dict[str, Any] = {
        "schema_version": "agent-context.v1",
        "role": task.role,
        "objective": clean_text(task.objective),
        "business": {
            "name": clean_text(capability.name),
            "objective": clean_text(capability.objective),
            "value_kind": capability.value_kind,
            "decision_unit": clean_text(capability.decision_unit),
        },
        "decision": None,
        "predictive_definition": None,
        "economics": None,
        "allowed_tools": list(ROLE_TOOLS.get(task.role, ())),
        "allowed_delegates": ["business", "data", "modeling", "value_evidence"]
        if task.role == "coordinator"
        else [],
        "datasets": [],
        "modeling_context": [],
        "children": [],
        "tool_evidence": [],
        "limitations": [
            "LLM_OUTPUT_IS_A_PROPOSAL",
            "HUMAN_CONFIRMATION_REQUIRED",
            "NO_PRODUCTION_DATA",
            "NO_METRICS_OR_ECONOMIC_VALUES_GENERATED_BY_LLM",
        ],
    }
    if task.role in {"coordinator", "data", "modeling", "value_evidence"}:
        from core.orchestration.data_tools import safe_profile

        cycles = (
            [caps.get_cycle(task.cycle_id)] if task.cycle_id else caps.list_cycles(capability.id)
        )
        datasets = [
            dataset
            for cycle in cycles
            for dataset in workflow.list_datasets(cycle.project_id, admission_status="ACCEPTED")
        ]
        datasets.sort(key=lambda row: (row["created_at"], row["id"]), reverse=True)
        for dataset in datasets[:20]:
            add_reference("dataset", dataset["id"], dataset["sha256"])
            item: dict[str, Any] = {
                "id": dataset["id"],
                "is_active": bool(dataset["is_active"]),
                "metadata_status": "REGISTERED_PROFILE_NOT_CURRENT_INSPECTION",
            }
            if task.role == "data":
                try:
                    item["profile"] = safe_profile(
                        DatasetProfileV1.model_validate_json(dataset["profile_json"])
                    )
                except ValueError:
                    item["profile"] = None
                    item["profile_status"] = "INVALID_DATASET_PROFILE"
            context["datasets"].append(item)
        if len(datasets) > 20:
            context["limitations"].append("DATASET_CATALOG_TRUNCATED_SELECT_A_CYCLE")
        if task.role in {"coordinator", "modeling", "value_evidence"}:
            for cycle in cycles[-20:]:
                current = workflow.get_latest_context(cycle.project_id)
                if current is None:
                    continue
                add_reference("context", current["id"], current["context_sha256"])
                item = {
                    "dataset_id": current["dataset_id"],
                    "context_id": current["id"],
                    "strategy_id": None,
                    "training_approval": "MISSING",
                    "run_id": None,
                    "run_status": None,
                }
                strategy = workflow.get_latest_strategy(str(current["id"]))
                if strategy is not None:
                    item["strategy_id"] = strategy["id"]
                    add_reference("strategy", strategy["id"], strategy["fingerprint"])
                    decision = workflow.get_latest_training_decision(str(strategy["id"]))
                    if decision is not None:
                        item["training_approval"] = decision["decision"]
                    run = workflow.get_latest_run(str(strategy["id"]))
                    if run is not None:
                        item.update(run_id=run["id"], run_status=run["status"])
                        add_reference("run", run["id"])
                context["modeling_context"].append(item)
    decisions = values.list_decision_versions(capability.id)
    if decisions:
        latest = decisions[-1]
        add_reference("decision_value", latest.id, latest.sha256)
        context["decision"] = {
            "id": latest.id,
            "confirmed_human": latest.confirmed_human,
            "missing_fields": latest.missing_fields,
            "owner_declared": latest.spec.owner_label is not None,
            "spec": {
                **{
                    name: clean_text(getattr(latest.spec, name))
                    for name in (
                        "question",
                        "action",
                        "alternative",
                        "kpi",
                    )
                },
                "constraints": [clean_text(item) for item in latest.spec.constraints[:20]],
                "predictive_version_id": latest.spec.predictive_version_id,
                "economic_version_id": latest.spec.economic_version_id,
            },
        }
    predictions = values.list_predictive_versions(capability.id)
    if predictions:
        latest = predictions[-1]
        add_reference("predictive_spec", latest.id, latest.sha256)
        context["predictive_definition"] = {
            "id": latest.id,
            "confirmed_human": latest.confirmed_human,
            "missing_fields": latest.missing_fields,
            "population": clean_text(latest.spec.population),
            "target_definition": clean_text(latest.spec.target_definition),
            "prediction_moment": clean_text(latest.spec.prediction_moment),
        }
    economics = values.list_economic_versions(capability.id)
    if economics:
        latest = economics[-1]
        add_reference("economic_scenario", latest.id, latest.sha256)
        context["economics"] = {
            "id": latest.id,
            "confirmed_human": latest.confirmed_human,
            "missing_fields": latest.missing_fields,
            "evidence_type": "synthetic_scenario",
            "adverse_event": True,
        }
    from core.domain.value_policy import ValuePolicyService

    policies = [
        policy
        for policy in ValuePolicyService(workflow).list_versions(capability.id)
        if task.cycle_id is None or policy.content.cycle_id == task.cycle_id
    ]
    if policies:
        latest_policy = policies[-1]
        add_reference("value_policy", latest_policy.id, latest_policy.sha256)
        context["policy"] = {
            "id": latest_policy.id,
            "version_no": latest_policy.version_no,
            "review_current": ValuePolicyService(workflow).is_review_current(latest_policy.id),
        }

    evidence_tasks = [task]
    if task.role == "coordinator":
        for child in tasks.list_children(task.id):
            context["children"].append(
                {
                    "id": child.id,
                    "role": child.role,
                    "state": child.state,
                    "objective": clean_text(child.objective),
                }
            )
            if child.state == "COMPLETED":
                evidence_tasks.append(child)
                from core.db.repositories.agent_turns import AgentTurnRepository

                turns = AgentTurnRepository(workflow).list_turns(child.id)
                final = next(
                    (
                        turn
                        for turn in reversed(turns)
                        if turn.proposal is not None
                        and turn.proposal.action == "finish"
                        and turn.dispatch_result is not None
                    ),
                    None,
                )
                if final is not None:
                    context["children"][-1]["proposal_result"] = {
                        "message": clean_text(final.proposal.message),
                        "evidence_refs": [
                            ref.model_dump(mode="json") for ref in final.proposal.evidence_refs
                        ],
                        "label": "AGENT_PROPOSAL_NOT_VERIFIED_FACT",
                    }
    for evidence_task in evidence_tasks:
        operations = [
            operation
            for operation in tasks.list_operations(evidence_task.id)
            if operation.call.tool_name
            in {
                "read_capability_context",
                "save_decision_draft",
                "evaluate_economic_scenario",
                "read_dataset_profile",
                "inspect_data_quality",
                "inspect_label_readiness",
                "aggregate_analysis",
                "inspect_model_viability",
                "propose_training_strategy",
                "train_model",
                "read_run_summary",
                "propose_value_policy",
            }
            and operation.result is not None
        ]
        for operation in operations[-10:]:
            result = operation.result
            assert result is not None
            for ref in result.output_refs:
                add_reference(ref.kind, ref.id, ref.sha256)
            summary = result.summary
            if operation.call.tool_name == "read_dataset_profile" and result.status == "COMPLETED":
                # Los resultados anteriores a MV-08 podían conservar nombres de columnas.
                # Revalidar el contrato permitido y sustituirlos también al recuperar evidencia.
                try:
                    profile = SafeProfileForLLMV1.model_validate(summary["profile"])
                    safe = profile.model_dump(mode="json")
                    for column in safe["columns"]:
                        column["name"] = column["ref"]
                    summary = {
                        "profile": safe,
                        "integrity_verified": summary.get("integrity_verified") is True,
                    }
                except (KeyError, ValueError, TypeError):
                    summary = {"profile_status": "INVALID_DATASET_PROFILE"}
            context["tool_evidence"].append(
                {
                    "role": evidence_task.role,
                    "tool_name": operation.call.tool_name,
                    "status": result.status,
                    "error_code": result.error_code,
                    "summary": summary,
                    "output_refs": [ref.model_dump(mode="json") for ref in result.output_refs],
                }
            )
    context["available_refs"] = list(refs.values())
    return context
