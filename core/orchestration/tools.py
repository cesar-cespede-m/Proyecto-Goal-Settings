"""MV-06: registro cerrado de herramientas con argumentos y salidas acotados."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.contracts.decision_value import DecisionValueSpecV1
from core.contracts.orchestration import AgentTaskV1, ArtifactRefV1, ToolResultV1
from core.data_engine.models import DatasetProfileV1
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.domain.capabilities import CapabilityService
from core.domain.decision_value import DecisionValueService
from core.economics.scenarios import evaluate_scenario_version


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


class EmptyArguments(_Arguments):
    pass


class DatasetArguments(_Arguments):
    dataset_id: str = Field(min_length=1, max_length=100)


class RunArguments(_Arguments):
    run_id: str = Field(min_length=1, max_length=100)


class EconomicArguments(_Arguments):
    economic_version_id: str = Field(min_length=1, max_length=100)
    selected_cases: int | None = Field(default=None, ge=0, le=1_000_000_000, strict=True)


class DecisionDraftArguments(_Arguments):
    spec: DecisionValueSpecV1
    based_on_version_id: str | None = Field(default=None, min_length=1, max_length=100)


@dataclass(frozen=True)
class ToolContext:
    workflow: WorkflowRepository
    task: AgentTaskV1
    actor_label: str
    operation_id: str
    storage_root: str | Path | None = None
    verify_current: Callable[[], None] | None = None


@dataclass(frozen=True)
class ToolOutcome:
    result: ToolResultV1
    # Cambios previstos por una escritura propia; el resto de la huella debe permanecer igual.
    context_updates: dict[str, Any] = field(default_factory=dict)
    project_updates: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    roles: frozenset[str]
    arguments: type[BaseModel]
    execute: Callable[[ToolContext, Any], ToolOutcome]
    writes: bool = False
    required_seconds: Callable[[ToolContext, Any], int] | None = None


def domain_operation_id(operation_id: str) -> str:
    return "orchestration:" + canonical_hash({"operation_id": operation_id})


def version_identity(version: Any) -> dict[str, Any]:
    return {"id": version.id, "sha256": version.sha256, "confirmed_human": version.confirmed_human}


def _capability(context: ToolContext, arguments: EmptyArguments) -> ToolOutcome:
    capabilities = CapabilityService(context.workflow)
    capability = capabilities.get_capability(context.task.capability_id)
    values = DecisionValueService(context.workflow)
    references = [
        ArtifactRefV1(kind="capability", id=capability.id, sha256=canonical_hash(capability))
    ]
    summary: dict[str, Any] = {"value_kind": capability.value_kind, "definitions": {}}
    for kind, versions in (
        ("decision_value", values.list_decision_versions(capability.id)),
        ("predictive_spec", values.list_predictive_versions(capability.id)),
        ("economic_scenario", values.list_economic_versions(capability.id)),
    ):
        if versions:
            latest = versions[-1]
            references.append(ArtifactRefV1(kind=kind, id=latest.id, sha256=latest.sha256))
            summary["definitions"][kind] = {
                "version_no": latest.version_no,
                "confirmed_human": latest.confirmed_human,
                "missing_fields": latest.missing_fields,
            }
    if context.task.cycle_id is not None:
        cycle = capabilities.get_cycle(context.task.cycle_id)
        references.append(ArtifactRefV1(kind="cycle", id=cycle.id, sha256=canonical_hash(cycle)))
    return ToolOutcome(ToolResultV1(status="COMPLETED", output_refs=references, summary=summary))


def _profile(context: ToolContext, arguments: DatasetArguments) -> ToolOutcome:
    from core.orchestration.data_tools import load_tool_source, safe_profile

    load_tool_source(context, arguments.dataset_id)
    dataset = context.workflow.get_dataset(arguments.dataset_id)
    profile = DatasetProfileV1.model_validate_json(dataset["profile_json"])
    return ToolOutcome(
        ToolResultV1(
            status="COMPLETED",
            output_refs=[
                ArtifactRefV1(kind="dataset", id=arguments.dataset_id, sha256=dataset["sha256"])
            ],
            summary={
                "row_count": profile.row_count,
                "column_count": profile.column_count,
                "profile": safe_profile(profile),
                "integrity_verified": True,
            },
        )
    )


def _numeric_metrics(value: Any) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "support",
        "positive_support",
        "negative_support",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "average_precision",
        "roc_auc",
        "threshold",
    }
    return {
        key: number
        for key, number in value.items()
        if key in allowed
        and isinstance(number, int | float)
        and not isinstance(number, bool)
        and math.isfinite(number)
    }


def _run(context: ToolContext, arguments: RunArguments) -> ToolOutcome:
    CapabilityService(context.workflow).require_scope(
        context.task.capability_id, cycle_id=context.task.cycle_id, run_id=arguments.run_id
    )
    run = context.workflow.get_run(arguments.run_id)
    evaluation = json.loads(run["candidate_metrics_json"])
    candidates = {"DummyClassifier", "LogisticRegression", "RandomForestClassifier"}
    validation = evaluation.get("validation", {})
    return ToolOutcome(
        ToolResultV1(
            status="COMPLETED",
            output_refs=[ArtifactRefV1(kind="run", id=arguments.run_id)],
            summary={
                "status": run["status"],
                "selected_model": run["selected_model"]
                if run["selected_model"] in candidates
                else None,
                "holdout_metrics": _numeric_metrics(evaluation.get("holdout")),
                "validation_metrics": {
                    name: _numeric_metrics(candidate.get("metrics"))
                    for name, candidate in validation.items()
                    if name in candidates and isinstance(candidate, dict)
                }
                if isinstance(validation, dict)
                else {},
            },
        )
    )


def _economy(context: ToolContext, arguments: EconomicArguments) -> ToolOutcome:
    from core.db.repositories.orchestration import OrchestrationError

    version = DecisionValueService(context.workflow).get_economic_version(
        arguments.economic_version_id
    )
    if version.capability_id != context.task.capability_id:
        raise OrchestrationError("TOOL_SCOPE_MISMATCH", "El escenario pertenece a otra capacidad")
    result = evaluate_scenario_version(version, selected_cases=arguments.selected_cases)
    return ToolOutcome(
        ToolResultV1(
            status="COMPLETED",
            output_refs=[
                ArtifactRefV1(kind="economic_scenario", id=version.id, sha256=version.sha256)
            ],
            summary=result.model_dump(mode="json", exclude={"period"}),
        )
    )


def _save_decision(context: ToolContext, arguments: DecisionDraftArguments) -> ToolOutcome:
    from core.db.repositories.orchestration import OrchestrationError

    if arguments.spec.owner_label is not None:
        raise OrchestrationError(
            "TOOL_HUMAN_FIELD_FORBIDDEN", "El responsable debe declararlo una persona"
        )
    service = DecisionValueService(context.workflow)
    owner = None
    if arguments.based_on_version_id is not None:
        previous = service.get_decision_version(arguments.based_on_version_id)
        if previous.capability_id != context.task.capability_id:
            raise OrchestrationError("TOOL_SCOPE_MISMATCH", "La decisión es de otra capacidad")
        owner = previous.spec.owner_label
    payload = {**arguments.spec.model_dump(mode="python"), "owner_label": owner}
    version = service.create_decision_version(
        context.task.capability_id,
        spec=payload,
        based_on_version_id=arguments.based_on_version_id,
        confirmed_human=False,
        actor_label=context.actor_label,
        operation_id=domain_operation_id(context.operation_id),
    )
    return ToolOutcome(
        ToolResultV1(
            status="COMPLETED",
            output_refs=[
                ArtifactRefV1(kind="decision_value", id=version.id, sha256=version.sha256)
            ],
            summary={
                "version_no": version.version_no,
                "confirmed_human": False,
                "missing_fields": version.missing_fields,
            },
        ),
        context_updates={"decision_version": version_identity(version)},
    )


def registered_tools() -> dict[str, RegisteredTool]:
    """Sólo funciones internas conocidas; ningún nombre importa módulos ni ejecuta código."""

    from core.orchestration.data_tools import data_tools
    from core.orchestration.modeling_tools import modeling_tools
    from core.orchestration.value_tools import value_tools

    tools = (
        RegisteredTool(
            "read_capability_context",
            frozenset({"coordinator", "business", "data", "modeling", "value_evidence"}),
            EmptyArguments,
            _capability,
        ),
        RegisteredTool("read_dataset_profile", frozenset({"data"}), DatasetArguments, _profile),
        RegisteredTool(
            "read_run_summary", frozenset({"modeling", "value_evidence"}), RunArguments, _run
        ),
        RegisteredTool(
            "evaluate_economic_scenario",
            frozenset({"business", "value_evidence"}),
            EconomicArguments,
            _economy,
        ),
        RegisteredTool(
            "save_decision_draft",
            frozenset({"business"}),
            DecisionDraftArguments,
            _save_decision,
            writes=True,
        ),
    )
    return {tool.name: tool for tool in (*tools, *data_tools(), *modeling_tools(), *value_tools())}
