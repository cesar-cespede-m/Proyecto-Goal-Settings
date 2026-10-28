"""MV-08: inspección local y agregados cerrados, sin filas ni categorías salientes."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, model_validator

from core.analysis.contracts import AnalysisPlanV1, ColumnType
from core.analysis.engine import AnalysisEngine, AnalysisLimits, LoadedSource, load_project_source
from core.analysis.quality import quality_findings
from core.contracts.orchestration import ArtifactRefV1, ToolResultV1
from core.data_engine.models import DatasetProfileV1
from core.db.repositories.orchestration import OrchestrationError
from core.domain.assistant import SafeProfileForLLMV1, column_refs
from core.domain.capabilities import CapabilityService
from core.orchestration.tools import DatasetArguments, RegisteredTool, ToolContext, ToolOutcome

MIN_AGGREGATE_SUPPORT = 5


class AggregateArguments(DatasetArguments):
    template: Literal["dataset_summary", "numeric_summary"]
    column_alias: str | None = Field(default=None, pattern=r"^C\d{3}$")

    @model_validator(mode="after")
    def column_matches_template(self) -> AggregateArguments:
        if (self.template == "numeric_summary") != (self.column_alias is not None):
            raise ValueError("La plantilla numérica requiere un alias y la general no lo admite")
        return self


def safe_profile(profile: DatasetProfileV1) -> dict[str, Any]:
    """Reutiliza el perfil permitido y sustituye todos los nombres por aliases locales."""

    safe = SafeProfileForLLMV1.from_profile(profile).model_dump(mode="json")
    for column in safe["columns"]:
        column["name"] = column["ref"]
    return safe


def load_tool_source(context: ToolContext, dataset_id: str) -> LoadedSource:
    CapabilityService(context.workflow).require_scope(
        context.task.capability_id, cycle_id=context.task.cycle_id, dataset_id=dataset_id
    )
    if context.storage_root is None:
        raise OrchestrationError("MISSING_STORAGE", "La inspección necesita storage configurado")
    source = load_project_source(context.workflow, context.storage_root, dataset_id)
    profile = DatasetProfileV1.model_validate_json(
        context.workflow.get_dataset(dataset_id)["profile_json"]
    )
    if list(source.frame.columns) != [column.name for column in profile.columns]:
        raise OrchestrationError("DATASET_PROFILE_MISMATCH", "El perfil no corresponde a la tabla")
    if len(source.frame) != profile.row_count or len(source.frame.columns) != profile.column_count:
        raise OrchestrationError("DATASET_PROFILE_MISMATCH", "El perfil no corresponde a la tabla")
    if len(source.frame) > AnalysisLimits().max_input_rows:
        raise OrchestrationError("ANALYSIS_INPUT_ROW_LIMIT_EXCEEDED", "La fuente excede el límite")
    return source


def _outcome(dataset_id: str, source: LoadedSource, summary: dict[str, Any]) -> ToolOutcome:
    return ToolOutcome(
        ToolResultV1(
            status="COMPLETED",
            output_refs=[ArtifactRefV1(kind="dataset", id=dataset_id, sha256=source.sha256)],
            summary=summary,
        )
    )


def _quality(context: ToolContext, arguments: DatasetArguments) -> ToolOutcome:
    source = load_tool_source(context, arguments.dataset_id)
    aliases = {name: f"C{index:03d}" for index, name in enumerate(source.frame.columns, 1)}
    rows = len(source.frame)
    findings = [
        {
            "column_alias": aliases[finding.column],
            "kind": finding.kind,
            "count": finding.count,
            "share": finding.share,
        }
        for finding in quality_findings({"source": source})
    ]
    missing = [
        {
            "column_alias": alias,
            "count": int(source.frame[column].isna().sum()),
            "share": float(source.frame[column].isna().mean()) if rows else 0.0,
        }
        for column, alias in aliases.items()
    ]
    return _outcome(
        arguments.dataset_id,
        source,
        {
            "kind": "data_quality",
            "row_count": rows,
            "column_count": len(aliases),
            "missing_counts": missing,
            "findings": findings,
            "limitations": ["HEURISTIC_FINDINGS_REQUIRE_REVIEW", "NO_TRAINING_AUTHORIZATION"],
        },
    )


def _aggregate(context: ToolContext, arguments: AggregateArguments) -> ToolOutcome:
    source = load_tool_source(context, arguments.dataset_id)
    profile = DatasetProfileV1.model_validate_json(
        context.workflow.get_dataset(arguments.dataset_id)["profile_json"]
    )
    aliases = column_refs(profile)
    column = aliases.get(arguments.column_alias or "")
    aggregations: list[dict[str, Any]] = [{"function": "count", "output_column": "count"}]
    if arguments.template == "numeric_summary":
        if column is None:
            raise OrchestrationError("UNKNOWN_COLUMN_ALIAS", "El alias no pertenece a esta fuente")
        info = next(item for item in profile.columns if item.name == column)
        if (
            source.schema[column] != ColumnType.NUMBER
            or info.possible_identifier
            or info.semantic_type in {"identifier", "personal"}
        ):
            raise OrchestrationError(
                "AGGREGATE_COLUMN_NOT_ALLOWED", "La columna no es numérica apta"
            )
        aggregations = [
            {"function": function, "column": column, "output_column": function}
            for function in ("count", "mean", "sum")
        ]
    plan = AnalysisPlanV1.model_validate(
        {
            "sources": ["source"],
            "steps": [{"op": "group_by", "by": [], "aggregations": aggregations}],
            "output": {"chart": "table"},
        }
    )
    result = AnalysisEngine().run(plan, {"source": source})
    # Sólo esta plantilla construida localmente puede llegar aquí. Nunca serializar el resultado
    # tabular completo: proyectar explícitamente los escalares agregados autorizados.
    metrics: dict[str, int | float | None] = {}
    for item, value in zip(result.columns, result.rows[0], strict=True):
        if item.name not in {"count", "mean", "sum"}:
            raise OrchestrationError("INVALID_AGGREGATE_OUTPUT", "Salida fuera de plantilla")
        metrics[item.name] = (
            value
            if isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            else None
        )
    warnings = []
    if arguments.template == "numeric_summary" and (metrics["count"] or 0) < MIN_AGGREGATE_SUPPORT:
        metrics.update(mean=None, sum=None)
        warnings.append("AGGREGATE_SUPPORT_TOO_SMALL")
    if arguments.template == "dataset_summary":
        metrics.update(
            column_count=len(source.frame.columns),
            missing_cells=int(source.frame.isna().sum().sum()),
            duplicate_rows=int(source.frame.duplicated().sum()),
        )
    return _outcome(
        arguments.dataset_id,
        source,
        {
            "kind": "aggregate_analysis",
            "template": arguments.template,
            "column_alias": arguments.column_alias,
            "metrics": metrics,
            "warnings": warnings,
            "limitations": ["DESCRIPTIVE_NOT_CAUSAL", "NO_TRAINING_AUTHORIZATION"],
        },
    )


def _readiness(context: ToolContext, arguments: DatasetArguments) -> ToolOutcome:
    from core.domain.data_readiness import DataReadinessService

    if context.storage_root is None:
        raise OrchestrationError("MISSING_STORAGE", "La inspección necesita storage configurado")
    result = DataReadinessService(context.workflow, context.storage_root).inspect_label_readiness(
        context.task.capability_id, dataset_id=arguments.dataset_id, cycle_id=context.task.cycle_id
    )
    allowed = {
        "schema_version",
        "status",
        "reason_codes",
        "reason_counts",
        "snapshot_readiness",
        "measurement_plan",
        "pending_source_count",
        "table_rows",
        "verified_table_rows",
        "grants_training_approval",
        "statistical_support_checked",
        "evidence_sha256",
        "recipe_sha256",
        "binding_sha256",
    }
    summary = result.model_dump(mode="json", include=allowed)
    return ToolOutcome(
        ToolResultV1(
            status="COMPLETED",
            output_refs=result.output_refs,
            summary={
                "kind": "label_readiness",
                **summary,
                "limitations": ["NO_TRAINING_AUTHORIZATION", "NO_GATE_CHANGED"],
            },
        )
    )


def data_tools() -> tuple[RegisteredTool, ...]:
    roles = frozenset({"data"})
    return (
        RegisteredTool("inspect_data_quality", roles, DatasetArguments, _quality),
        RegisteredTool("inspect_label_readiness", roles, DatasetArguments, _readiness),
        RegisteredTool("aggregate_analysis", roles, AggregateArguments, _aggregate),
    )
