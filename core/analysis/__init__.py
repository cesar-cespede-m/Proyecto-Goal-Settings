"""Contratos y validación del motor de análisis declarativo."""

from core.analysis.contracts import (
    AggregateSpecV1,
    AnalysisOutputV1,
    AnalysisPlanV1,
    AnalysisStepV1,
    ColumnType,
)
from core.analysis.engine import (
    ANALYSIS_ENGINE_VERSION,
    AnalysisColumnV1,
    AnalysisEngine,
    AnalysisLimits,
    AnalysisResultV1,
    LoadedSource,
    load_project_source,
)
from core.analysis.payload import (
    AnalysisLineageV1,
    LlmPayloadV1,
    analysis_lineage,
    omitted_columns,
    to_llm_payload,
)
from core.analysis.roles import (
    ColumnRoleConfirmationV1,
    ColumnRoleName,
    ColumnRolesV1,
    ColumnRoleV1,
    infer_roles,
)
from core.analysis.snapshot import (
    SNAPSHOT_CUTOFF_COLUMN,
    ObservationCoverageV1,
    SnapshotError,
    SnapshotReadinessV1,
    SnapshotResultV1,
    SnapshotSpecV1,
    SnapshotSummaryV1,
    build_snapshot,
)
from core.analysis.validation import AnalysisPlanError, canonical_plan_hash, validate_plan

__all__ = [
    "AggregateSpecV1",
    "ANALYSIS_ENGINE_VERSION",
    "AnalysisColumnV1",
    "AnalysisEngine",
    "AnalysisLineageV1",
    "AnalysisLimits",
    "AnalysisOutputV1",
    "AnalysisPlanError",
    "AnalysisPlanV1",
    "AnalysisResultV1",
    "AnalysisStepV1",
    "ColumnRoleConfirmationV1",
    "ColumnRoleName",
    "ColumnRolesV1",
    "ColumnRoleV1",
    "ColumnType",
    "LlmPayloadV1",
    "LoadedSource",
    "ObservationCoverageV1",
    "SNAPSHOT_CUTOFF_COLUMN",
    "SnapshotError",
    "SnapshotReadinessV1",
    "SnapshotResultV1",
    "SnapshotSpecV1",
    "SnapshotSummaryV1",
    "build_snapshot",
    "analysis_lineage",
    "canonical_plan_hash",
    "infer_roles",
    "load_project_source",
    "omitted_columns",
    "to_llm_payload",
    "validate_plan",
]
