"""Contratos agregados del motor ML; nunca contienen filas del dataset."""

from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, model_validator

from core.domain.canonical import canonical_hash
from core.domain.context import ClassSupport, StrictModel


class PartitionSummary(StrictModel):
    row_count: int = Field(gt=0)
    class_distribution: list[ClassSupport] = Field(min_length=2, max_length=2)
    index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TemporalSplitEvidenceV1(StrictModel):
    """Límites y exclusiones agregados; la disponibilidad no se infiere de las fechas."""

    time_column: str = Field(min_length=1)
    label_available_at_column: str | None = None
    timezone: Literal["UTC"] = "UTC"
    ties: Literal["same_timestamp_same_partition"] = "same_timestamp_same_partition"
    label_availability_verified: bool
    source_rows: int = Field(gt=0)
    validation_origin: str = Field(min_length=1)
    test_origin: str = Field(min_length=1)
    availability_rule: Literal["strictly_before_next_partition"] | None = None
    train_label_available_max: str | None = None
    validation_label_available_max: str | None = None
    test_label_available_max: str | None = None
    purged_train_rows: int = Field(ge=0)
    purged_validation_rows: int = Field(ge=0)
    purged_train_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    purged_validation_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SplitSummary(StrictModel):
    method: Literal["stratified_random", "temporal"] = "stratified_random"
    seed: int = Field(ge=0)
    train: PartitionSummary
    validation: PartitionSummary
    test: PartitionSummary
    temporal: TemporalSplitEvidenceV1 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class CandidateFitSummary(StrictModel):
    candidate: Literal[
        "DummyClassifier",
        "LogisticRegression",
        "RandomForestClassifier",
    ]
    fit_seconds: float = Field(ge=0)
    status: Literal["FITTED"] = "FITTED"


class TrainingSummaryV1(StrictModel):
    schema_version: Literal["training-summary.v1"] = "training-summary.v1"
    status: Literal["TRAINED_PENDING_EVALUATION"] = "TRAINED_PENDING_EVALUATION"
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_schema_version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    feature_columns: list[str] = Field(min_length=1)
    numeric_columns: list[str]
    categorical_columns: list[str]
    split: SplitSummary
    candidates: list[CandidateFitSummary] = Field(min_length=3, max_length=3)
    package_versions: dict[str, str]
    warnings: list[str]


TargetValue = str | int | float | bool


ThresholdSource = Literal["baseline_default", "fixed", "validation_max_f1"]


class BinaryMetricSet(StrictModel):
    threshold: float = Field(ge=0, le=1)
    threshold_source: ThresholdSource
    positive_class: TargetValue
    negative_class: TargetValue
    support: int = Field(gt=0)
    positive_support: int = Field(gt=0)
    negative_support: int = Field(gt=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    average_precision: float = Field(ge=0, le=1)
    roc_auc: float = Field(ge=0, le=1)
    confusion_matrix: list[list[int]]


class CandidateEvaluation(StrictModel):
    candidate: Literal[
        "DummyClassifier",
        "LogisticRegression",
        "RandomForestClassifier",
    ]
    partition: Literal["validation"] = "validation"
    metrics: BinaryMetricSet
    primary_metric_value: float = Field(ge=0, le=1)
    guardrails_passed: bool
    failed_guardrails: list[str]
    beats_baseline: bool | None = None


class SelectionSummary(StrictModel):
    primary_metric: str
    baseline_model: Literal["DummyClassifier"] = "DummyClassifier"
    baseline_value: float = Field(ge=0, le=1)
    eligible_candidates: list[str]
    selected_model: Literal["LogisticRegression", "RandomForestClassifier"] | None
    outcome: Literal["RECOMMENDED", "WITHOUT_RECOMMENDATION"]
    reason: str
    tie_breaker: Literal["secondary_metric_then_simplicity"] = "secondary_metric_then_simplicity"
    secondary_metric: Literal["f1", "balanced_accuracy"]
    threshold_policy: Literal["max_f1_on_validation", "fixed"]
    selected_threshold: float | None = Field(default=None, ge=0, le=1)


class FeatureImportance(StrictModel):
    feature: str = Field(min_length=1)
    importance_mean: float
    importance_std: float = Field(ge=0)


class MetricInterval(StrictModel):
    metric: str = Field(min_length=1)
    point: float = Field(ge=0, le=1)
    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)


class BootstrapSummary(StrictModel):
    """Incertidumbre del holdout con el modelo y el umbral ya congelados."""

    partition: Literal["holdout"] = "holdout"
    method: Literal["percentile"] = "percentile"
    confidence: Literal[0.95] = 0.95
    seed: int = Field(ge=0)
    resamples_requested: int = Field(ge=1)
    resamples_used: int = Field(ge=1)
    intervals: list[MetricInterval] = Field(min_length=1)


class OperatingPointV1(StrictModel):
    """V2-13: qué pasaría en la prueba final si se marcara una proporción de casos."""

    flagged_share: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    flagged: int = Field(ge=0)
    true_alerts: int = Field(ge=0)
    false_alarms: int = Field(ge=0)
    missed: int = Field(ge=0)
    support: int = Field(ge=1)
    positives: int = Field(ge=0)


def top_fraction_count(fraction: float, total: int) -> int:
    """Trunca la fracción decimal sin redondear hacia arriba ni errores binarios."""

    return int(Decimal(str(fraction)) * total)


class PolicyOperatingPointV1(StrictModel):
    selection_fraction: float = Field(ge=0, le=1)
    flagged_share: float = Field(ge=0, le=1)
    cutoff_score: float | None = Field(default=None, ge=0, le=1)
    flagged: int = Field(ge=0)
    true_alerts: int = Field(ge=0)
    false_alarms: int = Field(ge=0)
    missed: int = Field(ge=0)
    support: int = Field(ge=1)
    positives: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_match_policy(self) -> "PolicyOperatingPointV1":
        if (
            self.flagged != top_fraction_count(self.selection_fraction, self.support)
            or self.true_alerts + self.false_alarms != self.flagged
            or self.true_alerts + self.missed != self.positives
            or self.positives > self.support
            or self.false_alarms > self.support - self.positives
            or abs(self.flagged_share - self.flagged / self.support) > 1e-12
            or (self.cutoff_score is None) != (self.flagged == 0)
        ):
            raise ValueError("Los conteos no corresponden a la política por fracción")
        return self


class PolicySelectionDataV1(StrictModel):
    schema_version: Literal["policy-selection.v1"] = "policy-selection.v1"
    partition: Literal["validation"] = "validation"
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    partition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_model: Literal["LogisticRegression", "RandomForestClassifier"]
    positive_class: TargetValue
    selection_mode: Literal["top_fraction"] = "top_fraction"
    tie_breaker: Literal["input_order"] = "input_order"
    rounding: Literal["floor"] = "floor"
    points: list[PolicyOperatingPointV1] = Field(min_length=2)

    @model_validator(mode="after")
    def points_share_cohort(self) -> "PolicySelectionDataV1":
        fractions = [point.selection_fraction for point in self.points]
        if fractions != sorted(set(fractions)) or fractions[0] != 0 or fractions[-1] != 1:
            raise ValueError("La grilla debe estar ordenada e incluir no actuar y atender a todos")
        if len({(point.support, point.positives) for point in self.points}) != 1:
            raise ValueError("Los puntos deben usar la misma cohorte de validación")
        return self


class PolicySelectionArtifactV1(PolicySelectionDataV1):
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def compute_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json", exclude={"content_sha256"}))

    @classmethod
    def build(cls, **values: Any) -> "PolicySelectionArtifactV1":
        payload = PolicySelectionDataV1.model_validate(values).model_dump(mode="json")
        return cls(**payload, content_sha256=canonical_hash(payload))

    @model_validator(mode="after")
    def content_matches_hash(self) -> "PolicySelectionArtifactV1":
        if self.content_sha256 != self.compute_hash():
            raise ValueError("La huella del artefacto de selección no coincide")
        return self


class ReliabilityBinV1(StrictModel):
    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)
    support: int = Field(gt=0)
    mean_score: float = Field(ge=0, le=1)
    observed_rate: float = Field(ge=0, le=1)


class ProbabilityDiagnosticsV1(StrictModel):
    schema_version: Literal["probability-diagnostics.v1"] = "probability-diagnostics.v1"
    partition: Literal["validation", "holdout"]
    support: int = Field(gt=0)
    positive_support: int = Field(ge=0)
    negative_support: int = Field(ge=0)
    brier_score: float = Field(ge=0, le=1)
    log_loss: float = Field(ge=0)
    prevalence_baseline_probability: float = Field(ge=0, le=1)
    prevalence_baseline_brier: float = Field(ge=0, le=1)
    prevalence_baseline_log_loss: float = Field(ge=0)
    reliability_bins: list[ReliabilityBinV1]
    calibration_fit: Literal["none"] = "none"
    status: Literal["diagnostic_only", "insufficient_support"]
    allowed_use: Literal["ranking_only"] = "ranking_only"
    warnings: list[str]


class EvaluationSummaryV1(StrictModel):
    schema_version: Literal["evaluation-summary.v2"] = "evaluation-summary.v2"
    validation: dict[str, CandidateEvaluation]
    selection: SelectionSummary
    holdout: BinaryMetricSet | None
    holdout_evaluations: Literal[0, 1]
    holdout_bootstrap: BootstrapSummary | None = None
    permutation_importance_partition: Literal["validation"] | None
    permutation_importance_scorer: Literal["average_precision"] | None = None
    permutation_importance_repeats: int | None = Field(default=None, ge=1)
    permutation_importance: list[FeatureImportance]
    warnings: list[str]
    # V2-13: puntos de operación de la prueba final (sólo conteos, sin filas). Vacío en
    # entrenamientos anteriores: para decidir con el modelo hay que reentrenar.
    holdout_operating_points: list[OperatingPointV1] = Field(default_factory=list)
    # MV-03: artefacto agregado de validation, nunca sustituido por puntos de test.
    policy_selection: PolicySelectionArtifactV1 | None = None
    probability_diagnostics: list[ProbabilityDiagnosticsV1] = Field(default_factory=list)


class RunExecutionResult(StrictModel):
    run_id: str = Field(min_length=1)
    status: Literal["COMPLETED", "COMPLETED_WITHOUT_RECOMMENDATION"]
    selected_model: str | None
    artifact_path: str | None
    evaluation: "EvaluationSummaryV1 | TaskEvaluationV1"


class TaskEvaluationV1(StrictModel):
    """Evaluación de varias clases o cantidades (ADR-016): validación elige, test una vez."""

    schema_version: Literal["task-evaluation.v1"] = "task-evaluation.v1"
    task_kind: Literal["multiclass_classification", "regression"]
    primary_metric: str = Field(min_length=1)
    higher_is_better: bool
    baseline: str = Field(min_length=1)
    validation: dict[str, dict[str, float]]
    selected_model: str | None
    test: dict[str, float]
    split_method: Literal["random", "stratified", "temporal"]
    split_rows: dict[str, int]
    holdout_evaluations: Literal[0, 1]
    warnings: list[str]


RunExecutionResult.model_rebuild()
