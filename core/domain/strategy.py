"""Contrato estricto de la estrategia P0 revisable por una persona."""

from typing import Literal

from pydantic import Field, model_validator

from core.domain.context import (
    METRICS_BY_TASK,
    GuardrailDecision,
    PrimaryMetric,
    StrictModel,
    TaskMetric,
    TaskType,
)


class StrategySplit(StrictModel):
    # GA-07e: «temporal» entrena con lo más antiguo y prueba con lo más reciente según
    # `time_column`; evita que el modelo aprenda del futuro.
    method: Literal["stratified_random", "temporal"] = "stratified_random"
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)
    # Vacío no se serializa: las estrategias sin fechas conservan su huella de siempre.
    time_column: str | None = Field(default=None, exclude_if=lambda value: value is None)
    label_available_at_column: str | None = Field(
        default=None, min_length=1, max_length=255, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "StrategySplit":
        if (self.method == "temporal") != (self.time_column is not None):
            raise ValueError("La partición por tiempo requiere una columna de fecha, y sólo ella")
        if self.label_available_at_column is not None:
            if self.method != "temporal":
                raise ValueError("La disponibilidad del resultado requiere partición temporal")
            if self.label_available_at_column.casefold() == self.time_column.casefold():
                raise ValueError(
                    "El momento de predicción y la disponibilidad usan columnas distintas"
                )
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("Las fracciones train/validation/test deben sumar 1")
        return self


class PreprocessingConfig(StrictModel):
    numeric_imputation: Literal["median"] = "median"
    categorical_imputation: Literal["most_frequent"] = "most_frequent"
    categorical_encoding: Literal["one_hot_ignore_unknown"] = "one_hot_ignore_unknown"
    numeric_scaling: Literal["standard_for_linear"] = "standard_for_linear"


class DummyParameters(StrictModel):
    strategy: Literal["most_frequent"] = "most_frequent"


class LogisticRegressionParameters(StrictModel):
    c: float = Field(default=1.0, gt=0, le=100)
    max_iter: int = Field(default=1000, ge=100, le=5000)
    solver: Literal["liblinear"] = "liblinear"
    class_weight: Literal["balanced"] | None = None


class RandomForestParameters(StrictModel):
    n_estimators: int = Field(default=100, ge=10, le=500)
    max_depth: int | None = Field(default=12, ge=2, le=50)
    min_samples_leaf: int = Field(default=2, ge=1, le=100)
    class_weight: Literal["balanced"] | None = None
    n_jobs: Literal[1] = 1


class ModelParameters(StrictModel):
    dummy: DummyParameters = Field(default_factory=DummyParameters)
    logistic_regression: LogisticRegressionParameters = Field(
        default_factory=LogisticRegressionParameters
    )
    random_forest: RandomForestParameters = Field(default_factory=RandomForestParameters)


class DecisionThresholdPolicy(StrictModel):
    """Cómo se fija el umbral de decisión de cada candidato antes de tocar el holdout.

    `max_f1_on_validation` elige, por candidato, el umbral que maximiza F1 en validation;
    el holdout lo recibe congelado. Con clases desbalanceadas, 0.5 fijo suele anular el recall.
    """

    method: Literal["max_f1_on_validation", "fixed"] = "max_f1_on_validation"
    fixed_value: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def value_matches_method(self) -> "DecisionThresholdPolicy":
        if self.method == "fixed" and self.fixed_value is None:
            raise ValueError("El umbral fijo requiere fixed_value")
        if self.method != "fixed" and self.fixed_value is not None:
            raise ValueError("fixed_value sólo aplica al método fixed")
        return self


class ExecutionCaps(StrictModel):
    max_training_seconds: int = Field(default=300, ge=10, le=1800)
    max_parallel_jobs: Literal[1] = 1


CandidateName = Literal[
    "DummyClassifier",
    "LogisticRegression",
    "RandomForestClassifier",
    "DummyRegressor",
    "Ridge",
    "RandomForestRegressor",
]

# ADR-016: tipos que ya recorren todo el flujo (evaluación, reporte, uso y monitoreo).
FLOW_READY_TASK_KINDS = {"binary_classification", "multiclass_classification", "regression"}

# ADR-016: baseline y dos candidatos por tipo de modelo. Varias clases usa los mismos nombres
# que sí/no; las cantidades usan regresores.
CANDIDATES_BY_TASK: dict[str, list[str]] = {
    "binary_classification": ["DummyClassifier", "LogisticRegression", "RandomForestClassifier"],
    "multiclass_classification": [
        "DummyClassifier",
        "LogisticRegression",
        "RandomForestClassifier",
    ],
    "regression": ["DummyRegressor", "Ridge", "RandomForestRegressor"],
}


class StrategyProposalV1(StrictModel):
    schema_version: Literal["strategy.v2"] = "strategy.v2"
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_version_id: str = Field(min_length=1)
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_type: TaskType = "binary_classification"
    target_column: str = Field(min_length=1)
    positive_class: str | int | float | bool | None
    feature_columns: list[str] = Field(min_length=1)
    excluded_columns: list[str]
    primary_metric: PrimaryMetric | TaskMetric
    guardrails: GuardrailDecision
    split: StrategySplit
    preprocessing: PreprocessingConfig
    candidates: list[CandidateName] = Field(min_length=3, max_length=3)
    model_params: ModelParameters
    threshold_policy: DecisionThresholdPolicy = Field(default_factory=DecisionThresholdPolicy)
    seed: int = Field(ge=0, le=2_147_483_647)
    caps: ExecutionCaps
    source: Literal["TEMPLATE"] = "TEMPLATE"
    warnings: list[str]
    human_approval_required: Literal[True] = True

    @model_validator(mode="after")
    def validate_catalog_and_columns(self) -> "StrategyProposalV1":
        if (self.task_type == "binary_classification") != (self.positive_class is not None):
            raise ValueError("Sólo un modelo sí/no tiene valor de evento, y lo requiere")
        if str(self.primary_metric) not in METRICS_BY_TASK[self.task_type]:
            raise ValueError("La métrica principal no corresponde al tipo de modelo")
        if set(self.candidates) != set(CANDIDATES_BY_TASK[self.task_type]):
            raise ValueError(
                "La estrategia P0 requiere baseline, LogisticRegression y RandomForest"
            )
        if len(self.candidates) != len(set(self.candidates)):
            raise ValueError("Los candidatos no pueden repetirse")
        normalized_features = [name.casefold() for name in self.feature_columns]
        normalized_excluded = [name.casefold() for name in self.excluded_columns]
        if len(normalized_features) != len(set(normalized_features)):
            raise ValueError("Las features no pueden repetirse")
        if len(normalized_excluded) != len(set(normalized_excluded)):
            raise ValueError("Las exclusiones no pueden repetirse")
        if set(normalized_features) & set(normalized_excluded):
            raise ValueError("Una columna no puede incluirse y excluirse a la vez")
        if self.target_column.casefold() in set(normalized_features):
            raise ValueError("El target no puede usarse como feature")
        availability = self.split.label_available_at_column
        if availability is not None:
            if self.task_type != "binary_classification":
                raise ValueError(
                    "La purga por disponibilidad sólo está soportada para modelos sí/no"
                )
            if availability.casefold() in {*normalized_features, self.target_column.casefold()}:
                raise ValueError(
                    "La disponibilidad del resultado no puede usarse como feature o target"
                )
        return self


class StrategyVersionResult(StrictModel):
    strategy_version_id: str
    context_version_id: str
    dataset_id: str
    version_no: int = Field(gt=0)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal: StrategyProposalV1
