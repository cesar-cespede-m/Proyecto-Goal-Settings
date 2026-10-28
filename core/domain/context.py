"""Contrato estricto y validado del contexto de negocio CTX-01."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Los contratos críticos rechazan campos desconocidos."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PrimaryMetric(StrEnum):
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    BALANCED_ACCURACY = "balanced_accuracy"
    AVERAGE_PRECISION = "average_precision"
    ROC_AUC = "roc_auc"


class TaskMetric(StrEnum):
    """Métricas de los tipos de modelo distintos de sí/no (ADR-016)."""

    MACRO_F1 = "macro_f1"
    MAE = "mae"
    RMSE = "rmse"


MAX_CLASSES = 20
MIN_ROWS_PER_CLASS = 5
TaskType = Literal["binary_classification", "multiclass_classification", "regression"]
METRICS_BY_TASK: dict[str, set[str]] = {
    "binary_classification": {metric.value for metric in PrimaryMetric},
    "multiclass_classification": {TaskMetric.MACRO_F1.value, PrimaryMetric.BALANCED_ACCURACY.value},
    "regression": {TaskMetric.MAE.value, TaskMetric.RMSE.value},
}


class FeatureUse(StrEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class FeatureDecision(StrictModel):
    column: str = Field(min_length=1, max_length=255)
    use: FeatureUse
    available_at_prediction: bool
    reason: str = Field(min_length=3, max_length=1000)

    @model_validator(mode="after")
    def unavailable_features_cannot_be_included(self) -> "FeatureDecision":
        if self.use is FeatureUse.INCLUDE and not self.available_at_prediction:
            raise ValueError("Una variable no disponible al predecir no puede incluirse")
        return self


class GuardrailMode(StrEnum):
    NONE = "no_quantitative_guardrail"
    QUANTITATIVE = "quantitative"


class QuantitativeGuardrail(StrictModel):
    metric: PrimaryMetric
    operator: Literal[">="] = ">="
    threshold: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=3, max_length=1000)


class GuardrailDecision(StrictModel):
    mode: GuardrailMode
    rules: list[QuantitativeGuardrail] = Field(default_factory=list)

    @model_validator(mode="after")
    def mode_matches_rules(self) -> "GuardrailDecision":
        if self.mode is GuardrailMode.NONE and self.rules:
            raise ValueError("Sin guardarraíl cuantitativo no puede contener reglas")
        if self.mode is GuardrailMode.QUANTITATIVE and not self.rules:
            raise ValueError("El modo cuantitativo requiere al menos una regla")
        metrics = [rule.metric for rule in self.rules]
        if len(metrics) != len(set(metrics)):
            raise ValueError("No se permite repetir una métrica de guardarraíl")
        return self


class BusinessContextV1(StrictModel):
    """Decisiones humanas necesarias antes de proponer una estrategia."""

    schema_version: Literal["business-context.v1"] = "business-context.v1"
    objective: str = Field(min_length=3, max_length=2000)
    target_column: str = Field(min_length=1, max_length=255)
    positive_class: str | int | float | bool | None = None
    task_type: TaskType = "binary_classification"
    prediction_moment: str = Field(min_length=3, max_length=1000)
    false_positive_consequence: str = Field(min_length=3, max_length=1000)
    false_negative_consequence: str = Field(min_length=3, max_length=1000)
    primary_metric: PrimaryMetric | TaskMetric
    metric_rationale: str = Field(min_length=3, max_length=1000)
    decision_actor_label: str = Field(min_length=1, max_length=200)
    feature_decisions: list[FeatureDecision] = Field(min_length=1)
    guardrails: GuardrailDecision

    @model_validator(mode="after")
    def feature_columns_are_unique(self) -> "BusinessContextV1":
        names = [decision.column.casefold() for decision in self.feature_decisions]
        if len(names) != len(set(names)):
            raise ValueError("Cada variable debe tener una única decisión")
        if self.target_column.casefold() in names:
            raise ValueError("El target no puede figurar como feature")
        return self

    @model_validator(mode="after")
    def fields_match_task_type(self) -> "BusinessContextV1":
        binary = self.task_type == "binary_classification"
        if binary and self.positive_class is None:
            raise ValueError("Un modelo sí/no necesita indicar qué valor es el evento")
        if not binary and self.positive_class is not None:
            raise ValueError("Sólo un modelo sí/no tiene un valor de evento")
        if str(self.primary_metric) not in METRICS_BY_TASK[self.task_type]:
            raise ValueError("La métrica principal no corresponde al tipo de modelo")
        return self

    @property
    def allowed_features(self) -> list[str]:
        return [item.column for item in self.feature_decisions if item.use is FeatureUse.INCLUDE]

    @property
    def excluded_features(self) -> list[str]:
        return [item.column for item in self.feature_decisions if item.use is FeatureUse.EXCLUDE]


class ClassSupport(StrictModel):
    value: str | int | float | bool
    count: int = Field(gt=0)


class ContextValidationSnapshot(StrictModel):
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(gt=0)
    column_count: int = Field(gt=1)
    target_missing_count: int = Field(ge=0)
    # Dos clases en sí/no, hasta 20 en varias clases y vacía en cantidades (ADR-016).
    target_distribution: list[ClassSupport] = Field(max_length=20)
    split_viability_checked_for: Literal["60/20/20_stratified"] = "60/20/20_stratified"


class ContextVersionResult(StrictModel):
    context_version_id: str
    project_id: str
    dataset_id: str
    version_no: int = Field(gt=0)
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context: BusinessContextV1
    validation: ContextValidationSnapshot
