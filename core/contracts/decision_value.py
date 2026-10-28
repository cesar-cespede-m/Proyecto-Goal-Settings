"""MV-05: especificaciones declaradas y versionadas, separadas de los gates del experimento."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from core.analysis.snapshot import SnapshotSpecV1
from core.contracts.economics import EconomicScenarioV1
from core.domain.canonical import canonical_hash
from core.domain.context import FeatureDecision, PrimaryMetric, StrictModel


class FixedDaysWindowV1(StrictModel):
    kind: Literal["fixed_days"] = "fixed_days"
    days: int = Field(gt=0, le=3650, strict=True)


class DeadlineWindowV1(StrictModel):
    kind: Literal["deadline_column"] = "deadline_column"
    column: str = Field(min_length=1, max_length=255)


class _DraftSpec(StrictModel):
    @model_validator(mode="before")
    @classmethod
    def empty_values_are_missing(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: None if isinstance(item, str) and not item.strip() else item
                for key, item in value.items()
            }
        return value


class PredictiveSpecV1(_DraftSpec):
    schema_version: Literal["predictive-spec.v1"] = "predictive-spec.v1"
    task_type: Literal["binary_classification"] = "binary_classification"
    population: str | None = Field(default=None, min_length=1, max_length=2000)
    target_definition: str | None = Field(default=None, min_length=1, max_length=2000)
    target_column: str | None = Field(default=None, min_length=1, max_length=255)
    positive_class: str | int | float | bool | None = None
    prediction_moment: str | None = Field(default=None, min_length=1, max_length=1000)
    primary_metric: PrimaryMetric | None = None
    outcome_window: (
        Annotated[FixedDaysWindowV1 | DeadlineWindowV1, Field(discriminator="kind")] | None
    ) = None
    feature_decisions: list[FeatureDecision] = Field(default_factory=list)
    source_context_version_id: str | None = None
    source_context_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    snapshot_recipe: SnapshotSpecV1 | None = None

    @model_validator(mode="after")
    def consistent_features_and_origin(self) -> "PredictiveSpecV1":
        names = [item.column.casefold() for item in self.feature_decisions]
        if len(names) != len(set(names)):
            raise ValueError("No repitas decisiones de una variable")
        if self.target_column is not None and self.target_column.casefold() in names:
            raise ValueError("El target no puede ser una feature")
        if (self.source_context_version_id is None) != (self.source_context_sha256 is None):
            raise ValueError("El origen requiere contexto y hash juntos")
        self.feature_decisions.sort(key=lambda item: item.column.casefold())
        return self

    def missing_fields(self) -> list[str]:
        fields = (
            "population",
            "target_definition",
            "target_column",
            "positive_class",
            "prediction_moment",
            "primary_metric",
            "outcome_window",
        )
        missing = [field for field in fields if getattr(self, field) is None]
        if not any(item.use == "include" for item in self.feature_decisions):
            missing.append("feature_decisions")
        return missing

    @property
    def predictive_fingerprint(self) -> str:
        """La procedencia de un ciclo cambia el hash íntegro, no la definición predictiva."""

        return canonical_hash(
            self.model_dump(
                mode="json", exclude={"source_context_version_id", "source_context_sha256"}
            )
        )


class DecisionValueSpecV1(_DraftSpec):
    schema_version: Literal["decision-value-spec.v1"] = "decision-value-spec.v1"
    question: str | None = Field(default=None, min_length=1, max_length=2000)
    action: str | None = Field(default=None, min_length=1, max_length=2000)
    alternative: str | None = Field(default=None, min_length=1, max_length=2000)
    kpi: str | None = Field(default=None, min_length=1, max_length=1000)
    owner_label: str | None = Field(default=None, min_length=1, max_length=200)
    constraints: list[str] = Field(default_factory=list)
    predictive_version_id: str | None = None
    economic_version_id: str | None = None

    def missing_fields(self) -> list[str]:
        fields = (
            "question",
            "action",
            "alternative",
            "kpi",
            "owner_label",
            "predictive_version_id",
            "economic_version_id",
        )
        return [field for field in fields if getattr(self, field) is None]


class _Version(StrictModel):
    id: str
    capability_id: str
    version_no: int = Field(gt=0)
    based_on_version_id: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmed_human: bool
    actor_label: str
    created_at: datetime
    missing_fields: list[str]


class PredictiveSpecVersionV1(_Version):
    spec: PredictiveSpecV1

    @property
    def predictive_fingerprint(self) -> str:
        return self.spec.predictive_fingerprint


class EconomicScenarioVersionV1(_Version):
    scenario: EconomicScenarioV1


class DecisionValueSpecVersionV1(_Version):
    spec: DecisionValueSpecV1


class CyclePredictiveBindingV1(StrictModel):
    id: str
    capability_id: str
    cycle_id: str
    predictive_version_id: str
    actor_label: str
    created_at: datetime
