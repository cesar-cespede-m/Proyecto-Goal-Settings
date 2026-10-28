"""Contratos Pydantic mínimos para JSON persistido."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExtensibleModel(BaseModel):
    """Contrato versionado que conserva campos futuros sin omitir validación JSON."""

    model_config = ConfigDict(extra="allow")


class DatasetProfile(ExtensibleModel):
    schema_version: str = Field(min_length=1)


class ContextPayload(ExtensibleModel):
    schema_version: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    positive_class: str | int | float | bool | None  # None sólo fuera de sí/no (ADR-016)
    prediction_moment: str = Field(min_length=1)
    excluded_columns: list[str]
    primary_metric: str = Field(min_length=1)
    guardrails: dict[str, Any] | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["stratified_random", "temporal"] = "stratified_random"
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)
    # GA-07e: sólo con method temporal; vacío no se serializa, así las huellas no cambian.
    time_column: str | None = Field(default=None, exclude_if=lambda value: value is None)
    label_available_at_column: str | None = Field(
        default=None, min_length=1, max_length=255, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "SplitConfig":
        if self.label_available_at_column is not None:
            if self.method != "temporal" or self.time_column is None:
                raise ValueError("La disponibilidad del resultado requiere partición temporal")
            if self.label_available_at_column.casefold() == self.time_column.casefold():
                raise ValueError(
                    "El momento de predicción y la disponibilidad usan columnas distintas"
                )
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("Las fracciones train/validation/test deben sumar 1")
        return self


AllowedCandidate = Literal[
    "DummyClassifier",
    "LogisticRegression",
    "RandomForestClassifier",
    "DummyRegressor",
    "Ridge",
    "RandomForestRegressor",
]


class StrategyPayload(ExtensibleModel):
    split: SplitConfig
    preprocessing: dict[str, Any]
    candidates: list[AllowedCandidate] = Field(min_length=1)
    seed: int
    caps: dict[str, Any]


class Insight(BaseModel):
    kind: Literal["FACT", "INTERPRETATION", "HYPOTHESIS", "RECOMMENDATION"]
    claim: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    limitations: list[str]
    source: Literal["ML_ENGINE", "TEMPLATE", "LLM", "HUMAN"]


class ReportPayload(ExtensibleModel):
    metrics: dict[str, Any]
    insights: list[Insight]
    limitations: list[str]


class JsonObject(BaseModel):
    """Contenedor para validar que metadata libre siga siendo un objeto JSON."""

    value: dict[str, Any]


class RunManifest(ExtensibleModel):
    schema_version: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    versions: dict[str, str]
    hyperparameters: dict[str, Any]
    features: list[str]
    warnings: list[str]
