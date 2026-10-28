"""Contratos Pydantic de DATA-01."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

MAX_COMPOSITION_SOURCES = 5
MAX_COMPOSITION_INPUT_BYTES = 50 * 1024 * 1024


class DatasetClassification(StrEnum):
    SYNTHETIC = "synthetic"
    SANDBOX = "sandbox"
    ANONYMIZED = "anonymized"
    PENDING_REVIEW = "pending_review"


class AuthorizationStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class IngestionLimits(BaseModel):
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    max_rows: int = Field(default=100_000, gt=0)
    max_columns: int = Field(default=100, gt=0)
    high_cardinality_ratio: float = Field(default=0.9, gt=0, le=1)
    high_cardinality_min_unique: int = Field(default=20, gt=1)


class DatasetAdmissionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    original_filename_display: str = Field(min_length=1, max_length=255)
    source_name: str = Field(min_length=1, max_length=500)
    source_url: HttpUrl | None = None
    license: str | None = Field(default=None, max_length=200)
    classification: DatasetClassification
    authorization_status: AuthorizationStatus
    anonymization_declared: bool = False


class NumericSummary(BaseModel):
    minimum: float | None
    maximum: float | None
    mean: float | None


class ColumnProfile(BaseModel):
    name: str
    inferred_type: Literal["boolean", "numeric", "categorical", "string"]
    missing_count: int = Field(ge=0)
    missing_rate: float = Field(ge=0, le=1)
    unique_count: int = Field(ge=0)
    constant: bool
    high_cardinality: bool
    possible_identifier: bool
    numeric_summary: NumericSummary | None = None
    # GA-01: tipo para el análisis; None en perfiles guardados antes (ver analysis_schema).
    semantic_type: (
        Literal["number", "category", "datetime", "text", "identifier", "personal"] | None
    ) = None


class OutcomeAssociationV1(BaseModel):
    """Señal agregada de que una columna podría revelar un resultado.

    No guarda filas ni valores del archivo. Los perfiles anteriores no traen esta lista y
    siguen siendo válidos porque `DatasetProfileV1` usa una lista vacía por defecto.
    """

    target_column: str
    feature_column: str
    signal: Literal["categorical_separation", "numeric_correlation"]
    strength: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)


class DatasetProfileV1(BaseModel):
    schema_version: Literal["dataset-profile.v1"] = "dataset-profile.v1"
    data_classification: DatasetClassification
    row_count: int = Field(gt=0)
    column_count: int = Field(gt=0)
    duplicate_row_count: int = Field(ge=0)
    columns: list[ColumnProfile]
    numeric_columns: list[str]
    categorical_columns: list[str]
    high_cardinality_columns: list[str]
    constant_columns: list[str]
    warnings: list[str]
    outcome_associations: list[OutcomeAssociationV1] = Field(default_factory=list)


class DatasetMetadata(BaseModel):
    dataset_id: str
    project_id: str
    sha256_raw: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_filename_display: str
    storage_key_private: str
    source_name: str
    source_url: str | None
    license: str | None
    classification: DatasetClassification
    authorization_status: AuthorizationStatus
    admission_status: Literal["QUARANTINED", "ACCEPTED", "REJECTED"]
    file_size_bytes: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)
    column_count: int | None = Field(default=None, ge=0)
    schema_version: Literal["dataset-metadata.v1"] = "dataset-metadata.v1"
    created_at_utc: datetime


class AdmissionResult(BaseModel):
    metadata: DatasetMetadata
    profile: DatasetProfileV1 | None


class CompositionSourceRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_id: str = Field(min_length=1, max_length=100)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    classification: DatasetClassification

    @field_validator("classification")
    @classmethod
    def source_is_permitted(cls, value: DatasetClassification) -> DatasetClassification:
        if value is DatasetClassification.PENDING_REVIEW:
            raise ValueError("Una fuente pendiente de revisión no puede componerse")
        return value


class AuxiliaryJoinV1(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source: CompositionSourceRefV1
    alias: str = Field(pattern=r"^[a-z][a-z0-9_]{0,19}$")
    primary_key: str = Field(min_length=1, max_length=255)
    auxiliary_key: str = Field(min_length=1, max_length=255)
    cardinality: Literal["many_to_one", "one_to_one"] = "many_to_one"


class DatasetCompositionSpecV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dataset-composition.v1"] = "dataset-composition.v1"
    primary: CompositionSourceRefV1
    joins: list[AuxiliaryJoinV1] = Field(min_length=1, max_length=4)
    join_type: Literal["left"] = "left"

    @model_validator(mode="after")
    def sources_and_aliases_are_unique(self) -> "DatasetCompositionSpecV1":
        source_ids = [self.primary.dataset_id, *(item.source.dataset_id for item in self.joins)]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Cada dataset puede aparecer una sola vez en la composición")
        aliases = [item.alias.casefold() for item in self.joins]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Cada dataset auxiliar requiere un alias único")
        return self


class CompositionLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_sources: int = Field(default=MAX_COMPOSITION_SOURCES, ge=2, le=MAX_COMPOSITION_SOURCES)
    max_total_input_bytes: int = Field(
        default=MAX_COMPOSITION_INPUT_BYTES,
        gt=0,
        le=MAX_COMPOSITION_INPUT_BYTES,
    )


class JoinStatisticsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auxiliary_dataset_id: str
    alias: str
    primary_rows: int = Field(ge=0)
    auxiliary_rows: int = Field(ge=0)
    matched_primary_rows: int = Field(ge=0)
    unmatched_primary_rows: int = Field(ge=0)
    unmatched_auxiliary_rows: int = Field(ge=0)
    output_rows: int = Field(ge=0)


class DatasetCompositionSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dataset-composition-summary.v1"] = "dataset-composition-summary.v1"
    composition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_count: int = Field(ge=2, le=MAX_COMPOSITION_SOURCES)
    total_input_bytes: int = Field(gt=0, le=MAX_COMPOSITION_INPUT_BYTES)
    output_bytes: int = Field(gt=0)
    protected_join_columns: list[str]
    joins: list[JoinStatisticsV1] = Field(min_length=1, max_length=4)


class CompositionJoinRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_id: str = Field(min_length=1, max_length=100)
    alias: str = Field(pattern=r"^[a-z][a-z0-9_]{0,19}$")
    primary_key: str = Field(min_length=1, max_length=255)
    auxiliary_key: str = Field(min_length=1, max_length=255)
    cardinality: Literal["many_to_one", "one_to_one"] = "many_to_one"


class DatasetCompositionRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    primary_dataset_id: str = Field(min_length=1, max_length=100)
    joins: list[CompositionJoinRequestV1] = Field(min_length=1, max_length=4)
    actor_label: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def request_sources_and_aliases_are_unique(self) -> "DatasetCompositionRequestV1":
        source_ids = [self.primary_dataset_id, *(item.dataset_id for item in self.joins)]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Cada dataset puede aparecer una sola vez en la composición")
        aliases = [item.alias.casefold() for item in self.joins]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Cada dataset auxiliar requiere un alias único")
        return self


class PersistedDatasetCompositionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    composition_id: str
    output_dataset_id: str
    profile: DatasetProfileV1
    summary: DatasetCompositionSummaryV1
