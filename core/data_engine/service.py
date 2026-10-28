"""Orquestación de cuarentena, admisión y perfilado."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from uuid import uuid4

from core.data_engine.errors import DatasetRejectedError, DataValidationError
from core.data_engine.models import (
    MAX_COMPOSITION_SOURCES,
    AdmissionResult,
    AuthorizationStatus,
    DatasetAdmissionRequest,
    DatasetClassification,
    DatasetMetadata,
    DatasetProfileV1,
    IngestionLimits,
)
from core.data_engine.profiler import build_profile
from core.data_engine.validator import (
    read_limited_stream,
    validate_csv_bytes,
    validate_upload_envelope,
)
from core.db.repositories import WorkflowRepository


def validate_and_profile(
    filename: str,
    content: bytes,
    *,
    classification: DatasetClassification = DatasetClassification.SYNTHETIC,
    limits: IngestionLimits | None = None,
) -> DatasetProfileV1:
    """API pura de DATA-01 para CLI y pruebas, sin persistencia ni UI."""

    effective_limits = limits or IngestionLimits()
    validated = validate_csv_bytes(filename, content, effective_limits)
    return build_profile(
        validated,
        classification=classification,
        limits=effective_limits,
    )


class DataIngestionService:
    """Admite datasets permitidos y mantiene los inciertos en cuarentena."""

    def __init__(
        self,
        repository: WorkflowRepository,
        storage_root: str | Path,
        *,
        limits: IngestionLimits | None = None,
    ) -> None:
        self.repository = repository
        self.storage_root = Path(storage_root)
        self.limits = limits or IngestionLimits()

    def _storage_path(self, area: str, dataset_id: str) -> tuple[str, Path]:
        relative = str(PurePosixPath("data", area, dataset_id))
        absolute = (self.storage_root / Path(relative)).resolve()
        root = self.storage_root.resolve()
        if root not in absolute.parents:
            raise RuntimeError("La ruta generada salió del storage privado")
        return relative, absolute

    @staticmethod
    def _source_value(request: DatasetAdmissionRequest) -> str:
        if request.source_url is None:
            return request.source_name
        return f"{request.source_name} | {request.source_url}"

    @staticmethod
    def _pending_reason(request: DatasetAdmissionRequest) -> str | None:
        if request.authorization_status is AuthorizationStatus.PENDING:
            return "AUTHORIZATION_PENDING"
        if request.classification is DatasetClassification.PENDING_REVIEW:
            return "CLASSIFICATION_PENDING_REVIEW"
        if (
            request.classification is DatasetClassification.ANONYMIZED
            and not request.anonymization_declared
        ):
            return "ANONYMIZATION_NOT_DECLARED"
        return None

    def _metadata(
        self,
        *,
        request: DatasetAdmissionRequest,
        dataset_id: str,
        sha256_raw: str,
        storage_key: str,
        admission_status: str,
        byte_count: int,
        row_count: int | None,
        column_count: int | None,
    ) -> DatasetMetadata:
        row = self.repository.get_dataset(dataset_id)
        created_at = datetime.fromisoformat(str(row["created_at"]))
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return DatasetMetadata(
            dataset_id=dataset_id,
            project_id=str(row["project_id"]),
            sha256_raw=sha256_raw,
            original_filename_display=request.original_filename_display,
            storage_key_private=storage_key,
            source_name=request.source_name,
            source_url=None if request.source_url is None else str(request.source_url),
            license=request.license,
            classification=request.classification,
            authorization_status=request.authorization_status,
            admission_status=admission_status,
            file_size_bytes=byte_count,
            row_count=row_count,
            column_count=column_count,
            created_at_utc=created_at,
        )

    def _register(
        self,
        *,
        request: DatasetAdmissionRequest,
        project_id: str,
        dataset_id: str,
        sha256_raw: str,
        admission_status: str,
        rejection_reason: str | None,
        row_count: int | None,
        column_count: int | None,
        byte_count: int,
        profile: dict[str, Any],
    ) -> None:
        self.repository.add_dataset(
            project_id=project_id,
            dataset_id=dataset_id,
            sha256=sha256_raw,
            original_filename_display=request.original_filename_display,
            source=self._source_value(request),
            license_name=request.license,
            anonymization_declared=request.anonymization_declared,
            admission_status=admission_status,
            rejection_reason=rejection_reason,
            n_rows=row_count,
            n_cols=column_count,
            byte_count=byte_count,
            profile=profile,
        )

    def ingest_bytes(
        self,
        *,
        project_id: str,
        request: DatasetAdmissionRequest | dict[str, Any],
        content: bytes,
    ) -> AdmissionResult:
        """Persiste metadata y nunca usa el nombre recibido para construir paths."""

        request_model = DatasetAdmissionRequest.model_validate(request)
        will_be_accepted = (
            request_model.authorization_status is AuthorizationStatus.APPROVED
            and self._pending_reason(request_model) is None
        )
        dataset_id = str(uuid4())
        sha256_raw = hashlib.sha256(content).hexdigest()
        if any(row["sha256"] == sha256_raw for row in self.repository.list_datasets(project_id)):
            raise DataValidationError(
                "DATASET_ALREADY_REGISTERED",
                "Este mismo contenido ya está registrado en el proyecto",
            )
        accepted_sources = self.repository.list_datasets(
            project_id,
            admission_status="ACCEPTED",
            dataset_kind="SOURCE",
        )
        if will_be_accepted and len(accepted_sources) >= MAX_COMPOSITION_SOURCES:
            raise DataValidationError(
                "PROJECT_SOURCE_LIMIT_REACHED",
                f"El proyecto admite como máximo {MAX_COMPOSITION_SOURCES} fuentes CSV",
            )
        quarantine_key, quarantine_path = self._storage_path("quarantine", dataset_id)
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_path.write_bytes(content)

        try:
            validate_upload_envelope(
                request_model.original_filename_display,
                content,
                self.limits,
            )
        except DataValidationError as error:
            self._register(
                request=request_model,
                project_id=project_id,
                dataset_id=dataset_id,
                sha256_raw=sha256_raw,
                admission_status="REJECTED",
                rejection_reason=error.code,
                row_count=None,
                column_count=None,
                byte_count=len(content),
                profile={
                    "schema_version": "dataset-profile.v1",
                    "status": "NOT_PROFILED",
                    "reason": error.code,
                    "data_classification": request_model.classification.value,
                },
            )
            raise DatasetRejectedError(
                error.code,
                str(error),
                dataset_id=dataset_id,
            ) from error

        if request_model.authorization_status is AuthorizationStatus.REJECTED:
            reason = "AUTHORIZATION_REJECTED"
            self._register(
                request=request_model,
                project_id=project_id,
                dataset_id=dataset_id,
                sha256_raw=sha256_raw,
                admission_status="REJECTED",
                rejection_reason=reason,
                row_count=None,
                column_count=None,
                byte_count=len(content),
                profile={
                    "schema_version": "dataset-profile.v1",
                    "status": "NOT_PROFILED",
                    "reason": reason,
                    "data_classification": request_model.classification.value,
                },
            )
            raise DatasetRejectedError(
                reason,
                "La autorización del dataset fue rechazada",
                dataset_id=dataset_id,
            )

        pending_reason = self._pending_reason(request_model)
        if pending_reason is not None:
            self._register(
                request=request_model,
                project_id=project_id,
                dataset_id=dataset_id,
                sha256_raw=sha256_raw,
                admission_status="QUARANTINED",
                rejection_reason=None,
                row_count=None,
                column_count=None,
                byte_count=len(content),
                profile={
                    "schema_version": "dataset-profile.v1",
                    "status": "NOT_PROFILED",
                    "reason": pending_reason,
                    "data_classification": request_model.classification.value,
                },
            )
            metadata = self._metadata(
                request=request_model,
                dataset_id=dataset_id,
                sha256_raw=sha256_raw,
                storage_key=quarantine_key,
                admission_status="QUARANTINED",
                byte_count=len(content),
                row_count=None,
                column_count=None,
            )
            return AdmissionResult(metadata=metadata, profile=None)

        try:
            validated = validate_csv_bytes(
                request_model.original_filename_display,
                content,
                self.limits,
            )
            profile = build_profile(
                validated,
                classification=request_model.classification,
                limits=self.limits,
            )
        except DataValidationError as error:
            self._register(
                request=request_model,
                project_id=project_id,
                dataset_id=dataset_id,
                sha256_raw=sha256_raw,
                admission_status="REJECTED",
                rejection_reason=error.code,
                row_count=None,
                column_count=None,
                byte_count=len(content),
                profile={
                    "schema_version": "dataset-profile.v1",
                    "status": "NOT_PROFILED",
                    "reason": error.code,
                    "data_classification": request_model.classification.value,
                },
            )
            raise DatasetRejectedError(
                error.code,
                str(error),
                dataset_id=dataset_id,
            ) from error

        personal_warnings = [
            warning
            for warning in profile.warnings
            if warning.startswith("POSSIBLE_PERSONAL_DATA_COLUMNS:")
        ]
        if personal_warnings:
            reason = "POSSIBLE_PERSONAL_DATA"
            self._register(
                request=request_model,
                project_id=project_id,
                dataset_id=dataset_id,
                sha256_raw=sha256_raw,
                admission_status="REJECTED",
                rejection_reason=reason,
                row_count=profile.row_count,
                column_count=profile.column_count,
                byte_count=len(content),
                profile=profile.model_dump(mode="json"),
            )
            raise DatasetRejectedError(
                reason,
                "El archivo contiene columnas que podrían identificar personas",
                dataset_id=dataset_id,
                warnings=profile.warnings,
            )

        accepted_key, accepted_path = self._storage_path("accepted", dataset_id)
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_path.replace(accepted_path)
        try:
            self._register(
                request=request_model,
                project_id=project_id,
                dataset_id=dataset_id,
                sha256_raw=sha256_raw,
                admission_status="ACCEPTED",
                rejection_reason=None,
                row_count=profile.row_count,
                column_count=profile.column_count,
                byte_count=len(content),
                profile=profile.model_dump(mode="json"),
            )
        except Exception:
            accepted_path.replace(quarantine_path)
            raise
        metadata = self._metadata(
            request=request_model,
            dataset_id=dataset_id,
            sha256_raw=sha256_raw,
            storage_key=accepted_key,
            admission_status="ACCEPTED",
            byte_count=len(content),
            row_count=profile.row_count,
            column_count=profile.column_count,
        )
        return AdmissionResult(metadata=metadata, profile=profile)

    def ingest_stream(
        self,
        *,
        project_id: str,
        request: DatasetAdmissionRequest | dict[str, Any],
        stream: BinaryIO,
    ) -> AdmissionResult:
        """Limita la lectura del upload antes de delegar la admisión."""

        content = read_limited_stream(stream, self.limits)
        return self.ingest_bytes(project_id=project_id, request=request, content=content)
