"""Carga y revisión aislada de varios CSV con una admisión común."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.data_engine.errors import DatasetRejectedError, DataValidationError
from core.data_engine.models import (
    AuthorizationStatus,
    DatasetAdmissionRequest,
    DatasetClassification,
    IngestionLimits,
)
from core.data_engine.service import DataIngestionService, validate_and_profile
from core.db.repositories import WorkflowRepository

_ACTIONS = {
    "AMBIGUOUS_DELIMITER": "Guarda el archivo como CSV separado por comas y vuelve a revisarlo.",
    "ANONYMIZATION_NOT_DECLARED": (
        "Confirma la anonimización o deja el archivo pendiente de revisión."
    ),
    "AUTHORIZATION_PENDING": "Obtén la autorización y vuelve a confirmar la carga.",
    "AUTHORIZATION_REJECTED": "Usa otra fuente que tenga autorización vigente.",
    "BINARY_CONTENT": "Selecciona un CSV de texto en lugar de un archivo binario.",
    "CLASSIFICATION_PENDING_REVIEW": (
        "Revisa la clasificación y confirma que la fuente está permitida."
    ),
    "COMPRESSED_FILE": "Extrae el CSV permitido y vuelve a seleccionarlo.",
    "DATASET_ALREADY_REGISTERED": (
        "Usa la fuente ya registrada o selecciona una versión realmente diferente."
    ),
    "DUPLICATE_HEADERS": "Asigna un nombre único a cada columna y vuelve a cargar el archivo.",
    "EMPTY_FILE": "Incluye encabezados y al menos una fila de datos.",
    "EMPTY_HEADER": "Asigna un nombre a cada columna y vuelve a cargar el archivo.",
    "FILE_TOO_LARGE": "Reduce la muestra permitida y vuelve a cargarla.",
    "INVALID_ENCODING": "Guarda una copia como CSV UTF-8 y vuelve a cargarla.",
    "INVALID_EXTENSION": "Guarda una copia con extensión .csv y vuelve a seleccionarla.",
    "MALFORMED_CSV": "Corrige la estructura del CSV y vuelve a cargarlo.",
    "MALFORMED_ROW": "Corrige las filas para que coincidan con los encabezados.",
    "NO_DATA_ROWS": "Incluye al menos una fila de datos además de los encabezados.",
    "POSSIBLE_PERSONAL_DATA": ("Elimina esas columnas o carga una copia debidamente anonimizada."),
    "PROJECT_SOURCE_LIMIT_REACHED": (
        "Compón las fuentes disponibles o usa otro proyecto para un análisis diferente."
    ),
    "TOO_MANY_COLUMNS": "Conserva sólo las columnas necesarias y vuelve a cargar el archivo.",
    "TOO_MANY_ROWS": "Reduce la muestra al límite permitido y vuelve a cargarla.",
}


class SharedAdmissionV1(BaseModel):
    """Datos declarados una vez para todos los archivos del lote."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_name: str = Field(min_length=1, max_length=500)
    license: str | None = Field(default=None, max_length=200)
    classification: DatasetClassification
    authorization_status: AuthorizationStatus
    anonymization_declared: bool = False
    owner_label: str = Field(min_length=1, max_length=200)
    approver_label: str = Field(min_length=1, max_length=200)


class _SharedAdmissionOverridesV1(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_name: str | None = Field(default=None, min_length=1, max_length=500)
    license: str | None = Field(default=None, max_length=200)
    classification: DatasetClassification | None = None
    authorization_status: AuthorizationStatus | None = None
    anonymization_declared: bool | None = None
    owner_label: str | None = Field(default=None, min_length=1, max_length=200)
    approver_label: str | None = Field(default=None, min_length=1, max_length=200)


class BatchFileV1(BaseModel):
    """Contenido de un archivo y cambios explícitos sobre los datos compartidos."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    filename: str = Field(min_length=1, max_length=255)
    content: bytes
    overrides: _SharedAdmissionOverridesV1 = Field(default_factory=_SharedAdmissionOverridesV1)


class BatchFileResultV1(BaseModel):
    """Resultado independiente y accionable de un archivo del lote."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    status: Literal["accepted", "rejected", "duplicate", "error"]
    dataset_id: str | None = None
    rows: int | None = Field(default=None, ge=0)
    columns: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)
    code: str | None = None
    message: str | None = None

    @property
    def row_count(self) -> int | None:
        return self.rows

    @property
    def column_count(self) -> int | None:
        return self.columns


class BatchResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[BatchFileResultV1]

    @property
    def results(self) -> list[BatchFileResultV1]:
        return self.files


def _message(code: str, detail: str) -> str:
    action = _ACTIONS.get(
        code,
        "Revisa los datos declarados y el archivo antes de volver a intentarlo.",
    )
    return f"{detail.rstrip('.')}. {action}"


def _pending_reason(request: DatasetAdmissionRequest) -> tuple[str, str] | None:
    if request.authorization_status is AuthorizationStatus.REJECTED:
        return "AUTHORIZATION_REJECTED", "La autorización declarada fue rechazada"
    if request.authorization_status is AuthorizationStatus.PENDING:
        return "AUTHORIZATION_PENDING", "La autorización todavía está pendiente"
    if request.classification is DatasetClassification.PENDING_REVIEW:
        return "CLASSIFICATION_PENDING_REVIEW", "La clasificación todavía está pendiente"
    if (
        request.classification is DatasetClassification.ANONYMIZED
        and not request.anonymization_declared
    ):
        return "ANONYMIZATION_NOT_DECLARED", "La anonimización no fue declarada"
    return None


class BatchIngestionService:
    """Previsualiza y admite un lote sin propagar el fallo de un archivo a los demás."""

    def __init__(
        self,
        repository: WorkflowRepository,
        storage_root: str | Path,
        *,
        limits: IngestionLimits | None = None,
    ) -> None:
        self.repository = repository
        self.ingestion = DataIngestionService(repository, storage_root, limits=limits)
        self.limits = self.ingestion.limits

    @staticmethod
    def _models(
        files: list[BatchFileV1 | dict[str, Any]],
        shared: SharedAdmissionV1 | dict[str, Any],
    ) -> tuple[list[BatchFileV1], SharedAdmissionV1]:
        return (
            [BatchFileV1.model_validate(item) for item in files],
            SharedAdmissionV1.model_validate(shared),
        )

    @staticmethod
    def _request(item: BatchFileV1, shared: SharedAdmissionV1) -> DatasetAdmissionRequest:
        effective = shared.model_dump()
        effective.update(item.overrides.model_dump(exclude_unset=True))
        values = {
            key: value
            for key, value in effective.items()
            if key not in {"owner_label", "approver_label"}
        }
        return DatasetAdmissionRequest(
            original_filename_display=item.filename,
            **values,
        )

    @staticmethod
    def _rejected(
        filename: str,
        code: str,
        detail: str,
        *,
        warnings: list[str] | None = None,
        duplicate: bool = False,
    ) -> BatchFileResultV1:
        return BatchFileResultV1(
            filename=filename,
            status="duplicate" if duplicate else "rejected",
            warnings=warnings or [],
            code=code,
            message=_message(code, detail),
        )

    def preview(
        self,
        files: list[BatchFileV1 | dict[str, Any]],
        shared: SharedAdmissionV1 | dict[str, Any],
    ) -> BatchResultV1:
        """Valida y perfila en memoria; no toca repositorio, auditoría ni storage."""

        items, shared_model = self._models(files, shared)
        results: list[BatchFileResultV1] = []
        for item in items:
            request = self._request(item, shared_model)
            pending = _pending_reason(request)
            if pending is not None:
                code, detail = pending
                results.append(self._rejected(item.filename, code, detail))
                continue
            try:
                profile = validate_and_profile(
                    item.filename,
                    item.content,
                    classification=request.classification,
                    limits=self.limits,
                )
                personal = [
                    warning
                    for warning in profile.warnings
                    if warning.startswith("POSSIBLE_PERSONAL_DATA_COLUMNS:")
                ]
                if personal:
                    results.append(
                        self._rejected(
                            item.filename,
                            "POSSIBLE_PERSONAL_DATA",
                            "El archivo contiene columnas que podrían identificar personas",
                            warnings=profile.warnings,
                        )
                    )
                else:
                    results.append(
                        BatchFileResultV1(
                            filename=item.filename,
                            status="accepted",
                            rows=profile.row_count,
                            columns=profile.column_count,
                            warnings=profile.warnings,
                        )
                    )
            except DataValidationError as error:
                results.append(self._rejected(item.filename, error.code, str(error)))
            except Exception:
                results.append(
                    BatchFileResultV1(
                        filename=item.filename,
                        status="error",
                        code="BATCH_INGESTION_FAILED",
                        message=(
                            "No pudimos revisar este archivo. "
                            "Vuelve a seleccionarlo o revisa el registro del incidente."
                        ),
                    )
                )
        return BatchResultV1(files=results)

    def ingest(
        self,
        project_id: str,
        files: list[BatchFileV1 | dict[str, Any]],
        shared: SharedAdmissionV1 | dict[str, Any],
    ) -> BatchResultV1:
        items, shared_model = self._models(files, shared)
        if self.repository.get_project_roles(project_id) is None:
            self.repository.set_project_roles(
                project_id,
                owner_label=shared_model.owner_label,
                approver_label=shared_model.approver_label,
            )

        results: list[BatchFileResultV1] = []
        for item in items:
            try:
                request = self._request(item, shared_model)
                admitted = self.ingestion.ingest_bytes(
                    project_id=project_id,
                    request=request,
                    content=item.content,
                )
                if admitted.metadata.admission_status != "ACCEPTED":
                    pending = _pending_reason(request)
                    code, detail = pending or (
                        "CLASSIFICATION_PENDING_REVIEW",
                        "El archivo quedó pendiente de revisión",
                    )
                    results.append(self._rejected(item.filename, code, detail))
                    continue
                profile = admitted.profile
                results.append(
                    BatchFileResultV1(
                        filename=item.filename,
                        status="accepted",
                        dataset_id=admitted.metadata.dataset_id,
                        rows=admitted.metadata.row_count,
                        columns=admitted.metadata.column_count,
                        warnings=[] if profile is None else profile.warnings,
                    )
                )
            except DatasetRejectedError as error:
                results.append(
                    self._rejected(
                        item.filename,
                        error.code,
                        str(error),
                        warnings=error.warnings,
                    )
                )
            except DataValidationError as error:
                results.append(
                    self._rejected(
                        item.filename,
                        error.code,
                        str(error),
                        duplicate=error.code == "DATASET_ALREADY_REGISTERED",
                    )
                )
            except Exception:
                results.append(
                    BatchFileResultV1(
                        filename=item.filename,
                        status="error",
                        code="BATCH_INGESTION_FAILED",
                        message=(
                            "No pudimos procesar este archivo. "
                            "Vuelve a intentarlo o revisa el registro del incidente."
                        ),
                    )
                )
        return BatchResultV1(files=results)
