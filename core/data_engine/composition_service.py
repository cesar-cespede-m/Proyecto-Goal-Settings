"""Persistencia atómica del linaje y snapshot derivado de una composición."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

from core.data_engine.composition import (
    CSVSourceInput,
    DatasetCompositionResult,
    compose_csv_datasets,
)
from core.data_engine.errors import DatasetCompositionError
from core.data_engine.models import (
    AuxiliaryJoinV1,
    CompositionSourceRefV1,
    DatasetClassification,
    DatasetCompositionRequestV1,
    DatasetCompositionSpecV1,
    DatasetProfileV1,
    PersistedDatasetCompositionV1,
)
from core.db.repositories import WorkflowRepository

_PENDING_ACTORS = {"", "pendiente", "pending", "todo"}


@dataclass(frozen=True)
class _PreparedComposition:
    model: DatasetCompositionRequestV1
    spec: DatasetCompositionSpecV1
    result: DatasetCompositionResult
    source_entries: list[dict[str, object]]


class DatasetCompositionService:
    def __init__(self, repository: WorkflowRepository, storage_root: str | Path) -> None:
        self.repository = repository
        self.storage_root = Path(storage_root)

    def _source(self, project_id: str, dataset_id: str):
        row = self.repository.get_dataset(dataset_id)
        if (
            row["project_id"] != project_id
            or row["admission_status"] != "ACCEPTED"
            or row["dataset_kind"] != "SOURCE"
        ):
            raise DatasetCompositionError(
                "SOURCE_NOT_AVAILABLE",
                "La fuente no es un CSV aceptado del proyecto",
            )
        profile = DatasetProfileV1.model_validate(json.loads(row["profile_json"]))
        root = self.storage_root.resolve()
        path = (root / Path(str(row["stored_path"]))).resolve()
        if root not in path.parents or not path.is_file():
            raise DatasetCompositionError(
                "SOURCE_FILE_UNAVAILABLE",
                "El archivo privado de una fuente no está disponible",
            )
        return (
            row,
            profile,
            CSVSourceInput(
                filename=str(row["original_filename_display"]),
                content=path.read_bytes(),
            ),
        )

    def _prepare(
        self,
        *,
        project_id: str,
        request: DatasetCompositionRequestV1 | dict[str, object],
    ) -> _PreparedComposition:
        model = DatasetCompositionRequestV1.model_validate(request)
        if model.actor_label.strip().casefold() in _PENDING_ACTORS:
            raise DatasetCompositionError(
                "COMPOSITION_ACTOR_REQUIRED",
                "La composición exige una persona o rol responsable",
            )

        primary, primary_profile, primary_input = self._source(project_id, model.primary_dataset_id)
        sources = {model.primary_dataset_id: primary_input}
        source_entries: list[dict[str, object]] = [
            {
                "position": 0,
                "dataset_id": model.primary_dataset_id,
                "role": "PRIMARY",
                "source_sha256": str(primary["sha256"]),
            }
        ]
        joins: list[AuxiliaryJoinV1] = []
        for position, requested in enumerate(model.joins, start=1):
            row, profile, source_input = self._source(project_id, requested.dataset_id)
            sources[requested.dataset_id] = source_input
            joins.append(
                AuxiliaryJoinV1(
                    source=CompositionSourceRefV1(
                        dataset_id=requested.dataset_id,
                        dataset_sha256=str(row["sha256"]),
                        classification=profile.data_classification,
                    ),
                    alias=requested.alias,
                    primary_key=requested.primary_key,
                    auxiliary_key=requested.auxiliary_key,
                    cardinality=requested.cardinality,
                )
            )
            source_entries.append(
                {
                    "position": position,
                    "dataset_id": requested.dataset_id,
                    "role": "AUXILIARY",
                    "alias": requested.alias,
                    "source_sha256": str(row["sha256"]),
                    "primary_key": requested.primary_key,
                    "auxiliary_key": requested.auxiliary_key,
                    "cardinality": requested.cardinality,
                }
            )
        spec = DatasetCompositionSpecV1(
            primary=CompositionSourceRefV1(
                dataset_id=model.primary_dataset_id,
                dataset_sha256=str(primary["sha256"]),
                classification=primary_profile.data_classification,
            ),
            joins=joins,
        )
        result = compose_csv_datasets(spec, sources)
        return _PreparedComposition(
            model=model,
            spec=spec,
            result=result,
            source_entries=source_entries,
        )

    def preview(
        self,
        *,
        project_id: str,
        request: DatasetCompositionRequestV1 | dict[str, object],
    ) -> DatasetCompositionResult:
        return self._prepare(project_id=project_id, request=request).result

    def create(
        self,
        *,
        project_id: str,
        request: DatasetCompositionRequestV1 | dict[str, object],
    ) -> PersistedDatasetCompositionV1:
        prepared = self._prepare(project_id=project_id, request=request)
        model = prepared.model
        spec = prepared.spec
        result = prepared.result

        composition_id = str(uuid4())
        output_dataset_id = str(uuid4())
        relative = PurePosixPath("data", "accepted", output_dataset_id)
        root = self.storage_root.resolve()
        destination = (root / Path(relative)).resolve()
        if root not in destination.parents:
            raise DatasetCompositionError(
                "OUTPUT_PATH_INVALID", "La ruta derivada salió del storage privado"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(result.content)
        temporary.replace(destination)
        try:
            self.repository.persist_dataset_composition(
                composition_id=composition_id,
                project_id=project_id,
                output_dataset_id=output_dataset_id,
                output_sha256=result.summary.output_sha256,
                output_bytes=result.summary.output_bytes,
                output_rows=result.profile.row_count,
                output_columns=result.profile.column_count,
                output_anonymization_declared=(
                    result.profile.data_classification is DatasetClassification.ANONYMIZED
                ),
                output_profile=result.profile.model_dump(mode="json"),
                spec=spec.model_dump(mode="json"),
                summary=result.summary.model_dump(mode="json"),
                source_entries=prepared.source_entries,
                actor_label=model.actor_label,
            )
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return PersistedDatasetCompositionV1(
            composition_id=composition_id,
            output_dataset_id=output_dataset_id,
            profile=result.profile,
            summary=result.summary,
        )
