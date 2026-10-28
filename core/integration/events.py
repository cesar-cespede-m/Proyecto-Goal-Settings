"""Captura agnóstica y materialización de eventos observados (C2-01)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict

from core.contracts.platform import ObservationEventV1
from core.data_engine.models import DatasetClassification
from core.data_engine.semantics import looks_personal
from core.data_engine.service import validate_and_profile
from core.db.repositories import WorkflowRepository


class EventIngestionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    accepted: int
    duplicates: int
    rejected: list[tuple[int, str]]


def _required_label(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} no puede estar vacío")
    return normalized


def _storage_event(event: ObservationEventV1) -> dict[str, Any]:
    outcome_json = None
    if event.outcome is not None:
        outcome_json = json.dumps(
            {
                "value": event.outcome,
                "observed_at": event.outcome_observed_at.isoformat(),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    attributes_json = json.dumps(
        event.attributes, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    # Huella del contenido: un reenvío idéntico se descarta; una versión nueva del mismo caso
    # (por ejemplo, cuando llega su resultado) se guarda aparte.
    fingerprint = hashlib.sha256(
        json.dumps(
            [
                event.source,
                event.entity_type,
                event.entity_id,
                event.occurred_at.isoformat(),
                attributes_json,
                outcome_json,
            ],
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "id": fingerprint,
        "entity_id": event.entity_id,
        "source": event.source,
        "entity": event.entity_type,
        "occurred_at": event.occurred_at.isoformat(),
        "attributes_json": json.dumps(
            event.attributes,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "outcome_json": outcome_json,
    }


def _has_personal_attribute(event: ObservationEventV1) -> bool:
    return any(looks_personal(name, pd.Series([value])) for name, value in event.attributes.items())


def _attribute_columns(rows: list[dict[str, Any]], *, include_outcome: bool) -> dict[str, str]:
    reserved = {"id", "occurred_at"}
    if include_outcome:
        reserved.update({"outcome", "outcome_observed_at"})
    used = {name.casefold() for name in reserved}
    mapping: dict[str, str] = {}
    keys = sorted({key for row in rows for key in row["attributes"]}, key=str.casefold)
    for position, key in enumerate(keys, start=1):
        candidate = key.strip() or f"attribute_{position}"
        if candidate.casefold() in used:
            candidate = f"attribute__{candidate}"
        suffix = 2
        unique = candidate
        while unique.casefold() in used:
            unique = f"{candidate}_{suffix}"
            suffix += 1
        mapping[key] = unique
        used.add(unique.casefold())
    return mapping


class EventIngestionService:
    """Valida eventos, evita duplicados y los convierte en una fuente del catálogo."""

    def __init__(self, repository: WorkflowRepository, storage_root: str | Path) -> None:
        self.repository = repository
        self.storage_root = Path(storage_root)

    def ingest(
        self,
        project_id: str,
        events: list[ObservationEventV1 | dict[str, Any]],
        *,
        source_label: str,
        actor_label: str,
    ) -> EventIngestionResult:
        source = _required_label(source_label, "source_label")
        actor = _required_label(actor_label, "actor_label")
        accepted_events: list[dict[str, Any]] = []
        rejected: list[tuple[int, str]] = []
        for index, candidate in enumerate(events):
            try:
                event = ObservationEventV1.model_validate(candidate)
            except (TypeError, ValueError):
                rejected.append((index, "INVALID_OBSERVATION_EVENT"))
                continue
            if _has_personal_attribute(event):
                rejected.append((index, "PERSONAL_ATTRIBUTE_DETECTED"))
                continue
            accepted_events.append(_storage_event(event))

        accepted, duplicates = self.repository.append_observation_events(
            project_id=project_id,
            events=accepted_events,
            rejected_count=len(rejected),
            source_label=source,
            actor_label=actor,
        )
        return EventIngestionResult(
            accepted=accepted,
            duplicates=duplicates,
            rejected=rejected,
        )

    def _accepted_path(self, dataset_id: str) -> Path:
        relative = PurePosixPath("data", "accepted", dataset_id)
        root = self.storage_root.resolve()
        path = (root / Path(relative)).resolve()
        if root not in path.parents:
            raise RuntimeError("La ruta generada salió del storage privado")
        return path

    @staticmethod
    def _csv_content(rows: list[Any]) -> bytes:
        decoded: list[dict[str, Any]] = []
        include_outcome = False
        # Una fila por caso: la versión recibida más tarde, que trae el resultado si llegó.
        latest: dict[str, Any] = {}
        for row in rows:
            latest[str(row["entity_id"])] = row
        for row in latest.values():
            outcome = None if row["outcome_json"] is None else json.loads(row["outcome_json"])
            include_outcome = include_outcome or outcome is not None
            decoded.append(
                {
                    "id": str(row["entity_id"]),
                    "occurred_at": str(row["occurred_at"]),
                    "attributes": json.loads(str(row["attributes_json"])),
                    "outcome": outcome,
                }
            )

        attribute_columns = _attribute_columns(decoded, include_outcome=include_outcome)
        fieldnames = ["id", "occurred_at", *attribute_columns.values()]
        if include_outcome:
            fieldnames.extend(["outcome", "outcome_observed_at"])
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in decoded:
            flat: dict[str, Any] = {
                "id": item["id"],
                "occurred_at": item["occurred_at"],
            }
            flat.update(
                {attribute_columns[key]: value for key, value in item["attributes"].items()}
            )
            if include_outcome and item["outcome"] is not None:
                flat["outcome"] = item["outcome"]["value"]
                flat["outcome_observed_at"] = item["outcome"]["observed_at"]
            writer.writerow(flat)
        return output.getvalue().encode("utf-8")

    def materialize(
        self,
        project_id: str,
        *,
        entity: str,
        actor_label: str,
        display_name: str,
        classification: DatasetClassification | str,
    ) -> str:
        entity_name = _required_label(entity, "entity")
        actor = _required_label(actor_label, "actor_label")
        filename = Path(_required_label(display_name, "display_name")).name
        if Path(filename).suffix.casefold() != ".csv":
            filename = f"{filename}.csv"
        classification_value = DatasetClassification(classification)
        if classification_value not in {
            DatasetClassification.SYNTHETIC,
            DatasetClassification.SANDBOX,
        }:
            raise ValueError("classification debe ser synthetic o sandbox")

        rows = self.repository.list_observation_events(project_id, entity=entity_name)
        if not rows:
            raise ValueError("No hay eventos para la entidad solicitada")
        content = self._csv_content(rows)
        profile = validate_and_profile(
            filename,
            content,
            classification=classification_value,
        )
        dataset_id = str(uuid4())
        path = self._accepted_path(dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        try:
            self.repository.add_dataset(
                project_id=project_id,
                dataset_id=dataset_id,
                sha256=hashlib.sha256(content).hexdigest(),
                original_filename_display=filename,
                source=f"observation_events:{entity_name}",
                license_name=None,
                anonymization_declared=False,
                admission_status="ACCEPTED",
                rejection_reason=None,
                n_rows=profile.row_count,
                n_cols=profile.column_count,
                byte_count=len(content),
                profile=profile.model_dump(mode="json"),
                dataset_kind="SOURCE",
            )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        self.repository.record_observation_materialization(
            project_id=project_id,
            dataset_id=dataset_id,
            entity=entity_name,
            event_count=len(rows),
            classification=classification_value.value,
            actor_label=actor,
        )
        return dataset_id
