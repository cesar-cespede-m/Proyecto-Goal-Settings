"""Receta de una tabla armada desde un recorrido: guardarla y rehacerla con actividad nueva.

La receta (la especificación `SnapshotSpecV1` y el archivo de actividad del que salió, con su
huella) queda en la auditoría, sólo de inserción. Rehacer aplica exactamente la misma receta a un
archivo nuevo con las mismas columnas, así un reentrenamiento parte de los mismos pasos.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from core.analysis.snapshot import (
    ObservationCoverageV1,
    SnapshotReadinessV1,
    SnapshotResultV1,
    SnapshotSpecV1,
    build_snapshot,
)
from core.db.connection import transaction

EVENT_TYPE = "SNAPSHOT_TABLE_SAVED"
READINESS_EVENT_TYPE = "SNAPSHOT_READINESS_RECORDED"


def record_snapshot_recipe(
    repository: Any,
    *,
    project_id: str,
    table_dataset_id: str,
    source_dataset_id: str,
    spec: SnapshotSpecV1 | dict[str, Any],
    actor_label: str,
    entity_dataset_id: str | None = None,
    coverage: ObservationCoverageV1 | dict[str, Any] | None = None,
    readiness: SnapshotReadinessV1 | dict[str, Any] | None = None,
) -> None:
    model = SnapshotSpecV1.model_validate(spec)
    coverage_model = None if coverage is None else ObservationCoverageV1.model_validate(coverage)
    readiness_model = None if readiness is None else SnapshotReadinessV1.model_validate(readiness)
    if readiness_model is not None:
        if coverage_model is not None and coverage_model != readiness_model.coverage:
            raise ValueError("La cobertura no coincide con la usada para evaluar los resultados")
        coverage_model = readiness_model.coverage
    source = repository.get_dataset(source_dataset_id)
    if str(source["project_id"]) != project_id:
        raise ValueError("La fuente de la receta no pertenece al proyecto")
    if not actor_label.strip():
        raise ValueError("Indica quién declara la cobertura y guarda la receta")
    with transaction(repository.connection):
        repository._audit(
            project_id=project_id,
            event_type=EVENT_TYPE,
            entity_type="DATASET",
            entity_id=table_dataset_id,
            detail={
                "actor_label": actor_label.strip(),
                "source_dataset_id": source_dataset_id,
                "source_sha256": str(source["sha256"]),
                "spec": model.model_dump(mode="json"),
                # Datos fijos por caso: se reusan al rehacer (no cambian con la actividad).
                "entity_dataset_id": entity_dataset_id,
                # Es evidencia de este lote, no un parámetro reutilizable de la receta.
                "coverage": None
                if coverage_model is None
                else coverage_model.model_dump(mode="json"),
                "readiness": None
                if readiness_model is None
                else readiness_model.model_dump(mode="json"),
            },
        )


def record_snapshot_readiness(
    repository: Any,
    *,
    project_id: str,
    table_dataset_id: str,
    source_sha256: str,
    readiness: SnapshotReadinessV1 | dict[str, Any],
    actor_label: str,
) -> None:
    """Registra cobertura/exclusiones de un lote rehecho, sin IDs ni resultados individuales."""

    model = SnapshotReadinessV1.model_validate(readiness)
    if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise ValueError("La huella de la fuente debe ser SHA-256")
    if not actor_label.strip():
        raise ValueError("Indica quién declara la cobertura del lote")
    dataset = repository.get_dataset(table_dataset_id)
    if str(dataset["project_id"]) != project_id:
        raise ValueError("La tabla no pertenece al proyecto")
    with transaction(repository.connection):
        repository._audit(
            project_id=project_id,
            event_type=READINESS_EVENT_TYPE,
            entity_type="DATASET",
            entity_id=table_dataset_id,
            detail={
                "actor_label": actor_label.strip(),
                "source_sha256": source_sha256,
                "readiness": model.model_dump(mode="json"),
            },
        )


def snapshot_recipe(repository: Any, table_dataset_id: str) -> dict[str, Any] | None:
    row = repository.connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = ? AND entity_id = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (EVENT_TYPE, table_dataset_id),
    ).fetchone()
    return None if row is None else json.loads(row["detail_json"])


def rebuild_from_recipe(
    repository: Any,
    table_dataset_id: str,
    events: pd.DataFrame,
    entities: pd.DataFrame | None = None,
    *,
    coverage: ObservationCoverageV1 | dict[str, Any] | None = None,
) -> SnapshotResultV1:
    """Misma receta, cobertura nueva: nunca hereda la declaración de la fuente histórica."""

    recipe = snapshot_recipe(repository, table_dataset_id)
    if recipe is None:
        raise LookupError("Esta tabla no se armó desde un recorrido: no tiene receta que repetir")
    return build_snapshot(
        events, SnapshotSpecV1.model_validate(recipe["spec"]), entities, coverage=coverage
    )
