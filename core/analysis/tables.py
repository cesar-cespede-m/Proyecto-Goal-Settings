"""GA-06: guardar el resultado de un análisis como tabla del proyecto, con su receta.

La receta es el plan validado, las fuentes con su hash y la versión del motor: con ella la tabla
se puede rehacer igual o con datos nuevos (GA-08). Guardar no entrena nada: elegir la tabla para
entrenar y aprobar el entrenamiento siguen siendo decisiones humanas aparte.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import pandas as pd

from core.analysis.contracts import AnalysisPlanV1
from core.analysis.engine import (
    ANALYSIS_ENGINE_VERSION,
    RESULT_TRUNCATED,
    AnalysisEngine,
    AnalysisLimits,
    AnalysisResultV1,
    load_project_source,
)
from core.analysis.validation import AnalysisPlanError, canonical_plan_hash
from core.data_engine.errors import DataValidationError
from core.data_engine.models import DatasetClassification
from core.data_engine.service import validate_and_profile

# Una tabla para entrenar conserva todas las filas: el límite de salida es el de entrada.
TABLE_LIMITS = AnalysisLimits(max_output_rows=AnalysisLimits().max_input_rows)
MAX_NAME_LENGTH = 80


@dataclass(frozen=True, slots=True)
class SavedAnalysisTable:
    dataset_id: str
    display_name: str
    rows: int
    columns: int
    plan_sha256: str


def result_to_csv(result: AnalysisResultV1) -> bytes:
    """CSV canónico del resultado: mismas columnas, mismo orden, UTF-8 y saltos `\\n`."""

    frame = pd.DataFrame(result.rows, columns=[column.name for column in result.columns])
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


class AnalysisTableService:
    def __init__(self, repository: Any, storage_root: str | Path) -> None:
        self.repository = repository
        self.storage_root = Path(storage_root)

    def save(
        self,
        *,
        project_id: str,
        plan: AnalysisPlanV1,
        source_ids: Mapping[str, str],
        display_name: str,
        actor_label: str,
    ) -> SavedAnalysisTable:
        """Rehace el cálculo con todas las filas y lo registra como tabla con receta."""

        name = " ".join(display_name.split())[:MAX_NAME_LENGTH]
        if not name:
            raise AnalysisPlanError("TABLE_NAME_REQUIRED", "Ponle un nombre a la tabla")
        if not actor_label.strip():
            raise AnalysisPlanError("TABLE_ACTOR_REQUIRED", "Indica quién guarda la tabla")
        missing = [source for source in plan.sources if source not in source_ids]
        if missing:
            raise AnalysisPlanError("SOURCE_NOT_FOUND", "Falta un archivo usado por el cálculo")

        loaded = {
            source: load_project_source(self.repository, self.storage_root, source_ids[source])
            for source in plan.sources
        }
        result = AnalysisEngine(TABLE_LIMITS).run(plan, loaded)
        if RESULT_TRUNCATED in result.warnings:
            raise AnalysisPlanError(
                "TABLE_TOO_LARGE", "El resultado tiene demasiadas filas para guardarlo como tabla"
            )
        if not result.total_rows:
            raise AnalysisPlanError("TABLE_EMPTY", "El cálculo no dejó filas para guardar")

        anonymized = any(
            bool(self.repository.get_dataset(source_ids[source])["anonymization_declared"])
            for source in plan.sources
        )
        content = result_to_csv(result)
        try:
            profile = validate_and_profile(
                f"{name}.csv",
                content,
                classification=(
                    DatasetClassification.ANONYMIZED
                    if anonymized
                    else DatasetClassification.SYNTHETIC
                ),
            )
        except DataValidationError as error:
            raise AnalysisPlanError(
                "TABLE_NOT_VALID", "El resultado no forma una tabla válida para guardar"
            ) from error

        dataset_id = str(uuid4())
        root = self.storage_root.resolve()
        destination = (root / Path(PurePosixPath("data", "accepted", dataset_id))).resolve()
        if root not in destination.parents:
            raise AnalysisPlanError("INVALID_INTERNAL_PATH", "La ruta privada salió del storage")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
        plan_sha256 = canonical_plan_hash(plan)
        try:
            self.repository.persist_analysis_table(
                table_id=str(uuid4()),
                project_id=project_id,
                output_dataset_id=dataset_id,
                display_name=name,
                output_sha256=hashlib.sha256(content).hexdigest(),
                output_bytes=len(content),
                output_rows=profile.row_count,
                output_columns=profile.column_count,
                output_anonymization_declared=anonymized,
                output_profile=profile.model_dump(mode="json"),
                plan=plan.model_dump(mode="json"),
                plan_sha256=plan_sha256,
                sources=[
                    {
                        "name": source,
                        "dataset_id": source_ids[source],
                        "sha256": loaded[source].sha256,
                    }
                    for source in plan.sources
                ],
                engine_version=ANALYSIS_ENGINE_VERSION,
                warnings=list(result.warnings),
                actor_label=actor_label,
            )
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return SavedAnalysisTable(
            dataset_id=dataset_id,
            display_name=name,
            rows=profile.row_count,
            columns=profile.column_count,
            plan_sha256=plan_sha256,
        )
