"""Datos de un resultado autorizados para salir hacia un proveedor de IA."""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime

from pydantic import ConfigDict, Field, field_validator

from core.analysis.contracts import StrictAnalysisModel
from core.analysis.engine import AnalysisColumnV1, AnalysisResultV1, JSONScalar
from core.analysis.validation import AnalysisPlanError

MAX_LLM_ROWS = 500


class LlmPayloadV1(StrictAnalysisModel):
    """Vista exacta de las columnas que una persona autorizó enviar."""

    model_config = ConfigDict(extra="forbid", strict=True)

    columns: list[AnalysisColumnV1]
    rows: list[list[JSONScalar]]
    sent_row_count: int = Field(ge=0)
    excluded_row_count: int = Field(ge=0)
    truncated: bool


class AnalysisLineageV1(StrictAnalysisModel):
    """Identidad reproducible del cálculo que produjo un resultado."""

    model_config = ConfigDict(extra="forbid", strict=True)

    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_hashes: dict[str, str]
    engine_version: str = Field(min_length=1)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def created_at_has_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("La fecha de creación del linaje debe incluir zona horaria")
        return value


def analysis_lineage(result: AnalysisResultV1) -> AnalysisLineageV1:
    """Extrae el linaje sellado al ejecutar el resultado, sin recalcularlo."""

    return AnalysisLineageV1(
        plan_hash=result.plan_hash,
        source_hashes=result.source_hashes,
        engine_version=result.engine_version,
        created_at=result.created_at,
    )


def to_llm_payload(
    result: AnalysisResultV1,
    allowed_columns: Collection[str],
    max_rows: int = MAX_LLM_ROWS,
) -> LlmPayloadV1:
    """Construye la vista autorizada, preservando el orden del resultado."""

    if (
        isinstance(max_rows, bool)
        or not isinstance(max_rows, int)
        or not 1 <= max_rows <= MAX_LLM_ROWS
    ):
        raise ValueError(f"max_rows debe ser un entero entre 1 y {MAX_LLM_ROWS}")

    requested = list(allowed_columns)
    available = {column.name for column in result.columns}
    missing = [column for column in requested if column not in available]
    if missing:
        raise AnalysisPlanError(
            "COLUMN_NOT_FOUND",
            f"La columna solicitada {missing[0]!r} no existe en el resultado",
        )

    allowed = set(requested)
    selected = [
        (index, column) for index, column in enumerate(result.columns) if column.name in allowed
    ]
    sent_rows = result.rows[:max_rows]
    rows = [[row[index] for index, _ in selected] for row in sent_rows]
    sent_row_count = len(rows)
    excluded_row_count = max(result.total_rows - sent_row_count, 0)
    return LlmPayloadV1(
        columns=[column for _, column in selected],
        rows=rows,
        sent_row_count=sent_row_count,
        excluded_row_count=excluded_row_count,
        truncated=excluded_row_count > 0,
    )


def omitted_columns(result: AnalysisResultV1, allowed_columns: Collection[str]) -> list[str]:
    """Columnas que NO saldrán hacia la IA, sólo para la vista previa local en pantalla.

    No forman parte de `LlmPayloadV1`: ni siquiera sus nombres se envían.
    """

    allowed = set(allowed_columns)
    return [column.name for column in result.columns if column.name not in allowed]
