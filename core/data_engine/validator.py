"""Validación estructural de CSV sin ejecutar su contenido."""

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from core.data_engine.errors import DataValidationError
from core.data_engine.models import IngestionLimits

_COMPRESSED_SIGNATURES = (
    b"PK\x03\x04",  # ZIP
    b"\x1f\x8b",  # gzip
    b"BZh",  # bzip2
    b"\xfd7zXZ\x00",  # xz
)


@dataclass(frozen=True)
class ValidatedCSV:
    text: str
    header: tuple[str, ...]
    row_count: int
    column_count: int


def read_limited_stream(stream: BinaryIO, limits: IngestionLimits) -> bytes:
    """Lee como máximo el límite más un byte, sin consumir un upload enorme."""

    content = stream.read(limits.max_upload_bytes + 1)
    if len(content) > limits.max_upload_bytes:
        raise DataValidationError(
            "FILE_TOO_LARGE",
            f"El archivo supera el límite de {limits.max_upload_bytes} bytes",
        )
    return content


def validate_csv_bytes(
    filename: str,
    content: bytes,
    limits: IngestionLimits,
) -> ValidatedCSV:
    """Valida formato, encoding y dimensiones antes del perfilado."""

    text = validate_upload_envelope(filename, content, limits)

    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        raw_header = next(reader, None)
        if raw_header is None:
            raise DataValidationError("EMPTY_FILE", "El archivo CSV está vacío")
        if len(raw_header) == 1 and any(mark in raw_header[0] for mark in (";", "\t", "|")):
            raise DataValidationError(
                "AMBIGUOUS_DELIMITER",
                "El P0 requiere CSV delimitado por comas",
            )
        header = tuple(name.strip() for name in raw_header)
        if any(not name for name in header):
            raise DataValidationError(
                "EMPTY_HEADER",
                "Todos los encabezados deben tener un nombre",
            )
        normalized = [name.casefold() for name in header]
        if len(set(normalized)) != len(normalized):
            raise DataValidationError(
                "DUPLICATE_HEADERS",
                "Los encabezados duplicados no están permitidos",
            )
        if len(header) > limits.max_columns:
            raise DataValidationError(
                "TOO_MANY_COLUMNS",
                f"El CSV supera el límite de {limits.max_columns} columnas",
            )

        row_count = 0
        for row in reader:
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != len(header):
                raise DataValidationError(
                    "MALFORMED_ROW",
                    "Una fila no tiene la misma cantidad de campos que el encabezado",
                )
            row_count += 1
            if row_count > limits.max_rows:
                raise DataValidationError(
                    "TOO_MANY_ROWS",
                    f"El CSV supera el límite de {limits.max_rows} filas",
                )
    except csv.Error as error:
        raise DataValidationError("MALFORMED_CSV", "El CSV no se pudo interpretar") from error

    if row_count == 0:
        raise DataValidationError("NO_DATA_ROWS", "El CSV no contiene filas de datos")
    return ValidatedCSV(
        text=text,
        header=header,
        row_count=row_count,
        column_count=len(header),
    )


def validate_upload_envelope(
    filename: str,
    content: bytes,
    limits: IngestionLimits,
) -> str:
    """Aplica controles previos incluso si el archivo queda en cuarentena."""

    if Path(filename).suffix.casefold() != ".csv":
        raise DataValidationError("INVALID_EXTENSION", "Solo se aceptan archivos .csv")
    if not content:
        raise DataValidationError("EMPTY_FILE", "El archivo CSV está vacío")
    if len(content) > limits.max_upload_bytes:
        raise DataValidationError(
            "FILE_TOO_LARGE",
            f"El archivo supera el límite de {limits.max_upload_bytes} bytes",
        )
    if content.startswith(_COMPRESSED_SIGNATURES):
        raise DataValidationError(
            "COMPRESSED_FILE",
            "Los archivos comprimidos no están permitidos en el P0",
        )
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise DataValidationError(
            "INVALID_ENCODING",
            "El CSV debe usar UTF-8 o UTF-8-SIG",
        ) from error
    if "\x00" in text:
        raise DataValidationError("BINARY_CONTENT", "El archivo contiene bytes binarios")
    return text
