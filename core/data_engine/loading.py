"""Conversión única de un CSV ya validado a DataFrame.

Perfilado, contexto y entrenamiento deben inferir exactamente los mismos tipos: la clase
positiva se compara con tipo estricto, por lo que dos lecturas distintas romperían el contrato.
"""

import io

import pandas as pd

from core.data_engine.errors import DataValidationError
from core.data_engine.validator import ValidatedCSV


def read_validated_frame(validated: ValidatedCSV) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.StringIO(validated.text), low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise DataValidationError(
            "PARSING_FAILED",
            "El CSV validado no pudo convertirse en una tabla",
        ) from error
    frame.columns = list(validated.header)
    return frame
