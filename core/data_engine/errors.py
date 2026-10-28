"""Errores accionables de admisión de datos."""


class DataEngineError(RuntimeError):
    """Base de errores controlados del motor de datos."""


class DataValidationError(DataEngineError):
    """El archivo no cumple el contrato CSV del P0."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DatasetRejectedError(DataValidationError):
    """El rechazo fue persistido para auditoría del flujo."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        dataset_id: str,
        warnings: list[str] | None = None,
    ) -> None:
        super().__init__(code, message)
        self.dataset_id = dataset_id
        self.warnings = list(warnings or [])


class DatasetCompositionError(DataEngineError):
    """Una composición no puede ejecutarse sin perder integridad o trazabilidad."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
