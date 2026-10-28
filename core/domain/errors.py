"""Errores tipados del workflow de Goal Settings."""


class DomainError(RuntimeError):
    """Base para errores de precondición del dominio."""


class ApprovalRequiredError(DomainError):
    """La versión exacta no tiene una aprobación vigente."""


class InvalidStateTransitionError(DomainError):
    """La transición solicitada no pertenece a la máquina de estados."""


class ContextValidationError(DomainError):
    """El contexto no coincide con el dataset o deja una decisión obligatoria inválida."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class StrategyValidationError(DomainError):
    """La propuesta viola el contexto vigente o el catálogo técnico P0."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ReportValidationError(DomainError):
    """El reporte no puede construirse o contiene afirmaciones sin evidencia."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
