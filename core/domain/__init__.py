"""Reglas de dominio independientes de UI y proveedores externos."""

from core.domain.context import BusinessContextV1
from core.domain.errors import (
    ApprovalRequiredError,
    ContextValidationError,
    InvalidStateTransitionError,
    StrategyValidationError,
)

__all__ = [
    "ApprovalRequiredError",
    "BusinessContextV1",
    "ContextValidationError",
    "InvalidStateTransitionError",
    "StrategyValidationError",
]
