"""Contrato agnóstico del proveedor LLM y registro de adaptadores.

`core/` sólo conoce `LLMProvider`. Un proveedor nuevo se agrega con `register_provider()`;
ningún otro módulo cambia.
"""

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from core.ai_engine.config import LLMConfigError, LLMSettings


class LLMRequest(BaseModel):
    system: str = Field(min_length=1)
    user: str = Field(min_length=1)
    schema_name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    json_schema: dict[str, Any]
    # Límite propio de la llamada; si es None se usa LLM_MAX_OUTPUT_TOKENS.
    max_output_tokens: int | None = Field(default=None, ge=64, le=16000)
    # Espera mínima de la llamada; si la configurada (LLM_TIMEOUT_SECONDS) es mayor, se usa esa.
    min_timeout_seconds: float | None = Field(default=None, gt=0, le=120)
    # El supervisor puede acotar cada intento sin alterar el comportamiento legacy.
    max_timeout_seconds: float | None = Field(default=None, gt=0, le=120)
    max_retries: int | None = Field(default=None, ge=0, le=3, strict=True)


class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    response_id: str | None = None
    latency_ms: int = Field(ge=0)
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProviderError(RuntimeError):
    """Fallo del proveedor con código estable; nunca incluye la clave ni el prompt."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class LLMProvider(Protocol):
    name: str
    model: str

    def complete_json(self, request: LLMRequest) -> LLMResponse: ...


ProviderFactory = Callable[[LLMSettings], LLMProvider]
_REGISTRY: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    _REGISTRY[name.casefold()] = factory


def registered_providers() -> list[str]:
    return sorted(_REGISTRY)


def build_provider(settings: LLMSettings) -> LLMProvider | None:
    """Devuelve None si el LLM está deshabilitado: el producto opera en modo plantilla."""

    if not settings.active:
        return None
    factory = _REGISTRY.get(settings.provider)
    if factory is None:
        raise LLMConfigError(
            f"Proveedor LLM desconocido: {settings.provider}. "
            f"Disponibles: {', '.join(registered_providers())}"
        )
    return factory(settings)
