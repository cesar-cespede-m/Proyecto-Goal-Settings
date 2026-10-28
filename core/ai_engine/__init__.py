"""IA generativa agnóstica del proveedor: redacta, nunca calcula ni entrena."""

from core.ai_engine import openai_compatible as _openai_compatible  # noqa: F401  (registra)
from core.ai_engine.config import LLMConfigError, LLMSettings
from core.ai_engine.narrative import NarrativeResult, NarrativeService
from core.ai_engine.prompts import PROMPT_VERSION
from core.ai_engine.provider import (
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    build_provider,
    register_provider,
    registered_providers,
)

__all__ = [
    "PROMPT_VERSION",
    "LLMConfigError",
    "LLMProvider",
    "LLMProviderError",
    "LLMRequest",
    "LLMResponse",
    "LLMSettings",
    "NarrativeResult",
    "NarrativeService",
    "build_provider",
    "register_provider",
    "registered_providers",
]
