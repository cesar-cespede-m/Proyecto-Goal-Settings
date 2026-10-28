"""Configuración del proveedor LLM desde el entorno; la clave nunca se expone.

Cambiar de proveedor es cambiar variables, no código: `LLM_PROVIDER`, `LLM_BASE_URL`,
`LLM_MODEL` y `LLM_API_KEY_ENV` (el nombre de la variable que guarda la clave).
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_TRUE = {"1", "true", "yes", "si", "sí", "on"}
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class LLMConfigError(ValueError):
    """Configuración LLM inválida; el mensaje nunca incluye la clave."""


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    provider: str = "none"
    model: str = ""
    base_url: str = DEFAULT_BASE_URL
    api_key: SecretStr | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_retries: int = Field(default=1, ge=0, le=3)
    response_format: Literal["json_schema", "json_object"] = "json_schema"
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_output_tokens: int | None = Field(default=1200, ge=64, le=8000)
    max_output_tokens_field: Literal["max_completion_tokens", "max_tokens"] = (
        "max_completion_tokens"
    )

    @property
    def active(self) -> bool:
        return self.enabled and self.provider != "none"

    @property
    def is_local(self) -> bool:
        return urlparse(self.base_url).hostname in _LOCAL_HOSTS

    @model_validator(mode="after")
    def validate_active_configuration(self) -> "LLMSettings":
        if not self.active:
            return self
        if not self.model.strip():
            raise ValueError("LLM_MODEL es obligatorio cuando el LLM está habilitado")
        if urlparse(self.base_url).scheme != "https" and not self.is_local:
            raise ValueError("LLM_BASE_URL debe usar https; http sólo se admite en localhost")
        if self.api_key is None and not self.is_local:
            raise ValueError(f"Falta la clave en la variable {self.api_key_env}")
        return self

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        env_file: Path | None = None,
    ) -> "LLMSettings":
        """Lee `.env` (si se indica) y luego el entorno, que tiene prioridad."""

        values = _read_env_file(env_file) if env_file is not None else {}
        values.update(os.environ if environ is None else environ)

        def get(name: str, default: str | None = None) -> str | None:
            value = values.get(name)
            return default if value is None or not value.strip() else value.strip()

        key_env = get("LLM_API_KEY_ENV", "OPENAI_API_KEY") or "OPENAI_API_KEY"
        key = get(key_env)
        optional_float = get("LLM_TEMPERATURE")
        optional_tokens = get("LLM_MAX_OUTPUT_TOKENS", "1200")
        try:
            return cls(
                enabled=(get("LLM_ENABLED", "false") or "").casefold() in _TRUE,
                provider=(get("LLM_PROVIDER", "none") or "none").casefold(),
                model=get("LLM_MODEL", "") or "",
                base_url=(get("LLM_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/"),
                api_key=SecretStr(key) if key else None,
                api_key_env=key_env,
                timeout_seconds=float(get("LLM_TIMEOUT_SECONDS", "30") or 30),
                max_retries=int(get("LLM_MAX_RETRIES", "1") or 1),
                response_format=get("LLM_RESPONSE_FORMAT", "json_schema"),
                temperature=float(optional_float) if optional_float else None,
                max_output_tokens=int(optional_tokens) if optional_tokens else None,
                max_output_tokens_field=get("LLM_MAX_OUTPUT_TOKENS_FIELD", "max_completion_tokens"),
            )
        except ValidationError as error:
            messages = "; ".join(str(item["msg"]) for item in error.errors())
            raise LLMConfigError(messages) from None
        except ValueError as error:
            message = f"Valor numérico inválido en la configuración LLM: {error}"
            raise LLMConfigError(message) from None


def _read_env_file(path: Path) -> dict[str, str]:
    """Parser mínimo de `.env` (KEY=VALUE, comentarios con #); no ejecuta nada."""

    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values
