"""Adaptador para cualquier API compatible con `POST {base_url}/chat/completions`.

OpenAI es el proveedor por defecto; otros servicios o servidores locales que expongan el mismo
formato se usan cambiando `LLM_BASE_URL` y `LLM_MODEL`. Sin SDK: sólo `requests`.
"""

import time
from collections.abc import Callable
from time import perf_counter
from typing import Any

import requests

from core.ai_engine.config import LLMSettings
from core.ai_engine.provider import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    register_provider,
)


class OpenAICompatibleProvider:
    def __init__(
        self,
        settings: LLMSettings,
        *,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.name = settings.provider
        self.model = settings.model
        self._settings = settings
        self._session = session or requests.Session()
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"OpenAICompatibleProvider(name={self.name!r}, model={self.model!r})"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._settings.api_key is not None:
            headers["Authorization"] = f"Bearer {self._settings.api_key.get_secret_value()}"
        return headers

    def _body(self, request: LLMRequest) -> dict[str, Any]:
        settings = self._settings
        body: dict[str, Any] = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }
        if settings.response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "schema": request.json_schema,
                    "strict": True,
                },
            }
        else:
            body["response_format"] = {"type": "json_object"}
        if settings.temperature is not None:
            body["temperature"] = settings.temperature
        max_tokens = request.max_output_tokens or settings.max_output_tokens
        if max_tokens is not None:
            body[settings.max_output_tokens_field] = max_tokens
        return body

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        url = f"{self._settings.base_url}/chat/completions"
        body = self._body(request)
        timeout = max(self._settings.timeout_seconds, request.min_timeout_seconds or 0)
        if request.max_timeout_seconds is not None:
            timeout = min(timeout, request.max_timeout_seconds)
        retries = self._settings.max_retries if request.max_retries is None else request.max_retries
        attempts = 1 + retries
        for attempt in range(attempts):
            started = perf_counter()
            try:
                response = self._session.post(
                    url,
                    json=body,
                    headers=self._headers(),
                    timeout=timeout,
                )
            except requests.Timeout:
                error = LLMProviderError("PROVIDER_TIMEOUT", retryable=True)
            except requests.RequestException:
                error = LLMProviderError("PROVIDER_UNREACHABLE", retryable=True)
            else:
                if response.status_code == 200:
                    return self._parse(response, int((perf_counter() - started) * 1000))
                status = response.status_code
                if status == 429 and _error_code(response) == "insufficient_quota":
                    # Sin saldo: reintentar no sirve y la persona debe saberlo.
                    raise LLMProviderError("PROVIDER_QUOTA_EXHAUSTED")
                error = LLMProviderError(
                    f"PROVIDER_HTTP_{status}",
                    retryable=status == 429 or status >= 500,
                )
            if not error.retryable or attempt == attempts - 1:
                raise error
            self._sleep(min(2.0**attempt, 4.0))
        raise LLMProviderError("PROVIDER_UNREACHABLE")  # pragma: no cover

    def _parse(self, response: Any, latency_ms: int) -> LLMResponse:
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMProviderError("INVALID_PROVIDER_RESPONSE") from None
        if message.get("refusal"):
            raise LLMProviderError("PROVIDER_REFUSAL")
        if choice.get("finish_reason") == "length":
            raise LLMProviderError("PROVIDER_OUTPUT_TRUNCATED")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("EMPTY_PROVIDER_RESPONSE")
        usage = data.get("usage") or {}
        return LLMResponse(
            content=content,
            provider=self.name,
            model=str(data.get("model") or self.model),
            response_id=data.get("id"),
            latency_ms=latency_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


def _error_code(response: Any) -> str | None:
    """Lee sólo `error.code` del cuerpo; nunca registra el cuerpo completo."""

    try:
        error = response.json().get("error") or {}
    except (ValueError, AttributeError):
        return None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


def _factory(settings: LLMSettings) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(settings)


# "openai" usa la URL por defecto; "openai_compatible" deja explícito que se cambió.
register_provider("openai", _factory)
register_provider("openai_compatible", _factory)
