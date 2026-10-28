"""Aliases de un turno: las referencias internas nunca salen hacia el proveedor."""

from __future__ import annotations

import json
from typing import Any

from core.orchestration.runtime import OrchestrationError


def wire_context(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    identifiers: dict[str, str] = {}

    def collect(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                collect(item, name)
        elif isinstance(value, list):
            for item in value:
                collect(item, key)
        elif isinstance(value, str) and (key == "id" or key.endswith("_id")):
            identifiers.setdefault(value, f"R{len(identifiers) + 1:03d}")

    collect(context)

    def encode(value: Any, key: str = "") -> Any:
        if key == "sha256" or key.endswith("_sha256"):
            return None
        if isinstance(value, dict):
            return {name: encode(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [encode(item, key) for item in value]
        if isinstance(value, str) and value in identifiers:
            return identifiers[value]
        return value

    return encode(context), {alias: identifier for identifier, alias in identifiers.items()}


def decode_proposal(content: str, context: dict[str, Any]) -> str:
    if len(content.encode("utf-8")) > 65536:
        raise OrchestrationError("AGENT_INVALID_OUTPUT", "La respuesta excede el contrato")
    try:
        value = json.loads(content)
    except ValueError as error:
        raise OrchestrationError(
            "AGENT_INVALID_OUTPUT", "La respuesta no cumple el contrato"
        ) from error
    _, aliases = wire_context(context)

    def decode(item: Any, key: str = "") -> Any:
        if (key == "sha256" or key.endswith("_sha256")) and item is not None:
            raise OrchestrationError(
                "AGENT_REFERENCE_NOT_PROVIDED", "El proveedor no recibió hashes"
            )
        if isinstance(item, dict):
            return {name: decode(part, name) for name, part in item.items()}
        if isinstance(item, list):
            return [decode(part, key) for part in item]
        if isinstance(item, str) and (key == "id" or key.endswith("_id")):
            if item not in aliases:
                raise OrchestrationError(
                    "AGENT_REFERENCE_NOT_PROVIDED", "La referencia no fue enviada"
                )
            return aliases[item]
        return item

    return json.dumps(decode(value), ensure_ascii=False)
