"""Serialización canónica y hashes de artefactos versionados."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def canonical_json(value: Any) -> str:
    """Serializa JSON estable en UTF-8 lógico y rechaza NaN/Infinity."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def strategy_fingerprint(
    dataset_sha256: str,
    context_sha256: str,
    canonical_strategy: str,
    schema_version: str,
) -> str:
    """Hash compuesto exigido para autorizar una estrategia exacta."""

    return sha256_text(dataset_sha256 + context_sha256 + canonical_strategy + schema_version)


def report_fingerprint(
    *,
    run_id: str,
    metrics: dict[str, Any],
    insights: list[dict[str, Any]],
    limitations: list[str],
    version_no: int,
) -> str:
    """Hash del snapshot exacto que debe aprobarse antes de publicar."""

    return canonical_hash(
        {
            "insights": insights,
            "limitations": limitations,
            "metrics": metrics,
            "run_id": run_id,
            "version_no": version_no,
        }
    )
