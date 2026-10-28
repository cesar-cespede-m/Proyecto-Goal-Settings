"""C2-04 (ciclo 2): exportar e importar la receta y las decisiones de un modelo, sin datos.

El paquete lleva la receta (`RecipeV1`), la referencia al contexto (`BusinessContextRefV2`),
las decisiones confirmadas y, si la hubo, la preparación de datos. Nunca lleva filas ni
valores de los archivos. Al importarlo en otro proyecto las decisiones llegan como propuestas:
una persona las confirma una por una.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from core.contracts.platform import BusinessContextRefV2, RecipeV1, VersionRef
from core.domain.canonical import canonical_hash
from core.domain.context import BusinessContextV1
from core.domain.errors import DomainError
from core.domain.retraining_service import RetrainingService

MAX_BUNDLE_BYTES = 1_000_000


class RecipeBundleError(DomainError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RecipeBundleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["recipe-bundle.v1"] = "recipe-bundle.v1"
    recipe: RecipeV1
    context_ref: BusinessContextRefV2
    context: dict[str, Any]
    analysis_plan: dict[str, Any] | None
    content_sha256: str


def _content(bundle: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bundle.items() if key != "content_sha256"}


def export_bundle(repository: Any, storage_root: Any, run_id: str) -> bytes:
    recipe = RetrainingService(repository, storage_root).recipe(run_id)
    context_row = repository.get_context(recipe.context.id)
    context = json.loads(context_row["context_json"])["context"]
    plan = None
    if recipe.analysis_plan is not None:
        strategy = repository.get_strategy(recipe.strategy.id)
        table = repository.get_analysis_table(str(strategy["dataset_id"]))
        plan = json.loads(table["plan_json"])
    content = {
        "schema_version": "recipe-bundle.v1",
        "recipe": recipe.model_dump(mode="json"),
        "context_ref": BusinessContextRefV2(
            context=VersionRef(id=recipe.context.id, sha256=recipe.context.sha256),
            source_project_id=str(context_row["project_id"]),
        ).model_dump(mode="json"),
        "context": context,
        "analysis_plan": plan,
    }
    bundle = RecipeBundleV1.model_validate({**content, "content_sha256": canonical_hash(content)})
    return json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, indent=1).encode()


def parse_bundle(content: bytes) -> tuple[RecipeBundleV1, BusinessContextV1]:
    if len(content) > MAX_BUNDLE_BYTES:
        raise RecipeBundleError("BUNDLE_TOO_LARGE", "El archivo de receta es demasiado grande")
    try:
        raw = json.loads(content.decode("utf-8"))
        bundle = RecipeBundleV1.model_validate(raw)
        context = BusinessContextV1.model_validate(bundle.context)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise RecipeBundleError(
            "BUNDLE_INVALID", "El archivo no es una receta exportada por Goal Settings"
        ) from error
    if canonical_hash(_content(raw)) != bundle.content_sha256:
        raise RecipeBundleError(
            "BUNDLE_TAMPERED", "El archivo de receta fue modificado después de exportarse"
        )
    return bundle, context
