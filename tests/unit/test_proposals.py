"""GA-09: la IA propone opciones; el núcleo valida y siempre tiene respaldo."""

import json
from pathlib import Path

import pandas as pd
import pytest

from core.ai_engine.proposals import ProposalService, ProposalSetV1
from core.ai_engine.provider import LLMProviderError, LLMResponse
from core.data_engine import analysis_schema, validate_and_profile
from tests.support.domains import DOMAINS

SCHEMAS = {
    "fuente": {
        "momento": "datetime",
        "grupo": "category",
        "valor": "number",
        "codigo": "identifier",
        "contacto": "personal",
    }
}
CATEGORY_VALUES = {"fuente": {"grupo": ["a", "b"]}}


def _proposals(*, questions=None, analyses=None, models=None) -> str:
    return json.dumps(
        {
            "questions": questions or [],
            "analyses": analyses or [],
            "models": models or [],
        }
    )


def _model(**changes) -> dict:
    model = {
        "task_type": "binary_classification",
        "target_column": "grupo",
        "suggested_columns": ["valor", "momento"],
        "reason": "Permite anticipar una categoría con datos disponibles.",
        "leakage_risks": ["Revisar cuándo se conoce cada entrada."],
    }
    return {**model, **changes}


class FakeProvider:
    name, model = "fake", "fake"

    def __init__(self, *contents: str, error: str | None = None) -> None:
        self.contents, self.error, self.requests = list(contents), error, []

    def complete_json(self, request):
        self.requests.append(request)
        if self.error:
            raise LLMProviderError(self.error)
        return LLMResponse(
            content=self.contents.pop(0), provider=self.name, model=self.model, latency_ms=1
        )


def test_valid_ai_response_is_kept_and_uses_a_strict_schema() -> None:
    provider = FakeProvider(
        _proposals(
            questions=["¿Cómo cambia el valor entre grupos?"],
            analyses=[{"kind": "segmentation", "reason": "Compara números entre categorías."}],
            models=[_model()],
        )
    )

    answer = ProposalService(provider).propose(
        SCHEMAS,
        context_notes="Contexto autorizado",
        category_values=CATEGORY_VALUES,
    )

    assert answer.mode == "LLM"
    assert answer.proposals.models[0].target_column == "grupo"
    schema = provider.requests[0].json_schema
    assert schema["additionalProperties"] is False
    assert schema["properties"]["questions"]["maxItems"] == 5
    assert "valores_de_categorias_autorizados" in provider.requests[0].user


def test_personal_identifier_and_unknown_columns_are_removed() -> None:
    answer = ProposalService(
        FakeProvider(
            _proposals(
                questions=["¿Qué opciones conviene revisar?"],
                analyses=[
                    {"kind": "descriptive", "reason": "Resume los tipos disponibles."},
                    {"kind": "invented", "reason": "No existe."},
                ],
                models=[
                    _model(
                        suggested_columns=[
                            "valor",
                            "contacto",
                            "codigo",
                            "inventada",
                        ]
                    ),
                    _model(target_column="contacto"),
                ],
            )
        )
    ).propose(SCHEMAS, category_values=CATEGORY_VALUES)

    assert answer.mode == "LLM"
    assert [item.kind for item in answer.proposals.analyses] == ["descriptive"]
    assert len(answer.proposals.models) == 1
    assert answer.proposals.models[0].suggested_columns == ["valor"]


def test_incompatible_targets_are_discarded() -> None:
    answer = ProposalService(
        FakeProvider(
            _proposals(
                questions=["¿Qué resultado tendría sentido anticipar?"],
                models=[
                    _model(task_type="regression", target_column="grupo"),
                    _model(task_type="binary_classification", target_column="valor"),
                ],
            )
        )
    ).propose(SCHEMAS, category_values=CATEGORY_VALUES)

    assert answer.mode == "LLM"
    assert answer.proposals.models == []


def test_a_fully_rejected_answer_retries_with_the_exact_reason() -> None:
    bad = _proposals(analyses=[{"kind": "invented", "reason": "No existe."}])
    good = _proposals(analyses=[{"kind": "descriptive", "reason": "Resume los datos."}])
    provider = FakeProvider(bad, good)

    answer = ProposalService(provider).propose(SCHEMAS)

    assert answer.mode == "LLM" and len(provider.requests) == 2
    assert "analyses[0].kind: la clave «invented» no existe en el registro" in (
        provider.requests[1].user
    )


def test_provider_failure_uses_the_deterministic_fallback() -> None:
    answer = ProposalService(FakeProvider(error="PROVIDER_TIMEOUT")).propose(
        SCHEMAS, category_values=CATEGORY_VALUES
    )

    assert answer.mode == "FALLBACK" and answer.fallback_reason == "PROVIDER_TIMEOUT"
    assert answer.proposals.questions
    assert {item.kind for item in answer.proposals.analyses} <= {
        "descriptive",
        "temporal",
        "segmentation",
        "outcome",
        "quality",
    }
    assert answer.proposals.models


def _schema(path: Path) -> dict[str, str]:
    profile = validate_and_profile(path.name, path.read_bytes())
    return analysis_schema(profile.model_dump())


@pytest.mark.parametrize("domain", DOMAINS, ids=lambda domain: domain.key)
def test_disabled_fallback_is_valid_in_every_business(domain) -> None:
    schemas = {f"fuente_{index}": _schema(path) for index, path in enumerate(domain.files)}
    category_values: dict[str, dict[str, list[str]]] = {}
    for index, path in enumerate(domain.files):
        source = f"fuente_{index}"
        frame = pd.read_csv(path, low_memory=False)
        category_values[source] = {
            column: frame[column].dropna().astype(str).unique().tolist()
            for column, column_type in schemas[source].items()
            if column_type == "category"
        }

    answer = ProposalService(None).propose(schemas, category_values=category_values)

    assert answer.mode == "DISABLED"
    assert ProposalSetV1.model_validate(answer.proposals.model_dump()) == answer.proposals
    assert answer.proposals.questions or answer.proposals.analyses or answer.proposals.models
    for model in answer.proposals.models:
        all_types = {
            column: column_type
            for schema in schemas.values()
            for column, column_type in schema.items()
        }
        assert all_types[model.target_column] not in {"personal", "identifier"}
        assert all(
            all_types[column] not in {"personal", "identifier"}
            for column in model.suggested_columns
        )
