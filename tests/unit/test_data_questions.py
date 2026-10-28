"""GA-04a: la IA traduce preguntas a planes válidos; plantillas y explicación sin IA."""

import json
from pathlib import Path

import pytest

from core.ai_engine.data_questions import _STEP_FIELDS, DataQuestionService, to_contract
from core.ai_engine.provider import LLMProviderError, LLMResponse
from core.analysis.guided import describe_plan, suggested_questions
from core.analysis.validation import validate_plan
from core.data_engine import analysis_schema, validate_and_profile
from tests.support.domains import DOMAINS

SCHEMAS = {
    "f": {"momento": "datetime", "grupo": "category", "valor": "number", "codigo": "identifier"},
    "g": {"grupo": "category", "nivel": "category"},
}


def _step(op: str, **fields) -> dict:
    step = dict.fromkeys(_STEP_FIELDS)
    step.update(op=op, **fields)
    return step


def _answer(steps, *, sources=("f",), output=None, answerable=True, reason="") -> str:
    return json.dumps(
        {
            "answerable": answerable,
            "reason": reason,
            "sources": list(sources),
            "steps": steps,
            "output": output or {"chart": "table", "x": None, "y": None, "color": None},
        }
    )


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


HOURLY = [
    _step("derive_datetime", column="momento", output_column="hora", part="hour_of_day"),
    _step("derive_datetime", column="momento", output_column="dia", part="day_of_week"),
    _step("filter", column="dia", operator="lt", value=5),
    _step(
        "group_by",
        by=["hora"],
        aggregations=[{"function": "count", "column": None, "output_column": "casos"}],
    ),
    _step("sort", by=["hora"], descending=False),
]


def test_flat_steps_become_a_valid_contract_plan() -> None:
    provider = FakeProvider(
        _answer(HOURLY, output={"chart": "line", "x": "hora", "y": "casos", "color": None})
    )
    answer = DataQuestionService(provider).plan("¿A qué hora hay más casos?", SCHEMAS)

    assert answer.mode == "LLM" and answer.attempts == 1
    assert [step.op for step in answer.plan.steps][-1] == "sort"
    sent = provider.requests[0].user
    assert "momento" in sent and "datetime" in sent
    assert "valores_de_categorias_autorizados" not in sent


def test_every_step_shape_converts() -> None:
    steps = [
        _step("select", columns=["grupo", "valor", "momento"]),
        _step("rename", rename=[{"from_name": "valor", "to_name": "monto"}]),
        _step("filter", column="grupo", operator="in", value_list=["a", "b"]),
        _step("derive_bucket", column="monto", output_column="tramo", method="quantile", bins=4),
        _step("join", source="g", left_on="grupo", right_on="grupo", how="left"),
        _step("pivot", index=["tramo"], pivot_columns="nivel", values="monto", aggregation="sum"),
    ]
    plan = to_contract(json.loads(_answer(steps, sources=("f", "g"))))

    assert plan.steps[1].mapping == {"valor": "monto"}
    assert plan.steps[2].value == ["a", "b"]
    assert plan.steps[5].columns == "nivel"


def test_single_column_in_column_field_is_accepted_for_lists() -> None:
    plan = to_contract(json.loads(_answer([_step("sort", column="valor", descending=True)])))

    assert plan.steps[0].by == ["valor"]


def test_rejected_plan_gets_one_retry_with_the_reason() -> None:
    bad = _answer([_step("group_by", by=["inventada"], aggregations=[])])
    provider = FakeProvider(bad, _answer(HOURLY))
    answer = DataQuestionService(provider).plan("pregunta", SCHEMAS)

    assert answer.mode == "LLM" and answer.attempts == 2
    assert "RECHAZADO" in provider.requests[1].user


def test_two_rejections_fall_back_with_a_diagnosis() -> None:
    empty = _answer([])
    answer = DataQuestionService(FakeProvider(empty, empty)).plan("pregunta", SCHEMAS)

    assert answer.mode == "FALLBACK" and answer.fallback_reason == "INVALID_PLAN"
    assert "EMPTY_PLAN" in answer.last_rejection


def test_unanswerable_questions_always_explain_why() -> None:
    silent = _answer([], answerable=False, reason="")
    answer = DataQuestionService(FakeProvider(silent)).plan("pregunta", SCHEMAS)

    assert answer.mode == "NOT_ANSWERABLE" and answer.reason


def test_disabled_and_failing_provider_never_break() -> None:
    assert DataQuestionService(None).plan("pregunta", SCHEMAS).mode == "DISABLED"
    failing = DataQuestionService(FakeProvider(error="PROVIDER_TIMEOUT")).plan("x", SCHEMAS)
    assert failing.mode == "FALLBACK" and failing.fallback_reason == "PROVIDER_TIMEOUT"


def test_category_values_travel_only_when_authorized() -> None:
    provider = FakeProvider(_answer(HOURLY))
    DataQuestionService(provider).plan(
        "pregunta", SCHEMAS, category_values={"f": {"grupo": ["a", "b"]}}
    )

    sent = provider.requests[0].user
    assert "valores_de_categorias_autorizados" in sent and '"a"' in sent


def _schema(path: Path) -> dict[str, str]:
    return analysis_schema(validate_and_profile(path.name, path.read_bytes()).model_dump())


@pytest.mark.parametrize("domain", DOMAINS, ids=lambda domain: domain.key)
def test_suggested_questions_are_valid_in_every_business(domain) -> None:
    schemas = {f"fuente_{index}": _schema(path) for index, path in enumerate(domain.files)}
    suggestions = suggested_questions(schemas)

    assert suggestions
    for suggestion in suggestions:
        validate_plan(suggestion.plan, schemas)
        assert suggestion.text.startswith("¿")


def test_plan_is_explained_in_plain_words() -> None:
    plan = to_contract(json.loads(_answer(HOURLY)))
    lines = describe_plan(plan)

    assert lines[0] == "Usé «f»."
    assert "Saqué «hora del día» de «momento»." in lines
    assert "Dejé sólo los casos donde «dia» es menor que 5." in lines
    assert "Calculé cantidad de casos por «hora»." in lines


# -- GA-04b: interpretar con cifras verificadas -----------------------------------------


class _Payload:
    def __init__(self, columns, rows) -> None:
        from types import SimpleNamespace

        self.columns = [SimpleNamespace(name=name) for name in columns]
        self.rows = rows
        self.sent_row_count = len(rows)


PAYLOAD = _Payload(["grupo", "promedio", "proporcion"], [["a", 3.62, 0.25], ["b", 1234.5, 0.75]])


def _interpretation(**fields) -> str:
    base = {"explanation": "", "findings": [], "recommendation": "", "caveats": []}
    return json.dumps({**base, **fields})


def _interpret(provider):
    from core.ai_engine.data_questions import interpret

    return interpret(
        provider,
        question="¿Qué grupo es mayor?",
        plan_description=["Usé «f»."],
        payload=PAYLOAD,
        total_rows=2,
    )


def test_interpretation_with_numbers_from_the_result_is_kept() -> None:
    text = _interpretation(
        explanation="El grupo b llega a 1.234,5 y concentra el 75 % del total.",
        findings=["El grupo a promedia 3,62."],
        recommendation="Revisar primero el grupo b.",
        caveats=["Una diferencia no prueba una causa."],
    )
    answer = _interpret(FakeProvider(text))

    assert answer.mode == "LLM"
    assert answer.findings == ["El grupo a promedia 3,62."]


def test_invented_numbers_discard_the_answer() -> None:
    text = _interpretation(explanation="El grupo b creció 48 % el último mes.")
    answer = _interpret(FakeProvider(text))

    assert answer.mode == "FALLBACK" and answer.fallback_reason == "UNVERIFIED_CLAIMS"
    assert "0.75" in answer.explanation  # resumen fijo con cifras reales


def test_without_ai_a_fixed_summary_names_highest_and_lowest() -> None:
    answer = _interpret(None)

    assert answer.mode == "DISABLED"
    assert "mayor valor de «proporcion» es 0.75" in answer.explanation
    assert "menor es 0.25" in answer.explanation


def test_modeling_table_mode_rejects_grouped_plans_and_retries() -> None:
    row_level = [
        _step("join", source="g", left_on="grupo", right_on="grupo", how="left"),
        _step("threshold_label", column="valor", output_column="alto", percentile=0.5),
        _step("select", columns=["grupo", "nivel", "alto"]),
    ]
    provider = FakeProvider(_answer(HOURLY), _answer(row_level, sources=("f", "g")))
    answer = DataQuestionService(provider).plan(
        "Prepara una tabla para anticipar valores altos", SCHEMAS, modeling_table=True
    )

    assert answer.mode == "LLM" and answer.attempts == 2
    assert "MODELING_TABLE_NOT_ROW_LEVEL" in provider.requests[1].user
    assert "TABLA PARA ENTRENAR" in provider.requests[0].user


def test_confirmed_outcome_column_travels_by_name_only() -> None:
    """Recorrido 22/10: la IA definió el resultado con otra columna en vez de la confirmada."""
    column = next(iter(SCHEMAS["f"]))
    provider = FakeProvider(_answer(HOURLY), _answer(HOURLY))
    DataQuestionService(provider).plan("pregunta", SCHEMAS, outcome=("f", column))
    sent = json.loads(provider.requests[0].user.removeprefix("DATOS_NO_CONFIABLES:\n"))

    assert sent["resultado_del_proyecto"]["columna"] == column
    assert set(sent["resultado_del_proyecto"]) == {"fuente", "columna", "guia"}

    DataQuestionService(provider).plan("pregunta", SCHEMAS, outcome=("f", "no_existe"))
    assert "resultado_del_proyecto" not in provider.requests[1].user
