"""EXT-C: preguntas sobre resultados respondidas sólo con cifras verificadas."""

import json

import pytest

from core.ai_engine import LLMProviderError, LLMRequest, LLMResponse
from core.ai_engine.results_qa import ResultsQA, build_evidence
from core.domain.models import ReportPayload
from core.reporting import ReportService
from tests.support.ml_workflow import prepared_strategy


class _Provider:
    name = "pytest-fake"
    model = "fake-model"

    def __init__(self, answer: object = None, error: LLMProviderError | None = None) -> None:
        self.answer, self.error = answer, error
        self.requests: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error:
            raise self.error
        content = self.answer if isinstance(self.answer, str) else json.dumps(self.answer)
        return LLMResponse(content=content, provider=self.name, model=self.model, latency_ms=1)


@pytest.fixture(scope="module")
def payload(tmp_path_factory) -> ReportPayload:
    connection, repository, ml, strategy_id, _ = prepared_strategy(tmp_path_factory.mktemp("qa"))
    repository.decide_training(
        strategy_version_id=strategy_id, decision="APPROVED", actor_label="r"
    )
    run = ml.execute(strategy_id)
    report = repository.get_report(ReportService(repository).create_template_report(run.run_id))
    connection.close()
    return ReportPayload.model_validate(json.loads(report["payload_json"]))


def _answer(text: str, refs: list[str], kind: str = "FACT") -> dict[str, object]:
    return {"kind": kind, "answer": text, "evidence_refs": refs, "suggestion": "Nada más."}


def test_verified_answer_is_accepted_and_question_is_redacted(payload: ReportPayload) -> None:
    recall = payload.metrics["holdout"]["recall"]
    provider = _Provider(
        _answer(
            f"El modelo detecta {recall * 100:.1f} % de los eventos.", ["metrics.holdout.recall"]
        )
    )
    result = ResultsQA(provider).ask("¿Cuántos detecta? escríbeme a persona@example.test", payload)
    assert result.mode == "LLM" and result.kind == "FACT"
    sent = provider.requests[0].user
    assert "persona@example.test" not in sent
    assert '"positive_class"' not in sent and "negative_class" not in sent


def test_invented_number_falls_back(payload: ReportPayload) -> None:
    provider = _Provider(
        _answer("El modelo detecta 97.3 % de los eventos.", ["metrics.holdout.recall"])
    )
    result = ResultsQA(provider).ask("¿Cuántos detecta?", payload)
    assert result.mode == "FALLBACK" and result.fallback_reason.startswith("UNVERIFIED_CLAIMS")
    assert result.kind == "CANNOT_ANSWER"


@pytest.mark.parametrize(
    "answer",
    [
        "no es json",
        {
            "kind": "FACT",
            "answer": "Ahorrarás USD cada mes.",
            "evidence_refs": [],
            "suggestion": "",
        },
        {"kind": "FACT", "answer": "texto", "evidence_refs": [], "suggestion": "", "extra": 1},
    ],
)
def test_malformed_or_forbidden_answers_fall_back(payload: ReportPayload, answer) -> None:
    assert ResultsQA(_Provider(answer)).ask("¿Qué pasó?", payload).mode == "FALLBACK"


def test_cannot_answer_without_numbers_is_accepted(payload: ReportPayload) -> None:
    provider = _Provider(
        _answer(
            "Con esta evidencia no se puede saber qué pasaría si priorizas evitar falsas alarmas.",
            [],
            kind="CANNOT_ANSWER",
        )
    )
    result = ResultsQA(provider).ask("¿Y si priorizo evitar falsas alarmas?", payload)
    assert result.mode == "LLM" and result.kind == "CANNOT_ANSWER"


def test_comparison_evidence_can_be_cited(payload: ReportPayload) -> None:
    experiments = [{"run_id": "x", "primary_metric": "f1", "holdout_precision": 0.5}]
    evidence = build_evidence(payload, experiments)
    assert "run_id" not in evidence["experiments"][0]
    provider = _Provider(
        _answer("La versión anterior acertaba 50 % de sus alertas.", ["metrics.experiments.0"])
    )
    assert ResultsQA(provider).ask("¿Y la anterior?", payload, experiments).mode == "LLM"


def test_without_provider_or_on_error(payload: ReportPayload) -> None:
    assert ResultsQA(None).ask("¿Qué pasó?", payload).mode == "DISABLED"
    failed = ResultsQA(_Provider(error=LLMProviderError("PROVIDER_TIMEOUT"))).ask("?", payload)
    assert (failed.mode, failed.fallback_reason) == ("FALLBACK", "PROVIDER_TIMEOUT")


def test_variable_names_never_leave_in_questions(payload: ReportPayload) -> None:
    experiments = [{"run_id": "x", "features_used": ["NOMBRE-SECRETO"], "holdout_recall": 0.5}]
    provider = _Provider(_answer("Sin cifras que citar aquí.", [], kind="CANNOT_ANSWER"))
    ResultsQA(provider).ask("¿Qué variables pesan más?", payload, experiments)
    sent = provider.requests[0].user
    assert "NOMBRE-SECRETO" not in sent and "permutation_importance" not in sent


def test_true_number_from_evidence_is_accepted_even_if_not_cited(payload: ReportPayload) -> None:
    positives = payload.metrics["holdout"]["positive_support"]
    provider = _Provider(_answer(f"Hubo {positives} eventos reales en la prueba.", []))
    assert ResultsQA(provider).ask("¿Cuántos eventos hubo?", payload).mode == "LLM"
