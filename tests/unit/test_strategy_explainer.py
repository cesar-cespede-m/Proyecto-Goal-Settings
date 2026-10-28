"""AI-01.8: explicación de la estrategia fija, plantilla y reescritura verificada."""

import json
from pathlib import Path

import pytest

from core.ai_engine import LLMProviderError, LLMRequest, LLMResponse
from core.ai_engine.strategy_explainer import StrategyExplainer, template_explanation
from core.domain.strategy import StrategyProposalV1
from tests.support.ml_workflow import prepared_strategy


class _Provider:
    name = "pytest-fake"
    model = "fake-model"

    def __init__(self, content: object = None, error: LLMProviderError | None = None) -> None:
        self.content, self.error = content, error

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if self.error:
            raise self.error
        content = self.content if isinstance(self.content, str) else json.dumps(self.content)
        return LLMResponse(content=content, provider=self.name, model=self.model, latency_ms=1)


@pytest.fixture
def strategy(tmp_path: Path) -> tuple[StrategyProposalV1, object, str]:
    connection, repository, _, strategy_id, _ = prepared_strategy(tmp_path)
    row = repository.get_strategy(strategy_id)
    yield StrategyProposalV1.model_validate_json(row["strategy_json"]), repository, strategy_id
    connection.close()


def test_template_uses_plain_language_and_real_split(strategy) -> None:
    proposal, _, _ = strategy
    text = template_explanation(proposal).model_dump_json()
    assert "60 %" in text and "20 %" in text
    for jargon in ("recall", "split", "holdout", "seed", "RandomForest", "LogisticRegression"):
        assert jargon not in text


def _rewrite(template, **changes):
    data = template.model_dump(mode="json")
    data.update(changes)
    return data


def test_valid_llm_rewrite_is_accepted_and_stored(strategy) -> None:
    proposal, repository, strategy_id = strategy
    template = template_explanation(proposal)
    rewritten = _rewrite(template, summary="Así se entrenará, sólo cuando tú lo apruebes.")
    result = StrategyExplainer(_Provider(rewritten)).explain(proposal)
    assert result.source == "LLM"

    explanation_id = repository.add_strategy_explanation(
        strategy_version_id=strategy_id,
        source=result.source,
        explanation=result.explanation.model_dump(mode="json"),
        ai_trace=result.trace,
    )
    latest = repository.get_latest_strategy_explanation(strategy_id)
    assert latest["id"] == explanation_id and latest["source"] == "LLM"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"summary": "Esto mejorará la precisión al 97 % garantizado."}, "UNVERIFIED_CONTENT"),
        ({"summary": "Ahorrarás USD en mantenimiento cada mes."}, "UNVERIFIED_CONTENT"),
        ({"sections": []}, "SCHEMA_INVALID"),
        ({"extra": "x"}, "SCHEMA_INVALID"),
    ],
)
def test_unsafe_rewrite_falls_back_to_template(strategy, change, reason) -> None:
    proposal, _, _ = strategy
    template = template_explanation(proposal)
    result = StrategyExplainer(_Provider(_rewrite(template, **change))).explain(proposal)
    assert (result.source, result.trace["fallback_reason"]) == ("TEMPLATE", reason)
    assert result.explanation == template


def test_without_provider_or_on_error_the_template_is_used(strategy) -> None:
    proposal, _, _ = strategy
    assert StrategyExplainer(None).explain(proposal).source == "TEMPLATE"
    failed = StrategyExplainer(_Provider(error=LLMProviderError("PROVIDER_TIMEOUT"))).explain(
        proposal
    )
    assert (failed.source, failed.trace["fallback_reason"]) == ("TEMPLATE", "PROVIDER_TIMEOUT")


def test_metric_sentence_reads_as_a_complete_phrase(strategy) -> None:
    """Recorrido 22/10: «la medida principal mira de los eventos reales…» quedaba trunca."""
    proposal, _, _ = strategy
    recall = proposal.model_copy(update={"primary_metric": "recall"})
    text = template_explanation(recall).model_dump_json()
    assert "mira cuántos de los casos reales logra detectar" in text
    assert "mira de los" not in text
