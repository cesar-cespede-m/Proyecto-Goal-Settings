"""V2-24e: la IA propone qué anticipar en el recorrido, validada contra las etapas reales."""

import json
from pathlib import Path

import pandas as pd

from core.ai_engine.journey_proposal import propose_outcome
from core.ai_engine.provider import LLMRequest, LLMResponse
from core.analysis.funnel import funnel

DATA = Path(__file__).parents[2] / "data" / "synthetic" / "recorridos" / "altas_eventos.csv"
ORDER = ["registro", "verificacion", "primera_compra", "segunda_compra"]


def _funnel():
    frame = pd.read_csv(DATA)
    entity, moment, stage = frame.columns[:3]
    return funnel(frame, entity=entity, moment=moment, stage=stage, order=ORDER)


class _Provider:
    name, model = "fake", "fake"

    def __init__(self, answer: dict) -> None:
        self.answer = answer
        self.requests: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content=json.dumps(self.answer), provider="fake", model="fake", latency_ms=1
        )


def test_valid_ai_proposal_is_used_and_no_rows_are_sent() -> None:
    provider = _Provider(
        {
            "target_stage": "segunda_compra",
            "observe_days": 7,
            "within_days": 60,
            "reason": "Tu objetivo es que vuelvan a comprar, así que se mira la segunda compra.",
        }
    )
    answer = propose_outcome(
        provider, objective="Quiero que vuelvan a comprar", funnel_result=_funnel()
    )

    assert answer.mode == "LLM" and answer.suggestion.target_stage == "segunda_compra"
    assert "INC-0" not in provider.requests[0].user  # sólo agregados, nunca casos


def test_invented_stage_falls_back_to_rules() -> None:
    provider = _Provider(
        {"target_stage": "otra", "observe_days": 3, "within_days": 30, "reason": "x" * 20}
    )
    answer = propose_outcome(provider, objective="algo", funnel_result=_funnel())

    assert answer.mode == "RULES" and answer.fallback_reason == "UNKNOWN_STAGE"
    assert answer.suggestion is not None and answer.suggestion.target_stage in ORDER


def test_without_ai_uses_rules() -> None:
    assert propose_outcome(None, objective="x", funnel_result=_funnel()).mode == "RULES"
