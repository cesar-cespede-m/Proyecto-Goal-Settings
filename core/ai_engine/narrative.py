"""Redacción del reporte con LLM, verificada; ante cualquier falla, reporte por plantilla."""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.ai_engine.prompts import EVENT_PLACEHOLDER, PROMPT_VERSION, build_narrative_request
from core.ai_engine.provider import LLMProvider, LLMProviderError, LLMResponse
from core.domain.models import ReportPayload
from core.reporting.verification import verify_report

NarrativeMode = Literal["LLM", "TEMPLATE_FALLBACK", "TEMPLATE_DISABLED"]
# Dinero e impacto económico no se midieron: un LLM no puede introducirlos.
_FORBIDDEN = re.compile(r"[$€£]|\b(usd|clp|eur|pesos|d[oó]lares|euros)\b", re.IGNORECASE)
# UX-24: el reporte es para personas sin formación técnica; con jerga se usa la plantilla.
_JARGON = re.compile(
    r"Classifier|LogisticRegression|\b(recall|precision|baseline|holdout|umbral|"
    r"validaci[oó]n|permutaci[oó]n|intervalo de confianza|F1)\b",
    re.IGNORECASE,
)
_LLM_LIMITATION = (
    "Redacción de IA generativa verificada contra las métricas; la persona revisora valida "
    "el sentido."
)
_AMBIGUOUS_REPORT_LANGUAGE = re.compile(
    r"casos del evento|punto desde el que avisa",
    re.IGNORECASE,
)


class _NarrativeInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_index: int = Field(ge=0)
    kind: Literal["FACT", "INTERPRETATION", "HYPOTHESIS", "RECOMMENDATION"]
    claim: str = Field(min_length=10, max_length=600)
    evidence_refs: list[str] = Field(min_length=1, max_length=8)


class _NarrativeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    insights: list[_NarrativeInsight] = Field(min_length=1, max_length=8)


class NarrativeResult(BaseModel):
    payload: ReportPayload
    mode: NarrativeMode
    reason: str | None = None


class NarrativeService:
    def __init__(self, provider: LLMProvider | None) -> None:
        self.provider = provider

    def _trace(
        self,
        mode: NarrativeMode,
        *,
        response: LLMResponse | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "mode": mode,
            "provider": self.provider.name if self.provider else None,
            "model": response.model if response else getattr(self.provider, "model", None),
            "prompt_template_version": PROMPT_VERSION if self.provider else None,
            "response_id": response.response_id if response else None,
            "latency_ms": response.latency_ms if response else None,
            "input_tokens": response.input_tokens if response else None,
            "output_tokens": response.output_tokens if response else None,
            "fallback_reason": reason,
        }

    def _with(
        self,
        template: ReportPayload,
        *,
        generator: str,
        trace: dict[str, Any],
        insights: list[dict[str, Any]] | None = None,
    ) -> ReportPayload:
        data = template.model_dump(mode="json")
        data["generator"] = generator
        data["ai_trace"] = trace
        if insights is not None:
            data["insights"] = insights
        return ReportPayload.model_validate(data)

    def _fallback(
        self, template: ReportPayload, reason: str, response: LLMResponse | None = None
    ) -> NarrativeResult:
        trace = self._trace("TEMPLATE_FALLBACK", response=response, reason=reason)
        return NarrativeResult(
            payload=self._with(template, generator="TEMPLATE", trace=trace),
            mode="TEMPLATE_FALLBACK",
            reason=reason,
        )

    def narrate(self, template: ReportPayload) -> NarrativeResult:
        """`template` debe venir ya verificado; nunca se pierde su contenido."""

        if self.provider is None:
            trace = self._trace("TEMPLATE_DISABLED")
            return NarrativeResult(
                payload=self._with(template, generator="TEMPLATE", trace=trace),
                mode="TEMPLATE_DISABLED",
            )
        try:
            response = self.provider.complete_json(build_narrative_request(template))
        except LLMProviderError as error:
            return self._fallback(template, error.code)
        try:
            output = _NarrativeOutput.model_validate(json.loads(response.content))
        except json.JSONDecodeError:
            return self._fallback(template, "INVALID_JSON", response)
        except ValidationError:
            return self._fallback(template, "SCHEMA_INVALID", response)
        if any(_FORBIDDEN.search(item.claim) for item in output.insights):
            return self._fallback(template, "FORBIDDEN_CONTENT", response)

        expected_indices = list(range(len(template.insights)))
        if [item.template_index for item in output.insights] != expected_indices:
            return self._fallback(template, "NARRATIVE_CONTRACT_MISMATCH", response)
        for item in output.insights:
            original = template.insights[item.template_index]
            if item.kind != original.kind or item.evidence_refs != original.evidence_refs:
                return self._fallback(template, "NARRATIVE_CONTRACT_MISMATCH", response)

        label = str(template.metrics.get("event_cases_label") or "casos del resultado elegido")
        insights = [
            {
                **item.model_dump(mode="json", exclude={"template_index"}),
                "claim": item.claim.replace(EVENT_PLACEHOLDER, label),
                "limitations": [_LLM_LIMITATION],
                "source": "LLM",
            }
            for item in output.insights
        ]
        candidate = self._with(
            template,
            generator="LLM",
            trace=self._trace("LLM", response=response),
            insights=insights,
        )
        errors = verify_report(candidate)
        if errors:
            return self._fallback(template, f"UNVERIFIED_CLAIMS:{errors[0]}", response)
        if any(
            _JARGON.search(item.claim) or _AMBIGUOUS_REPORT_LANGUAGE.search(item.claim)
            for item in output.insights
        ):
            return self._fallback(template, "TECHNICAL_JARGON", response)
        return NarrativeResult(payload=candidate, mode="LLM")
