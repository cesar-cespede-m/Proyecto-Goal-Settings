"""Valida acciones y referencias del agente antes de permitir cualquier efecto."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from core.ai_engine.agents.context import ROLE_TOOLS, clean_text
from core.contracts.agent_reasoning import AgentProposalV1
from core.orchestration.runtime import OrchestrationError
from core.orchestration.tools import registered_tools

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_MONEY = re.compile(r"[$€£%]|\b(?:USD|CLP|EUR|mill[oó]n|millones)\b", re.IGNORECASE)
_METRIC = re.compile(
    r"\b(?:precision|precisión|recall|accuracy|auc|f1|lift|exactitud)\b", re.IGNORECASE
)


def _human_text(context: dict[str, Any]) -> str:
    decision = context.get("decision") or {}
    spec = decision.get("spec") or {}
    values = [context.get("objective"), *context["business"].values()]
    values.extend(spec.get(name) for name in ("question", "action", "alternative", "kpi"))
    values.extend(spec.get("constraints", []))
    predictive = context.get("predictive_definition") or {}
    values.extend(
        predictive.get(name)
        for name in (
            "population",
            "target_definition",
            "prediction_moment",
        )
    )
    return " ".join(value for value in values if isinstance(value, str))


def validate_proposal(content: str, context: dict[str, Any]) -> AgentProposalV1:
    if len(content.encode("utf-8")) > 65536:
        raise OrchestrationError("AGENT_INVALID_OUTPUT", "La respuesta excede el contrato")
    try:
        proposal = AgentProposalV1.model_validate(json.loads(content))
    except (ValueError, ValidationError) as error:
        raise OrchestrationError(
            "AGENT_INVALID_OUTPUT", "La respuesta no cumple el contrato"
        ) from error
    refs = {(ref["kind"], ref["id"]): ref for ref in context["available_refs"]}
    for ref in proposal.evidence_refs:
        expected = refs.get((ref.kind, ref.id))
        if expected is None or (ref.sha256 is not None and ref.sha256 != expected["sha256"]):
            raise OrchestrationError(
                "AGENT_REFERENCE_NOT_PROVIDED", "La referencia no estaba disponible"
            )
    texts = [proposal.message, *proposal.questions]
    if proposal.delegate_objective:
        texts.append(proposal.delegate_objective)
    if proposal.action == "delegate" and (
        context["role"] != "coordinator"
        or proposal.delegate_role not in {"business", "data", "modeling", "value_evidence"}
    ):
        raise OrchestrationError("DELEGATION_ROLE_DENIED", "Ese especialista no está habilitado")
    if proposal.tool_call is not None:
        name = proposal.tool_call.tool_name
        if name not in ROLE_TOOLS[context["role"]]:
            raise OrchestrationError(
                "TOOL_ROLE_DENIED", "La herramienta no está habilitada para este rol"
            )
        try:
            arguments = registered_tools()[name].arguments.model_validate(
                proposal.tool_call.arguments
            )
        except ValidationError as error:
            raise OrchestrationError(
                "INVALID_TOOL_ARGUMENTS", "La llamada contiene argumentos no permitidos"
            ) from error
        if hasattr(arguments, "dataset_id"):
            if ("dataset", arguments.dataset_id) not in refs:
                raise OrchestrationError(
                    "AGENT_REFERENCE_NOT_PROVIDED", "La fuente no estaba disponible"
                )
            if name == "aggregate_analysis" and arguments.column_alias is not None:
                dataset = next(
                    (item for item in context["datasets"] if item["id"] == arguments.dataset_id),
                    {},
                )
                columns = (dataset.get("profile") or {}).get("columns", [])
                if arguments.column_alias not in {column["ref"] for column in columns}:
                    raise OrchestrationError(
                        "INVALID_TOOL_ARGUMENTS", "El alias no estaba disponible"
                    )
        for field, kind in (("strategy_id", "strategy"), ("run_id", "run")):
            identifier = getattr(arguments, field, None)
            if identifier is not None and (kind, identifier) not in refs:
                raise OrchestrationError(
                    "AGENT_REFERENCE_NOT_PROVIDED", "El artefacto no estaba disponible"
                )
        if name == "propose_value_policy":
            if ("economic_scenario", arguments.economic_version_id) not in refs:
                raise OrchestrationError("AGENT_REFERENCE_NOT_PROVIDED", "Falta el escenario")
            current_policy = context.get("policy")
            if arguments.based_on_version_id != (current_policy["id"] if current_policy else None):
                raise OrchestrationError("POLICY_VERSION_CONFLICT", "La política cambió")
        if name == "save_decision_draft":
            if arguments.spec.owner_label is not None:
                raise OrchestrationError(
                    "TOOL_HUMAN_FIELD_FORBIDDEN", "El responsable debe declararlo una persona"
                )
            current = context.get("decision")
            if arguments.based_on_version_id != (current["id"] if current else None):
                raise OrchestrationError(
                    "SPEC_VERSION_CONFLICT", "La propuesta no parte de la versión revisada"
                )
            for kind, identifier in (
                ("predictive_spec", arguments.spec.predictive_version_id),
                ("economic_scenario", arguments.spec.economic_version_id),
            ):
                if identifier is not None and (kind, identifier) not in refs:
                    raise OrchestrationError(
                        "AGENT_REFERENCE_NOT_PROVIDED", "La referencia no estaba disponible"
                    )
            texts.extend(
                getattr(arguments.spec, name)
                for name in (
                    "question",
                    "action",
                    "alternative",
                    "kpi",
                )
                if getattr(arguments.spec, name) is not None
            )
            texts.extend(arguments.spec.constraints)
        if name == "evaluate_economic_scenario":
            if ("economic_scenario", arguments.economic_version_id) not in refs:
                raise OrchestrationError(
                    "AGENT_REFERENCE_NOT_PROVIDED", "El escenario no estaba disponible"
                )
            if arguments.selected_cases is not None:
                raise OrchestrationError(
                    "TOOL_HUMAN_FIELD_FORBIDDEN", "Este turno usa el cupo declarado por la persona"
                )
    known_numbers = set(_NUMBER.findall(_human_text(context)))
    for text in texts:
        numbers = set(_NUMBER.findall(text))
        if (
            (numbers and _MONEY.search(text))
            or not numbers.issubset(known_numbers)
            or (numbers and _METRIC.search(text))
        ):
            raise OrchestrationError(
                "AGENT_UNSUPPORTED_CLAIM", "Las cifras deben venir de evidencia determinista"
            )
        if clean_text(text, limit=max(len(text), 1)) != text:
            raise OrchestrationError(
                "AGENT_SENSITIVE_OUTPUT", "La propuesta contiene texto que requiere revisión"
            )
    return proposal
