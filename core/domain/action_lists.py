"""V2-17: listas de acción ligadas a una predicción y su resultado observado.

La lista conserva sólo identificadores de caso. La auditoría y la medición contienen únicamente
conteos agregados: nunca filas, resultados individuales ni los identificadores de los casos.
"""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.db.connection import transaction
from core.domain.decisions import DecisionService
from core.domain.errors import DomainError
from core.ml_engine.decision import DecisionPlanV1


class ActionListError(DomainError):
    """La lista de acción o su medición no cumple el contrato."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ActionListV1(_Strict):
    schema_version: Literal["action-list.v1"] = "action-list.v1"
    id: str = Field(min_length=1, max_length=100)
    project_id: str = Field(min_length=1, max_length=100)
    run_id: str = Field(min_length=1, max_length=100)
    scoring_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_ids: list[str] = Field(min_length=1)
    attended_ids: list[str]
    control_ids: list[str]
    not_attended_ids: list[str]
    control_share: float = Field(ge=0, lt=1)
    seed: int
    plan: DecisionPlanV1 | None = None
    decision_id: str = Field(min_length=1, max_length=100)
    actor_label: str = Field(min_length=1, max_length=200)
    created_at: datetime

    @model_validator(mode="after")
    def groups_partition_cases(self) -> ActionListV1:
        all_ids = self.attended_ids + self.control_ids + self.not_attended_ids
        if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(self.case_ids):
            raise ValueError("Los grupos deben repartir todos los casos una sola vez")
        return self


class ActionGroupOutcomeV1(_Strict):
    cases: int = Field(ge=0)
    observed: int = Field(ge=0)
    positive: int = Field(ge=0)
    proportion: float | None = Field(default=None, ge=0, le=1)


class ApproximateIntervalV1(_Strict):
    lower: float = Field(ge=-1, le=1)
    upper: float = Field(ge=-1, le=1)
    confidence: Literal[0.95] = 0.95


class ActionListOutcomeV1(_Strict):
    schema_version: Literal["action-list-outcome.v1"] = "action-list-outcome.v1"
    id: str = Field(min_length=1, max_length=100)
    action_list_id: str = Field(min_length=1, max_length=100)
    attended: ActionGroupOutcomeV1
    control: ActionGroupOutcomeV1
    not_attended: ActionGroupOutcomeV1
    attended_minus_control: float | None = Field(default=None, ge=-1, le=1)
    approximate_interval: ApproximateIntervalV1 | None = None
    unknown_ids_ignored: int = Field(ge=0)
    warnings: list[str]
    summary: str = Field(min_length=1, max_length=2000)
    decision_outcome_id: str = Field(min_length=1, max_length=100)
    actor_label: str = Field(min_length=1, max_length=200)
    observed_at: datetime


class ActionListService:
    def __init__(self, repository: Any, storage_root: str | Path) -> None:
        self.repository = repository
        self.connection = repository.connection
        # Firma común de los servicios; V2-17 no crea una segunda copia en archivos.
        self.storage_root = Path(storage_root)

    def create(
        self,
        project_id: str,
        run_id: str,
        *,
        scoring_input_sha256: str,
        case_ids: list[str],
        attended_ids: list[str],
        control_share: float = 0.0,
        seed: int,
        plan: DecisionPlanV1 | None,
        actor_label: str,
    ) -> ActionListV1:
        cases = _clean_ids(case_ids)
        intended = _clean_ids(attended_ids)
        actor = actor_label.strip()
        if (
            not project_id.strip()
            or not run_id.strip()
            or not actor
            or len(actor) > 200
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not math.isfinite(control_share)
            or control_share < 0
            or control_share >= 1
            or len(cases) != len(set(cases))
            or len(intended) != len(set(intended))
            or not cases
            or not intended
            or not set(intended) <= set(cases)
            or len(scoring_input_sha256) != 64
            or any(character not in "0123456789abcdef" for character in scoring_input_sha256)
        ):
            raise ActionListError(
                "INVALID_ACTION_LIST",
                "La lista necesita casos únicos, atendidos válidos, un hash y un responsable",
            )
        if control_share > 0 and len(intended) < 2:
            raise ActionListError(
                "INVALID_ACTION_LIST",
                "Se necesitan al menos dos casos seleccionados para separar un control",
            )
        try:
            run_project_id = self.repository.get_run_project_id(run_id)
        except (LookupError, ValueError) as error:
            raise ActionListError(
                "ACTION_LIST_RUN_MISMATCH", "El modelo no existe en este proyecto"
            ) from error
        if run_project_id != project_id:
            raise ActionListError(
                "ACTION_LIST_RUN_MISMATCH", "El modelo no pertenece a este proyecto"
            )

        control_count = 0
        if control_share > 0:
            requested = math.floor(len(intended) * control_share + 0.5)
            control_count = max(1, min(len(intended) - 1, requested))
        selected_control = set(random.Random(seed).sample(intended, control_count))
        control = [case_id for case_id in intended if case_id in selected_control]
        attended = [case_id for case_id in intended if case_id not in selected_control]
        intended_set = set(intended)
        not_attended = [case_id for case_id in cases if case_id not in intended_set]

        decision_text = f"Atender {len(attended)} casos priorizados por el modelo"
        if control:
            decision_text += f" y reservar {len(control)} como grupo de control no atendido"
        decision_id = DecisionService(self.repository).record(
            project_id,
            decision=decision_text + ".",
            based_on="prediction",
            reference_id=run_id,
            actor_label=actor,
        )
        action_list = ActionListV1(
            id=str(uuid4()),
            project_id=project_id,
            run_id=run_id,
            scoring_input_sha256=scoring_input_sha256,
            case_ids=cases,
            attended_ids=attended,
            control_ids=control,
            not_attended_ids=not_attended,
            control_share=control_share,
            seed=seed,
            plan=plan,
            decision_id=decision_id,
            actor_label=actor,
            created_at=datetime.now(UTC),
        )
        with transaction(self.connection):
            self.connection.execute(
                """
                INSERT INTO action_lists(
                    id, project_id, run_id, decision_id, scoring_input_sha256, action_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    action_list.id,
                    project_id,
                    run_id,
                    decision_id,
                    scoring_input_sha256,
                    action_list.model_dump_json(),
                ),
            )
            self.repository._audit(
                project_id=project_id,
                event_type="ACTION_LIST_CREATED",
                entity_type="ACTION_LIST",
                entity_id=action_list.id,
                detail={
                    "actor_label": actor,
                    "cases": len(cases),
                    "attended": len(attended),
                    "control": len(control),
                    "not_attended": len(not_attended),
                },
            )
        return action_list

    def measure(
        self,
        action_list_id: str,
        *,
        outcomes: dict[str, Any],
        positive_class: Any,
        actor_label: str,
    ) -> ActionListOutcomeV1:
        actor = actor_label.strip()
        if not actor or len(actor) > 200 or positive_class is None:
            raise ActionListError(
                "INVALID_ACTION_OUTCOME",
                "La medición necesita el resultado esperado y quién lo observó",
            )
        row = self.connection.execute(
            "SELECT project_id, action_json FROM action_lists WHERE id = ?", (action_list_id,)
        ).fetchone()
        if row is None:
            raise ActionListError("ACTION_LIST_NOT_FOUND", "La lista de acción no existe")
        action_list = ActionListV1.model_validate_json(row["action_json"])
        known = set(action_list.case_ids)
        unknown_ids_ignored = sum(case_id not in known for case_id in outcomes)
        positive_token = _value_token(positive_class)

        def aggregate(ids: list[str]) -> ActionGroupOutcomeV1:
            values = [outcomes[case_id] for case_id in ids if outcomes.get(case_id) is not None]
            positives = sum(_value_token(value) == positive_token for value in values)
            return ActionGroupOutcomeV1(
                cases=len(ids),
                observed=len(values),
                positive=positives,
                proportion=None if not values else positives / len(values),
            )

        attended = aggregate(action_list.attended_ids)
        control = aggregate(action_list.control_ids)
        not_attended = aggregate(action_list.not_attended_ids)
        warnings: list[str] = []
        difference = None
        interval = None
        if action_list.control_ids:
            small = [
                name
                for name, group in (("atendidos", attended), ("control", control))
                if group.observed < 30
            ]
            if small:
                warnings.append(
                    "La comparación es inestable porque hay menos de 30 resultados observados "
                    f"en: {', '.join(small)}."
                )
            if attended.proportion is None or control.proportion is None:
                warnings.append(
                    "Faltan resultados observados en atendidos o control; todavía no se puede "
                    "calcular la diferencia."
                )
                summary = (
                    "Todavía no hay resultados suficientes en atendidos y control para comparar "
                    "los grupos."
                )
            else:
                difference = attended.proportion - control.proportion
                standard_error = math.sqrt(
                    attended.proportion * (1 - attended.proportion) / attended.observed
                    + control.proportion * (1 - control.proportion) / control.observed
                )
                margin = 1.96 * standard_error
                interval = ApproximateIntervalV1(
                    lower=max(-1.0, difference - margin),
                    upper=min(1.0, difference + margin),
                )
                summary = (
                    f"En atendidos, {attended.positive} de {attended.observed} resultados "
                    f"observados fueron el resultado definido ({attended.proportion:.1%}); en "
                    f"control, {control.positive} de {control.observed} "
                    f"({control.proportion:.1%}). "
                    f"La diferencia atendidos menos control fue {difference:.1%}, con intervalo "
                    f"aproximado del 95% de {interval.lower:.1%} a {interval.upper:.1%}. Esta "
                    "comparación muestra una diferencia entre grupos, pero por sí sola no prueba "
                    "que la acción la haya causado."
                )
        else:
            warnings.append("Sin grupo de control no se puede saber si actuar ayudó.")
            observed_rate = (
                "sin resultados observados"
                if attended.proportion is None
                else (
                    f"{attended.positive} de {attended.observed} resultados observados "
                    f"({attended.proportion:.1%})"
                )
            )
            summary = (
                f"En los casos atendidos hubo {observed_rate}. Sin grupo de control no se puede "
                "saber si actuar ayudó."
            )
        if unknown_ids_ignored:
            summary += (
                f" Se ignoraron {unknown_ids_ignored} identificadores que no pertenecen a la "
                "lista de acción."
            )

        decision_outcome_id = DecisionService(self.repository).observe(
            action_list.decision_id,
            observed_result=summary,
            actor_label=actor,
        )
        outcome = ActionListOutcomeV1(
            id=str(uuid4()),
            action_list_id=action_list.id,
            attended=attended,
            control=control,
            not_attended=not_attended,
            attended_minus_control=difference,
            approximate_interval=interval,
            unknown_ids_ignored=unknown_ids_ignored,
            warnings=warnings,
            summary=summary,
            decision_outcome_id=decision_outcome_id,
            actor_label=actor,
            observed_at=datetime.now(UTC),
        )
        with transaction(self.connection):
            self.connection.execute(
                """
                INSERT INTO action_list_outcomes(
                    id, action_list_id, decision_outcome_id, outcome_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    outcome.id,
                    action_list.id,
                    decision_outcome_id,
                    outcome.model_dump_json(),
                ),
            )
            self.repository._audit(
                project_id=str(row["project_id"]),
                event_type="ACTION_LIST_MEASURED",
                entity_type="ACTION_LIST",
                entity_id=action_list.id,
                detail={
                    "actor_label": actor,
                    "attended_observed": attended.observed,
                    "control_observed": control.observed,
                    "not_attended_observed": not_attended.observed,
                    "unknown_ids_ignored": unknown_ids_ignored,
                },
            )
        return outcome


def _clean_ids(values: list[str]) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        return []
    cleaned = [value.strip() for value in values]
    if any(not value or len(value) > 200 for value in cleaned):
        return []
    return cleaned


def _value_token(value: Any) -> str:
    """Compara clases conservando la diferencia entre, por ejemplo, ``1`` y ``True``."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
