"""C2-02 (ciclo 2): registro de decisiones tomadas con datos y de lo que pasó después.

Cerrar el ciclo «dato → decisión → resultado» es lo que permite medir si los análisis y los
modelos sirven. Todo queda sólo de inserción: el resultado observado se agrega, nunca reemplaza
la decisión original.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from core.contracts.platform import DecisionRecordV1
from core.db.connection import transaction
from core.domain.errors import DomainError


class DecisionRecordError(DomainError):
    """Decisión o resultado que no cumple el contrato."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class StoredDecision:
    id: str
    record: DecisionRecordV1
    outcomes: list[dict[str, Any]]


class DecisionService:
    def __init__(self, repository: Any) -> None:
        self.repository = repository
        self.connection = repository.connection

    def record(
        self,
        project_id: str,
        *,
        decision: str,
        based_on: str,
        reference_id: str,
        actor_label: str,
    ) -> str:
        try:
            record = DecisionRecordV1(
                decision=decision.strip(),
                based_on=based_on,
                reference_id=reference_id,
                actor_label=actor_label.strip(),
                decided_at=datetime.now(UTC),
            )
        except ValidationError as error:
            raise DecisionRecordError(
                "INVALID_DECISION_RECORD",
                "La decisión necesita texto, a qué se refiere y quién la toma",
            ) from error
        decision_id = str(uuid4())
        with transaction(self.connection):
            self.connection.execute(
                "INSERT INTO decision_records(id, project_id, record_json) VALUES (?, ?, ?)",
                (decision_id, project_id, record.model_dump_json()),
            )
            self.repository._audit(
                project_id=project_id,
                event_type="DECISION_RECORDED",
                entity_type="DECISION",
                entity_id=decision_id,
                detail={
                    "actor_label": record.actor_label,
                    "based_on": record.based_on,
                    "reference_id": record.reference_id,
                },
            )
        return decision_id

    def observe(self, decision_id: str, *, observed_result: str, actor_label: str) -> str:
        row = self.connection.execute(
            "SELECT project_id, record_json FROM decision_records WHERE id = ?", (decision_id,)
        ).fetchone()
        if row is None:
            raise DecisionRecordError("DECISION_NOT_FOUND", "La decisión no existe")
        record = DecisionRecordV1.model_validate_json(row["record_json"])
        observed_at = datetime.now(UTC)
        try:
            DecisionRecordV1.model_validate(
                {
                    **record.model_dump(),
                    "observed_result": observed_result.strip(),
                    "observed_at": observed_at,
                }
            )
        except ValidationError as error:
            raise DecisionRecordError(
                "INVALID_DECISION_OUTCOME", "Describe qué pasó después de la decisión"
            ) from error
        if not observed_result.strip() or not actor_label.strip():
            raise DecisionRecordError(
                "INVALID_DECISION_OUTCOME", "Describe qué pasó y quién lo observó"
            )
        outcome_id = str(uuid4())
        with transaction(self.connection):
            self.connection.execute(
                """
                INSERT INTO decision_outcomes(
                    id, decision_id, observed_result, observed_at, actor_label
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    outcome_id,
                    decision_id,
                    observed_result.strip(),
                    observed_at.isoformat(),
                    actor_label.strip(),
                ),
            )
            self.repository._audit(
                project_id=str(row["project_id"]),
                event_type="DECISION_OUTCOME_OBSERVED",
                entity_type="DECISION",
                entity_id=decision_id,
                detail={"actor_label": actor_label.strip()},
            )
        return outcome_id

    def list(self, project_id: str) -> list[StoredDecision]:
        rows = self.connection.execute(
            "SELECT id, record_json FROM decision_records WHERE project_id = ? "
            "ORDER BY created_at DESC, rowid DESC",
            (project_id,),
        ).fetchall()
        decisions = []
        for row in rows:
            outcomes = [
                dict(item)
                for item in self.connection.execute(
                    "SELECT observed_result, observed_at, actor_label FROM decision_outcomes "
                    "WHERE decision_id = ? ORDER BY created_at, rowid",
                    (row["id"],),
                ).fetchall()
            ]
            decisions.append(
                StoredDecision(
                    id=str(row["id"]),
                    record=DecisionRecordV1.model_validate(json.loads(row["record_json"])),
                    outcomes=outcomes,
                )
            )
        return decisions
