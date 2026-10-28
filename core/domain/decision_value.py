"""MV-05: versiones de decisión, predicción y economía sin alterar gates de proyectos."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from core.analysis.snapshot_recipe import snapshot_recipe
from core.contracts.decision_value import (
    CyclePredictiveBindingV1,
    DecisionValueSpecV1,
    DecisionValueSpecVersionV1,
    EconomicScenarioVersionV1,
    PredictiveSpecV1,
    PredictiveSpecVersionV1,
)
from core.contracts.economics import EconomicScenarioV1
from core.db.connection import transaction
from core.domain.canonical import canonical_hash, canonical_json
from core.domain.capabilities import CapabilityService
from core.domain.errors import DomainError

_CATALOG = {
    "predictive": (
        "predictive_spec_versions",
        PredictiveSpecV1,
        PredictiveSpecVersionV1,
        "spec",
        "PREDICTIVE_SPEC_CREATED",
    ),
    "economic": (
        "economic_scenario_versions",
        EconomicScenarioV1,
        EconomicScenarioVersionV1,
        "scenario",
        "ECONOMIC_SCENARIO_CREATED",
    ),
    "decision": (
        "decision_value_spec_versions",
        DecisionValueSpecV1,
        DecisionValueSpecVersionV1,
        "spec",
        "DECISION_VALUE_SPEC_CREATED",
    ),
}


class DecisionValueError(DomainError, ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _label(value: str, field: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 200:
        raise DecisionValueError("INVALID_OPERATION", f"{field} requiere entre 1 y 200 caracteres")
    return value.strip()


class DecisionValueService:
    def __init__(self, repository: Any) -> None:
        self.repository = repository
        self.connection = repository.connection
        self.capabilities = CapabilityService(repository)

    def create_predictive_version(
        self,
        capability_id: str,
        *,
        spec: PredictiveSpecV1 | dict[str, Any],
        based_on_version_id: str | None = None,
        confirmed_human: bool = False,
        actor_label: str,
        operation_id: str,
    ) -> PredictiveSpecVersionV1:
        return self._create(
            "predictive",
            capability_id,
            spec,
            based_on_version_id,
            confirmed_human,
            actor_label,
            operation_id,
        )

    def create_economic_version(
        self,
        capability_id: str,
        *,
        scenario: EconomicScenarioV1 | dict[str, Any],
        based_on_version_id: str | None = None,
        confirmed_human: bool = False,
        actor_label: str,
        operation_id: str,
    ) -> EconomicScenarioVersionV1:
        return self._create(
            "economic",
            capability_id,
            scenario,
            based_on_version_id,
            confirmed_human,
            actor_label,
            operation_id,
        )

    def create_decision_version(
        self,
        capability_id: str,
        *,
        spec: DecisionValueSpecV1 | dict[str, Any],
        based_on_version_id: str | None = None,
        confirmed_human: bool = False,
        actor_label: str,
        operation_id: str,
    ) -> DecisionValueSpecVersionV1:
        return self._create(
            "decision",
            capability_id,
            spec,
            based_on_version_id,
            confirmed_human,
            actor_label,
            operation_id,
        )

    def get_predictive_version(self, version_id: str) -> PredictiveSpecVersionV1:
        return self._get("predictive", version_id)

    def get_economic_version(self, version_id: str) -> EconomicScenarioVersionV1:
        return self._get("economic", version_id)

    def get_decision_version(self, version_id: str) -> DecisionValueSpecVersionV1:
        return self._get("decision", version_id)

    def list_predictive_versions(self, capability_id: str) -> list[PredictiveSpecVersionV1]:
        return self._list("predictive", capability_id)

    def list_economic_versions(self, capability_id: str) -> list[EconomicScenarioVersionV1]:
        return self._list("economic", capability_id)

    def list_decision_versions(self, capability_id: str) -> list[DecisionValueSpecVersionV1]:
        return self._list("decision", capability_id)

    def _get(self, kind: str, version_id: str) -> Any:
        table, model_type, wrapper_type, field, _event = _CATALOG[kind]
        row = self.connection.execute(f"SELECT * FROM {table} WHERE id=?", (version_id,)).fetchone()
        if row is None:
            raise DecisionValueError("SPEC_VERSION_NOT_FOUND", "La versión no existe")
        model = model_type.model_validate_json(row["content_json"])
        if canonical_hash(model) != row["sha256"]:
            raise DecisionValueError("SPEC_HASH_MISMATCH", "El contenido de la versión cambió")
        return wrapper_type(
            **{
                name: row[name]
                for name in (
                    "id",
                    "capability_id",
                    "version_no",
                    "based_on_version_id",
                    "sha256",
                    "confirmed_human",
                    "actor_label",
                    "created_at",
                )
            },
            missing_fields=self._missing_with_references(kind, model, str(row["capability_id"])),
            **{field: model},
        )

    def _list(self, kind: str, capability_id: str) -> list[Any]:
        self.capabilities.get_capability(capability_id)
        rows = self.connection.execute(
            f"SELECT id FROM {_CATALOG[kind][0]} WHERE capability_id=? ORDER BY version_no",
            (capability_id,),
        ).fetchall()
        return [self._get(kind, str(row["id"])) for row in rows]

    def _missing_with_references(self, kind: str, model: Any, capability_id: str) -> list[str]:
        missing = model.missing_fields()
        if kind == "decision":
            for dependency, reference in (
                ("predictive", model.predictive_version_id),
                ("economic", model.economic_version_id),
            ):
                if reference is None:
                    continue
                version = self._get(dependency, reference)
                if version.capability_id != capability_id:
                    raise DecisionValueError(
                        "SPEC_SCOPE_MISMATCH", "La referencia es de otra capacidad"
                    )
                if not version.confirmed_human:
                    missing.append(f"{dependency}_version_confirmed")
        return missing

    def _create(
        self,
        kind: str,
        capability_id: str,
        content: Any,
        based_on_version_id: str | None,
        confirmed_human: bool,
        actor_label: str,
        operation_id: str,
    ) -> Any:
        actor, operation = _label(actor_label, "actor_label"), _label(operation_id, "operation_id")
        if type(confirmed_human) is not bool:
            raise DecisionValueError("INVALID_CONFIRMATION", "La confirmación debe ser explícita")
        table, model_type, _wrapper, _field, event = _CATALOG[kind]
        model = model_type.model_validate(
            content.model_dump(mode="python", warnings=False)
            if isinstance(content, model_type)
            else content
        )
        payload = model.model_dump(mode="json")
        digest = canonical_hash(payload)
        request = {
            "capability_id": capability_id,
            "content": payload,
            "based_on_version_id": based_on_version_id,
            "confirmed_human": confirmed_human,
            "actor_label": actor,
        }
        with transaction(self.connection):
            replay = self._replay(operation, event, request)
            if replay is not None:
                return self._get(kind, replay)
            capability = self.capabilities.get_capability(capability_id)
            if capability.status != "active":
                raise DecisionValueError("ENTITY_ARCHIVED", "La capacidad está archivada")
            latest = self.connection.execute(
                f"SELECT id,version_no FROM {table} WHERE capability_id=? "
                "ORDER BY version_no DESC LIMIT 1",
                (capability_id,),
            ).fetchone()
            if based_on_version_id != (None if latest is None else latest["id"]):
                raise DecisionValueError(
                    "SPEC_VERSION_CONFLICT", "Recarga la versión vigente antes de editar"
                )
            missing = self._missing_with_references(kind, model, capability_id)
            if confirmed_human and missing:
                raise DecisionValueError(
                    "SPEC_INCOMPLETE", "Falta completar o confirmar: " + ", ".join(missing)
                )
            if kind == "predictive" and model.source_context_version_id is not None:
                original = self.draft_predictive_from_context(
                    capability_id, context_version_id=model.source_context_version_id
                )
                if (
                    model.source_context_sha256 != original.source_context_sha256
                    or any(
                        canonical_json(getattr(model, name))
                        != canonical_json(getattr(original, name))
                        for name in (
                            "target_column",
                            "positive_class",
                            "prediction_moment",
                            "primary_metric",
                        )
                    )
                    or (
                        original.feature_decisions
                        and [item.model_dump(mode="json") for item in model.feature_decisions]
                        != [item.model_dump(mode="json") for item in original.feature_decisions]
                    )
                ):
                    raise DecisionValueError(
                        "SOURCE_CONTEXT_MISMATCH", "La especificación no coincide con su origen"
                    )
            if kind == "economic" and confirmed_human:
                previous = self.connection.execute(
                    "SELECT content_json FROM economic_scenario_versions "
                    "WHERE capability_id=? AND confirmed_human=1 LIMIT 1",
                    (capability_id,),
                ).fetchone()
                if previous is not None:
                    old = EconomicScenarioV1.model_validate_json(previous[0])
                    if (old.currency, old.currency_minor_units) != (
                        model.currency,
                        model.currency_minor_units,
                    ):
                        raise DecisionValueError(
                            "CURRENCY_MISMATCH",
                            "La capacidad conserva una divisa y unidad de valor",
                        )
            values = {
                "id": str(uuid4()),
                "capability_id": capability_id,
                "version_no": 1 if latest is None else int(latest["version_no"]) + 1,
                "based_on_version_id": based_on_version_id,
                "content_json": canonical_json(payload),
                "sha256": digest,
                "confirmed_human": int(confirmed_human),
                "actor_label": actor,
                "created_at": datetime.now(UTC).isoformat(),
            }
            if kind == "predictive":
                values["source_context_version_id"] = model.source_context_version_id
            if kind == "decision":
                values["predictive_version_id"] = model.predictive_version_id
                values["economic_version_id"] = model.economic_version_id
            self.connection.execute(
                f"INSERT INTO {table} ({','.join(values)}) "
                f"VALUES ({','.join('?' for _ in values)})",
                tuple(values.values()),
            )
            self._event(
                capability_id,
                event,
                values["id"],
                operation,
                request,
                actor,
                {
                    "sha256": digest,
                    "version_no": values["version_no"],
                    "confirmed_human": confirmed_human,
                    "missing_fields": missing,
                },
            )
            return self._get(kind, str(values["id"]))

    def draft_predictive_from_context(
        self,
        capability_id: str,
        *,
        context_version_id: str,
    ) -> PredictiveSpecV1:
        row = self.repository.get_context(context_version_id)
        self.capabilities.require_scope(capability_id, project_id=str(row["project_id"]))
        payload = json.loads(row["context_json"])
        if canonical_hash(payload) != row["context_sha256"]:
            raise DecisionValueError(
                "SOURCE_CONTEXT_MISMATCH", "La huella del contexto no coincide"
            )
        business = payload.get("context") or {}
        if business.get("task_type", "binary_classification") != "binary_classification":
            raise DecisionValueError(
                "UNSUPPORTED_TASK", "El nuevo recorrido requiere un objetivo binario"
            )
        recipe = snapshot_recipe(self.repository, str(row["dataset_id"]))
        return PredictiveSpecV1(
            target_column=payload["target_column"],
            positive_class=payload["positive_class"],
            prediction_moment=payload["prediction_moment"],
            primary_metric=payload["primary_metric"],
            feature_decisions=business.get("feature_decisions", []),
            source_context_version_id=context_version_id,
            source_context_sha256=str(row["context_sha256"]),
            snapshot_recipe=None if recipe is None else recipe["spec"],
        )

    def bind_predictive_to_cycle(
        self,
        capability_id: str,
        *,
        cycle_id: str,
        predictive_version_id: str,
        actor_label: str,
        operation_id: str,
    ) -> CyclePredictiveBindingV1:
        actor, operation = _label(actor_label, "actor_label"), _label(operation_id, "operation_id")
        request = {
            "capability_id": capability_id,
            "cycle_id": cycle_id,
            "predictive_version_id": predictive_version_id,
            "actor_label": actor,
        }
        with transaction(self.connection):
            replay = self._replay(operation, "PREDICTIVE_SPEC_BOUND", request)
            if replay is not None:
                return self._binding(replay)
            self.capabilities.require_scope(capability_id, cycle_id=cycle_id)
            version = self.get_predictive_version(predictive_version_id)
            if version.capability_id != capability_id:
                raise DecisionValueError("SPEC_SCOPE_MISMATCH", "La versión es de otra capacidad")
            if not version.confirmed_human or version.missing_fields:
                raise DecisionValueError(
                    "SPEC_INCOMPLETE", "Confirma una especificación completa antes de vincular"
                )
            cycle = self.capabilities.get_cycle(cycle_id)
            latest_context = self.repository.get_latest_context(cycle.project_id)
            if latest_context is not None:
                current = self.draft_predictive_from_context(
                    capability_id, context_version_id=str(latest_context["id"])
                )
                fields = (
                    "target_column",
                    "positive_class",
                    "prediction_moment",
                    "primary_metric",
                    "feature_decisions",
                )
                declared = version.spec.model_dump(mode="json", include=set(fields))
                actual = current.model_dump(mode="json", include=set(fields))
                if canonical_json(declared) != canonical_json(actual):
                    raise DecisionValueError(
                        "CYCLE_CONTEXT_MISMATCH",
                        "La especificación no coincide con el contexto vigente de este ciclo",
                    )
            if self.get_cycle_binding(cycle_id) is not None:
                raise DecisionValueError(
                    "CYCLE_ALREADY_BOUND", "El ciclo ya conserva una especificación inmutable"
                )
            binding = CyclePredictiveBindingV1(
                id=str(uuid4()),
                capability_id=capability_id,
                cycle_id=cycle_id,
                predictive_version_id=predictive_version_id,
                actor_label=actor,
                created_at=datetime.now(UTC),
            )
            values = binding.model_dump(mode="json")
            self.connection.execute(
                "INSERT INTO cycle_predictive_bindings "
                f"({','.join(values)}) VALUES ({','.join('?' for _ in values)})",
                tuple(values.values()),
            )
            self._event(
                capability_id,
                "PREDICTIVE_SPEC_BOUND",
                binding.id,
                operation,
                request,
                actor,
                {
                    "cycle_id": cycle_id,
                    "predictive_version_id": predictive_version_id,
                    "sha256": version.sha256,
                },
            )
            return binding

    def _binding(self, binding_id: str) -> CyclePredictiveBindingV1:
        row = self.connection.execute(
            "SELECT * FROM cycle_predictive_bindings WHERE id=?", (binding_id,)
        ).fetchone()
        if row is None:
            raise DecisionValueError("BINDING_NOT_FOUND", "El vínculo no existe")
        return CyclePredictiveBindingV1.model_validate(dict(row))

    def get_cycle_binding(self, cycle_id: str) -> CyclePredictiveBindingV1 | None:
        self.capabilities.get_cycle(cycle_id)
        row = self.connection.execute(
            "SELECT id FROM cycle_predictive_bindings WHERE cycle_id=?", (cycle_id,)
        ).fetchone()
        return None if row is None else self._binding(str(row[0]))

    def _replay(self, operation_id: str, event: str, request: dict[str, Any]) -> str | None:
        row = self.connection.execute(
            "SELECT * FROM value_spec_events WHERE operation_id=?", (operation_id,)
        ).fetchone()
        if row is None:
            return None
        if row["event_type"] != event or row["request_sha256"] != canonical_hash(request):
            raise DecisionValueError("OPERATION_CONFLICT", "La operación ya tiene otra solicitud")
        return str(row["entity_id"])

    def _event(
        self,
        capability_id: str,
        event: str,
        entity_id: str,
        operation: str,
        request: dict[str, Any],
        actor: str,
        detail: dict[str, Any],
    ) -> None:
        self.connection.execute(
            "INSERT INTO value_spec_events (id,capability_id,event_type,entity_id,operation_id,"
            "request_sha256,actor_label,detail_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                str(uuid4()),
                capability_id,
                event,
                entity_id,
                operation,
                canonical_hash(request),
                actor,
                canonical_json(detail),
                datetime.now(UTC).isoformat(),
            ),
        )
