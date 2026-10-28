"""MV-04: capacidades persistentes con ciclos técnicos aislados e idempotentes."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from core.contracts.product import CapabilityCycleV1, CapabilityV1, ProductV1
from core.db.connection import transaction
from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash, canonical_json
from core.domain.errors import DomainError


class CapabilityError(DomainError, ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 200:
        raise CapabilityError("INVALID_OPERATION", f"{name} requiere entre 1 y 200 caracteres")
    return value.strip()


class CapabilityService:
    def __init__(self, repository: WorkflowRepository) -> None:
        self.repository = repository
        self.connection = repository.connection

    def create_product(
        self,
        *,
        name: str,
        description: str = "",
        stage: str = "building",
        owner_label: str | None = None,
        actor_label: str,
        operation_id: str,
    ) -> ProductV1:
        actor = _label(actor_label, "actor_label")
        operation = _label(operation_id, "operation_id")
        product = ProductV1.model_validate(
            {
                "id": str(uuid4()),
                "name": name,
                "description": description,
                "stage": stage,
                "owner_label": owner_label,
                "created_at": datetime.now(UTC),
            }
        )
        request = product.model_dump(mode="json", exclude={"id", "created_at"})
        request["actor_label"] = actor
        with transaction(self.connection):
            replay = self._replay(operation, "PRODUCT_CREATED", request)
            if replay is not None:
                return self.get_product(replay)
            self._insert("products", product)
            self._event(
                product_id=product.id,
                event_type="PRODUCT_CREATED",
                entity_id=product.id,
                operation_id=operation,
                request=request,
                actor_label=actor,
                detail={"owner_declared": product.owner_label is not None},
            )
        return product

    def create_capability(
        self,
        product_id: str,
        *,
        name: str,
        objective: str,
        value_kind: str,
        decision_unit: str,
        actor_label: str,
        operation_id: str,
    ) -> CapabilityV1:
        actor = _label(actor_label, "actor_label")
        operation = _label(operation_id, "operation_id")
        capability = CapabilityV1.model_validate(
            {
                "id": str(uuid4()),
                "product_id": product_id,
                "name": name,
                "objective": objective,
                "value_kind": value_kind,
                "decision_unit": decision_unit,
                "created_at": datetime.now(UTC),
            }
        )
        request = capability.model_dump(mode="json", exclude={"id", "created_at"})
        request["actor_label"] = actor
        with transaction(self.connection):
            replay = self._replay(operation, "CAPABILITY_CREATED", request)
            if replay is not None:
                return self.get_capability(replay)
            product = self.get_product(capability.product_id)
            self._require_active(product.status)
            self._insert("capabilities", capability)
            self._event(
                product_id=product.id,
                capability_id=capability.id,
                event_type="CAPABILITY_CREATED",
                entity_id=capability.id,
                operation_id=operation,
                request=request,
                actor_label=actor,
                detail={"value_kind": capability.value_kind},
            )
        return capability

    def create_cycle(
        self,
        capability_id: str,
        *,
        reason: str,
        actor_label: str,
        operation_id: str,
        source_cycle_id: str | None = None,
    ) -> CapabilityCycleV1:
        actor = _label(actor_label, "actor_label")
        operation = _label(operation_id, "operation_id")
        # Valida entradas antes de producir efectos; los IDs y el número finales
        # se asignan dentro de la transacción que también crea el proyecto.
        draft = CapabilityCycleV1(
            id=str(uuid4()),
            capability_id=capability_id,
            project_id="pending",
            cycle_no=1,
            origin="initial" if source_cycle_id is None else "update",
            source_cycle_id=source_cycle_id,
            reason=reason,
            created_at=datetime.now(UTC),
        )
        request = {
            "capability_id": draft.capability_id,
            "reason": draft.reason,
            "source_cycle_id": draft.source_cycle_id,
            "actor_label": actor,
        }
        with transaction(self.connection):
            replay = self._replay(operation, "CAPABILITY_CYCLE_CREATED", request)
            if replay is not None:
                return self.get_cycle(replay)
            capability = self.get_capability(draft.capability_id)
            product = self.get_product(capability.product_id)
            self._require_active(product.status)
            self._require_active(capability.status)
            cycle_no = self.connection.execute(
                "SELECT COALESCE(MAX(cycle_no), 0) + 1 FROM capability_cycles "
                "WHERE capability_id = ?",
                (capability.id,),
            ).fetchone()[0]
            if draft.source_cycle_id is not None:
                self.require_scope(capability.id, cycle_id=draft.source_cycle_id)
            elif cycle_no > 1:
                raise CapabilityError(
                    "CYCLE_SOURCE_REQUIRED", "Elige el ciclo de origen para la actualización"
                )
            project_id = self.repository._insert_project(f"{capability.name} · ciclo {cycle_no}")
            cycle = CapabilityCycleV1.model_validate(
                {**draft.model_dump(), "project_id": project_id, "cycle_no": cycle_no}
            )
            self._insert("capability_cycles", cycle)
            self.repository._audit(
                project_id=project_id,
                event_type="CAPABILITY_CYCLE_CREATED",
                entity_type="CAPABILITY_CYCLE",
                entity_id=cycle.id,
                detail={
                    "product_id": product.id,
                    "capability_id": capability.id,
                    "cycle_no": cycle.cycle_no,
                    "source_cycle_id": cycle.source_cycle_id,
                    "actor_label": actor,
                    "identity_verified": False,
                },
            )
            self._event(
                product_id=product.id,
                capability_id=capability.id,
                cycle_id=cycle.id,
                event_type="CAPABILITY_CYCLE_CREATED",
                entity_id=cycle.id,
                operation_id=operation,
                request=request,
                actor_label=actor,
                detail={
                    "project_id": project_id,
                    "cycle_no": cycle.cycle_no,
                    "source_cycle_id": cycle.source_cycle_id,
                },
            )
        return cycle

    def adopt_update_project(
        self,
        capability_id: str,
        *,
        source_cycle_id: str,
        project_id: str,
        reason: str,
        actor_label: str,
        operation_id: str,
    ) -> CapabilityCycleV1:
        """Vincula una candidata ya preparada, sin sustituir el ciclo de origen."""

        actor = _label(actor_label, "actor_label")
        operation = _label(operation_id, "operation_id")
        explanation = _label(reason, "reason")
        request = {
            "capability_id": capability_id,
            "source_cycle_id": source_cycle_id,
            "project_id": project_id,
            "reason": explanation,
            "actor_label": actor,
        }
        with transaction(self.connection):
            replay = self._replay(operation, "CAPABILITY_CYCLE_CREATED", request)
            if replay is not None:
                return self.get_cycle(replay)
            capability = self.get_capability(capability_id)
            product = self.get_product(capability.product_id)
            self._require_active(product.status)
            self._require_active(capability.status)
            source = self.get_cycle(source_cycle_id)
            if source.capability_id != capability_id:
                raise CapabilityError("CYCLE_SCOPE_MISMATCH", "El origen es de otra capacidad")
            row = self.connection.execute(
                "SELECT status FROM projects WHERE id=?", (project_id,)
            ).fetchone()
            if row is None or row["status"] == "ARCHIVED":
                raise CapabilityError("CANDIDATE_PROJECT_NOT_ACTIVE", "La candidata no está activa")
            source_links = [
                link
                for link in self.repository.list_retraining_links()
                if link["project_id"] == project_id
            ]
            if len(source_links) != 1 or source_links[0]["source_project_id"] != source.project_id:
                raise CapabilityError(
                    "CANDIDATE_PROVENANCE_MISMATCH",
                    "El proyecto no proviene de este ciclo",
                )
            if self.get_cycle_for_project(project_id) is not None:
                raise CapabilityError(
                    "PROJECT_ALREADY_BOUND", "El proyecto ya pertenece a un ciclo"
                )
            cycle_no = self.connection.execute(
                "SELECT COALESCE(MAX(cycle_no),0)+1 FROM capability_cycles WHERE capability_id=?",
                (capability_id,),
            ).fetchone()[0]
            cycle = CapabilityCycleV1(
                id=str(uuid4()),
                capability_id=capability_id,
                project_id=project_id,
                cycle_no=cycle_no,
                origin="update",
                source_cycle_id=source.id,
                reason=explanation,
                created_at=datetime.now(UTC),
            )
            self._insert("capability_cycles", cycle)
            self.repository._audit(
                project_id=project_id,
                event_type="CAPABILITY_UPDATE_ADOPTED",
                entity_type="CAPABILITY_CYCLE",
                entity_id=cycle.id,
                detail={
                    "capability_id": capability_id,
                    "source_cycle_id": source.id,
                    "actor_label": actor,
                },
            )
            self._event(
                product_id=product.id,
                capability_id=capability_id,
                cycle_id=cycle.id,
                event_type="CAPABILITY_CYCLE_CREATED",
                entity_id=cycle.id,
                operation_id=operation,
                request=request,
                actor_label=actor,
                detail={
                    "project_id": project_id,
                    "source_cycle_id": source.id,
                    "prepared_candidate_adopted": True,
                },
            )
        return cycle

    def get_product(self, product_id: str) -> ProductV1:
        return ProductV1.model_validate(dict(self._get("products", product_id)))

    def get_capability(self, capability_id: str) -> CapabilityV1:
        return CapabilityV1.model_validate(dict(self._get("capabilities", capability_id)))

    def get_cycle(self, cycle_id: str) -> CapabilityCycleV1:
        return CapabilityCycleV1.model_validate(dict(self._get("capability_cycles", cycle_id)))

    def get_cycle_for_project(self, project_id: str) -> CapabilityCycleV1 | None:
        row = self.connection.execute(
            "SELECT * FROM capability_cycles WHERE project_id = ?", (project_id,)
        ).fetchone()
        return None if row is None else CapabilityCycleV1.model_validate(dict(row))

    def list_products(self, *, include_archived: bool = True) -> list[ProductV1]:
        rows = self.connection.execute(
            "SELECT * FROM products WHERE (? OR status = 'active') ORDER BY created_at, rowid",
            (include_archived,),
        ).fetchall()
        return [ProductV1.model_validate(dict(row)) for row in rows]

    def list_capabilities(
        self, product_id: str, *, include_archived: bool = True
    ) -> list[CapabilityV1]:
        self.get_product(product_id)
        rows = self.connection.execute(
            "SELECT * FROM capabilities WHERE product_id = ? AND (? OR status = 'active') "
            "ORDER BY created_at, rowid",
            (product_id, include_archived),
        ).fetchall()
        return [CapabilityV1.model_validate(dict(row)) for row in rows]

    def list_cycles(self, capability_id: str) -> list[CapabilityCycleV1]:
        self.get_capability(capability_id)
        rows = self.connection.execute(
            "SELECT * FROM capability_cycles WHERE capability_id = ? ORDER BY cycle_no",
            (capability_id,),
        ).fetchall()
        return [CapabilityCycleV1.model_validate(dict(row)) for row in rows]

    def require_scope(
        self,
        capability_id: str,
        *,
        cycle_id: str | None = None,
        project_id: str | None = None,
        run_id: str | None = None,
        dataset_id: str | None = None,
    ) -> None:
        """Todas las referencias dadas deben pertenecer al mismo ciclo y capacidad."""

        self.get_capability(capability_id)
        projects: list[str] = []
        if cycle_id is not None:
            projects.append(self.get_cycle(cycle_id).project_id)
        if project_id is not None:
            projects.append(project_id)
        if run_id is not None:
            row = self.connection.execute(
                "SELECT context.project_id FROM runs AS run "
                "JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id "
                "JOIN context_versions AS context ON context.id = strategy.context_version_id "
                "WHERE run.id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise CapabilityError("RUN_NOT_FOUND", "La ejecución no existe")
            projects.append(str(row["project_id"]))
        if dataset_id is not None:
            row = self.connection.execute(
                "SELECT project_id FROM datasets WHERE id = ?", (dataset_id,)
            ).fetchone()
            if row is None:
                raise CapabilityError("DATASET_NOT_FOUND", "El dataset no existe")
            projects.append(str(row["project_id"]))
        if len(set(projects)) > 1:
            raise CapabilityError(
                "CAPABILITY_SCOPE_MISMATCH", "Las referencias pertenecen a ciclos distintos"
            )
        for project in projects:
            cycle = self.get_cycle_for_project(project)
            if cycle is None or cycle.capability_id != capability_id:
                raise CapabilityError(
                    "CAPABILITY_SCOPE_MISMATCH", "La referencia no pertenece a esta capacidad"
                )

    def _get(self, table: str, entity_id: str) -> sqlite3.Row:
        codes = {
            "products": ("PRODUCT_NOT_FOUND", "El producto no existe"),
            "capabilities": ("CAPABILITY_NOT_FOUND", "La capacidad no existe"),
            "capability_cycles": ("CYCLE_NOT_FOUND", "El ciclo no existe"),
        }
        if table not in codes:
            raise ValueError("Tabla de producto no permitida")
        row = self.connection.execute(
            f"SELECT * FROM {table} WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            raise CapabilityError(*codes[table])
        return row

    def _insert(self, table: str, model: ProductV1 | CapabilityV1 | CapabilityCycleV1) -> None:
        if table not in {"products", "capabilities", "capability_cycles"}:
            raise ValueError("Tabla de producto no permitida")
        values = model.model_dump(mode="json", exclude={"schema_version"})
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        self.connection.execute(
            f"INSERT INTO {table}({columns}) VALUES ({placeholders})", tuple(values.values())
        )

    @staticmethod
    def _require_active(status: str) -> None:
        if status != "active":
            raise CapabilityError("ENTITY_ARCHIVED", "La entidad está archivada")

    def _replay(self, operation_id: str, event_type: str, request: dict[str, Any]) -> str | None:
        row = self.connection.execute(
            "SELECT event_type, request_sha256, entity_id FROM product_events "
            "WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        if row["event_type"] != event_type or row["request_sha256"] != canonical_hash(request):
            raise CapabilityError(
                "OPERATION_CONFLICT", "La operación ya fue usada con una solicitud diferente"
            )
        return str(row["entity_id"])

    def _event(
        self,
        *,
        product_id: str,
        event_type: str,
        entity_id: str,
        operation_id: str,
        request: dict[str, Any],
        actor_label: str,
        detail: dict[str, Any],
        capability_id: str | None = None,
        cycle_id: str | None = None,
    ) -> None:
        self.connection.execute(
            "INSERT INTO product_events(id, product_id, capability_id, cycle_id, event_type, "
            "entity_id, operation_id, request_sha256, actor_label, detail_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                product_id,
                capability_id,
                cycle_id,
                event_type,
                entity_id,
                operation_id,
                canonical_hash(request),
                actor_label,
                canonical_json({**detail, "identity_verified": False}),
                datetime.now(UTC).isoformat(),
            ),
        )
