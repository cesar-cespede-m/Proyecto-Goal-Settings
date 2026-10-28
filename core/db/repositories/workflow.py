"""Repositorio mínimo del workflow persistido de Goal Settings."""

import json
import sqlite3
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from core.db.connection import transaction
from core.domain.canonical import (
    canonical_hash,
    canonical_json,
    report_fingerprint,
    strategy_fingerprint,
)
from core.domain.errors import ApprovalRequiredError, InvalidStateTransitionError
from core.domain.gates import (
    is_report_current,
    require_publication_approval,
    require_training_approval,
)
from core.domain.models import (
    ContextPayload,
    DatasetProfile,
    JsonObject,
    ReportPayload,
    RunManifest,
    StrategyPayload,
)
from core.domain.states import ApprovalDecision, ReportStatus, RunStatus

_SENSITIVE_AUDIT_KEYS = {"api_key", "csv_rows", "password", "raw_data", "secret", "token"}


def _new_id() -> str:
    return str(uuid4())


def _validate_audit_detail(detail: dict[str, Any]) -> str:
    if {key.lower() for key in detail} & _SENSITIVE_AUDIT_KEYS:
        raise ValueError("audit detail contiene una clave sensible o datos crudos")
    return canonical_json(JsonObject(value=detail).value)


class WorkflowRepository:
    """Escribe cambios de estado y su auditoría en la misma transacción."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def _audit(
        self,
        *,
        project_id: str,
        event_type: str,
        entity_type: str,
        entity_id: str,
        detail: dict[str, Any],
        event_id: str | None = None,
    ) -> str:
        audit_id = event_id or _new_id()
        self.connection.execute(
            """
            INSERT INTO audit_events(
                id, project_id, event_type, entity_type, entity_id, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                project_id,
                event_type,
                entity_type,
                entity_id,
                _validate_audit_detail(detail),
            ),
        )
        return audit_id

    def create_project(self, name: str, *, status: str = "CREATED") -> str:
        with transaction(self.connection):
            return self._insert_project(name, status=status)

    def _insert_project(self, name: str, *, status: str = "CREATED") -> str:
        """Inserta proyecto y auditoría dentro de una transacción del llamador."""

        if not self.connection.in_transaction:
            raise RuntimeError("La inserción de proyecto requiere una transacción activa")
        project_id = _new_id()
        self.connection.execute(
            "INSERT INTO projects(id, name, status) VALUES (?, ?, ?)",
            (project_id, name, status),
        )
        self._audit(
            project_id=project_id,
            event_type="PROJECT_CREATED",
            entity_type="PROJECT",
            entity_id=project_id,
            detail={"status": status},
        )
        return project_id

    def set_project_roles(
        self,
        project_id: str,
        *,
        owner_label: str,
        approver_label: str,
    ) -> int:
        """Declara una nueva versión; las etiquetas no prueban identidad."""

        owner = owner_label.strip()
        approver = approver_label.strip()
        if not owner or len(owner) > 200:
            raise ValueError("owner_label debe tener entre 1 y 200 caracteres")
        if not approver or len(approver) > 200:
            raise ValueError("approver_label debe tener entre 1 y 200 caracteres")
        with transaction(self.connection):
            current = self.connection.execute(
                "SELECT COALESCE(MAX(version_no), 0) FROM project_roles WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
            version_no = int(current) + 1
            self.connection.execute(
                """
                INSERT INTO project_roles(
                    id, project_id, version_no, owner_label, approver_label
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (_new_id(), project_id, version_no, owner, approver),
            )
            self._audit(
                project_id=project_id,
                event_type="PROJECT_ROLES_DECLARED",
                entity_type="PROJECT",
                entity_id=project_id,
                detail={
                    "owner_label": owner,
                    "approver_label": approver,
                    "version_no": version_no,
                    "identity_verified": False,
                },
            )
        return version_no

    def get_project_roles(self, project_id: str) -> sqlite3.Row | None:
        """Devuelve la versión vigente de las etiquetas declaradas del proyecto."""

        return self.connection.execute(
            """
            SELECT * FROM project_roles
            WHERE project_id = ?
            ORDER BY version_no DESC, rowid DESC
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()

    def add_dataset(
        self,
        *,
        project_id: str,
        dataset_id: str | None = None,
        sha256: str,
        original_filename_display: str,
        source: str,
        license_name: str | None,
        anonymization_declared: bool,
        admission_status: str,
        rejection_reason: str | None,
        n_rows: int | None,
        n_cols: int | None,
        byte_count: int,
        profile: DatasetProfile | dict[str, Any],
        dataset_kind: str = "SOURCE",
        make_active: bool | None = None,
    ) -> str:
        dataset_id = dataset_id or _new_id()
        storage_area = "accepted" if admission_status == "ACCEPTED" else "quarantine"
        stored_path = str(PurePosixPath("data", storage_area, dataset_id))
        profile_model = DatasetProfile.model_validate(profile)
        with transaction(self.connection):
            if make_active is None:
                make_active = admission_status == "ACCEPTED" and not bool(
                    self.connection.execute(
                        """
                        SELECT 1 FROM datasets
                        WHERE project_id = ? AND admission_status = 'ACCEPTED' AND is_active = 1
                        LIMIT 1
                        """,
                        (project_id,),
                    ).fetchone()
                )
            self.connection.execute(
                """
                INSERT INTO datasets(
                    id, project_id, sha256, original_filename_display, stored_path,
                    source, license, anonymization_declared, admission_status,
                    rejection_reason, n_rows, n_cols, bytes, profile_json,
                    dataset_kind, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    project_id,
                    sha256,
                    original_filename_display,
                    stored_path,
                    source,
                    license_name,
                    int(anonymization_declared),
                    admission_status,
                    rejection_reason,
                    n_rows,
                    n_cols,
                    byte_count,
                    canonical_json(profile_model),
                    dataset_kind,
                    int(bool(make_active)),
                ),
            )
            self._audit(
                project_id=project_id,
                event_type="DATASET_REGISTERED",
                entity_type="DATASET",
                entity_id=dataset_id,
                detail={"admission_status": admission_status, "sha256": sha256},
            )
        return dataset_id

    def get_dataset(self, dataset_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM datasets WHERE id = ?",
            (dataset_id,),
        ).fetchone()
        if row is None:
            raise ValueError("dataset inexistente")
        return row

    def list_projects(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM projects ORDER BY created_at DESC, rowid DESC"
        ).fetchall()

    def list_active_projects(self) -> list[sqlite3.Row]:
        """Proyectos visibles en la interfaz: los archivados quedan fuera de la lista."""

        return self.connection.execute(
            """
            SELECT * FROM projects WHERE status <> 'ARCHIVED'
            ORDER BY created_at DESC, rowid DESC
            """
        ).fetchall()

    def archive_project(self, project_id: str, *, actor_label: str) -> None:
        """«Eliminar» desde la interfaz: oculta el proyecto sin borrar su trazabilidad.

        Aprobaciones y auditoría son inmutables por diseño (HITL), así que el proyecto no se
        borra: pasa a ARCHIVED y queda un evento PROJECT_ARCHIVED con quién lo hizo.
        """

        if not actor_label or not actor_label.strip():
            raise ValueError("Indica quién elimina el proyecto")
        with transaction(self.connection):
            updated = self.connection.execute(
                """
                UPDATE projects SET status = 'ARCHIVED', updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status <> 'ARCHIVED'
                """,
                (project_id,),
            ).rowcount
            if updated != 1:
                raise LookupError("El proyecto no existe o ya fue eliminado")
            self._audit(
                project_id=project_id,
                event_type="PROJECT_ARCHIVED",
                entity_type="PROJECT",
                entity_id=project_id,
                detail={"actor_label": actor_label.strip()},
            )

    def list_datasets(
        self,
        project_id: str,
        *,
        admission_status: str | None = None,
        dataset_kind: str | None = None,
        active_only: bool = False,
    ) -> list[sqlite3.Row]:
        clauses = ["project_id = ?"]
        parameters: list[Any] = [project_id]
        if admission_status is not None:
            clauses.append("admission_status = ?")
            parameters.append(admission_status)
        if dataset_kind is not None:
            clauses.append("dataset_kind = ?")
            parameters.append(dataset_kind)
        if active_only:
            clauses.append("is_active = 1")
        return self.connection.execute(
            f"""SELECT * FROM datasets WHERE {" AND ".join(clauses)}
            ORDER BY created_at DESC, rowid DESC""",
            parameters,
        ).fetchall()

    def get_active_dataset(self, project_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM datasets
            WHERE project_id = ? AND admission_status = 'ACCEPTED' AND is_active = 1
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()

    def append_observation_events(
        self,
        *,
        project_id: str,
        events: list[dict[str, Any]],
        rejected_count: int,
        source_label: str,
        actor_label: str,
    ) -> tuple[int, int]:
        """Inserta eventos nuevos y audita sólo etiquetas y conteos, nunca sus valores."""

        accepted = 0
        duplicates = 0
        with transaction(self.connection):
            for event in events:
                cursor = self.connection.execute(
                    """
                    INSERT INTO observation_events(
                        id, project_id, source, entity, entity_id, occurred_at,
                        attributes_json, outcome_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, id) DO NOTHING
                    """,
                    (
                        event["id"],
                        project_id,
                        event["source"],
                        event["entity"],
                        event["entity_id"],
                        event["occurred_at"],
                        event["attributes_json"],
                        event["outcome_json"],
                    ),
                )
                if cursor.rowcount == 1:
                    accepted += 1
                else:
                    duplicates += 1
            self._audit(
                project_id=project_id,
                event_type="OBSERVATION_EVENTS_INGESTED",
                entity_type="OBSERVATION_EVENTS",
                entity_id=project_id,
                detail={
                    "accepted": accepted,
                    "duplicates": duplicates,
                    "rejected": rejected_count,
                    "source_label": source_label,
                    "actor_label": actor_label,
                },
            )
        return accepted, duplicates

    def list_observation_events(self, project_id: str, *, entity: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT * FROM observation_events
            WHERE project_id = ? AND entity = ?
            ORDER BY occurred_at, rowid
            """,
            (project_id, entity),
        ).fetchall()

    def record_observation_materialization(
        self,
        *,
        project_id: str,
        dataset_id: str,
        entity: str,
        event_count: int,
        classification: str,
        actor_label: str,
    ) -> str:
        with transaction(self.connection):
            return self._audit(
                project_id=project_id,
                event_type="OBSERVATION_EVENTS_MATERIALIZED",
                entity_type="DATASET",
                entity_id=dataset_id,
                detail={
                    "entity": entity,
                    "event_count": event_count,
                    "classification": classification,
                    "actor_label": actor_label,
                },
            )

    def record_measurement_refresh(
        self,
        *,
        project_id: str,
        actor_label: str,
        classification: str,
        created_files: int,
        unchanged_entities: int,
        measured_models: int,
        warnings: int,
    ) -> str:
        """Audita C2-06 sólo con clasificación y conteos, nunca con valores de filas."""

        with transaction(self.connection):
            return self._audit(
                project_id=project_id,
                event_type="MEASUREMENT_REFRESHED",
                entity_type="PROJECT",
                entity_id=project_id,
                detail={
                    "actor_label": actor_label,
                    "classification": classification,
                    "created_files": created_files,
                    "unchanged_entities": unchanged_entities,
                    "measured_models": measured_models,
                    "warnings": warnings,
                },
            )

    def set_modeling_table(self, project_id: str, dataset_id: str, *, actor_label: str) -> None:
        """Elige qué archivo aceptado del catálogo se usa para entrenar (GA-01).

        Sólo puede haber una tabla de modelado activa por proyecto. Cambiarla no borra nada:
        las decisiones y modelos de la tabla anterior se conservan con su linaje.
        """

        if not actor_label or not actor_label.strip():
            raise ValueError("Indica quién elige la tabla para entrenar")
        with transaction(self.connection):
            row = self.connection.execute(
                "SELECT project_id, admission_status FROM datasets WHERE id = ?",
                (dataset_id,),
            ).fetchone()
            if row is None or row["project_id"] != project_id:
                raise LookupError("El archivo no pertenece a este proyecto")
            if row["admission_status"] != "ACCEPTED":
                raise ValueError("Sólo un archivo aceptado puede usarse para entrenar")
            self.connection.execute(
                "UPDATE datasets SET is_active = 0 WHERE project_id = ? AND is_active = 1",
                (project_id,),
            )
            self.connection.execute(
                "UPDATE datasets SET is_active = 1 WHERE id = ?",
                (dataset_id,),
            )
            self._audit(
                project_id=project_id,
                event_type="MODELING_TABLE_SELECTED",
                entity_type="DATASET",
                entity_id=dataset_id,
                detail={"actor_label": actor_label.strip()},
            )

    def persist_dataset_composition(
        self,
        *,
        composition_id: str,
        project_id: str,
        output_dataset_id: str,
        output_sha256: str,
        output_bytes: int,
        output_rows: int,
        output_columns: int,
        output_anonymization_declared: bool,
        output_profile: DatasetProfile | dict[str, Any],
        spec: dict[str, Any],
        summary: dict[str, Any],
        source_entries: list[dict[str, Any]],
        actor_label: str,
    ) -> None:
        profile_model = DatasetProfile.model_validate(output_profile)
        spec_json = canonical_json(JsonObject(value=spec).value)
        summary_json = canonical_json(JsonObject(value=summary).value)
        spec_sha256 = canonical_hash(spec)
        if summary.get("composition_sha256") != spec_sha256:
            raise ValueError("composition hash mismatch")
        if summary.get("output_sha256") != output_sha256:
            raise ValueError("composition output hash mismatch")
        if not actor_label.strip():
            raise ValueError("composition actor required")

        with transaction(self.connection):
            for entry in source_entries:
                source = self.connection.execute(
                    """
                    SELECT project_id, sha256, admission_status, dataset_kind
                    FROM datasets WHERE id = ?
                    """,
                    (entry["dataset_id"],),
                ).fetchone()
                if (
                    source is None
                    or source["project_id"] != project_id
                    or source["admission_status"] != "ACCEPTED"
                    or source["dataset_kind"] != "SOURCE"
                    or source["sha256"] != entry["source_sha256"]
                ):
                    raise ValueError("composition source is not an accepted project source")

            stored_path = str(PurePosixPath("data", "accepted", output_dataset_id))
            self.connection.execute(
                """
                INSERT INTO datasets(
                    id, project_id, sha256, original_filename_display, stored_path,
                    source, license, anonymization_declared, admission_status,
                    rejection_reason, n_rows, n_cols, bytes, profile_json,
                    dataset_kind, is_active
                ) VALUES (?, ?, ?, 'composicion-goal_settings.csv', ?,
                          'Goal Settings dataset composition', NULL, ?, 'ACCEPTED',
                          NULL, ?, ?, ?, ?, 'DERIVED', 0)
                """,
                (
                    output_dataset_id,
                    project_id,
                    output_sha256,
                    stored_path,
                    int(output_anonymization_declared),
                    output_rows,
                    output_columns,
                    output_bytes,
                    canonical_json(profile_model),
                ),
            )
            self.connection.execute(
                """
                INSERT INTO dataset_compositions(
                    id, project_id, output_dataset_id, spec_json, spec_sha256,
                    summary_json, actor_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    composition_id,
                    project_id,
                    output_dataset_id,
                    spec_json,
                    spec_sha256,
                    summary_json,
                    actor_label.strip(),
                ),
            )
            for entry in source_entries:
                self.connection.execute(
                    """
                    INSERT INTO dataset_composition_sources(
                        composition_id, position, dataset_id, role, alias, source_sha256,
                        primary_key, auxiliary_key, cardinality
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        composition_id,
                        entry["position"],
                        entry["dataset_id"],
                        entry["role"],
                        entry.get("alias"),
                        entry["source_sha256"],
                        entry.get("primary_key"),
                        entry.get("auxiliary_key"),
                        entry.get("cardinality"),
                    ),
                )
            self.connection.execute(
                "UPDATE datasets SET is_active = 0 WHERE project_id = ? AND is_active = 1",
                (project_id,),
            )
            self.connection.execute(
                "UPDATE datasets SET is_active = 1 WHERE id = ?",
                (output_dataset_id,),
            )
            self._audit(
                project_id=project_id,
                event_type="DATASET_COMPOSITION_CREATED",
                entity_type="DATASET_COMPOSITION",
                entity_id=composition_id,
                detail={
                    "actor_label": actor_label.strip(),
                    "composition_sha256": spec_sha256,
                    "output_dataset_id": output_dataset_id,
                    "output_sha256": output_sha256,
                    "source_count": len(source_entries),
                },
            )

    def persist_analysis_table(
        self,
        *,
        table_id: str,
        project_id: str,
        output_dataset_id: str,
        display_name: str,
        output_sha256: str,
        output_bytes: int,
        output_rows: int,
        output_columns: int,
        output_anonymization_declared: bool,
        output_profile: DatasetProfile | dict[str, Any],
        plan: dict[str, Any],
        plan_sha256: str,
        sources: list[dict[str, Any]],
        engine_version: str,
        warnings: list[str],
        actor_label: str,
    ) -> None:
        """Registra el resultado de un análisis como tabla derivada con su receta (GA-06).

        La tabla queda disponible para analizar; usarla para entrenar es otra decisión humana
        (`set_modeling_table`) seguida de la aprobación de entrenamiento de siempre.
        """

        profile_model = DatasetProfile.model_validate(output_profile)
        if not actor_label.strip():
            raise ValueError("Indica quién guarda la tabla")
        if not display_name.strip():
            raise ValueError("La tabla necesita un nombre")
        with transaction(self.connection):
            for entry in sources:
                source = self.connection.execute(
                    "SELECT project_id, sha256, admission_status FROM datasets WHERE id = ?",
                    (entry["dataset_id"],),
                ).fetchone()
                if (
                    source is None
                    or source["project_id"] != project_id
                    or source["admission_status"] != "ACCEPTED"
                    or source["sha256"] != entry["sha256"]
                ):
                    raise ValueError("analysis table source is not an accepted project file")
            self.connection.execute(
                """
                INSERT INTO datasets(
                    id, project_id, sha256, original_filename_display, stored_path,
                    source, license, anonymization_declared, admission_status,
                    rejection_reason, n_rows, n_cols, bytes, profile_json,
                    dataset_kind, is_active
                ) VALUES (?, ?, ?, ?, ?, 'Goal Settings analysis table', NULL, ?, 'ACCEPTED',
                          NULL, ?, ?, ?, ?, 'DERIVED', 0)
                """,
                (
                    output_dataset_id,
                    project_id,
                    output_sha256,
                    f"{display_name.strip()}.csv",
                    str(PurePosixPath("data", "accepted", output_dataset_id)),
                    int(output_anonymization_declared),
                    output_rows,
                    output_columns,
                    output_bytes,
                    canonical_json(profile_model),
                ),
            )
            self.connection.execute(
                """
                INSERT INTO analysis_tables(
                    id, project_id, output_dataset_id, display_name, plan_json, plan_sha256,
                    sources_json, engine_version, warnings_json, actor_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    table_id,
                    project_id,
                    output_dataset_id,
                    display_name.strip(),
                    canonical_json(JsonObject(value=plan).value),
                    plan_sha256,
                    json.dumps(sources, sort_keys=True, ensure_ascii=False),
                    engine_version,
                    json.dumps(warnings, ensure_ascii=False),
                    actor_label.strip(),
                ),
            )
            self._audit(
                project_id=project_id,
                event_type="ANALYSIS_TABLE_SAVED",
                entity_type="DATASET",
                entity_id=output_dataset_id,
                detail={
                    "actor_label": actor_label.strip(),
                    "plan_sha256": plan_sha256,
                    "output_sha256": output_sha256,
                    "source_count": len(sources),
                },
            )

    def get_analysis_table(self, output_dataset_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM analysis_tables WHERE output_dataset_id = ?",
            (output_dataset_id,),
        ).fetchone()

    def get_dataset_composition(self, output_dataset_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM dataset_compositions WHERE output_dataset_id = ?",
            (output_dataset_id,),
        ).fetchone()

    def get_context(self, context_version_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM context_versions WHERE id = ?",
            (context_version_id,),
        ).fetchone()
        if row is None:
            raise ValueError("context_version inexistente")
        return row

    def get_latest_context(self, project_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT context.* FROM context_versions AS context
            JOIN datasets AS dataset ON dataset.id = context.dataset_id
            WHERE context.project_id = ? AND dataset.is_active = 1
            ORDER BY context.version_no DESC
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()

    def list_reusable_contexts(self, exclude_project_id: str) -> list[sqlite3.Row]:
        """GA-13: la última versión de contexto de cada otro proyecto visible."""

        return self.connection.execute(
            """
            SELECT context.id, context.project_id, context.version_no, context.target_column,
                   context.context_json, context.created_at, project.name AS project_name
            FROM context_versions AS context
            JOIN projects AS project ON project.id = context.project_id
            WHERE context.project_id <> ? AND project.status <> 'ARCHIVED'
              AND context.version_no = (
                  SELECT MAX(other.version_no) FROM context_versions AS other
                  WHERE other.project_id = context.project_id
              )
            ORDER BY context.created_at DESC, context.rowid DESC
            """,
            (exclude_project_id,),
        ).fetchall()

    def record_context_reused(
        self, *, project_id: str, source_context_id: str, actor_label: str, proposed: int
    ) -> str:
        with transaction(self.connection):
            return self._audit(
                project_id=project_id,
                event_type="CONTEXT_REUSED",
                entity_type="PROJECT",
                entity_id=project_id,
                detail={
                    "source_context_id": source_context_id,
                    "proposed": proposed,
                    "actor_label": actor_label,
                },
            )

    def add_context(
        self,
        *,
        project_id: str,
        dataset_id: str,
        version_no: int | None,
        payload: ContextPayload | dict[str, Any],
    ) -> str:
        context_id = _new_id()
        model = ContextPayload.model_validate(payload)
        context_json = canonical_json(model)
        context_sha256 = canonical_hash(model)
        with transaction(self.connection):
            if version_no is None:
                version_no = int(
                    self.connection.execute(
                        """
                        SELECT COALESCE(MAX(version_no), 0) + 1
                        FROM context_versions
                        WHERE project_id = ?
                        """,
                        (project_id,),
                    ).fetchone()[0]
                )
            self.connection.execute(
                """
                INSERT INTO context_versions(
                    id, project_id, dataset_id, version_no, target_column,
                    positive_class, prediction_moment, excluded_columns_json,
                    primary_metric, guardrails_json, context_json, context_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    context_id,
                    project_id,
                    dataset_id,
                    version_no,
                    model.target_column,
                    canonical_json(model.positive_class),
                    model.prediction_moment,
                    canonical_json(model.excluded_columns),
                    model.primary_metric,
                    None if model.guardrails is None else canonical_json(model.guardrails),
                    context_json,
                    context_sha256,
                ),
            )
            self._audit(
                project_id=project_id,
                event_type="CONTEXT_VERSION_CREATED",
                entity_type="CONTEXT_VERSION",
                entity_id=context_id,
                detail={"context_sha256": context_sha256, "version_no": version_no},
            )
        return context_id

    def add_strategy(
        self,
        *,
        context_version_id: str,
        dataset_id: str,
        version_no: int | None,
        payload: StrategyPayload | dict[str, Any],
        schema_version: str,
        source: str,
        operation_id: str | None = None,
        actor_label: str | None = None,
        verify_current: Callable[[], None] | None = None,
    ) -> str:
        strategy_id = _new_id()
        model = StrategyPayload.model_validate(payload)
        strategy_json = canonical_json(model)
        parent = self.connection.execute(
            """
            SELECT context.project_id, context.context_sha256, dataset.sha256
            FROM context_versions AS context
            JOIN datasets AS dataset ON dataset.id = ?
            WHERE context.id = ?
            """,
            (dataset_id, context_version_id),
        ).fetchone()
        if parent is None:
            raise ValueError("dataset o contexto inexistente")
        fingerprint = strategy_fingerprint(
            parent["sha256"],
            parent["context_sha256"],
            strategy_json,
            schema_version,
        )
        operation_detail: dict[str, Any] = {}
        if operation_id is not None:
            if not isinstance(operation_id, str) or not 1 <= len(operation_id.strip()) <= 200:
                raise ValueError("Indica una operación de estrategia válida")
            if not isinstance(actor_label, str) or not 1 <= len(actor_label.strip()) <= 200:
                raise ValueError("Indica quién solicita la estrategia")
            operation_id, actor_label = operation_id.strip(), actor_label.strip()
            operation_detail = {
                "operation_id": operation_id,
                "actor_label": actor_label,
                "request_sha256": canonical_hash(
                    {
                        "context_version_id": context_version_id,
                        "dataset_id": dataset_id,
                        "version_no": version_no,
                        "strategy_json": strategy_json,
                        "schema_version": schema_version,
                        "source": source,
                        "actor_label": actor_label,
                        "fingerprint": fingerprint,
                    }
                ),
            }
        with transaction(self.connection):
            if operation_id is not None:
                previous = self.connection.execute(
                    "SELECT entity_id, detail_json FROM audit_events "
                    "WHERE event_type='STRATEGY_VERSION_CREATED' "
                    "AND json_extract(detail_json, '$.operation_id')=?",
                    (operation_id,),
                ).fetchone()
                if previous is not None:
                    detail = json.loads(previous["detail_json"])
                    if detail.get("request_sha256") != operation_detail["request_sha256"]:
                        raise InvalidStateTransitionError("STRATEGY_OPERATION_CONFLICT")
                    return str(previous["entity_id"])
                current = self.get_latest_context(str(parent["project_id"]))
                dataset = self.get_dataset(dataset_id)
                if (
                    current is None
                    or current["id"] != context_version_id
                    or current["dataset_id"] != dataset_id
                    or dataset["admission_status"] != "ACCEPTED"
                    or not dataset["is_active"]
                ):
                    raise InvalidStateTransitionError("STRATEGY_CONTEXT_NOT_CURRENT")
                latest = self.get_latest_strategy(context_version_id)
                if latest is not None and latest["strategy_json"] == strategy_json:
                    raise InvalidStateTransitionError("STRATEGY_UNCHANGED")
            if verify_current is not None:
                verify_current()
            if version_no is None:
                version_no = int(
                    self.connection.execute(
                        """
                        SELECT COALESCE(MAX(version_no), 0) + 1
                        FROM strategy_versions
                        WHERE context_version_id = ?
                        """,
                        (context_version_id,),
                    ).fetchone()[0]
                )
            self.connection.execute(
                """
                INSERT INTO strategy_versions(
                    id, context_version_id, dataset_id, version_no, strategy_json,
                    schema_version, fingerprint, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_id,
                    context_version_id,
                    dataset_id,
                    version_no,
                    strategy_json,
                    schema_version,
                    fingerprint,
                    source,
                ),
            )
            self._audit(
                project_id=parent["project_id"],
                event_type="STRATEGY_VERSION_CREATED",
                entity_type="STRATEGY_VERSION",
                entity_id=strategy_id,
                detail={
                    "fingerprint": fingerprint,
                    "version_no": version_no,
                    **operation_detail,
                },
            )
        return strategy_id

    def get_strategy(self, strategy_version_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM strategy_versions WHERE id = ?",
            (strategy_version_id,),
        ).fetchone()
        if row is None:
            raise ValueError("strategy_version inexistente")
        return row

    def get_latest_strategy(self, context_version_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM strategy_versions
            WHERE context_version_id = ?
            ORDER BY version_no DESC
            LIMIT 1
            """,
            (context_version_id,),
        ).fetchone()

    def get_latest_training_decision(self, strategy_version_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM approval_events
            WHERE stage = 'TRAINING' AND strategy_version_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (strategy_version_id,),
        ).fetchone()

    def get_run(self, run_id: str) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("run inexistente")
        return row

    def get_latest_run(self, strategy_version_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM runs
            WHERE strategy_version_id = ?
            ORDER BY started_at DESC, rowid DESC
            LIMIT 1
            """,
            (strategy_version_id,),
        ).fetchone()

    def get_report(self, report_id: str) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            raise ValueError("report inexistente")
        return row

    def get_latest_report(self, run_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM reports
            WHERE run_id = ?
            ORDER BY version_no DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()

    def get_latest_publication_decision(self, report_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM approval_events
            WHERE stage = 'PUBLICATION' AND report_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (report_id,),
        ).fetchone()

    def decide_training(
        self,
        *,
        strategy_version_id: str,
        decision: ApprovalDecision | str,
        actor_label: str,
        comment: str | None = None,
    ) -> str:
        decision_value = ApprovalDecision(decision).value
        strategy = self.connection.execute(
            """
            SELECT
                strategy.fingerprint,
                context.project_id,
                dataset.is_active = 1
                AND NOT EXISTS (
                    SELECT 1
                    FROM context_versions AS newer_context
                    WHERE newer_context.project_id = context.project_id
                      AND newer_context.version_no > context.version_no
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM strategy_versions AS newer_strategy
                    WHERE newer_strategy.context_version_id = strategy.context_version_id
                      AND newer_strategy.version_no > strategy.version_no
                ) AS is_current
            FROM strategy_versions AS strategy
            JOIN context_versions AS context ON context.id = strategy.context_version_id
            JOIN datasets AS dataset ON dataset.id = context.dataset_id
            WHERE strategy.id = ?
            """,
            (strategy_version_id,),
        ).fetchone()
        if strategy is None:
            raise ValueError("strategy_version inexistente")
        if not strategy["is_current"]:
            raise InvalidStateTransitionError("STRATEGY_NOT_CURRENT")
        approval_id = _new_id()
        with transaction(self.connection):
            self.connection.execute(
                """
                INSERT INTO approval_events(
                    id, stage, decision, strategy_version_id, report_id,
                    approved_hash, actor_label, comment
                ) VALUES (?, 'TRAINING', ?, ?, NULL, ?, ?, ?)
                """,
                (
                    approval_id,
                    decision_value,
                    strategy_version_id,
                    strategy["fingerprint"],
                    actor_label,
                    comment,
                ),
            )
            self._audit(
                project_id=strategy["project_id"],
                event_type="TRAINING_DECISION_RECORDED",
                entity_type="STRATEGY_VERSION",
                entity_id=strategy_version_id,
                detail={"decision": decision_value, "approval_id": approval_id},
            )
        return approval_id

    def start_run(
        self,
        *,
        strategy_version_id: str,
        seed: int,
        split_info: dict[str, Any],
        candidate_metrics: dict[str, Any],
        manifest: RunManifest | dict[str, Any],
        verify_current: Callable[[], None] | None = None,
    ) -> str:
        run_id, _created = self._start_run(
            strategy_version_id=strategy_version_id,
            seed=seed,
            split_info=split_info,
            candidate_metrics=candidate_metrics,
            manifest=manifest,
            operation_id=None,
            verify_current=verify_current,
        )
        return run_id

    def start_run_once(
        self,
        *,
        operation_id: str,
        strategy_version_id: str,
        seed: int,
        split_info: dict[str, Any],
        candidate_metrics: dict[str, Any],
        manifest: RunManifest | dict[str, Any],
        verify_current: Callable[[], None] | None = None,
    ) -> tuple[str, bool]:
        """MV-06: gate, run y vínculo de operación se confirman en la misma transacción.

        Devuelve (run_id, creado_ahora). El llamador nunca vuelve a entrenar si recibe False.
        Una operación fallida/incompleta tampoco crea otra ejecución al reintentarse.
        """

        if not isinstance(operation_id, str) or not 1 <= len(operation_id.strip()) <= 200:
            raise ValueError("La operación de entrenamiento requiere un identificador válido")
        return self._start_run(
            strategy_version_id=strategy_version_id,
            seed=seed,
            split_info=split_info,
            candidate_metrics=candidate_metrics,
            manifest=manifest,
            operation_id=operation_id.strip(),
            verify_current=verify_current,
        )

    def _start_run(
        self,
        *,
        strategy_version_id: str,
        seed: int,
        split_info: dict[str, Any],
        candidate_metrics: dict[str, Any],
        manifest: RunManifest | dict[str, Any],
        operation_id: str | None,
        verify_current: Callable[[], None] | None = None,
    ) -> tuple[str, bool]:
        strategy = self.connection.execute(
            """
            SELECT context.project_id, strategy.fingerprint
            FROM strategy_versions AS strategy
            JOIN context_versions AS context ON context.id = strategy.context_version_id
            WHERE strategy.id = ?
            """,
            (strategy_version_id,),
        ).fetchone()
        if strategy is None:
            raise ValueError("strategy_version inexistente")
        split_json = canonical_json(JsonObject(value=split_info).value)
        metrics_json = canonical_json(JsonObject(value=candidate_metrics).value)
        manifest_json = canonical_json(RunManifest.model_validate(manifest))
        request_sha256 = canonical_hash(
            {
                "strategy_version_id": strategy_version_id,
                "strategy_fingerprint": strategy["fingerprint"],
                "seed": seed,
                "split_info": json.loads(split_json),
                "candidate_metrics": json.loads(metrics_json),
                "manifest": json.loads(manifest_json),
            }
        )
        try:
            with transaction(self.connection):
                previous = None
                if operation_id is not None:
                    previous = self.connection.execute(
                        "SELECT strategy_version_id, run_id, request_sha256 "
                        "FROM training_operations WHERE operation_id = ?",
                        (operation_id,),
                    ).fetchone()
                    if previous is not None and (
                        previous["strategy_version_id"] != strategy_version_id
                        or previous["request_sha256"] != request_sha256
                    ):
                        raise InvalidStateTransitionError("TRAINING_OPERATION_CONFLICT")
                require_training_approval(self.connection, strategy_version_id)
                # Las capacidades nuevas no pueden entrenar saltándose su definición
                # confirmada; también se aplica al flujo de UI y a replays.
                from core.domain.predictive_guard import require_capability_training_binding

                binding = require_capability_training_binding(self, strategy_version_id)
                recorded_binding = json.loads(manifest_json).get("capability_training_binding")
                if binding is not None and recorded_binding is not None:
                    if canonical_json(recorded_binding) != canonical_json(binding):
                        raise InvalidStateTransitionError("CAPABILITY_EVIDENCE_CHANGED")
                if previous is not None:
                    return str(previous["run_id"]), False
                if verify_current is not None:
                    verify_current()
                run_id = _new_id()
                self.connection.execute(
                    """
                    INSERT INTO runs(
                        id, strategy_version_id, status, seed, split_info_json,
                        candidate_metrics_json, manifest_json
                    ) VALUES (?, ?, 'RUNNING', ?, ?, ?, ?)
                    """,
                    (run_id, strategy_version_id, seed, split_json, metrics_json, manifest_json),
                )
                if operation_id is not None:
                    self.connection.execute(
                        "INSERT INTO training_operations("
                        "operation_id, strategy_version_id, run_id, request_sha256) "
                        "VALUES (?, ?, ?, ?)",
                        (operation_id, strategy_version_id, run_id, request_sha256),
                    )
                self._audit(
                    project_id=strategy["project_id"],
                    event_type="RUN_STARTED",
                    entity_type="RUN",
                    entity_id=run_id,
                    detail={
                        "strategy_version_id": strategy_version_id,
                        **({"operation_id": operation_id} if operation_id is not None else {}),
                    },
                )
        except ApprovalRequiredError:
            with transaction(self.connection):
                self._audit(
                    project_id=strategy["project_id"],
                    event_type="TRAINING_DENIED",
                    entity_type="STRATEGY_VERSION",
                    entity_id=strategy_version_id,
                    detail={"reason": "STRATEGY_NOT_APPROVED"},
                )
            raise
        return run_id, True

    def recover_stale_runs(self, *, grace_seconds: int = 300) -> list[str]:
        """Cierra como FAILED los runs RUNNING que superaron su cap más un margen.

        Un proceso caído deja el run abierto y el índice de run único bloquearía todo
        entrenamiento posterior. El cap se lee de la estrategia aprobada de cada run.
        """

        stale = self.connection.execute(
            """
            SELECT run.id, run.manifest_json, context.project_id
            FROM runs AS run
            JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
            JOIN context_versions AS context ON context.id = strategy.context_version_id
            WHERE run.status = 'RUNNING'
              AND (julianday('now') - julianday(run.started_at)) * 86400.0 >
                  COALESCE(
                      json_extract(strategy.strategy_json, '$.caps.max_training_seconds'),
                      1800
                  ) + ?
            """,
            (grace_seconds,),
        ).fetchall()
        recovered: list[str] = []
        for row in stale:
            manifest = json.loads(row["manifest_json"])
            manifest["execution_status"] = RunStatus.FAILED.value
            manifest["warnings"] = sorted({*manifest.get("warnings", []), "STALE_RUN_RECOVERED"})
            with transaction(self.connection):
                updated = self.connection.execute(
                    """
                    UPDATE runs
                    SET status = 'FAILED', finished_at = CURRENT_TIMESTAMP,
                        manifest_json = ?, error_message = 'STALE_RUN_RECOVERED'
                    WHERE id = ? AND status = 'RUNNING'
                    """,
                    (canonical_json(RunManifest.model_validate(manifest)), row["id"]),
                ).rowcount
                if updated:
                    self._audit(
                        project_id=row["project_id"],
                        event_type="RUN_RECOVERED_AS_FAILED",
                        entity_type="RUN",
                        entity_id=row["id"],
                        detail={"reason": "STALE_RUN_RECOVERED", "grace_seconds": grace_seconds},
                    )
                    recovered.append(row["id"])
        return recovered

    def finish_run(
        self,
        run_id: str,
        *,
        status: RunStatus | str,
        split_info: dict[str, Any] | None = None,
        candidate_metrics: dict[str, Any],
        selected_model: str | None = None,
        artifact_path: str | None = None,
        manifest: RunManifest | dict[str, Any],
        error_message: str | None = None,
        audit_event_id: str | None = None,
    ) -> None:
        status_value = RunStatus(status).value
        if status_value == RunStatus.RUNNING:
            raise InvalidStateTransitionError("RUNNING no es un estado final")
        run = self.connection.execute(
            """
            SELECT context.project_id, run.split_info_json
            FROM runs AS run
            JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
            JOIN context_versions AS context ON context.id = strategy.context_version_id
            WHERE run.id = ?
            """,
            (run_id,),
        ).fetchone()
        if run is None:
            raise ValueError("run inexistente")
        split_json = (
            run["split_info_json"]
            if split_info is None
            else canonical_json(JsonObject(value=split_info).value)
        )
        metrics_json = canonical_json(JsonObject(value=candidate_metrics).value)
        manifest_json = canonical_json(RunManifest.model_validate(manifest))
        with transaction(self.connection):
            self.connection.execute(
                """
                UPDATE runs
                SET status = ?, finished_at = CURRENT_TIMESTAMP,
                    split_info_json = ?, candidate_metrics_json = ?,
                    selected_model = ?, artifact_path = ?,
                    manifest_json = ?, error_message = ?
                WHERE id = ?
                """,
                (
                    status_value,
                    split_json,
                    metrics_json,
                    selected_model,
                    artifact_path,
                    manifest_json,
                    error_message,
                    run_id,
                ),
            )
            self._audit(
                project_id=run["project_id"],
                event_type="RUN_FINISHED",
                entity_type="RUN",
                entity_id=run_id,
                detail={"status": status_value},
                event_id=audit_event_id,
            )

    def create_report(
        self,
        *,
        run_id: str,
        payload: ReportPayload | dict[str, Any],
    ) -> str:
        model = ReportPayload.model_validate(payload)
        payload_data = model.model_dump(mode="json")
        report = self.connection.execute(
            """
            SELECT context.project_id, COALESCE(MAX(report.version_no), 0) AS max_version
            FROM runs AS run
            JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
            JOIN context_versions AS context ON context.id = strategy.context_version_id
            LEFT JOIN reports AS report ON report.run_id = run.id
            WHERE run.id = ?
            GROUP BY context.project_id
            """,
            (run_id,),
        ).fetchone()
        if report is None:
            raise ValueError("run inexistente")
        version_no = report["max_version"] + 1
        report_id = _new_id()
        report_sha256 = report_fingerprint(
            run_id=run_id,
            metrics=payload_data["metrics"],
            insights=payload_data["insights"],
            limitations=payload_data["limitations"],
            version_no=version_no,
        )
        with transaction(self.connection):
            self.connection.execute(
                """
                UPDATE reports
                SET status = 'SUPERSEDED'
                WHERE run_id = ? AND status <> 'SUPERSEDED'
                """,
                (run_id,),
            )
            self.connection.execute(
                """
                INSERT INTO reports(
                    id, run_id, version_no, payload_json, report_sha256, status
                ) VALUES (?, ?, ?, ?, ?, 'DRAFT')
                """,
                (
                    report_id,
                    run_id,
                    version_no,
                    canonical_json(model),
                    report_sha256,
                ),
            )
            self._audit(
                project_id=report["project_id"],
                event_type="REPORT_VERSION_CREATED",
                entity_type="REPORT",
                entity_id=report_id,
                detail={"report_sha256": report_sha256, "version_no": version_no},
            )
        return report_id

    def decide_publication(
        self,
        *,
        report_id: str,
        decision: ApprovalDecision | str,
        actor_label: str,
        comment: str | None = None,
    ) -> str:
        decision_value = ApprovalDecision(decision).value
        report = self.connection.execute(
            """
            SELECT report.report_sha256, report.status, context.project_id
            FROM reports AS report
            JOIN runs AS run ON run.id = report.run_id
            JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
            JOIN context_versions AS context ON context.id = strategy.context_version_id
            WHERE report.id = ?
            """,
            (report_id,),
        ).fetchone()
        if report is None:
            raise ValueError("report inexistente")
        if decision_value == ApprovalDecision.APPROVED and not is_report_current(
            self.connection, report_id
        ):
            raise InvalidStateTransitionError("REPORT_NOT_CURRENT")
        next_status = (
            ReportStatus.APPROVED.value
            if decision_value == ApprovalDecision.APPROVED
            else ReportStatus.REJECTED.value
        )
        approval_id = _new_id()
        with transaction(self.connection):
            self.connection.execute(
                """
                INSERT INTO approval_events(
                    id, stage, decision, strategy_version_id, report_id,
                    approved_hash, actor_label, comment
                ) VALUES (?, 'PUBLICATION', ?, NULL, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    decision_value,
                    report_id,
                    report["report_sha256"],
                    actor_label,
                    comment,
                ),
            )
            if report["status"] != next_status:
                self.connection.execute(
                    "UPDATE reports SET status = ? WHERE id = ?",
                    (next_status, report_id),
                )
            self._audit(
                project_id=report["project_id"],
                event_type="PUBLICATION_DECISION_RECORDED",
                entity_type="REPORT",
                entity_id=report_id,
                detail={"decision": decision_value, "approval_id": approval_id},
            )
        return approval_id

    def publish_report(self, report_id: str) -> None:
        report = self.connection.execute(
            """
            SELECT report.status, context.project_id
            FROM reports AS report
            JOIN runs AS run ON run.id = report.run_id
            JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
            JOIN context_versions AS context ON context.id = strategy.context_version_id
            WHERE report.id = ?
            """,
            (report_id,),
        ).fetchone()
        if report is None:
            raise ValueError("report inexistente")
        try:
            require_publication_approval(self.connection, report_id)
        except ApprovalRequiredError:
            with transaction(self.connection):
                self._audit(
                    project_id=report["project_id"],
                    event_type="PUBLICATION_DENIED",
                    entity_type="REPORT",
                    entity_id=report_id,
                    detail={"reason": "REPORT_NOT_APPROVED"},
                )
            raise
        if report["status"] != ReportStatus.APPROVED:
            raise InvalidStateTransitionError("solo un reporte APPROVED puede publicarse")
        with transaction(self.connection):
            self.connection.execute(
                "UPDATE reports SET status = 'PUBLISHED' WHERE id = ?",
                (report_id,),
            )
            self._audit(
                project_id=report["project_id"],
                event_type="REPORT_PUBLISHED",
                entity_type="REPORT",
                entity_id=report_id,
                detail={"visibility": "PRIVATE_DEMO"},
            )

    # -- evidencia de release para integración SDLC (SDLC-01) -------------------------

    def get_release_evidence_source(self, report_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            """
            SELECT
                report.id AS report_id,
                report.status AS report_status,
                report.report_sha256,
                run.id AS run_id,
                run.status AS run_status,
                run.selected_model,
                run.artifact_path,
                run.manifest_json,
                run.candidate_metrics_json,
                strategy.id AS strategy_version_id,
                strategy.version_no AS strategy_version_no,
                strategy.fingerprint AS strategy_fingerprint,
                context.id AS context_version_id,
                context.version_no AS context_version_no,
                context.context_sha256,
                context.project_id,
                dataset.sha256 AS dataset_sha256
            FROM reports AS report
            JOIN runs AS run ON run.id = report.run_id
            JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
            JOIN context_versions AS context ON context.id = strategy.context_version_id
            JOIN datasets AS dataset ON dataset.id = strategy.dataset_id
            WHERE report.id = ?
            """,
            (report_id,),
        ).fetchone()
        if row is None:
            raise ValueError("report inexistente")
        return row

    def record_release_evidence_export(
        self,
        *,
        report_id: str,
        actor_label: str,
        detail: dict[str, Any],
    ) -> str:
        source = self.get_release_evidence_source(report_id)
        with transaction(self.connection):
            return self._audit(
                project_id=source["project_id"],
                event_type="RELEASE_EVIDENCE_EXPORTED",
                entity_type="REPORT",
                entity_id=report_id,
                detail={**detail, "actor_label": actor_label},
            )

    # -- borradores de contexto (CTX-01.6) -------------------------------------------

    def get_active_draft(
        self, project_id: str, dataset_id: str, dataset_sha256: str
    ) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM context_drafts
            WHERE project_id = ? AND dataset_id = ? AND dataset_sha256 = ?
              AND status = 'ACTIVE'
            """,
            (project_id, dataset_id, dataset_sha256),
        ).fetchone()

    def create_draft(
        self,
        *,
        project_id: str,
        dataset_id: str,
        dataset_sha256: str,
        revision: int,
        draft_json: str,
    ) -> str:
        draft_id = _new_id()
        with transaction(self.connection):
            self.connection.execute(
                """
                INSERT INTO context_drafts(
                    id, project_id, dataset_id, dataset_sha256, status, revision, draft_json
                ) VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?)
                """,
                (draft_id, project_id, dataset_id, dataset_sha256, revision, draft_json),
            )
            self._audit(
                project_id=project_id,
                event_type="CONTEXT_DRAFT_STARTED",
                entity_type="CONTEXT_DRAFT",
                entity_id=draft_id,
                detail={"dataset_id": dataset_id},
            )
        return draft_id

    def update_draft(
        self, draft_id: str, *, expected_revision: int, revision: int, draft_json: str
    ) -> bool:
        """Escritura optimista: falla si otra pestaña guardó antes."""

        with transaction(self.connection):
            cursor = self.connection.execute(
                """
                UPDATE context_drafts
                SET revision = ?, draft_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'ACTIVE' AND revision = ?
                """,
                (revision, draft_json, draft_id, expected_revision),
            )
        return cursor.rowcount == 1

    def close_draft(
        self,
        draft_id: str,
        *,
        status: str,
        project_id: str,
        context_version_id: str | None = None,
    ) -> None:
        with transaction(self.connection):
            cursor = self.connection.execute(
                """
                UPDATE context_drafts
                SET status = ?, context_version_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'ACTIVE'
                """,
                (status, context_version_id, draft_id),
            )
            if cursor.rowcount != 1:
                raise InvalidStateTransitionError("DRAFT_NOT_ACTIVE")
            self._audit(
                project_id=project_id,
                event_type=f"CONTEXT_DRAFT_{status}",
                entity_type="CONTEXT_DRAFT",
                entity_id=draft_id,
                detail={"context_version_id": context_version_id},
            )

    # -- explicación de estrategia (AI-01.8) -----------------------------------------

    def add_strategy_explanation(
        self,
        *,
        strategy_version_id: str,
        source: str,
        explanation: dict[str, Any],
        ai_trace: dict[str, Any] | None,
    ) -> str:
        strategy = self.get_strategy(strategy_version_id)
        context = self.get_context(strategy["context_version_id"])
        explanation_id = _new_id()
        with transaction(self.connection):
            self.connection.execute(
                """
                INSERT INTO strategy_explanations(
                    id, strategy_version_id, strategy_fingerprint, source,
                    explanation_json, ai_trace_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    explanation_id,
                    strategy_version_id,
                    strategy["fingerprint"],
                    source,
                    canonical_json(JsonObject(value=explanation).value),
                    None if ai_trace is None else _validate_audit_detail(ai_trace),
                ),
            )
            self._audit(
                project_id=context["project_id"],
                event_type="STRATEGY_EXPLANATION_CREATED",
                entity_type="STRATEGY_VERSION",
                entity_id=strategy_version_id,
                detail={"source": source, "explanation_id": explanation_id},
            )
        return explanation_id

    def get_latest_strategy_explanation(self, strategy_version_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT explanation.*
            FROM strategy_explanations AS explanation
            JOIN strategy_versions AS strategy ON strategy.id = explanation.strategy_version_id
            WHERE explanation.strategy_version_id = ?
              AND explanation.strategy_fingerprint = strategy.fingerprint
            ORDER BY explanation.created_at DESC, explanation.rowid DESC
            LIMIT 1
            """,
            (strategy_version_id,),
        ).fetchone()

    # -- uso de modelos aprobados (EXT-A) ---------------------------------------------

    def get_run_project_id(self, run_id: str) -> str:
        row = self.connection.execute(
            """
            SELECT context.project_id
            FROM runs AS run
            JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
            JOIN context_versions AS context ON context.id = strategy.context_version_id
            WHERE run.id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError("run inexistente")
        return str(row["project_id"])

    def save_project_notes(self, project_id: str, body: str, *, actor_label: str) -> str:
        """GA-13: nueva versión de las notas libres del negocio; nunca reemplaza la anterior."""

        if not actor_label or not actor_label.strip():
            raise ValueError("Indica quién escribe las notas")
        text = body.strip()
        if len(text) > 4000:
            raise ValueError("Las notas pueden tener hasta 4000 caracteres")
        note_id = _new_id()
        with transaction(self.connection):
            self.connection.execute(
                "INSERT INTO project_notes(id, project_id, body, actor_label) VALUES (?, ?, ?, ?)",
                (note_id, project_id, text, actor_label.strip()),
            )
            self._audit(
                project_id=project_id,
                event_type="PROJECT_NOTES_SAVED",
                entity_type="PROJECT",
                entity_id=project_id,
                detail={"actor_label": actor_label.strip(), "characters": len(text)},
            )
        return note_id

    def latest_project_notes(self, project_id: str) -> str:
        row = self.connection.execute(
            "SELECT body FROM project_notes WHERE project_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (project_id,),
        ).fetchone()
        return "" if row is None else str(row["body"])

    def record_ai_data_shared(
        self, *, project_id: str, purpose: str, columns: list[str], rows: int
    ) -> str:
        """ADR-014: qué columnas y cuántas filas salieron hacia la IA, sin sus valores."""

        with transaction(self.connection):
            return self._audit(
                project_id=project_id,
                event_type="AI_DATA_SHARED",
                entity_type="PROJECT",
                entity_id=project_id,
                detail={"purpose": purpose, "columns": sorted(columns), "rows": rows},
            )

    def record_scoring(self, *, run_id: str, actor_label: str, detail: dict[str, Any]) -> str:
        """Audita un uso del modelo con conteos y hash; nunca guarda las filas evaluadas."""

        project_id = self.get_run_project_id(run_id)
        with transaction(self.connection):
            return self._audit(
                project_id=project_id,
                event_type="MODEL_USED_FOR_SCORING",
                entity_type="RUN",
                entity_id=run_id,
                detail={**detail, "actor_label": actor_label},
            )

    def list_scorings(self, run_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT * FROM audit_events
            WHERE entity_type = 'RUN' AND entity_id = ? AND event_type = 'MODEL_USED_FOR_SCORING'
            ORDER BY created_at DESC, rowid DESC
            """,
            (run_id,),
        ).fetchall()

    # -- reentrenamiento con HITL (MON-03) ---------------------------------------------

    def record_retraining_started(
        self, *, project_id: str, actor_label: str, detail: dict[str, Any]
    ) -> str:
        """Enlaza un proyecto de reentrenamiento con el modelo del que nace."""

        with transaction(self.connection):
            return self._audit(
                project_id=project_id,
                event_type="RETRAINING_STARTED",
                entity_type="PROJECT",
                entity_id=project_id,
                detail={**detail, "actor_label": actor_label},
            )

    def list_retraining_links(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT project_id, detail_json, created_at FROM audit_events
            WHERE event_type = 'RETRAINING_STARTED'
            ORDER BY created_at, rowid
            """
        ).fetchall()
        return [{**json.loads(row["detail_json"]), "project_id": row["project_id"]} for row in rows]

    def record_model_comparison(
        self, *, run_id: str, actor_label: str, detail: dict[str, Any]
    ) -> str:
        project_id = self.get_run_project_id(run_id)
        with transaction(self.connection):
            return self._audit(
                project_id=project_id,
                event_type="MODEL_COMPARISON",
                entity_type="RUN",
                entity_id=run_id,
                detail={**detail, "actor_label": actor_label},
            )

    # -- comparación de experimentos (EXT-B) ------------------------------------------

    def list_project_runs(self, project_id: str) -> list[sqlite3.Row]:
        """Ejecuciones terminadas del proyecto con su contexto, estrategia y último reporte."""

        return self.connection.execute(
            """
            SELECT
                run.id AS run_id,
                run.status AS run_status,
                run.started_at,
                run.selected_model,
                run.candidate_metrics_json,
                strategy.id AS strategy_version_id,
                strategy.version_no AS strategy_version_no,
                context.id AS context_version_id,
                context.version_no AS context_version_no,
                context.context_json,
                (
                    SELECT report.status FROM reports AS report
                    WHERE report.run_id = run.id
                    ORDER BY report.version_no DESC LIMIT 1
                ) AS report_status
            FROM runs AS run
            JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
            JOIN context_versions AS context ON context.id = strategy.context_version_id
            WHERE context.project_id = ?
              AND run.status IN ('COMPLETED', 'COMPLETED_WITHOUT_RECOMMENDATION')
            ORDER BY run.started_at, run.rowid
            """,
            (project_id,),
        ).fetchall()
