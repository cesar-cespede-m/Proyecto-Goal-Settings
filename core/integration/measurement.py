"""C2-06/C2-08: materialización y seguimiento automático desde eventos observados.

El refresco sigue siendo local y por lotes. La auditoría de cada materialización actúa como
cursor acumulado por entidad. Las tablas preparadas se rehacen sólo en memoria con su receta;
no se guardan valores de filas en el reporte ni en la auditoría.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.analysis.contracts import AnalysisPlanV1
from core.analysis.engine import AnalysisEngine, load_project_source
from core.analysis.tables import TABLE_LIMITS, result_to_csv
from core.data_engine.models import DatasetClassification
from core.db.repositories import WorkflowRepository
from core.integration.events import EventIngestionService
from core.ml_engine.models import EvaluationSummaryV1
from core.ml_engine.scoring import DEGRADATION_TOLERANCE, ScoringService


class _StrictMeasurementModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class CreatedMeasurementFile(_StrictMeasurementModel):
    entity: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)


class MeasuredModel(_StrictMeasurementModel):
    run_id: str = Field(min_length=1)
    entity: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    metric_value: float
    degraded: bool


class MeasurementReport(_StrictMeasurementModel):
    schema_version: Literal["measurement-report.v1"] = "measurement-report.v1"
    created_files: list[CreatedMeasurementFile]
    unchanged_entities: list[str]
    measured_models: list[MeasuredModel]
    warnings: list[str]


def _required_label(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} no puede estar vacío")
    return normalized


def _display_name(entity: str, now: datetime) -> str:
    safe_entity = re.sub(r"[^\w.-]+", "-", entity, flags=re.UNICODE).strip(".-")
    safe_entity = safe_entity[:80].rstrip(".-") or "entity"
    return f"{safe_entity}-{now.astimezone(UTC):%Y%m%d-%H%M}.csv"


class MeasurementService:
    """Convierte eventos nuevos y mide modelos publicados sin registrar un uso manual."""

    def __init__(self, repository: WorkflowRepository, storage_root: str | Path) -> None:
        self.repository = repository
        self.storage_root = Path(storage_root)

    def _entity_state(self, project_id: str) -> list[tuple[str, int, int]]:
        """Devuelve entidad, total y resultados nuevos desde la última materialización."""

        previous: dict[str, int] = {}
        audits = self.repository.connection.execute(
            """
            SELECT detail_json FROM audit_events
            WHERE project_id = ? AND event_type = 'OBSERVATION_EVENTS_MATERIALIZED'
            ORDER BY rowid DESC
            """,
            (project_id,),
        ).fetchall()
        for audit in audits:
            detail = json.loads(str(audit["detail_json"]))
            entity = detail.get("entity")
            event_count = detail.get("event_count")
            if isinstance(entity, str) and isinstance(event_count, int):
                previous.setdefault(entity, event_count)

        totals = self.repository.connection.execute(
            """
            SELECT entity, COUNT(*) AS event_count
            FROM observation_events
            WHERE project_id = ?
            GROUP BY entity
            ORDER BY entity COLLATE NOCASE, entity
            """,
            (project_id,),
        ).fetchall()
        state = []
        for row in totals:
            entity = str(row["entity"])
            total = int(row["event_count"])
            offset = min(previous.get(entity, 0), total)
            new_outcomes = self.repository.connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT outcome_json
                    FROM observation_events
                    WHERE project_id = ? AND entity = ?
                    ORDER BY rowid
                    LIMIT -1 OFFSET ?
                ) AS new_events
                WHERE outcome_json IS NOT NULL
                """,
                (project_id, entity, offset),
            ).fetchone()[0]
            state.append((entity, total - offset, int(new_outcomes)))
        return state

    def _dataset_content(self, dataset_id: str) -> tuple[str, bytes]:
        dataset = self.repository.get_dataset(dataset_id)
        relative = PurePosixPath(str(dataset["stored_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("La ruta privada del archivo no es válida")
        root = self.storage_root.resolve()
        path = (root / Path(*relative.parts)).resolve()
        if root not in path.parents:
            raise RuntimeError("La ruta privada del archivo salió del storage")
        return str(dataset["original_filename_display"]), path.read_bytes()

    @staticmethod
    def _warning(run_id: str, reason: str) -> str:
        return f"El seguimiento del modelo {run_id} no aplica en este refresco ({reason})."

    def _rebuild_analysis_table(
        self,
        *,
        run_id: str,
        table: sqlite3.Row,
        files_by_entity: dict[str, CreatedMeasurementFile],
        warnings: list[str],
    ) -> tuple[str, bytes] | None:
        """Rehace una receta con las conversiones nuevas, sin persistir su resultado."""

        recipe_sources = json.loads(str(table["sources_json"]))
        loaded = {}
        entities: list[str] = []
        missing: list[str] = []
        prefix = "observation_events:"
        for recipe_source in recipe_sources:
            name = str(recipe_source["name"])
            original = self.repository.get_dataset(str(recipe_source["dataset_id"]))
            source = str(original["source"])
            if not source.startswith(prefix):
                missing.append(name)
                continue
            entity = source[len(prefix) :]
            replacement = files_by_entity.get(entity)
            if replacement is None:
                missing.append(name)
                continue
            loaded[name] = load_project_source(
                self.repository,
                self.storage_root,
                replacement.dataset_id,
            )
            if entity not in entities:
                entities.append(entity)

        if missing:
            warnings.extend(
                self._warning(run_id, f"falta versión nueva de {name}") for name in missing
            )
            return None

        plan = AnalysisPlanV1.model_validate(json.loads(str(table["plan_json"])))
        result = AnalysisEngine(TABLE_LIMITS).run(plan, loaded)
        return ", ".join(entities), result_to_csv(result)

    def _measure_models(
        self,
        project_id: str,
        *,
        files_by_entity: dict[str, CreatedMeasurementFile],
        entities_with_new_outcomes: set[str],
        warnings: list[str],
    ) -> list[MeasuredModel]:
        scoring = ScoringService(self.repository, self.storage_root)
        measured: list[MeasuredModel] = []
        content_by_entity: dict[str, tuple[str, bytes]] = {}
        for candidate in self.repository.list_project_runs(project_id):
            if candidate["report_status"] != "PUBLISHED":
                continue
            run_id = str(candidate["run_id"])
            try:
                run, _, proposal = scoring.approved_run(run_id)
            except (LookupError, RuntimeError, ValueError):
                # Un reporte histórico puede seguir publicado aunque ya no sea vigente.
                continue

            training_dataset = self.repository.get_dataset(proposal.dataset_id)
            table = self.repository.get_analysis_table(proposal.dataset_id)
            if table is None:
                source = str(training_dataset["source"])
                prefix = "observation_events:"
                if not source.startswith(prefix):
                    continue
                entity = source[len(prefix) :]
                if entity not in entities_with_new_outcomes or entity not in files_by_entity:
                    continue
                filename, content = content_by_entity.setdefault(
                    entity,
                    self._dataset_content(files_by_entity[entity].dataset_id),
                )
            else:
                try:
                    rebuilt = self._rebuild_analysis_table(
                        run_id=run_id,
                        table=table,
                        files_by_entity=files_by_entity,
                        warnings=warnings,
                    )
                except Exception as error:
                    reason = str(getattr(error, "code", "receta no disponible"))
                    warnings.append(self._warning(run_id, reason))
                    continue
                if rebuilt is None:
                    continue
                entity, content = rebuilt
                filename = str(training_dataset["original_filename_display"])

            try:
                if proposal.task_type == "binary_classification":
                    evaluate = getattr(scoring, "evaluate", None)
                    if not callable(evaluate):
                        warnings.append(self._warning(run_id, "método de evaluación no disponible"))
                        continue
                    result = evaluate(run_id, filename=filename, content=content)
                    metric = str(proposal.primary_metric)
                    value = None
                    if result.follow_up is not None:
                        value = getattr(result.follow_up, metric, None)
                    if value is None:
                        warnings.append(self._warning(run_id, "métrica principal no disponible"))
                        continue
                    evaluation = EvaluationSummaryV1.model_validate(
                        json.loads(str(run["candidate_metrics_json"]))
                    )
                    reference = (
                        None
                        if evaluation.holdout is None
                        else getattr(evaluation.holdout, metric, None)
                    )
                    degraded = bool(
                        reference is not None
                        and (float(reference) - float(value)) / max(abs(float(reference)), 1e-9)
                        > DEGRADATION_TOLERANCE
                    )
                else:
                    evaluate_prediction = getattr(scoring, "evaluate_prediction", None)
                    if not callable(evaluate_prediction):
                        warnings.append(self._warning(run_id, "método de evaluación no disponible"))
                        continue
                    result = evaluate_prediction(run_id, filename=filename, content=content)
                    metric = result.primary_metric
                    value = None if result.follow_up is None else result.follow_up.get(metric)
                    if value is None:
                        warnings.append(self._warning(run_id, "métrica principal no disponible"))
                        continue
                    degraded = result.degraded
            except Exception as error:  # una medición no bloquea las demás entidades
                reason = str(getattr(error, "code", "evaluación no disponible"))
                warnings.append(self._warning(run_id, reason))
                continue

            measured.append(
                MeasuredModel(
                    run_id=run_id,
                    entity=entity,
                    task_type=proposal.task_type,
                    primary_metric=metric,
                    metric_value=float(value),
                    degraded=bool(degraded),
                )
            )
        return measured

    def refresh(
        self,
        project_id: str,
        *,
        actor_label: str,
        classification: DatasetClassification | str,
    ) -> MeasurementReport:
        actor = _required_label(actor_label, "actor_label")
        classification_value = DatasetClassification(classification)
        if classification_value not in {
            DatasetClassification.SYNTHETIC,
            DatasetClassification.SANDBOX,
        }:
            raise ValueError("classification debe ser synthetic o sandbox")

        ingestion = EventIngestionService(self.repository, self.storage_root)
        created: list[CreatedMeasurementFile] = []
        unchanged: list[str] = []
        entities_with_new_outcomes: set[str] = set()
        now = datetime.now(UTC)
        for entity, new_events, new_outcomes in self._entity_state(project_id):
            if new_events == 0:
                unchanged.append(entity)
                continue
            filename = _display_name(entity, now)
            dataset_id = ingestion.materialize(
                project_id,
                entity=entity,
                actor_label=actor,
                display_name=filename,
                classification=classification_value,
            )
            created.append(
                CreatedMeasurementFile(
                    entity=entity,
                    dataset_id=dataset_id,
                    filename=filename,
                )
            )
            if new_outcomes:
                entities_with_new_outcomes.add(entity)

        warnings: list[str] = []
        files_by_entity = {item.entity: item for item in created}
        measured = self._measure_models(
            project_id,
            files_by_entity=files_by_entity,
            entities_with_new_outcomes=entities_with_new_outcomes,
            warnings=warnings,
        )
        report = MeasurementReport(
            created_files=created,
            unchanged_entities=unchanged,
            measured_models=measured,
            warnings=warnings,
        )
        self.repository.record_measurement_refresh(
            project_id=project_id,
            actor_label=actor,
            classification=classification_value.value,
            created_files=len(created),
            unchanged_entities=len(unchanged),
            measured_models=len(measured),
            warnings=len(warnings),
        )
        return report
