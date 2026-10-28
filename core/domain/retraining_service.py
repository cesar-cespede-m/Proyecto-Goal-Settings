"""Reentrenamiento con aprobación humana (MON-03, ADR-013).

Reentrenar no ejecuta nada por su cuenta: crea un proyecto hermano con un archivo reciente y
las mismas decisiones de negocio, confirmadas por la persona que lo propone. El modelo anterior
sigue vigente y utilizable mientras el nuevo pasa por las mismas aprobaciones de entrenamiento
y publicación. Se usa un proyecto aparte porque guardar decisiones nuevas en el mismo proyecto
dejaría sin vigencia al modelo actual antes de tener reemplazo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.analysis.contracts import AnalysisPlanV1
from core.analysis.engine import load_project_source
from core.analysis.snapshot import ObservationCoverageV1
from core.analysis.snapshot_recipe import (
    rebuild_from_recipe,
    record_snapshot_recipe,
    snapshot_recipe,
)
from core.analysis.tables import AnalysisTableService
from core.analysis.validation import AnalysisPlanError
from core.contracts.platform import RecipeV1, VersionRef
from core.data_engine import DataIngestionService, IngestionLimits
from core.data_engine.loading import read_validated_frame
from core.data_engine.validator import validate_csv_bytes
from core.db.connection import transaction
from core.db.repositories import WorkflowRepository
from core.domain.context_service import BusinessContextService
from core.domain.strategy import StrategyProposalV1
from core.ml_engine.execution import RunBlockedError
from core.ml_engine.scoring import ScoringService

_PENDING_ACTORS = {"", "pendiente", "pending", "todo"}


@dataclass(frozen=True)
class RetrainingStarted:
    project_id: str
    dataset_id: str
    context_version_id: str


class RetrainingService:
    def __init__(self, repository: WorkflowRepository, storage_root: str | Path) -> None:
        self.repository = repository
        self.storage_root = Path(storage_root)

    def start(
        self,
        *,
        source_run_id: str,
        filename: str,
        content: bytes,
        actor_label: str,
        data_authorized: bool,
    ) -> RetrainingStarted:
        if actor_label.strip().casefold() in _PENDING_ACTORS:
            raise RunBlockedError("ACTOR_REQUIRED", "Indica quién propone el reentrenamiento")
        if not data_authorized:
            raise RunBlockedError(
                "DATA_AUTHORIZATION_REQUIRED",
                "Confirma que el archivo es sintético, de sandbox o anonimizado y está autorizado",
            )
        run, report, proposal = ScoringService(self.repository, self.storage_root).approved_run(
            source_run_id
        )
        strategy = self.repository.get_strategy(run["strategy_version_id"])
        context_row = self.repository.get_context(strategy["context_version_id"])
        source_project_id = str(context_row["project_id"])
        source_dataset = self.repository.get_dataset(proposal.dataset_id)
        classification = json.loads(source_dataset["profile_json"])["data_classification"]

        previous = self.list_retrainings(source_project_id)
        name = f"{self._project_name(source_project_id)} · reentrenamiento {len(previous) + 1}"
        project_id = self.repository.create_project(name)
        admission = DataIngestionService(self.repository, self.storage_root).ingest_bytes(
            project_id=project_id,
            request={
                "original_filename_display": filename,
                "source_name": str(source_dataset["source"]),
                "license": source_dataset["license"],
                "classification": classification,
                "authorization_status": "approved",
                "anonymization_declared": bool(source_dataset["anonymization_declared"]),
            },
            content=content,
        )
        if admission.metadata.admission_status != "ACCEPTED":
            raise RunBlockedError(
                "RETRAINING_DATASET_NOT_ACCEPTED",
                "El archivo reciente no quedó aceptado; revisa su procedencia y formato",
            )
        context: dict[str, Any] = json.loads(context_row["context_json"])["context"]
        context["decision_actor_label"] = actor_label.strip()
        created = BusinessContextService(self.repository, self.storage_root).create_version(
            project_id=project_id,
            dataset_id=admission.metadata.dataset_id,
            context=context,
        )
        self.repository.record_retraining_started(
            project_id=project_id,
            actor_label=actor_label.strip(),
            detail={
                "source_project_id": source_project_id,
                "source_run_id": source_run_id,
                "source_report_id": str(report["id"]),
                "dataset_sha256": admission.metadata.sha256_raw,
                "context_version_id": created.context_version_id,
            },
        )
        return RetrainingStarted(
            project_id=project_id,
            dataset_id=admission.metadata.dataset_id,
            context_version_id=created.context_version_id,
        )

    def start_from_snapshot(
        self,
        *,
        source_run_id: str,
        filename: str,
        content: bytes,
        coverage: ObservationCoverageV1 | dict[str, Any] | None,
        actor_label: str,
        data_authorized: bool,
    ) -> RetrainingStarted:
        """Reconstruye resultados maduros y conserva la receta para el siguiente lote.

        La cobertura siempre corresponde al archivo nuevo. Las fuentes se guardan en el
        proyecto candidato, de modo que su receta no depende de archivos de otro proyecto.
        No se crea una candidata cuando faltan autorización, cobertura o resultados maduros.
        """

        if actor_label.strip().casefold() in _PENDING_ACTORS:
            raise RunBlockedError("ACTOR_REQUIRED", "Indica quién propone el reentrenamiento")
        if not data_authorized:
            raise RunBlockedError(
                "DATA_AUTHORIZATION_REQUIRED",
                "Confirma que el archivo es sintético, de sandbox o anonimizado y está autorizado",
            )
        _run, _report, proposal = ScoringService(self.repository, self.storage_root).approved_run(
            source_run_id
        )
        recipe = snapshot_recipe(self.repository, proposal.dataset_id)
        if recipe is None:
            raise RunBlockedError(
                "SNAPSHOT_RECIPE_NOT_AVAILABLE", "Esta tabla no tiene una receta de recorrido"
            )
        source = self.repository.get_dataset(recipe["source_dataset_id"])
        events = read_validated_frame(validate_csv_bytes(filename, content, IngestionLimits()))
        entity_id = recipe.get("entity_dataset_id")
        entities = (
            load_project_source(self.repository, self.storage_root, entity_id).frame
            if entity_id
            else None
        )
        rebuilt = rebuild_from_recipe(
            self.repository, proposal.dataset_id, events, entities, coverage=coverage
        )
        eligible = rebuilt.training_frame()
        if eligible.empty:
            raise RunBlockedError(
                "SNAPSHOT_OUTCOMES_NOT_READY",
                "No hay resultados completos: confirma la cobertura del lote nuevo o espera "
                "a que termine el plazo de observación.",
            )

        started = self.start(
            source_run_id=source_run_id,
            filename="snapshot-reciente.csv",
            content=eligible.to_csv(index=False).encode("utf-8"),
            actor_label=actor_label,
            data_authorized=data_authorized,
        )
        try:
            new_source_id = self._copy_snapshot_source(
                started.project_id, source, filename, content
            )
            new_entity_id = None
            if entities is not None:
                entity_source = self.repository.get_dataset(entity_id)
                new_entity_id = self._copy_snapshot_source(
                    started.project_id,
                    entity_source,
                    str(entity_source["original_filename_display"]),
                    entities.to_csv(index=False).encode("utf-8"),
                )
            record_snapshot_recipe(
                self.repository,
                project_id=started.project_id,
                table_dataset_id=started.dataset_id,
                source_dataset_id=new_source_id,
                spec=recipe["spec"],
                actor_label=actor_label,
                entity_dataset_id=new_entity_id,
                coverage=rebuilt.readiness.coverage,
                readiness=rebuilt.readiness,
            )
        except Exception as error:
            # Los servicios anteriores ya confirmaron sus propias transacciones. No devolver
            # éxito ni continuar la navegación si la candidata quedó sin receta reproducible.
            with transaction(self.repository.connection):
                self.repository._audit(
                    project_id=started.project_id,
                    event_type="SNAPSHOT_RETRAINING_INCOMPLETE",
                    entity_type="DATASET",
                    entity_id=started.dataset_id,
                    detail={
                        "actor_label": actor_label.strip(),
                        "source_run_id": source_run_id,
                        "error_type": type(error).__name__,
                        "error_code": getattr(error, "code", "SNAPSHOT_PERSISTENCE_FAILED"),
                    },
                )
            self.repository.archive_project(started.project_id, actor_label=actor_label.strip())
            raise RunBlockedError(
                "SNAPSHOT_RETRAINING_INCOMPLETE",
                "Se archivó la candidata porque no se pudieron guardar sus fuentes y receta. "
                "La auditoría conserva el fallo; el modelo anterior sigue vigente.",
            ) from error
        return started

    def _copy_snapshot_source(
        self, project_id: str, source: Any, filename: str, content: bytes
    ) -> str:
        admission = DataIngestionService(self.repository, self.storage_root).ingest_bytes(
            project_id=project_id,
            request={
                "original_filename_display": filename,
                "source_name": str(source["source"]),
                "license": source["license"],
                "classification": json.loads(source["profile_json"])["data_classification"],
                "authorization_status": "approved",
                "anonymization_declared": bool(source["anonymization_declared"]),
            },
            content=content,
        )
        if admission.metadata.admission_status != "ACCEPTED":
            raise RunBlockedError(
                "RETRAINING_DATASET_NOT_ACCEPTED", "Una fuente del recorrido no quedó aceptada"
            )
        return admission.metadata.dataset_id

    def _project_name(self, project_id: str) -> str:
        names = {row["id"]: row["name"] for row in self.repository.list_projects()}
        return str(names.get(project_id, "Proyecto"))

    def list_retrainings(self, source_project_id: str) -> list[dict[str, Any]]:
        return [
            detail
            for detail in self.repository.list_retraining_links()
            if detail["source_project_id"] == source_project_id
        ]

    def source_of(self, project_id: str) -> dict[str, Any] | None:
        """Modelo anterior del que nace un proyecto de reentrenamiento, si lo hay."""

        for detail in self.repository.list_retraining_links():
            if detail["project_id"] == project_id:
                return detail
        return None

    # -- GA-08: receta completa ---------------------------------------------------------

    def recipe(self, run_id: str) -> RecipeV1:
        """Todo lo que formó el modelo: decisiones, preparación, estrategia, tipo y fuentes."""

        run = self.repository.get_run(run_id)
        strategy = self.repository.get_strategy(run["strategy_version_id"])
        proposal = StrategyProposalV1.model_validate(json.loads(strategy["strategy_json"]))
        context_row = self.repository.get_context(strategy["context_version_id"])
        table = self.repository.get_analysis_table(proposal.dataset_id)
        if table is None:
            sources = [VersionRef(id=proposal.dataset_id, sha256=proposal.dataset_sha256)]
            plan = None
        else:
            sources = [
                VersionRef(id=item["dataset_id"], sha256=item["sha256"])
                for item in json.loads(table["sources_json"])
            ]
            plan = VersionRef(id=str(table["id"]), sha256=str(table["plan_sha256"]))
        return RecipeV1(
            context=VersionRef(
                id=str(context_row["id"]), sha256=str(context_row["context_sha256"])
            ),
            analysis_plan=plan,
            strategy=VersionRef(id=str(strategy["id"]), sha256=str(strategy["fingerprint"])),
            task_kind=proposal.task_type,
            sources=sources,
        )

    def recipe_sources(self, run_id: str) -> list[str] | None:
        """Nombres de los archivos que pide la receta; None si el modelo no tiene preparación."""

        run = self.repository.get_run(run_id)
        strategy = self.repository.get_strategy(run["strategy_version_id"])
        table = self.repository.get_analysis_table(str(strategy["dataset_id"]))
        if table is None:
            return None
        return [str(item["name"]) for item in json.loads(table["sources_json"])]

    def start_from_recipe(
        self,
        *,
        source_run_id: str,
        files: dict[str, tuple[str, bytes]],
        actor_label: str,
        data_authorized: bool,
    ) -> RetrainingStarted:
        """Reentrena rehaciendo la tabla con versiones nuevas de cada archivo de la receta.

        Mismos pasos de preparación y mismas decisiones; la persona sigue aprobando estrategia,
        entrenamiento y publicación del modelo nuevo como siempre.
        """

        if actor_label.strip().casefold() in _PENDING_ACTORS:
            raise RunBlockedError("ACTOR_REQUIRED", "Indica quién propone el reentrenamiento")
        if not data_authorized:
            raise RunBlockedError(
                "DATA_AUTHORIZATION_REQUIRED",
                "Confirma que el archivo es sintético, de sandbox o anonimizado y está autorizado",
            )
        run, report, proposal = ScoringService(self.repository, self.storage_root).approved_run(
            source_run_id
        )
        table = self.repository.get_analysis_table(proposal.dataset_id)
        if table is None:
            raise RunBlockedError(
                "RECIPE_NOT_AVAILABLE",
                "Este modelo se entrenó con un archivo directo: reentrena subiendo ese archivo",
            )
        needed = [str(item["name"]) for item in json.loads(table["sources_json"])]
        missing = [name for name in needed if name not in files]
        if missing:
            raise RunBlockedError(
                "RECIPE_FILES_MISSING",
                "Faltan versiones nuevas de: " + ", ".join(missing),
            )
        strategy = self.repository.get_strategy(run["strategy_version_id"])
        context_row = self.repository.get_context(strategy["context_version_id"])
        source_project_id = str(context_row["project_id"])
        source_dataset = self.repository.get_dataset(proposal.dataset_id)
        classification = json.loads(source_dataset["profile_json"])["data_classification"]
        previous = self.list_retrainings(source_project_id)
        name = f"{self._project_name(source_project_id)} · reentrenamiento {len(previous) + 1}"
        project_id = self.repository.create_project(name)
        source_ids: dict[str, str] = {}
        for source_name in needed:
            filename, content = files[source_name]
            admission = DataIngestionService(self.repository, self.storage_root).ingest_bytes(
                project_id=project_id,
                request={
                    "original_filename_display": filename,
                    "source_name": str(source_dataset["source"]),
                    "license": source_dataset["license"],
                    "classification": classification,
                    "authorization_status": "approved",
                    "anonymization_declared": bool(source_dataset["anonymization_declared"]),
                },
                content=content,
            )
            if admission.metadata.admission_status != "ACCEPTED":
                raise RunBlockedError(
                    "RETRAINING_DATASET_NOT_ACCEPTED",
                    f"El archivo nuevo para «{source_name}» no quedó aceptado",
                )
            source_ids[source_name] = admission.metadata.dataset_id
        plan = AnalysisPlanV1.model_validate(json.loads(table["plan_json"]))
        try:
            saved = AnalysisTableService(self.repository, self.storage_root).save(
                project_id=project_id,
                plan=plan,
                source_ids=source_ids,
                display_name=str(table["display_name"]),
                actor_label=actor_label.strip(),
            )
        except AnalysisPlanError as error:
            raise RunBlockedError(error.code, error.message) from error
        self.repository.set_modeling_table(
            project_id, saved.dataset_id, actor_label=actor_label.strip()
        )
        context: dict[str, Any] = json.loads(context_row["context_json"])["context"]
        context["decision_actor_label"] = actor_label.strip()
        created = BusinessContextService(self.repository, self.storage_root).create_version(
            project_id=project_id,
            dataset_id=saved.dataset_id,
            context=context,
        )
        new_table = self.repository.get_dataset(saved.dataset_id)
        self.repository.record_retraining_started(
            project_id=project_id,
            actor_label=actor_label.strip(),
            detail={
                "source_project_id": source_project_id,
                "source_run_id": source_run_id,
                "source_report_id": str(report["id"]),
                "dataset_sha256": str(new_table["sha256"]),
                "context_version_id": created.context_version_id,
                "recipe_plan_sha256": str(table["plan_sha256"]),
            },
        )
        return RetrainingStarted(
            project_id=project_id,
            dataset_id=saved.dataset_id,
            context_version_id=created.context_version_id,
        )
