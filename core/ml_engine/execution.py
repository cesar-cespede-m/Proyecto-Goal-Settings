"""Orquestador síncrono ML-02 con gate, tracking SQLite y artefacto privado."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import pandas as pd

from core.data_engine.errors import DataValidationError
from core.data_engine.loading import read_validated_frame
from core.data_engine.models import IngestionLimits
from core.data_engine.validator import validate_csv_bytes
from core.db.repositories import WorkflowRepository
from core.domain.data_readiness import DataReadinessService
from core.domain.errors import DomainError
from core.domain.gates import require_training_approval
from core.domain.models import RunManifest
from core.domain.predictive_guard import require_capability_training_binding
from core.domain.states import RunStatus
from core.domain.strategy import StrategyProposalV1
from core.ml_engine.errors import CandidateTrainingError, MLValidationError
from core.ml_engine.models import RunExecutionResult
from core.ml_engine.task_kinds import get_task_kind


class RunBlockedError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class MLRunService:
    """Ejecuta un intento completo; nunca carga pickle/joblib recibido del usuario."""

    def __init__(
        self,
        repository: WorkflowRepository,
        storage_root: str | Path,
        *,
        limits: IngestionLimits | None = None,
    ) -> None:
        self.repository = repository
        self.storage_root = Path(storage_root)
        self.limits = limits or IngestionLimits()

    def _internal_path(self, relative: str) -> Path:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise RunBlockedError("INVALID_INTERNAL_PATH", "La ruta privada no es válida")
        root = self.storage_root.resolve()
        absolute = (root / Path(pure)).resolve()
        if root != absolute and root not in absolute.parents:
            raise RunBlockedError("INVALID_INTERNAL_PATH", "La ruta privada salió del storage")
        return absolute

    def _load_frame(self, dataset: Any, proposal: StrategyProposalV1) -> pd.DataFrame:
        path = self._internal_path(str(dataset["stored_path"]))
        try:
            content = path.read_bytes()
        except FileNotFoundError as error:
            raise RunBlockedError(
                "ACCEPTED_DATASET_MISSING",
                "El dataset aceptado no está disponible en el storage privado",
            ) from error
        observed_hash = hashlib.sha256(content).hexdigest()
        if observed_hash != dataset["sha256"] or observed_hash != proposal.dataset_sha256:
            raise RunBlockedError(
                "DATASET_HASH_MISMATCH",
                "El contenido del dataset ya no coincide con la versión aprobada",
            )
        validated = validate_csv_bytes(
            str(dataset["original_filename_display"]),
            content,
            self.limits,
        )
        return read_validated_frame(validated)

    @staticmethod
    def _manifest(
        proposal: StrategyProposalV1,
        *,
        versions: dict[str, str],
        warnings: list[str],
        **extra: Any,
    ) -> RunManifest:
        value = {
            "schema_version": "run-manifest.v1",
            "dataset_sha256": proposal.dataset_sha256,
            "context_sha256": proposal.context_sha256,
            "strategy_fingerprint": extra.pop("strategy_fingerprint"),
            "versions": versions,
            "hyperparameters": proposal.model_params.model_dump(mode="json"),
            "features": proposal.feature_columns,
            "warnings": sorted(set(warnings)),
            "seed": proposal.seed,
            "split": proposal.split.model_dump(mode="json"),
            "primary_metric": proposal.primary_metric.value,
            "positive_class": proposal.positive_class,
            "guardrails": proposal.guardrails.model_dump(mode="json"),
            **extra,
        }
        return RunManifest.model_validate(value)

    def _save_artifact(self, run_id: str, pipeline: Any) -> tuple[str, str]:
        relative = str(PurePosixPath("artifacts", "runs", run_id, "pipeline.joblib"))
        destination = self._internal_path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".joblib.tmp")
        joblib.dump(pipeline, temporary)
        temporary.replace(destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        return relative, digest

    def execute(
        self,
        strategy_version_id: str,
        *,
        operation_id: str | None = None,
        verify_start: Callable[[], None] | None = None,
    ) -> RunExecutionResult:
        strategy = self.repository.get_strategy(strategy_version_id)
        proposal = StrategyProposalV1.model_validate(json.loads(strategy["strategy_json"]))
        dataset = self.repository.get_dataset(str(strategy["dataset_id"]))
        if dataset["admission_status"] != "ACCEPTED":
            raise RunBlockedError("DATASET_NOT_ACCEPTED", "El dataset ya no está aceptado")

        capability_evidence = self._capability_evidence(strategy_version_id)

        self.repository.recover_stale_runs()
        initial_manifest = self._manifest(
            proposal,
            strategy_fingerprint=str(strategy["fingerprint"]),
            versions={"strategy": proposal.schema_version},
            warnings=[*proposal.warnings, "RUN_STARTED_PENDING_RESULTS"],
            execution_status="RUNNING",
            **capability_evidence,
        )
        start_inputs = {
            "strategy_version_id": strategy_version_id,
            "seed": proposal.seed,
            "split_info": {},
            "candidate_metrics": {},
            "manifest": initial_manifest,
        }
        if verify_start is not None:
            start_inputs["verify_current"] = verify_start
        if operation_id is None:
            run_id = self.repository.start_run(**start_inputs)
        else:
            run_id, created = self.repository.start_run_once(
                operation_id=operation_id, **start_inputs
            )
            if not created:
                return self._existing_result(run_id)
        artifact_path: str | None = None
        try:
            frame = self._load_frame(dataset, proposal)
            if capability_evidence:
                require_training_approval(self.repository.connection, strategy_version_id)
                if self._capability_evidence(strategy_version_id) != capability_evidence:
                    raise RunBlockedError(
                        "CAPABILITY_EVIDENCE_CHANGED",
                        "La definición o la evidencia de maduración cambió antes del ajuste",
                    )
            outcome = get_task_kind(proposal.task_type).execute(frame, proposal)
            selected = outcome.selected
            artifact_sha256: str | None = None
            if selected is not None:
                if outcome.pipeline is None:
                    raise RuntimeError("La selección no devolvió el pipeline congelado")
                artifact_path, artifact_sha256 = self._save_artifact(run_id, outcome.pipeline)
            status = (
                RunStatus.COMPLETED
                if selected is not None
                else RunStatus.COMPLETED_WITHOUT_RECOMMENDATION
            )
            final_manifest = self._manifest(
                proposal,
                strategy_fingerprint=str(strategy["fingerprint"]),
                versions={**outcome.versions, "strategy": proposal.schema_version},
                warnings=outcome.warnings,
                run_id=run_id,
                execution_status=status.value,
                selected_model=selected,
                artifact_sha256=artifact_sha256,
                holdout_evaluations=outcome.summary.holdout_evaluations,
                **capability_evidence,
            )
            self.repository.finish_run(
                run_id,
                status=status,
                split_info=outcome.split_info,
                candidate_metrics=outcome.summary.model_dump(mode="json"),
                selected_model=selected,
                artifact_path=artifact_path,
                manifest=final_manifest,
            )
            return RunExecutionResult(
                run_id=run_id,
                status=status.value,
                selected_model=selected,
                artifact_path=artifact_path,
                evaluation=outcome.summary,
            )
        except (RunBlockedError, DataValidationError, MLValidationError, DomainError) as error:
            self._close_unsuccessful(
                run_id,
                artifact_path,
                proposal,
                strategy_fingerprint=str(strategy["fingerprint"]),
                status=RunStatus.BLOCKED,
                error_message=str(getattr(error, "code", "ML_PRECONDITION_FAILED")),
                error=error,
            )
            raise
        except CandidateTrainingError as error:
            self._close_unsuccessful(
                run_id,
                artifact_path,
                proposal,
                strategy_fingerprint=str(strategy["fingerprint"]),
                status=RunStatus.FAILED,
                error_message=f"CANDIDATE_TRAINING_FAILED:{error.candidate}",
                error=error,
            )
            raise
        except Exception as error:
            self._close_unsuccessful(
                run_id,
                artifact_path,
                proposal,
                strategy_fingerprint=str(strategy["fingerprint"]),
                status=RunStatus.FAILED,
                error_message="UNEXPECTED_EXECUTION_ERROR",
                error=error,
            )
            raise

    def _capability_evidence(self, strategy_version_id: str) -> dict[str, Any]:
        """Comprueba la definición y reconstruye la cohorte antes de ajustar ML."""
        binding = require_capability_training_binding(self.repository, strategy_version_id)
        if binding is None:
            return {}
        readiness = DataReadinessService(
            self.repository, self.storage_root
        ).inspect_label_readiness(
            binding.capability_id,
            cycle_id=binding.cycle_id,
            dataset_id=binding.dataset_id,
            predictive_version_id=binding.predictive_version_id,
        )
        if readiness.status != "ready":
            raise RunBlockedError(
                "CAPABILITY_DATA_NOT_READY",
                "La tabla aún no acredita resultados maduros y una receta compatible",
            )
        return {
            "capability_training_binding": binding.model_dump(mode="json"),
            "label_readiness_sha256": readiness.evidence_sha256,
            "label_readiness_recipe_sha256": readiness.recipe_sha256,
        }

    def _existing_result(self, run_id: str) -> RunExecutionResult:
        """Reproduce evidencia persistida; no carga ni vuelve a ajustar el estimador."""

        run = self.repository.get_run(run_id)
        if run["status"] == RunStatus.RUNNING.value:
            raise RunBlockedError(
                "TRAINING_OPERATION_IN_PROGRESS", "Esta operación ya tiene un entrenamiento activo"
            )
        if run["status"] not in {
            RunStatus.COMPLETED.value,
            RunStatus.COMPLETED_WITHOUT_RECOMMENDATION.value,
        }:
            raise RunBlockedError(
                "TRAINING_OPERATION_TERMINATED",
                "La operación anterior terminó sin éxito; revisa su evidencia antes de reintentar",
            )
        if run["status"] == RunStatus.COMPLETED.value:
            manifest = json.loads(run["manifest_json"])
            if not run["artifact_path"]:
                raise RunBlockedError("MODEL_ARTIFACT_MISSING", "Falta el modelo guardado")
            path = self._internal_path(str(run["artifact_path"]))
            if not path.is_file():
                raise RunBlockedError("MODEL_ARTIFACT_MISSING", "Falta el modelo guardado")
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("artifact_sha256"):
                raise RunBlockedError("MODEL_ARTIFACT_CHANGED", "El modelo guardado cambió")
        return RunExecutionResult.model_validate(
            {
                "run_id": run_id,
                "status": run["status"],
                "selected_model": run["selected_model"],
                "artifact_path": run["artifact_path"],
                "evaluation": json.loads(run["candidate_metrics_json"]),
            }
        )

    def _close_unsuccessful(
        self,
        run_id: str,
        artifact_path: str | None,
        proposal: StrategyProposalV1,
        *,
        strategy_fingerprint: str,
        status: RunStatus,
        error_message: str,
        error: BaseException,
    ) -> None:
        """Cierra el run sin ocultar la causa original si el propio cierre falla."""

        try:
            started = self.repository.get_run(run_id)
            if started["status"] != RunStatus.RUNNING.value:
                # Un fallo al devolver la respuesta puede ocurrir después del commit exitoso.
                # Su artefacto y resultado confirmado deben seguir disponibles para el replay.
                return
            initial = json.loads(started["manifest_json"])
            capability_evidence = {
                key: initial[key]
                for key in (
                    "capability_training_binding",
                    "label_readiness_sha256",
                    "label_readiness_recipe_sha256",
                )
                if key in initial
            }
            self._remove_artifact(artifact_path)
            self.repository.finish_run(
                run_id,
                status=status,
                candidate_metrics={},
                selected_model=None,
                artifact_path=None,
                manifest=self._manifest(
                    proposal,
                    strategy_fingerprint=strategy_fingerprint,
                    versions={"strategy": proposal.schema_version},
                    warnings=[*proposal.warnings, f"RUN_{status.value}"],
                    run_id=run_id,
                    execution_status=status.value,
                    **capability_evidence,
                ),
                error_message=error_message,
            )
        except Exception as close_error:
            error.add_note(
                f"Además, no se pudo cerrar el run {run_id}: "
                f"{type(close_error).__name__}: {close_error}. "
                "recover_stale_runs lo cerrará al vencer su cap."
            )

    def _remove_artifact(self, relative: str | None) -> None:
        if relative is None:
            return
        path = self._internal_path(relative)
        if path.is_file():
            path.unlink()
