"""Uso de un modelo aprobado con casos nuevos (EXT-A, ADR-010).

Prioriza casos nuevos con el modelo de una ejecución cuyo reporte una persona aprobó y que
sigue vigente. Es un uso local y por lotes: no expone un servicio a otros sistemas, no
reentrena y no guarda las filas evaluadas (sólo un evento de auditoría con conteos y hash).
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import average_precision_score, roc_auc_score

from core.data_engine.loading import read_validated_frame
from core.data_engine.models import IngestionLimits
from core.data_engine.validator import validate_csv_bytes
from core.db.repositories import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.domain.gates import has_publication_approval, has_training_approval
from core.domain.strategy import StrategyProposalV1
from core.ml_engine.drift import DriftReportV1, compute_drift
from core.ml_engine.errors import MLValidationError
from core.ml_engine.execution import RunBlockedError
from core.ml_engine.health import ModelHealthV1, assess_model_health
from core.ml_engine.models import EvaluationSummaryV1
from core.ml_engine.task_kinds import get_task_kind


class MissingFeatureColumnsError(RunBlockedError):
    """Conserva el esquema esperado y el recibido para dar una ayuda concreta en la UI."""

    def __init__(self, missing_columns: list[str], available_columns: list[str]) -> None:
        self.missing_columns = tuple(missing_columns)
        self.available_columns = tuple(available_columns)
        super().__init__(
            "MISSING_FEATURE_COLUMNS",
            "Faltan columnas con las que se entrenó el modelo: " + ", ".join(missing_columns),
        )


class ScoredCase(BaseModel):
    rank: int = Field(ge=1)
    row_number: int = Field(ge=1, description="Fila en el archivo cargado (1 = primera fila)")
    case_id: str | None
    score: float = Field(ge=0, le=1)
    alert: bool
    seen_in_training: bool = False
    # V2-16: por qué tiene este puntaje (relación del modelo, no causa); None si no se calculó.
    reasons: str | None = None


class FollowUpMetrics(BaseModel):
    """Cómo le fue al modelo cuando el archivo ya trae el resultado real."""

    rows_with_outcome: int = Field(ge=1)
    prevalence: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float | None = Field(default=None, ge=0, le=1)
    balanced_accuracy: float | None = Field(default=None, ge=0, le=1)
    average_precision: float | None = Field(default=None, ge=0, le=1)
    roc_auc: float | None = Field(default=None, ge=0, le=1)
    training_average_precision: float | None = Field(default=None, ge=0, le=1)


class ScoringResultV1(BaseModel):
    schema_version: str = "scoring-result.v1"
    run_id: str
    report_id: str
    selected_model: str
    positive_class: str | int | float | bool
    threshold: float = Field(ge=0, le=1)
    id_column: str | None
    rows_scored: int = Field(ge=1)
    alerts: int = Field(ge=0)
    input_sha256: str
    cases: list[ScoredCase]
    ignored_columns: list[str]
    target_present: bool
    follow_up: FollowUpMetrics | None = None
    # Filas idénticas a las del archivo con que se entrenó y evaluó: su resultado no mide
    # el desempeño con casos nuevos, por eso se excluyen del seguimiento (UX-04).
    seen_in_training: int = Field(default=0, ge=0)
    # MON-01: cambios en los datos frente al entrenamiento; None si el archivo original no
    # está disponible o cambió (el control advierte, nunca bloquea).
    drift: DriftReportV1 | None = None

    def to_csv(self) -> bytes:
        frame = pd.DataFrame(
            [
                {
                    "prioridad": case.rank,
                    "fila_original": case.row_number,
                    **({self.id_column: case.case_id} if self.id_column else {}),
                    "puntaje": round(case.score, 4),
                    "alerta": "sí" if case.alert else "no",
                    "por_que": case.reasons or "",
                }
                for case in self.cases
            ]
        )
        return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


class PredictedCase(BaseModel):
    row_number: int = Field(ge=1)
    case_id: str | None
    prediction: str | float
    # V2-15: seguridad (varias categorías) o rango probable (cantidades); los dudosos se
    # marcan para revisión humana.
    confidence: float | None = Field(default=None, ge=0, le=1)
    low: float | None = None
    high: float | None = None
    review: bool = False


class PredictionResultV1(BaseModel):
    """Uso de un modelo de varias clases o de cantidades (ADR-016)."""

    schema_version: str = "prediction-result.v1"
    run_id: str
    report_id: str
    task_kind: str
    selected_model: str
    primary_metric: str
    higher_is_better: bool
    id_column: str | None
    rows_scored: int = Field(ge=1)
    input_sha256: str
    cases: list[PredictedCase]
    target_present: bool
    # Con el resultado real en el archivo: la métrica principal en estos casos frente a la
    # prueba final del entrenamiento; `degraded` marca una caída que pide revisar o reentrenar.
    follow_up: dict[str, float] | None = None
    rows_with_outcome: int = Field(default=0, ge=0)
    training_value: float | None = None
    degraded: bool = False
    drift: DriftReportV1 | None = None

    def to_csv(self) -> bytes:
        frame = pd.DataFrame(
            [
                {
                    "fila_original": case.row_number,
                    **({self.id_column: case.case_id} if self.id_column else {}),
                    "prediccion": case.prediction,
                    **(
                        {"seguridad": round(case.confidence, 3), "revisar": case.review}
                        if case.confidence is not None
                        else {}
                    ),
                    **(
                        {"desde": round(case.low, 3), "hasta": round(case.high, 3)}
                        if case.low is not None and case.high is not None
                        else {}
                    ),
                }
                for case in self.cases
            ]
        )
        return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


class ComparedPredictionMetricV1(BaseModel):
    run_id: str
    metric: str
    value: float


class PredictionComparisonV1(BaseModel):
    """Comparación justa de dos modelos del mismo tipo sobre los mismos resultados."""

    schema_version: str = "prediction-comparison.v1"
    task_kind: str
    higher_is_better: bool
    current: ComparedPredictionMetricV1
    candidate: ComparedPredictionMetricV1
    better: Literal["CURRENT", "CANDIDATE", "TIE"]
    input_sha256: str


# Por debajo de esta seguridad, una categoría se marca para revisión humana (V2-15).
DOUBT_CONFIDENCE = 0.5

# Caída relativa de la métrica principal que se considera degradación del modelo en uso.
DEGRADATION_TOLERANCE = 0.2


class ScoringService:
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

    def _artifact(self, relative: str) -> Path:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise RunBlockedError("INVALID_INTERNAL_PATH", "La ruta privada no es válida")
        root = self.storage_root.resolve()
        path = (root / Path(pure)).resolve()
        if root not in path.parents:
            raise RunBlockedError("INVALID_INTERNAL_PATH", "La ruta privada salió del storage")
        return path

    def approved_run(
        self, run_id: str, *, _for_promotion: bool = False
    ) -> tuple[Any, Any, StrategyProposalV1]:
        """Ejecución utilizable: completada, con modelo y reporte aprobado y vigente."""

        run = self.repository.get_run(run_id)
        if run["status"] != "COMPLETED" or not run["artifact_path"]:
            raise RunBlockedError(
                "RUN_HAS_NO_MODEL", "La ejecución no terminó con un modelo recomendado"
            )
        report = self.repository.get_latest_report(run_id)
        if report is None or report["status"] not in {"APPROVED", "PUBLISHED"}:
            raise RunBlockedError(
                "MODEL_NOT_APPROVED_FOR_USE",
                "Una persona debe aprobar el reporte de este modelo antes de usarlo",
            )
        if not has_publication_approval(self.repository.connection, report["id"]):
            raise RunBlockedError(
                "MODEL_NOT_CURRENT",
                "El modelo corresponde a una versión superada del contexto o la estrategia",
            )
        if not _for_promotion:
            project_id = self.repository.get_run_project_id(run_id)
            cycle = self.repository.connection.execute(
                "SELECT capability_id,origin FROM capability_cycles WHERE project_id=?",
                (project_id,),
            ).fetchone()
            if cycle is not None:
                assignment = self.repository.connection.execute(
                    "SELECT p.content_json FROM capability_model_assignments a "
                    "JOIN policy_versions p ON p.id=a.policy_version_id "
                    "WHERE a.capability_id=? ORDER BY a.rowid DESC LIMIT 1",
                    (cycle["capability_id"],),
                ).fetchone()
                assigned_run = (
                    None if assignment is None else json.loads(assignment["content_json"])["run_id"]
                )
                if (cycle["origin"] == "update" and assigned_run != run_id) or (
                    assigned_run is not None and assigned_run != run_id
                ):
                    raise RunBlockedError(
                        "MODEL_NOT_ACTIVE", "Esta versión aún no es la vigente de la capacidad"
                    )
        strategy = self.repository.get_strategy(run["strategy_version_id"])
        proposal = StrategyProposalV1.model_validate(json.loads(strategy["strategy_json"]))
        return run, report, proposal

    def score(
        self, run_id: str, *, filename: str, content: bytes, actor_label: str
    ) -> ScoringResultV1:
        if not actor_label or not actor_label.strip():
            raise RunBlockedError("ACTOR_REQUIRED", "Indica quién usa el modelo para auditarlo")
        result = self.evaluate(run_id, filename=filename, content=content)
        self.repository.record_scoring(
            run_id=run_id,
            actor_label=actor_label,
            detail={
                "report_id": result.report_id,
                "rows_scored": result.rows_scored,
                "alerts": result.alerts,
                "seen_in_training": result.seen_in_training,
                "threshold": result.threshold,
                "input_sha256": result.input_sha256,
                "result_sha256": canonical_hash(result),
                "follow_up": None if result.follow_up is None else result.follow_up.model_dump(),
                "drift": None if result.drift is None else result.drift.model_dump(),
            },
        )
        return result

    def compare(
        self,
        *,
        current_run_id: str,
        candidate_run_id: str,
        filename: str,
        content: bytes,
        actor_label: str,
    ) -> tuple[ScoringResultV1, ScoringResultV1]:
        """MON-03: ambos modelos puntúan exactamente el mismo archivo reciente.

        No cuenta como uso de ningún modelo (no entra en su salud); se audita una sola vez en
        el modelo candidato con los aciertos agregados de ambos.
        """

        if not actor_label or not actor_label.strip():
            raise RunBlockedError("ACTOR_REQUIRED", "Indica quién compara los modelos")
        current = self.evaluate(current_run_id, filename=filename, content=content)
        candidate = self.evaluate(candidate_run_id, filename=filename, content=content)
        if current.follow_up is None or candidate.follow_up is None:
            raise RunBlockedError(
                "COMPARISON_NEEDS_OUTCOMES",
                "Para comparar, el archivo debe traer el resultado real de casos que ninguno de "
                "los dos modelos usó al entrenar",
            )
        self.repository.record_model_comparison(
            run_id=candidate_run_id,
            actor_label=actor_label,
            detail={
                "current_run_id": current_run_id,
                "input_sha256": candidate.input_sha256,
                "current": current.follow_up.model_dump(),
                "candidate": candidate.follow_up.model_dump(),
            },
        )
        return current, candidate

    def compare_predictions(
        self,
        *,
        current_run_id: str,
        candidate_run_id: str,
        filename: str,
        content: bytes,
        actor_label: str,
    ) -> PredictionComparisonV1:
        """Compara varias clases o cantidades sin contar la evaluación como uso."""

        from core.ml_engine.models import TaskEvaluationV1

        if not actor_label or not actor_label.strip():
            raise RunBlockedError("ACTOR_REQUIRED", "Indica quién compara los modelos")
        current_run, _, current_proposal = self.approved_run(current_run_id)
        candidate_run, _, candidate_proposal = self.approved_run(candidate_run_id)
        if current_proposal.task_type != candidate_proposal.task_type:
            raise RunBlockedError(
                "COMPARISON_TASK_KIND_MISMATCH",
                "Los dos modelos deben ser del mismo tipo para poder compararlos",
            )
        if current_proposal.task_type == "binary_classification":
            raise RunBlockedError(
                "USE_COMPARE_FOR_BINARY", "Los modelos sí/no se comparan con «compare»"
            )

        current_evaluation = TaskEvaluationV1.model_validate(
            json.loads(current_run["candidate_metrics_json"])
        )
        candidate_evaluation = TaskEvaluationV1.model_validate(
            json.loads(candidate_run["candidate_metrics_json"])
        )
        if (
            current_evaluation.primary_metric != candidate_evaluation.primary_metric
            or current_evaluation.higher_is_better != candidate_evaluation.higher_is_better
        ):
            raise RunBlockedError(
                "COMPARISON_METRIC_MISMATCH",
                "Los dos modelos deben usar la misma métrica principal para decidir cuál es mejor",
            )

        current_prediction = self.evaluate_prediction(
            current_run_id, filename=filename, content=content
        )
        candidate_prediction = self.evaluate_prediction(
            candidate_run_id, filename=filename, content=content
        )
        if current_prediction.follow_up is None or candidate_prediction.follow_up is None:
            raise RunBlockedError(
                "COMPARISON_NEEDS_OUTCOMES",
                "Para comparar, el archivo debe traer el resultado real",
            )
        metric = current_evaluation.primary_metric
        current_value = current_prediction.follow_up[metric]
        candidate_value = candidate_prediction.follow_up[metric]
        if current_value == candidate_value:
            better: Literal["CURRENT", "CANDIDATE", "TIE"] = "TIE"
        elif (candidate_value > current_value) == current_evaluation.higher_is_better:
            better = "CANDIDATE"
        else:
            better = "CURRENT"
        result = PredictionComparisonV1(
            task_kind=current_proposal.task_type,
            higher_is_better=current_evaluation.higher_is_better,
            current=ComparedPredictionMetricV1(
                run_id=current_run_id, metric=metric, value=current_value
            ),
            candidate=ComparedPredictionMetricV1(
                run_id=candidate_run_id, metric=metric, value=candidate_value
            ),
            better=better,
            input_sha256=current_prediction.input_sha256,
        )
        self.repository.record_model_comparison(
            run_id=candidate_run_id,
            actor_label=actor_label,
            detail=result.model_dump(),
        )
        return result

    def predict(
        self, run_id: str, *, filename: str, content: bytes, actor_label: str
    ) -> PredictionResultV1:
        """Varias clases o cantidades: predice cada fila con un modelo aprobado y vigente."""

        if not actor_label or not actor_label.strip():
            raise RunBlockedError("ACTOR_REQUIRED", "Indica quién usa el modelo para auditarlo")
        result = self.evaluate_prediction(run_id, filename=filename, content=content)
        self.repository.record_scoring(
            run_id=run_id,
            actor_label=actor_label,
            detail={
                "report_id": result.report_id,
                "task_kind": result.task_kind,
                "primary_metric": result.primary_metric,
                "higher_is_better": result.higher_is_better,
                "rows_scored": result.rows_scored,
                "rows_with_outcome": result.rows_with_outcome,
                "input_sha256": result.input_sha256,
                "follow_up": result.follow_up,
                "training_value": result.training_value,
                "degraded": result.degraded,
                "drift": None if result.drift is None else result.drift.model_dump(),
            },
        )
        return result

    def evaluate_prediction(
        self, run_id: str, *, filename: str, content: bytes
    ) -> PredictionResultV1:
        """Predice varias clases o cantidades sin registrar el archivo como uso."""

        from core.ml_engine.models import TaskEvaluationV1
        from core.ml_engine.task_kernels import _classification_metrics, _regression_metrics

        run, report, proposal = self.approved_run(run_id)
        if proposal.task_type == "binary_classification":
            raise RunBlockedError("USE_SCORE_FOR_BINARY", "Un modelo sí/no se usa con «score»")
        manifest = json.loads(run["manifest_json"])
        path = self._artifact(str(run["artifact_path"]))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("artifact_sha256"):
            raise RunBlockedError(
                "ARTIFACT_INTEGRITY_FAILED", "El archivo del modelo no coincide con su registro"
            )
        evaluation = TaskEvaluationV1.model_validate(json.loads(run["candidate_metrics_json"]))
        validated = validate_csv_bytes(filename, content, self.limits)
        frame = read_validated_frame(validated)
        missing = [column for column in proposal.feature_columns if column not in frame.columns]
        if missing:
            raise MissingFeatureColumnsError(missing, [str(column) for column in frame.columns])
        pipeline = joblib.load(path)  # artefacto propio, verificado por hash; nunca del usuario
        numeric_features = _numeric_features(pipeline)
        _require_numeric_values(frame, numeric_features)
        predictions = get_task_kind(proposal.task_type).score(pipeline, frame, proposal)
        id_column = next(
            (
                column
                for column in frame.columns
                if column not in proposal.feature_columns
                and column != proposal.target_column
                and frame[column].notna().all()
                and frame[column].is_unique
            ),
            None,
        )
        regression = proposal.task_type == "regression"
        confidence: list[float | None] = [None] * len(predictions)
        if not regression and hasattr(pipeline, "predict_proba"):
            confidence = [
                float(value)
                for value in pipeline.predict_proba(frame[proposal.feature_columns]).max(axis=1)
            ]
        typical_error = evaluation.test.get("mae") if regression else None
        cases = [
            PredictedCase(
                row_number=index + 1,
                case_id=None if id_column is None else str(frame[id_column].iloc[index]),
                prediction=float(value) if regression else str(value),
                confidence=confidence[index],
                low=None if typical_error is None else float(value) - typical_error,
                high=None if typical_error is None else float(value) + typical_error,
                review=confidence[index] is not None and confidence[index] < DOUBT_CONFIDENCE,
            )
            for index, value in enumerate(predictions)
        ]
        follow_up = None
        rows_with_outcome = 0
        degraded = False
        training_value = evaluation.test.get(evaluation.primary_metric)
        target_present = proposal.target_column in frame.columns
        if target_present:
            known = frame[proposal.target_column].notna().to_numpy()
            if known.any():
                rows_with_outcome = int(known.sum())
                truth = frame[proposal.target_column][known]
                guessed = np.asarray(predictions)[known]
                follow_up = (
                    _regression_metrics(pd.to_numeric(truth), guessed.astype(float))
                    if regression
                    else _classification_metrics(truth.astype(str), guessed.astype(str))
                )
                current = follow_up[evaluation.primary_metric]
                if training_value is not None:
                    change = (current - training_value) / max(abs(training_value), 1e-9)
                    degraded = (
                        change < -DEGRADATION_TOLERANCE
                        if evaluation.higher_is_better
                        else change > DEGRADATION_TOLERANCE
                    )
        training = self._training_frame(proposal)
        drift = None
        if training is not None:
            drift = compute_drift(
                training[proposal.feature_columns],
                frame[proposal.feature_columns],
                numeric_columns=numeric_features,
                categorical_columns=[
                    column for column in proposal.feature_columns if column not in numeric_features
                ],
            )
        result = PredictionResultV1(
            run_id=run_id,
            report_id=str(report["id"]),
            task_kind=proposal.task_type,
            selected_model=str(run["selected_model"]),
            primary_metric=evaluation.primary_metric,
            higher_is_better=evaluation.higher_is_better,
            id_column=id_column,
            rows_scored=len(cases),
            input_sha256=hashlib.sha256(content).hexdigest(),
            cases=cases,
            target_present=target_present,
            follow_up=follow_up,
            rows_with_outcome=rows_with_outcome,
            training_value=training_value,
            degraded=degraded,
            drift=drift,
        )
        return result

    def evaluate(
        self, run_id: str, *, filename: str, content: bytes, _candidate_internal: bool = False
    ) -> ScoringResultV1:
        """Puntúa un archivo con un modelo aprobado y vigente, sin registrar el uso."""

        if _candidate_internal:
            run = self.repository.get_run(run_id)
            if run["status"] != "COMPLETED" or not run["artifact_path"]:
                raise RunBlockedError("RUN_HAS_NO_MODEL", "La candidata no terminó con modelo")
            strategy = self.repository.get_strategy(run["strategy_version_id"])
            if not has_training_approval(self.repository.connection, str(strategy["id"])):
                raise RunBlockedError("TRAINING_NOT_APPROVED", "Falta aprobación de entrenamiento")
            report = self.repository.get_latest_report(run_id)
            if report is None:
                raise RunBlockedError("CANDIDATE_REPORT_REQUIRED", "Crea el reporte de candidata")
            proposal = StrategyProposalV1.model_validate(json.loads(strategy["strategy_json"]))
        else:
            run, report, proposal = self.approved_run(run_id)
        manifest = json.loads(run["manifest_json"])
        path = self._artifact(str(run["artifact_path"]))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("artifact_sha256"):
            raise RunBlockedError(
                "ARTIFACT_INTEGRITY_FAILED", "El archivo del modelo no coincide con su registro"
            )
        evaluation = EvaluationSummaryV1.model_validate(json.loads(run["candidate_metrics_json"]))
        threshold = evaluation.selection.selected_threshold
        if threshold is None:
            raise RunBlockedError("RUN_HAS_NO_THRESHOLD", "La ejecución no fijó un punto de corte")

        validated = validate_csv_bytes(filename, content, self.limits)
        frame = read_validated_frame(validated)
        missing = [column for column in proposal.feature_columns if column not in frame.columns]
        if missing:
            raise MissingFeatureColumnsError(missing, [str(column) for column in frame.columns])
        pipeline = joblib.load(path)  # artefacto propio, verificado por hash; nunca del usuario
        numeric_features = _numeric_features(pipeline)
        _require_numeric_values(frame, numeric_features)
        try:
            probability = get_task_kind(proposal.task_type).score(pipeline, frame, proposal)
        except MLValidationError as error:
            raise RunBlockedError(error.code, str(error)) from error

        id_column = next(
            (
                column
                for column in frame.columns
                if column not in proposal.feature_columns
                and column != proposal.target_column
                and frame[column].notna().all()
                and frame[column].is_unique
            ),
            None,
        )
        training = self._training_frame(proposal)
        seen = _seen_in_training(frame, training, proposal)
        drift = None
        if training is not None:
            drift = compute_drift(
                training[proposal.feature_columns],
                frame[proposal.feature_columns],
                numeric_columns=numeric_features,
                categorical_columns=[
                    column for column in proposal.feature_columns if column not in numeric_features
                ],
            )
        order = np.argsort(-probability, kind="stable")
        explained = _explain_top(pipeline, frame, proposal, training, order)
        cases = [
            ScoredCase(
                rank=rank,
                row_number=int(index) + 1,
                case_id=None if id_column is None else str(frame[id_column].iloc[index]),
                score=float(probability[index]),
                alert=bool(probability[index] >= threshold),
                seen_in_training=bool(seen[index]),
                reasons=explained.get(int(index)),
            )
            for rank, index in enumerate(order, start=1)
        ]
        follow_up = _follow_up(frame, proposal, probability, threshold, evaluation, seen)
        result = ScoringResultV1(
            run_id=run_id,
            report_id=str(report["id"]),
            selected_model=str(run["selected_model"]),
            positive_class=proposal.positive_class,
            threshold=threshold,
            id_column=id_column,
            rows_scored=len(cases),
            alerts=sum(case.alert for case in cases),
            input_sha256=hashlib.sha256(content).hexdigest(),
            cases=cases,
            ignored_columns=[
                column
                for column in frame.columns
                if column not in proposal.feature_columns and column != id_column
            ],
            target_present=proposal.target_column in frame.columns,
            follow_up=follow_up,
            seen_in_training=int(seen.sum()),
            drift=drift,
        )
        return result

    def health(self, run_id: str) -> ModelHealthV1:
        """MON-02: salud del modelo según sus usos registrados (sólo lectura)."""

        run = self.repository.get_run(run_id)
        evaluation = json.loads(run["candidate_metrics_json"] or "{}")
        holdout = evaluation.get("holdout") or {}
        task_kind = evaluation.get("task_kind")
        primary_metric = evaluation.get("primary_metric") if task_kind else None
        test_metrics = evaluation.get("test") or {}
        events = [
            (str(event["created_at"]), json.loads(event["detail_json"]))
            for event in reversed(self.repository.list_scorings(run_id))
        ]
        return assess_model_health(
            events,
            reference_precision=holdout.get("precision"),
            reference_recall=holdout.get("recall"),
            reference_metric=primary_metric,
            reference_value=test_metrics.get(primary_metric) if primary_metric else None,
        )

    def _training_frame(self, proposal: StrategyProposalV1) -> pd.DataFrame | None:
        """Archivo aprobado para entrenar, sólo si sigue disponible e íntegro."""

        try:
            dataset = self.repository.get_dataset(proposal.dataset_id)
            content = self._artifact(str(dataset["stored_path"])).read_bytes()
            if hashlib.sha256(content).hexdigest() != proposal.dataset_sha256:
                return None
            return read_validated_frame(
                validate_csv_bytes(str(dataset["original_filename_display"]), content, self.limits)
            )
        except (OSError, LookupError, RunBlockedError, ValueError):
            return None


def _seen_in_training(
    frame: pd.DataFrame, training: pd.DataFrame | None, proposal: StrategyProposalV1
) -> np.ndarray:
    """Marca filas idénticas a alguna del archivo aprobado (sin mirar el resultado).

    Compara las columnas presentes en ambos archivos, excepto el resultado. Si el archivo
    original no está disponible o cambió, no marca nada: este control sólo advierte y
    nunca bloquea el uso.
    """

    unseen = np.zeros(len(frame), dtype=bool)
    if training is None:
        return unseen
    columns = [
        column
        for column in frame.columns
        if column in training.columns and column != proposal.target_column
    ]
    if not columns:
        return unseen

    def row_keys(source: pd.DataFrame) -> pd.Series:
        return pd.util.hash_pandas_object(source[columns].astype(str), index=False)

    return row_keys(frame).isin(set(row_keys(training))).to_numpy()


def _numeric_features(pipeline: Any) -> list[str]:
    """Columnas que el modelo aprendió como números, leídas del preprocesador ajustado."""

    preprocessor = pipeline.named_steps.get("preprocessor")
    return [
        str(column)
        for name, _, columns in getattr(preprocessor, "transformers_", [])
        if name == "numeric"
        for column in columns
    ]


def _require_numeric_values(frame: pd.DataFrame, columns: list[str]) -> None:
    """Bloquea con un mensaje claro si una columna numérica trae texto (p. ej. «n/d»)."""

    problems = []
    for column in columns:
        values = frame[column]
        invalid = values.notna() & pd.to_numeric(values, errors="coerce").isna()
        if invalid.any():
            problems.append(f"{column} ({int(invalid.sum())} filas)")
    if problems:
        raise RunBlockedError(
            "NON_NUMERIC_FEATURE_VALUES",
            "Hay texto donde el modelo espera números: " + ", ".join(problems),
        )


def _follow_up(
    frame: pd.DataFrame,
    proposal: StrategyProposalV1,
    probability: np.ndarray,
    threshold: float,
    evaluation: EvaluationSummaryV1,
    seen: np.ndarray | None = None,
) -> FollowUpMetrics | None:
    if proposal.target_column not in frame.columns:
        return None
    outcome = frame[proposal.target_column]
    known = outcome.notna().to_numpy()
    if seen is not None:
        known = known & ~seen
    if not known.any():
        return None
    positive = json.dumps(proposal.positive_class, default=str)
    holdout = evaluation.holdout
    allowed = {positive}
    if holdout is not None:
        allowed.add(json.dumps(holdout.negative_class, default=str))
    observed = [json.dumps(value, default=str) for value in outcome[known]]
    unexpected = sum(value not in allowed for value in observed)
    if unexpected:
        # Un valor desconocido no puede contarse como «no ocurrió»: distorsionaría el seguimiento.
        raise RunBlockedError(
            "UNKNOWN_OUTCOME_VALUES",
            f"{unexpected} filas traen un resultado distinto de los dos valores del entrenamiento",
        )
    actual = np.asarray([value == positive for value in observed])
    alert = probability[known] >= threshold
    hits = int(np.sum(alert & actual))
    precision = hits / int(alert.sum()) if alert.any() else 0.0
    recall = hits / int(actual.sum()) if actual.any() else 0.0
    metric_total = precision + recall
    true_negatives = int(np.sum(~alert & ~actual))
    negatives = int((~actual).sum())
    average_precision = None
    roc_auc = None
    if actual.any() and not actual.all():
        average_precision = float(average_precision_score(actual, probability[known]))
        roc_auc = float(roc_auc_score(actual, probability[known]))
    return FollowUpMetrics(
        rows_with_outcome=int(known.sum()),
        prevalence=float(actual.mean()),
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / metric_total if metric_total else 0.0,
        balanced_accuracy=(recall + true_negatives / negatives) / 2 if negatives else None,
        average_precision=average_precision,
        roc_auc=roc_auc,
        training_average_precision=None if holdout is None else holdout.average_precision,
    )


def _explain_top(
    pipeline: Any,
    frame: pd.DataFrame,
    proposal: StrategyProposalV1,
    training: pd.DataFrame | None,
    order: np.ndarray,
) -> dict[int, str]:
    """V2-16: razones de los casos de mayor puntaje, con valores típicos del entrenamiento."""

    if training is None:
        return {}
    from core.ml_engine.explain import (
        MAX_EXPLAINED_CASES,
        case_reasons,
        reasons_text,
        typical_values,
    )

    top = [int(index) for index in order[:MAX_EXPLAINED_CASES]]
    columns = list(proposal.feature_columns)
    kind = get_task_kind(proposal.task_type)
    reasons = case_reasons(
        lambda cases: kind.score(pipeline, cases, proposal),
        frame.iloc[top][columns].reset_index(drop=True),
        columns,
        typical_values(training, columns),
    )
    return {index: reasons_text(items) for index, items in zip(top, reasons, strict=True)}
