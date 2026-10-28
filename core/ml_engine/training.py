"""Entrenamiento ML-01 de tres candidatos; evaluación queda en ML-02."""

from dataclasses import dataclass
from importlib.metadata import version
from time import perf_counter

import pandas as pd
from sklearn.pipeline import Pipeline

from core.domain.strategy import StrategyProposalV1
from core.ml_engine.errors import CandidateTrainingError, MLValidationError
from core.ml_engine.models import CandidateFitSummary, TrainingSummaryV1
from core.ml_engine.preprocessing import infer_feature_types
from core.ml_engine.splitting import DatasetSplits, split_dataset
from core.ml_engine.task_kinds import ESTIMATOR_STEP, get_task_kind


@dataclass(frozen=True)
class TrainingSession:
    splits: DatasetSplits
    pipelines: dict[str, Pipeline]
    summary: TrainingSummaryV1


def _fit_candidate(
    name: str,
    pipeline: Pipeline,
    splits: DatasetSplits,
) -> CandidateFitSummary:
    started = perf_counter()
    try:
        pipeline.fit(splits.X_train, splits.y_train)
    except Exception as error:
        raise CandidateTrainingError(name, f"Falló el ajuste de {name}") from error
    return CandidateFitSummary(candidate=name, fit_seconds=perf_counter() - started)


def train_candidates(frame: pd.DataFrame, proposal: StrategyProposalV1) -> TrainingSession:
    """Kernel sin UI/LLM; no selecciona modelo ni consulta el holdout."""

    splits = split_dataset(frame, proposal)
    numeric_columns, categorical_columns = infer_feature_types(splits.X_train)
    estimators = get_task_kind(proposal.task_type).build_estimators(
        proposal, numeric_columns, categorical_columns
    )
    pipelines: dict[str, Pipeline] = {}
    fit_summaries: list[CandidateFitSummary] = []
    training_started = perf_counter()
    for candidate in proposal.candidates:
        preprocessor, estimator = estimators[candidate]
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (ESTIMATOR_STEP, estimator),
            ]
        )
        fit_summaries.append(_fit_candidate(candidate, pipeline, splits))
        pipelines[candidate] = pipeline
        if perf_counter() - training_started > proposal.caps.max_training_seconds:
            raise MLValidationError(
                "TRAINING_TIME_CAP_EXCEEDED",
                "El entrenamiento superó el cap técnico después de completar un candidato",
            )

    summary = TrainingSummaryV1(
        dataset_sha256=proposal.dataset_sha256,
        context_sha256=proposal.context_sha256,
        strategy_schema_version=proposal.schema_version,
        seed=proposal.seed,
        feature_columns=proposal.feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        split=splits.summary,
        candidates=fit_summaries,
        package_versions={
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "python": __import__("platform").python_version(),
            "scikit-learn": version("scikit-learn"),
            "scipy": version("scipy"),
        },
        warnings=sorted({*proposal.warnings, "HOLDOUT_NOT_EVALUATED_ML01"}),
    )
    return TrainingSession(splits=splits, pipelines=pipelines, summary=summary)
