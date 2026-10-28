"""GA-07a: registro de tipos de modelo.

Cada tipo de tarea (sí/no, varias clases, cantidades) declara cómo se arman sus candidatos,
cómo se evalúan y cómo se puntúa un caso nuevo. El resto del motor (partición, ejecución,
artefactos, aprobaciones) es común y no conoce el tipo. Por ahora sólo existe sí/no, con la
misma lógica que tenía el motor antes del registro.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from core.ml_engine.errors import MLValidationError
from core.ml_engine.preprocessing import build_preprocessor

if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    from core.domain.strategy import StrategyProposalV1
    from core.ml_engine.evaluation import EvaluationSession
    from core.ml_engine.training import TrainingSession

UNSUPPORTED_TASK_KIND = "UNSUPPORTED_TASK_KIND"


@dataclass(frozen=True)
class TaskOutcome:
    """Lo que la ejecución guarda de cualquier tipo de modelo (ADR-016)."""

    selected: str | None
    pipeline: Any | None
    split_info: dict[str, Any]
    summary: Any  # EvaluationSummaryV1 o TaskEvaluationV1
    warnings: list[str]
    versions: dict[str, str]


# Nombre del paso final del pipeline; los artefactos guardados dependen de él.
ESTIMATOR_STEP = "classifier"


class TaskKind(Protocol):
    name: str
    baseline: str  # candidato de referencia que todo modelo debe superar
    higher_is_better: bool  # dirección de la métrica principal

    def build_estimators(
        self,
        proposal: StrategyProposalV1,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> dict[str, tuple[ColumnTransformer, Any]]: ...

    def evaluate(
        self, training: TrainingSession, proposal: StrategyProposalV1, **options: Any
    ) -> EvaluationSession: ...

    def score(
        self, pipeline: Pipeline, frame: pd.DataFrame, proposal: StrategyProposalV1
    ) -> np.ndarray: ...

    def execute(self, frame: pd.DataFrame, proposal: StrategyProposalV1) -> TaskOutcome: ...


class BinaryClassification:
    """Sí/no: probabilidad del evento aprobado, umbral y métricas de clasificación."""

    name = "binary_classification"
    baseline = "DummyClassifier"
    higher_is_better = True

    def build_estimators(
        self,
        proposal: StrategyProposalV1,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> dict[str, tuple[ColumnTransformer, Any]]:
        dummy = proposal.model_params.dummy
        logistic = proposal.model_params.logistic_regression
        forest = proposal.model_params.random_forest
        return {
            "DummyClassifier": (
                build_preprocessor(numeric_columns, categorical_columns, scale_numeric=False),
                DummyClassifier(strategy=dummy.strategy),
            ),
            "LogisticRegression": (
                build_preprocessor(numeric_columns, categorical_columns, scale_numeric=True),
                LogisticRegression(
                    C=logistic.c,
                    max_iter=logistic.max_iter,
                    solver=logistic.solver,
                    class_weight=logistic.class_weight,
                    random_state=proposal.seed,
                ),
            ),
            "RandomForestClassifier": (
                build_preprocessor(numeric_columns, categorical_columns, scale_numeric=False),
                RandomForestClassifier(
                    n_estimators=forest.n_estimators,
                    max_depth=forest.max_depth,
                    min_samples_leaf=forest.min_samples_leaf,
                    class_weight=forest.class_weight,
                    n_jobs=forest.n_jobs,
                    random_state=proposal.seed,
                ),
            ),
        }

    def evaluate(
        self, training: TrainingSession, proposal: StrategyProposalV1, **options: Any
    ) -> EvaluationSession:
        from core.ml_engine.evaluation import evaluate_candidates

        return evaluate_candidates(training, proposal, **options)

    def execute(self, frame: pd.DataFrame, proposal: StrategyProposalV1) -> TaskOutcome:
        from core.ml_engine.training import train_candidates

        training = train_candidates(frame, proposal)
        evaluation = self.evaluate(training, proposal)
        return TaskOutcome(
            selected=evaluation.summary.selection.selected_model,
            pipeline=evaluation.selected_pipeline,
            split_info=training.summary.split.model_dump(mode="json"),
            summary=evaluation.summary,
            warnings=[*training.summary.warnings, *evaluation.summary.warnings],
            versions={
                **training.summary.package_versions,
                "training_summary": training.summary.schema_version,
                "evaluation_summary": evaluation.summary.schema_version,
            },
        )

    def score(
        self, pipeline: Pipeline, frame: pd.DataFrame, proposal: StrategyProposalV1
    ) -> np.ndarray:
        """Probabilidad del evento aprobado para cada caso."""

        estimator = pipeline.named_steps[ESTIMATOR_STEP]
        classes = [json.dumps(value, default=str) for value in estimator.classes_.tolist()]
        positive_key = json.dumps(proposal.positive_class, default=str)
        if positive_key not in classes:
            raise MLValidationError(
                "ESTIMATOR_POSITIVE_CLASS_MISMATCH", "El modelo no contiene el evento aprobado"
            )
        return pipeline.predict_proba(frame[proposal.feature_columns])[
            :, classes.index(positive_key)
        ]


class _KernelTask:
    """Varias clases y cantidades sobre los kernels de `task_kernels.py`."""

    name = ""
    baseline = ""
    higher_is_better = True

    def _train(self, frame: pd.DataFrame, proposal: StrategyProposalV1) -> Any:
        raise NotImplementedError

    def build_estimators(self, proposal, numeric_columns, categorical_columns):
        raise MLValidationError(UNSUPPORTED_TASK_KIND, "Este tipo arma sus candidatos en el kernel")

    def evaluate(self, training, proposal, **options):
        raise MLValidationError(UNSUPPORTED_TASK_KIND, "Este tipo se evalúa dentro del kernel")

    def score(self, pipeline: Pipeline, frame: pd.DataFrame, proposal: StrategyProposalV1):
        return pipeline.predict(frame[proposal.feature_columns])

    def execute(self, frame: pd.DataFrame, proposal: StrategyProposalV1) -> TaskOutcome:
        from importlib.metadata import version

        from core.ml_engine.models import TaskEvaluationV1

        result = self._train(frame, proposal)
        summary = TaskEvaluationV1(
            task_kind=self.name,
            primary_metric=result.primary_metric,
            higher_is_better=result.higher_is_better,
            baseline=result.baseline,
            validation=result.validation,
            selected_model=result.selected,
            test=result.test,
            split_method=result.split_method,
            split_rows=result.split_rows,
            holdout_evaluations=1 if result.selected else 0,
            warnings=sorted({*proposal.warnings, *result.warnings}),
        )
        return TaskOutcome(
            selected=result.selected,
            pipeline=result.pipeline,
            split_info={"method": result.split_method, "rows": result.split_rows},
            summary=summary,
            warnings=summary.warnings,
            versions={
                "scikit-learn": version("scikit-learn"),
                "pandas": version("pandas"),
                "evaluation_summary": summary.schema_version,
            },
        )


class MulticlassClassification(_KernelTask):
    name = "multiclass_classification"
    baseline = "DummyClassifier"

    def _train(self, frame, proposal):
        from core.ml_engine.task_kernels import train_multiclass

        return train_multiclass(
            frame,
            proposal.target_column,
            proposal.feature_columns,
            seed=proposal.seed,
            time_column=proposal.split.time_column,
        )


class Regression(_KernelTask):
    name = "regression"
    baseline = "DummyRegressor"
    higher_is_better = False

    def _train(self, frame, proposal):
        from core.ml_engine.task_kernels import train_regression

        return train_regression(
            frame,
            proposal.target_column,
            proposal.feature_columns,
            seed=proposal.seed,
            time_column=proposal.split.time_column,
        )


TASK_KINDS: dict[str, Callable[[], TaskKind]] = {
    BinaryClassification.name: BinaryClassification,
    MulticlassClassification.name: MulticlassClassification,
    Regression.name: Regression,
}


def get_task_kind(name: str) -> TaskKind:
    try:
        return TASK_KINDS[name]()
    except KeyError as error:
        raise MLValidationError(
            UNSUPPORTED_TASK_KIND, f"Este tipo de modelo todavía no está disponible: {name}"
        ) from error
