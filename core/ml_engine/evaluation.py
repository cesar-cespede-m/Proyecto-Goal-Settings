"""Evaluación ML-02: validation para seleccionar y holdout una sola vez."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from core.domain.canonical import canonical_json
from core.domain.context import PrimaryMetric
from core.domain.strategy import StrategyProposalV1
from core.ml_engine.errors import MLValidationError
from core.ml_engine.models import (
    BinaryMetricSet,
    BootstrapSummary,
    CandidateEvaluation,
    EvaluationSummaryV1,
    FeatureImportance,
    MetricInterval,
    PolicyOperatingPointV1,
    PolicySelectionArtifactV1,
    ProbabilityDiagnosticsV1,
    ReliabilityBinV1,
    SelectionSummary,
    ThresholdSource,
    top_fraction_count,
)
from core.ml_engine.training import TrainingSession

_EPSILON = 1e-12
_SIMPLICITY_ORDER = {"LogisticRegression": 0, "RandomForestClassifier": 1}
_BASELINE_THRESHOLD = 0.5
# Por debajo de este soporte positivo, una sola fila mueve recall/precision más de un punto.
_LOW_POSITIVE_SUPPORT = 100
_BOOTSTRAP_RESAMPLES = 1000


@dataclass(frozen=True)
class EvaluationSession:
    summary: EvaluationSummaryV1
    selected_pipeline: Pipeline | None


@dataclass(frozen=True)
class _Scored:
    y_true: np.ndarray
    probability: np.ndarray
    negative_class: Any


def _python_scalar(value: Any) -> str | int | float | bool:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str | bool | int | float):
        return value
    raise MLValidationError("UNSUPPORTED_TARGET_VALUE", "El target contiene un valor no soportado")


def _typed_key(value: Any) -> str:
    scalar = _python_scalar(value)
    return f"{type(scalar).__name__}:{canonical_json(scalar)}"


def _binary_target(y: pd.Series, positive_class: Any) -> tuple[np.ndarray, Any]:
    positive_key = _typed_key(positive_class)
    values = [_python_scalar(value) for value in y]
    unique = {_typed_key(value): value for value in values}
    if positive_key not in unique or len(unique) != 2:
        raise MLValidationError(
            "EVALUATION_TARGET_MISMATCH",
            "La partición no contiene exactamente la clase positiva y una negativa",
        )
    negative = next(value for key, value in unique.items() if key != positive_key)
    encoded = np.asarray([int(_typed_key(value) == positive_key) for value in values])
    return encoded, negative


def _positive_probability(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    positive_class: Any,
) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    class_keys = [_typed_key(value) for value in classifier.classes_]
    positive_key = _typed_key(positive_class)
    if positive_key not in class_keys:
        raise MLValidationError(
            "ESTIMATOR_POSITIVE_CLASS_MISMATCH",
            "El estimador no contiene la clase positiva aprobada",
        )
    return pipeline.predict_proba(frame)[:, class_keys.index(positive_key)]


def _score(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    target: pd.Series,
    positive_class: Any,
) -> _Scored:
    y_true, negative_class = _binary_target(target, positive_class)
    return _Scored(
        y_true=y_true,
        probability=_positive_probability(pipeline, frame, positive_class),
        negative_class=negative_class,
    )


def _unit_interval(value: Any) -> float:
    """Absorbe únicamente deriva de coma flotante alrededor de [0, 1]."""

    return min(1.0, max(0.0, float(value)))


def _threshold_metric(metric: str, y_true: np.ndarray, prediction: np.ndarray) -> float:
    """Métricas de umbral desde conteos; vectorizado para que el bootstrap sea barato."""

    positive = y_true == 1
    true_positive = int(np.sum(prediction & positive))
    false_positive = int(np.sum(prediction & ~positive))
    false_negative = int(np.sum(~prediction & positive))
    true_negative = int(np.sum(~prediction & ~positive))
    precision = true_positive / (true_positive + false_positive) if prediction.any() else 0.0
    recall = true_positive / (true_positive + false_negative) if positive.any() else 0.0
    if metric == "precision":
        return precision
    if metric == "recall":
        return recall
    if metric == "f1":
        total = precision + recall
        return 2 * precision * recall / total if total > 0 else 0.0
    if metric == "balanced_accuracy":
        negatives = true_negative + false_positive
        specificity = true_negative / negatives if negatives else 0.0
        return (recall + specificity) / 2
    raise ValueError(f"Métrica de umbral desconocida: {metric}")


def _metric(metric: str, y_true: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    if metric == "average_precision":
        return _unit_interval(average_precision_score(y_true, probability))
    if metric == "roc_auc":
        return _unit_interval(roc_auc_score(y_true, probability))
    return _unit_interval(_threshold_metric(metric, y_true, probability >= threshold))


def _metric_set(
    scored: _Scored,
    positive_class: Any,
    *,
    threshold: float,
    threshold_source: ThresholdSource,
) -> BinaryMetricSet:
    y_true, probability = scored.y_true, scored.probability
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    values = {
        metric.value: _metric(metric.value, y_true, probability, threshold)
        for metric in PrimaryMetric
    }
    return BinaryMetricSet(
        threshold=_unit_interval(threshold),
        threshold_source=threshold_source,
        positive_class=_python_scalar(positive_class),
        negative_class=scored.negative_class,
        support=len(y_true),
        positive_support=int(y_true.sum()),
        negative_support=int((1 - y_true).sum()),
        confusion_matrix=matrix.astype(int).tolist(),
        **values,
    )


def _max_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Umbral de validation con mayor F1; ante empate, el menor (favorece recall).

    Se devuelve el punto medio entre ese puntaje y el puntaje inmediatamente inferior:
    validation queda idéntica, pero el corte no queda pegado al positivo más bajo observado.
    """

    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    best = float(thresholds[int(np.argmax(f1))])
    below = probability[probability < best]
    return (best + float(below.max())) / 2 if below.size else best


def _candidate_threshold(
    scored: _Scored,
    proposal: StrategyProposalV1,
) -> tuple[float, ThresholdSource]:
    policy = proposal.threshold_policy
    if policy.method == "fixed" and policy.fixed_value is not None:
        return policy.fixed_value, "fixed"
    return _max_f1_threshold(scored.y_true, scored.probability), "validation_max_f1"


def _metric_value(metrics: BinaryMetricSet, metric: PrimaryMetric) -> float:
    return float(getattr(metrics, metric.value))


def _failed_guardrails(
    metrics: BinaryMetricSet,
    proposal: StrategyProposalV1,
) -> list[str]:
    failed: list[str] = []
    for rule in proposal.guardrails.rules:
        actual = _metric_value(metrics, rule.metric)
        if actual + _EPSILON < rule.threshold:
            failed.append(f"{rule.metric.value}>={rule.threshold}")
    return failed


def _secondary_metric(primary: PrimaryMetric) -> str:
    return "balanced_accuracy" if primary is PrimaryMetric.F1 else "f1"


def _permutation_importances(
    pipeline: Pipeline,
    training: TrainingSession,
    proposal: StrategyProposalV1,
    *,
    repeats: int,
) -> list[FeatureImportance]:
    """Usa average precision: no depende del umbral y no se anula con clases raras."""

    def scorer(estimator: Pipeline, frame: pd.DataFrame, target: pd.Series) -> float:
        scored = _score(estimator, frame, target, proposal.positive_class)
        return float(average_precision_score(scored.y_true, scored.probability))

    result = permutation_importance(
        pipeline,
        training.splits.X_validation,
        training.splits.y_validation,
        scoring=scorer,
        n_repeats=repeats,
        random_state=proposal.seed,
        n_jobs=1,
    )
    values = [
        FeatureImportance(
            feature=feature,
            importance_mean=float(result.importances_mean[index]),
            importance_std=float(result.importances_std[index]),
        )
        for index, feature in enumerate(proposal.feature_columns)
    ]
    return sorted(values, key=lambda item: (-item.importance_mean, item.feature.casefold()))


def _bootstrap_holdout(
    scored: _Scored,
    holdout: BinaryMetricSet,
    proposal: StrategyProposalV1,
    *,
    resamples: int,
) -> BootstrapSummary | None:
    metrics = list(
        dict.fromkeys([proposal.primary_metric.value, "precision", "recall", "average_precision"])
    )
    rng = np.random.default_rng(proposal.seed)
    size = len(scored.y_true)
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(resamples):
        index = rng.integers(0, size, size)
        y_true = scored.y_true[index]
        if y_true.min() == y_true.max():
            continue
        probability = scored.probability[index]
        for metric in metrics:
            samples[metric].append(_metric(metric, y_true, probability, holdout.threshold))
    used = len(samples[metrics[0]])
    if used == 0:
        return None
    return BootstrapSummary(
        seed=proposal.seed,
        resamples_requested=resamples,
        resamples_used=used,
        intervals=[
            MetricInterval(
                metric=metric,
                point=float(getattr(holdout, metric)),
                lower=_unit_interval(np.percentile(samples[metric], 2.5)),
                upper=_unit_interval(np.percentile(samples[metric], 97.5)),
            )
            for metric in metrics
        ],
    )


def _support_warnings(partition: str, metrics: BinaryMetricSet) -> list[str]:
    if metrics.positive_support < _LOW_POSITIVE_SUPPORT:
        return [f"LOW_POSITIVE_SUPPORT_{partition.upper()}"]
    return []


def evaluate_candidates(
    training: TrainingSession,
    proposal: StrategyProposalV1,
    *,
    permutation_repeats: int = 5,
    bootstrap_resamples: int = _BOOTSTRAP_RESAMPLES,
) -> EvaluationSession:
    """Selecciona solo con validation; consulta test únicamente para el modelo congelado."""

    validation: dict[str, CandidateEvaluation] = {}
    validation_scores: dict[str, _Scored] = {}
    baseline_metrics: BinaryMetricSet | None = None
    for candidate in proposal.candidates:
        scored = _score(
            training.pipelines[candidate],
            training.splits.X_validation,
            training.splits.y_validation,
            proposal.positive_class,
        )
        validation_scores[candidate] = scored
        if candidate == "DummyClassifier":
            # El baseline conserva su predicción nativa: ajustarle el umbral lo convertiría en
            # "todo positivo", que no es la referencia que se quiere superar.
            threshold, source = _BASELINE_THRESHOLD, "baseline_default"
        else:
            threshold, source = _candidate_threshold(scored, proposal)
        metrics = _metric_set(
            scored,
            proposal.positive_class,
            threshold=threshold,
            threshold_source=source,
        )
        failed = _failed_guardrails(metrics, proposal)
        primary_value = _metric_value(metrics, proposal.primary_metric)
        if candidate == "DummyClassifier":
            baseline_metrics = metrics
            beats_baseline = None
        else:
            if baseline_metrics is None:
                raise MLValidationError(
                    "BASELINE_ORDER_INVALID",
                    "DummyClassifier debe evaluarse antes que los candidatos",
                )
            beats_baseline = (
                primary_value > _metric_value(baseline_metrics, proposal.primary_metric) + _EPSILON
            )
        validation[candidate] = CandidateEvaluation(
            candidate=candidate,
            metrics=metrics,
            primary_metric_value=primary_value,
            guardrails_passed=not failed,
            failed_guardrails=failed,
            beats_baseline=beats_baseline,
        )
    if baseline_metrics is None:
        raise MLValidationError("BASELINE_MISSING", "La estrategia no produjo baseline")

    secondary = _secondary_metric(proposal.primary_metric)
    eligible = [
        candidate
        for candidate in ("LogisticRegression", "RandomForestClassifier")
        if validation[candidate].guardrails_passed and validation[candidate].beats_baseline
    ]
    selected: str | None = None
    if eligible:
        selected = max(
            eligible,
            key=lambda candidate: (
                validation[candidate].primary_metric_value,
                float(getattr(validation[candidate].metrics, secondary)),
                -_SIMPLICITY_ORDER[candidate],
            ),
        )
        outcome = "RECOMMENDED"
        reason = "BEST_VALIDATION_CANDIDATE_ABOVE_BASELINE_AND_GUARDRAILS"
    else:
        outcome = "WITHOUT_RECOMMENDATION"
        reason = "NO_CANDIDATE_ABOVE_BASELINE_AND_GUARDRAILS"

    selection = SelectionSummary(
        primary_metric=proposal.primary_metric.value,
        baseline_value=_metric_value(baseline_metrics, proposal.primary_metric),
        eligible_candidates=eligible,
        selected_model=selected,
        outcome=outcome,
        reason=reason,
        secondary_metric=secondary,
        threshold_policy=proposal.threshold_policy.method,
        selected_threshold=validation[selected].metrics.threshold if selected else None,
    )
    validation_warnings = _support_warnings("validation", baseline_metrics)
    if selected is None:
        summary = EvaluationSummaryV1(
            validation=validation,
            selection=selection,
            holdout=None,
            holdout_evaluations=0,
            permutation_importance_partition=None,
            permutation_importance_repeats=None,
            permutation_importance=[],
            warnings=sorted(
                {
                    "NO_MODEL_RECOMMENDED",
                    "HOLDOUT_NOT_EVALUATED_WITHOUT_RECOMMENDATION",
                    *validation_warnings,
                }
            ),
        )
        return EvaluationSession(summary=summary, selected_pipeline=None)

    selected_pipeline = training.pipelines[selected]
    selected_validation = validation_scores[selected]
    policy_selection = PolicySelectionArtifactV1.build(
        dataset_sha256=training.summary.dataset_sha256,
        partition_sha256=training.summary.split.validation.index_sha256,
        selected_model=selected,
        positive_class=proposal.positive_class,
        points=operating_points(selected_validation.y_true, selected_validation.probability),
    )
    train_target, _negative = _binary_target(training.splits.y_train, proposal.positive_class)
    baseline_probability = float(train_target.mean())
    validation_diagnostics = probability_diagnostics(
        selected_validation.y_true,
        selected_validation.probability,
        partition="validation",
        baseline_probability=baseline_probability,
    )
    importances = _permutation_importances(
        selected_pipeline,
        training,
        proposal,
        repeats=permutation_repeats,
    )
    frozen = validation[selected].metrics
    holdout_scored = _score(
        selected_pipeline,
        training.splits.X_test,
        training.splits.y_test,
        proposal.positive_class,
    )
    holdout = _metric_set(
        holdout_scored,
        proposal.positive_class,
        threshold=frozen.threshold,
        threshold_source=frozen.threshold_source,
    )
    bootstrap = _bootstrap_holdout(
        holdout_scored,
        holdout,
        proposal,
        resamples=bootstrap_resamples,
    )
    warnings = {
        "HOLDOUT_EVALUATED_ONCE_AFTER_SELECTION",
        "PERMUTATION_IMPORTANCE_IS_NOT_CAUSAL",
        "POLICY_SELECTED_ON_VALIDATION_NOT_INDEPENDENT_EVIDENCE",
        "PROBABILITY_DIAGNOSTICS_DO_NOT_AUTHORIZE_MONETARY_USE",
        *validation_warnings,
        *_support_warnings("holdout", holdout),
    }
    if frozen.threshold_source == "validation_max_f1":
        warnings.add("THRESHOLD_TUNED_ON_VALIDATION_HOLDOUT_UNTOUCHED")
    summary = EvaluationSummaryV1(
        validation=validation,
        selection=selection,
        holdout=holdout,
        holdout_evaluations=1,
        holdout_bootstrap=bootstrap,
        permutation_importance_partition="validation",
        permutation_importance_scorer="average_precision",
        permutation_importance_repeats=permutation_repeats,
        permutation_importance=importances,
        policy_selection=policy_selection,
        probability_diagnostics=[
            validation_diagnostics,
            probability_diagnostics(
                holdout_scored.y_true,
                holdout_scored.probability,
                partition="holdout",
                baseline_probability=baseline_probability,
            ),
        ],
        warnings=sorted(warnings),
    )
    return EvaluationSession(summary=summary, selected_pipeline=selected_pipeline)


# Proporciones de casos marcados para las que se guarda qué pasaría (V2-13).
OPERATING_SHARES = (
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)


def _validate_scores(y_true: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth, scores = np.asarray(y_true), np.asarray(probability, dtype=float)
    if (
        truth.ndim != 1
        or scores.ndim != 1
        or len(truth) == 0
        or len(truth) != len(scores)
        or not np.isin(truth, [0, 1]).all()
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise MLValidationError("INVALID_POLICY_SCORES", "Los resultados y puntajes no son válidos")
    return truth.astype(int), scores


def operating_points(y_true: np.ndarray, probability: np.ndarray) -> list[PolicyOperatingPointV1]:
    """Grilla explícita por fracción; un empate se resuelve por orden original, nunca umbral."""

    y_true, probability = _validate_scores(y_true, probability)
    order = np.argsort(-probability, kind="stable")
    ordered_truth = np.asarray(y_true)[order]
    ordered_probability = np.asarray(probability)[order]
    support = len(ordered_truth)
    positives = int(ordered_truth.sum())
    points = []
    for share in OPERATING_SHARES:
        flagged = top_fraction_count(share, support)
        hits = int(ordered_truth[:flagged].sum())
        points.append(
            PolicyOperatingPointV1(
                selection_fraction=share,
                flagged_share=flagged / support,
                cutoff_score=_unit_interval(ordered_probability[flagged - 1]) if flagged else None,
                flagged=flagged,
                true_alerts=hits,
                false_alarms=flagged - hits,
                missed=positives - hits,
                support=support,
                positives=positives,
            )
        )
    return points


def probability_diagnostics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    partition: Literal["validation", "holdout"],
    baseline_probability: float,
) -> ProbabilityDiagnosticsV1:
    """Diagnóstico sin calibrar ni certificar probabilidades para valoración monetaria."""

    truth, scores = _validate_scores(y_true, probability)
    if not np.isfinite(baseline_probability) or not 0 <= baseline_probability <= 1:
        raise MLValidationError(
            "INVALID_PROBABILITY_BASELINE", "La prevalencia de train no es válida"
        )
    baseline = np.full(len(truth), baseline_probability)
    bin_indices = np.minimum((scores * 10).astype(int), 9)
    bins = []
    for index in range(10):
        inside = bin_indices == index
        if inside.any():
            bins.append(
                ReliabilityBinV1(
                    lower=index / 10,
                    upper=(index + 1) / 10,
                    support=int(inside.sum()),
                    mean_score=float(scores[inside].mean()),
                    observed_rate=float(truth[inside].mean()),
                )
            )
    positives = int(truth.sum())
    negatives = len(truth) - positives
    # Umbrales de advertencia de soporte, nunca criterios para certificar calibración.
    insufficient = min(positives, negatives) < 30 or any(item.support < 20 for item in bins)
    return ProbabilityDiagnosticsV1(
        partition=partition,
        support=len(truth),
        positive_support=positives,
        negative_support=negatives,
        brier_score=float(brier_score_loss(truth, scores)),
        log_loss=float(log_loss(truth, scores, labels=[0, 1])),
        prevalence_baseline_probability=baseline_probability,
        prevalence_baseline_brier=float(brier_score_loss(truth, baseline)),
        prevalence_baseline_log_loss=float(log_loss(truth, baseline, labels=[0, 1])),
        reliability_bins=bins,
        status="insufficient_support" if insufficient else "diagnostic_only",
        warnings=[
            "BRIER_AND_LOG_LOSS_ARE_NOT_CALIBRATION_CERTIFICATION",
            "MONETARY_PROBABILITY_USE_NOT_VERIFIED",
            *(["LOW_RELIABILITY_SUPPORT"] if insufficient else []),
        ],
    )
