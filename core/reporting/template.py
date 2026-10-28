"""Reporte determinista construido sólo con valores calculados por el motor ML."""

from typing import Any

from core.domain.models import Insight, ReportPayload
from core.ml_engine.models import EvaluationSummaryV1, TaskEvaluationV1

_LIMITATIONS = {
    "LOW_POSITIVE_SUPPORT_VALIDATION": (
        "Hubo menos de cien {event_cases} en los datos de comparación: cada caso mueve "
        "bastante las cifras."
    ),
    "LOW_POSITIVE_SUPPORT_HOLDOUT": (
        "Hubo menos de cien {event_cases} en la prueba final: lee cada cifra junto con su "
        "rango probable."
    ),
    "THRESHOLD_TUNED_ON_VALIDATION_HOLDOUT_UNTOUCHED": (
        "El modelo y el puntaje de riesgo necesario para avisar se eligieron con los datos de "
        "comparación; la prueba final no intervino en esas decisiones."
    ),
    "PERMUTATION_IMPORTANCE_IS_NOT_CAUSAL": (
        "Que una columna influya en la predicción muestra asociación, no que cause el resultado."
    ),
    "NO_MODEL_RECOMMENDED": (
        "Ningún modelo cumplió las condiciones aprobadas; no hay recomendación."
    ),
    "HOLDOUT_NOT_EVALUATED_WITHOUT_RECOMMENDATION": (
        "La prueba final no se usó porque no hubo modelo recomendado."
    ),
}
_IGNORED_WARNINGS = {
    "HOLDOUT_EVALUATED_ONCE_AFTER_SELECTION",
    "DETERMINISTIC_TEMPLATE_NO_LLM",
    "HOLDOUT_NOT_EVALUATED_ML01",
}
# V2-12: avisos del perfil del archivo, en palabras (antes se mostraban como códigos).
_PROFILE_LIMITATIONS = {
    "CONSTANT_COLUMNS": "{cols} tiene el mismo valor en casi todas las filas y no aporta.",
    "POSSIBLE_IDENTIFIER_COLUMNS": (
        "{cols} parece un identificador (distinto en cada fila); el modelo no debería usarlo."
    ),
    "HIGH_CARDINALITY_COLUMNS": "{cols} tiene muchísimos valores distintos; aporta poco.",
    "HIGH_MISSING_COLUMNS": "{cols} tiene muchos valores vacíos.",
    "DUPLICATE_ROWS": "El archivo tenía {cols} filas repetidas.",
    "POSSIBLE_PERSONAL_DATA_COLUMNS": "{cols} podría contener datos personales.",
}


def limitation_text(warning: str, *, event_cases: str = "casos del evento") -> str:
    """Traduce un aviso técnico a una frase; nunca muestra el código en mayúsculas."""

    if warning in _LIMITATIONS:
        return _LIMITATIONS[warning].format(event_cases=event_cases)
    name, _, detail = warning.removeprefix("DATA_PROFILE:").partition(":")
    if name in _PROFILE_LIMITATIONS:
        columns = ", ".join(f"«{item}»" for item in detail.split(",") if item) or "Una columna"
        return _PROFILE_LIMITATIONS[name].format(cols=columns)
    return "Hay un aviso técnico del entrenamiento; revisa el detalle técnico si lo necesitas."


_CLASSIFICATION_LIMITATION = {
    "synthetic": (
        "Los datos son inventados: el resultado no demuestra cómo funcionaría en la operación real."
    ),
    "sandbox": (
        "Los datos son de un entorno de pruebas: el resultado no demuestra cómo funcionaría en "
        "la operación real."
    ),
    "anonymized": (
        "Los datos están anonimizados: valida con el dueño de los datos antes de generalizar."
    ),
}
_ALWAYS = (
    "El modelo todavía no se usó en la operación real: es una evaluación offline, hecha con "
    "datos guardados."
)

# Nombres en lenguaje común; los nombres técnicos quedan en `metrics` (detalle técnico).
_MODEL_NAMES = {
    "DummyClassifier": "la respuesta de referencia simple",
    "LogisticRegression": "la alternativa lineal",
    "RandomForestClassifier": "la alternativa de reglas combinadas",
}
_METRIC_NAMES = {
    "recall": "casos reales detectados",
    "precision": "avisos que aciertan",
    "f1": "equilibrio entre detectar y acertar",
    "average_precision": "calidad al ordenar por riesgo",
    "balanced_accuracy": "acierto equilibrado entre ambos resultados",
    "roc_auc": "capacidad para separar ambos resultados",
}


def _fmt(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def _pct(value: float) -> str:
    """Proporción como porcentaje con coma decimal; el verificador acepta «64,7 %»."""

    return f"{value * 100:.1f}".replace(".", ",") + " %"


def event_cases_label(positive_class: Any | None, target_column: str | None = None) -> str:
    """Nombra en lenguaje común el resultado que la persona eligió anticipar."""

    if positive_class is None or not str(positive_class).strip():
        return "casos del evento"
    if isinstance(positive_class, bool):
        label = "Sí" if positive_class else "No"
    else:
        label = str(positive_class).strip()
    if target_column:
        return f"casos en que «{target_column}» fue «{label}»"
    return f"casos de «{label}»"


def build_metrics(
    *,
    run_id: str,
    run_status: str,
    evaluation: EvaluationSummaryV1,
    dataset_classification: str | None,
    target_column: str | None,
    positive_class: Any | None,
) -> dict[str, Any]:
    """Bloque `metrics` del reporte: sólo agregados ya persistidos en el run."""

    selection = evaluation.selection
    bootstrap = evaluation.holdout_bootstrap
    return {
        "run_id": run_id,
        "run_status": run_status,
        "dataset_classification": dataset_classification,
        "event_cases_label": event_cases_label(positive_class, target_column),
        "risk_score_range": {"minimum": 0.0, "maximum": 1.0},
        "primary_metric": selection.primary_metric,
        "selection": selection.model_dump(mode="json"),
        "validation": {
            name: {
                "primary_metric_value": candidate.primary_metric_value,
                "beats_baseline": candidate.beats_baseline,
                "guardrails_passed": candidate.guardrails_passed,
                **candidate.metrics.model_dump(mode="json"),
            }
            for name, candidate in evaluation.validation.items()
        },
        "holdout": evaluation.holdout.model_dump(mode="json") if evaluation.holdout else None,
        "bootstrap_confidence": bootstrap.confidence if bootstrap else None,
        "holdout_bootstrap": (
            {interval.metric: interval.model_dump(mode="json") for interval in bootstrap.intervals}
            if bootstrap
            else None
        ),
        "permutation_importance": [
            item.model_dump(mode="json") for item in evaluation.permutation_importance
        ],
        "warnings": evaluation.warnings,
    }


def _safe_use_recommendation(evidence_ref: str) -> Insight:
    return Insight(
        kind="RECOMMENDATION",
        claim=(
            "Usa los avisos como una señal para revisar los casos, no como una decisión "
            "automática, y pruébalos en paralelo con el proceso actual antes de depender "
            "de ellos."
        ),
        evidence_refs=[evidence_ref],
        limitations=[
            "Es una recomendación prudente de uso; el modelo aún no se probó en la operación."
        ],
        source="TEMPLATE",
    )


def _recommended_insights(metrics: dict[str, Any]) -> list[Insight]:
    primary = metrics["primary_metric"]
    metric_name = _METRIC_NAMES.get(primary, primary)
    selection = metrics["selection"]
    model = selection["selected_model"]
    model_name = _MODEL_NAMES.get(model, "la alternativa elegida")
    validation = metrics["validation"][model]
    holdout = metrics["holdout"]
    insights = [
        Insight(
            kind="FACT",
            claim=(
                f"Al comparar modelos, {model_name} obtuvo "
                f"{_pct(validation['primary_metric_value'])} en «{metric_name}», frente a "
                f"{_pct(selection['baseline_value'])} de la respuesta de referencia simple."
            ),
            evidence_refs=[
                f"metrics.validation.{model}.primary_metric_value",
                "metrics.selection.baseline_value",
            ],
            limitations=[
                "Estos datos de comparación también se usaron para elegir el modelo y el "
                "punto desde el que avisa."
            ],
            source="TEMPLATE",
        )
    ]
    interval = (metrics["holdout_bootstrap"] or {}).get(primary)
    holdout_claim = (
        f"En la prueba final, usada una sola vez, obtuvo {_pct(holdout[primary])} en "
        f"«{metric_name}»"
    )
    holdout_refs = [f"metrics.holdout.{primary}"]
    if interval:
        holdout_claim += (
            f" (rango probable con {round(metrics['bootstrap_confidence'] * 100)} % de "
            f"confianza: entre {_pct(interval['lower'])} y {_pct(interval['upper'])})"
        )
        holdout_refs += [
            "metrics.bootstrap_confidence",
            f"metrics.holdout_bootstrap.{primary}.lower",
            f"metrics.holdout_bootstrap.{primary}.upper",
        ]
    insights.append(
        Insight(
            kind="FACT",
            claim=holdout_claim + ".",
            evidence_refs=holdout_refs,
            limitations=["Es una sola prueba; el rango muestra cuánto podría variar la cifra."],
            source="TEMPLATE",
        )
    )
    matrix = holdout["confusion_matrix"]
    event_cases = metrics["event_cases_label"]
    insights.append(
        Insight(
            kind="FACT",
            claim=(
                "El modelo avisa cuando el puntaje de riesgo del caso llega a "
                f"{_fmt(holdout['threshold'])} o más (de 0 a 1). Con ese criterio, detectó "
                f"{matrix[1][1]} de {holdout['positive_support']} {event_cases} "
                f"({_pct(holdout['recall'])}) y dio {matrix[0][1]} falsas alarmas; de sus "
                f"avisos, acertó el {_pct(holdout['precision'])}."
            ),
            evidence_refs=[
                "metrics.holdout.threshold",
                "metrics.risk_score_range.minimum",
                "metrics.risk_score_range.maximum",
                "metrics.holdout.confusion_matrix",
                "metrics.holdout.positive_support",
                "metrics.holdout.recall",
                "metrics.holdout.precision",
                "metrics.event_cases_label",
            ],
            limitations=[
                "Avisar desde otro puntaje cambia el equilibrio entre detectar y dar falsas "
                "alarmas."
            ],
            source="TEMPLATE",
        )
    )
    importances = metrics["permutation_importance"]
    top = [(index, item) for index, item in enumerate(importances) if item["importance_mean"] > 0][
        :2
    ]
    if top:
        names = " y ".join(f"«{item['feature']}»" for _, item in top)
        insights.append(
            Insight(
                kind="INTERPRETATION",
                claim=(
                    f"En esta prueba, alterar {names} redujo en promedio la calidad al ordenar "
                    "los casos. Es una señal de asociación dentro de esta muestra."
                ),
                evidence_refs=[
                    f"metrics.permutation_importance.{index}.{field}"
                    for index, _ in top
                    for field in ("feature", "importance_mean")
                ],
                limitations=[
                    "Muestra asociación, no causa; columnas parecidas se reparten la influencia."
                ],
                source="TEMPLATE",
            )
        )
    elif importances:
        insights.append(
            Insight(
                kind="INTERPRETATION",
                claim=(
                    "Esta prueba no encontró una caída promedio de la calidad al alterar "
                    "cada columna por separado. No permite destacar una columna como más "
                    "influyente ni concluir que los datos no sirven."
                ),
                evidence_refs=[
                    f"metrics.permutation_importance.{index}.importance_mean"
                    for index in range(len(importances))
                ],
                limitations=[
                    "Depende de la muestra y del modelo; columnas parecidas pueden "
                    "compensarse entre sí."
                ],
                source="TEMPLATE",
            )
        )
    insights.append(_safe_use_recommendation("metrics.selection.selected_model"))
    return insights


def _no_recommendation_insights(metrics: dict[str, Any]) -> list[Insight]:
    primary = metrics["primary_metric"]
    metric_name = _METRIC_NAMES.get(primary, primary)
    selection = metrics["selection"]
    return [
        Insight(
            kind="FACT",
            claim=(
                "Ninguna alternativa le ganó a la respuesta de referencia simple "
                f"({_pct(selection['baseline_value'])} en «{metric_name}») cumpliendo las "
                "condiciones aprobadas; no se recomienda un modelo y la prueba final no se usó."
            ),
            evidence_refs=["metrics.selection.baseline_value", "metrics.selection.outcome"],
            limitations=["Es un resultado válido: con estos datos y columnas no alcanza hoy."],
            source="TEMPLATE",
        ),
        Insight(
            kind="RECOMMENDATION",
            claim=(
                "Revisar con el responsable si faltan datos que se conozcan antes de decidir, "
                "o si la medida elegida y la condición mínima reflejan el objetivo."
            ),
            evidence_refs=["metrics.selection.reason"],
            limitations=["Recomendación de proceso, no de modelo."],
            source="TEMPLATE",
        ),
    ]


def build_template_report(
    *,
    run_id: str,
    run_status: str,
    evaluation: EvaluationSummaryV1,
    dataset_classification: str | None,
    target_column: str | None = None,
    positive_class: Any | None = None,
) -> ReportPayload:
    metrics = build_metrics(
        run_id=run_id,
        run_status=run_status,
        evaluation=evaluation,
        dataset_classification=dataset_classification,
        target_column=target_column,
        positive_class=positive_class,
    )
    recommended = evaluation.selection.selected_model is not None
    insights = (
        _recommended_insights(metrics) if recommended else _no_recommendation_insights(metrics)
    )
    limitations: list[str] = []
    for warning in evaluation.warnings:
        if warning in _IGNORED_WARNINGS:
            continue
        limitations.append(limitation_text(warning, event_cases=metrics["event_cases_label"]))
    if dataset_classification in _CLASSIFICATION_LIMITATION:
        limitations.append(_CLASSIFICATION_LIMITATION[dataset_classification])
    limitations.append(_ALWAYS)
    return ReportPayload(
        schema_version="report.v1",
        generator="TEMPLATE",
        metrics=metrics,
        insights=insights,
        limitations=limitations,
    )


_TASK_METRIC_NAMES = {
    "macro_f1": "F1 promedio entre clases",
    "balanced_accuracy": "exactitud balanceada",
    "accuracy": "exactitud",
    "mae": "error absoluto medio",
    "rmse": "error cuadrático medio (raíz)",
    "r2": "R²",
}


def build_task_report(
    *,
    run_id: str,
    run_status: str,
    evaluation: TaskEvaluationV1,
    dataset_classification: str | None,
) -> ReportPayload:
    """Reporte por plantilla de varias clases o cantidades (ADR-016); cifras con evidencia."""

    metric = evaluation.primary_metric
    name = _TASK_METRIC_NAMES.get(metric, metric)
    selected = evaluation.selected_model
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "run_status": run_status,
        "task_kind": evaluation.task_kind,
        "primary_metric": metric,
        "higher_is_better": evaluation.higher_is_better,
        "baseline": evaluation.baseline,
        "selected_model": selected,
        "validation": evaluation.validation,
        "test": evaluation.test,
        "split": {"method": evaluation.split_method, "rows": evaluation.split_rows},
    }
    baseline_value = evaluation.validation[evaluation.baseline][metric]
    if selected is None:
        insights = [
            Insight(
                kind="FACT",
                claim=(
                    "Ninguna alternativa superó a la respuesta de referencia simple "
                    f"({_fmt(baseline_value)} en «{name}»); no se recomienda un modelo y la "
                    "prueba final no se usó."
                ),
                evidence_refs=[f"metrics.validation.{evaluation.baseline}.{metric}"],
                limitations=["Es un resultado válido: con estos datos hoy no alcanza."],
                source="TEMPLATE",
            )
        ]
    else:
        direction = "mayor es mejor" if evaluation.higher_is_better else "menor es mejor"
        insights = [
            Insight(
                kind="FACT",
                claim=(
                    f"En validación, «{selected}» logró "
                    f"{_fmt(evaluation.validation[selected][metric])} "
                    f"en «{name}» ({direction}), frente a {_fmt(baseline_value)} de la referencia "
                    "simple."
                ),
                evidence_refs=[
                    f"metrics.validation.{selected}.{metric}",
                    f"metrics.validation.{evaluation.baseline}.{metric}",
                ],
                limitations=["Validación sirve para elegir; la prueba final mide una sola vez."],
                source="TEMPLATE",
            ),
            Insight(
                kind="FACT",
                claim=(
                    "En la prueba final, nunca vista al elegir, obtuvo "
                    f"{_fmt(evaluation.test[metric])} "
                    f"en «{name}»."
                ),
                evidence_refs=[f"metrics.test.{metric}"],
                limitations=["Una sola medición: puede variar con datos nuevos."],
                source="TEMPLATE",
            ),
        ]
        insights.append(_safe_use_recommendation("metrics.selected_model"))
    limitations = [
        limitation_text(warning)
        for warning in evaluation.warnings
        if warning not in _IGNORED_WARNINGS
    ]
    if dataset_classification in _CLASSIFICATION_LIMITATION:
        limitations.append(_CLASSIFICATION_LIMITATION[dataset_classification])
    limitations.append(_ALWAYS)
    return ReportPayload(
        schema_version="report.v1",
        generator="TEMPLATE",
        metrics=metrics,
        insights=insights,
        limitations=limitations,
    )
