"""Exporta un reporte como HTML autocontenido (EXT-D, P1, FR-017).

El HTML se abre en cualquier navegador y se imprime como PDF desde ahí; no requiere
dependencias nuevas. Sólo muestra datos ya persistidos y verificados; todo texto se escapa.
"""

import json
from datetime import UTC, datetime
from html import escape
from typing import Any

from core.db.repositories import WorkflowRepository
from core.domain.context import BusinessContextV1
from core.domain.gates import has_publication_approval
from core.domain.models import ReportPayload
from core.domain.strategy import StrategyProposalV1

STATUS_TEXT = {
    "DRAFT": ("Borrador — pendiente de revisión humana", "#8a5a00"),
    "APPROVED": ("Aprobado por una persona — todavía no publicado", "#14532d"),
    "PUBLISHED": ("Publicado de forma privada tras aprobación humana", "#14532d"),
    "REJECTED": ("Rechazado por una persona", "#8e1b1b"),
    "SUPERSEDED": ("Reemplazado por una versión posterior", "#555555"),
}
METRIC_TEXT = {
    "recall": "De los eventos reales, cuántos detecta",
    "precision": "De las alertas emitidas, cuántas aciertan",
    "f1": "Equilibrio entre detectar y acertar",
    "balanced_accuracy": "Acierto equilibrado entre ambos resultados",
    "average_precision": "Calidad global de las alertas ordenadas por riesgo",
    "roc_auc": "Capacidad global para ordenar ambos resultados",
}
KIND_TEXT = {
    "FACT": "Dato comprobado",
    "INTERPRETATION": "Lectura",
    "HYPOTHESIS": "Hipótesis",
    "RECOMMENDATION": "Recomendación",
}
MODEL_TEXT = {
    "LogisticRegression": "Regresión logística (combina variables de forma lineal)",
    "RandomForestClassifier": "Bosque aleatorio (combina muchas reglas sencillas)",
    "DummyClassifier": "Modelo simple de referencia",
}
GENERATOR_TEXT = {
    "LLM": "Redactado por IA generativa y verificado contra las cifras calculadas",
    "TEMPLATE": "Plantilla sin IA generativa, construida con cifras calculadas",
}


def _e(value: Any) -> str:
    return escape(str(value), quote=True)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f} %"


def _rows(pairs: list[tuple[str, Any]]) -> str:
    return "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in pairs)


def render_report_html(repository: WorkflowRepository, report_id: str) -> str:
    report = repository.get_report(report_id)
    payload = ReportPayload.model_validate(json.loads(report["payload_json"]))
    run = repository.get_run(report["run_id"])
    strategy_row = repository.get_strategy(run["strategy_version_id"])
    proposal = StrategyProposalV1.model_validate(json.loads(strategy_row["strategy_json"]))
    context_row = repository.get_context(strategy_row["context_version_id"])
    context = BusinessContextV1.model_validate(json.loads(context_row["context_json"])["context"])
    project = next(
        (row for row in repository.list_projects() if row["id"] == context_row["project_id"]),
        None,
    )
    dataset = repository.get_dataset(context_row["dataset_id"])
    publication = repository.get_latest_publication_decision(report_id)
    training = repository.get_latest_training_decision(strategy_row["id"])
    metrics = payload.metrics
    holdout = metrics.get("holdout") or {}
    selection = metrics.get("selection") or {}
    status_label, status_color = STATUS_TEXT.get(report["status"], (report["status"], "#555"))
    current = has_publication_approval(repository.connection, report_id)
    generator = getattr(payload, "generator", None) or "TEMPLATE"

    matrix = holdout.get("confusion_matrix")
    matrix_html = ""
    if matrix:
        (tn, fp), (fn, tp) = matrix
        matrix_html = (
            "<table class='matrix'><tr><th></th><th>El modelo alertó</th>"
            "<th>El modelo no alertó</th></tr>"
            f"<tr><th>El evento ocurrió</th><td>{tp} aciertos</td><td>{fn} casos omitidos</td></tr>"
            f"<tr><th>El evento no ocurrió</th><td>{fp} falsas alarmas</td>"
            f"<td>{tn} descartes correctos</td></tr></table>"
        )
    importance = "".join(
        f"<li>{_e(item['feature'])}</li>" for item in metrics.get("permutation_importance", [])[:5]
    )
    insights = "".join(
        f"<li><strong>{_e(KIND_TEXT.get(item.kind, item.kind))}.</strong> {_e(item.claim)}"
        f"<div class='refs'>Evidencia: {_e(', '.join(item.evidence_refs))}</div></li>"
        for item in payload.insights
    )
    limitations = "".join(f"<li>{_e(item)}</li>" for item in payload.limitations)
    approvals = []
    if training is not None:
        approvals.append(
            (
                "Aprobación 1 (antes de entrenar)",
                f"{training['decision']} · {training['actor_label']} · {training['created_at']}",
            )
        )
    if publication is not None:
        approvals.append(
            (
                "Aprobación 2 (antes de publicar)",
                f"{publication['decision']} · {publication['actor_label']} · "
                f"{publication['created_at']}",
            )
        )
    split = proposal.split
    selected = selection.get("selected_model")

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte Goal Settings · {_e(project["name"] if project else "")}</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#1c1c19;background:#fcf9f4;
max-width:880px;margin:0 auto;padding:32px 20px;line-height:1.5}}
h1{{font-size:1.6rem;margin:0 0 4px}} h2{{font-size:1.15rem;margin:28px 0 8px;
border-bottom:1px solid #e5e2de;padding-bottom:4px}}
.status{{display:inline-block;padding:6px 10px;border-radius:6px;color:#fff;
background:{status_color};font-weight:600}}
.warn{{background:#fff0d5;border:1px solid #e5c58a;padding:10px;border-radius:6px}}
table{{border-collapse:collapse;width:100%;margin:8px 0}} th,td{{text-align:left;
padding:6px 8px;border-bottom:1px solid #e5e2de;vertical-align:top}} th{{width:38%;color:#4a4a45}}
.matrix td,.matrix th{{text-align:center;width:auto}} .refs{{font-size:.85rem;color:#666}}
.meta{{font-size:.85rem;color:#666;word-break:break-all}}
@media print{{body{{background:#fff;padding:0}} .noprint{{display:none}}}}
</style></head><body>
<h1>Reporte de experimento</h1>
<p>{_e(project["name"] if project else "")} · archivo {_e(dataset["original_filename_display"])}
· reporte versión {report["version_no"]}</p>
<p><span class="status">{_e(status_label)}</span></p>
{
        ""
        if current or report["status"] == "DRAFT"
        else "<p class='warn'>Este reporte ya no corresponde a la versión vigente del "
        "contexto o la estrategia.</p>"
    }
<p class="noprint meta">Para guardarlo como PDF, usa «Imprimir» en tu navegador.</p>

<h2>Qué se quiso anticipar</h2>
<table>{
        _rows(
            [
                ("Decisión que se quiere apoyar", context.objective),
                ("Resultado que se anticipa", context.target_column),
                ("Evento que se detecta", context.positive_class),
                ("Momento de la decisión", context.prediction_moment),
                ("Si se alerta y no ocurre", context.false_positive_consequence),
                ("Si no se alerta y sí ocurre", context.false_negative_consequence),
                (
                    "Medida principal",
                    METRIC_TEXT.get(context.primary_metric.value, context.primary_metric.value),
                ),
                ("Razón de la medida", context.metric_rationale),
                ("Variables usadas", ", ".join(context.allowed_features)),
                ("Variables excluidas", ", ".join(context.excluded_features) or "ninguna"),
                ("Responsable declarado", context.decision_actor_label),
            ]
        )
    }</table>

<h2>Cómo se entrenó</h2>
<table>{
        _rows(
            [
                (
                    "Reparto de datos",
                    f"{split.train:.0%} aprender · {split.validation:.0%} comparar · "
                    f"{split.test:.0%} prueba final reservada",
                ),
                ("Modelos comparados", "Modelo simple de referencia y dos alternativas"),
                (
                    "Modelo elegido",
                    MODEL_TEXT.get(selected, "Ninguno superó a la referencia")
                    if selected
                    else "Ninguno superó a la referencia",
                ),
                (
                    "Punto de corte para alertar",
                    "—"
                    if selection.get("selected_threshold") is None
                    else f"{selection['selected_threshold']:.3f}",
                ),
                ("Motivo de la selección", selection.get("reason", "—")),
            ]
        )
    }</table>

<h2>Resultado en la prueba final</h2>
<table>{
        _rows(
            [
                ("Casos evaluados", holdout.get("support", "—")),
                (
                    "Proporción del evento",
                    _pct(holdout["positive_support"] / holdout["support"]) if holdout else "—",
                ),
                (METRIC_TEXT["precision"], _pct(holdout.get("precision"))),
                (METRIC_TEXT["recall"], _pct(holdout.get("recall"))),
                (METRIC_TEXT["average_precision"], _pct(holdout.get("average_precision"))),
            ]
        )
    }</table>
{matrix_html}
{
        f"<p>Variables más influyentes (asociación, no causa):</p><ol>{importance}</ol>"
        if importance
        else ""
    }

<h2>Hallazgos</h2>
<p class="meta">{_e(GENERATOR_TEXT.get(generator, generator))}.</p>
<ul>{insights}</ul>

<h2>Limitaciones</h2>
<ul>{limitations}</ul>

<h2>Decisiones humanas y trazabilidad</h2>
<table>{_rows(approvals or [("Aprobaciones", "sin decisiones registradas")])}</table>
<p class="meta">Ejecución {_e(run["id"])} · huella del reporte {_e(report["report_sha256"])} ·
huella de la estrategia {_e(strategy_row["fingerprint"])} · exportado
{datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}. Las etiquetas de responsable son
declaradas, no identidades verificadas. Datos de prueba; no son resultados de negocio.</p>
</body></html>
"""
