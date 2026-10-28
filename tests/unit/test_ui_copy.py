"""Contratos de microcopy y límites visibles de la interfaz."""

import tomllib
from pathlib import Path

from app.ui.copy import FIELD_COPY, GLOSSARY
from app.ui.messages import ERROR_MESSAGES

ROOT = Path(__file__).parents[2]

REQUIRED_FIELDS = {
    "project_name",
    "active_project",
    "csv_file",
    "source",
    "license",
    "classification",
    "authorization",
    "anonymization",
    "active_file",
    "task_type",
    "assistant_conversation",
    "outgoing_preview",
    "objective",
    "target_column",
    "positive_class",
    "prediction_moment",
    "false_positive",
    "false_negative",
    "feature_use",
    "feature_availability",
    "primary_metric",
    "metric_rationale",
    "guardrail_mode",
    "guardrail_metric",
    "guardrail_threshold",
    "guardrail_rationale",
    "decision_actor",
    "strategy_origin",
    "strategy_features",
    "strategy_split",
    "strategy_seed",
    "preprocessing",
    "candidate_models",
    "decision_threshold",
    "runtime_limits",
    "strategy_warnings",
    "strategy_explanation",
    "training_decision",
    "strategy_reviewer",
    "strategy_comment",
    "train_action",
    "run_status",
    "primary_result",
    "translated_metrics",
    "baseline_comparison",
    "uncertainty_range",
    "confusion_matrix",
    "feature_importance",
    "generate_report",
    "generation_mode",
    "insight_kind",
    "evidence",
    "report_decision",
    "report_reviewer",
    "report_comment",
    "publish_action",
    "compare_versions",
    "compare_selection",
    "results_question",
    "report_download",
    "new_cases",
    "new_cases_file",
    "scoring_actor",
}


TEMPLATE_LABELS = ("Qué es", "Para qué sirve", "Alcance y límites", "Quién decide", "Ejemplo:")
JARGON = (
    "dataset",
    "SQLite",
    " run ",
    "binario",
    "cuarentena",
    "perfilado",
    "auditado",
    "holdout",
    "baseline",
    "recall",
    "threshold",
)


def test_every_field_explains_itself_in_natural_prose() -> None:
    """DU-1: cada campo tiene ayuda en prosa, sin plantilla de etiquetas ni jerga."""
    assert set(FIELD_COPY) == REQUIRED_FIELDS
    for field_key, content in FIELD_COPY.items():
        text = content.text
        assert 30 <= len(text) <= 420, field_key
        assert text.rstrip().endswith((".", "?", "»")), field_key
        assert not any(label in text for label in TEMPLATE_LABELS), field_key
        assert not any(term.lower() in f" {text.lower()} " for term in JARGON), field_key


def test_glossary_contains_every_required_technical_term() -> None:
    assert {
        "target",
        "positive_class",
        "recall",
        "precision",
        "f1",
        "average_precision",
        "split",
        "train",
        "validation",
        "test",
        "seed",
        "holdout",
        "leakage",
        "guardrail",
        "threshold",
        "baseline",
        "run",
    } <= set(GLOSSARY)


def test_streamlit_upload_limit_matches_the_visible_product_limit() -> None:
    with (ROOT / ".streamlit" / "config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    assert config["server"]["maxUploadSize"] == 25


def test_common_messages_cover_every_workflow_area() -> None:
    required_codes = {
        "FILE_TOO_LARGE",
        "MALFORMED_CSV",
        "TARGET_NOT_BINARY",
        "INVALID_CONTEXT",
        "CONTEXT_UNCHANGED",
        "INVALID_STRATEGY",
        "STRATEGY_UNCHANGED",
        "TRAINING_FAILED",
        "RUN_NOT_REPORTABLE",
        "INVALID_REPORT",
        "REPORT_NOT_APPROVED",
    }

    assert required_codes <= set(ERROR_MESSAGES)
    for code in required_codes:
        message = ERROR_MESSAGES[code]
        assert message.message.strip()
        assert message.action.strip()
