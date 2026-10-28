"""EXT-D: exportar el reporte como HTML autocontenido y escapado."""

from pathlib import Path

from core.reporting import ReportService
from core.reporting.export import render_report_html
from tests.support.ml_workflow import prepared_strategy

ACTOR = "pytest-<b>usuario</b>"


def test_exported_report_is_plain_language_escaped_and_shows_status(tmp_path: Path) -> None:
    connection, repository, ml, strategy_id, _ = prepared_strategy(tmp_path)
    repository.decide_training(
        strategy_version_id=strategy_id, decision="APPROVED", actor_label=ACTOR
    )
    run = ml.execute(strategy_id)
    report_id = ReportService(repository).create_template_report(run.run_id)

    draft = render_report_html(repository, report_id)
    assert draft.startswith("<!doctype html>")
    assert "Borrador — pendiente de revisión humana" in draft
    assert "De los eventos reales, cuántos detecta" in draft
    assert "<b>usuario</b>" not in draft and "&lt;b&gt;usuario&lt;/b&gt;" in draft
    assert "<script" not in draft.lower()

    repository.decide_publication(report_id=report_id, decision="APPROVED", actor_label="r")
    approved = render_report_html(repository, report_id)
    assert "Aprobado por una persona" in approved
    assert "Aprobación 2 (antes de publicar)" in approved
    connection.close()
