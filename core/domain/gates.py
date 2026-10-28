"""Gates de dominio que revalidan la aprobación exacta.

Una estrategia es vigente si usa el snapshot analítico activo y no existe un contexto más nuevo
en su proyecto ni una estrategia más nueva para su contexto.
"""

import sqlite3

from core.domain.errors import ApprovalRequiredError

# Requiere los alias `context` y `strategy` en la consulta que lo incluye.
_STRATEGY_IS_CURRENT = """
    EXISTS (
        SELECT 1
        FROM datasets AS active_dataset
        WHERE active_dataset.id = context.dataset_id
          AND active_dataset.admission_status = 'ACCEPTED'
          AND active_dataset.is_active = 1
    )
    AND NOT EXISTS (
        SELECT 1
        FROM context_versions AS newer_context
        WHERE newer_context.project_id = context.project_id
          AND newer_context.version_no > context.version_no
    )
    AND NOT EXISTS (
        SELECT 1
        FROM strategy_versions AS newer_strategy
        WHERE newer_strategy.context_version_id = strategy.context_version_id
          AND newer_strategy.version_no > strategy.version_no
    )
"""


def is_strategy_current(connection: sqlite3.Connection, strategy_version_id: str) -> bool:
    row = connection.execute(
        f"""
        SELECT 1
        FROM strategy_versions AS strategy
        JOIN context_versions AS context ON context.id = strategy.context_version_id
        WHERE strategy.id = ?
          AND {_STRATEGY_IS_CURRENT}
        """,
        (strategy_version_id,),
    ).fetchone()
    return row is not None


def has_training_approval(connection: sqlite3.Connection, strategy_version_id: str) -> bool:
    row = connection.execute(
        f"""
        SELECT approval.decision
        FROM approval_events AS approval
        JOIN strategy_versions AS strategy
          ON strategy.id = approval.strategy_version_id
        JOIN context_versions AS context
          ON context.id = strategy.context_version_id
        WHERE approval.stage = 'TRAINING'
          AND approval.strategy_version_id = ?
          AND approval.approved_hash = strategy.fingerprint
          AND {_STRATEGY_IS_CURRENT}
        ORDER BY approval.created_at DESC, approval.rowid DESC
        LIMIT 1
        """,
        (strategy_version_id,),
    ).fetchone()
    return row is not None and row["decision"] == "APPROVED"


def require_training_approval(connection: sqlite3.Connection, strategy_version_id: str) -> None:
    if not has_training_approval(connection, strategy_version_id):
        raise ApprovalRequiredError("STRATEGY_NOT_APPROVED")


def is_report_current(connection: sqlite3.Connection, report_id: str) -> bool:
    """El reporte proviene de una estrategia vigente y aún aprobada para entrenar."""

    row = connection.execute(
        """
        SELECT run.strategy_version_id
        FROM reports AS report
        JOIN runs AS run ON run.id = report.run_id
        WHERE report.id = ?
        """,
        (report_id,),
    ).fetchone()
    return row is not None and has_training_approval(connection, row["strategy_version_id"])


def has_publication_approval(connection: sqlite3.Connection, report_id: str) -> bool:
    row = connection.execute(
        """
        SELECT approval.decision
        FROM approval_events AS approval
        JOIN reports AS report ON report.id = approval.report_id
        WHERE approval.stage = 'PUBLICATION'
          AND approval.report_id = ?
          AND approval.approved_hash = report.report_sha256
        ORDER BY approval.created_at DESC, approval.rowid DESC
        LIMIT 1
        """,
        (report_id,),
    ).fetchone()
    return (
        row is not None
        and row["decision"] == "APPROVED"
        and is_report_current(connection, report_id)
    )


def require_publication_approval(connection: sqlite3.Connection, report_id: str) -> None:
    if not has_publication_approval(connection, report_id):
        raise ApprovalRequiredError("REPORT_NOT_APPROVED")
