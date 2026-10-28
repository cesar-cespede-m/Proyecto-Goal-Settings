-- HITL-01.6 (alcance decidido el 19/10/2026: un archivo por proyecto).
-- 1. Un proyecto admite como máximo un dataset ACCEPTED; los rechazados o en cuarentena no
--    cuentan, para poder reintentar. Para usar otro archivo se crea otro proyecto.
-- 2. Publicar exige además que el contexto, la estrategia y la aprobación de entrenamiento
--    del reporte sigan vigentes: un reporte de una versión superada no se publica.

CREATE TRIGGER IF NOT EXISTS trg_datasets_single_accepted_per_project
BEFORE INSERT ON datasets
FOR EACH ROW
WHEN NEW.admission_status = 'ACCEPTED' AND EXISTS (
    SELECT 1
    FROM datasets AS existing
    WHERE existing.project_id = NEW.project_id
      AND existing.admission_status = 'ACCEPTED'
)
BEGIN
    SELECT RAISE(ABORT, 'project already has an accepted dataset');
END;

DROP TRIGGER IF EXISTS trg_reports_require_publication_approval;

CREATE TRIGGER trg_reports_require_publication_approval
BEFORE UPDATE OF status ON reports
FOR EACH ROW
WHEN NEW.status = 'PUBLISHED' AND (
    COALESCE((
        SELECT approval.decision
        FROM approval_events AS approval
        WHERE approval.stage = 'PUBLICATION'
          AND approval.report_id = NEW.id
          AND approval.approved_hash = NEW.report_sha256
        ORDER BY approval.created_at DESC, approval.rowid DESC
        LIMIT 1
    ), '') <> 'APPROVED'
    OR NOT EXISTS (
        SELECT 1
        FROM runs AS run
        JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
        JOIN context_versions AS context ON context.id = strategy.context_version_id
        WHERE run.id = NEW.run_id
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
          AND COALESCE((
              SELECT approval.decision
              FROM approval_events AS approval
              WHERE approval.stage = 'TRAINING'
                AND approval.strategy_version_id = strategy.id
                AND approval.approved_hash = strategy.fingerprint
              ORDER BY approval.created_at DESC, approval.rowid DESC
              LIMIT 1
          ), '') = 'APPROVED'
    )
)
BEGIN
    SELECT RAISE(ABORT, 'publication approval required for a current report');
END;
