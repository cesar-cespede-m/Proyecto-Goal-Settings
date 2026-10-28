CREATE TRIGGER IF NOT EXISTS trg_strategy_requires_current_context
BEFORE INSERT ON strategy_versions
FOR EACH ROW
WHEN EXISTS (
    SELECT 1
    FROM context_versions AS selected_context
    JOIN context_versions AS newer_context
      ON newer_context.project_id = selected_context.project_id
     AND newer_context.version_no > selected_context.version_no
    WHERE selected_context.id = NEW.context_version_id
)
BEGIN
    SELECT RAISE(ABORT, 'strategy requires the current context');
END;

DROP TRIGGER IF EXISTS trg_runs_require_training_approval;

CREATE TRIGGER trg_runs_require_training_approval
BEFORE INSERT ON runs
FOR EACH ROW
WHEN
    NOT EXISTS (
        SELECT 1
        FROM strategy_versions AS strategy
        JOIN context_versions AS context
          ON context.id = strategy.context_version_id
        WHERE strategy.id = NEW.strategy_version_id
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
    )
    OR COALESCE((
        SELECT approval.decision
        FROM approval_events AS approval
        JOIN strategy_versions AS strategy
          ON strategy.id = NEW.strategy_version_id
        WHERE approval.stage = 'TRAINING'
          AND approval.strategy_version_id = NEW.strategy_version_id
          AND approval.approved_hash = strategy.fingerprint
        ORDER BY approval.created_at DESC, approval.rowid DESC
        LIMIT 1
    ), '') <> 'APPROVED'
BEGIN
    SELECT RAISE(ABORT, 'current context or strategy training approval required');
END;
