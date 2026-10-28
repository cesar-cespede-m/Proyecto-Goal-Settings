CREATE TRIGGER IF NOT EXISTS trg_context_requires_accepted_dataset
BEFORE INSERT ON context_versions
FOR EACH ROW
WHEN NOT EXISTS (
    SELECT 1
    FROM datasets
    WHERE id = NEW.dataset_id
      AND project_id = NEW.project_id
      AND admission_status = 'ACCEPTED'
)
BEGIN
    SELECT RAISE(ABORT, 'context requires an accepted dataset');
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
    SELECT RAISE(ABORT, 'current context training approval required');
END;
