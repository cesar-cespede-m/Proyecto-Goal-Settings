CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    sha256 TEXT NOT NULL
        CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    original_filename_display TEXT NOT NULL,
    stored_path TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    license TEXT,
    anonymization_declared INTEGER NOT NULL
        CHECK (anonymization_declared IN (0, 1)),
    admission_status TEXT NOT NULL
        CHECK (admission_status IN ('QUARANTINED', 'ACCEPTED', 'REJECTED')),
    rejection_reason TEXT,
    n_rows INTEGER CHECK (n_rows IS NULL OR n_rows >= 0),
    n_cols INTEGER CHECK (n_cols IS NULL OR n_cols >= 0),
    bytes INTEGER NOT NULL CHECK (bytes >= 0),
    profile_json TEXT NOT NULL
        CHECK (json_valid(profile_json) AND json_type(profile_json) = 'object'),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (project_id, sha256),
    CHECK (
        (admission_status = 'REJECTED' AND rejection_reason IS NOT NULL)
        OR admission_status <> 'REJECTED'
    )
);

CREATE TABLE IF NOT EXISTS context_versions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
    version_no INTEGER NOT NULL CHECK (version_no > 0),
    target_column TEXT NOT NULL CHECK (length(trim(target_column)) > 0),
    positive_class TEXT NOT NULL,
    prediction_moment TEXT NOT NULL CHECK (length(trim(prediction_moment)) > 0),
    excluded_columns_json TEXT NOT NULL
        CHECK (json_valid(excluded_columns_json) AND json_type(excluded_columns_json) = 'array'),
    primary_metric TEXT NOT NULL CHECK (length(trim(primary_metric)) > 0),
    guardrails_json TEXT
        CHECK (
            guardrails_json IS NULL
            OR (json_valid(guardrails_json) AND json_type(guardrails_json) = 'object')
        ),
    context_json TEXT NOT NULL
        CHECK (json_valid(context_json) AND json_type(context_json) = 'object'),
    context_sha256 TEXT NOT NULL
        CHECK (length(context_sha256) = 64 AND context_sha256 NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (project_id, version_no),
    UNIQUE (project_id, context_sha256)
);

CREATE TABLE IF NOT EXISTS strategy_versions (
    id TEXT PRIMARY KEY,
    context_version_id TEXT NOT NULL REFERENCES context_versions(id) ON DELETE RESTRICT,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
    version_no INTEGER NOT NULL CHECK (version_no > 0),
    strategy_json TEXT NOT NULL
        CHECK (json_valid(strategy_json) AND json_type(strategy_json) = 'object'),
    schema_version TEXT NOT NULL CHECK (length(trim(schema_version)) > 0),
    fingerprint TEXT NOT NULL
        CHECK (length(fingerprint) = 64 AND fingerprint NOT GLOB '*[^0-9a-f]*'),
    source TEXT NOT NULL CHECK (source IN ('TEMPLATE', 'LLM')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (context_version_id, version_no)
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    strategy_version_id TEXT NOT NULL REFERENCES strategy_versions(id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK (
        status IN (
            'RUNNING',
            'COMPLETED',
            'COMPLETED_WITHOUT_RECOMMENDATION',
            'FAILED',
            'BLOCKED'
        )
    ),
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    seed INTEGER NOT NULL,
    split_info_json TEXT NOT NULL
        CHECK (json_valid(split_info_json) AND json_type(split_info_json) = 'object'),
    candidate_metrics_json TEXT NOT NULL
        CHECK (json_valid(candidate_metrics_json) AND json_type(candidate_metrics_json) = 'object'),
    selected_model TEXT,
    artifact_path TEXT,
    manifest_json TEXT NOT NULL
        CHECK (json_valid(manifest_json) AND json_type(manifest_json) = 'object'),
    error_message TEXT,
    CHECK (
        (status = 'RUNNING' AND finished_at IS NULL)
        OR (status <> 'RUNNING' AND finished_at IS NOT NULL)
    ),
    CHECK (status <> 'FAILED' OR error_message IS NOT NULL),
    CHECK (status <> 'COMPLETED' OR selected_model IS NOT NULL),
    CHECK (status <> 'COMPLETED_WITHOUT_RECOMMENDATION' OR selected_model IS NULL)
);

CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE RESTRICT,
    version_no INTEGER NOT NULL CHECK (version_no > 0),
    payload_json TEXT NOT NULL
        CHECK (json_valid(payload_json) AND json_type(payload_json) = 'object'),
    report_sha256 TEXT NOT NULL
        CHECK (length(report_sha256) = 64 AND report_sha256 NOT GLOB '*[^0-9a-f]*'),
    status TEXT NOT NULL CHECK (
        status IN ('DRAFT', 'APPROVED', 'PUBLISHED', 'REJECTED', 'SUPERSEDED')
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (run_id, version_no),
    UNIQUE (report_sha256)
);

CREATE TABLE IF NOT EXISTS approval_events (
    id TEXT PRIMARY KEY,
    stage TEXT NOT NULL CHECK (stage IN ('TRAINING', 'PUBLICATION')),
    decision TEXT NOT NULL CHECK (decision IN ('APPROVED', 'REJECTED')),
    strategy_version_id TEXT REFERENCES strategy_versions(id) ON DELETE RESTRICT,
    report_id TEXT REFERENCES reports(id) ON DELETE RESTRICT,
    approved_hash TEXT NOT NULL
        CHECK (length(approved_hash) = 64 AND approved_hash NOT GLOB '*[^0-9a-f]*'),
    actor_label TEXT NOT NULL CHECK (length(trim(actor_label)) > 0),
    comment TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        (stage = 'TRAINING' AND strategy_version_id IS NOT NULL AND report_id IS NULL)
        OR (stage = 'PUBLICATION' AND strategy_version_id IS NULL AND report_id IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    event_type TEXT NOT NULL CHECK (length(trim(event_type)) > 0),
    entity_type TEXT NOT NULL CHECK (length(trim(entity_type)) > 0),
    entity_id TEXT NOT NULL,
    detail_json TEXT NOT NULL
        CHECK (json_valid(detail_json) AND json_type(detail_json) = 'object'),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_datasets_project ON datasets(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_context_project ON context_versions(project_id, version_no);
CREATE INDEX IF NOT EXISTS idx_strategy_context ON strategy_versions(context_version_id, version_no);
CREATE INDEX IF NOT EXISTS idx_strategy_dataset ON strategy_versions(dataset_id);
CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs(strategy_version_id, started_at);
CREATE INDEX IF NOT EXISTS idx_approvals_strategy
    ON approval_events(strategy_version_id, approved_hash, created_at);
CREATE INDEX IF NOT EXISTS idx_approvals_report
    ON approval_events(report_id, approved_hash, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_project ON audit_events(project_id, created_at);

CREATE UNIQUE INDEX IF NOT EXISTS uq_single_running_run
    ON runs((1)) WHERE status = 'RUNNING';

CREATE TRIGGER IF NOT EXISTS trg_context_project_matches_dataset
BEFORE INSERT ON context_versions
FOR EACH ROW
WHEN NOT EXISTS (
    SELECT 1 FROM datasets
    WHERE id = NEW.dataset_id AND project_id = NEW.project_id
)
BEGIN
    SELECT RAISE(ABORT, 'context project and dataset mismatch');
END;

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

CREATE TRIGGER IF NOT EXISTS trg_strategy_dataset_matches_context
BEFORE INSERT ON strategy_versions
FOR EACH ROW
WHEN NOT EXISTS (
    SELECT 1
    FROM context_versions AS context
    WHERE context.id = NEW.context_version_id
      AND context.dataset_id = NEW.dataset_id
)
BEGIN
    SELECT RAISE(ABORT, 'strategy dataset and context mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_approval_training_hash
BEFORE INSERT ON approval_events
FOR EACH ROW
WHEN NEW.stage = 'TRAINING' AND NOT EXISTS (
    SELECT 1 FROM strategy_versions
    WHERE id = NEW.strategy_version_id AND fingerprint = NEW.approved_hash
)
BEGIN
    SELECT RAISE(ABORT, 'training approval hash mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_approval_publication_hash
BEFORE INSERT ON approval_events
FOR EACH ROW
WHEN NEW.stage = 'PUBLICATION' AND NOT EXISTS (
    SELECT 1 FROM reports
    WHERE id = NEW.report_id AND report_sha256 = NEW.approved_hash
)
BEGIN
    SELECT RAISE(ABORT, 'publication approval hash mismatch');
END;

CREATE TRIGGER IF NOT EXISTS trg_runs_require_training_approval
BEFORE INSERT ON runs
FOR EACH ROW
WHEN COALESCE((
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
    SELECT RAISE(ABORT, 'training approval required');
END;

CREATE TRIGGER IF NOT EXISTS trg_runs_state_transition
BEFORE UPDATE OF status ON runs
FOR EACH ROW
WHEN NOT (
    OLD.status = 'RUNNING'
    AND NEW.status IN (
        'COMPLETED',
        'COMPLETED_WITHOUT_RECOMMENDATION',
        'FAILED',
        'BLOCKED'
    )
)
BEGIN
    SELECT RAISE(ABORT, 'invalid run state transition');
END;

CREATE TRIGGER IF NOT EXISTS trg_reports_start_draft
BEFORE INSERT ON reports
FOR EACH ROW
WHEN NEW.status <> 'DRAFT'
BEGIN
    SELECT RAISE(ABORT, 'reports must start as DRAFT');
END;

CREATE TRIGGER IF NOT EXISTS trg_reports_state_transition
BEFORE UPDATE OF status ON reports
FOR EACH ROW
WHEN NOT (
    (OLD.status = 'DRAFT' AND NEW.status IN ('APPROVED', 'REJECTED', 'SUPERSEDED'))
    OR (OLD.status = 'APPROVED' AND NEW.status IN ('PUBLISHED', 'REJECTED', 'SUPERSEDED'))
    OR (OLD.status IN ('PUBLISHED', 'REJECTED') AND NEW.status = 'SUPERSEDED')
)
BEGIN
    SELECT RAISE(ABORT, 'invalid report state transition');
END;

CREATE TRIGGER IF NOT EXISTS trg_reports_content_no_update
BEFORE UPDATE OF run_id, version_no, payload_json, report_sha256 ON reports
BEGIN
    SELECT RAISE(ABORT, 'report content is immutable; create a new version');
END;

CREATE TRIGGER IF NOT EXISTS trg_reports_require_publication_approval
BEFORE UPDATE OF status ON reports
FOR EACH ROW
WHEN NEW.status = 'PUBLISHED' AND COALESCE((
    SELECT approval.decision
    FROM approval_events AS approval
    WHERE approval.stage = 'PUBLICATION'
      AND approval.report_id = NEW.id
      AND approval.approved_hash = NEW.report_sha256
    ORDER BY approval.created_at DESC, approval.rowid DESC
    LIMIT 1
), '') <> 'APPROVED'
BEGIN
    SELECT RAISE(ABORT, 'publication approval required');
END;

CREATE TRIGGER IF NOT EXISTS trg_context_versions_no_update
BEFORE UPDATE ON context_versions
BEGIN
    SELECT RAISE(ABORT, 'context_versions is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_context_versions_no_delete
BEFORE DELETE ON context_versions
BEGIN
    SELECT RAISE(ABORT, 'context_versions is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_strategy_versions_no_update
BEFORE UPDATE ON strategy_versions
BEGIN
    SELECT RAISE(ABORT, 'strategy_versions is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_strategy_versions_no_delete
BEFORE DELETE ON strategy_versions
BEGIN
    SELECT RAISE(ABORT, 'strategy_versions is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_approval_events_no_update
BEFORE UPDATE ON approval_events
BEGIN
    SELECT RAISE(ABORT, 'approval_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_approval_events_no_delete
BEFORE DELETE ON approval_events
BEGIN
    SELECT RAISE(ABORT, 'approval_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_audit_events_no_update
BEFORE UPDATE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_audit_events_no_delete
BEFORE DELETE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit_events is append-only');
END;
