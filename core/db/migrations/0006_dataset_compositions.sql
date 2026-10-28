-- DATA-02.2: varias fuentes aceptadas, un solo snapshot analítico activo y linaje inmutable.

DROP TRIGGER IF EXISTS trg_datasets_single_accepted_per_project;

ALTER TABLE datasets ADD COLUMN dataset_kind TEXT NOT NULL DEFAULT 'SOURCE'
    CHECK (dataset_kind IN ('SOURCE', 'DERIVED'));
ALTER TABLE datasets ADD COLUMN is_active INTEGER NOT NULL DEFAULT 0
    CHECK (is_active IN (0, 1));

UPDATE datasets
SET is_active = 1
WHERE admission_status = 'ACCEPTED';

CREATE UNIQUE INDEX uq_datasets_active_per_project
ON datasets(project_id)
WHERE admission_status = 'ACCEPTED' AND is_active = 1;

CREATE TABLE dataset_compositions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    output_dataset_id TEXT NOT NULL UNIQUE REFERENCES datasets(id) ON DELETE RESTRICT,
    spec_json TEXT NOT NULL
        CHECK (json_valid(spec_json) AND json_type(spec_json) = 'object'),
    spec_sha256 TEXT NOT NULL
        CHECK (length(spec_sha256) = 64 AND spec_sha256 NOT GLOB '*[^0-9a-f]*'),
    summary_json TEXT NOT NULL
        CHECK (json_valid(summary_json) AND json_type(summary_json) = 'object'),
    actor_label TEXT NOT NULL CHECK (length(trim(actor_label)) > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, spec_sha256)
);

CREATE TABLE dataset_composition_sources (
    composition_id TEXT NOT NULL REFERENCES dataset_compositions(id) ON DELETE RESTRICT,
    position INTEGER NOT NULL CHECK (position BETWEEN 0 AND 4),
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
    role TEXT NOT NULL CHECK (role IN ('PRIMARY', 'AUXILIARY')),
    alias TEXT,
    source_sha256 TEXT NOT NULL
        CHECK (length(source_sha256) = 64 AND source_sha256 NOT GLOB '*[^0-9a-f]*'),
    primary_key TEXT,
    auxiliary_key TEXT,
    cardinality TEXT CHECK (cardinality IN ('many_to_one', 'one_to_one')),
    PRIMARY KEY(composition_id, position),
    UNIQUE(composition_id, dataset_id),
    CHECK (
        (role = 'PRIMARY' AND position = 0 AND alias IS NULL
         AND primary_key IS NULL AND auxiliary_key IS NULL AND cardinality IS NULL)
        OR
        (role = 'AUXILIARY' AND position > 0 AND alias IS NOT NULL
         AND primary_key IS NOT NULL AND auxiliary_key IS NOT NULL AND cardinality IS NOT NULL)
    )
);

CREATE INDEX idx_compositions_project ON dataset_compositions(project_id, created_at);
CREATE INDEX idx_composition_sources_dataset ON dataset_composition_sources(dataset_id);

CREATE TRIGGER trg_datasets_active_requires_accepted
BEFORE INSERT ON datasets
FOR EACH ROW
WHEN NEW.is_active = 1 AND NEW.admission_status <> 'ACCEPTED'
BEGIN
    SELECT RAISE(ABORT, 'active dataset must be accepted');
END;

CREATE TRIGGER trg_datasets_active_update_requires_accepted
BEFORE UPDATE OF is_active, admission_status ON datasets
FOR EACH ROW
WHEN NEW.is_active = 1 AND NEW.admission_status <> 'ACCEPTED'
BEGIN
    SELECT RAISE(ABORT, 'active dataset must be accepted');
END;

DROP TRIGGER IF EXISTS trg_context_requires_accepted_dataset;
CREATE TRIGGER trg_context_requires_accepted_dataset
BEFORE INSERT ON context_versions
FOR EACH ROW
WHEN NOT EXISTS (
    SELECT 1 FROM datasets
    WHERE id = NEW.dataset_id
      AND project_id = NEW.project_id
      AND admission_status = 'ACCEPTED'
      AND is_active = 1
)
BEGIN
    SELECT RAISE(ABORT, 'context requires the active accepted dataset');
END;

DROP TRIGGER IF EXISTS trg_context_drafts_dataset_matches;
CREATE TRIGGER trg_context_drafts_dataset_matches
BEFORE INSERT ON context_drafts
FOR EACH ROW
WHEN NOT EXISTS (
    SELECT 1 FROM datasets
    WHERE id = NEW.dataset_id
      AND project_id = NEW.project_id
      AND sha256 = NEW.dataset_sha256
      AND admission_status = 'ACCEPTED'
      AND is_active = 1
)
BEGIN
    SELECT RAISE(ABORT, 'context draft dataset mismatch');
END;

DROP TRIGGER IF EXISTS trg_strategy_requires_current_context;
CREATE TRIGGER trg_strategy_requires_current_context
BEFORE INSERT ON strategy_versions
FOR EACH ROW
WHEN NOT EXISTS (
    SELECT 1
    FROM context_versions AS selected_context
    JOIN datasets AS dataset ON dataset.id = selected_context.dataset_id
    WHERE selected_context.id = NEW.context_version_id
      AND dataset.is_active = 1
      AND NOT EXISTS (
          SELECT 1 FROM context_versions AS newer_context
          WHERE newer_context.project_id = selected_context.project_id
            AND newer_context.version_no > selected_context.version_no
      )
)
BEGIN
    SELECT RAISE(ABORT, 'strategy requires the current context on active dataset');
END;

DROP TRIGGER IF EXISTS trg_runs_require_training_approval;
CREATE TRIGGER trg_runs_require_training_approval
BEFORE INSERT ON runs
FOR EACH ROW
WHEN
    NOT EXISTS (
        SELECT 1
        FROM strategy_versions AS strategy
        JOIN context_versions AS context ON context.id = strategy.context_version_id
        JOIN datasets AS dataset ON dataset.id = context.dataset_id
        WHERE strategy.id = NEW.strategy_version_id
          AND dataset.is_active = 1
          AND NOT EXISTS (
              SELECT 1 FROM context_versions AS newer_context
              WHERE newer_context.project_id = context.project_id
                AND newer_context.version_no > context.version_no
          )
          AND NOT EXISTS (
              SELECT 1 FROM strategy_versions AS newer_strategy
              WHERE newer_strategy.context_version_id = strategy.context_version_id
                AND newer_strategy.version_no > strategy.version_no
          )
    )
    OR COALESCE((
        SELECT approval.decision
        FROM approval_events AS approval
        JOIN strategy_versions AS strategy ON strategy.id = NEW.strategy_version_id
        WHERE approval.stage = 'TRAINING'
          AND approval.strategy_version_id = NEW.strategy_version_id
          AND approval.approved_hash = strategy.fingerprint
        ORDER BY approval.created_at DESC, approval.rowid DESC
        LIMIT 1
    ), '') <> 'APPROVED'
BEGIN
    SELECT RAISE(ABORT, 'current context or strategy on active dataset and training approval required');
END;

DROP TRIGGER IF EXISTS trg_reports_require_publication_approval;
CREATE TRIGGER trg_reports_require_publication_approval
BEFORE UPDATE OF status ON reports
FOR EACH ROW
WHEN NEW.status = 'PUBLISHED' AND (
    COALESCE((
        SELECT approval.decision FROM approval_events AS approval
        WHERE approval.stage = 'PUBLICATION'
          AND approval.report_id = NEW.id
          AND approval.approved_hash = NEW.report_sha256
        ORDER BY approval.created_at DESC, approval.rowid DESC LIMIT 1
    ), '') <> 'APPROVED'
    OR NOT EXISTS (
        SELECT 1
        FROM runs AS run
        JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
        JOIN context_versions AS context ON context.id = strategy.context_version_id
        JOIN datasets AS dataset ON dataset.id = context.dataset_id
        WHERE run.id = NEW.run_id
          AND dataset.is_active = 1
          AND NOT EXISTS (
              SELECT 1 FROM context_versions AS newer_context
              WHERE newer_context.project_id = context.project_id
                AND newer_context.version_no > context.version_no
          )
          AND NOT EXISTS (
              SELECT 1 FROM strategy_versions AS newer_strategy
              WHERE newer_strategy.context_version_id = strategy.context_version_id
                AND newer_strategy.version_no > strategy.version_no
          )
          AND COALESCE((
              SELECT approval.decision FROM approval_events AS approval
              WHERE approval.stage = 'TRAINING'
                AND approval.strategy_version_id = strategy.id
                AND approval.approved_hash = strategy.fingerprint
              ORDER BY approval.created_at DESC, approval.rowid DESC LIMIT 1
          ), '') = 'APPROVED'
    )
)
BEGIN
    SELECT RAISE(ABORT, 'publication approval required for a current report on active dataset');
END;

CREATE TRIGGER dataset_compositions_no_update BEFORE UPDATE ON dataset_compositions
BEGIN SELECT RAISE(ABORT, 'dataset_compositions is append-only'); END;
CREATE TRIGGER dataset_compositions_no_delete BEFORE DELETE ON dataset_compositions
BEGIN SELECT RAISE(ABORT, 'dataset_compositions is append-only'); END;
CREATE TRIGGER dataset_composition_sources_no_update BEFORE UPDATE ON dataset_composition_sources
BEGIN SELECT RAISE(ABORT, 'dataset_composition_sources is append-only'); END;
CREATE TRIGGER dataset_composition_sources_no_delete BEFORE DELETE ON dataset_composition_sources
BEGIN SELECT RAISE(ABORT, 'dataset_composition_sources is append-only'); END;
