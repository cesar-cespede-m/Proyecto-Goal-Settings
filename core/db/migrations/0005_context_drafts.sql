-- CTX-01.6 / AI-01.8: borrador de contexto persistente y explicación de la estrategia.
-- El borrador guarda valores, origen y confirmaciones por campo; nunca la conversación.

CREATE TABLE IF NOT EXISTS context_drafts (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
    dataset_sha256 TEXT NOT NULL
        CHECK (length(dataset_sha256) = 64 AND dataset_sha256 NOT GLOB '*[^0-9a-f]*'),
    status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'FINALIZED', 'DISCARDED')),
    revision INTEGER NOT NULL CHECK (revision >= 0),
    draft_json TEXT NOT NULL
        CHECK (json_valid(draft_json) AND json_type(draft_json) = 'object'),
    context_version_id TEXT REFERENCES context_versions(id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (status <> 'FINALIZED' OR context_version_id IS NOT NULL)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_context_drafts_one_active
ON context_drafts(project_id, dataset_id, dataset_sha256)
WHERE status = 'ACTIVE';

CREATE TRIGGER IF NOT EXISTS trg_context_drafts_dataset_matches
BEFORE INSERT ON context_drafts
FOR EACH ROW
WHEN NOT EXISTS (
    SELECT 1
    FROM datasets
    WHERE id = NEW.dataset_id
      AND project_id = NEW.project_id
      AND sha256 = NEW.dataset_sha256
      AND admission_status = 'ACCEPTED'
)
BEGIN
    SELECT RAISE(ABORT, 'draft requires the accepted dataset of its project');
END;

CREATE TRIGGER IF NOT EXISTS trg_context_drafts_closed_are_final
BEFORE UPDATE ON context_drafts
FOR EACH ROW
WHEN OLD.status <> 'ACTIVE'
BEGIN
    SELECT RAISE(ABORT, 'a closed draft cannot change');
END;

CREATE TABLE IF NOT EXISTS strategy_explanations (
    id TEXT PRIMARY KEY,
    strategy_version_id TEXT NOT NULL REFERENCES strategy_versions(id) ON DELETE RESTRICT,
    strategy_fingerprint TEXT NOT NULL
        CHECK (length(strategy_fingerprint) = 64
               AND strategy_fingerprint NOT GLOB '*[^0-9a-f]*'),
    source TEXT NOT NULL CHECK (source IN ('LLM', 'TEMPLATE')),
    explanation_json TEXT NOT NULL
        CHECK (json_valid(explanation_json) AND json_type(explanation_json) = 'object'),
    ai_trace_json TEXT
        CHECK (ai_trace_json IS NULL
               OR (json_valid(ai_trace_json) AND json_type(ai_trace_json) = 'object')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER IF NOT EXISTS trg_strategy_explanations_fingerprint
BEFORE INSERT ON strategy_explanations
FOR EACH ROW
WHEN NOT EXISTS (
    SELECT 1
    FROM strategy_versions
    WHERE id = NEW.strategy_version_id
      AND fingerprint = NEW.strategy_fingerprint
)
BEGIN
    SELECT RAISE(ABORT, 'explanation must match the strategy fingerprint');
END;

CREATE TRIGGER IF NOT EXISTS trg_strategy_explanations_no_update
BEFORE UPDATE ON strategy_explanations
BEGIN
    SELECT RAISE(ABORT, 'strategy explanations are immutable');
END;
