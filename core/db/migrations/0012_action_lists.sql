-- V2-17: listas de acción y mediciones agregadas. Ambas tablas son append-only.
-- Los identificadores de caso viven sólo en action_json; jamás en audit_events.

CREATE TABLE action_lists (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE RESTRICT,
    decision_id TEXT NOT NULL UNIQUE REFERENCES decision_records(id) ON DELETE RESTRICT,
    scoring_input_sha256 TEXT NOT NULL CHECK (
        length(scoring_input_sha256) = 64
        AND scoring_input_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    action_json TEXT NOT NULL CHECK (
        json_valid(action_json) AND json_type(action_json) = 'object'
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE action_list_outcomes (
    id TEXT PRIMARY KEY,
    action_list_id TEXT NOT NULL REFERENCES action_lists(id) ON DELETE RESTRICT,
    decision_outcome_id TEXT NOT NULL UNIQUE
        REFERENCES decision_outcomes(id) ON DELETE RESTRICT,
    outcome_json TEXT NOT NULL CHECK (
        json_valid(outcome_json) AND json_type(outcome_json) = 'object'
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_action_lists_project ON action_lists(project_id, created_at);
CREATE INDEX idx_action_lists_run ON action_lists(run_id, created_at);
CREATE INDEX idx_action_list_outcomes_list
ON action_list_outcomes(action_list_id, created_at);

CREATE TRIGGER action_list_project_matches_run BEFORE INSERT ON action_lists
FOR EACH ROW WHEN NOT EXISTS (
    SELECT 1
    FROM runs AS run
    JOIN strategy_versions AS strategy ON strategy.id = run.strategy_version_id
    JOIN context_versions AS context ON context.id = strategy.context_version_id
    WHERE run.id = NEW.run_id AND context.project_id = NEW.project_id
)
BEGIN SELECT RAISE(ABORT, 'action list project and run mismatch'); END;

CREATE TRIGGER action_lists_no_update BEFORE UPDATE ON action_lists
BEGIN SELECT RAISE(ABORT, 'action_lists is append-only'); END;
CREATE TRIGGER action_lists_no_delete BEFORE DELETE ON action_lists
BEGIN SELECT RAISE(ABORT, 'action_lists is append-only'); END;
CREATE TRIGGER action_list_outcomes_no_update BEFORE UPDATE ON action_list_outcomes
BEGIN SELECT RAISE(ABORT, 'action_list_outcomes is append-only'); END;
CREATE TRIGGER action_list_outcomes_no_delete BEFORE DELETE ON action_list_outcomes
BEGIN SELECT RAISE(ABORT, 'action_list_outcomes is append-only'); END;
