-- C2-02 (ciclo 2, B3/B4): decisiones tomadas con análisis, predicciones o reportes, y el
-- resultado que se observó después. Ambas tablas son sólo de inserción.

CREATE TABLE decision_records (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    record_json TEXT NOT NULL
        CHECK (json_valid(record_json) AND json_type(record_json) = 'object'),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE decision_outcomes (
    id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL REFERENCES decision_records(id) ON DELETE RESTRICT,
    observed_result TEXT NOT NULL CHECK (length(trim(observed_result)) > 0),
    observed_at TEXT NOT NULL,
    actor_label TEXT NOT NULL CHECK (length(trim(actor_label)) > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_decision_records_project ON decision_records(project_id, created_at);
CREATE INDEX idx_decision_outcomes_decision ON decision_outcomes(decision_id, created_at);

CREATE TRIGGER decision_records_no_update BEFORE UPDATE ON decision_records
BEGIN SELECT RAISE(ABORT, 'decision_records is append-only'); END;
CREATE TRIGGER decision_records_no_delete BEFORE DELETE ON decision_records
BEGIN SELECT RAISE(ABORT, 'decision_records is append-only'); END;
CREATE TRIGGER decision_outcomes_no_update BEFORE UPDATE ON decision_outcomes
BEGIN SELECT RAISE(ABORT, 'decision_outcomes is append-only'); END;
CREATE TRIGGER decision_outcomes_no_delete BEFORE DELETE ON decision_outcomes
BEGIN SELECT RAISE(ABORT, 'decision_outcomes is append-only'); END;
