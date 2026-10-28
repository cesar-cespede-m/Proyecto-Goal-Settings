-- GA-06: resultado de un plan de análisis guardado como tabla con receta inmutable.
-- La receta (plan + fuentes con su hash + versión del motor) permite rehacer la tabla (GA-08).

CREATE TABLE analysis_tables (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    output_dataset_id TEXT NOT NULL UNIQUE REFERENCES datasets(id) ON DELETE RESTRICT,
    display_name TEXT NOT NULL CHECK (length(trim(display_name)) > 0),
    plan_json TEXT NOT NULL
        CHECK (json_valid(plan_json) AND json_type(plan_json) = 'object'),
    plan_sha256 TEXT NOT NULL
        CHECK (length(plan_sha256) = 64 AND plan_sha256 NOT GLOB '*[^0-9a-f]*'),
    sources_json TEXT NOT NULL
        CHECK (json_valid(sources_json) AND json_type(sources_json) = 'array'),
    engine_version TEXT NOT NULL CHECK (length(trim(engine_version)) > 0),
    warnings_json TEXT NOT NULL
        CHECK (json_valid(warnings_json) AND json_type(warnings_json) = 'array'),
    actor_label TEXT NOT NULL CHECK (length(trim(actor_label)) > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analysis_tables_project ON analysis_tables(project_id, created_at);

CREATE TRIGGER analysis_tables_no_update BEFORE UPDATE ON analysis_tables
BEGIN SELECT RAISE(ABORT, 'analysis_tables is append-only'); END;
CREATE TRIGGER analysis_tables_no_delete BEFORE DELETE ON analysis_tables
BEGIN SELECT RAISE(ABORT, 'analysis_tables is append-only'); END;
