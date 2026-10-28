-- C2-01: `id` es la huella del contenido del evento: reenviar exactamente lo mismo no
-- duplica, pero una versión nueva del mismo caso (por ejemplo, con su resultado) sí se
-- guarda. Al convertir a archivo se usa la versión más reciente de cada caso.
CREATE TABLE IF NOT EXISTS observation_events (
    id TEXT NOT NULL CHECK (length(trim(id)) > 0),
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    source TEXT NOT NULL CHECK (length(trim(source)) > 0),
    entity TEXT NOT NULL CHECK (length(trim(entity)) > 0),
    entity_id TEXT NOT NULL CHECK (length(trim(entity_id)) > 0),
    occurred_at TEXT NOT NULL CHECK (length(trim(occurred_at)) > 0),
    attributes_json TEXT NOT NULL
        CHECK (json_valid(attributes_json) AND json_type(attributes_json) = 'object'),
    outcome_json TEXT CHECK (outcome_json IS NULL OR json_valid(outcome_json)),
    received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (project_id, id)
);

CREATE INDEX IF NOT EXISTS idx_observation_events_project_entity
    ON observation_events(project_id, entity, occurred_at);

CREATE TRIGGER IF NOT EXISTS observation_events_no_update
BEFORE UPDATE ON observation_events
BEGIN
    SELECT RAISE(ABORT, 'observation_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS observation_events_no_delete
BEFORE DELETE ON observation_events
BEGIN
    SELECT RAISE(ABORT, 'observation_events is append-only');
END;
