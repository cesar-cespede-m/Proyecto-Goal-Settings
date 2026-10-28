-- GA-13: notas libres del negocio por proyecto, versionadas y sólo de inserción.
-- Viven fuera de las decisiones confirmadas para no alterar sus huellas aprobadas.

CREATE TABLE project_notes (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    body TEXT NOT NULL CHECK (length(body) <= 4000),
    actor_label TEXT NOT NULL CHECK (length(trim(actor_label)) > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_project_notes_project ON project_notes(project_id, created_at);

CREATE TRIGGER project_notes_no_update BEFORE UPDATE ON project_notes
BEGIN SELECT RAISE(ABORT, 'project_notes is append-only'); END;
CREATE TRIGGER project_notes_no_delete BEFORE DELETE ON project_notes
BEGIN SELECT RAISE(ABORT, 'project_notes is append-only'); END;
