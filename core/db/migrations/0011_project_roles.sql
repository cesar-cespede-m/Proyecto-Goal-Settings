-- V2-09: responsable y aprobador declarados del proyecto, con historial append-only.

CREATE TABLE project_roles (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    version_no INTEGER NOT NULL CHECK (version_no > 0),
    owner_label TEXT NOT NULL CHECK (
        length(trim(owner_label)) BETWEEN 1 AND 200
    ),
    approver_label TEXT NOT NULL CHECK (
        length(trim(approver_label)) BETWEEN 1 AND 200
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, version_no)
);

CREATE INDEX idx_project_roles_current
ON project_roles(project_id, version_no DESC);

CREATE TRIGGER project_roles_no_update BEFORE UPDATE ON project_roles
BEGIN SELECT RAISE(ABORT, 'project_roles is append-only'); END;
CREATE TRIGGER project_roles_no_delete BEFORE DELETE ON project_roles
BEGIN SELECT RAISE(ABORT, 'project_roles is append-only'); END;
