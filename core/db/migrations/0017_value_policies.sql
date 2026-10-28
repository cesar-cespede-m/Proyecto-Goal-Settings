-- MV-10: política y revisión humana independientes del gate predictivo.
CREATE TABLE policy_versions (
    id TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL REFERENCES capabilities(id) ON DELETE RESTRICT,
    version_no INTEGER NOT NULL CHECK(version_no > 0),
    based_on_version_id TEXT REFERENCES policy_versions(id) ON DELETE RESTRICT,
    content_json TEXT NOT NULL CHECK(json_valid(content_json)),
    sha256 TEXT NOT NULL CHECK(length(sha256)=64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    actor_label TEXT NOT NULL CHECK(length(trim(actor_label)) BETWEEN 1 AND 200),
    created_at TEXT NOT NULL,
    UNIQUE(capability_id, version_no)
);
CREATE TABLE policy_review_events (
    id TEXT PRIMARY KEY,
    policy_version_id TEXT NOT NULL REFERENCES policy_versions(id) ON DELETE RESTRICT,
    approved_hash TEXT NOT NULL CHECK(length(approved_hash)=64),
    decision TEXT NOT NULL CHECK(decision IN ('APPROVED','REJECTED')),
    actor_label TEXT NOT NULL CHECK(length(trim(actor_label)) BETWEEN 1 AND 200),
    created_at TEXT NOT NULL
);
CREATE TABLE policy_operations (
    operation_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL CHECK(event_type IN ('POLICY_PROPOSED','POLICY_REVIEWED')),
    entity_id TEXT NOT NULL,
    request_sha256 TEXT NOT NULL CHECK(length(request_sha256)=64),
    actor_label TEXT NOT NULL CHECK(length(trim(actor_label)) BETWEEN 1 AND 200),
    created_at TEXT NOT NULL
);
CREATE TRIGGER policy_versions_parent BEFORE INSERT ON policy_versions
WHEN NEW.based_on_version_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM policy_versions p WHERE p.id=NEW.based_on_version_id
    AND p.capability_id=NEW.capability_id)
BEGIN SELECT RAISE(ABORT,'policy parent capability mismatch'); END;
CREATE TRIGGER policy_versions_no_update BEFORE UPDATE ON policy_versions
BEGIN SELECT RAISE(ABORT,'policy append-only'); END;
CREATE TRIGGER policy_versions_no_delete BEFORE DELETE ON policy_versions
BEGIN SELECT RAISE(ABORT,'policy append-only'); END;
CREATE TRIGGER policy_review_no_update BEFORE UPDATE ON policy_review_events
BEGIN SELECT RAISE(ABORT,'policy review append-only'); END;
CREATE TRIGGER policy_review_no_delete BEFORE DELETE ON policy_review_events
BEGIN SELECT RAISE(ABORT,'policy review append-only'); END;
