-- MV-11: lote privado y hechos de intervención separados, append-only.
CREATE TABLE prediction_batches (
    id TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL REFERENCES capabilities(id) ON DELETE RESTRICT,
    policy_version_id TEXT NOT NULL REFERENCES policy_versions(id) ON DELETE RESTRICT,
    scoring_event_id TEXT NOT NULL UNIQUE REFERENCES audit_events(id) ON DELETE RESTRICT,
    period TEXT NOT NULL,
    batch_json TEXT NOT NULL CHECK(json_valid(batch_json)),
    sha256 TEXT NOT NULL CHECK(length(sha256)=64),
    actor_label TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE action_observations (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES prediction_batches(id) ON DELETE RESTRICT,
    case_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('ACTED','OUTCOME_PENDING','OUTCOME_OBSERVED')),
    event_json TEXT NOT NULL CHECK(json_valid(event_json)),
    actor_label TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE prediction_batch_operations (
    operation_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    request_sha256 TEXT NOT NULL CHECK(length(request_sha256)=64),
    created_at TEXT NOT NULL
);
CREATE INDEX idx_prediction_batches_scope ON prediction_batches(capability_id,period);
CREATE INDEX idx_action_observations_case ON action_observations(batch_id,case_id,created_at);
CREATE TRIGGER prediction_batch_policy_scope BEFORE INSERT ON prediction_batches
WHEN NOT EXISTS (SELECT 1 FROM policy_versions p WHERE p.id=NEW.policy_version_id
                 AND p.capability_id=NEW.capability_id)
BEGIN SELECT RAISE(ABORT,'prediction batch policy scope mismatch'); END;
CREATE TRIGGER prediction_batches_no_update BEFORE UPDATE ON prediction_batches
BEGIN SELECT RAISE(ABORT,'prediction batches append-only'); END;
CREATE TRIGGER prediction_batches_no_delete BEFORE DELETE ON prediction_batches
BEGIN SELECT RAISE(ABORT,'prediction batches append-only'); END;
CREATE TRIGGER action_observations_no_update BEFORE UPDATE ON action_observations
BEGIN SELECT RAISE(ABORT,'action observations append-only'); END;
CREATE TRIGGER action_observations_no_delete BEFORE DELETE ON action_observations
BEGIN SELECT RAISE(ABORT,'action observations append-only'); END;
CREATE TRIGGER prediction_batch_operations_no_update BEFORE UPDATE ON prediction_batch_operations
BEGIN SELECT RAISE(ABORT,'prediction operations append-only'); END;
CREATE TRIGGER prediction_batch_operations_no_delete BEFORE DELETE ON prediction_batch_operations
BEGIN SELECT RAISE(ABORT,'prediction operations append-only'); END;
