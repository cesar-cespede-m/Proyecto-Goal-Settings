-- MV-12: candidata aislada vinculada al ciclo y archivo de actualización.
CREATE TABLE capability_update_operations (
    operation_id TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL REFERENCES capabilities(id) ON DELETE RESTRICT,
    source_run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE RESTRICT,
    source_data_sha256 TEXT NOT NULL CHECK(length(source_data_sha256)=64),
    request_sha256 TEXT NOT NULL CHECK(length(request_sha256)=64),
    candidate_cycle_id TEXT NOT NULL UNIQUE REFERENCES capability_cycles(id) ON DELETE RESTRICT,
    candidate_dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
    actor_label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(capability_id,source_run_id,source_data_sha256)
);
CREATE TRIGGER capability_update_operations_no_update BEFORE UPDATE ON capability_update_operations
BEGIN SELECT RAISE(ABORT,'capability updates append-only'); END;
CREATE TRIGGER capability_update_operations_no_delete BEFORE DELETE ON capability_update_operations
BEGIN SELECT RAISE(ABORT,'capability updates append-only'); END;
