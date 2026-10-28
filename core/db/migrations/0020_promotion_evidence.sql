-- MV-13: cohorte sellada, comparación y elección humana append-only.
CREATE TABLE promotion_cohorts (
    id TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL REFERENCES capabilities(id) ON DELETE RESTRICT,
    current_policy_id TEXT NOT NULL REFERENCES policy_versions(id) ON DELETE RESTRICT,
    candidate_policy_id TEXT NOT NULL REFERENCES policy_versions(id) ON DELETE RESTRICT,
    input_sha256 TEXT NOT NULL CHECK(length(input_sha256)=64),
    manifest_json TEXT NOT NULL CHECK(json_valid(manifest_json)),
    sha256 TEXT NOT NULL CHECK(length(sha256)=64),
    actor_label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(capability_id,input_sha256)
);
CREATE TABLE promotion_evaluations (
    id TEXT PRIMARY KEY,
    cohort_id TEXT NOT NULL UNIQUE REFERENCES promotion_cohorts(id) ON DELETE RESTRICT,
    evaluation_json TEXT NOT NULL CHECK(json_valid(evaluation_json)),
    sha256 TEXT NOT NULL CHECK(length(sha256)=64),
    created_at TEXT NOT NULL
);
CREATE TABLE promotion_reviews (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL REFERENCES promotion_evaluations(id) ON DELETE RESTRICT,
    approved_hash TEXT NOT NULL CHECK(length(approved_hash)=64),
    decision TEXT NOT NULL CHECK(decision IN ('APPROVED','REJECTED')),
    actor_label TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE capability_model_assignments (
    id TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL REFERENCES capabilities(id) ON DELETE RESTRICT,
    previous_policy_id TEXT NOT NULL REFERENCES policy_versions(id) ON DELETE RESTRICT,
    policy_version_id TEXT NOT NULL REFERENCES policy_versions(id) ON DELETE RESTRICT,
    evaluation_id TEXT NOT NULL UNIQUE REFERENCES promotion_evaluations(id) ON DELETE RESTRICT,
    review_id TEXT NOT NULL UNIQUE REFERENCES promotion_reviews(id) ON DELETE RESTRICT,
    actor_label TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_promotion_cohorts_cap ON promotion_cohorts(capability_id,created_at);
CREATE INDEX idx_model_assignments_cap ON capability_model_assignments(capability_id,created_at);
CREATE TRIGGER promotion_cohorts_no_update BEFORE UPDATE ON promotion_cohorts
BEGIN SELECT RAISE(ABORT,'promotion cohorts append-only'); END;
CREATE TRIGGER promotion_cohorts_no_delete BEFORE DELETE ON promotion_cohorts
BEGIN SELECT RAISE(ABORT,'promotion cohorts append-only'); END;
CREATE TRIGGER promotion_evaluations_no_update BEFORE UPDATE ON promotion_evaluations
BEGIN SELECT RAISE(ABORT,'promotion evaluations append-only'); END;
CREATE TRIGGER promotion_evaluations_no_delete BEFORE DELETE ON promotion_evaluations
BEGIN SELECT RAISE(ABORT,'promotion evaluations append-only'); END;
CREATE TRIGGER promotion_reviews_no_update BEFORE UPDATE ON promotion_reviews
BEGIN SELECT RAISE(ABORT,'promotion reviews append-only'); END;
CREATE TRIGGER promotion_reviews_no_delete BEFORE DELETE ON promotion_reviews
BEGIN SELECT RAISE(ABORT,'promotion reviews append-only'); END;
CREATE TRIGGER capability_model_assignments_no_update BEFORE UPDATE ON capability_model_assignments
BEGIN SELECT RAISE(ABORT,'model assignments append-only'); END;
CREATE TRIGGER capability_model_assignments_no_delete BEFORE DELETE ON capability_model_assignments
BEGIN SELECT RAISE(ABORT,'model assignments append-only'); END;
