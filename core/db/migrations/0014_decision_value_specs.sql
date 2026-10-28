-- MV-05: especificaciones independientes; no modifica context_versions ni sus gates.

CREATE TABLE predictive_spec_versions (
    id TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL REFERENCES capabilities(id) ON DELETE RESTRICT,
    version_no INTEGER NOT NULL CHECK(version_no > 0),
    based_on_version_id TEXT REFERENCES predictive_spec_versions(id) ON DELETE RESTRICT,
    source_context_version_id TEXT REFERENCES context_versions(id) ON DELETE RESTRICT,
    content_json TEXT NOT NULL CHECK(json_valid(content_json)),
    sha256 TEXT NOT NULL CHECK(length(sha256)=64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    confirmed_human INTEGER NOT NULL CHECK(confirmed_human IN (0,1)),
    actor_label TEXT NOT NULL CHECK(length(trim(actor_label)) > 0),
    created_at TEXT NOT NULL,
    UNIQUE(capability_id,version_no)
);
CREATE TRIGGER predictive_spec_versions_parent_scope BEFORE INSERT ON predictive_spec_versions
WHEN NEW.based_on_version_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM predictive_spec_versions old
    WHERE old.id=NEW.based_on_version_id AND old.capability_id=NEW.capability_id
)
BEGIN SELECT RAISE(ABORT,'spec version capability mismatch'); END;

CREATE TABLE economic_scenario_versions (
    id TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL REFERENCES capabilities(id) ON DELETE RESTRICT,
    version_no INTEGER NOT NULL CHECK(version_no > 0),
    based_on_version_id TEXT REFERENCES economic_scenario_versions(id) ON DELETE RESTRICT,
    content_json TEXT NOT NULL CHECK(json_valid(content_json)),
    sha256 TEXT NOT NULL CHECK(length(sha256)=64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    confirmed_human INTEGER NOT NULL CHECK(confirmed_human IN (0,1)),
    actor_label TEXT NOT NULL CHECK(length(trim(actor_label)) > 0),
    created_at TEXT NOT NULL,
    UNIQUE(capability_id,version_no)
);
CREATE TRIGGER economic_scenario_versions_parent_scope BEFORE INSERT ON economic_scenario_versions
WHEN NEW.based_on_version_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM economic_scenario_versions old
    WHERE old.id=NEW.based_on_version_id AND old.capability_id=NEW.capability_id
)
BEGIN SELECT RAISE(ABORT,'spec version capability mismatch'); END;

CREATE TABLE decision_value_spec_versions (
    id TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL REFERENCES capabilities(id) ON DELETE RESTRICT,
    version_no INTEGER NOT NULL CHECK(version_no > 0),
    based_on_version_id TEXT REFERENCES decision_value_spec_versions(id) ON DELETE RESTRICT,
    predictive_version_id TEXT REFERENCES predictive_spec_versions(id) ON DELETE RESTRICT,
    economic_version_id TEXT REFERENCES economic_scenario_versions(id) ON DELETE RESTRICT,
    content_json TEXT NOT NULL CHECK(json_valid(content_json)),
    sha256 TEXT NOT NULL CHECK(length(sha256)=64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    confirmed_human INTEGER NOT NULL CHECK(confirmed_human IN (0,1)),
    actor_label TEXT NOT NULL CHECK(length(trim(actor_label)) > 0),
    created_at TEXT NOT NULL,
    UNIQUE(capability_id,version_no)
);
CREATE TRIGGER decision_value_spec_versions_parent_scope BEFORE INSERT ON decision_value_spec_versions
WHEN NEW.based_on_version_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM decision_value_spec_versions old
    WHERE old.id=NEW.based_on_version_id AND old.capability_id=NEW.capability_id
)
BEGIN SELECT RAISE(ABORT,'spec version capability mismatch'); END;

CREATE TABLE cycle_predictive_bindings (
    id TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL REFERENCES capabilities(id) ON DELETE RESTRICT,
    cycle_id TEXT NOT NULL UNIQUE REFERENCES capability_cycles(id) ON DELETE RESTRICT,
    predictive_version_id TEXT NOT NULL REFERENCES predictive_spec_versions(id) ON DELETE RESTRICT,
    actor_label TEXT NOT NULL CHECK(length(trim(actor_label)) > 0),
    created_at TEXT NOT NULL
);
CREATE TABLE value_spec_events (
    id TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL REFERENCES capabilities(id) ON DELETE RESTRICT,
    event_type TEXT NOT NULL CHECK(event_type IN (
        'PREDICTIVE_SPEC_CREATED','ECONOMIC_SCENARIO_CREATED','DECISION_VALUE_SPEC_CREATED',
        'PREDICTIVE_SPEC_BOUND'
    )),
    entity_id TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE CHECK(length(trim(operation_id)) > 0),
    request_sha256 TEXT NOT NULL CHECK(length(request_sha256)=64),
    actor_label TEXT NOT NULL CHECK(length(trim(actor_label)) > 0),
    detail_json TEXT NOT NULL CHECK(json_valid(detail_json)),
    created_at TEXT NOT NULL
);
CREATE TRIGGER predictive_context_scope BEFORE INSERT ON predictive_spec_versions
WHEN NEW.source_context_version_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM context_versions context
    JOIN capability_cycles cycle ON cycle.project_id=context.project_id
    WHERE context.id=NEW.source_context_version_id AND cycle.capability_id=NEW.capability_id
)
BEGIN SELECT RAISE(ABORT,'source context capability mismatch'); END;

CREATE TRIGGER economic_currency_consistency BEFORE INSERT ON economic_scenario_versions
WHEN NEW.confirmed_human=1 AND EXISTS (
    SELECT 1 FROM economic_scenario_versions old
    WHERE old.capability_id=NEW.capability_id AND old.confirmed_human=1 AND (
        json_extract(old.content_json,'$.currency') != json_extract(NEW.content_json,'$.currency')
        OR json_extract(old.content_json,'$.currency_minor_units') !=
            json_extract(NEW.content_json,'$.currency_minor_units')
    )
)
BEGIN SELECT RAISE(ABORT,'capability currency cannot change'); END;

CREATE TRIGGER decision_value_reference_scope BEFORE INSERT ON decision_value_spec_versions
WHEN (NEW.predictive_version_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM predictive_spec_versions spec
    WHERE spec.id=NEW.predictive_version_id AND spec.capability_id=NEW.capability_id
        AND (NEW.confirmed_human=0 OR spec.confirmed_human=1)
)) OR (NEW.economic_version_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM economic_scenario_versions spec
    WHERE spec.id=NEW.economic_version_id AND spec.capability_id=NEW.capability_id
        AND (NEW.confirmed_human=0 OR spec.confirmed_human=1)
))
BEGIN SELECT RAISE(ABORT,'decision spec reference mismatch'); END;

CREATE TRIGGER cycle_predictive_binding_scope BEFORE INSERT ON cycle_predictive_bindings
WHEN NOT EXISTS (
    SELECT 1 FROM capability_cycles cycle
    JOIN predictive_spec_versions spec ON spec.capability_id=cycle.capability_id
    WHERE cycle.id=NEW.cycle_id AND spec.id=NEW.predictive_version_id
        AND cycle.capability_id=NEW.capability_id AND spec.confirmed_human=1
)
BEGIN SELECT RAISE(ABORT,'cycle predictive binding mismatch'); END;

CREATE TRIGGER predictive_spec_versions_no_update BEFORE UPDATE ON predictive_spec_versions
BEGIN SELECT RAISE(ABORT,'predictive_spec_versions is append-only'); END;

CREATE TRIGGER predictive_spec_versions_no_delete BEFORE DELETE ON predictive_spec_versions
BEGIN SELECT RAISE(ABORT,'predictive_spec_versions is append-only'); END;

CREATE TRIGGER economic_scenario_versions_no_update BEFORE UPDATE ON economic_scenario_versions
BEGIN SELECT RAISE(ABORT,'economic_scenario_versions is append-only'); END;

CREATE TRIGGER economic_scenario_versions_no_delete BEFORE DELETE ON economic_scenario_versions
BEGIN SELECT RAISE(ABORT,'economic_scenario_versions is append-only'); END;

CREATE TRIGGER decision_value_spec_versions_no_update BEFORE UPDATE ON decision_value_spec_versions
BEGIN SELECT RAISE(ABORT,'decision_value_spec_versions is append-only'); END;

CREATE TRIGGER decision_value_spec_versions_no_delete BEFORE DELETE ON decision_value_spec_versions
BEGIN SELECT RAISE(ABORT,'decision_value_spec_versions is append-only'); END;

CREATE TRIGGER cycle_predictive_bindings_no_update BEFORE UPDATE ON cycle_predictive_bindings
BEGIN SELECT RAISE(ABORT,'cycle_predictive_bindings is append-only'); END;

CREATE TRIGGER cycle_predictive_bindings_no_delete BEFORE DELETE ON cycle_predictive_bindings
BEGIN SELECT RAISE(ABORT,'cycle_predictive_bindings is append-only'); END;

CREATE TRIGGER value_spec_events_no_update BEFORE UPDATE ON value_spec_events
BEGIN SELECT RAISE(ABORT,'value_spec_events is append-only'); END;

CREATE TRIGGER value_spec_events_no_delete BEFORE DELETE ON value_spec_events
BEGIN SELECT RAISE(ABORT,'value_spec_events is append-only'); END;
