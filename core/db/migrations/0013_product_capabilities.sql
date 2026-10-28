-- MV-04: jerarquía aditiva. Cada ciclo posee un proyecto técnico distinto.
CREATE TABLE products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL CHECK (length(trim(name)) BETWEEN 1 AND 200),
    description TEXT NOT NULL CHECK (length(description) <= 4000),
    stage TEXT NOT NULL CHECK (stage IN ('building', 'operating')),
    owner_label TEXT CHECK (owner_label IS NULL OR length(trim(owner_label)) BETWEEN 1 AND 200),
    status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
    created_at TEXT NOT NULL
);

CREATE TABLE capabilities (
    id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    name TEXT NOT NULL CHECK (length(trim(name)) BETWEEN 1 AND 200),
    objective TEXT NOT NULL CHECK (length(trim(objective)) BETWEEN 1 AND 4000),
    value_kind TEXT NOT NULL CHECK (value_kind IN ('revenue', 'savings')),
    decision_unit TEXT NOT NULL CHECK (length(trim(decision_unit)) BETWEEN 1 AND 200),
    status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
    created_at TEXT NOT NULL
);

CREATE TABLE capability_cycles (
    id TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL REFERENCES capabilities(id) ON DELETE RESTRICT,
    project_id TEXT NOT NULL UNIQUE REFERENCES projects(id) ON DELETE RESTRICT,
    cycle_no INTEGER NOT NULL CHECK (cycle_no >= 1),
    origin TEXT NOT NULL CHECK (origin IN ('initial', 'update')),
    source_cycle_id TEXT REFERENCES capability_cycles(id) ON DELETE RESTRICT,
    predictive_spec_id TEXT,
    reason TEXT NOT NULL CHECK (length(trim(reason)) BETWEEN 1 AND 2000),
    created_at TEXT NOT NULL,
    UNIQUE (capability_id, cycle_no),
    CHECK ((origin = 'initial' AND source_cycle_id IS NULL)
        OR (origin = 'update' AND source_cycle_id IS NOT NULL))
);

-- No requiere dataset ni project_id: también audita productos sin datos.
CREATE TABLE product_events (
    id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    capability_id TEXT REFERENCES capabilities(id) ON DELETE RESTRICT,
    cycle_id TEXT REFERENCES capability_cycles(id) ON DELETE RESTRICT,
    event_type TEXT NOT NULL CHECK (event_type IN
        ('PRODUCT_CREATED', 'CAPABILITY_CREATED', 'CAPABILITY_CYCLE_CREATED')),
    entity_id TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE CHECK (length(trim(operation_id)) BETWEEN 1 AND 200),
    request_sha256 TEXT NOT NULL CHECK (
        length(request_sha256) = 64 AND request_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    actor_label TEXT NOT NULL CHECK (length(trim(actor_label)) BETWEEN 1 AND 200),
    detail_json TEXT NOT NULL CHECK (json_valid(detail_json) AND json_type(detail_json) = 'object'),
    created_at TEXT NOT NULL,
    CHECK ((event_type = 'PRODUCT_CREATED' AND entity_id = product_id
                AND capability_id IS NULL AND cycle_id IS NULL)
        OR (event_type = 'CAPABILITY_CREATED' AND entity_id = capability_id
                AND capability_id IS NOT NULL AND cycle_id IS NULL)
        OR (event_type = 'CAPABILITY_CYCLE_CREATED' AND entity_id = cycle_id
                AND capability_id IS NOT NULL AND cycle_id IS NOT NULL))
);

CREATE INDEX idx_capabilities_product ON capabilities(product_id, created_at);
CREATE INDEX idx_capability_cycles_capability ON capability_cycles(capability_id, cycle_no);
CREATE INDEX idx_product_events_product ON product_events(product_id, created_at);

CREATE TRIGGER capability_cycle_source_matches BEFORE INSERT ON capability_cycles
FOR EACH ROW WHEN NEW.source_cycle_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM capability_cycles WHERE id = NEW.source_cycle_id
        AND capability_id = NEW.capability_id AND cycle_no < NEW.cycle_no
)
BEGIN SELECT RAISE(ABORT, 'cycle source capability mismatch'); END;

CREATE TRIGGER product_event_scope_matches BEFORE INSERT ON product_events
FOR EACH ROW WHEN (
    NEW.capability_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM capabilities WHERE id = NEW.capability_id AND product_id = NEW.product_id
    )
) OR (
    NEW.cycle_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM capability_cycles WHERE id = NEW.cycle_id
            AND capability_id = NEW.capability_id
    )
)
BEGIN SELECT RAISE(ABORT, 'product event scope mismatch'); END;

CREATE TRIGGER products_no_update BEFORE UPDATE ON products
BEGIN SELECT RAISE(ABORT, 'products is append-only'); END;
CREATE TRIGGER products_no_delete BEFORE DELETE ON products
BEGIN SELECT RAISE(ABORT, 'products is append-only'); END;
CREATE TRIGGER capabilities_no_update BEFORE UPDATE ON capabilities
BEGIN SELECT RAISE(ABORT, 'capabilities is append-only'); END;
CREATE TRIGGER capabilities_no_delete BEFORE DELETE ON capabilities
BEGIN SELECT RAISE(ABORT, 'capabilities is append-only'); END;
CREATE TRIGGER capability_cycles_no_update BEFORE UPDATE ON capability_cycles
BEGIN SELECT RAISE(ABORT, 'capability_cycles is append-only'); END;
CREATE TRIGGER capability_cycles_no_delete BEFORE DELETE ON capability_cycles
BEGIN SELECT RAISE(ABORT, 'capability_cycles is append-only'); END;
CREATE TRIGGER product_events_no_update BEFORE UPDATE ON product_events
BEGIN SELECT RAISE(ABORT, 'product_events is append-only'); END;
CREATE TRIGGER product_events_no_delete BEFORE DELETE ON product_events
BEGIN SELECT RAISE(ABORT, 'product_events is append-only'); END;
