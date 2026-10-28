-- MV-06: supervisor local persistente; no workers ni llamadas de red en transacciones.
CREATE TABLE agent_tasks (
    id TEXT PRIMARY KEY,
    capability_id TEXT NOT NULL REFERENCES capabilities(id) ON DELETE RESTRICT,
    cycle_id TEXT REFERENCES capability_cycles(id) ON DELETE RESTRICT,
    parent_task_id TEXT REFERENCES agent_tasks(id) ON DELETE RESTRICT,
    root_task_id TEXT NOT NULL REFERENCES agent_tasks(id) ON DELETE RESTRICT,
    role TEXT NOT NULL CHECK(role IN ('coordinator','business','data','modeling','value_evidence')),
    objective TEXT CHECK(objective IS NULL OR length(trim(objective)) BETWEEN 1 AND 2000),
    state TEXT NOT NULL CHECK(state IN (
        'PENDING','RUNNING','WAITING_HUMAN','WAITING_DATA','COMPLETED','FAILED','CANCELLED'
    )),
    input_refs_json TEXT NOT NULL CHECK(json_valid(input_refs_json)),
    context_sha256 TEXT NOT NULL CHECK(length(context_sha256)=64 AND context_sha256 NOT GLOB '*[^0-9a-f]*'),
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
    lease_owner TEXT,
    lease_until TEXT,
    last_error_code TEXT,
    actor_label TEXT NOT NULL CHECK(length(trim(actor_label)) > 0),
    create_operation_id TEXT NOT NULL UNIQUE CHECK(length(trim(create_operation_id)) > 0),
    create_request_sha256 TEXT NOT NULL CHECK(length(create_request_sha256)=64),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK((lease_owner IS NULL) = (lease_until IS NULL))
);
CREATE INDEX agent_tasks_root ON agent_tasks(root_task_id,created_at);
CREATE TRIGGER agent_task_scope BEFORE INSERT ON agent_tasks
WHEN (NEW.cycle_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM capability_cycles c WHERE c.id=NEW.cycle_id AND c.capability_id=NEW.capability_id
)) OR (NEW.parent_task_id IS NULL AND NEW.root_task_id != NEW.id)
OR (NEW.parent_task_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM agent_tasks p WHERE p.id=NEW.parent_task_id
        AND p.root_task_id=NEW.root_task_id AND p.capability_id=NEW.capability_id
        AND (p.cycle_id IS NULL OR p.cycle_id=NEW.cycle_id)
))
BEGIN SELECT RAISE(ABORT,'agent task scope mismatch'); END;
CREATE TRIGGER agent_task_identity_immutable BEFORE UPDATE OF
    id,capability_id,cycle_id,parent_task_id,root_task_id,role,objective,input_refs_json,
    actor_label,create_operation_id,create_request_sha256,created_at ON agent_tasks
BEGIN SELECT RAISE(ABORT,'agent task identity is immutable'); END;
CREATE TRIGGER agent_tasks_no_delete BEFORE DELETE ON agent_tasks
BEGIN SELECT RAISE(ABORT,'agent tasks cannot be deleted'); END;

CREATE TABLE agent_budgets (
    root_task_id TEXT PRIMARY KEY REFERENCES agent_tasks(id) ON DELETE RESTRICT,
    max_steps INTEGER NOT NULL CHECK(max_steps BETWEEN 1 AND 1000),
    max_calls INTEGER NOT NULL CHECK(max_calls BETWEEN 0 AND 1000),
    max_seconds INTEGER NOT NULL CHECK(max_seconds BETWEEN 1 AND 3600),
    steps_used INTEGER NOT NULL DEFAULT 0 CHECK(steps_used >= 0),
    calls_used INTEGER NOT NULL DEFAULT 0 CHECK(calls_used >= 0),
    elapsed_ms INTEGER NOT NULL DEFAULT 0 CHECK(elapsed_ms >= 0),
    reserved_ms INTEGER NOT NULL DEFAULT 0 CHECK(reserved_ms >= 0)
);
CREATE TRIGGER agent_budget_root BEFORE INSERT ON agent_budgets
WHEN NOT EXISTS (SELECT 1 FROM agent_tasks t WHERE t.id=NEW.root_task_id AND t.parent_task_id IS NULL)
BEGIN SELECT RAISE(ABORT,'agent budget requires a root task'); END;
CREATE TRIGGER agent_budget_limits_immutable BEFORE UPDATE OF
    root_task_id,max_steps,max_calls,max_seconds ON agent_budgets
BEGIN SELECT RAISE(ABORT,'agent budget limits are immutable'); END;
CREATE TRIGGER agent_budgets_no_delete BEFORE DELETE ON agent_budgets
BEGIN SELECT RAISE(ABORT,'agent budgets cannot be deleted'); END;

CREATE TABLE tool_operations (
    operation_id TEXT PRIMARY KEY CHECK(length(trim(operation_id)) > 0),
    task_id TEXT NOT NULL REFERENCES agent_tasks(id) ON DELETE RESTRICT,
    root_task_id TEXT NOT NULL REFERENCES agent_tasks(id) ON DELETE RESTRICT,
    call_json TEXT NOT NULL CHECK(json_valid(call_json)),
    request_sha256 TEXT NOT NULL CHECK(length(request_sha256)=64 AND request_sha256 NOT GLOB '*[^0-9a-f]*'),
    status TEXT NOT NULL CHECK(status IN ('CLAIMED','COMPLETED','FAILED')),
    task_revision INTEGER NOT NULL CHECK(task_revision >= 0),
    owner TEXT NOT NULL CHECK(length(trim(owner)) > 0),
    lease_until TEXT NOT NULL,
    reserved_ms INTEGER NOT NULL CHECK(reserved_ms > 0),
    calls_cost INTEGER NOT NULL CHECK(calls_cost >= 0),
    result_json TEXT CHECK(result_json IS NULL OR json_valid(result_json)),
    created_at TEXT NOT NULL,
    finished_at TEXT,
    CHECK((status='CLAIMED' AND result_json IS NULL AND finished_at IS NULL) OR
          (status!='CLAIMED' AND result_json IS NOT NULL AND finished_at IS NOT NULL))
);
CREATE UNIQUE INDEX one_claim_per_agent_root ON tool_operations(root_task_id) WHERE status='CLAIMED';
CREATE TRIGGER tool_operation_scope BEFORE INSERT ON tool_operations
WHEN NOT EXISTS (SELECT 1 FROM agent_tasks t WHERE t.id=NEW.task_id AND t.root_task_id=NEW.root_task_id)
BEGIN SELECT RAISE(ABORT,'tool operation scope mismatch'); END;
CREATE TRIGGER tool_operation_identity_immutable BEFORE UPDATE OF
    operation_id,task_id,root_task_id,call_json,request_sha256,task_revision,owner,
    lease_until,reserved_ms,calls_cost,created_at ON tool_operations
BEGIN SELECT RAISE(ABORT,'tool operation identity is immutable'); END;
CREATE TRIGGER tool_operation_terminal_immutable BEFORE UPDATE ON tool_operations
WHEN OLD.status != 'CLAIMED'
BEGIN SELECT RAISE(ABORT,'tool operation outcome is immutable'); END;
CREATE TRIGGER tool_operations_no_delete BEFORE DELETE ON tool_operations
BEGIN SELECT RAISE(ABORT,'tool operations cannot be deleted'); END;

CREATE TABLE orchestration_events (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES agent_tasks(id) ON DELETE RESTRICT,
    event_type TEXT NOT NULL,
    actor_label TEXT NOT NULL CHECK(length(trim(actor_label)) > 0),
    detail_json TEXT NOT NULL CHECK(json_valid(detail_json)),
    created_at TEXT NOT NULL
);
CREATE TRIGGER orchestration_events_no_update BEFORE UPDATE ON orchestration_events
BEGIN SELECT RAISE(ABORT,'orchestration events are append-only'); END;
CREATE TRIGGER orchestration_events_no_delete BEFORE DELETE ON orchestration_events
BEGIN SELECT RAISE(ABORT,'orchestration events are append-only'); END;

-- Vínculo creado en la misma transacción que runs; un retry recupera el mismo intento.
CREATE TABLE training_operations (
    operation_id TEXT PRIMARY KEY CHECK(length(trim(operation_id)) > 0),
    strategy_version_id TEXT NOT NULL REFERENCES strategy_versions(id) ON DELETE RESTRICT,
    run_id TEXT NOT NULL UNIQUE REFERENCES runs(id) ON DELETE RESTRICT,
    request_sha256 TEXT NOT NULL CHECK(length(request_sha256)=64 AND request_sha256 NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TRIGGER training_operation_scope BEFORE INSERT ON training_operations
WHEN NOT EXISTS (SELECT 1 FROM runs r WHERE r.id=NEW.run_id AND r.strategy_version_id=NEW.strategy_version_id)
BEGIN SELECT RAISE(ABORT,'training operation strategy mismatch'); END;
CREATE TRIGGER training_operations_no_update BEFORE UPDATE ON training_operations
BEGIN SELECT RAISE(ABORT,'training operations are append-only'); END;
CREATE TRIGGER training_operations_no_delete BEFORE DELETE ON training_operations
BEGIN SELECT RAISE(ABORT,'training operations are append-only'); END;
