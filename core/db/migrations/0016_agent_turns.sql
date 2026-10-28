-- MV-07: sólo hashes del prompt/respuesta y propuesta validada; nunca texto crudo del proveedor.
CREATE TABLE agent_turns (
    operation_id TEXT PRIMARY KEY REFERENCES tool_operations(operation_id) ON DELETE RESTRICT,
    repair_of_operation_id TEXT UNIQUE REFERENCES agent_turns(operation_id) ON DELETE RESTRICT,
    context_sha256 TEXT NOT NULL CHECK(length(context_sha256)=64 AND context_sha256 NOT GLOB '*[^0-9a-f]*'),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL CHECK(length(prompt_sha256)=64 AND prompt_sha256 NOT GLOB '*[^0-9a-f]*'),
    response_recorded INTEGER NOT NULL DEFAULT 0 CHECK(response_recorded IN (0,1)),
    actual_provider TEXT,
    actual_model TEXT,
    response_id TEXT,
    response_sha256 TEXT CHECK(response_sha256 IS NULL OR
        (length(response_sha256)=64 AND response_sha256 NOT GLOB '*[^0-9a-f]*')),
    input_tokens INTEGER CHECK(input_tokens IS NULL OR input_tokens >= 0),
    output_tokens INTEGER CHECK(output_tokens IS NULL OR output_tokens >= 0),
    latency_ms INTEGER CHECK(latency_ms IS NULL OR latency_ms >= 0),
    proposal_json TEXT CHECK(proposal_json IS NULL OR json_valid(proposal_json)),
    proposal_sha256 TEXT CHECK(proposal_sha256 IS NULL OR
        (length(proposal_sha256)=64 AND proposal_sha256 NOT GLOB '*[^0-9a-f]*')),
    error_code TEXT,
    dispatch_result_json TEXT CHECK(dispatch_result_json IS NULL OR json_valid(dispatch_result_json)),
    dispatch_sha256 TEXT CHECK(dispatch_sha256 IS NULL OR length(dispatch_sha256)=64),
    dispatched_at TEXT,
    CHECK((proposal_json IS NULL) = (proposal_sha256 IS NULL)),
    CHECK((dispatch_result_json IS NULL) = (dispatch_sha256 IS NULL)),
    CHECK((dispatch_result_json IS NULL) = (dispatched_at IS NULL))
);
CREATE TRIGGER agent_turn_operation_scope BEFORE INSERT ON agent_turns
WHEN NOT EXISTS (
    SELECT 1 FROM tool_operations o WHERE o.operation_id=NEW.operation_id
        AND json_extract(o.call_json,'$.tool_name')='agent_turn' AND o.calls_cost=1 AND o.status='CLAIMED'
) OR (NEW.repair_of_operation_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM agent_turns parent
    JOIN tool_operations old ON old.operation_id=parent.operation_id
    JOIN tool_operations current ON current.operation_id=NEW.operation_id
    WHERE parent.operation_id=NEW.repair_of_operation_id AND parent.repair_of_operation_id IS NULL
        AND old.task_id=current.task_id AND old.status='FAILED' AND parent.proposal_json IS NULL
        AND parent.response_recorded=1 AND parent.dispatch_result_json IS NULL
))
BEGIN SELECT RAISE(ABORT,'agent turn operation mismatch'); END;
CREATE TRIGGER agent_turn_identity_immutable BEFORE UPDATE OF
    operation_id,repair_of_operation_id,context_sha256,provider,model,prompt_version,prompt_sha256 ON agent_turns
BEGIN SELECT RAISE(ABORT,'agent turn identity is immutable'); END;
CREATE TRIGGER agent_turn_response_immutable BEFORE UPDATE OF
    response_recorded,actual_provider,actual_model,response_id,response_sha256,input_tokens,
    output_tokens,latency_ms,proposal_json,proposal_sha256,error_code ON agent_turns
WHEN OLD.response_recorded=1
BEGIN SELECT RAISE(ABORT,'agent turn response is immutable'); END;
CREATE TRIGGER agent_turn_dispatch_immutable BEFORE UPDATE OF
    dispatch_result_json,dispatch_sha256,dispatched_at ON agent_turns
WHEN OLD.dispatch_result_json IS NOT NULL
BEGIN SELECT RAISE(ABORT,'agent turn dispatch is immutable'); END;
CREATE TRIGGER agent_turn_no_delete BEFORE DELETE ON agent_turns
BEGIN SELECT RAISE(ABORT,'agent turns cannot be deleted'); END;
