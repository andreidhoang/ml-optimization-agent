# cosmos_lab/harness — Adapter Contract

> Per PLAN_V2.md §0.4 (library architecture) + §0.4.5 (two-layer decision) + §0.6 (v6 — 9 agents on ml-intern primitives), cosmos-lab is a Python library that plugs into agent runtimes via thin adapters. This document specifies the contract every adapter must satisfy.

## Two adapter families (v5.1/v6 reality)

The two shipped adapters serve different purposes and have different contracts. Honest naming:

### Family A — Execution Substrate Adapter

**Purpose**: install cosmos-lab governance (capability-scoped tool calls + identity-aware audit) INTO an autonomous agent runtime. The host runs the actual ReAct loop; cosmos-lab governs every tool call inside it.

**Examples**:
- `cosmos_lab.harness.ml_intern.install_into_session` (P0.5 D2 — shipped)
- `cosmos_lab.harness.claude_sdk.install_into_agent` (v1.1 — planned)
- `cosmos_lab.harness.openai_agents.install_into_agent` (v1.2 — planned)

**Contract signature**:
```python
def install(host: HostType, identity: AgentIdentity, audit_log: AuditLog) -> None:
    """Wrap host's tool router with CapabilityScopedRouter."""
```

**Behavior**:
1. Validate host has the prerequisite (tool_router for Session, equivalent for other hosts)
2. Wrap host's tool router with `CapabilityScopedRouter(identity, audit_log)`
3. Mark host as installed (idempotency tag)
4. Mutate host in place; return `None`

### Family B — Deployment Surface Adapter

**Purpose**: register cosmos-lab as an invokable tool in a workflow runtime. The workflow runtime invokes cosmos-lab CLI; cosmos-lab CLI orchestrates the actual work using the family A execution substrate adapter.

**Examples**:
- `cosmos_lab.harness.nat.register_as_nat_tool` (P0.5 D3 — shipped)
- `cosmos_lab.harness.langgraph.register_as_langgraph_node` (v1.2 — planned)
- `cosmos_lab.harness.airflow.register_as_airflow_operator` (future)

**Contract signature**:
```python
def register_as_X_tool(builder: BuilderType) -> None:
    """Register cosmos_lab_principal as an invokable tool in the workflow builder."""
```

**Behavior**:
1. Validate builder has a recognizable tool-registration method
2. Register one tool named `cosmos_lab_principal` (uniform across deployment surfaces for predictability)
3. Tool callable accepts `(task: str, budget_usd: float, timeout_sec: int)`, returns structured dict
4. Mark builder as registered (idempotency check)
5. Mutate builder in place; return `None`

## Shared requirements (both families)

These requirements apply to ALL adapters regardless of family. Tested cross-family in `test_adapter_contract.py`.

### S1 — Idempotency
Re-installing/re-registering on the same host raises a clear error (RuntimeError or equivalent) with explanation. Silent re-installation could shadow audit history or duplicate tool registrations — both are correctness bugs.

### S2 — Composition only (Invariant 1)
Adapter never modifies upstream files. Wraps existing host attributes via composition. Adapter must work against any duck-typed host that satisfies the protocol.

### S3 — Input validation
Adapter raises a clear error (ValueError, AttributeError, TypeError) when host lacks prerequisites. Errors must name the specific missing prerequisite to aid debugging.

### S4 — In-place mutation, returns None
Adapter mutates the host in place and returns None. No new host instance returned. This makes adapter usage uniform: `adapter.install(host, ...); use(host)`.

### S5 — No partial state on failure
If adapter raises during install/register, host must remain unchanged from pre-call state. No half-installed governance. (Atomicity guarantee.)

## Per-adapter specifics

### `cosmos_lab.harness.ml_intern.install_into_session`

- **HostType**: `agent.core.session.Session` (or duck-typed equivalent with `.tool_router`)
- **Prerequisite**: `host.tool_router is not None`
- **Wraps**: `host.tool_router` with `CapabilityScopedRouter`
- **Idempotency tag**: `host._cosmos_lab_installed = True`
- **Tested in**: `tests/optimization/harness/test_ml_intern_adapter.py` (6 tests)

### `cosmos_lab.harness.nat.register_as_nat_tool`

- **HostType**: nat `Builder` (or duck-typed `BuilderLike` with one of `add_function`/`register_function`/`register_tool`/`add_tool`)
- **Prerequisite**: builder has a recognizable registration method
- **Registers**: one tool named `cosmos_lab_principal`
- **Idempotency check**: `cosmos_lab_principal` not already in `builder.functions`/`tools`/`_functions`/`_tools`
- **Tool body**: stub in P0.5 D3; real CLI invocation lands in P3 (PrincipalAgent v0 ships specialty agents)
- **Tested in**: `tests/optimization/harness/test_nat_adapter.py` (11 tests)

## Future adapters (v1.1+)

When adding a new adapter:

1. **Determine family**: Family A (execution substrate) or Family B (deployment surface)?
2. **Match family contract signature**: install vs register
3. **Satisfy all 5 shared requirements** (S1-S5)
4. **Document per-adapter specifics** in this file
5. **Add adapter-specific test file**: `tests/optimization/harness/test_X_adapter.py`
6. **Add cross-family parametrization** in `test_adapter_contract.py` for the shared requirements (S1-S5)

## Anti-patterns explicitly rejected

- ❌ One unified adapter signature for both families — they serve different purposes, forcing one signature loses clarity
- ❌ Abstract base class enforcing contract — duck typing + Protocol + test suite is more flexible (Pythonic)
- ❌ Adapters that modify upstream files — violates Invariant 1
- ❌ Silent re-installation — shadows audit history
- ❌ Partial-state failure — host left in inconsistent state on error

## References

- PLAN_V2.md §0.4 — library architecture decision
- PLAN_V2.md §0.4.5 — two-layer architecture (cosmos-lab CLI + ml-intern Session, nat as deployment wrapper)
- PLAN_V2.md §0.6 (v6) — what cosmos-lab ships vs leverages from ml-intern
- `cosmos_lab/harness/ml_intern.py` — Family A reference implementation
- `cosmos_lab/harness/nat.py` — Family B reference implementation
