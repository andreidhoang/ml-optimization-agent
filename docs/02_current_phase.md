# Current Phase — 🎉 P0.5 COMPLETE → Next: P1 (Eval Infrastructure)

> ⚡ **LIVE FILE** — updated as we move through phases. If this is stale, fix it before doing more work.

**Today's date**: 2026-05-03 (P0 + P0.5 D1/D2/D3/D4 + v3 → v6 plan evolution all shipped same-day)
**Active phase**: **P0.5 COMPLETE** ✅
**Next phase**: **P1 — Eval infrastructure** (foundation for EvalAgent + used by all 6 specialty agents)

---

## P0.5 — DONE (recap of all 4 days)

### D1 — Library restructure ✅ (14/14 verifier)
- `cosmos_lab/__init__.py` + `cosmos_lab/identity/__init__.py` re-export shims
- `pyproject.toml` updated: `cosmos_lab*` in packages.find + `[nat]`/`[ml_intern]`/`[claude_sdk]` extras

### D2 — ml_intern adapter (Family A — execution substrate) ✅ (11/11 verifier)
- `cosmos_lab/harness/ml_intern.py` — `install_into_session()` wraps Session.tool_router with CapabilityScopedRouter
- 6 smoke tests (3 contract + 3 e2e behavior)
- **This is THE primary product surface**: every specialty agent (P3+) constructs ml-intern Sessions and calls `install_into_session` to install governance

### D3 — nat wrapper (Family B — deployment surface) ✅ (10/10 verifier)
- `cosmos_lab/harness/nat.py` — `register_as_nat_tool()` registers `cosmos_lab_principal` as nat workflow tool
- 11 smoke tests covering registration mechanics + tool callable contract
- Tool body is v0 stub; real CLI invocation lands in P3 when PrincipalAgent v0 ships

### D4 — Adapter contract + dual-adapter test matrix ✅ (14/14 verifier)
- `cosmos_lab/harness/CONTRACT.md` — formal contract documentation:
  - Family A vs Family B distinction (per v5.1/v6 architecture)
  - 5 shared requirements (S1 idempotency, S2 composition only, S3 input validation, S4 returns None, S5 no partial state on failure)
  - Per-adapter specifics + future adapter checklist
- `tests/optimization/harness/test_adapter_contract.py` — parametrized contract tests:
  - 9 tests run across BOTH shipped adapters (`ml_intern`, `nat`)
  - When v1.1 adds claude_sdk or langgraph, just add row to ADAPTERS registry — automatic contract enforcement

### v3 → v6 plan evolution (same day, captured in commits)
- v3.1 / v3.2 / v4 / v5 / v5.1 / v5.2 / **v6 (final)** — see PLAN_V2.md §0.5 deltas table rows 1-13
- v6 final framing: 6 Cosmos-specialty agents + 3 governance agents + ~16 infrastructure components, on ml-intern primitives leveraged AS-IS, ~19 weeks

---

## Final P0.5 metrics

| Metric | Value |
|---|---|
| **Verifier scores** | D1: 14/14, D2: 11/11, D3: 10/10, D4: 14/14 (all green) |
| **Total cosmos-lab tests** | 42 (16 P0 + 6 D2 + 11 D3 + 9 D4 contract) |
| **Upstream baseline** | 237 pass / 3 known-broken (no regression across all 4 days) |
| **Total commits on branch** | 12 (P0 + 4 P0.5 days + 5 plan evolutions + 2 fixups) |
| **LOC added** | ~3500 (cosmos_lab/ + tests/ + bin/ + docs/ + planning) |
| **Zero-diff invariant** | ✅ holds throughout |

---

## P0.5 LEARN (cumulative, across all 4 days)

1. **D1 — `uv.lock` regenerates on `uv sync`** → verifier exclusion list must include it
2. **D2 — `uv sync` without `--extra dev` removes pytest** → CLAUDE.md updated
3. **D2 — `uv run pytest` PATH-leaks to system pytest** → use `uv run python -m pytest` always
4. **D2 — Smoke test design** → duck-typed mocks > real host construction (MockSession vs real ml-intern Session)
5. **D3 — Editable install metadata stale** when new submodule added → re-run `uv sync` after package changes
6. **D4 — Two adapter families honest** → `ml_intern` (execution substrate) and `nat` (deployment surface) have DIFFERENT contracts. Forcing one signature loses clarity. Two families + 5 shared requirements is the right design.

---

## P1 spec — Phase 1 of workflow (DEFINE) — eval infrastructure

> **Per PLAN_V2.md §1 v6 phase table**: P1 ships eval infrastructure that becomes the foundation for EvalAgent (P4a) and is used by all 6 specialty agents. Schedule: 2 weeks.

### Goal (one sentence)
Ship the eval infrastructure: `TrajectorySink` Protocol, `OTelGenAIEmitter` (Phoenix backend default), 4 sentinel types per §3.1, `MultiJudge` with bootstrap CIs, Inspect AI bridge, 5 seed Inspect tasks, `evaluate` CLI — so every cosmos-lab specialty agent (P3+) inherits sentinel-gated evaluation + OTel observability + Inspect AI integration for free.

### Spec — what it does (P1 deliverables, ~2w)

**Module: `cosmos_lab/trajectory/`**
- `sink.py` — `TrajectorySink` Protocol
- `otel_emitter.py` — `OTelGenAIEmitter` emits `gen_ai.*` spans → Phoenix
- `duckdb_sink.py` — opt-in analytics layer
- `hf_sink.py` — opt-in HF dataset upload (P8 flywheel)

**Module: `cosmos_lab/eval/`**
- `judge.py` — `LLMJudge` (single-pass)
- `multi_judge.py` — `MultiJudge` (N=3 with bootstrap CI, no debate per arxiv:2508.17536)
- `sentinels/` — 4 sentinel types per §3.1 taxonomy:
  - `deterministic.py`
  - `output_format.py`
  - `side_effect.py`
  - `no_op.py` (mandatory on every task)
- `inspect_bridge.py` — exposes scorers/judges as Inspect AI Scorers
- `tool_judge.py` — ToolAugmentedJudge (judge can call read-only tools)

**Tasks: `tasks/seed/*.py`**
- 5 Inspect AI tasks: dataset inspect, code task, ML debug, paper summary, profiling
- Each ships with one judge scorer + one composed sentinel

**CLI: `agent/optimization/cli/evaluate.py`**
- `cosmos-lab evaluate --suite seed --judge multi --sinks otel,duckdb,hf`

### Acceptance criteria (P1 numerical targets per §0.7)
- 5 Inspect tasks × 3 runs = 15 trajectories: spans land in Phoenix via OTel; round-trip p99 < 500ms
- `MultiJudge` reports bootstrap 95% CI; CI width ≤ 8pp at N=15 runs
- Sentinel/judge agreement ≥ 98% on green seed runs
- A capability-denied call is blocked BEFORE approval policy is consulted (ordering test)
- One golden Phoenix screenshot committed to docs/

### Spec — what it does NOT do (today)
- Does NOT build any of the 6 specialty agents (those are P3+)
- Does NOT add KMS for audit log signing (deferred to P10)
- Does NOT integrate with real Cosmos NIM endpoints (P2 + P9)
- Does NOT include S5 monthly red-team sprint (P8)

### Verifier
`./bin/verify.sh p1` — checks: all modules exist, 5 Inspect tasks load, evaluate CLI works, 5 P1 numerical targets met.

---

## After P1 → v7 phase progression (frontier-aligned final)

> **v7 update**: 3-audit frontier verification confirmed v6 was ~60% aligned with 6 specific issues. v7 fixes them. See PLAN_V2.md §0.5 row 14 for cited rationale.

Per PLAN_V2 §1 v7 phase table:
- P1 (2w): Eval infra — **5 sentinel types** (incl. judge-hacking per Gaia2), **cross-family MultiJudge** (3× Sonnet + 1× non-Anthropic), Inspect AI bridge via Anthropic PostToolUse hooks contract
- P2 (1w): Cosmos toolset (NIMProvider + cosmos_reason/predict/transfer wrappers)
- **P3 (2w): 🤖 PrincipalAgent foundation (NEW v7)** — LangGraph durable supervisor + Magentic-One Task/Progress Ledger pattern + 4-scope hybrid memory (Mem0/Letta)
- **P4a (1.5w): 🤖 DataAgent** (worker #1)
- P4b (2w): Identity v2 (MCP OAuth + RFC 8707 + RFC 8693 + signed audit; standard delegation only)
- **P5 (2w): 🤖 EvalAgent + 🤖 TrainOrchestrator** (workers #2 + #3 — first real GPU run)
- P5.5 (1w): PyTorch depth artifact
- **P6 (1.5w): 🤖 OptimizeAgent** (worker #4)
- **P7 (1w): CodeWork Skill** (Anthropic Skills pattern, NOT separate agent) + **CapabilityProbe in CI/CD lane** (NOT standing agent)
- **P8 (1.5w): GepaOptimizer offline batch tool** (NOT standing agent — Decagon pattern)
- **P9 (1.5w): MultimodalPipeline DEMO** (orchestrate existing 4 workers via PrincipalAgent on real Cosmos NIM endpoint)
- **P10 (2w): CrossAgentEvaluator offline + production deploy + nat YAML + OSS PR + demo**

**~18 weeks of agent + governance work after P0.5 completion.**

**v7 final agent count**: 5 production agents (1 PrincipalAgent supervisor + 4 specialty workers) + 1+ Skills (CodeWork) + 3 OFFLINE governance tools (NOT standing agents per frontier convergence). Total: 5 production + Skills + offline tools + ~16 infrastructure on LangGraph + Magentic-One + ml-intern primitives.

---

## Branch state

`p0_5_library_restructure` — 12 commits, all green, ready for PR.

After P0.5 completion commit (this one):
- Open PR? (recommended for review of foundation before P1 starts)
- Or continue to P1 D1 immediately?
