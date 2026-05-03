# Current Phase — P0.5 D4: dual-adapter test matrix + CONTRACT.md

> ⚡ **LIVE FILE** — updated as we move through phases. If this is stale, fix it before doing more work.

**Today's date**: 2026-05-03 (D1 + D2 + D3 + v5.1 architecture decision shipped same-day)
**Active phase**: P0.5 (Library restructure + harness adapters, 4 days)
**Active day**: **D4 of 4 — final** — dual-adapter test matrix + CONTRACT.md

---

## D1 + D2 + D3 — DONE (recap)

### D1 (cosmos_lab restructure) ✅
- `cosmos_lab/__init__.py` + `cosmos_lab/identity/__init__.py` re-export shims
- `pyproject.toml` updated: `cosmos_lab*` in packages.find + `[nat]`/`[ml_intern]`/`[claude_sdk]` extras
- Verifier: 14/14 pass

### D2 (ml_intern adapter — execution substrate) ✅
- `cosmos_lab/harness/ml_intern.py` — `install_into_session()` wraps Session.tool_router
- 6 smoke tests (3 contract + 3 e2e behavior)
- Per v5.1: this is the **execution substrate adapter** (PrincipalAgent constructs Sessions, installs governance, runs tasks)
- Verifier: 11/11 pass

### D3 (nat wrapper — deployment surface) ✅
- `cosmos_lab/harness/nat.py` (132 LOC, ~78 non-comment) — `register_as_nat_tool()` registers `cosmos_lab_principal` as nat tool
- 11 smoke tests covering registration mechanics + tool callable contract
- Per v5.1: nat is **deployment wrapper, not runtime substrate** — invokes cosmos-lab CLI from a nat workflow
- Tool body is v0 stub; real CLI invocation lands in P3 (PrincipalAgent v0)
- Verifier: 10/10 pass

### v5.1 architectural decision (PLAN_V2 §0.4.5) ✅
After auditing `agent_loop.py:1771` (queue-based `submission_loop`), committed to **2-layer architecture**:
- **Layer 1**: cosmos-lab CLI (primary entry point) — PrincipalAgent + governance + sentinels + memory + sub-agent spawning
- **Layer 2**: ml-intern Session as execution substrate (per task) — debugged ReAct, 16 tools, MCP, sandbox
- **Deployment wrappers** (P10): nat workflow YAML, Modal/HF Spaces endpoint
- Avoided: 1-2 weeks of async-bridge engineering for 3-layer runtime
- Banked: schedule + complexity budget for sentinels/memory/capability domains

**LEARN from D3 + v5.1 decision**:
1. **Always read substrate code before architecting on it** — the queue-based submission_loop made 3-layer runtime non-trivial; should have read agent_loop.py earlier
2. **`uv sync` without `--extra dev` removes pytest** — must use `uv sync --extra dev` to keep test deps. CLAUDE.md updated.
3. **Two layers > three when one is sufficient** — workflow anti-pattern #4 generalized: don't add a layer that doesn't earn its complexity

---

## D4 spec — Phase 1 of workflow (DEFINE)

### Goal (one sentence)
Ship `cosmos_lab/harness/CONTRACT.md` documenting the adapter contract that ALL harness adapters (current: ml_intern + nat; future: claude_sdk) must satisfy, plus a parametrized test matrix that runs the same contract tests against both shipped adapters.

### Spec — what it does
- New file `cosmos_lab/harness/CONTRACT.md` documenting:
  - The 3-method adapter contract (registration, execution interface, lifecycle)
  - What each adapter is responsible for vs what cosmos-lab core handles
  - Per-adapter exceptions (e.g., ml_intern needs Session, nat needs Builder)
- New test file `tests/optimization/harness/test_adapter_contract.py` parametrizing:
  - `@pytest.mark.parametrize("adapter", ["ml_intern", "nat"])`
  - Each contract assertion runs against both adapters
  - Both must pass identically for any contract assertion that's adapter-shape-agnostic

### Spec — what it does NOT do (today)
- Does NOT add a third adapter (claude_sdk is v1.1)
- Does NOT change ml_intern or nat adapter implementations
- Does NOT enforce contract at runtime via abstract base class — uses test-suite enforcement (more flexible for ducked Protocol patterns)

### Verifier
`./bin/verify.sh p0_5_d4` — checks: CONTRACT.md exists, test_adapter_contract.py exists, parametrized matrix passes for both adapters, P0.5 closes (D1+D2+D3+D4 all green).

### After D4 → P0.5 COMPLETE
P0.5 ships in 4 days as planned. P1 starts: eval infrastructure (sentinels + OTel + Inspect AI + MultiJudge) — foundation for EvalAgent (P4a) and supporting all other specialty agents.

### v6 schedule reminder (~17 more weeks after P0.5)

Per PLAN_V2 §1 v6 phase table:
- P1 (2w): Eval infrastructure (sentinels, OTel, Inspect AI, MultiJudge — foundation for EvalAgent + used by all)
- P2 (1w): Cosmos toolset (NIMProvider + tool wrappers — used by all specialty agents)
- **P3 (1.5w): 🤖 DataAgent** (Cosmos-specialty agent #1)
- **P4a (1w): 🤖 EvalAgent** (Cosmos-specialty agent #2)
- P4b (2w): Identity v2 (MCP OAuth + RFC 8693 + signed audit — substrate for capability expansion)
- **P5 (1.5w): 🤖 TrainOrchestrator** (Cosmos-specialty agent #3 — first real GPU run)
- P5.5 (1w): PyTorch depth artifact
- **P6 (1.5w): 🤖 OptimizeAgent** (Cosmos-specialty agent #4)
- **P7 (1w): 🤖 CapabilityProbe** (governance agent #1) + 3-tier memory
- **P8 (2w): 🤖 GepaOptimizer** (governance agent #2)
- **P9 (2w): 🤖 MultimodalPipelineAgent + CodeAgent** (Cosmos-specialty agents #5 + #6)
- **P10 (2w): 🤖 CrossAgentEvaluator** (governance agent #3) + production deploy + nat YAML + OSS PR + demo

Total: 9 NEW agents (6 specialty + 3 governance) + ~16 infrastructure components. Built on ml-intern's tool primitives leveraged AS-IS.

---

## D3 spec (archived, for reference)

### Goal (one sentence)
Ship `cosmos_lab/harness/nat.py` (~50 LOC) — a lightweight nat-tool registration shim that lets a nat workflow invoke `cosmos-lab principal --task <spec>` as a callable tool, with smoke test verifying registration contract works against mocked nat Builder API.

(See git log for D1, D2 specs.)

---

## Branch state

`p0_5_library_restructure` — 9 commits planned (currently 8, D3 + v5.1 ready to commit as #9).

After D4 → 10 commits, P0.5 complete, ready for PR.
