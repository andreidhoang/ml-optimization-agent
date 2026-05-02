# Current Phase — P0.5 D2: ml_intern adapter

> ⚡ **LIVE FILE** — updated as we move through phases. If this is stale, fix it before doing more work.

**Today's date**: 2026-05-03 (D1 shipped same-day)
**Active phase**: P0.5 (Library restructure + harness adapters, 4 days)
**Active day**: **D2 of 4** — write `cosmos_lab/harness/ml_intern.py` adapter

---

## D1 — DONE (recap)

✅ Created `cosmos_lab/` package with `__init__.py` + `identity/__init__.py` re-export shims  
✅ Updated `pyproject.toml` — added `cosmos_lab*` to packages.find + `[nat]`/`[ml_intern]`/`[claude_sdk]` extras  
✅ Verifier `./bin/verify.sh p0_5_d1` → 14/14 pass  
✅ Upstream baseline preserved: 237 pass / 3 known-broken  
✅ Both import paths work: `from cosmos_lab.identity import AgentIdentity` AND `from agent.optimization.identity import AgentIdentity`  
✅ Zero-diff: only `pyproject.toml` + `uv.lock` modified (config we own); no upstream code touched

**LEARN from D1**: One verifier surprise — `uv.lock` regenerates on `uv sync` after `pyproject.toml` change. Verifier exclusion list updated to allow `pyproject.toml` and `uv.lock`. Captured in `bin/verify_p0_5_d1.sh` comment block. Pattern for future: any file we *intentionally* modify must be in verifier exclusion list, OR added with rationale.

Branch: `p0_5_library_restructure` (not yet committed — waiting for user OK)

---

## D2 — DEFINE (Phase 1 of workflow)

---

## D2 spec — Phase 1 of workflow (DEFINE)

### Goal (one sentence)
Ship `cosmos_lab/harness/ml_intern.py` (≤200 LOC) — an adapter that installs cosmos-lab governance (identity + audit + capability-scoped router) into an existing `agent.core.session.Session` instance, with a smoke test proving capability denial + audit emission works inside a real ml-intern session.

### Spec — what it does
- New module `cosmos_lab/harness/__init__.py` + `cosmos_lab/harness/ml_intern.py`
- One public function: `install_into_session(session, identity, audit) -> None` that wraps `session.tool_router` with `CapabilityScopedRouter`
- One smoke test in `tests/optimization/harness/test_ml_intern_adapter.py` verifying: (a) capability denial blocks unauthorized tool, (b) audit log records denial, (c) authorized tool call passes through to base router

### Spec — what it does NOT do (today)
- Does NOT touch `agent.core.session` (Invariant 1 — composition only)
- Does NOT write the `nat` adapter (D3)
- Does NOT parametrize all 16 Phase 0 tests across both adapters (D4)
- Does NOT add OTel span emission yet (P1)

### Verifier
`./bin/verify.sh p0_5_d2` — checks: module exists, smoke test passes, ≤200 LOC, zero-diff invariant.

---

## D1 spec (archived, for reference)

### Goal (one sentence)
Make `from cosmos_lab.identity import AgentIdentity` work, while preserving `from agent.optimization.identity import AgentIdentity` and all 16 Phase 0 tests passing unchanged.

### Spec — what it does
- Create `cosmos_lab/` package directory at repo root with re-export modules (`__init__.py`, `identity/__init__.py`, `trajectory/__init__.py`, `eval/__init__.py`, `harness/__init__.py`)
- Each `cosmos_lab/X/__init__.py` re-exports from `agent.optimization.X` (no code moves yet — only re-export shims)
- Update `pyproject.toml` to add `cosmos-lab` package + `[project.optional-dependencies]` block: `nat`, `ml-intern`, `claude-sdk`, `all`
- Both import paths work; both pass the same 16 tests

### Spec — what it does NOT do (today)
- Does NOT move actual code yet (D2-D3 work)
- Does NOT write nat or ml-intern adapters yet (D2-D3 work)
- Does NOT change Phase 0 test semantics — only adds parametrization
- Does NOT touch any upstream `agent/core/` or `agent/config.py` file

### Verifier (script, not prose)
`./bin/verify.sh p0_5_d1` — must exit 0. Checks:
1. `cosmos_lab/__init__.py` exists
2. `python -c "from cosmos_lab.identity import AgentIdentity, AuditLog, CapabilityScopedRouter, CapabilityDenied"` succeeds
3. `python -c "from agent.optimization.identity import AgentIdentity, AuditLog, CapabilityScopedRouter, CapabilityDenied"` still succeeds (backward compat)
4. `pyproject.toml` contains `cosmos-lab` and `[project.optional-dependencies]`
5. `uv run pytest tests/optimization/ -q` exits 0

### Invariants this phase must respect
- Invariant 1 (zero-diff): no upstream file edited
- Invariant 2 (no commit): wait for user
- Invariant 3 (baseline): `pytest tests/unit/ -q` ≤ 3 failures (the 3 known upstream-broken)
- Invariant 4 (owned paths): `cosmos_lab/` is new owned path; add to CLAUDE.md ✓ (already done)

### What I will NOT build today (write it down)
- nat adapter (D2)
- ml-intern adapter (D3)
- dual-adapter test matrix (D4)
- any Phase 1 trajectory code

---

## Phase 2 of workflow — PROBE (skip for D1)

D1 is mechanical packaging — no LLM-on-hard-task probing needed. Resume PROBE in P1 D1 when we start sentinel design.

---

## Phase 3 of workflow — BUILD (the loop)

### Plan (before code)
1. `mkdir cosmos_lab/{identity,trajectory,eval,governance,memory,providers,compute,sandbox,harness}`
2. Write top-level `cosmos_lab/__init__.py` with re-exports
3. Write each subpackage `__init__.py` re-exporting from `agent.optimization.X`
4. Update `pyproject.toml` — add `cosmos-lab` package + extras
5. Run `uv sync` to verify packaging picks it up
6. Run `./bin/verify.sh p0_5_d1`
7. Run dual-import smoke test
8. Run `uv run pytest tests/optimization/ -q`

### Build rule
3 concurrent agents max. D1 is single-agent (mechanical work). Use Edit/Write tools directly.

---

## Phase 4 of workflow — REVIEW (human gate)

Before declaring D1 done:
- [ ] Test file FIRST: do tests still check spec, or were they accidentally moved?
- [ ] Diff in reverse: last change to first
- [ ] For every new `__init__.py`: 3am failure mode? (silent re-import cycle = reject; ImportError = acceptable)
- [ ] Substrate check: identity/audit/router code unchanged in semantics? Just relocated import paths?
- [ ] Run `git diff upstream/main --name-only` — owned paths only

---

## Phase 5 of workflow — SHIP (n/a for D1)

D1 is internal restructure — no deploy. Resume SHIP in P10.

**KILL conditions for the day**: if `pytest tests/unit/` shows >3 failures (we broke upstream baseline) → STOP, revert, root-cause.

---

## Phase 6 of workflow — LEARN

After D1:
- What surprised you? (write 1 paragraph)
- New verifier needed? Add to `./bin/verify.sh`
- Update `docs/02_current_phase.md` to D2 (write nat adapter)

---

## After D1 ships, next:
- **D2**: write `cosmos_lab/harness/ml_intern.py` (≤200 LOC adapter — the one we already implicitly use)
- **D3**: write `cosmos_lab/harness/nat.py` (≤200 LOC adapter — primary)
- **D4**: dual-adapter test matrix; both adapters pass same 16 Phase 0 tests; CONTRACT.md

Full P0.5 spec → `PLAN_V2.md` §2.5 (read only when needed).
