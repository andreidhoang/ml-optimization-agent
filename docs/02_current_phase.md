# Current Phase — P0.5 D3: nat adapter

> ⚡ **LIVE FILE** — updated as we move through phases. If this is stale, fix it before doing more work.

**Today's date**: 2026-05-03 (D1 + D2 shipped same-day)
**Active phase**: P0.5 (Library restructure + harness adapters, 4 days)
**Active day**: **D3 of 4** — write `cosmos_lab/harness/nat.py` adapter (primary harness for Cosmos pitch)

---

## D1 + D2 — DONE (recap)

### D1 (cosmos_lab restructure) ✅
- `cosmos_lab/__init__.py` + `cosmos_lab/identity/__init__.py` re-export shims
- `pyproject.toml` updated: `cosmos_lab*` in packages.find + `[nat]`/`[ml_intern]`/`[claude_sdk]` extras
- Verifier `./bin/verify.sh p0_5_d1` → 14/14 pass

### D2 (ml_intern adapter) ✅
- `cosmos_lab/harness/__init__.py` + `cosmos_lab/harness/ml_intern.py` (84 LOC adapter)
- `tests/optimization/harness/test_ml_intern_adapter.py` — 6 smoke tests (3 contract + 3 e2e behavior)
- Public API: `from cosmos_lab.harness import install_into_session`
- Verifier `./bin/verify.sh p0_5_d2` → 11/11 pass

**LEARN from D2** — three surprises captured:

1. **Editable install staleness**: adding `cosmos_lab/harness/` after the previous `uv sync` left the editable install metadata stale; new submodule was undiscoverable until re-sync. Pattern: any new package directory needs `uv sync` (and `uv sync --extra dev` for test deps) before tests pass.

2. **`uv run pytest` is ambiguous**: PATH leak — `uv run pytest` resolved to `/opt/miniconda3/bin/pytest` (system Python with stale editable install) instead of venv. Symptom: `python -c "import cosmos_lab.harness"` succeeded everywhere but pytest collection failed with `ModuleNotFoundError: No module named 'cosmos_lab'`. **Fix**: always use `uv run python -m pytest` for deterministic venv resolution. All verifier scripts updated.

3. **Smoke test design**: writing a test that constructs a real `agent.core.session.Session` would require Config + ContextManager + event_queue + sandbox + more — fighting against composition philosophy. Used a duck-typed `MockSession` (just `.tool_router`) since adapter only touches that one attribute. **Pattern**: smoke test verifies the adapter contract, not the host's internals.

---

## D3 spec — Phase 1 of workflow (DEFINE)

### Goal (one sentence)
Ship `cosmos_lab/harness/nat.py` (≤200 LOC) — an adapter that registers cosmos-lab governance (CapabilityScopedRouter wrapping nat's tool registry) inside a `nvidia-nat` Builder, with a smoke test proving capability denial works inside a `nat run` of a trivial workflow YAML.

### Spec — what it does
- New module `cosmos_lab/harness/nat.py`
- One public function: `install_into_nat(builder, identity, audit) -> None` that wraps the nat builder's tool router
- Optional: register `OTelGenAIEmitter` placeholder (real emitter lands in P1)
- Smoke test in `tests/optimization/harness/test_nat_adapter.py` verifying contract:
  - `install_into_nat()` registers wrapped router as nat plugin
  - Tool denial path works when called via nat workflow
  - Tool authorized path passes through

### Spec — what it does NOT do (today)
- Does NOT modify `nvidia-nat` package (Invariant 1 — composition)
- Does NOT parametrize Phase 0 tests across both adapters yet (D4)
- Does NOT add full OTel emitter (P1)
- Does NOT validate against real `nat run` (mock the Builder protocol — pattern from D2 with MockSession)

### Verifier
`./bin/verify.sh p0_5_d3` — checks: module exists, ≤200 LOC, smoke test passes, dual-adapter (`ml_intern` + `nat`) parity check.

### Open question for D3 D-day
**`nvidia-nat` API surface** — we haven't pinned the exact Builder API yet. If `pip install nvidia-nat` fails on local Python 3.12, fall back to a Protocol-typed `BuilderLike` and treat the smoke test as contract-only (real nat integration deferred to D-day with hands-on verification). This is honest scoping, not retreat.

---

## D2 spec (archived, for reference)

### Goal (one sentence)
Ship `cosmos_lab/harness/ml_intern.py` (≤200 LOC) — an adapter that installs cosmos-lab governance (identity + audit + capability-scoped router) into an existing `agent.core.session.Session` instance, with a smoke test proving capability denial + audit emission works inside a real ml-intern session.

(See git log for D1 spec.)

---

## After D3, next:
- **D4**: dual-adapter test matrix — every Phase 0 test parameterized via `@pytest.mark.parametrize("harness", ["nat", "ml_intern"])`; both must pass identically. Document contract in `cosmos_lab/harness/CONTRACT.md`.

Full P0.5 spec → `PLAN_V2.md` §2.5 (read only when needed).
