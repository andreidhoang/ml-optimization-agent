# cosmos-lab — Context Harness

> **What this is**: zero-diff fork of `huggingface/ml-intern` that ships the **cosmos-lab** library — **one exceptional autonomous PrincipalAgent** doing long-horizon ML lifecycle work, with 6 demonstrated capability domains (data, eval, train, optimize, multimodal, code), built on ml-intern's `agent_loop.py` substrate. Governance (sentinels, MCP-OAuth identity, GEPA self-improvement) exists as **enabler of autonomy**, not constraint. Runs natively on `nvidia-nat`. See `docs/01_north_star.md` for vision; PLAN_V2.md §0.9 for the autonomous-agent thesis; PLAN_V2.md §3.2 for PrincipalAgent architecture.

**Current phase** → `docs/02_current_phase.md` (LIVE — read this first when starting work)

---

## Invariants — never break

1. **ZERO-DIFF**: never edit any file existing in `huggingface/ml-intern`. Use subclass / composition / new path.
2. Never `git commit` or `git push` without explicit user request.
3. `uv run pytest tests/unit/` must match baseline (237 pass / 3 upstream-broken — see PLAN_V2 invariant 2).
4. New code only in owned paths (table below).
5. **One-optimization-per-experiment** + **measured-peak over vendor-peak** (EVAL_SPEC.md, applies P6+).
6. **Trajectory-on-by-default**: from P1, no agent run is unobserved.
7. **OTel-GenAI-on-by-default**: from P1, every span uses `gen_ai.*` semconv.
8. **No judge-only metric reaches a gate**: every quality gate requires (judge, structural-verifier) pair.
9. **No GPU phase exits without one measured real run** (P5/P5.5/P6/P9a; ~$200-400 budget).

---

## Owned paths

| Write here | Never touch (upstream) |
|---|---|
| `cosmos_lab/` (P0.5+) | `agent/core/` |
| `agent/optimization/` (Phase 0 home) | `agent/config.py` |
| `agent/tools/{profiling,training_opt,inference_opt,multimodal_opt,vla_opt}/` | `agent/context_manager/manager.py` |
| `agent/tools/hardware_specs.py` | `agent/tools/*.py` (existing) |
| `agent/prompts/system_prompt_optimization_*.yaml` | `backend/`, `frontend/`, `tests/unit/` |
| `configs/optimization_agent_config.json` | |
| `tests/optimization/` | |
| `docs/`, `bin/` (this harness) | |

**Verify ownership**: `git diff upstream/main --name-only` must show only owned paths.

---

## Anti-patterns (catch yourself)

1. Editing the prompt when the bug is in the data.
2. Editing the data when the bug is in the spec.
3. Trusting "I have verified this" from an agent — re-run the verifier yourself.
4. Building a pipeline that should have been one model call.
5. Adding a fourth concurrent agent. You will regret it.
6. Saying "the agent decided" — replace with "P(output | context) was high."

---

## Workflow (every task)

`DEFINE → PROBE → BUILD → REVIEW → SHIP → LEARN` — see `docs/00_workflow.md`.

**Hard rule**: if you can't write the verifier, the goal is wrong. Fix the goal, not the agent.

---

## Dev commands

```bash
uv sync                                                    # install
uv run pytest tests/unit/ -q                               # upstream baseline (must match)
uv run pytest tests/optimization/ -q                       # cosmos-lab tests (must pass)
PYTHONPATH=. ruff check agent/ --ignore E501,F401,E402     # lint
./bin/verify.sh <phase>                                    # phase verifier (e.g. p0_5)
git fetch upstream && git merge upstream/main              # daily upstream sync
```

---

## Pointer index (load on demand)

| Need | Read |
|---|---|
| What we're building this week | `docs/02_current_phase.md` |
| Vision in 1 screen | `docs/01_north_star.md` |
| Workflow phases | `docs/00_workflow.md` |
| Phase → PLAN_V2 anchor map | `docs/03_pointers.md` |
| Full plan (24 sections, 837L) | `PLAN_V2.md` (read specific section, not whole file) |
| Architecture deep WHY (Vietnamese, 1167L) | `SYSTEM.md` (rare — only for upstream debugging) |
| Eval spec — ML output (perplexity, KL, latency p99) | `EVAL_SPEC.md` |
| Eval spec — agent system (trajectory, plan, replan, capability boundary, reward-hack, cross-agent) | `AGENTIC_EVAL_SPEC.md` |
| Self-improvement research | `RESEARCH_AHE_ANALYSIS.md` |
| Dev server / deploy notes | `AGENTS.md` |
| NVIDIA Cosmos JD | `docs/04_jd.md` |

---

## Git remotes

```
upstream → https://github.com/huggingface/ml-intern  (read-only, never push)
origin   → https://github.com/andreidhoang/ml-optimization-agent
```
