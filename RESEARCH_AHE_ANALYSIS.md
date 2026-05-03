# AHE Paper Analysis — Senior Engineering Review

> **Paper**: *Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses*
> Lin, Liu, Pan, Lin, Dou, Huang, Yan, Han, Gui — Fudan / Peking / Shanghai Qiji Zhifeng
> arXiv:2604.25850v2, April 29, 2026
> **Reviewed**: 2026-04-30
> **Reviewer perspective**: Senior agentic-harness engineer, frontier-lab lens
> **Verification status**: All numeric claims cross-checked against the paper extract. Project mappings cross-checked against `CLAUDE.md`. Items I have *not* personally verified in source code are explicitly flagged with `[unverified]`.

---

## 0. TL;DR

AHE is the most directly relevant paper to our harness architecture published to date. Its three-pillar framing (component / experience / decision observability) **validates the architectural instinct already baked into our `CLAUDE.md`** (zero-diff fork, owned-paths split, minimal-seed phase 0). It also exposes three things we *do not yet have* that the paper provides empirical evidence are load-bearing: structured trajectory analysis, falsifiable change manifests, and layered evidence distillation.

**Verdict (revised 2026-05-01)**: implement the full AHE stack for our ML optimization domain — 3 agents, 7-slot substrate, manifest contracts, Algorithm 1 loop, cross-model evaluation — sequenced by dependency across 9 stages (A-I, see Section 6). Stage A (slot decomposition) and Stage D (manual manifest discipline) start now. Stages E-I (meta-stack: Debugger, Verifier, Evolve Agent, orchestration, transfer testing) require Stage C (≥50-task scored suite) as a hard precondition.

**Confidence in this assessment**: Medium-high on architecture, lower on schedule. The paper's empirical case is real but its hard-task tier loss, regression-attribution weakness (precision 11.8%), and sub-additive component interactions mean every stage needs measured acceptance criteria — not blind replication. Stage C (scored task suite) and Stage H compute budget are the rate-limiters, not the code itself.

---

## 1. The Paper in One Page

### 1.1 Thesis

The bottleneck for evolving coding-agent harnesses is **observability, not model capability**. Given a decoupled action space, structured trajectory evidence, and falsifiable change manifests, an evolve-agent self-improves a harness without collapsing into trial-and-error.

### 1.2 Three Observability Pillars

| Pillar | Mechanism | What it buys |
|---|---|---|
| **Component** (NexAU) | Harness exposed as 7 orthogonal *files*: system prompt, tool description, tool implementation, middleware, skill, sub-agent config, long-term memory | Each failure pattern maps to one component class — clean attribution, no entanglement |
| **Experience** (Agent Debugger) | Million-token traces converted into per-task analysis reports + benchmark overview, navigable as a file environment | Evolve-agent reasons over structured root causes, not raw logs |
| **Decision** (Change Manifest) | Every edit ships with predicted fixes and predicted regressions | Next round verifies the contract → falsifiable evolution |

### 1.3 The Loop (Algorithm 1)

```
Rollout → Clean → Attribute / Rollback → Distill → Evolve → Commit
```

Governed by two hard constraints:

- **Controllability**: only the harness workspace is writable; infrastructure is read-only.
- **Falsifiability**: manifest predictions are checked against actual task-level deltas next round.

### 1.4 Headline Numbers (Terminal-Bench 2, 89 tasks)

| Method | All | Easy (4) | Medium (55) | Hard (30) |
|---|---|---|---|---|
| NexAU₀ (seed) | 69.7% | 87.5% | 78.2% | 51.7% |
| ACE (prompt-only self-evolve) | 68.9% | 91.7% | 78.2% | 48.9% |
| TF-GRPO (RL-style) | 72.3% | 100.0% | 79.4% | **55.6%** |
| **AHE** | **77.0%** | 100.0% | **88.2%** | 53.3% |

Also beats human-designed harnesses: Codex 71.9%, terminus-2 62.9%.

### 1.5 Operating Mode — What the Paper Validates vs. What We Choose

> *Verified by direct query against the paper, 2026-05-01. Earlier drafts of this doc and conversational answers asserted things the paper does not actually claim. This section corrects the record.*

The paper describes and empirically validates exactly **one** operating mode:

- **Offline batch evolution against fixed benchmarks.** Algorithm 1 runs sequentially through `N=10` iterations. Each iteration runs all 89 Terminal-Bench 2 tasks, analyzes traces, proposes edits, commits. **~32 hours total wall time** for the full 10-iteration run on one benchmark. All three role agents (Code Agent, Agent Debugger, Evolve Agent) share one base model (GPT-5.4 high reasoning), differing by prompt + tools + role.

The paper does **NOT** describe or validate any of the following — these are gaps to be aware of, not facts to extract:

- A **deployment / production scenario** for the evolved harness. The paper presents the evolved harness as a research artifact that *transfers* across benchmarks and models. It does not address how end users would receive, run, or interact with it.
- **Online evolution during user sessions.** No experiment evolves the harness while a user (rather than a benchmark) is the trace source.
- **Continuous / test-time evolution at inference.** The paper *motivates* "test-time learning" as a direction in its introduction, but does not implement or evaluate it. Treat as future work.
- **Adaptive termination.** `N=10` is fixed. No plateau-detection. No early-stop rule.
- **Cost amortization across user sessions.** The 32-hour figure is treated as a one-time research expense; the paper offers no model for spreading it across deployment usage.

#### Operating-mode options for our project

| Option | Description | Paper status |
|---|---|---|
| **A — Offline-only (paper-faithful)** | AHE runs as a separate batch process, by us, on Phase 8 benchmark. User `ml-intern` sessions invoke only the Code Agent with whatever harness is currently committed in the repo. | **Empirically validated by paper.** |
| **B — Offline evolution + user-trace logging** | Mode A, plus user session traces accumulate as additional evolution data for future batch runs. | Suggested by paper's "test-time learning" framing but **not implemented, not evaluated**. |
| **C — Online evolution during user sessions** | All 3 agents run during user sessions; harness evolves continuously per-user. | **Not described in paper.** Speculation. |

#### Our default: Option A

It is the only mode with empirical evidence. Phases 10 and 11 in `PLAN.md` are written for this mode. Moving to B or C is a deliberate scope expansion past the paper's evidence base, and should require its own validation work — not blind adoption.

#### Practical implication for users

When a user runs `uv run ml-intern`:

- **Only the Code Agent runs.** Same as today.
- **Debugger and Evolve Agent do not run.** They are offline tools used by the engineering team to update the harness between releases.
- **No manifests are written, no traces are analyzed, no rollbacks happen during user sessions.**
- The user simply benefits from a harness that has been evolved against the Phase 8 benchmark in a prior offline run committed to the repo.

This matches the paper's empirical setup. Deviating from it (Option B or C) would put us off the paper's evidence and require us to validate the new mode independently.

---

## 2. Detailed Mechanisms — What Actually Does the Work

### 2.1 NexAU: Decoupled Harness Substrate

Seven orthogonal component types as *explicit files*:

1. System prompt
2. Tool description (the schema/docstring the model sees)
3. Tool implementation (the executable code)
4. Middleware (cross-cutting concerns: retries, parsing, side-effects)
5. Skill (reusable procedural knowledge)
6. Sub-agent configuration (delegation patterns)
7. Long-term memory

**Why the split matters**: Tool-description and tool-implementation are *two separate slots*. A failure mode where "the model misuses the tool because the docstring is wrong" must be debuggable without touching the tool's actual code path — and vice versa. This is a genuine engineering insight: most harnesses I have seen conflate these.

**Minimal seed (H₀)**: bash-only, no middleware, no sub-agents. Deliberately spartan. Forces every added component to *justify itself through measured rollouts* and prevents hidden-prior leakage where a fat baseline silently drives gains. This is the same instinct as `git bisect` against a clean ancestor.

### 2.2 Agent Debugger: Layered Trajectory Evidence

The non-obvious move: traces are treated as a **navigable file environment**, not a flat log. The debugger agent has tools to drill in (per-task) and zoom out (benchmark overview), and it produces *structured reports* — quoted directly: "structured root causes rather than raw logs." This is what allows the evolve-agent to consume analysis without context explosion.

This is the part of the paper most under-appreciated by casual readers. A million-token trace cannot be fed to an LLM directly; the question is how you compress it without destroying causal information. Their answer: progressive disclosure with task-level and benchmark-level views.

### 2.3 Evolve Agent: Change Manifests as Falsifiable Contracts

Each edit is committed with a manifest containing:

- The failure evidence that motivated it
- The diagnosed root cause
- The targeted fix
- The expected impact (which tasks should flip, which might regress)

Next round verifies. This is the discipline that separates AHE from "let an LLM rewrite the prompt and pray." The manifest *is the contract*; if predictions don't hold, you have evidence the diagnosis was wrong, not just the fix.

---

## 3. Honest Reading of the Results

The headline number is real, but the supporting evidence has cracks worth naming.

### 3.1 Strengths (Real Signal)

- **+7.3pp over a competitive seed** with a clear, attributable methodology. Not a noise-level win.
- **Beats prompt-only self-evolution (ACE) and an RL-style baseline (TF-GRPO)** on aggregate. The methodology, not the budget, is doing the work.
- **Cross-model transfer is the most impressive result**: +10.1pp on deepseek-v4-flash, +6.3pp on qwen-3.6-plus, +5.1pp on gemini-3.1-flash-lite. The author's read — *"less capable models lean more heavily on coordination patterns AHE has fixed"* — is plausible and is genuine evidence against benchmark overfitting.
- **Cross-benchmark transfer to SWE-bench-verified**: 75.6% with 32% fewer tokens than ACE. The harness encodes general engineering experience, not Terminal-Bench-specific tricks.

### 3.2 Weaknesses (Caveats Not to Smooth Over)

- **Hard tier: AHE 53.3% vs TF-GRPO 55.6%**. AHE *loses* on the hardest 30 tasks. Gains concentrate in medium difficulty. The hard-task ceiling is unclear and not addressed convincingly.
- **Component ablation is sub-additive and partly negative**:

  | Component (alone) | Δ vs seed |
  |---|---|
  | Memory | +5.6 pp |
  | Tools | +3.3 pp |
  | Middleware | +2.2 pp |
  | System prompt | **−2.3 pp** |

  System-prompt-only evolution makes things *worse*. The full stack is sub-additive — the paper acknowledges *"components interact non-additively, capping aggregate gain"*. The implication: evolving everything simultaneously is not a free lunch; interference is real.

- **Attribution is barely above random for regressions**:
  - Fix precision 33.7% / recall 51.4% (≈5× random)
  - Regression precision 11.8% / recall 11.1% (≈2× random)

  The paper calls this "regression blindness" and acknowledges it limits convergence predictability. This means the "falsifiable contract" works for *fixes* but is shaky for *regressions* — which is the more dangerous failure mode.

- **Step budgets fitted to GPT-5.4-high**. Cross-model results are sensitive to timeout conventions. Not necessarily wrong, but a confound to track.

- **Engineering overhead is not quantified**. Trajectory analysis + workspace management has compute cost that the paper does not put a number on.

### 3.3 Net Reading

The methodology is real and the architectural insights generalize. The empirical case is strong on average and on transfer, weak on hard-task tier and on regression attribution. The paper itself frames AHE as *"a controlled research prototype rather than a fully mature autonomous system"* — that framing is honest and should anchor our adoption decisions.

---

## 4. Architectural Principles (What to Extract)

Independent of whether we ever implement the AHE *loop*, four principles from the paper stand on their own as architecture guidance.

### 4.1 Decouple the Substrate Aggressively

If a failure pattern can be caused by either of two components, those components must live in separate files. The 7-slot decomposition is an existence proof of how granular this can usefully go.

### 4.2 Start From a Minimal Seed

A fat baseline hides where gains come from. Anything you add must *earn* its place against a measured floor. This is also the cheapest insurance against premature abstraction (which our project's `RULES.md` already enforces).

### 4.3 Treat Edits as Falsifiable Contracts

An edit without a prediction is not a hypothesis, it is a hope. Manifests with predicted-fix and predicted-regression fields convert harness evolution from vibes-based to evidence-based — even before any automation.

### 4.4 Observability Beats Cleverness

The paper's central insight: *"once the evolution agent receives structured context over a clear action space, it reliably converges on better designs."* The implication for human engineers is the same. Invest in trace structure before investing in cleverer agents.

---

## 5. Mapping to Our Project

Our `CLAUDE.md` already encodes a *partial* version of these principles. The mapping is partial, not exact — I want to be precise about what we have and what we do not.

### 5.1 What We Already Have (Verified Against `CLAUDE.md`)

| Our invariant | NexAU equivalent | Status |
|---|---|---|
| Zero-diff: never edit upstream files | Controllability constraint (infrastructure read-only) | Present |
| Owned-paths table: `agent/optimization/`, `agent/tools/profiling/`, `prompts/`, `configs/` | Decoupled component substrate | Partial — split exists, but not by NexAU's 7 categories |
| Phase 0 = baseline verification before anything else | Minimal seed H₀ | Present |
| `pytest tests/unit/ -q` exit-0 gate | Rollout + verification | Present, but coarse-grained (binary, not per-task) |

### 5.2 What We Do Not Yet Have

| Missing capability | Why it matters | Cost to add |
|---|---|---|
| **Trajectory observability layer** (Agent Debugger analogue) | When optimization agent fails on a profiling/quantization task, the failure is raw logs, not structured root causes | Medium — needs a benchmark first |
| **Change manifests** | Edits to prompts/tools have no predicted-fix / predicted-regression fields. We cannot tell, after the fact, whether a change behaved as designed. | Low — schema + discipline |
| **Layered distillation** | As Phase 1+ knowledge tools grow, context-explosion will hit. Per-task → per-domain distillation is the answer | Medium — defer until pain is real |
| **Per-component ownership in the 7-slot sense** | Our current owned-paths split is by *concern* (profiling, training opt, inference opt), not by NexAU's *component type* (tool description vs tool impl vs middleware) | Low — refactor intent, not files |

### 5.3 Mapping Caveats

- I have **not personally verified** the internals of `agent/core/agent_loop.py`, `agent/core/session.py`, or `agent/context_manager/manager.py` `[unverified]`. The mapping above is based on `CLAUDE.md` declarations and standard layering assumptions. Before any refactor, those files should be read end-to-end to confirm where seams actually are.
- Our project is *building an ML optimization agent* — the agent optimizes ML workloads. AHE is about *optimizing the harness itself*. These are adjacent but not identical objectives. Some AHE machinery is overkill for our current phase.

---

## 6. Implementation Roadmap — Build Everything From AHE, Sequenced by Dependency

**Decision (recorded 2026-05-01)**: implement the full AHE stack (3 agents + 7-slot substrate + manifest contract + Algorithm 1 loop + cross-model evaluation) for the ML optimization domain. Sequence below respects technical dependencies — order is not preference, it is what the paper itself requires for the loop to converge.

### 6.0 Dependency Graph (read this first)

```
[A] NexAU 7-slot decomposition  ◄── Phase 0/1
       │
       ▼
[B] Code Agent = ML optimization agent  ◄── Phases 1-5 (existing PLAN.md)
       │
       ├─► [D] Change-manifest discipline (manual)  ◄── Phase 1+, in parallel
       │
       ▼
[C] Scored ML-opt task suite (≥50 tasks, deterministic)  ◄── Phase 8 (rate-limiter)
       │
       ├─► [E] Agent Debugger (structured failure reports)  ◄── Phase 9
       │
       ▼
[F] Manifest auto-verification (falsifiability loop)  ◄── Phase 9
       │
       ▼
[G] Evolve Agent (proposes edits, writes manifests)  ◄── Phase 10
       │
       ▼
[H] Full orchestration (Algorithm 1)  ◄── Phase 10
       │
       ▼
[I] Cross-model transfer evaluation  ◄── Phase 11
```

Skipping the order does not save time — it amplifies the paper's known failure modes (regression blindness, sub-additive interactions).

### 6.A — NexAU 7-slot decomposition

| Field | Value |
|---|---|
| **Phase** | 0 / 1 |
| **Path** | `agent/optimization/config_ext.py`, `agent/optimization/` directory layout |
| **Scope** | Declare all 7 slots in `OptimizationConfig` (system prompt, tool description, tool implementation, middleware, skill, sub-agent config, long-term memory). Empty slots reserved by name. |
| **Acceptance** | Every future PR can answer "which slot does this belong to." Lint check that every new file lives under one of the 7 slot directories. |
| **Dependencies** | None. Start here. |
| **Risk** | Low. Cost: hours. |

```python
# agent/optimization/config_ext.py
class OptimizationConfig(Config):
    # NexAU-aligned slots (some empty initially)
    system_prompt_path: str = "agent/prompts/system_prompt_optimization_v1.yaml"
    tool_descriptions_path: str | None = None
    tool_implementations_dir: str = "agent/tools/"
    middleware_dir: str | None = None
    skills_dir: str | None = None
    subagent_config_dir: str | None = None
    long_term_memory_path: str | None = None
```

### 6.B — Code Agent (the ML optimization agent itself)

| Field | Value |
|---|---|
| **Phase** | 1-5 (entire existing `PLAN.md`) |
| **Path** | `agent/optimization/`, `agent/tools/profiling/`, `training_opt/`, `inference_opt/`, `multimodal_opt/`, `vla_opt/` |
| **Scope** | Everything in current `PLAN.md` Phases 1-5: system prompt, knowledge tools, profiling suite, training/inference/multimodal/VLA optimizations. |
| **Acceptance** | Per existing `PLAN.md` Definition of Done. |
| **Dependencies** | A. |
| **Why this is Stage B not Stage Z** | AHE without a Code Agent has nothing to evolve. The Code Agent IS the harness that gets optimized in stages G-H. The existing roadmap is not a preliminary — it is the substrate. |

### 6.C — Scored ML-optimization task suite

| Field | Value |
|---|---|
| **Phase** | 5-6 (extension of current Phase 6) |
| **Path** | `tests/optimization/benchmarks/`, `tests/optimization/scoring/` |
| **Scope** | ≥50 tasks with deterministic scoring. Each task: input (model + script + hardware target), success criterion (e.g. "fits in 40GB", "MMLU drop <1%"), reproducible scoring. |
| **Examples** | "Fit Llama-3-8B QLoRA on A100 40GB", "Reduce Mixtral inference latency 30% on 4×H100", "Quantize Gemma-7B <8GB with <1% MMLU drop". |
| **Acceptance** | Same task scored twice = same result. 3 baseline runs of seed harness yield <2% pass-rate variance. |
| **Dependencies** | B. |
| **Risk** | **HIGHEST risk in the roadmap.** Hardware availability, determinism, and scoring oracles for ML tasks are genuinely hard. This stage is the rate-limiter for the entire meta-stack. |

### 6.D — Change-manifest discipline (manual, parallel with B)

| Field | Value |
|---|---|
| **Phase** | 1+, in parallel with B |
| **Path** | `manifests/<date>-<slug>.yaml` |
| **Scope** | Every non-trivial harness edit ships with a manifest. Schema: `edit`, `slot`, `evidence`, `root_cause`, `expected_fix[]`, `expected_regression_risk`, `verification_round`. |
| **Acceptance** | 100% of harness PRs after Stage A include a manifest. Pre-commit hook enforces schema. |
| **Dependencies** | A. |
| **Risk** | Near zero cost, near zero risk. Highest ROI item in the entire roadmap before Stage F is online. |
| **Why parallel** | This is discipline, not automation. It also builds the dataset of human-authored manifests that informs Stage G's evolve-agent prompt. |

```yaml
# manifests/2026-05-15-add-hardware-specs-tool.yaml
edit: "Add hardware-spec retrieval tool"
slot: tool_implementation
evidence: "Optimization tasks failing because agent guesses wrong VRAM bounds"
root_cause: "No tool exposes ground-truth hardware specs"
expected_fix: ["task-id-014", "task-id-022", "task-id-031"]
expected_regression_risk: "May increase token usage on simple tasks (~5%)"
verification_round: 3
```

### 6.E — Agent Debugger

| Field | Value |
|---|---|
| **Phase** | 9 (new — appended to `PLAN.md`) |
| **Path** | `agent/optimization/meta/debugger/` |
| **Scope** | Separate agent. Different prompt + tools, **same base model as Code Agent** (per AHE pattern). Reads `runs/` trace files (read-only). Outputs structured per-task reports + benchmark-level overview using progressive disclosure. |
| **Acceptance** | On a synthetic 20-task failure suite, Debugger correctly classifies root cause for ≥70% of cases. Output token count ≤5% of input trace token count. |
| **Dependencies** | C (needs scored task runs to analyze). |
| **Risk** | Medium. The compression target is the hard part — paper achieves it via "navigable file environment" framing, which we will replicate. |

### 6.F — Manifest auto-verification

| Field | Value |
|---|---|
| **Phase** | 9 |
| **Path** | `agent/optimization/meta/verifier.py` |
| **Scope** | Code module (NOT an agent). Compares manifest predictions to actual task-level deltas. Computes fix-precision, fix-recall, regression-precision, regression-recall. |
| **Acceptance** | Replicate paper's metrics on our own data. Baseline computed on ≥10 manual manifests from Stage D. Watch for "regression blindness" — paper's regression precision was 11.8%; our number tells us how reliable Stage G's contracts can be. |
| **Dependencies** | C, D. |
| **Risk** | Low (pure code). |

### 6.G — Evolve Agent

| Field | Value |
|---|---|
| **Phase** | 10 (new — appended to `PLAN.md`) |
| **Path** | `agent/optimization/meta/evolver/` |
| **Scope** | Separate agent. Reads Debugger reports, proposes harness edits, writes manifests. Hard constraints: **controllability** (writes only to `workspace/`, infrastructure read-only) and **falsifiability** (every edit must include a manifest with predictions). |
| **Acceptance** | On a held-out task subset, Evolve Agent's proposed edits improve pass-rate over 3 rounds. Rollback rate <30%. System prompt slot edits gated through held-out validation (per paper's −2.3pp warning). |
| **Dependencies** | E, F. |
| **Risk** | High. Sub-additive component interactions are real — stage component additions, measure between each. |

### 6.H — Full orchestration (Algorithm 1)

| Field | Value |
|---|---|
| **Phase** | 10 |
| **Path** | `agent/optimization/meta/loop.py` |
| **Scope** | Pure Python orchestrator. Implements `Rollout → Clean → Attribute/Rollback → Distill → Evolve → Commit`. Calls Code Agent / Debugger / Evolve Agent at the right phases. |
| **Acceptance** | Full loop runs end-to-end on the 50-task suite. Improvement over seed harness ≥+5pp over 5 rounds (calibrated against paper's +7.3pp; lower bar reflects ML-opt domain narrowness). |
| **Dependencies** | B, C, E, F, G. |
| **Risk** | Compute budget. **Each round ≈ N_tasks × (rollout cost + debugger cost + evolve cost).** Paper does not quantify; we should expect ≥10K LLM calls per round on a 50-task suite. Budget compute before launching this stage. |

### 6.I — Cross-model transfer evaluation

| Field | Value |
|---|---|
| **Phase** | 11 (new — appended to `PLAN.md`) |
| **Path** | `tests/optimization/transfer/` |
| **Scope** | Take auto-evolved harness from Stage H, run with alternate base models. Measure pass-rate transfer. |
| **Acceptance** | ≥3 alternate models tested. Transfer gain ≥+3pp (paper showed +5-10pp, we lower the bar because ML-opt domain is narrower than terminal-bench). |
| **Dependencies** | H. |
| **Risk** | Low — read-only evaluation. |

### 6.X — Phase mapping summary

| AHE Stage | `PLAN.md` Phase |
|---|---|
| A — 7-slot decomposition | Phase 0/1 (existing) |
| B — Code Agent | Phases 1-5 (existing) |
| D — Manifest discipline | Phase 1+ (cross-cutting, parallel) |
| C — Scored task suite | **Phase 8** (appended) |
| E — Agent Debugger | **Phase 9** (appended) |
| F — Manifest Verifier | **Phase 9** (appended) |
| G — Evolve Agent | **Phase 10** (appended) |
| H — Algorithm 1 orchestration | **Phase 10** (appended) |
| I — Cross-model transfer | **Phase 11** (appended) |

Phases 8-11 are now in `PLAN.md` with operational step-by-step detail (Steps 8.1-8.4, 9.1-9.4, 10.1-10.5, 11.1-11.2).

**Note on numbering:** earlier drafts of this doc proposed Phases 6.5/7/8 for the AHE meta-stack. That conflicted with existing Phase 7 (CUDA Kernel Generation), so the meta-stack was renumbered to 8/9/10/11 to avoid collision. Existing Phases 0-7 unchanged.

### 6.Y — What we are explicitly committing to

Everything from the paper, in this order:
- 3 LLM-driven agents (Code Agent, Agent Debugger, Evolve Agent) — all sharing one base model (per paper)
- 7-slot NexAU substrate
- Change manifests with `expected_fix` and `expected_regression_risk`
- Layered trajectory distillation (per-task → benchmark-level)
- Algorithm 1 orchestration (`Rollout → Clean → Attribute/Rollback → Distill → Evolve → Commit`)
- Controllability + falsifiability invariants
- Cross-model transfer testing

What does NOT change about the project:
- Zero-diff invariant (still applies; meta-stack lives entirely in owned paths)
- Existing `PLAN.md` Phases 0-5 unchanged
- ML optimization remains the *product*; AHE is the *meta-layer*

### 6.Z — What can break this plan

1. **Stage C harder than expected.** Building a stable 50-task scored suite for ML optimization may take longer than building the meta-stack itself. Hardware availability, determinism, and scoring oracles are real engineering. Watch this stage closely.
2. **Compute budget for Stage H.** Paper does not quantify AHE's compute overhead. Ballpark: 10K+ LLM calls per round × 5+ rounds = 50K+ calls per benchmark cycle. Budget before launching.
3. **Regression attribution stays unreliable.** If Stage F's measured regression precision/recall mirror the paper's (33%/11%), Stage G's `expected_regression_risk` becomes a flag for human review, not an autonomous filter.
4. **Sub-additive interactions.** Per Table 3 of paper, full stack < sum of individual gains. Our 50-task suite may not have the statistical power to detect interaction effects cleanly. Stage component additions one at a time.

---

## 7. Risks, Watch-Outs, Anti-Patterns

### 7.1 Don't Let an LLM Rewrite the System Prompt Without Held-Out Validation

The paper's own ablation: system-prompt-only evolution scored **−2.3 pp**. The system prompt is the highest-leverage and highest-risk slot. Any automated edits there must be gated on held-out task performance, not just inner-loop scores.

### 7.2 Treat Regression Attribution as Unreliable

Their numbers: regression precision 11.8%, recall 11.1%. If we adopt change manifests, the `expected_regression_risk` field should be treated as a *flag for human review*, not a trustworthy filter. Until our attribution layer beats theirs (we have no reason to expect it will, initially), every committed harness change deserves a sanity-check rollout on a held-out subset.

### 7.3 Don't Optimize the Harness Before It Has a Workload

Our project's *core* objective is the ML optimization agent itself — the harness is a means. AHE-style evolution requires a stable, scored task suite to drive learning. The roadmap (Section 6) sequences Stage C (scored task suite) before Stages E-H (the meta-stack) for exactly this reason. Burning effort on Stages E-H before C is online means optimizing against a noisy metric, which is the failure mode the paper itself warns about (sub-additive interactions, regression blindness).

### 7.4 Component Interaction Is Real

Sub-additive aggregate gain is the empirical reality in the paper. Translation: do not assume that adding a memory tool *and* a middleware layer *and* a sub-agent config will give you the sum of their individual gains. They will interact, sometimes negatively. Stage additions, measure between each.

### 7.5 Don't Conflate Tool-Description and Tool-Implementation

This is one of the cleanest insights in the paper. In our `agent/tools/`, treat the docstring/schema (what the model sees) as a separate evolvable artifact from the executable code. A failure caused by "model misuses tool because docstring is misleading" should be fixable without touching the implementation, and vice versa.

---

## 8. Decision Framework

When deciding whether to apply an AHE technique to our project, run this check:

1. **Does our project currently have a stable scored task suite (Stage C complete)?** If no → only Stages A, B, D are unlocked. Stages E-I require C as a precondition.
2. **Is the proposed change auditable post-hoc?** If no → add a change manifest first. No edit without a hypothesis.
3. **Are we touching the system prompt slot?** If yes → mandatory held-out validation, manual review, no full automation (per paper's −2.3pp ablation).
4. **Is the gain we're targeting on the *hard* task tier?** If yes → AHE has weak evidence here (paper lost to TF-GRPO on hard tier). Be skeptical of expected ROI.
5. **Are we adding multiple components at once?** If yes → stage them, measure between. Non-additivity is real (Table 3).
6. **What's the compute budget for the round we're about to run?** If unknown → estimate before launching. Stage H rounds are not cheap.

---

## 9. Open Questions (For Further Investigation)

- **What is the actual compute overhead of the AHE loop?** Paper does not quantify. If it is 10× a single rollout, the cost-benefit calculus changes.
- **Does change-manifest discipline alone (without automation) capture most of the benefit?** I suspect it captures a large fraction — perhaps 50%+ of the architectural value, at near-zero cost. The paper does not isolate this.
- **How does AHE behave on tasks where the failure mode is *missing knowledge*, not *missing structure*?** ML optimization tasks lean heavily on domain knowledge. The paper's tasks are coding-flavored. Transfer is plausible but not demonstrated.
- **What is the regression-attribution failure mode in practice?** Paper gives precision/recall but does not characterize the *type* of regression that gets missed. Without that, we cannot design compensating controls.
- **Does the 7-slot decomposition over-fit to terminal-bench-style tasks?** Some slots (e.g. middleware, sub-agent config) may be more or less load-bearing for ML optimization workflows. We will not know until we have data.

---

## 10. References

- Lin et al., *Agentic Harness Engineering*, arXiv:2604.25850v2, April 2026.
- Project context: `CLAUDE.md`, `PLAN.md`, `SYSTEM.md` (this repo).
- Related baselines mentioned in the paper: ACE (prompt-only self-evolution), TF-GRPO (RL-style), Codex, terminus-2.

---

## 11. Final Engineering Judgment

**Decision (2026-05-01): build everything from AHE, sequenced by dependency.** Section 6 lays out the 9-stage roadmap (A-I). The paper's stack only converges when each component has its preconditions met: Code Agent before Debugger, scored task suite before Evolve Agent, manual manifest discipline before automation. Skipping order amplifies the paper's known failure modes.

**Stages A and D are free wins — start now.** 7-slot substrate declaration + manual manifest discipline cost near zero and produce immediate clarity benefits.

**Stage B is the entire current `PLAN.md` (Phases 1-5).** AHE does not displace the existing roadmap — it *frames* it. The Code Agent we are building IS the harness AHE will eventually evolve.

**Stage C (scored 50-task suite) is the rate-limiter.** Hardware availability, determinism, and scoring oracles for ML optimization are the hardest part of the entire program — harder than building the meta-stack on top.

**Stages E-I (meta-stack) are real engineering, not a sprint.** Compute budget and stable scoring are the binding constraints, not code complexity. Append Phases 6.5, 7, 8 to `PLAN.md` to capture them in operational detail.

**Treat the paper's empirical results with calibrated trust.** The headline win is real (+7.3pp Terminal-Bench 2); the hard-task tier loss, sub-additive ablation, and regression-blindness numbers are real warnings. Acceptance criteria at each stage are calibrated against these, not against best-case headline numbers. The authors themselves call AHE *"a controlled research prototype rather than a fully mature autonomous system"* — we are committing to make it production-grade for the ML optimization domain, eyes open about what that costs.

The bottom-line claim from the paper that I most agree with as a senior engineer: *the bottleneck is observability, not capability*. The roadmap above commits to building exactly that observability stack — across 9 stages, in dependency order.
