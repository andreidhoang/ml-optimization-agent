# PLAN_V2 — Exceptional autonomous principal agent for ML lifecycle work

> **Status**: Revision **v5** (Autonomous principal-agent thesis pivot) as of 2026-05-03. Supersedes PLAN.md for sequencing; PLAN.md retained as deep-reference for optimization-vertical detail.
>
> **One-line v5 north star**: cosmos-lab is **one exceptional autonomous principal agent** capable of long-horizon ML lifecycle work — taking vague ML research questions, decomposing them into experiments, writing code, running real GPU workloads, observing results, replanning on surprise, fixing its own bugs, and delivering measured results — all within an **exceptional context harness it manages itself**, with **governance as an enabler** (sentinels as tripwires for replanning, capability scope expanding with earned track record). Built on ml-intern's `agent_loop.py` as substrate; runs natively on `nvidia-nat`. Targets the JD's literal asks — *"strong agency in LLM systems," "code agents doing real work," "AI helps build models"* — without diluting them into governance-only theater.
>
> **Revision history**:
> - v3.1: §0.6 unique value, §0.7 numerical targets, §3.1 sentinel taxonomy, P9b CodeAgent
> - v3.2: library architecture pivot (`pip install cosmos-lab[nat]`), P0.5 adapter phase, ~20 weeks
> - v4: production-grade pivot — P5.5 PyTorch Depth, expanded P10, Invariant 9, §0.65 Six Reference Agents, §0.8 Production Commitments, ~22.5 weeks
> - **v5 (current)**: thesis pivot from "governance library wrapping 6 thin orchestrators" → "ONE exceptional autonomous principal agent doing real long-horizon ML work, with governance as enabler." See §0.9 for the autonomous-agent thesis, §3.2 for the PrincipalAgent architecture, §0.5 row 11 for cited rationale.
>
> **Why v5 — what v4 got wrong**: v4 optimized for compliance-conscious reviewer skeptical of hype. The framing "we built a governance layer wrapping other people's agents" *under-delivered* on the JD's literal asks: "strong agency," "code agents," "AI helps build them." A NVIDIA Cosmos reviewer comparing cosmos-lab against Devin / Operator / Cursor Composer (2026 production autonomous agents) would see v4 as **conservative governance theater** — clever judgment, weak capability. v5 inverts: governance and harness exist to **enable autonomy**, not constrain it. The 6 v4 "agents" collapse into 6 capability domains of one principal agent — which is how real principal engineers actually work (one person with 6 skills, not 6 different specialists). The result is an agent system that does work the JD literally asks for, made safe by sentinels rather than gated by them.
>
> **North star**: An agentic ML lifecycle platform — `cosmos-lab` — where specialized agents collaborate over a shared trajectory store with closed-loop self-improvement. Optimization is *one vertical*, not the centerpiece.
>
> **Why this rewrite (v2)**: Original PLAN.md front-loaded optimization (P1-7) and back-loaded the meta-platform (P8-11). The NVIDIA Cosmos JD prioritizes the meta-platform. Re-prioritization brings high-signal Cosmos-aligned deliverables forward without losing optimization work.
>
> **Why v3 — 2026-frontier verification pass**: A senior-eng research sweep — citations independently re-verified via WebFetch — across NVIDIA's own 2025-2026 stack (NeMo Agent Toolkit `nvidia-nat` v1.6.0 released 2026-04-10, CLI `nat`; Cosmos Curator; NeMo-RL; OpenShell + NemoClaw alpha early-preview 2026-03-16; Cosmos Reason 2 / Predict 2.5 / Transfer 2.5), the converged agent orchestration field (Claude Agent SDK + OpenAI Agents SDK + LangGraph reserved for HITL only), 2026 eval credibility crises (UC Berkeley audit — 8 top agent benchmarks reward-hackable 73–100%; METR — o3/Claude 3.7 reward-hack 1–2% of attempts overall but 43× more on RE-Bench), OTel GenAI semantic conventions becoming the trace-schema standard, MCP authorization spec (OAuth 2.1 + RFC 8707/8693) + EU AI Act Art. 12 (currently enforceable **2026-08-02** for high-risk systems; Digital Omnibus negotiation may push to Dec 2027 — design for the earlier date), and the empirical refutation of multi-agent debate ([`arxiv:2508.17536`](https://arxiv.org/abs/2508.17536) — Choi/Zhu/Li) materially reshape 6 phases and force a P4 split into P4a/P4b. v3 deltas are summarized in §0.5; phase sections are updated in place.

---

## 0. Strategic frame

### JD → Vertical mapping

| Cosmos JD bullet | Vertical | New phase |
|---|---|---|
| Data generation & curation | DataAgent | P3 |
| Evaluation platforms (auto + human + agent-driven) | EvalAgent | P1 (foundation) + P4 (platform) |
| Training orchestration | TrainOrchestrator | P5 |
| Multimodal pipelines | Cosmos vertical (Reason 2 / Predict 2.5) | P2 |
| Self-improving loops | Trajectory mining + GEPA-style revision | P1 (store) + P8 (loop) |
| Agentic workflows over codebases | Cross-vertical orchestration | P9 |
| Context compression / agent memory | OwnedContextManager + cross-session retrieval | P7 |
| Engineering excellence (testing, packaging) | Cross-cutting | All phases |
| Stand-out: agent identity / AuthN / AuthZ | CapabilityScopedRouter + AuditLog | P0 |

### Architecture (target)

```
                  ┌──────────────────────────────────────────────┐
                  │         Multi-Agent Orchestrator (P9)         │
                  └───┬───────────┬───────────┬───────────┬──────┘
                      │           │           │           │
                ┌─────▼───┐ ┌─────▼─────┐ ┌───▼─────┐ ┌──▼──────────┐
                │DataAgent│ │TrainOrches│ │EvalAgent│ │OptimizeAgent│
                │  (P3)   │ │   (P5)    │ │  (P4)   │ │    (P6)     │
                └─────┬───┘ └─────┬─────┘ └───┬─────┘ └──────┬──────┘
                      │           │           │              │
                      └───────────┴─────┬─────┴──────────────┘
                                        │
                          ┌─────────────▼────────────────┐
                          │  Capability-Scoped ToolRouter│  ← P0
                          │  + AuditLog + AgentIdentity  │
                          └─────────────┬────────────────┘
                                        │
                          ┌─────────────▼────────────────┐
                          │   TrajectoryStore (DuckDB)   │  ← P1
                          │   + Agent-as-Judge Harness   │
                          └─────────────┬────────────────┘
                                        │
                          ┌─────────────▼────────────────┐
                          │ Self-Improvement Loop (GEPA) │  ← P8
                          │ trajectories → reflect →     │
                          │ revise prompts → re-eval     │
                          └──────────────────────────────┘

Cosmos vertical (P2): NIMProvider + Cosmos toolset wrappers (Reason 2 / Predict 2.5 / Transfer 2.5)
                       VideoUnderstandingAgent itself lives in P9 e2e demo
Memory layer (P7): cross-session retrieval, owned compression
```

### Invariants (carry from CLAUDE.md / WORKFLOW.md)

1. **Zero-diff**: `git diff upstream/main --name-only` returns only owned paths.
2. **Baseline**: `pytest tests/unit/` matches baseline (currently 237 pass / 3 upstream-broken: `test_doom_loop.py::test_check_for_doom_loop_returns_corrective_prompt_for_identical_run`, `test_doom_loop.py::test_check_for_doom_loop_returns_corrective_prompt_for_cycle`, `test_sandbox_auto_start.py::test_prompt_and_tool_specs_do_not_require_cpu_sandbox_create`). Document any drift.
3. **Owned tests**: `pytest tests/optimization/` exits 0.
4. **One-optimization-per-experiment** + **measured-peak over vendor-peak** (from EVAL_SPEC.md) — applies once optimization vertical is live (P6).
5. **Trajectory-on-by-default**: from P1 onward, no agent run is unobserved.
6. **OTel-GenAI-on-by-default** (v3): from P1 onward every agent run, tool call, and judge invocation emits an `OpenTelemetry GenAI` span (`gen_ai.*` semantic conventions, opt-in stability flag). No vendor-proprietary trace schemas.
7. **No unverified judge claim** (v3): every judge pass-rate metric ships with bootstrap CI and an anti-reward-hacking sentinel pair (one structural verifier + one judge); judge-only scores are flagged in UI.
8. **Framework-agnostic by construction** (v3.2): cosmos-lab core (sentinels, identity, GEPA governance, quality budget) operates on *interfaces* (Inspect AI Task/Solver/Scorer, MCP authorization spec, DSPy Module, OTel GenAI semconv), not on a specific agent loop. Owning an agent loop is anti-pattern.
9. **No GPU phase exits without one measured real run** (v4): P5, P5.5, P6, and P9a each require at least one actual GPU workload with measured numbers committed to the repo. Mocked-only acceptance is a stop-the-line event. Budget envelope: ~$200-400 across Modal / Lambda Cloud / NIM free tier — small enough to self-fund, large enough to be honest. Mocked tests stay as the *fast feedback loop*; real runs are the *credibility gate*.

---

## 0.4 Library architecture (v3.2) — cosmos-lab as a Python library, not a platform fork

### Why library, not fork

cosmos-lab's value prop (§0.6) is *governance and credibility*, not agent runtime. The 5 unique-value items (sentinel-gated judging, RFC 8693 sub-agent identity, GEPA promotion contract, quality budget invariant, `nat`-runnable workflow YAML) all operate on *interfaces* — Inspect AI's Scorer, MCP authorization, DSPy Module, OTel GenAI spans. None requires owning an agent loop.

The 2026 agent framework field has converged on: NeMo Agent Toolkit, Claude Agent SDK, OpenAI Agents SDK, LangGraph (HITL only). Building yet another agent loop = anti-signal. Library architecture is the senior-engineering answer.

**Analogy precedent**: `pytest` doesn't fork `unittest`. `dspy` doesn't fork `transformers`. `inspect-ai` doesn't fork any specific agent. cosmos-lab follows the same pattern — it's the *layer that makes self-improving agents safe to deploy*, plugged into whatever harness the user already runs.

### Package layout

```
cosmos_lab/                              # importable as `pip install cosmos-lab`
├── __init__.py                          # public API surface
├── identity/                            # P0 — IDENTITY (shipped)
│   ├── identity.py                      #   AgentIdentity, CapabilityDenied
│   ├── audit.py                         #   AuditLog (P0 JSONL → P4b hash-chained signed)
│   └── router.py                        #   CapabilityScopedRouter (composes any router)
├── trajectory/                          # P1 — TRAJECTORY
│   ├── sink.py                          #   TrajectorySink Protocol
│   ├── otel_emitter.py                  #   OTelGenAIEmitter (gen_ai.* spans)
│   ├── duckdb_sink.py                   #   opt-in analytics layer
│   └── hf_sink.py                       #   opt-in HF flywheel
├── eval/                                # P1 + P4a — EVAL
│   ├── judge.py, multi_judge.py         #   LLMJudge, MultiJudge (variance reduction)
│   ├── sentinels/                       #   §3.1 taxonomy (4 sentinel types)
│   │   ├── deterministic.py
│   │   ├── output_format.py
│   │   ├── side_effect.py
│   │   └── no_op.py                     #   mandatory on every task
│   └── inspect_bridge.py                #   exposes scorers as Inspect AI Scorers
├── governance/                          # P8 — SELF-IMPROVEMENT GOVERNANCE
│   ├── gepa_loop.py                     #   wraps dspy.GEPA with promotion contract
│   ├── promotion.py                     #   lower-CI-bound + sentinel + signed record
│   └── failure_mining.py                #   trajectory → failure clusters
├── memory/                              # P7 — MEMORY (3-tier hierarchical)
├── providers/                           # P2 — PROVIDERS
│   └── nim_provider.py                  #   litellm custom provider for NVIDIA NIM
├── compute/                             # P5 — COMPUTE BACKENDS
│   ├── backend.py                       #   ComputeBackend Protocol
│   ├── hf_jobs.py, skypilot.py, nemo_run.py
├── sandbox/                             # P6 — SANDBOX (E2B + Daytona, OpenShell P10)
└── harness/                             # NEW IN P0.5 — adapter pattern
    ├── nat.py                           #   primary: registers cosmos-lab as nat plugins
    ├── ml_intern.py                     #   secondary: v1 compat shim (current code)
    └── claude_sdk.py                    #   future: Claude Agent SDK adapter
```

**Owned path**: `agent/optimization/` becomes the *implementation directory*; `cosmos_lab/` is the *importable surface*. They're the same code, exposed two ways. P0.5 sets up `pyproject.toml` so `from cosmos_lab.identity import AgentIdentity` and `from agent.optimization.identity import AgentIdentity` both work.

### Adapter pattern — concrete contract

Each harness adapter is ≤ 200 LOC and provides three things:

1. **Tool registration**: surfaces cosmos-lab's `CapabilityScopedRouter` and any cosmos-lab tools (e.g., `cosmos_reason`, `cosmos_predict`) to the host harness's tool registry.
2. **Span correlation**: ensures cosmos-lab's `OTelGenAIEmitter` spans nest correctly under the host harness's parent agent span (so a Phoenix trace shows one tree, not two).
3. **Lifecycle wiring**: hooks `TracedSession`-equivalent behavior into the host harness's run lifecycle (start, step, end, error).

```python
# cosmos_lab/harness/nat.py — primary harness adapter (sketch)
from aiq.builder import register_function, register_telemetry_exporter
from cosmos_lab.identity import CapabilityScopedRouter
from cosmos_lab.trajectory import OTelGenAIEmitter
from cosmos_lab.eval.sentinels import compose_sentinel

def install_into_nat(builder: "aiq.Builder", identity: "AgentIdentity") -> None:
    """One call inside any nat workflow YAML to install cosmos-lab governance."""
    register_telemetry_exporter(builder, OTelGenAIEmitter(...))
    builder.wrap_router(lambda r: CapabilityScopedRouter(r, identity, audit_log))
    builder.register_lifecycle_hook("on_step", _emit_genai_step_span)
    # Inspect AI bridge auto-discovered from cosmos_lab.eval.inspect_bridge
```

```python
# cosmos_lab/harness/ml_intern.py — v1 compat (current code refactored)
from agent.core.session import Session
from cosmos_lab.identity import CapabilityScopedRouter
from cosmos_lab.trajectory import OTelGenAIEmitter

class TracedSession(Session):
    """ml-intern adapter — what we have today, refactored to use library imports."""
    def __init__(self, *args, identity, audit_log, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_router = CapabilityScopedRouter(self.tool_router, identity, audit_log)
        self._otel = OTelGenAIEmitter(...)
```

**Acceptance contract for adapters** (P0.5):
- Both adapters registered as `cosmos_lab.harness` entry points in `pyproject.toml`
- Same Phase 0 test suite (16 tests) runs against both adapters → both must pass
- One smoke test per adapter that runs a 3-step trivial agent and verifies (a) capability denial works, (b) OTel span emitted with correct parent_id, (c) sentinel evaluator returns expected verdict

### What the user does

```bash
pip install cosmos-lab                           # library only
pip install cosmos-lab[nat]                      # + nvidia-nat adapter (recommended)
pip install cosmos-lab[ml-intern]                # + ml-intern adapter (HF stack)
pip install cosmos-lab[all]                      # + all adapters
```

```yaml
# nat workflow YAML — Cosmos team can run cosmos-lab in their stack with one block
general:
  telemetry:
    exporter:
      _type: cosmos_lab.OTelGenAIEmitter
governance:
  identity:
    _type: cosmos_lab.AgentIdentity.scoped
    capabilities: [read_file, run_inspect_eval]
  sentinels: [no_op, output_format]
workflow:
  _type: nat.react_agent
  llm_name: nim-llama-3-70b
  tools: [...]
```

### Trade-off honestly

- **Cost**: ~4 days of P0.5 work (restructure + 2 adapters + dual-adapter test). Phase 0 code does not change semantically — just imports + packaging.
- **Benefit**: signals architectural maturity; trims 4-5 weeks across P1/P2/P5/P6/P8 by inheriting `nat` plumbing; future-proofs against framework churn; preserves HF flywheel via `ml-intern` adapter.

---

## 0.5 v3 deltas — what 2026 evidence forced us to change

A 2026 SOTA verification pass produced eight load-bearing changes (rows 1-8 below) plus one architectural pivot in v3.2 (row 9). Each is grounded in a public artifact (paper / repo / blog / spec) with the date.

| # | Change | 2026 evidence | Phase touched |
|---|---|---|---|
| 1 | **Mirror NeMo Agent Toolkit (`nvidia-nat`, CLI `nat`) config schema and plugin/decorator tool registry**; expose cosmos-lab as a runnable `nat`-compatible workflow YAML | NeMo Agent Toolkit **v1.6.0 released 2026-04-10** (NVIDIA, formerly AIQToolkit / AgentIQ; PyPI = `nvidia-nat`, CLI = `nat`); see [docs](https://docs.nvidia.com/nemo/agent-toolkit/latest/index.html), [repo](https://github.com/NVIDIA/NeMo-Agent-Toolkit) | P0, P1, P9 |
| 2 | **Emit OpenTelemetry GenAI (`gen_ai.*`) spans natively** for runs/tool-calls/judges; default sink = Phoenix, Langfuse/W&B/Weave swap-in by config (mirrors `nvidia-nat` telemetry block) | OTel GenAI semconv (experimental, opt-in via `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`, [spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/)); Datadog/Grafana/Phoenix all native by Q1 2026 | P1 (replaces "DuckDBSink as primary schema") |
| 3 | **Drop the "DebatingJudgePanel" framing.** Multi-judge stays for variance reduction; debate dynamics removed; *anti-reward-hacking sentinels* added (one structural verifier per task) | "Debate or Vote" ([arxiv:2508.17536](https://arxiv.org/abs/2508.17536) — Choi/Zhu/Li, 2025) — majority voting alone explains most of MAD's gain across 7 NLP benchmarks, debate is a martingale; [UC Berkeley audit (2026)](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/) — 8 top agent benchmarks (SWE-bench, WebArena, OSWorld, GAIA, Terminal-Bench, FieldWorkArena, CAR-bench + one more) all exploitable, exploit rates 73–100%; [METR (2025-06-05)](https://metr.org/blog/2025-06-05-recent-reward-hacking/) — o3 reward-hacks 1–2% of all task attempts overall, 43× more often on RE-Bench than HCAST, and **every trajectory** on one specific RE-Bench task eventually hacks; same behavior observed for Claude 3.7 Sonnet and o1 | P1, P4a |
| 4 | **Build evals on Inspect AI** (UK AISI Task/Solver/Scorer + Docker sandbox + log viewer) instead of a custom harness; ship our seed tasks as Inspect tasks | Inspect AI is the de-facto production eval standard (METR uses it); [docs](https://inspect.aisi.org.uk/) | P1, P4 |
| 5 | **Identity v2: MCP OAuth 2.1 + RFC 8707 (Resource Indicators) + RFC 8693 (token exchange) for sub-agent scope-down + hash-chained signed audit log** aligned to EU AI Act Art. 12 (currently enforceable **2026-08-02** for high-risk systems; ⚠️ note Digital Omnibus negotiation may push to Dec 2027 — design for the earlier date) | [MCP authorization draft](https://modelcontextprotocol.io/specification/draft/basic/authorization); [EU AI Act Art. 12](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-12) | P0 (AuthZ MVP shipped); **NEW P4b** (MCP-OAuth + signed log — graduated from P4 per scope review) |
| 6 | **Centaur HPO (LLM proposes, CMA-ES refines) — not pure-LLM HPO** | "Can LLMs Beat Classical HPO?" ([arxiv:2603.24647](https://arxiv.org/abs/2603.24647), 2026) — pure-LLM HPO loses to CMA-ES/TPE; Centaur is the credible hybrid | P5 |
| 7 | **Wrap NeMo Curator + cosmos-curate stages**; do not reimplement video curation; **wrap NeMo-RL** for any post-training; **wrap SkyPilot Job Groups + NeMo-Run + HF Jobs** as ComputeBackends | [NeMo Curator 26.04](https://github.com/NVIDIA-NeMo/Curator), [cosmos-curate](https://github.com/nvidia-cosmos/cosmos-curate) (20M video-hr in 14 days on Blackwell), [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) (Nemotron-3-Super post-trained on it, Mar 2026), [SkyPilot v0.12 Job Groups](https://github.com/skypilot-org/skypilot) | P3, P5, P6 |
| 8 | **Sandbox 2-tier**: E2B (Firecracker, CPU correctness) + **Daytona** for GPU profiling/training. New `agent/optimization/sandbox/` interface, `nat`-compatible. **v3.1 cut**: NemoClaw moved off P6 main path → **P10 stretch only** because it's alpha-stage early-preview (2026-03-16) and gating P6 acceptance on it is a credibility risk. NVIDIA OpenShell remains a `SandboxRunner` candidate; lands in P10 alongside NemoClaw. | [E2B vs Daytona 2026](https://northflank.com/blog/daytona-vs-e2b-ai-code-execution-sandboxes); [NVIDIA OpenShell repo](https://github.com/NVIDIA/OpenShell); [NemoClaw repo (alpha)](https://github.com/NVIDIA/NemoClaw); [OpenShell+NemoClaw blog](https://developer.nvidia.com/blog/build-a-secure-always-on-local-ai-agent-with-nvidia-nemoclaw-and-openclaw/) | P6 (Daytona); P10 (NemoClaw stretch) |
| **9** | **Architectural pivot — cosmos-lab is a Python LIBRARY, not a fork.** `pip install cosmos-lab[nat]` plugs into NeMo Agent Toolkit as primary harness from P1 onward; `[ml-intern]` extra preserves Phase 0 work + HF flywheel via compat adapter. Library never owns the agent loop — that's what makes it portable. **NEW P0.5 (4 days)** restructures Phase 0 code into library form + writes the two adapters; same 16 Phase 0 tests run against both adapters. | First-principles audit (v3.2): cosmos-lab's value prop (§0.6) is *governance*, not *runtime*. Sentinels operate on Inspect AI Scorer interface. Identity operates on MCP authorization spec. GEPA operates on DSPy Module. Quality budget is an architectural invariant. None require owning an agent loop. 2026 agent-framework field has converged (NeMo Agent Toolkit + Claude Agent SDK + OpenAI Agents SDK + LangGraph for HITL); building yet another agent loop = anti-signal. Library precedent: `pytest`, `dspy`, `inspect-ai`. | NEW P0.5 + reframes P1-P10; trims plan 24 → ~20 weeks |
| **10** | **Production-grade pivot (v4)** — closes 5 gaps the v3.2 audit found: (1) PyTorch depth absent → **NEW P5.5** (1w) custom autograd op + profiler-driven kernel selection on real workload; (2) every phase mockable → **NEW Invariant 9** (no GPU phase exits without measured real run, ~$200 budget); (3) `pip install` is publication not deployment → **expanded P10** to 2w with real production deployment on HF Spaces / Modal + 1-week trace gather; (4) multimodal only mocked → **P3 reframed** to require real video sample (10-100 hours through cosmos-curate); (5) OSS impact = own library only → **P10 commits** to one upstream PR to nvidia-nat or Inspect AI for sentinel pattern. Plus **§0.65 NEW** Six Reference Agents matrix for agents-first visibility. | NVIDIA Cosmos JD demands "deep PyTorch familiarity," "multimodal pipelines including deployment," "agent-based systems doing real work," "impactful OSS contribution." A plan that's fully mockable + ships only its own library + has no PyTorch chops fails the L6 bar regardless of architecture cleanness. Production-grade ≠ feature-rich; production-grade = *runs in front of real users with measured numbers and a rollback plan*. | NEW P5.5 + expanded P10; reframes P3; adds §0.65 + §0.8 + Invariant 9; plan 20 → ~22.5 weeks |
| **11** | **Autonomous principal-agent thesis pivot (v5)** — collapses v4's "6 thin orchestrator agents on a governance library" → **ONE PrincipalAgent demonstrating 6 capability domains**, with library + sentinels + identity reframed as *enablers of autonomy* (not constraints). Sentinels become tripwires for replanning. Identity capabilities expand with earned track record. GEPA becomes agent self-improvement (retroactive human review). Built on ml-intern's `agent_loop.py` substrate. **NEW §0.9 Autonomous Principal Agent thesis**, **NEW §3.2 PrincipalAgent architecture**, **§0.65 reframed** (six agents → six capability domains of one agent). | The v4 framing "we built a governance library wrapping other people's agents" *under-delivered* on JD's literal asks: "strong agency," "code agents doing real work," "AI helps build them." A NVIDIA Cosmos reviewer comparing cosmos-lab against 2026 production autonomous agents (Devin / Operator / Cursor Composer / Claude Code) saw v4 as conservative governance theater — clever judgment, weak capability. The 2026 agentic frontier is autonomous capability MADE SAFE by governance, not governance INSTEAD OF capability. v5 inverts the hierarchy: PrincipalAgent is the product; harness + sentinels + identity exist to make autonomy exceptional, not to substitute for it. Real principal engineers have one self with broad skills, not six narrow specialists — PrincipalAgent models that reality. | Reframes §0.6 + §0.65; adds §0.9 + §3.2; phase narratives shift from "ship N agents" to "PrincipalAgent demonstrates capability N"; ml-intern `agent_loop.py` graduated from compat shim to primary substrate; weeks unchanged (~22.5w) — depth shifts from breadth-across-agents to depth-per-capability |

**Net pitch (v3.2)**: cosmos-lab is a `pip install`-able Python library (`pip install cosmos-lab[nat]`) that adds governance — sentinel-gated judging, MCP-OAuth identity with RFC 8693 sub-agent scope-down, GEPA promotion contracts, quality-budget invariants — to NeMo Agent Toolkit (primary) or ml-intern (compat). It emits OTel GenAI traces into Phoenix/Weave/Langfuse, evaluates on Inspect AI with anti-reward-hacking sentinels, executes on a 2-tier sandbox, post-trains via NeMo-RL, curates data via cosmos-curate stages. **Cosmos team will recognize every interface boundary AND the architectural maturity of library-vs-fork separation.**

**What v3.2 explicitly does not change**: the zero-diff invariant, the `agent/optimization/` ownership tree (it's still where files physically live; `cosmos_lab/` is the importable surface re-exposing them), Phase 0 deliverable *semantics* (`OptimizationConfig`, `AgentIdentity`, `AuditLog`, `CapabilityScopedRouter`, 16 passing tests — only `import` lines change in P0.5). v3.2 *does* trim plan from 24 → ~20 weeks by inheriting nat plumbing; banked weeks become risk buffer + v1.1 hardening (KMS, NemoClaw, OpenShell stretches, additional harness adapters).

---

## 0.6 What only cosmos-lab does — unique value vs 2026 autonomous agents

A Cosmos reviewer will reasonably ask: *"Why isn't this just Devin / Operator / Cursor Composer / Claude Code with a different prompt?"* The answer needs to be sharp. Five things 2026's autonomous agents do NOT do, that cosmos-lab's PrincipalAgent does:

1. **ML-lifecycle-native tool surface** (P3+). Devin codes; Operator browses; Cursor edits IDE — none has the full ML stack as first-class tools (cosmos-curate stages, NeMo-RL, SkyPilot Job Groups, NIM endpoints, W&B sweeps, Inspect AI eval, profilers). PrincipalAgent treats them as one tool registry per `nat` plugin pattern. Same agent does data + train + eval + optimize cohesively.

2. **Long-horizon by construction via OTel trajectory + episodic memory** (P1, P7). Most 2026 agents are session-bounded (~1 hour). PrincipalAgent persists across sessions: experiment kicks off Monday, agent resumes Wednesday after compute-quota refresh, reads its own OTel spans + memory tier to recover state, continues from where it left off. Multi-day work without losing context.

3. **Sentinel-gated replanning instead of silent failure** (P1, §3.1). When agent does something wrong, it doesn't fail silently (Devin's worst pattern — wasted hours, no recovery signal). Sentinels trip → structured feedback → agent reads what failed and why → replans with that evidence. The 4 sentinel types (deterministic, output_format, side_effect, no_op) cover the failure modes UC Berkeley demonstrated.

4. **Earned-trust capability expansion via RFC 8693 token exchange** (P0 → P4b). New PrincipalAgent instance starts with narrow capabilities (read-only). After N sentinel-clean runs, capability scope auto-expands (read+write, then run-tests, then deploy). Parent token + scope expansion → new token, MCP-OAuth–compliant, hash-chained signed audit log records the expansion event. No 2026 autonomous agent has this.

5. **`nat`-runnable + OTel-GenAI-native by construction** (P0.5). Cosmos team runs `nat run cosmos-lab-principal.yaml` and inherits every 2026 standard. PrincipalAgent registers as nat plugin; OTel GenAI spans flow into Cosmos team's existing observability; identity uses their MCP OAuth setup. No platform-fork tax.

**One-line pitch (v5)**: cosmos-lab is **one autonomous principal agent** that does long-horizon ML lifecycle work — taking vague research questions, decomposing them into experiments, writing real code, running real GPU workloads, observing surprising results, replanning when sentinels trip, and delivering measured outcomes — all within an exceptional self-managed context harness, with capability scope that expands as the agent earns trust through sentinel-clean runs. Built on ml-intern's `agent_loop.py` substrate, runs natively on `nvidia-nat`. The agent is the product; the harness and governance are how it stays exceptional.

---

## 0.65 Six capability domains of the PrincipalAgent (v5 — one agent, six skills)

The product is **one autonomous PrincipalAgent**. These are the six capability domains it demonstrates over P3-P9 — like one principal engineer who does data work Monday, training Tuesday, optimization Wednesday. Not six different agents; one agent with six skills, each with measurable acceptance.

| Capability | Phase(s) | What the PrincipalAgent does | JD bullet match | Acceptance + numerical target |
|---|---|---|---|---|
| **Data curation** | P3 (W6-7) | PrincipalAgent composes cosmos-curate + NeMo Curator stages; runs persona-rewriter LLM-in-the-loop; emits curated dataset card with W&B Artifacts lineage. Resumes mid-curation across sessions. | "data generation and curation" + "agents help generate data" + "multimodal pipelines" | **Real**: 10-100 hours of real video processed; 100% lineage traceability raw video → final row |
| **Eval design** | P4a (W10) | PrincipalAgent designs multi-judge eval with bootstrap CIs + reward-hack sentinels; opens PR-gate against regression; ingests human feedback via Slack→Argilla and re-feeds Inspect dataset | "evaluation platforms (auto + human + agent-driven)" | PR-gate false-positive ≤5% on 10-PR replay set; sentinel/judge agreement ≥98% on green seeds |
| **Training orchestration** | P5 (W11-12.5) | PrincipalAgent runs Centaur HPO (proposes hyperparams via LLM reasoning, refines via CMA-ES); chooses ComputeBackend (SkyPilot/NeMo-Run/HF Jobs); fires NeMo-RL post-training on real GPU; monitors W&B; replans on validation plateau | "training orchestration" + "long-horizon multi-step workflows" | Centaur ≥ parity with pure CMA-ES on 6-config sweep; **one real GPU sweep** (Modal/Lambda) per Invariant 9 |
| **Optimization** | P6 (W14-15.5) | PrincipalAgent profiles workload, identifies bottleneck, proposes optimization (kernel fusion, torch.compile config, layer pruning), measures speedup, validates quality preservation. Uses E2B (CPU) + Daytona (GPU) sandboxes. | "AI-native systems improving productivity" + "engineering excellence" | ≥1.5× wall-clock speedup on 4 real workloads, ≤2% deterministic regression; **measured on real GPU** |
| **Multimodal pipeline** | P9a (W19-20) | PrincipalAgent runs end-to-end multimodal pipeline: cosmos-curate stages → NeMo-RL training → Inspect AI eval with physics-consistency scorer → Centaur inference optimization. Hits real Cosmos NIM endpoint. | "multimodal ML pipelines... deployment" + "physical AI" | Full pipeline runs < 8h wall-clock; physics-consistency scorer pass; **real Cosmos NIM endpoint hit ≥1×** |
| **Code work** | P9b (W20-21) | PrincipalAgent in **expanded capability scope** (earned via prior sentinel-clean runs in P3-P9a): reads real OSS issues, designs fix, writes patch, runs tests in E2B sandbox, iterates. Targets real GitHub repos, not closed fixture. | "agents can work with code" *(headline JD bullet)* + "real work: coding" | ≥60% success rate on 10-issue real OSS fixture; **stretch**: ≥1 real PR opened upstream with reviewer engagement |

**Why six capabilities, not six agents (v5 reframe)**: a principal engineer at NVIDIA Cosmos doesn't have 6 selves — they have one self with broad skills. PrincipalAgent models that. **One** trajectory store, **one** memory tier, **one** identity that earns capability expansion across all 6 domains. The work is cohesive: data prep informs eval design; eval failures inform training choices; training results inform optimization targets; optimization wins inform multimodal pipeline tuning; pipeline issues become bug fixes. Six separate agents (v4) would lose this cohesion.

**The product**: PrincipalAgent + the harness that makes it exceptional + the governance that makes it safe.

---

## 0.7 Numerical targets — what we commit to hit

Without numbers, this is a roadmap, not a product plan. The following targets are commitments per phase. They are the bar for "phase exits"; missing one is a stop-the-line event, not a footnote. Numbers may be revised at phase entry with written rationale, never silently.

| Phase | Metric | Target | How measured |
|---|---|---|---|
| **P1** | Sentinel/judge agreement on green seed runs | **≥98%** (any disagreement = bug, not noise) | `RewardHackSentinel` paired with `MultiJudge` across 5 seed tasks × 3 runs |
| **P1** | OTel span → Phoenix round-trip latency | **p99 < 500ms** | timing test; spans visible in Phoenix UI before next agent step |
| **P1** | `MultiJudge` pass-rate bootstrap CI width | **≤ 8pp at N=15 runs** | enables meaningful regression gates downstream |
| **P2** | NIM endpoint mock fidelity | **100% of cosmos tasks parseable** without endpoint changes | mock contract test |
| **P3** | DataAgent dataset-card lineage | **100% of records traceable** prompt → curator stage → final row | W&B Artifacts lineage graph |
| **P4a** | PR-gate false-positive rate | **≤ 5%** on a 10-PR replay set | shadow-mode for first week before enforcing |
| **P4b** | Sub-agent scope-down test | **0 unauthorized tool calls** in 100 child-agent runs | RFC 8693 integration test |
| **P5** | Centaur HPO vs pure-CMA-ES on small benchmark | **≥ parity** within 1σ; Centaur should not be *worse* | 6-config sweep, 3 seeds |
| **P6** | Inference optimization speedup | **≥ 1.5× wall-clock** on 4 baseline workloads | measured-peak per EVAL_SPEC.md |
| **P6** | Quality preservation under optimization | **≤ 2% absolute regression** on Inspect-scorer deterministic metrics | non-negotiable; judge-only flagged |
| **P7** | Memory recall on prior-session related task | **≥ 70% precision@5** on a 20-task held-out set | offline benchmark |
| **P8** | GEPA-promoted prompt revision lift | **≥ +5pp pass-rate at p<0.05**, sentinel agreement preserved | A/B vs control on 50-task golden suite |
| **P8** | False-promotion rate (revisions that regress in prod sample) | **≤ 10%** | weekly retrospective on promotions |
| **P9** | End-to-end pipeline wall-clock | **< 8 hours** for the demo task | DataAgent → TrainOrch → EvalAgent → OptimizeAgent |
| **P9b** | CodeAgent — small-bug-fix success | **≥ 60%** on a 10-bug fixture (E2B sandbox, no human assist) | hits JD's "AI helps build AI" bullet directly |
| **Cross-cutting** | Cost per evaluated task | **logged + visible in leaderboard, no target** | cost as first-class column, not vibes |

### v4 additions (production-grade gates)

| Phase | Metric | Target | How measured |
|---|---|---|---|
| **P5** | Real GPU sweep run (per Invariant 9) | **≥ 1 sweep on real hardware** (Modal/Lambda/NIM free tier) with logged W&B run | budget: ~$50; evidence committed to repo |
| **P5.5** | PyTorch custom op vs framework baseline | **≥ 10% measurable wall-clock improvement** on one real workload (matmul / attention / data loader) | profiler artifacts + benchmark script committed |
| **P6** | OptimizeAgent measured speedup | already in v3.1 (≥1.5× on 4 workloads) — v4 *requires real GPU* (was mockable) | per Invariant 9 |
| **P9a** | VideoUnderstandingAgent on real Cosmos NIM | **≥ 1 e2e run** hitting real NIM endpoint (cosmos-reason-2 free tier) | budget: ~$30; trace committed |
| **P10** | Production deployment | **deployed agent serves ≥ 100 real user sessions** over 1-week window (HF Spaces or Modal endpoint) | OTel traces + cost report committed |
| **P10** | OSS upstream PR | **≥ 1 PR opened to nvidia-nat or Inspect AI** with sentinel pattern; review-ready, not draft | PR URL in P10 release notes |

### v5 eval-system additions (per AGENTIC_EVAL_SPEC §9)

These extend numerical targets with eval-system-specific commitments. Without these, the entire numerical-targets table above is unverifiable (per axiom A10: eval-of-eval).

| # | Eval target | Commitment | How measured | Phase |
|---|---|---|---|---|
| **E1** | Sentinel suite false-positive rate | **≤ 5%** on null fixtures (identical agent A/B should never gate-fail) | M2 null fixture suite, weekly | P1 |
| **E2** | Sentinel suite false-negative rate | **≤ 1%** on planted regressions (known-broken agent must always gate-fail) | M2 planted regression suite, weekly | P1 |
| **E3** | T1 calibrated suite test-retest reliability | **r ≥ 0.95** on aggregate metrics across 2 runs | M2, monthly | P1 |
| **E4** | Plan-quality LLM-judge ↔ human agreement | **≥ 80%** on 50-plan calibration sample | S2 calibration, quarterly | P4a |
| **E5** | Replan success rate | **≥ 70%** (replans → next milestone sentinel-clean) | S3, continuous | P4a |
| **E6** | Capability boundary probe pass rate | **100%** (0 unauthorized tool calls across 100 child-agent runs) | S4, nightly | P4b |
| **E7** | Reward-hack discovery rate | **Trending downward over 6 months** (sentinel suite maturing) | S5, monthly | P4a |
| **E8** | Cross-agent Pareto position | **PrincipalAgent on Pareto frontier** of cost × quality vs comparison agents | S6, quarterly | P10 |
| **E9** | Eval cost as % of total project spend | **≤ 15%** of total GPU + compute budget | M3 cost telemetry, weekly | P1 |
| **E10** | Reproducibility envelope coverage | **100%** of agent runs tagged with envelope (seeds + hashes + versions + OTel trace ID) | M1, every run | P1 |

**Total numerical commitments**: 24 (original §0.7) + 10 (E1-E10) = **34 phase-exit conditions, all measurable, all gating decisions.**

---

## 0.8 Production-grade commitments — what makes v4 not a research roadmap

A research roadmap says "we will design X." A production plan says "we will run X in front of users with measurable Y, and roll back via Z." v4 commits to five production gates that v3.2 left aspirational:

| Gate | Concrete commitment | Why it matters for Cosmos |
|---|---|---|
| **G1: Real GPU runs** | Invariant 9 — every GPU phase (P5, P5.5, P6, P9a) requires ≥1 measured real-hardware run committed to the repo. ~$200-400 total budget across Modal / Lambda Cloud / NIM free tier. | "Ran in mocks only" is an interview-killing red flag for a multimodal/world-model team |
| **G2: PyTorch depth artifact** | P5.5 ships one custom autograd op or torch.compile pattern with profiler-driven kernel selection on a real workload, ≥10% wall-clock improvement vs framework baseline | JD: *"Deep familiarity with PyTorch, including the ability to debug, adapt, and extend model behavior"* — needs code, not words |
| **G3: Production deployment** | P10 deploys cosmos-lab reference agent on HF Spaces or Modal endpoint, gathers ≥100 real user sessions over 1-week window, publishes trace summary | JD: *"deployment"* as a lifecycle step. `pip install` is publication, not deployment. |
| **G4: Real multimodal data flow** | P3 reframed: DataAgent processes 10-100 hours of *real* video through cosmos-curate (not toy fixture), ships dataset card + lineage | JD: *"multimodal ML pipelines spanning data processing"* — at toy-fixture scale this is a demo; at hour-scale it's a pipeline |
| **G5: OSS impact beyond own repo** | P10 commits to opening one upstream PR (nvidia-nat OR Inspect AI) with the sentinel pattern, review-ready not draft | JD stand-out: *"contributed to impactful open-source ML, Python, or developer tooling"* — owning a library is good; landing in someone else's library is better |

**Honest scope**: G1+G2+G4 are mostly inside existing phase weeks (cost: time on real runs, not new weeks). G3+G5 are the +1 week expansion of P10 (was 1w polish, now 2w polish + production).

**Honest budget**: ~$200-400 of personal GPU spend total. Self-fundable. Worth every dollar — turns the plan from "smart paper" into "shipped product with numbers."

---

## 0.9 The Autonomous Principal Agent thesis (v5 — core)

This is the load-bearing reframe of v5. Read this before any phase section.

### The product

cosmos-lab is **one autonomous principal agent** that does long-horizon ML lifecycle work the way a real principal engineer does — not a coordination layer over six narrow specialists.

```
Real principal engineer's workday:
  Vague goal → reason for hours → decompose into experiments → write code →
  run experiments → observe surprising result → replan → iterate → ship

cosmos-lab PrincipalAgent's workday:
  Same loop. Long-horizon (multi-day). Real GPU. Real codebases. Real surprises.
```

The 6 v4 "agents" (Data, Eval, Train, Optimize, Video, Code) are now **6 capability domains** the same principal agent demonstrates over P3-P9. Like one senior ML engineer who does data work Monday, training Tuesday, optimization Wednesday — not six different people.

### Three pillars (each "exceptional," not "adequate")

**Pillar 1 — Long-horizon autonomous reasoning**
- Multi-step planning (decompose goal → experiments)
- Execution (real code, real runs, real surprises)
- Replanning on surprise (sentinel trips → agent gets feedback → adapts)
- Persistence across sessions (resume after compute interruption, after sleep, after week)

**Pillar 2 — Exceptional context harness (agent-managed)**
- 3-tier hierarchical memory: working / episodic / semantic
- Self-edited current state — agent updates its own `docs/CURRENT.md` equivalent
- Self-written verifiers — agent generates pass/fail scripts per task it takes on
- Trajectory store as replay buffer — past failures inform future planning
- Pointer index agent maintains as it learns codebase shape

**Pillar 3 — Governance as enabler (not fence)**
- Sentinels are **tripwires**, not gates: trip → agent gets structured feedback → replans → tries different approach
- Capability scope **expands with track record**: agent earning sentinel-clean runs gets broader tool access
- Audit log = **replay buffer**, not just compliance artifact: agent reads its own history during planning
- GEPA = **agent self-improvement** with retroactive human review (signed promotion record), not pre-approval gate

The inversion vs v4: governance compounds autonomy, doesn't constrain it. Like guardrails on a highway — they enable speed, not prevent it.

### What makes this an "exceptional" agent (not just "another agent")

In 2026 the agent landscape has Devin, Operator, Cursor Composer, Claude Code, Codex CLI, Aider. **Why is cosmos-lab different / better for ML lifecycle work?**

1. **ML-lifecycle-native tools**: not just coding tools — full ML stack (cosmos-curate stages, NeMo-RL, SkyPilot, NIM endpoints, W&B, Inspect AI, profilers). Same agent uses all.
2. **Long-horizon by construction**: most 2026 agents are session-bounded (hour). PrincipalAgent persists across sessions via OTel trajectory store + episodic memory. Resume mid-experiment after 24h.
3. **Sentinel-gated replanning**: when agent does something wrong, it doesn't fail loudly — it gets structured feedback (which sentinel tripped, why, what evidence) and replans. This is the differentiator over Devin (silent failure = wasted hours).
4. **Earned-trust capability expansion**: new agent starts with narrow capabilities; with N sentinel-clean runs, capability scope auto-expands. Devin doesn't have this.
5. **`nat`-runnable + OTel-GenAI-native**: drops into Cosmos team's stack with one command.

### The demo (this is the product)

A Cosmos hiring manager opens cosmos-lab, says:

> *"Take this Cosmos Reason 2 evaluation task. Improve pass-rate by ≥3pp. You have one week and $400 of GPU budget. Show me everything you did."*

Agent runs unattended for ~5 days. At end:
- Submits result with measured numbers (pass-rate before/after, p-value, sentinel agreement)
- Provides full Inspect View replay log
- Provides full OTel trajectory (every tool call, every reasoning step)
- Provides W&B sweep showing experiments tried (winners and losers)
- Provides 1-page report explaining hypothesis, what worked, what didn't, what to try next
- Cost report: $383 / $400 budget; 47 sentinel trips → agent replanned 47 times, all replans cleanly logged

That's **AI helping build AI** in production. Not orchestrating. Not gating. *Doing the work.* This is what cosmos-lab v5 ships.

### What we're NOT pretending to do (anti-hype, carried from v4)

- ❌ Not pretending the agent invents novel architectures (it composes known patterns intelligently)
- ❌ Not pretending zero human oversight (sentinel trips visible, weekly review by human, signed promotions)
- ❌ Not pretending it works on arbitrary research questions (scoped to ML lifecycle; cosmos vertical first)
- ❌ Not pretending online self-improvement (offline GEPA between sessions, with human review)
- ❌ Not pretending capability progression is unbounded (capability scope expansion is policy-bounded, not unbounded)

### Why this matches NVIDIA Cosmos JD better than v4

| JD literal text | v4 delivery | v5 delivery |
|---|---|---|
| *"strong agency in LLM-based systems"* | Library wrappers around other agents' agency | One agent with deep agency |
| *"code agents doing real work"* | CodeAgent on 10-bug fixture (1 week, demo-grade) | PrincipalAgent does coding as one of its capabilities, on real OSS issues + real ML codebases |
| *"AI helps build them"* | AI gates other things that build | AI does the building, observed + audited |
| *"long-horizon multi-step workflows"* | 6 agents in pipeline (orchestration) | One agent reasoning multi-day across phases |
| *"automation over data and experiments"* | Data + experiment as separate pipeline phases | Same agent runs data prep AND experiments AND eval cohesively |

v5 doesn't abandon v4's governance work. **It reframes that work as the substrate making autonomy safe** — sentinels as tripwires, identity for capability scoping, GEPA for self-improvement, OTel for replay/memory.

---

## 1. Phase table (~22.5 weeks — v5: PrincipalAgent capability progression)

> **v5 framing**: phases are no longer "ship N agents." They're **capability progression milestones for the PrincipalAgent**. Each phase adds a skill the same agent uses. Schedule unchanged from v4 (~22.5 weeks); depth shifts from breadth-across-agents to depth-per-capability. P0/P0.5/P1/P4b are *substrate* phases (identity, library, trajectory, governance). P3-P9 are *capability* phases. P10 is *production deployment* of the now-fully-capable PrincipalAgent.

> **v4 schedule rationale**: v3.2 trimmed to 20 weeks by inheriting nat plumbing. v4 adds **P5.5 PyTorch Depth (1w)** + **P10 expansion (1w)** to close production-grade gaps (§0.8). Net: 20 → ~22.5 weeks; still inside original 24-week budget. Banked ~1.5 weeks remain as risk buffer.
>
> **v3 split rationale (carried)**: P4 split into **P4a EvalAgent (1w)** + **P4b Identity v2 (2w)** because Identity v2 alone is 3-4w of work; honest > clean.

| New | Wks | Phase | Type | What PrincipalAgent gains here (v5) | Real GPU? |
|---|---|---|---|---|---|
| P0 | 1 | Foundation + identity (AuthZ MVP) | substrate | Identity primitive used for capability scoping; AuditLog used as agent's replay buffer *(shipped)* | no |
| **P0.5** | **0.6** *(4 days)* | **Library restructure + harness adapters** | substrate | `cosmos_lab/` package importable; PrincipalAgent will live in `cosmos_lab.principal/` (P3); `nat` adapter (D3) for Cosmos-stack deployment | no |
| P1 | 2 | TrajectorySink + Agent-as-Judge + OTel-GenAI + Inspect AI + Sentinel taxonomy | substrate | OTel trajectory store = PrincipalAgent's long-horizon memory substrate; sentinels = tripwires for replanning loop; Inspect AI = eval + capability validation | no |
| P2 | **1** | Cosmos provider scaffolding | substrate | NIMProvider + Cosmos toolset added to nat tool registry — PrincipalAgent will use these as tools in P3+ | no |
| P3 | 2 | **PrincipalAgent v0 + Data curation capability** | **CAPABILITY 1** | First version of PrincipalAgent (P5.2 architecture) ships with planner + executor + working memory + REPLAN loop. Demonstrates capability on real cosmos-curate data work (10-100 hours real video). | yes (cosmos-curate Ray cluster) |
| **P4a** | **1** | **Eval-design capability + EvalAgent platform UI** | **CAPABILITY 2** | PrincipalAgent gains eval-design capability: multi-judge + bootstrap CIs + reward-hack sentinels + PR-gating. Inspect View embed for human review. | no |
| **P4b** | 2 | **Identity v2 + capability-expansion mechanism** | substrate | MCP OAuth 2.1 + RFC 8707 + RFC 8693; **PrincipalAgent earned-trust capability expansion logic** lands here (initial scope → expanded after K sentinel-clean runs). Hash-chained signed audit log records expansion events. | no |
| P5 | **1.5** | **Training-orchestration capability** | **CAPABILITY 3** | PrincipalAgent gains training capability: Centaur HPO, ComputeBackend over SkyPilot+NeMo-Run+HF Jobs, NeMo-RL wrapping. **First real GPU sweep run autonomously by PrincipalAgent.** | **YES** (Inv 9: ≥1 real sweep) |
| **P5.5** | **1** | **PyTorch Depth artifact (PrincipalAgent demo)** | substrate + capability proof | PrincipalAgent autonomously delivers one PyTorch artifact: custom autograd op OR torch.compile pattern with profiler-driven kernel selection, ≥10% wall-clock improvement on real workload. Demonstrates "deep PyTorch familiarity" JD bullet via *agent action*, not human action. | **YES** (Inv 9) |
| P6 | **2** | **Optimization capability** | **CAPABILITY 4** | PrincipalAgent gains optimization capability: profiling + training_opt + inference_opt unified. **≥1.5× speedup on 4 real workloads autonomously selected and measured.** Sandbox 2-tier (E2B + Daytona). | **YES** (Inv 9) |
| P7 | 2 | **Memory & compression — agent's own 3-tier memory** | substrate | The 3-tier hierarchical memory (working/episodic/semantic) PrincipalAgent uses, shipped here. `MEMORY.md` pointer index + Anthropic `memory_*` tool API + Letta-compatible archival. **Agent self-edits its own memory.** | no |
| P8 | **2** | **Self-improvement (GEPA) — agent learning from own trajectory** | substrate | DSPy 3.x + `dspy.GEPA` offline pass; PrincipalAgent reads its own past trajectories from episodic memory, mines failures, proposes prompt/tool-description revisions, A/B tests, ratchets only on lower-CI improvement. **Retroactive human review + signed promotion record.** | no |
| P9a | 1 | **Multimodal pipeline capability** | **CAPABILITY 5** | PrincipalAgent runs end-to-end multimodal pipeline as ONE long-horizon task: data → train → eval → optimize on Cosmos Predict 2.5 + π₀.₅. Real Cosmos NIM endpoint hit. Single agent, single trajectory, < 8h wall-clock. | **YES** (Inv 9: real Cosmos NIM ≥1×) |
| P9b | 1 | **Code work capability (with expanded scope)** | **CAPABILITY 6** | PrincipalAgent in earned-expanded capability scope (per P4b mechanism, after P3-P9a sentinel-clean runs): writes real OSS bug fixes on real GitHub issues. Stretch: ≥1 real PR opened with reviewer engagement. **No closed fixture.** | no (E2B sandbox, CPU) |
| **P10** | **2** | **Production deployment + OSS PR + demo (EXPANDED IN v4)** | substrate + ship | (a) Deploy PrincipalAgent on HF Spaces or Modal endpoint, gather **≥100 real user sessions** over 1-week window — agent serves real ML tasks for real users; (b) **Open ≥1 upstream PR** to nvidia-nat or Inspect AI with sentinel/replanning pattern (review-ready, not draft); (c) `pip install cosmos-lab[all]`; (d) `nat run cosmos-lab.yaml` reference workflow; (e) KMS migration, NemoClaw/OpenShell stretch; (f) 5-min demo video showing PrincipalAgent solving a Cosmos task end-to-end | yes (production endpoint) |
| **Total** | **~22.5** | | **1 PrincipalAgent + 6 capabilities + library** | | **5 phases real GPU** |

---

## 1.5 Reuse map — existing assets we wrap, not rebuild

> **v3.2 reframe**: ml-intern moves from "platform we extend" to "**v1 reference harness via `cosmos_lab.harness.ml_intern` adapter**." The library does not depend on ml-intern at runtime; the adapter ports cosmos-lab into ml-intern's session model for users who want the HF stack + web UI. Primary harness from P1 onward is `nvidia-nat`.

A senior-eng audit of both the upstream `ml-intern` codebase and external 2026 SOTA components surfaced multiple components that materially overlap with planned phases. Default posture: **wrap behind cosmos-lab interfaces; rebuild only the genuinely new abstraction**.

| Existing upstream (ml-intern) asset | What it already does | Where used in cosmos-lab |
|---|---|---|
| `agent/core/session_uploader.py` | Detached subprocess uploader; ships every session to a configurable HF dataset | `cosmos_lab.trajectory.HFDatasetSink` (opt-in, P8 flywheel) |
| `agent/core/session_persistence.py` | Mongo-backed durable session store, gated on `MONGODB_URI` | Used directly via ml-intern adapter; no separate cosmos-lab Mongo sink (MongoSink cut in v3.1) |
| `agent/core/telemetry.py` + `HeartbeatSaver` | Mid-turn save/upload every N seconds | `cosmos_lab.harness.ml_intern` adapter hooks into these for mid-run flushes |
| `agent/core/approval_policy.py` (newly merged) | Approval policy + YOLO budget over tool calls | `cosmos_lab.identity.CapabilityScopedRouter` composes with this when running under ml-intern adapter |
| `agent/core/cost_estimation.py` (newly merged) | Per-call cost tracking | `cosmos_lab.eval` consumes for cost column in leaderboard |
| `agent/sft/tagger.py` | Tags trajectory events for SFT extraction | `cosmos_lab.governance.failure_mining` (P8) consumes via ml-intern adapter |
| `backend/kpis_scheduler.py` | APScheduler hourly rollup → HF KPI dataset | Optional integration via ml-intern adapter; cosmos-lab leaderboard CLI is the primary surface |
| `backend/session_manager.py::EventBroadcaster` | Per-session SSE fan-out | Used by ml-intern web UI only — not part of cosmos-lab core |
| `agent/tools/jobs_tool.py` | HF Training Jobs wrapper | `cosmos_lab.compute.HFJobsBackend` wraps this when ml-intern adapter active |

**Net effect (v3.2)**: cosmos-lab core does not depend on ml-intern at runtime; ml-intern is exposed via the `ml_intern` adapter for users who want HF stack + web UI. Phase 0 work (identity, audit log, router) is preserved as library code, runnable under either adapter.

### External assets — `nvidia-nat` as primary harness (v3.2 reframe)

| External asset | What it already does | Where used in cosmos-lab |
|---|---|---|
| **NeMo Agent Toolkit (`nvidia-nat`, CLI `nat`)** v1.6.0 (2026-04-10) | YAML workflow config, plugin/decorator tool registry, OTel-native telemetry chain, `nat eval` artifact pipeline, profiler, MCP-compatible tool surface, multi-framework adapters (LangGraph/CrewAI/LlamaIndex) | **PRIMARY HARNESS** from P1 onward via `cosmos_lab.harness.nat`. Not "wrap" — cosmos-lab plugs *into* nat as plugins. nat handles workflow runtime, tool registry, OTel exporter chain; cosmos-lab adds governance (sentinels, identity, GEPA, quality budget). |
| **OpenTelemetry GenAI semconv** (`gen_ai.*`, experimental) | Cross-vendor span schema for LLM calls, tool calls, agents | P1 — primary trace schema; replaces "DuckDBSink as schema source-of-truth" |
| **Phoenix (Arize)** OSS | OTel-native trace UI, eval rigor (drift, embeddings) | P1 default backend |
| **Inspect AI** (UK AISI) | Task / Solver / Scorer primitives + Docker sandbox + log viewer; production eval standard | P1, P4 — primary eval harness; our seed tasks ship as Inspect tasks |
| **DSPy 3.x + `dspy.GEPA`** | GEPA reflective prompt optimization, production-traction (Databricks, VMware) | P8 — offline self-improvement loop |
| **NeMo Curator 26.04** + **cosmos-curate** | Ray-based GPU text/image/video/audio curation; cosmos-curate processed 20M video-hr in 14 days on Blackwell | P3 — DataAgent composes these stages |
| **NeMo-RL** (formerly Reinforcer) | Production post-training framework; Nemotron-3-Super trained on it (Mar 2026) | P5, P6 — wrap as a post-training backend |
| **SkyPilot v0.12 Job Groups** | Multi-cloud job orchestration, RL job groups, Slurm support | P5 — `ComputeBackend` impl alongside `HFJobsBackend` |
| **NVIDIA OpenShell + NemoClaw** | OS-level syscall interception, declarative allow-lists, designed for "always-on" agents on GPU | P6/P9 — GPU sandbox tier |
| **E2B (Firecracker)** | Hardware-isolated CPU sandbox; ~150ms cold start | P6 — CPU correctness sandbox tier |
| **MCP authorization spec (draft 2026)** + **WorkOS AuthKit / Auth0 MCP AS** | OAuth 2.1 + RFC 8707 + RFC 8693 | P4 — Identity v2 (do not roll our own OAuth) |
| **Anthropic memory tool API** (`memory_*`) + **Letta** | File-based memory storage; agent loop with subagents | P7 — storage layer for memory; do not rebuild MemGPT-style paging |

**v3 net effect**: roughly half of "platform code" becomes integration glue against well-known boundaries. Cosmos team reading the plan recognizes every interface — credibility through *fluency in their stack*, not novel reinvention.

### Companion specification documents (v5)

These live at repo root as deep references; agents load on-demand:

| Document | Scope | When to read |
|---|---|---|
| `EVAL_SPEC.md` | ML-output evaluation (perplexity, KL divergence, latency p99, GPU OOM) — model under test | Working on P5/P6 (training/optimization), or any task with model output as deliverable |
| `AGENTIC_EVAL_SPEC.md` | Agent-system evaluation (trajectory quality, plan quality, replan quality, capability boundary, reward-hacking, cross-agent comparison) — agent itself as artifact-under-eval per axiom A8 | Working on any P1+ phase that builds or evaluates the PrincipalAgent itself |
| `PLAN.md` | Original 16-week ML optimization plan (superseded by PLAN_V2 v5) | Historical reference for optimization-vertical depth |
| `SYSTEM.md` | Full architecture deep-dive (Vietnamese, 1167L) | Rare — only for upstream debugging |
| `RESEARCH_AHE_ANALYSIS.md` | AHE (Agentic Harness Engineering) research informing P8 GEPA decisions | Working on P8 self-improvement loop |

---

## 2. Phase 0 — Foundation + Identity Skeleton (Week 1)

### Why now
Agent identity is a stand-out JD bullet ("AuthN, AuthZ, IAM"). Cheap to skeleton on day 1 and would be expensive to retrofit. Also unblocks safer multi-agent work in P9.

### Day-by-day

| Day | Deliverable | Owned path | Acceptance |
|---|---|---|---|
| 1 | Sync upstream + lock baseline | — | 237 pass / 3 upstream-broken documented |
| 1 | `agent/optimization/__init__.py` + `config_ext.py` | `agent/optimization/` | `from agent.optimization import OptimizationConfig` works |
| 2 | `configs/optimization_agent_config.json` | `configs/` | Loadable via existing `load_config()` |
| 2-3 | `AgentIdentity` (frozen dataclass) | `agent/optimization/identity/identity.py` | Root identity, scoped identity, `can_call()` |
| 2-3 | `AuditLog` (append-only JSONL, thread-safe) | `agent/optimization/identity/audit.py` | Atomic writes, `read_all()` round-trip |
| 3 | `CapabilityScopedRouter` (composition over `ToolRouter`) | `agent/optimization/identity/router.py` | Filters tool specs, denies unauthorized calls, audits all paths |
| 4 | `tests/optimization/test_identity_scoping.py` (≥6 tests) | `tests/optimization/` | All pass; full suite no regression |
| 5 | Phase 0 retro + zero-diff verification | — | `git diff upstream/main --name-only` = owned only |

### Acceptance criteria

- [ ] `pytest tests/optimization/ -q` exits 0
- [ ] `pytest tests/unit/ -q` shows ≤ 3 failures (the 3 upstream-broken; no new)
- [ ] `git diff upstream/main --name-only` lists only owned paths
- [ ] `OptimizationConfig` round-trips through `Config.model_validate()`
- [ ] `CapabilityScopedRouter` denies + audits when capability not granted
- [ ] `CapabilityScopedRouter` allows + audits before/after when granted
- [ ] `AuditLog` JSONL is parseable and chronologically ordered

### Design decisions

- **Composition over inheritance** for `CapabilityScopedRouter`: `ToolRouter.__init__` instantiates the full builtin tool stack (sandbox tools require HF auth) — heavy and unsuitable for unit tests. Wrapping a base router (or a duck-typed mock) keeps tests fast and decoupled.
- **JSONL audit**: human-readable, append-only, easy to ship to S3 / Phoenix later. No DB dependency in Phase 0.
- **`"*"` wildcard capability**: the only special-cased capability. No glob/prefix matching in Phase 0 — keep semantics trivially provable.
- **Composes with upstream `approval_policy.py`**: capability check is the *coarse, before-the-fact* allowlist; approval policy is the *fine, per-call, budget-aware* gate. Order: capability allow → approval policy → execute. Integration ticket lives in P1 D1; Phase 0 ships standalone enforcement.

### Scope clarification (AuthN vs AuthZ)

Phase 0 ships **AuthZ + audit**, not AuthN/IAM. `AgentIdentity` is unsigned — any caller can construct one. This is intentional for Phase 0: the goal is to prove the *enforcement and audit surface* against a known principal. AuthN (signed identity tokens, IAM provider integration) layers on later, once we have a real principal source (HF OAuth in `backend/`, sub-agent spawning in P9). Document this in module docstrings so a reader doesn't read more into it than is there.

### Risks & mitigations

- **Risk**: Upstream `ToolRouter.call_tool` signature changes in a future merge. **Mitigation**: composition + duck typing limits the blast radius to one method.
- **Risk**: AuditLog write contention if used from many threads. **Mitigation**: `threading.Lock` on write; if hot, switch to a queue+writer thread in P1.

---

## 2.5 Phase 0.5 — Library restructure + harness adapters (4 days, NEW IN v3.2)

### Why this exists
v3.2 reframes cosmos-lab as a Python library (`pip install cosmos-lab`), not a fork of ml-intern (§0.4 explains why). P0.5 is the architectural refactor that makes this real. **No semantic change to Phase 0 code** — only packaging, imports, and adapter wiring.

### Day-by-day

| Day | Deliverable | Acceptance |
|---|---|---|
| D1 | Restructure `agent/optimization/` into the §0.4 package layout: `cosmos_lab/{identity,trajectory,eval,governance,memory,providers,compute,sandbox,harness}/`. Both old (`agent.optimization.*`) and new (`cosmos_lab.*`) import paths work via `__init__.py` re-exports. Update `pyproject.toml` with `cosmos-lab` package + `[nat]`, `[ml-intern]`, `[all]` extras. | `from cosmos_lab.identity import AgentIdentity` works; existing 16 Phase 0 tests still pass without modification (only `import` lines change) |
| D2 | Write `cosmos_lab/harness/ml_intern.py` adapter — refactor existing `CapabilityScopedRouter` integration into adapter pattern. Provides: tool registration shim, span correlation hook, lifecycle wiring (per §0.4 adapter contract). | One smoke test: `cosmos_lab.harness.ml_intern.install(session, identity)` makes a 3-step trivial agent run with capability denial + OTel span emission + sentinel evaluation. ≤200 LOC. |
| D3 | Write `cosmos_lab/harness/nat.py` adapter — `install_into_nat(builder, identity)` API per §0.4 sketch. Registers `OTelGenAIEmitter` as nat exporter, wraps nat tool router with `CapabilityScopedRouter`, hooks lifecycle. | Same smoke test contract as ml-intern adapter, but running inside `nat run cosmos_lab_smoke.yaml`. ≤200 LOC. Both adapters parameterized over the same `AdapterContract` Protocol so future adapters (Claude SDK, OpenAI Agents) follow the pattern mechanically. |
| D4 | Dual-adapter test matrix: every Phase 0 test parameterized via `@pytest.mark.parametrize("harness", ["nat", "ml_intern"])`. Both must pass identically. Document the contract in `cosmos_lab/harness/CONTRACT.md`. | `pytest tests/optimization/ -q` runs 16 tests × 2 harnesses = 32 tests; all pass. CI matrix added. |

### Acceptance criteria

- [ ] `pip install -e .[nat]` and `pip install -e .[ml-intern]` both work; importable as `cosmos_lab`
- [ ] `tests/optimization/` runs 32 tests (16 × 2 adapters), all green
- [ ] `nat run examples/cosmos_lab_smoke.yaml` exits 0 and produces an OTel trace with capability denial event recorded
- [ ] `cosmos_lab/harness/CONTRACT.md` documents the 3-method adapter contract (tool registration, span correlation, lifecycle wiring)
- [ ] No existing Phase 0 test deleted; only `import` lines updated
- [ ] Zero-diff invariant still holds: `git diff upstream/main --name-only` returns only owned paths

### Why 4 days, not 1 week

Phase 0 code is ~600 LOC across `identity/`, no semantic change required. Adapter pattern is well-understood (entry-point pattern in `pyproject.toml`, ~150-200 LOC each). Dual-adapter test parametrization is `@pytest.mark.parametrize` — trivial. The risk is mostly in nat-side wiring (we haven't run nat locally yet); D3 reserves ~6h for that learning curve.

### What this unlocks

From P1 onward, every owned cosmos-lab module is written **once** against the framework-agnostic interfaces. The two adapters translate to/from nat and ml-intern. When OpenAI Agents SDK 2.0 ships, adding a third adapter is ~200 LOC, not a refactor.

This is the architectural difference between "we built on HF" and "we built a library that runs anywhere — and our default is Cosmos-team's stack." The four days pay for themselves in pitch credibility alone.

---

## 3. Phase 1 — TrajectorySink + OTel-GenAI + Inspect AI Judging (Weeks 2–3) — v3

### Reframe (vs v2)

v2 said "TrajectorySink with DuckDB primary." v3 says: **trace schema is OpenTelemetry GenAI semantic conventions**, not anything we invent — and the eval rig is **Inspect AI** (UK AISI Task/Solver/Scorer), not a custom `TaskRunner`. Why: (1) the field has converged on OTel `gen_ai.*` spans (see DataDog/Grafana/Phoenix native support, Q1 2026), and a custom DuckDB schema would be a one-off that loses portability and Cosmos-team recognition. (2) Inspect AI is the production eval standard (METR uses it), with Docker sandbox, log viewer, and bootstrap CIs built in. Building a custom harness is reinventing the field's primitive.

DuckDB stays — but as a *query/analytics layer over OTel spans*, not as the schema source-of-truth.

### Goals (v3)
1. Define a `TrajectorySink` interface that emits **OTel GenAI spans** + persists derived rows for fast query.
2. Wire seed tasks as **Inspect AI Tasks**; use Inspect's Solver/Scorer + Docker sandbox; add cosmos-lab-specific scorers.
3. Make every judged metric ship with **bootstrap CI + an anti-reward-hacking sentinel** (paired structural verifier).
4. Integrate `CapabilityScopedRouter` with upstream `approval_policy.py` — capability allow → policy gate → execute (ordering test mandatory).

### Deliverables (v3)

| Module | Path | Notes / reuse |
|---|---|---|
| `TrajectorySink` (Protocol) | `agent/optimization/trajectory/sink.py` | `record_span(GenAISpan)`, `record_run(RunRecord)`, `query(...)` — `GenAISpan` follows OTel `gen_ai.*` semconv |
| `OTelGenAIEmitter` | `agent/optimization/trajectory/otel_emitter.py` | **NEW** — emits `gen_ai.*` spans via OTel SDK; backend = Phoenix by default, swap-in via `OTEL_EXPORTER_OTLP_ENDPOINT`. **Default = sole sink in P1.** |
| `DuckDBSink` | `agent/optimization/trajectory/duckdb_sink.py` | **Opt-in** local query layer over OTel-shaped rows (`runs`, `spans`, `judge_scores`); enabled when analytics needed (P4a leaderboard) |
| `HFDatasetSink` | `agent/optimization/trajectory/hf_sink.py` | **Opt-in** thin adapter over existing `session_uploader.py`; enabled for AHE/SFT flywheel (P8) |
| `MultiSink` | same | Fan-out wrapper. **P1 default = `[OTelGenAIEmitter]` only**; chain grows as later phases need them. v3.1 cut: ship one sink working before three. |
| ~~`MongoSink`~~ | — | **CUT in v3.1**: `session_persistence.py` already provides Mongo durability via upstream's existing path; a separate `MongoSink` was feature creep with no consumer in any phase. |
| `TracedSession(Session)` | `agent/optimization/trajectory/session.py` | Subclass; routes `logged_events` to active sink chain. Reuses `HeartbeatSaver` for mid-run flushes |
| Capability/Policy integration | `agent/optimization/identity/router.py` (extend) | Compose with `approval_policy.py`; capability check first, then policy, then execute |
| `LLMJudge` | `agent/optimization/eval/judge.py` | Single-pass baseline; uses `cost_estimation.py` to log judge cost |
| `MultiJudge` | `agent/optimization/eval/multi_judge.py` | **N-judge variance reduction** (no debate dynamics — see [`arxiv:2508.17536`](https://arxiv.org/abs/2508.17536)). Reports bootstrap CI on pass-rate. Default N=3 (Sonnet 4.6 ×3); tie-break with Opus 4.7 only if CI bound straddles threshold |
| `RewardHackSentinel` | `agent/optimization/eval/sentinel.py` | **NEW** — paired structural verifier. Each task pairs a deterministic check (file written? speedup measured? parity asserted?) with the judge. Disagreement → flag run for review, do not silently accept |
| `ToolAugmentedJudge` | `agent/optimization/eval/tool_judge.py` | Judge can call read-only tools; identity = `judge-readonly` capability set |
| Inspect AI bridge | `agent/optimization/eval/inspect_bridge.py` | **NEW** — exposes our scorers/judges as Inspect AI `Scorer`s; lets a cosmos-lab task run inside `inspect eval` and produce Inspect View logs |
| Seed tasks (Inspect format) | `tasks/seed/*.py` | 5 tasks as Inspect `@task`s: dataset inspect, code task, ML debug, paper summary, profiling. Each ships with one judge scorer + one structural sentinel |
| `evaluate` CLI | `agent/optimization/cli/evaluate.py` | Wraps `inspect eval` with cosmos-lab defaults: `cosmos-lab evaluate --suite seed --judge multi --sinks otel,duckdb,hf` |

### Acceptance (v3.1)
- 5 Inspect tasks × 3 runs = 15 trajectories: spans land in Phoenix via OTel; round-trip p99 < 500ms (P1 numerical target)
- `MultiJudge` reports bootstrap 95% CI on pass-rate; CI width ≤ 8pp at N=15 runs (P1 numerical target)
- For every judged task, `RewardHackSentinel` runs and flags any judge/structural disagreement; **sentinel/judge agreement ≥ 98% on green seed runs** (P1 numerical target — any disagreement is a bug, not noise)
- `evaluate` CLI prints leaderboard with pass-rate ± CI, p50/p99 latency, **cost (via upstream `cost_estimation`) as a first-class column**, sentinel disagreement count
- A capability-denied call is blocked *before* approval policy is consulted (ordering test in `tests/optimization/test_router_policy_integration.py`)
- One golden Phoenix screenshot of an `agent` span tree committed under `docs/` so a Cosmos reviewer can see OTel wiring at a glance
- `DuckDBSink` and `HFDatasetSink` exist as opt-in modules with passing unit tests but are NOT in the default `MultiSink` chain (v3.1 — ship one sink working before three)

### What we explicitly do NOT ship in P1 (deferred or dropped)
- ❌ `DebatingJudgePanel` — dropped per `arxiv:2508.17536`. Multi-judge variance reduction stays; debate dynamics removed.
- ⏸ Custom leaderboard UI — uses Inspect View in P1; cosmos-lab leaderboard extension lands in P4.
- ⏸ MCP-OAuth identity — P0's `AgentIdentity` is the AuthZ MVP; OAuth 2.1 + RFC 8707/8693 + signed audit log lands in P4 (Identity v2). Document the gap explicitly so no one reads more into P1's audit log than is there.

### References (v3)
- [OpenTelemetry GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) (experimental; opt-in stability flag)
- [Inspect AI (UK AISI)](https://inspect.aisi.org.uk/) — production eval standard
- [NeMo Agent Toolkit `nat eval`](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/observe/observe.html) — schema we mirror for `nat`-runnable artifacts
- [Survey on Agent-as-a-Judge (2601.05111)](https://arxiv.org/pdf/2601.05111)
- [Debate or Vote — multi-agent debate refutation (`arxiv:2508.17536`)](https://arxiv.org/abs/2508.17536)
- [UC Berkeley — How We Broke Top AI Agent Benchmarks (2026)](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/)
- [METR — RE-Bench, reward-hacking measurements](https://metr.org/AI_R_D_Evaluation_Report.pdf)

---

## 3.1 Sentinel taxonomy (v3.1) — the platform's spine made concrete

The single highest-leverage thing cosmos-lab does is **block judge-only metrics from reaching gates**. To enforce that across phases, we need a *taxonomy* of structural verifiers, not a vague "structural check." Every Inspect task contributes one judge `Scorer` and one sentinel from this taxonomy. Sentinel/judge disagreement → run flagged for review, never silently accepted.

| Sentinel type | What it checks | Failure mode it blocks | Example (P1 seed task) |
|---|---|---|---|
| **`DeterministicStateCheck`** | A boolean function on post-run filesystem / DB / object state | "Judge said it worked, but nothing actually changed" | *dataset-inspect task*: `assert (workdir / "schema.json").exists() and parse_json(...) has expected keys` |
| **`OutputFormatCheck`** | Strict schema/regex/parse on the agent's final tool output | "Judge said the answer was good, but it's not parseable" | *paper-summary task*: response must be parseable JSON with `{title, key_findings: list[str], limitations: str}` |
| **`SideEffectCheck`** | A boolean over emitted OTel spans (specific tool was called, in expected order, with expected args) | "Judge said the agent reasoned through it, but the agent never actually ran the profiler" | *profiling task*: assert one `gen_ai.tool.call` with `tool.name = "torch_profiler"` and non-empty result span |
| **`NoOpCheck`** | The agent did *something* — at least N tool calls, modified at least one file, latency above lower bound | "Judge said pass on a no-op trajectory" — the canonical reward-hack pattern Berkeley audited | every task: assert `tool_call_count >= 1 AND wall_clock >= 100ms AND not all(span.result == "")` |

**Composition rule**: a task's sentinel = `DeterministicStateCheck OR OutputFormatCheck OR SideEffectCheck`, **always AND-ed with `NoOpCheck`**. NoOpCheck is mandatory on every task — it catches the cheapest reward-hack class for free.

**Why these four**: directly map to the failure modes UC Berkeley demonstrated (no work done, fake output, judge-prompt injection, monkey-patching the grader). Berkeley's "near-perfect scores with zero LLM calls" is *exactly* what `NoOpCheck` blocks.

**Owned path**: `agent/optimization/eval/sentinels/{deterministic.py,output_format.py,side_effect.py,no_op.py}`. Each ships with ≥3 unit tests covering green path + intended failure detection + edge case (empty workdir, malformed JSON, missing span).

**Task-author contract**: every Inspect `@task` we ship registers exactly one judge Scorer and exactly one composed sentinel via `@sentinel(...)`. Tasks without both fail CI before merge.

---

## 3.2 PrincipalAgent architecture (v5 — the autonomous heart)

The architectural answer to §0.9's autonomous-principal-agent thesis. This is the technical specification for the agent itself — the loop, memory tiers, planning module, replanning logic, capability expansion mechanism.

### 3.2.1 Substrate choice — ml-intern's `agent_loop.py`, not from scratch

**Decision**: PrincipalAgent runs **inside ml-intern's existing `agent_loop.py`** (1626 lines, production-debugged, 16 built-in tools, MCP integration). cosmos-lab adds capabilities ON TOP, never replaces the loop.

**Rationale**:
- ml-intern's loop already handles: tool calling, retries, context window management, doom-loop detection, session persistence (Mongo + HF), heartbeat saving, approval policy, cost estimation
- Re-implementing this is 6+ weeks of debugged code we'd recreate. Anti-pattern #4 (workflow): "Building a pipeline that should have been one model call" — generalize: building a substrate that should have been an existing one
- nvidia-nat as the *deployment harness* (P0.5 D3 adapter) — but the agent loop INSIDE nat is ml-intern's, wrapped through `cosmos_lab.harness.nat`

**Owned path**: `cosmos_lab/principal/` — agent definition, planner, memory, capability expansion logic. Substrate stays at `agent/core/agent_loop.py` (zero-diff).

### 3.2.2 The autonomous loop — long-horizon, multi-day

```
┌────────────────────────────────────────────────────────────────┐
│  PrincipalAgent.run(goal, budget)                              │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ 1. PLAN PHASE (LLM reasoning, no tool calls yet)      │     │
│  │    - Read goal + episodic memory of similar past work │     │
│  │    - Decompose into N experiment milestones           │     │
│  │    - Write plan to memory/working/plan.md             │     │
│  │    - Generate verifier scripts per milestone          │     │
│  └──────────────────────────────────────────────────────┘     │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ 2. EXECUTE PHASE (one milestone at a time)            │     │
│  │    - Pull next milestone from plan                    │     │
│  │    - Hand to ml-intern agent_loop with scoped tools  │     │
│  │    - Loop runs full ReAct: read → tool → observe → … │     │
│  │    - Every tool call audited, OTel span emitted       │     │
│  └──────────────────────────────────────────────────────┘     │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ 3. VERIFY PHASE (sentinel check after each milestone) │     │
│  │    - Run milestone's auto-generated verifier          │     │
│  │    - Run paired sentinels (judge + structural)       │     │
│  │    - GREEN → mark milestone done, persist to memory  │     │
│  │    - RED → ───────────────────────────────────┐      │     │
│  └────────────────────────────────────────────────┼─────┘     │
│                       │                            │            │
│                       ▼                            ▼            │
│              ┌───────────────────┐      ┌─────────────────┐    │
│              │ 4a. ALL DONE       │      │ 4b. REPLAN     │    │
│              │     → ship report  │      │  - Read failure │    │
│              └───────────────────┘      │  - Update plan  │    │
│                                         │  - Loop to (2) │    │
│                                         └─────────────────┘    │
│                                                                 │
│  Persisting throughout: working memory + episodic + semantic   │
└────────────────────────────────────────────────────────────────┘
```

**Long-horizon** = this loop runs across sessions. Mid-execute when compute quota expires? Trajectory persists. Resume the next day reads `memory/working/plan.md` + last OTel span ID, picks up at the next milestone.

### 3.2.3 Three memory tiers (agent-managed)

| Tier | Path | Lifetime | What lives here |
|---|---|---|---|
| **Working** | `memory/working/` | Current task | Plan, current milestone, scratchpad, in-progress reasoning |
| **Episodic** | `memory/episodic/` | All past tasks | Trajectories of completed work, indexed by goal/domain. "Last time I did Cosmos eval, I tried X and it failed because Y" |
| **Semantic** | `memory/semantic/` | Distilled facts | Curated lessons (e.g., "torch.compile mode='reduce-overhead' wins on attention but loses on data loaders") — promoted from episodic via GEPA |

Agent reads from semantic + relevant episodic during PLAN phase. Writes to working during EXECUTE. After milestone done, distills surprising-evidence into episodic. Weekly GEPA pass distills episodic → semantic.

### 3.2.4 Replanning logic (sentinels as enabler, not gate)

When a sentinel trips:
1. Sentinel emits **structured failure record**: `{sentinel_type, expected, actual, evidence_path}`
2. Agent reads record + relevant context (last 3 milestones, related episodic memory)
3. Agent generates **replan**: hypothesis about what went wrong, alternative approach to try
4. Updated plan written to working memory
5. Loop continues from updated milestone

**Critical**: sentinel trip is NOT failure. Sentinel trip is **information**. Failure is when agent can't replan after N tries (default N=3).

This is the difference vs Devin (silent failure → wasted hours): **sentinels make failure loud and actionable**.

### 3.2.5 Capability expansion (earned trust)

```
PrincipalAgent.spawn(parent_token):
  initial_caps = {read_file, list_directory, search_huggingface}  # narrow
  
After K consecutive sentinel-clean milestones:
  PrincipalAgent.expand_capabilities(K):
    K >=  5:  + {write_file, run_tests}              # expanded
    K >= 20:  + {git_diff, git_commit_local}         # broader
    K >= 50:  + {run_real_gpu_job, deploy_to_modal}  # production
  
  Each expansion = RFC 8693 token exchange:
    parent_token + new_scope_subset → new_token
    Logged to hash-chained signed audit (P4b)
    Visible in OTel: gen_ai.agent.capability_expansion event
```

Capability expansion is **policy-bounded**: expansion table is config, not unbounded. Human review of expansion happens weekly (read the audit log).

### 3.2.6 Concrete demo per §0.9

> Cosmos hiring manager: *"Take this Cosmos Reason 2 task. Improve pass-rate ≥3pp. One week, $400 budget."*

PrincipalAgent.run() executes:
- **Hour 0-2 PLAN**: read task, read episodic ("similar Cosmos Reason 2 task done 2 weeks ago, found mid-layer attention was bottleneck"), decompose into 8 milestones (data inspection → baseline → 3 experiments → eval → optimization → final eval → report)
- **Day 1-2 EXECUTE milestones 1-4**: data work, baseline, experiment 1 (LR tuning, sentinel-clean), experiment 2 (data augmentation, sentinel TRIPS — eval drift > tolerance), REPLAN, experiment 2-revised
- **Day 3 EXECUTE milestones 5-6**: experiment 3 (LoRA on attention layers), eval shows +4.2pp pass-rate, sentinel-clean
- **Day 4 EXECUTE milestones 7-8**: optimization (torch.compile + selective layer pruning, +6% wall-clock), final eval (+4.1pp pass-rate held)
- **Day 5 SHIP**: report committed, OTel trajectory archived, W&B sweep linked, cost report ($383/$400)

That's the demo. **Real autonomous principal-engineer work**, observed + replannable + auditable.

### 3.2.7 What lives in `cosmos_lab/principal/`

```
cosmos_lab/principal/
├── __init__.py            # exports PrincipalAgent
├── agent.py               # PrincipalAgent class — the orchestrator
├── planner.py             # PLAN phase: goal → milestones + verifiers
├── executor.py            # EXECUTE phase: hands milestone to ml-intern agent_loop
├── verifier_gen.py        # auto-generates milestone verifiers from goal
├── replanner.py           # REPLAN phase: sentinel trip → new plan
├── memory/
│   ├── working.py         # in-task memory
│   ├── episodic.py        # cross-task memory (DuckDB-backed for query)
│   └── semantic.py        # distilled facts (file-based, GEPA-curated)
└── capability_expansion.py  # earned-trust capability scope expansion
```

P3-P9 phases each ADD a capability domain to PrincipalAgent (data/eval/train/optimize/multimodal/code) — they're not separate agents, they're skill modules the same agent uses.

---

## 3.3 Agentic eval architecture (v5 — pointer to AGENTIC_EVAL_SPEC.md)

The sentinel taxonomy (§3.1) is one piece of agentic eval. The full architecture lives in **`AGENTIC_EVAL_SPEC.md`** — companion to `EVAL_SPEC.md` (which covers ML-output eval; this companion covers agent-system eval per axiom A8 *"the agent is itself an artifact-under-eval"*).

### Why a separate spec doc

EVAL_SPEC.md evaluates models. AGENTIC_EVAL_SPEC.md evaluates agents. Three distinctions (per AGENTIC_EVAL_SPEC §1):
1. **Trajectory is the deliverable, not just output** — two agents producing identical correct outputs can have radically different trajectory quality (one took 47 tool calls + 12 replans, the other took 3 calls correct first time)
2. **Agent is itself artifact-under-eval (A8)** — strong MMLU + strong HumanEval ≠ strong agentic tool-use; need separate eval surface for agent decisions
3. **Long-horizon eval is non-fungible with short-horizon eval (NEW axiom A13)** — a 5-day task is not 120 1-hour tasks; cross-session memory, plan staleness, capability expansion mid-task are new failure modes

### What agentic eval architecture adds (over §3.1 sentinels alone)

**5-tier ladder** (transfers from EVAL_SPEC, specialized for agentic):
- T0 smoke / T1 calibrated quality / T2 long-horizon / T3 shadow / T4 canary

**6 agentic-specific surfaces** (NEW — don't exist in EVAL_SPEC):
- **S1 Trajectory eval** — tool-call efficiency, replan ratio, wasted-work, doom-loop frequency
- **S2 Plan-quality eval** — LLM-judge on PLAN-phase decomposition (gates EXECUTE per §3.2)
- **S3 Replan-quality eval** — sentinel trips → response quality (success rate, diversity, time-to-recovery)
- **S4 Capability boundary eval** — 50-task denied-tool probe suite (security-critical for capability expansion per AGENTIC_EVAL_SPEC axiom A12)
- **S5 Reward-hacking adversarial eval** — monthly red-team sprint (covers what UC Berkeley's 8/8-hackable-benchmarks crisis demands)
- **S6 Cross-agent comparison eval** — PrincipalAgent vs Devin vs Claude Code vs human, quarterly Pareto chart (the differentiator pitch)

**3 cross-cutting meta layers** (transfer from EVAL_SPEC):
- M1 Reproducibility envelope, M2 Eval-of-eval, M3 Cost telemetry

**3 input types** (per JD bullet 5):
- I1 Automated metrics, I2 Human feedback (5% sampling + weekly review), I3 Agent-driven analysis

### Operational cadence summary

| Cadence | What runs | Cost budget |
|---|---|---|
| Every commit | T0 + S1 sanity | <$0.10 |
| Every PR to main | T1 + S1 + S2 + S3 | $10-$50 |
| Nightly | T1 + T2 + S4 | $50-$200 |
| Weekly | T3 + S6 sample + I2 5% human review | $300-$700 |
| Monthly | S5 red-team sprint + M2 eval-of-eval | $500-$1000 |
| Per release | T4 canary + production monitoring | $300-$1000 + risk |
| Quarterly | S6 cross-agent full Pareto comparison | $500-$2000 |

### Integration with v5 phases

Per AGENTIC_EVAL_SPEC §10, this architecture integrates into v5 phases without adding a new phase:
- **P1** establishes T0/T1 + S1 + M1 + sentinel taxonomy
- **P4a** EvalAgent capability builds S2 + S3 + I2 + S5 monthly red-team kickoff
- **P4b** ships S4 capability boundary probe suite (security-critical, blocks identity v2 ship)
- **P5/P6** runs T2 long-horizon eval on real GPU sweeps (per Invariant 9)
- **P9b** ships first S6 cross-agent comparison (PrincipalAgent vs Claude Code on bug fixture)
- **P10** runs T4 canary + first quarterly S6 full comparison

**Net cost**: ~3-4 days additional spec/test work spread across phases.

### 10 numerical eval-system targets

Extends §0.7 (see "v5 eval-system additions" subtable). Examples:
- E1: sentinel suite FPR ≤ 5% on null fixtures
- E2: sentinel suite FNR ≤ 1% on planted regressions
- E5: replan success rate ≥ 70%
- E6: capability boundary 100% (0 unauthorized calls in 100 child runs)
- E8: PrincipalAgent on Pareto frontier of cost × quality vs comparison agents

Full target list in AGENTIC_EVAL_SPEC §9.

---

## 4. Phase 2 — Cosmos Vertical (Weeks 4–5) — v3.1 reframed

### Reframe (v3.1)
v2/v3 promised `VideoUnderstandingAgent` as the proof point — but real video understanding requires GPU access we don't have committed. v3.1 honestly scopes P2 as: **provider abstraction + Cosmos tool wrappers + mocked-endpoint validation**. The agent itself is deferred to P9 where it lives inside the e2e pipeline (real or simulated). This is honest scoping, not retreat — it removes a credibility risk (a "demo" that only runs against mocks looks worse than no demo).

### Goals (v3.1)
1. First-class Cosmos integration scaffolding (Reason 2 / Predict 2.5 / Transfer 2.5) via `NIMProvider`.
2. Tool wrappers callable from any cosmos-lab agent — no agent ships in P2 itself.
3. 5 cosmos eval tasks (Inspect format) plugged into P1 harness, validated against mocked NIM.

### Deliverables (v3.1)

| Module | Path | Notes |
|---|---|---|
| `NIMProvider` | `agent/optimization/providers/nim_provider.py` | OpenAI-compatible litellm custom provider pointed at NVIDIA NIM endpoints. **First non-HF provider** in the platform — establishes the abstraction other providers (Modal, Lambda, NGC) will follow. |
| `CosmosToolset` | `agent/optimization/cosmos/{reason,predict,transfer}.py` | Tool wrappers calling Cosmos Reason 2 / Predict 2.5 / Transfer 2.5 *via* `NIMProvider` (preferred) with HF-hosted fallback |
| Physical-AI eval pack (Inspect format) | `tasks/cosmos/*.py` | 5 Inspect `@task`s: object localization, motion prediction, scene QA, future-state gen, sim-to-real. Each carries one judge + one sentinel per §3.1 taxonomy. |
| NIM mock contract test | `tests/optimization/cosmos/test_nim_contract.py` | 100% of cosmos tasks parseable against mock without endpoint changes (P2 numerical target) |
| ~~`VideoUnderstandingAgent`~~ | — | **Deferred to P9 (e2e demo)**: shipping a real agent without GPU is demo-ware. The toolset is ready; P9 wires it. |

### Acceptance (v3.1)
- All 5 cosmos tasks load + execute against mocked NIM endpoint (no real-endpoint dependency in CI)
- `NIMProvider` registered with litellm; basic completion + tool-call paths covered by tests
- Real-endpoint smoke test documented as a 1-page README runbook (manual until GPU access secured)
- **No agent ships in P2** — this is honest scoping per v3.1 reframe

### References
- [Cosmos technical blog](https://developer.nvidia.com/blog/scale-synthetic-data-and-physical-ai-reasoning-with-nvidia-cosmos-world-foundation-models/)
- [Cosmos GitHub org](https://github.com/nvidia-cosmos)

---

## 5. Phases 3–10 — sketches (v3)

(Detailed when entered. High-altitude only here. v3 deltas marked with ⚡.)

### P3 DataAgent (Wks 6–7) ⚡ — v4 reframed for real multimodal data

**v4 reframe**: v3.2 said "compose curator stages" but didn't require any real data flow. v4 requires DataAgent to process **10-100 hours of real video sample** through cosmos-curate, ship dataset card with full lineage. This closes §0.8 production gate G4 (real multimodal data flow) and credibly demonstrates the JD's "multimodal ML pipelines spanning data processing" bullet at non-toy scale.

- ⚡ Compose **NeMo Curator 26.04** stages (text/image/video) + **cosmos-curate** (video-specific) — do NOT reimplement curation
- Synthetic data gen via Cosmos Predict 2.5 + Nemotron-style instruct→reward filter pipeline (Magpie-class for instruction data)
- Curation primitives via NeMo Curator: GPU-accelerated heuristics + classifiers (domain/quality/safety) + fuzzy/semantic dedup
- ⚡ **Real video sample (v4)**: pick a public dataset slice (e.g., Open-Sora samples, YouTube-CC0 batch, or Cosmos community sample) at **10-100 hour scale**. Run through cosmos-curate Ray pipeline on Modal/Lambda GPU (~$50 budget). Output a real dataset card with classifier verdicts + dedup statistics + provenance.
- Lineage tracking via **W&B Artifacts** (v3.1: pick one, not both — W&B aligns with the model registry story we'll need in P5/P6; DVC was redundant)
- DataAgent = pipeline DAG with LLM-in-the-loop nodes (judge filters, persona-rewriters, taxonomy samplers) + agent that proposes new gen prompts from eval-failure clusters. **Not** "agent autonomously curates a corpus" — that's hype in 2026.
- Output: dataset card committed to HF Hub with full lineage manifest; **100% of records traceable from raw video → curator stage → final row** (P3 numerical target)
- Acceptance:
  - `cosmos-lab data --task <spec>` produces a curated dataset card with provenance + a NeMo-Curator-compatible recipe file
  - W&B Artifacts lineage graph renders end-to-end
  - **Real video processed**: at least one cosmos-curate run on 10-100 hours of public video, results committed (or HF dataset link); cost report ≤$50

### P4a EvalAgent platform (Wks 8–9) ⚡ — v3.1 simplified
- ⚡ **v3.1 cut**: dropped the custom frontend overlay. Use **Inspect View embed** as the leaderboard UI — it's the 2026 standard reviewers expect to see, and §6.5 explicitly said "porting frontend has no Cosmos-pitch value." The custom UI was contradicting our own framing.
- ⚡ Regression PR gate via Inspect AI bootstrap CI — block if lower CI bound regresses by >δ. **PR-gate false-positive rate ≤ 5% on 10-PR replay set** (P4a numerical target); shadow-mode for first week before enforcing
- Cosmos-lab–specific leaderboard data exposed as a **single CLI command** (`cosmos-lab leaderboard --format markdown|json`) reading from `DuckDBSink` (now opt-in-enabled here for analytics) — no custom React work
- Human-feedback ingestion via existing Slack tool → Argilla-style labeling sink → re-feeds Inspect dataset
- Acceptance: PR opened with regression auto-blocks; `cosmos-lab leaderboard` renders pass-rate ± CI + sentinel disagreement count + cost; one human-labeled batch flows back into Inspect dataset cleanly; Inspect View link from PR comment

### P4b Identity v2 — v1 cut (Wks 10–11) ⚡ *(promoted to its own phase; v1 scope tightened per advisor)*

**Honest scope cut**: full Identity v2 (MCP OAuth client + 3 RFCs + AS integration + hash-chained signed log + KMS custody + tamper mutation test + sub-agent propagation + OTel wiring) is 3-4 weeks of work. P4b ships a **v1 cut in 2 weeks**; KMS key custody and the tamper mutation test are explicitly deferred to P10. Document the gap in §8 as a known v1 limitation.

**P4b v1 (in scope, 2w)**:
- ⚡ Graduate P0 `AgentIdentity` to **MCP OAuth 2.1 client** (RFC 6749 + RFC 9728 Protected Resource Metadata discovery) + **RFC 8707 Resource Indicators** (closes confused-deputy hole, mandatory in MCP June 2025+ spec) + **RFC 8693 token exchange** for sub-agent scope-down
- Replace JSONL audit log with **hash-chained signed log** (linear chain): each entry = `H(prev_hash || canonical_event_json)`, root signed by **software-held Ed25519 key** (libsodium / `cryptography` library) — KMS migration deferred to P10
- AS choice: **D1 evaluation** of WorkOS AuthKit, Auth0, and self-hosted Hydra; pick based on hands-on (research subagent's preference is WorkOS but unverified — confirm in D1 spike). Do not roll our own OAuth.
- Sub-agent identity propagation: parent's token + scope subset → token exchange → child token; child token visible in OTel `gen_ai.agent.parent_id`
- Acceptance: integration test spawns a sub-agent via RFC 8693 with strictly-narrower scope (verified by attempting a denied tool call from the child); README cites exact RFC numbers and EU AI Act Article 12

**P4b v1 deferred to P10 polish**:
- KMS key custody (AWS KMS / GCP KMS) — software-key signing in v1 is sufficient for research-platform threat model; KMS is a deployment hardening, not a design change
- Tamper-evidence mutation test (mutate one entry → verification fails) — manual review of audit-log integrity in v1
- Document both as known v1 gaps; promote in P10 with the Cosmos-team review

### P5 TrainOrchestrator (Wks 11–12.5) ⚡
- ⚡ **Centaur HPO**: LLM proposes candidates, **CMA-ES/TPE refines via shared mean/step-size** state. Pure-LLM HPO is empirically inferior (`arxiv:2603.24647`, Mar 2026).
- ⚡ `ComputeBackend` interface with three impls: `HFJobsBackend` (existing wrapper), `SkyPilotBackend` (multi-cloud Job Groups), `NeMoRunBackend` (NVIDIA-native multi-node)
- ⚡ For post-training paths, wrap **NeMo-RL** (production-mature, Nemotron-3-Super trained on it); do not reimplement
- Early-stop policy (validation plateau detection)
- W&B/MLflow monitor, both swap-in via OTel
- ⚡ **v4 Invariant 9**: at least one Centaur sweep runs on real GPU (Modal/Lambda Cloud, ~$50 budget); W&B run committed to repo as `docs/p5_real_sweep.md`
- Acceptance: agent runs a 6-config Centaur sweep, real GPU, selects winner with statistical justification (CI-aware, not point-estimate); Centaur ≥ parity with pure CMA-ES baseline within 1σ

### P5.5 PyTorch Depth (Wk 13) ⚡ — NEW IN v4

**Why this exists**: NVIDIA Cosmos JD requires *"Deep familiarity with PyTorch, including the ability to debug, adapt, and extend model behavior within larger software systems."* No phase in v3.2 demonstrated this. v4 ships a 1-week phase that produces *one PyTorch artifact reviewers can read and run* — code, profiler traces, before/after numbers.

**Three menu options (pick one in P5.5 D1, ship in D2-D5)**:

| Option | What | Why credible |
|---|---|---|
| **A. Custom autograd op** | Write a custom `torch.autograd.Function` (forward + backward) for one operation in a real workload (e.g., a fused attention variant, a custom loss with non-trivial gradient, a quantization-aware op). Validate gradient via `torch.autograd.gradcheck`. | Demonstrates understanding of autograd internals — *the* PyTorch depth signal |
| **B. `torch.compile` + kernel selection** | Take a real workload (Cosmos-relevant: video preprocessing, multimodal encoder, or attention block). Profile with `torch.profiler` (CUPTI / kineto traces). Identify hot kernel. Apply `torch.compile` with mode/options tuned by profiler evidence. | Demonstrates production PyTorch optimization — exactly what OptimizeAgent automates later in P6 |
| **C. Distributed training pattern** | Implement one non-trivial distributed pattern: tensor parallelism for a 1-2B model layer, or activation checkpointing tuned for a real memory budget, or a custom collective via `torch.distributed`. Validate via 2-GPU run on Modal. | Demonstrates depth in the area Cosmos team cares about most (multi-GPU world model training) |

**Recommendation**: Option B is the cheapest credibility-per-dollar — fits naturally with P6 OptimizeAgent (the work composes), uses cheaper single-GPU runs, and hits the JD's "debug, adapt, extend model behavior" exactly. Option A is the most impressive technically. Option C is the most Cosmos-aligned.

**Day-by-day**:

| Day | Deliverable | Acceptance |
|---|---|---|
| D1 | Pick option (A/B/C); commit choice + workload selection in `agent/optimization/pytorch_depth/CHOICE.md`; baseline measurement | baseline number recorded |
| D2-3 | Implement; for A: gradcheck passes; for B: profiler trace identifies bottleneck, compile pattern applied; for C: 2-GPU run completes | code committed; profiler artifacts in `docs/p5_5_profiler/` |
| D4 | Benchmark vs baseline; produce `docs/p5_5_speedup.md` with reproducible benchmark script | **≥10% wall-clock improvement** vs baseline (P5.5 numerical target) |
| D5 | Write 1-page README explaining the choice, the bottleneck, the fix, and the limitation. This becomes a portfolio artifact for interview. | README readable by ML systems eng; reproducible by `python bench.py` |

**Acceptance**:
- Code passes `pytest tests/optimization/pytorch_depth/`
- ≥10% measured wall-clock improvement on real workload (not synthetic)
- Profiler traces (kineto JSON) committed
- 1-page README explains the work in terms a Cosmos hiring manager can audit
- Stretch: open a discussion on PyTorch forum or file an issue if the optimization reveals a framework gap

**Why 1 week is enough**: this is *one* artifact, not a research thesis. Senior PyTorch engineers ship Option B in 2-3 days; we budget 5 days for setup + writeup + Cosmos-grade polish.

### P6 OptimizeAgent (Wks 14–15.5) ⚡
- Collapses old P2-4: profiling + training_opt + inference_opt
- ⚡ Quality budget enforced via **Inspect AI scorers** + paired structural verifiers (per §3.1 sentinel taxonomy); judge-only metrics flagged as advisory. **Hard gate**: ≤2% absolute regression on deterministic metrics; **target**: ≥1.5× wall-clock speedup on 4 baseline workloads (P6 numerical targets)
- ⚡ **Sandbox 2-tier (v3.1 trimmed)**: E2B (Firecracker) for CPU correctness checks; **Daytona** for GPU profiling/training. NemoClaw moved off the main path — it's alpha-stage early-preview (2026-03-16) and gating P6 on it is a credibility risk. NemoClaw becomes a **P10 stretch** for the on-prem-NVIDIA-aligned story.
- Single `SandboxRunner` interface so swapping Daytona → OpenShell/NemoClaw later is a config change, not a refactor
- Acceptance: 4 baseline workloads optimized with measured speedup + quality preservation; ≥1.5× speedup achieved on at least 3/4; speedup numbers committed to repo as `docs/p6_speedup_baseline.md` so any regression is auditable

### P7 Memory & compression (Wks 16–17) ⚡
- ⚡ **3-tier hierarchical memory** (not flat): (1) Core/index — Claude-Code-style `MEMORY.md` ≤25KB always-resident pointer index; (2) Recall — file-based, Anthropic `memory_*` tool API compatible; (3) Archival — Letta-compatible KG/vector hybrid
- Do NOT rebuild MemGPT-style paging; use Anthropic's memory tool surface or Letta server as the storage layer
- Owned `OptimizationContextManager` extending upstream — adds hierarchical summarization + retrieval over recall tier; emits `gen_ai.tool.memory.*` spans
- Acceptance: agent recalls a relevant prior session result on a new related task; memory tool API surface usable by any Claude-native client

### P8 Self-improvement loop (Wks 18–19) ⚡
- ⚡ Use **DSPy 3.x + `dspy.GEPA`** (do not roll our own GEPA) — mine failures from trajectory store → propose prompt/tool-description revisions → A/B vs control on Inspect AI golden suite → ratchet only on lower-CI-bound improvement (not point estimate)
- Per `RESEARCH_AHE_ANALYSIS.md` decision: offline-batch evolution (Option A), not online-session — **online self-improvement remains hype in 2026**
- Weekly cadence; every promotion requires human review and a signed promotion record in the audit log
- Acceptance: 1 prompt revision shipped via the loop, demonstrably improving judge pass-rate at p<0.05 with sentinel agreement preserved

### P9 Multi-agent e2e (Wks 19.5–21.5) ⚡

**Two demos, one phase** — split intentionally to hit two distinct JD bullets:

**P9a (W22) — Vertical pipeline demo**:
- ⚡ AV scenario gen via Cosmos Predict 2.5 + π₀.₅-class policy eval — maps directly onto the Cosmos team's internal stack
- DataAgent (cosmos-curate stages) → TrainOrch (NeMo-RL on SkyPilot) → EvalAgent (Inspect AI + physics-consistency scorer) → OptimizeAgent (Centaur over inference config)
- ⚡ Sub-agent identity propagated via RFC 8693 token exchange (P4b deliverable); each sub-agent's traces show parent_agent_id in OTel `gen_ai.agent.parent_id`
- Wall-clock target: **< 8 hours end-to-end** (P9 numerical target)

**P9b (W23) — CodeAgent demo (NEW in v3.1)**:
- Hits the JD's headline bullet: *"systems where AI doesn't just run models but helps build them"* — the data/train/eval pipeline alone doesn't prove this.
- A small `CodeAgent` (capability-scoped to `{read_file, write_file, run_tests, git_diff}`) iterates on a 10-bug fixture: read failing test → propose fix → write file → re-run tests → repeat (max N iterations).
- Runs in **E2B sandbox** with full OTel span capture; sentinel = `OutputFormatCheck(diff_is_valid_unified) AND SideEffectCheck(test_command_was_called) AND NoOpCheck`
- Target: **≥60% small-bug-fix success on the 10-bug fixture** (P9 numerical target). Below that, CodeAgent is a research artifact, not a demo. Above that, it's a credible "AI helps build AI" story.
- Owned path: `agent/optimization/agents/code_agent.py` + `tasks/codebench/*.py` (Inspect tasks built from a curated bug fixture)

**Joint acceptance**:
- P9a pipeline runs to completion in < 8h; one "trajectory of trajectories" captured; all four sub-agent identities verifiable in audit log
- P9b CodeAgent achieves ≥60% on bug fixture; replay logs viewable via Inspect View
- Both demos produce a 60-second screen-capture each for the P10 demo video

### P10 Production deployment + OSS upstream PR + demo (Wks 21.5–22.5) ⚡ — EXPANDED IN v4

**Why 2 weeks instead of 1**: v3.2's P10 was "publication" (`pip install` + demo video + blog). v4's P10 is "production" — actual users, actual traces, actual upstream contribution. This is the difference between "smart project" and "ship-grade product." Per §0.8 production commitments G3 and G5.

**Week 1 — Production deployment (W21.5–22)**:
- ⚡ **Deploy cosmos-lab reference agent on HF Spaces or Modal endpoint.** Choice TBD by free-tier feature parity at deploy time. Surface = a simple "submit ML task, see judged trajectory" UI; bring-your-own API key.
- Open public access; share in 2-3 ML communities (HF Discord, r/MachineLearning, NVIDIA developer forum)
- Gather **≥100 real user sessions over 1-week window** (Invariant 9 / §0.8 G3)
- Cost telemetry visible — total spend + per-session avg in deploy README
- One incident playbook: what to do if sentinel disagreement spikes / cost runaway / endpoint returns 5xx

**Week 2 — OSS upstream PR + polish (W22–22.5)**:
- ⚡ **Open ≥1 upstream PR** (§0.8 G5 — review-ready, not draft). Two candidates:
  - **PR to nvidia-nat**: contribute the sentinel taxonomy as a `nat`-native scorer plugin; reframes our `RewardHackSentinel` as `aiq.sentinel.NoOpCheck` + 3 siblings. *Strongest Cosmos signal.*
  - **PR to Inspect AI**: contribute a `paired_sentinel_scorer(judge, structural)` Scorer combinator that wraps any judge with mandatory structural verification. *Broadest community impact.*
  - Pick whichever has the cleanest reception window (check recent PR merge cadence in D1); ship the other as v1.1.
- ⚡ **`pip install cosmos-lab[all]` final release** with `nat`-runnable workflow YAML reference example
- ⚡ **P4b promotions** (deferred from P4b v1): KMS key custody (AWS KMS or GCP KMS) for the audit-log signing key + tamper-evidence mutation test
- ⚡ **NemoClaw stretch** (deferred from P6): wire NemoClaw as an alternative `SandboxRunner` backend, validated against the same 4 P6 workloads. Only ship if NemoClaw repo has stabilized between P6 and P10; otherwise punt to v1.2.
- ⚡ **5-minute demo video** showing: P9a pipeline Inspect View log + Phoenix span tree, P9b CodeAgent fixing a bug live, signed audit log entry, sentinel-caught reward-hack attempt (deliberately constructed), production endpoint dashboard with real user count + cost
- ⚡ **Blog post**: "Building a production-grade closed-loop ML platform — 6 agents on a sentinel-gated library, deployed in front of users, contributed upstream"

**Acceptance**:
- ≥100 real user sessions captured + summary committed (`docs/p10_production_traces.md`)
- ≥1 upstream PR opened, reviewer engaged (any response = success; merge = stretch goal)
- Demo video published; pip install works from clean env
- Total deployment cost in the budget envelope (~$200-400 across all of v4 GPU + deploy)

### References (v3)
- [Trajectory-Informed Memory Generation](https://arxiv.org/abs/2603.10600)
- [Self-Evolving Agents Survey](https://arxiv.org/html/2507.21046v4)
- [TRACE benchmark evolution](https://arxiv.org/html/2510.00415)
- [Can LLMs Beat Classical HPO? (`arxiv:2603.24647`)](https://arxiv.org/abs/2603.24647)
- [DSPy GEPA Optimizer docs](https://dspy.ai/api/optimizers/GEPA/overview/)
- [π₀.₅ paper (`arxiv:2504.16054`)](https://arxiv.org/abs/2504.16054)
- [NVIDIA NeMo-RL repo](https://github.com/NVIDIA-NeMo/RL)
- [SkyPilot v0.12 Job Groups](https://github.com/skypilot-org/skypilot)
- [NVIDIA OpenShell + NemoClaw](https://developer.nvidia.com/blog/run-autonomous-self-evolving-agents-more-safely-with-nvidia-openshell/)
- [Anthropic memory tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)
- [Letta v1 agent loop](https://www.letta.com/blog/letta-v1-agent)
- [MCP authorization draft](https://modelcontextprotocol.io/specification/draft/basic/authorization)
- [EU AI Act Art. 12 record-keeping](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-12)

---

## 6. Cross-cutting engineering practices (v3)

- **Test-first for owned modules**: every owned class has a unit test before integration
- **Lint**: `ruff check agent/optimization/ --ignore E501,F401,E402` clean each phase
- ⚡ **OTel GenAI from day one (P1+)**: every owned module emits `gen_ai.*` spans through the central `OTelGenAIEmitter`; `logging.getLogger("cosmos_lab.*")` reserved for structured app-level logs (not traces)
- **Reproducibility**: every `evaluate` run dumps a self-contained replay manifest (config + seeds + tool versions + sandbox image digest)
- **No hidden state**: all persistent state (audit log, trajectory store, memory) lives under `~/.cosmos_lab/` with explicit paths in config
- ⚡ **No judge-only metric reaches a gate**: every quality budget / PR gate / promotion decision requires a deterministic structural verifier alongside any judge score (anti-reward-hacking invariant)
- ⚡ **Bootstrap CI on every aggregated metric**: pass-rate, judge agreement, latency p50/p99 — point estimates without CIs are not committed to the leaderboard

---

## 6.5 Vendor independence — library boundary IS the answer (v3.2)

> **v3.2 reframe**: vendor independence used to be "three pluggable interfaces" (Sink, Provider, Compute). That's still true and still useful. But the *deeper* answer added in v3.2 is: cosmos-lab is a **library** (`pip install cosmos-lab`), not a fork of any platform. The library plugs into multiple agent harnesses via thin adapters (§0.4). This is the strongest possible vendor-independence story — stronger than any number of pluggable interfaces — because **the agent loop itself is swappable**.

### Harness adapters (the real vendor-independence layer)

| Harness | Adapter | Status | Use case |
|---|---|---|---|
| **`nvidia-nat`** | `cosmos_lab.harness.nat` | **Primary** (P0.5+) | Cosmos pitch, NVIDIA-native deployments |
| **`ml-intern`** | `cosmos_lab.harness.ml_intern` | v1 compat (P0.5+) | HF stack + web UI + existing flywheel |
| Claude Agent SDK | `cosmos_lab.harness.claude_sdk` | future (v1.1) | Anthropic-native deployments |
| OpenAI Agents SDK | `cosmos_lab.harness.openai_agents` | future (v1.1) | OpenAI-native deployments |
| LangGraph | `cosmos_lab.harness.langgraph` | future (v1.2) | HITL durable workflows |

### Three pluggable interfaces (still in scope, complement the harness adapters)

| Interface | Phase introduced | HF-native impl (default) | NVIDIA-native impl | Other backends |
|---|---|---|---|---|
| **Sink** (`TrajectorySink`) | P1 | `OTelGenAIEmitter → Phoenix` (default); `HFDatasetSink` opt-in (P8 flywheel) | `S3Sink` / `NGCArtifactSink` (P4) | `DuckDBSink` opt-in (P4a analytics); MongoSink cut in v3.1 |
| **Provider** (`LLMProvider` via litellm) | P2 | HF router (already in `litellm`) | `NIMProvider` (P2) | Anthropic / OpenAI direct, Modal endpoints (P5+) |
| **Compute** (`ComputeBackend`) | P5 | `HFJobsBackend` (wraps `jobs_tool`) | `DGXBackend` stub + `NGCJobsBackend` (P5 stretch) | `ModalBackend` (P5), `LambdaBackend` (P5) |

### Observability

- ⚡ **Primary (v3)**: `OpenTelemetry GenAI` semantic conventions (`gen_ai.*` spans), opt-in via `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`. Backend = Phoenix (OSS) by default; Langfuse / W&B / Weave / DataDog all swap-in by config (mirrors `nvidia-nat` telemetry block).
- HF dataset upload remains as the *data flywheel sink* (feeds `kpis_scheduler.py` rollups and AHE/SFT loops); not removed, just one of N OTel exporters
- ⚡ Local secondary: `logging.getLogger("cosmos_lab.*")` for structured app logs only (not traces) — keeps non-OTel debugging cheap

### Identity (P0 AuthZ MVP → P4 MCP-OAuth → P9 federated multi-agent) ⚡

- Phase 0 (shipped): in-process `AgentIdentity`, unsigned, JSONL audit log — explicitly an AuthZ MVP, no AuthN
- ⚡ Phase 4: graduate to **MCP OAuth 2.1 + RFC 8707 Resource Indicators** (closes confused-deputy hole) + **RFC 8693 token exchange** for sub-agent scope-down + **hash-chained signed audit log** aligned to EU AI Act Art. 12 (enforceable 2026-08-02). Use **WorkOS AuthKit** or **Auth0** as the OAuth 2.1 AS — do not roll our own.
- Phase 9: when multi-agent orchestration lands, wire HF OAuth + a generic OIDC seam so the platform can authenticate against NVIDIA IDP when integrated

### Sandbox (v3)

- ⚡ **2-tier `SandboxRunner` interface** (lands in P6): (1) **E2B (Firecracker)** for CPU-only correctness checks (~150ms cold start, hardware isolation, cheap); (2) **Daytona** OR **NVIDIA OpenShell + NemoClaw** for GPU profiling/training (kernel-level policy, persistent stateful workspaces). OpenShell is the on-prem default for Cosmos-aligned deployments.
- Anthropic-managed code execution: explicitly NOT used — opaque, no GPU, no on-prem.

### What we deliberately do *not* multi-vendor

- Tool registry / `ToolRouter` interface stays unchanged — it's already provider-agnostic
- Frontend / SSE transport stays HF-stack — porting it has no Cosmos-pitch value
- Memory storage layer: pick Anthropic `memory_*` tool API OR Letta — not both; both is feature-creep

**Net pitch (v3.2)**: "cosmos-lab is a `pip install`-able library that adds governance — sentinel-gated judging, MCP-OAuth identity with RFC 8693 sub-agent scope-down, GEPA promotion contracts, quality budget invariants — to whatever agent harness you already run. Default install ships adapters for **`nvidia-nat`** (Cosmos team's stack) and **ml-intern** (HF stack); Claude Agent SDK and OpenAI Agents SDK adapters land in v1.1. The library never owns the agent loop — that's what makes it portable. Cosmos team can `pip install cosmos-lab[nat]` and `nat run cosmos-lab.yaml` today; everything else is a config swap."

---

## 7. What's deprioritized vs PLAN.md

- **Deep optimization sub-phases** (old P2 training opt → P3 inference opt → P4 multimodal opt → P5 VLA opt → P6 custom kernels): collapsed into single P6 OptimizeAgent vertical. Custom CUDA kernel generation removed from main path (re-add as P11 stretch goal if e2e ships early).
- **AHE Stages A–I detail**: subsumed into P8 self-improvement loop. The 7-slot decomposition from `RESEARCH_AHE_ANALYSIS.md` informs which slots get evolved (system_prompt, tool_description) but isn't sequenced as 9 stages.
- **Two-level benchmarking with Amdahl deviation**: deferred to P6 entry; P1 eval harness handles the simpler agent-task case first.

---

## 8. Open questions to resolve before P1 (v3)

Resolved-in-v3 (carried from v2):
- ✅ **DuckDB vs SQLite** → DuckDB stays as analytics layer **over OTel-shaped rows** (not as primary schema)
- ✅ **Judge model** → Sonnet 4.6 ×3 default in `MultiJudge`; Opus 4.7 only for CI-straddle tie-break
- ✅ **Multi-judge agreement metric** → bootstrap CI on pass-rate is the headline metric; pairwise Cohen's κ as diagnostic only (debate framing dropped)
- ✅ **NIM endpoint** → `cosmos-reason-2` first (smaller surface, easier to mock + judge)

Still-open (v3 raises):
1. **OTel GenAI semconv stability** — spec is experimental as of 2026-Q1, opt-in via stability flag. *Risk*: schema may change before 1.0. *Mitigation*: pin to a specific draft date in `requirements.txt`, add a v1.0-migration test that runs against both versions. *Decision needed before P1 D1.*
2. **Inspect AI vs `nat eval` as primary harness** — both are credible; Inspect AI has wider production adoption (METR), `nat eval` has direct Cosmos-team alignment. *Recommendation*: Inspect AI as primary; emit Inspect logs in a `nat eval`-compatible artifact directory layout so a Cosmos reviewer can also `nat eval --reuse-artifacts` them. *Confirm before P1 D2.*
3. **Capability × approval-policy ordering** — P1 D1 ticket: confirm capability check fires *before* approval policy (cheaper to deny early; preserves policy budget for genuinely-allowed but-expensive calls). *Verify with an ordering test in `tests/optimization/test_router_policy_integration.py`. (Carried — still open.)*
4. **OAuth 2.1 AS choice** (P4b D1 spike) — WorkOS AuthKit vs Auth0 vs self-hosted Hydra. *Recommendation* (research-subagent's preference, **not yet hands-on verified**): WorkOS for cleanest MCP-native docs as of Q2 2026. *D1 action*: 1-day spike comparing all three; pick by hands-on integration friction, not by doc browsing. Don't lock in before the spike.
5. **Sandbox GPU tier** (P6) — Daytona vs NVIDIA OpenShell as default. Daytona is faster to integrate; OpenShell is the Cosmos-pitch-aligned choice. *Recommendation*: ship both behind `SandboxRunner`; default = Daytona for OSS path, OpenShell for "NVIDIA-aligned profile" config. *Confirm before P6.*
6. **DSPy 3.x version pin** (P8) — `dspy.GEPA` is verified as a public API (per [DSPy docs](https://dspy.ai/api/optimizers/GEPA/overview/), production-ready, integrated under Optimizers section). API surface still evolving. *Mitigation*: pin minor version in `requirements.txt`, add a smoke test that runs one GEPA pass on a fixture. *Confirm before P8.*
7. **EU AI Act Art. 12 hash-chain construction** (P4b) — Merkle vs linear hash chain; signing key custody (HSM vs KMS vs software). *Decision*: linear hash chain (simpler; sufficient for Art. 12 tamper-evidence). *v1 cut*: software-held Ed25519 key in P4b; KMS migration + tamper-mutation test deferred to P10. *Known v1 gap*: software-key signing is acceptable for research-platform threat model but not for shipping to enterprise customers; document explicitly in P4b README and call out in P10 promotion.
8. **P9 vertical choice** — AV scenario gen (Cosmos-aligned, GPU-heavy) vs code-pass@1 (cheap to demo, less Cosmos-credible). *Recommendation*: AV scenario gen, with a code-pass@1 fallback in v0 of the pipeline so we can demo the orchestration even before GPU access. *Confirm before P5 entry.*
