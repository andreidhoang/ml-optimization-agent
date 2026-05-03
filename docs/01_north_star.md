# North Star — cosmos-lab in 1 screen (v6)

## What we are building

**9 NEW agents specialized for NVIDIA Cosmos team's ML lifecycle work, with production governance, built on ml-intern's tool primitives.**

Three layers:
1. **6 Cosmos-specialty agents** — DataAgent, EvalAgent, TrainOrchestrator, OptimizeAgent, MultimodalPipelineAgent, CodeAgent
2. **3 governance agents** — GepaOptimizer (self-improvement), CapabilityProbe (adversarial security), CrossAgentEvaluator (vendor comparison)
3. **~16 governance infrastructure components** — sentinels, identity v2, audit log, OTel emitter, memory tiers, Inspect AI bridge, ComputeBackend, etc.

All built on **ml-intern's tool primitives** (agent_loop blocks, 16 generic tools, sandbox, MCP, cost estimation, doom-loop detection) leveraged AS-IS — not reimplemented.

## Why we are building it

NVIDIA Cosmos team JD literally asks: *"Create self-improving loops where agents (plural) help generate data, surface failures, evaluate outputs"* + stand-out *"agent-based systems doing real work — coding, eval, data gen, triage, experimentation, orchestration"*.

This describes **multiple specialty agents for different ML lifecycle stages** — not one PrincipalAgent (v5/v5.1 over-correction), not zero agents with just governance (v5.2 over-correction), but multiple specialty agents + governance.

Plus 2026 reward-hacking crisis (METR + UC Berkeley) means specialty agents need sentinels + signed audit + capability expansion + GEPA self-improvement to be production-deployable.

## The 9 agents

### Layer 1 — 6 Cosmos-specialty agents (ML lifecycle work)

| Agent | Phase | Real work |
|---|---|---|
| **DataAgent** | P3 | Curate 10-100 hours real video through cosmos-curate; ship dataset card with W&B Artifacts lineage |
| **EvalAgent** | P4a | Multi-judge with bootstrap CIs + reward-hack sentinels; physics-consistency scorers; PR-gating |
| **TrainOrchestrator** | P5 | Centaur HPO; ComputeBackend over SkyPilot/NeMo-Run/HF Jobs; NeMo-RL post-training; real GPU sweep |
| **OptimizeAgent** | P6 | Profile workload + apply optimization; ≥1.5× speedup on 4 real workloads, ≤2% regression |
| **MultimodalPipelineAgent** | P9 | E2E Cosmos workflow on Predict 2.5 + π₀.₅; real Cosmos NIM endpoint |
| **CodeAgent** | P9 | Capability-scoped {read_file, write_file, run_tests, git_diff}; real OSS bug fixes |

### Layer 2 — 3 governance agents (meta-layer)

| Agent | Phase | What it does |
|---|---|---|
| **GepaOptimizer** | P8 | Weekly: mine failures → DSPy GEPA prompt revisions → A/B test → signed promotion |
| **CapabilityProbe** | P7 | Adversarial: red-team capability scope before each expansion event |
| **CrossAgentEvaluator** | P10 | Quarterly: spawn ml-intern+cosmos-lab vs Devin vs Claude Code vs human on identical task; Pareto chart |

## The demonstration

Cosmos team uses cosmos-lab via nat workflow:

```bash
$ nat run cosmos-lab.yaml --task "Improve Cosmos Reason 2 by 3pp"
```

What happens:
- DataAgent prepares data → TrainOrchestrator runs sweep on real GPU → EvalAgent scores → OptimizeAgent compresses winner → MultimodalPipelineAgent orchestrates the e2e workflow
- Background: GepaOptimizer mines for prompt improvements; CapabilityProbe tested scope before this run; CrossAgentEvaluator records data for next quarterly Pareto
- Each specialty agent uses ml-intern session + scoped CapabilityScopedRouter + sentinel paired evaluation + OTel span emission + signed audit log entry
- Cross-session memory persists across compute interruptions

Final: measured pass-rate +4.2pp (with bootstrap CI + p-value + sentinel agreement), full Phoenix trajectory across all 6 specialty agents, signed audit log, cost report ($383/$400).

## Schedule

~19 weeks. P0 + P0.5 (~3 days work) shipped. ~17 weeks remaining for P1-P10.

Between v5/v5.1's 22.5w (too long — included reimplementation) and v5.2's 13w (too short — removed agents JD asks for). v6 is honest middle ground.

## When done (Week ~19)

A Cosmos hiring manager opens the repo and sees:
- README → `pip install cosmos-lab[nat]` → `nat run cosmos-lab.yaml`
- 6 specialty agents + 3 governance agents + ~16 infrastructure components
- ml-intern's tool primitives leveraged as substrate (no reimplementation)
- Real GPU runs with measured numbers (Invariant 9)
- Real OSS upstream PR (P10)
- Production endpoint dashboard with real users
- Signed audit log
- 5-minute demo video showing all 9 agents in action

> *"AI helps build AI"* — 6 specialty agents doing real Cosmos work + 3 governance agents keeping them safe + production infrastructure that lets you deploy them.
