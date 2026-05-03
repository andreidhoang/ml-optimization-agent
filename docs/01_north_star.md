# North Star — cosmos-lab in 1 screen (v7 — frontier-aligned, final)

## What we are building

**A frontier-aligned production agentic system for NVIDIA Cosmos team's ML lifecycle work.**

5 production agents + 1 Skill + 3 offline tools + ~16 infrastructure, on LangGraph durable substrate + Magentic-One ledger pattern + ml-intern's tool primitives. Verified against 2026 frontier patterns at Anthropic, NVIDIA, LangGraph (Uber/JPMC production), Microsoft Agent Framework, Inspect AI (UK AISI), Mem0, MCP authorization spec.

## Why we are building it

NVIDIA Cosmos team JD literally asks: *"agentic systems that reason about, build, evaluate, and improve AI systems themselves"* + *"agents (plural) help generate data, surface failures, evaluate outputs"* + stand-out *"agent-based systems doing real work — coding, eval, data gen, triage, experimentation, orchestration"*.

This describes **multiple specialty agents** for different ML lifecycle stages PLUS production governance. ml-intern primitives are HF-flavored building blocks; v7 specializes them into Cosmos-aligned production agents using 2026-converged patterns.

## The 5 production agents (v7 honest count after frontier audit)

### Layer 1 — PrincipalAgent supervisor + 4 specialty workers

| Agent | Phase | Distinct tool surface | Frontier pattern |
|---|---|---|---|
| **PrincipalAgent** | P3 | LangGraph supervisor + Magentic-One Task/Progress Ledger + Skills loader | Hierarchical orchestrator-worker (Anthropic Multi-Agent Research, Magentic-One, LangGraph supervisor — convergent 2026) |
| **DataAgent** | P4a | cosmos-curate/NeMo Curator/synthetic gen | Magentic-One worker pattern |
| **EvalAgent** | P5 | Inspect AI/MultiJudge/5-type sentinels | Inspect AI standard substrate |
| **TrainOrchestrator** | P5 | NeMo-RL/SkyPilot/HF Jobs | nat plugin pattern + production training |
| **OptimizeAgent** | P6 | profiler/kernel/sandbox 2-tier | Production optimization pattern |

### Layer 2 — Skills (loaded by PrincipalAgent — Anthropic Skills pattern)

| Skill | Phase | Why a Skill not Agent |
|---|---|---|
| **CodeWork** | P7 | Commodity tools (file ops + tests in E2B); Anthropic Skills blog rejects per-domain agents for commodity capabilities |

### Layer 3 — 3 offline governance tools (NOT standing agents — frontier convergence)

| Tool | Cadence | Why offline |
|---|---|---|
| **GepaOptimizer** | Monthly cron | Decagon ships GEPA offline only; NO production deployment as standing agent |
| **CapabilityProbe** | CI/CD on capability expansion | METR pattern; co-resident standing would poison trace store |
| **CrossAgentEvaluator** | Quarterly | Inspect AI cross-agent comparison standard |

### Layer 4 — ~16 infrastructure components

Identity (P0 + RFC 8693) | 5-type sentinels via Anthropic PostToolUse hooks | OTel + 4-scope hybrid memory (Mem0/Letta) | Inspect AI + cross-family MultiJudge | **LangGraph durable supervisor + Magentic-One ledger** | **Context engineering discipline** (cache-aware prompt structure + 75% compaction + just-in-time retrieval + cosmos-progress.md state file + behavior-vs-capability staleness check) | ComputeBackend + sandbox 2-tier | reproducibility envelope (incl. CUDA versions) | nat deployment wrapper

### Layer 5 — ml-intern primitives (LEVERAGED inside LangGraph worker nodes)

agent_loop, 16 generic tools, sandbox, MCP, cost estimation, doom-loop detection.

## v7 frontier-fixed (vs v6)

6 issues caught by 3 parallel audit agents, all addressed in v7:

| Issue | v6 | v7 fix |
|---|---|---|
| Per-domain agents = anti-pattern (Anthropic Skills blog) | 6 specialty agents | 4 specialty workers (distinct tool surfaces) + 1 PrincipalAgent supervisor + CodeWork Skill |
| GEPA standing agent has no production precedent | GepaOptimizer as standing agent | Offline batch tool (Decagon pattern) |
| Sentinel "tripwire-replan" not in production | Novel mechanism | Anthropic PostToolUse hooks contract |
| 3-tier memory hierarchy is research not convergent | 3-tier hierarchical | 4-scope hybrid (Mem0/Letta) |
| Co-resident probe poisons trace store | Standing agent | CI/CD eval lane via Inspect AI snapshots |
| "Earned-trust capability expansion" oversold | Custom semantics | Standard RFC 8693 delegation only |

Plus 8 frontier additions: LangGraph durable substrate, Magentic-One ledgers, 5th sentinel (judge-hacking per Gaia2), cross-family MultiJudge, CodeWork Skill, RFC 8707 day-one, reward-hack Pareto axis, CUDA versions in envelope.

## Schedule

~21 weeks. P0 + P0.5 (~3 days work) shipped. ~18 weeks remaining for P1-P10.

Slightly more than v6's 19w because LangGraph integration + PrincipalAgent foundation + Magentic-One ledger pattern + 5th sentinel are all frontier-required additions per audit. Honest scoping, not optimistic.

## When done (Week ~21)

A Cosmos hiring manager opens the repo and sees:
- README → `pip install cosmos-lab[nat]` → `nat run cosmos-lab.yaml`
- 5 agents + CodeWork Skill + 3 offline tools + ~16 infrastructure
- LangGraph durable supervisor with Magentic-One ledger pattern
- 5-type sentinel suite (incl. judge-hacking detector)
- Cross-family MultiJudge (3× Sonnet + 1× non-Anthropic)
- MCP OAuth + RFC 8693 + signed audit (EU AI Act Art. 12 compliant)
- 4-scope hybrid memory via Mem0/Letta
- Real GPU runs with measured numbers (Invariant 9)
- Real OSS upstream PR
- Production endpoint dashboard with real users
- 5-min demo video showing all 5 agents + CodeWork Skill solving Cosmos task

> *"AI helps build AI"* — frontier-aligned production agentic system for ML lifecycle work, with Cosmos team's actual stack (NIM + cosmos-curate + NeMo-RL).

## Confidence anchor — why v7 IS final

3 independent senior-engineer research agents conducted parallel audits and converged on the same 6 fixes + 8 additions. This is the synthesis. Future audit findings document as v1.1+ work, not v8 — process needs to converge.
