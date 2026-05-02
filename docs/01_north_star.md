# North Star — cosmos-lab in 1 screen (v5)

## What we are building

**One exceptional autonomous principal agent** that does long-horizon ML lifecycle work the way a real principal engineer does — taking vague research questions, decomposing them into experiments, writing real code, running real GPU workloads, observing surprising results, replanning when sentinels trip, and delivering measured outcomes. All within an exceptional self-managed context harness, governed by sentinels that compound trust over time.

Built on ml-intern's `agent_loop.py` as substrate. Runs natively on `nvidia-nat`.

## Why we are building it

NVIDIA Cosmos team JD: *"AI doesn't just run models but helps build them."* "Strong agency in LLM-based systems, code agents doing real work."

In 2026, the agentic frontier IS autonomous capability — Devin, Operator, Cursor Composer, Claude Code. The differentiator that makes cosmos-lab L6-grade vs commodity isn't "we wrap autonomous agents in a governance layer" (that's CI for agents — table stakes). The differentiator is **exceptional autonomy made safe** — sentinels as tripwires for replanning (not gates), capability scope expanding with earned track record, OTel trajectory as long-horizon memory.

## The PrincipalAgent

One agent. Six capability domains it demonstrates over P3-P9. Same agent, six skills. Like one principal engineer who does data work Monday, training Tuesday, optimization Wednesday — not six different specialists.

| Capability | Phase | Real work |
|---|---|---|
| **Data curation** | P3 | Process 10-100 hours of real video through cosmos-curate, ship dataset card with full lineage |
| **Eval design** | P4a | Design multi-judge eval with bootstrap CIs + reward-hack sentinels, gate PRs |
| **Training orchestration** | P5 | Centaur HPO sweep on real GPU (Modal/Lambda), pick winner with statistical justification |
| **PyTorch optimization** | P5.5 + P6 | Custom autograd op or torch.compile pattern, ≥10% wall-clock improvement; ≥1.5× speedup on 4 real workloads |
| **Multimodal pipeline** | P9a | E2E pipeline on real Cosmos NIM endpoint, < 8h wall-clock |
| **Code work** | P9b | Real OSS bug fixes on real GitHub issues (capability scope earned in P3-P9a) |

## The three pillars (each "exceptional," not "adequate")

**1. Long-horizon autonomous reasoning** — multi-day work persisting across sessions via OTel trajectory + episodic memory; PLAN→EXECUTE→VERIFY→REPLAN loop; resume mid-experiment after compute interruption.

**2. Exceptional context harness (agent-managed)** — 3-tier memory (working/episodic/semantic), self-edited current state, self-written verifiers, trajectory store as replay buffer, pointer index agent maintains.

**3. Governance as enabler (not fence)** — sentinels are tripwires (trip → structured feedback → agent replans, not blocks); capability scope expands with earned trust (RFC 8693 token exchange after K sentinel-clean runs); audit log = replay buffer not just compliance artifact; GEPA = agent self-improvement.

## What makes this exceptional vs 2026 baseline (Devin/Operator/Cursor/Claude Code)

1. **ML-lifecycle-native tools** — full ML stack (cosmos-curate, NeMo-RL, SkyPilot, NIM, W&B, Inspect AI, profilers) as one tool registry
2. **Long-horizon by construction** — multi-day persistence via OTel + episodic memory (most 2026 agents are session-bounded)
3. **Sentinel-gated replanning** instead of silent failure — 4 sentinel types cover Berkeley/METR failure modes
4. **Earned-trust capability expansion** via RFC 8693 — narrow start, broaden with track record
5. **`nat`-runnable + OTel-GenAI-native** — drops into Cosmos team's stack with one command

## The 5 production gates (Invariant 9 + §0.8)

- **G1**: Real GPU runs (~$200-400 budget across P5/P5.5/P6/P9a)
- **G2**: PyTorch depth artifact — P5.5 ≥10% wall-clock improvement on real workload
- **G3**: Real production deployment — P10 ≥100 user sessions over 1-week window
- **G4**: Real multimodal data — P3 10-100 hours real video processed
- **G5**: Upstream OSS PR — P10 ≥1 review-ready PR to nvidia-nat or Inspect AI

## The demo (this is the product)

Cosmos hiring manager opens cosmos-lab and says: *"Take this Cosmos Reason 2 task. Improve pass-rate ≥3pp. One week, $400 budget."*

PrincipalAgent runs unattended ~5 days. Delivers: measured pass-rate improvement (with p-value + sentinel agreement), full Inspect View replay log, full OTel trajectory, W&B sweep showing winners and losers, 1-page report explaining hypothesis + what worked + what didn't, cost report.

That's **AI helping build AI** in production. Not orchestrating. Not gating. *Doing the work.*

## Schedule

22.5 weeks. 13 phases (P0 + P0.5 shipped, P1-P10 ahead). See `PLAN_V2.md §1` for full table, `docs/03_pointers.md` for phase → anchor map.

## When done (Week 22.5)

A Cosmos hiring manager opens the repo and in 5 minutes sees: README → `pip install cosmos-lab[nat]` → `nat run cosmos-lab.yaml` → PrincipalAgent solving a fresh ML task end-to-end → 6 demonstrated capabilities with measured numbers → upstream PR linked → production endpoint dashboard with real users → signed audit log → demo video.

> *"AI helps build AI"* — in production, by an agent that reasons, plans, executes, observes, replans, and ships.
