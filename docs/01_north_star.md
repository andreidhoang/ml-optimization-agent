# North Star — cosmos-lab in 1 screen

## What we are building

A Python library (`pip install cosmos-lab`) that ships **6 reference ML lifecycle agents** running on a shared **governance runtime** — sentinel-gated judging, MCP-OAuth identity with sub-agent scope-down, GEPA self-improvement with signed promotion, quality-budget invariants. Plugs into `nvidia-nat` (primary harness), `ml-intern` (compat), Claude SDK (v1.1).

## Why we are building it

NVIDIA Cosmos team JD: *"AI doesn't just run models but helps build them."* In 2026, agents can do real ML work — but they reward-hack (METR: o3 hacks 30%+ RE-Bench tasks). Every agent platform gives you the *runtime*; nobody packages the *governance* needed to deploy these agents in production with compliance + credibility + rollback. **cosmos-lab is that governance layer.**

## The 6 agents

| Agent | Phase | Real work |
|---|---|---|
| **DataAgent** | P3 | Curate 10-100 hours of real video through cosmos-curate |
| **EvalAgent** | P4a | Multi-judge + sentinels + PR-gate regression block |
| **TrainOrchestrator** | P5 | Centaur HPO, real GPU sweep on Modal/Lambda |
| **OptimizeAgent** | P6 | ≥1.5× wall-clock speedup on 4 real workloads, ≤2% regression |
| **VideoUnderstandingAgent** | P9a | E2E pipeline with real Cosmos NIM endpoint |
| **CodeAgent** | P9b | ≥60% small-bug-fix on 10-bug fixture in E2B sandbox |

## The 5 governance pillars (the unique IP)

1. **Sentinel-gated judging** — `(judge, structural-verifier)` pair on every quality gate; blocks reward-hacking by design.
2. **MCP OAuth 2.1 + RFC 8707/8693 identity** — sub-agent token scope-down, hash-chained signed audit log (EU AI Act Art. 12).
3. **GEPA promotion contract** — every prompt revision carries lower-CI evidence + sentinel agreement + human signature + signed log entry.
4. **Quality budget as architectural invariant** — no judge-only metric reaches a gate (Invariant 8).
5. **OTel GenAI spans by default** — portable observability, swap Phoenix/Langfuse/Weave/Datadog by config.

## The 5 production gates (Invariant 9 + §0.8)

- **G1**: Real GPU runs (~$200-400 budget across P5/P5.5/P6/P9a)
- **G2**: PyTorch depth artifact — P5.5 ships custom autograd or torch.compile, ≥10% wall-clock improvement
- **G3**: Real production deployment — ≥100 user sessions over 1-week window
- **G4**: Real multimodal data — 10-100 hours real video processed in P3
- **G5**: Upstream OSS PR — ≥1 review-ready PR to nvidia-nat or Inspect AI for sentinel pattern

## Schedule

22.5 weeks. 13 phases. P0 shipped, P0.5 starting now (4 days). See `PLAN_V2.md §1` for full table, `docs/03_pointers.md` for phase → anchor map.

## When done (Week 22.5)

A Cosmos hiring manager opens the repo and in 5 minutes sees: README → `pip install cosmos-lab[nat]` → `nat run cosmos-lab.yaml` reproduces demo → 6 agents live with measured numbers → upstream PR linked → production endpoint dashboard with real users → signed audit log → demo video.

> *"AI helps build AI"* — in production, with sentinels that prove it's safe.
