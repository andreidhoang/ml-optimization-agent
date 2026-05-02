# Phase → PLAN_V2.md anchor map

> Use this when you need deep detail on a phase. Read the *specific section* of `PLAN_V2.md`, not the whole file.

| Phase | What it ships | PLAN_V2.md section |
|---|---|---|
| **P0** (shipped) | Identity AuthZ MVP | `§2 Phase 0` |
| **P0.5** (current, 4 days) | Library restructure + harness adapters | `§2.5 Phase 0.5` |
| **P1** (W2-3) | TrajectorySink + OTel GenAI + Inspect AI + sentinel taxonomy | `§3 Phase 1` + `§3.1 Sentinel taxonomy` |
| **P2** (W4) | Cosmos provider scaffolding (NIMProvider + tool wrappers) | `§4 Phase 2` |
| **P3** (W5-6) | DataAgent — real video curation through cosmos-curate | `§5 P3` |
| **P4a** (W7) | EvalAgent platform — multi-judge + bootstrap CI + PR gate | `§5 P4a` |
| **P4b** (W8-9) | Identity v2 — MCP OAuth + RFC 8707/8693 + signed log | `§5 P4b` |
| **P5** (W10-11.5) | TrainOrchestrator — Centaur HPO + ComputeBackend | `§5 P5` |
| **P5.5** (W12) | PyTorch Depth — custom autograd OR torch.compile pattern | `§5 P5.5` |
| **P6** (W13-14.5) | OptimizeAgent — ≥1.5× speedup, ≤2% regression on real GPU | `§5 P6` |
| **P7** (W15-16) | Memory & compression (3-tier hierarchical) | `§5 P7` |
| **P8** (W17-18) | GEPA self-improvement loop (offline DSPy) | `§5 P8` |
| **P9a** (W18.5-19.5) | Multi-agent e2e — Cosmos Predict 2.5 + π₀.₅ pipeline | `§5 P9` |
| **P9b** (W19.5-20.5) | CodeAgent — bug fix in E2B sandbox | `§5 P9` |
| **P10** (W20.5-22.5) | Production deploy + OSS upstream PR + demo video | `§5 P10` |

## Cross-cutting

| Topic | PLAN_V2.md section |
|---|---|
| Library architecture (why) | `§0.4 Library architecture` |
| What only cosmos-lab does | `§0.6 unique value` |
| 6 reference agents matrix | `§0.65 Six reference agents` |
| 24 numerical targets | `§0.7 Numerical targets` |
| 5 production commitments | `§0.8 Production commitments` |
| 9 invariants | `§0 Invariants` |
| Reuse map (upstream + external) | `§1.5 Reuse map` |
| Sentinel taxonomy | `§3.1 Sentinel taxonomy` |
| Vendor independence | `§6.5 Vendor independence` |
| Open questions | `§8 Open questions` |

## How to read PLAN_V2.md efficiently

```bash
# Don't load the whole file. Use grep + Read with offset:
grep -n "^## " PLAN_V2.md          # see all section anchors
# Then Read tool with offset pointing to the section you need.
```
