# North Star — cosmos-lab in 1 screen (v5.2)

## What we are building

**The production governance layer that makes ml-intern (or any autonomous ML agent) safe to deploy at NVIDIA Cosmos scale.**

Two components:
1. **ml-intern** — already a fully-autonomous ML engineering agent (HF's product). Has planning (`plan_tool`), sub-agent spawning (`research_tool`), 20+ ML tools, sandbox, HF integration, doom-loop detection. Leveraged AS-IS.
2. **cosmos-lab** — what we ship: 10 production-governance components ml-intern doesn't have.

## Why we are building it

NVIDIA Cosmos team JD: *"AI doesn't just run models but helps build them."* "Strong agency in LLM-based systems." "Design and scale evaluation platforms."

In 2026, autonomous agents are commoditizing (Devin, Operator, Claude Code, ml-intern, Cursor Composer all exist). What's NOT commoditized — what NVIDIA Cosmos team specifically needs for production deployment — is the **governance layer** that makes these agents safe + auditable + improvable. cosmos-lab fills that gap.

## The 10 governance components cosmos-lab adds

| # | Component | What ml-intern has | What cosmos-lab adds |
|---|---|---|---|
| 1 | Sentinel-gated quality | basic eval | 4 sentinel types paired with judge — no judge-only metric reaches a gate |
| 2 | Cross-session memory | per-session `logged_events` | 3-tier (working/episodic/semantic) persistent memory |
| 3 | RFC 8693 capability expansion | static `tool_router` scope | Earned-trust expansion via token exchange |
| 4 | Hash-chained signed audit | basic JSON logging | Tamper-evident; EU AI Act Art. 12 compliant |
| 5 | OTel-GenAI native observability | HF telemetry | `gen_ai.*` semconv; portable to any backend |
| 6 | GEPA self-improvement | none | DSPy 3.x offline pass; ratchet on lower-CI improvement |
| 7 | MultiJudge with bootstrap CIs | ad-hoc | N=3 judges; no debate dynamics |
| 8 | Inspect AI integration | none | UK AISI standard adoption |
| 9 | PR-gating + canary deployment | none | Block regressions; sequential testing |
| 10 | AGENTIC_EVAL_SPEC discipline | none | Full eval architecture (T0-T4 + S1-S6 + E1-E10) |

## The demonstration

```bash
$ ml-intern --task "Improve Cosmos Reason 2 by 3pp" \
            --cosmos-lab-governance \
            --identity researcher@cosmos \
            --budget $400 \
            --timeout 1week
```

What happens:
- ml-intern's autonomous agent runs the actual ML work (planning, experimentation, training, evaluation)
- cosmos-lab governance wraps every step: identity check + sentinel evaluation + OTel span emission + signed audit log
- Cross-session memory persists across compute interruptions
- Capability scope expands when sentinel-clean runs accumulate
- Weekly GEPA pass mines trajectory for prompt-revision candidates

End of week: measured pass-rate +3pp (with bootstrap CI + p-value + sentinel agreement), full Phoenix trajectory, signed audit log, cost report ($383/$400).

## Schedule

~13 weeks. P0 + P0.5 (~3 days work) shipped. ~10 weeks remaining for P1-P9 governance enhancements + production deployment.

Compressed from v5/v5.1's 22.5 weeks because v5.2 doesn't re-implement what ml-intern already has (planner, executor, memory tier internals, sub-agent spawning).

## When done (Week ~13)

A Cosmos hiring manager opens the repo and in 5 minutes sees:
- README → `pip install cosmos-lab[nat]` → `nat run cosmos-lab.yaml`
- ml-intern + cosmos-lab solving a fresh Cosmos task end-to-end
- 10 governance components live with measured numbers
- Upstream OSS PR linked
- Production endpoint dashboard with real users
- Signed audit log
- 5-minute demo video

> *"AI helps build AI"* — autonomous agent (ml-intern) + production governance (cosmos-lab) = deployable ML lifecycle automation for Cosmos team.
