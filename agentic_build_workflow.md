# Agentic Build Workflow

> A simple, repeatable workflow for any engineering or building project —
> distilled from `cowork_os_karpathy.md`.
> Works for features, prototypes, training runs, infra, research probes,
> production systems.
> Six phases. No more. No checklists you wouldn't actually run.

---

## The whole thing on one screen

```text
   DEFINE  →  PROBE  →  BUILD  →  REVIEW  →  SHIP  →  LEARN
     ↑                                                   │
     └────── update spec / verifier on every surprise ──┘
```

---

## Phase 1 — DEFINE (before any code)

```text
[ ] Write the goal in ONE sentence.
[ ] Ask: can ONE model call do this end-to-end? If yes — stop, use it.
[ ] Write the spec: what it does, what it must NOT do, success criteria.
[ ] Write the verifier: a script that returns pass / fail. Not a description.
[ ] List the invariants that must never break (identity, security, data, $).
[ ] Decide what you will NOT build today. Write it down.
```

> If you can't write the verifier, the goal is wrong. Fix the goal, not the agent.

---

## Phase 2 — PROBE (5 minutes, before trusting the agent)

```text
[ ] Run 3–5 small known-answer test cases of the task on the model.
[ ] Decide which regime you're in:
```

```text
mostly works     inside trained circuits   delegate freely
sometimes works  at the edge               delegate + verifier + spot review
mostly fails     outside                   do it yourself, fine-tune, or rescope
```

Skipping the probe is how you discover, the expensive way, that the model has
a hole exactly where you needed competence.

---

## Phase 3 — BUILD (the agentic loop)

```text
[ ] Hand the spec to agent #1. Ask for a PLAN, not code.
[ ] Read the plan. Reject anything that violates an invariant.
[ ] Agent #1 implements + writes its own tests.
[ ] Agent #2 critiques: edge cases, security, missing invariants.
[ ] Run the verifier. It must pass.
[ ] Run the verifier on a deliberately broken input. It must fail.
```

Cap: 3 concurrent agents max. Above that you rubber-stamp.

A verifier that always passes is a decoration. Step 6 catches the most subtle bugs.

---

## Phase 4 — REVIEW (the human gate)

```text
[ ] Read the test file FIRST. Are tests checking the spec, or the bug?
[ ] Read the diff in reverse — last change to first.
[ ] For every new function: what's the 3am failure mode?
       Silent corruption = reject. Loud crash = acceptable.
[ ] For every deleted line: was it load-bearing?
       If you can't answer in 10 seconds, don't merge.
[ ] Substrate check: identity, money, security, state — never delegated.
```

If you can't defend a decision from first principles, you don't merge.

---

## Phase 5 — SHIP (with sensors)

```text
[ ] Define KILL conditions BEFORE launch — concrete numeric thresholds.
       e.g. "loss spike >2× baseline for 50 steps", "p99 latency >200ms",
            "error rate >1%", "MFU <30%".
[ ] Add observability: logs, metrics, audit trail.
[ ] Add a rollback path. Test it once.
[ ] Deploy. Watch the sensors.
```

> Rule: never add an actuator (anything that changes state) without a matching
> sensor (a way to observe what changed). An agent that acts without verifying
> its action is a bomb with a timer.

---

## Phase 6 — LEARN (per surprise)

```text
[ ] What surprised you? Write one paragraph.
[ ] Does the surprise become a new verifier? Add it now.
[ ] Update the spec to cover the failure mode.
[ ] If you rubber-stamped anything in REVIEW, flag it for tomorrow.
```

> A postmortem that doesn't update a verifier is a diary, not engineering.

---

## When something fails, walk this in order

```text
Rare in pretraining?      → put canonical examples in context
Outside RL coverage?      → build a verifier, loop yourself
Bad context?              → fix the context first; most failures live here
Wrong layer (1.0/2.0/3.0)?→ stop editing prompts when the bug is in the code/data
```

Replaces frustration with a debugging procedure.

---

## What stays yours, always

```text
identity, permissions, money / accounting logic, security boundaries
secrets, distributed state, schema migrations, irreversible deploys
performance-critical architecture, the choice of what to build at all
memory layout, data movement, numerical precision, concurrency
```

The rest is fair game for delegation. Verify everything.

---

## Anti-patterns to catch yourself in

```text
1. Editing the prompt when the bug is in the data.
2. Editing the data when the bug is in the spec.
3. Trusting "I have verified this" from an agent — it hasn't.
4. Building a pipeline that should have been one model call.
5. Adding a fourth concurrent agent. You will regret it.
6. Saying "the agent decided" — replace with "P(output | context) was high."
```

---

## The mantra

> Design the environment so jagged agents can safely produce high-quality work,
> and never let typing speed substitute for understanding.

That's the whole game. Six phases. One screen.
