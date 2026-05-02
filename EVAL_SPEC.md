# EVAL_SPEC.md — Engineering Specification for Agentic Optimization Evaluation

> **Author perspective**: senior AI performance evaluation engineer at a frontier-AI lab, 2026.
> **Project**: ML Optimization Agent (`ml-intern` fork) — an agentic system that recommends, applies, and verifies model-optimization techniques (quantization, kernel selection, parallelism, speculative decoding, scheduling) for training, inference, multimodal, and VLA workloads.
> **Goal of this doc**: make the eval design *defensible to a skeptical reviewer*. Every choice traces to either an information-theoretic bound, a measurement-physics constraint, or an empirically documented failure mode in the published literature.
> **Companion docs**: `PLAN.md` (16-week build plan), `SYSTEM.md` (architecture, Vietnamese), `WORKFLOW.md` (file-ownership rules).
> **Status**: design spec. Implementation tracked in `PLAN.md` Phase 8 (to be added).

---

## Table of contents

0. [Document scope and reading order](#0-document-scope-and-reading-order)
1. [Problem statement: why eval is the load-bearing wall](#1-problem-statement-why-eval-is-the-load-bearing-wall)
2. [Foundational axioms (with derivations and citations)](#2-foundational-axioms)
3. [The 5-tier eval architecture](#3-the-5-tier-eval-architecture)
4. [Statistical framework](#4-statistical-framework)
5. [Metric taxonomy](#5-metric-taxonomy)
6. [Benchmark suite (2026 frontier choices)](#6-benchmark-suite)
7. [Agent-level evaluation](#7-agent-level-evaluation)
8. [Reproducibility envelope](#8-reproducibility-envelope)
9. [Implementation specification](#9-implementation-specification)
10. [Operational runbooks](#10-operational-runbooks)
11. [Risks, anti-patterns, and known limitations](#11-risks-anti-patterns-and-known-limitations)
12. [References](#12-references)

---

## 0. Document scope and reading order

This spec covers **evaluation of optimizations applied by the agent**, **evaluation of the agent itself**, and the **statistical/operational substrate** that makes either claim defensible.

**Reading order by role:**

- **First-time reader** → §1 → §2 → §3 → §11. ~30 min, gives you the full mental model.
- **Implementer** → §3 → §4 → §9. Concrete code spec.
- **Reviewer / skeptic** → §2 → §4 → §6 → §11 → §12. Sources and limitations.
- **On-call** → §10. Runbooks for "the eval is failing, what now".

**Out of scope of this doc:**

- The optimization techniques themselves — covered in `PLAN.md` Phases 2–5.
- Training-side eval (model pre/post-training); we eval *optimization deltas*, not training quality. The two share infrastructure (statistical framework, benchmarks) but differ in subject of measurement.
- Cost/billing dashboards — operational concern, not eval methodology.

---

## 1. Problem statement: why eval is the load-bearing wall

### 1.1 What the agent does, and why eval is harder than it looks

The ml-intern agent ingests a model + workload + hardware spec, runs profiling tools, and recommends optimizations. A naive eval is: "did the optimization improve the target metric?" This is wrong on at least four counts:

1. **The target metric is multi-dimensional.** A quantization that improves throughput 30% but degrades GPQA-Diamond 4 points is a *regression in disguise*. A single-axis eval cannot adjudicate.
2. **Sample noise dominates small effects.** A 0.3% MMLU improvement on a 1k subsample has Minimum Detectable Effect (MDE) ≈ 1.4% — the claim is *literally below the noise floor* (derivation in §4.3).
3. **The agent itself is a system under test.** Even if a recommended optimization is good in isolation, the agent might have selected the wrong tool, mis-diagnosed the bottleneck, or thrashed in a doom-loop. Eval-of-the-agent is distinct from eval-of-the-optimization.
4. **Distribution shift between bench and prod.** Static benchmarks (ShareGPT samples, wikitext-2) do not match production traffic shape (request length distribution, concurrency, arrival burstiness). Optimizations that look great on bench can degrade goodput@SLO in real serving.

A defensible eval system addresses all four. This document specifies how.

### 1.2 The cost of getting it wrong

Two failure modes, asymmetric in cost:

- **False positive (ship a bad optimization)**: silent quality regression reaches users. Recovery cost = rollback + customer trust + on-call burn. Often *not detected for weeks* if eval was bench-only and the regression manifests on real-traffic distribution.
- **False negative (reject a good optimization)**: lost compute savings + dev velocity. Recoverable via re-run, but each false rejection burns ~1 engineer-day.

A single-threshold gate cannot optimize both. A **tiered system** (§3) is the only way to give each gate the right Type-I/Type-II tradeoff for its decision surface.

---

## 2. Foundational axioms

Every downstream design choice is derived from one of these. If you disagree with a downstream choice, find the axiom you disagree with first — that's where the real argument is.

### A1 — Every measurement is a sample from a distribution

A benchmark score is a random variable over (data, seed, hardware nondeterminism, kernel-launch ordering). Reporting a point estimate without quantified uncertainty is mathematically meaningless: you cannot distinguish signal from noise.

**Operational consequence**: every eval result must include a confidence interval (CI) or be marked "indicative-only, do not gate".

**Source**: standard frequentist statistics; for ML-specific evidence of how badly results degrade without CIs, see Henderson et al. 2018 "Deep Reinforcement Learning that Matters" (AAAI), which showed seed variance explained more outcome variance than algorithmic differences.

### A2 — You cannot detect what you cannot measure (MDE)

For sample size *n*, measurement variance σ², and significance level α, the Minimum Detectable Effect at power 1−β is approximately:

```
MDE ≈ (z_{1-α/2} + z_{1-β}) × σ × √(2/n)
```

(For two-sample comparison; paired tests reduce by √2.) If MDE > the claimed effect, the claim is *unfalsifiable from this experiment*.

**Operational consequence**: every eval must publish its MDE and refuse to gate on effects below MDE.

**Source**: classical statistical power analysis (Cohen 1988). For ML application, see Madaan et al. 2024 and Schaeffer et al. 2023 ("Are Emergent Abilities of Large Language Models a Mirage?", NeurIPS 2023 best paper) — both demonstrate how lack of power analysis manufactured false claims at scale.

### A3 — Composite metrics destroy information

Collapsing multiple dimensions into a single "quality score" loses the structure of *which* dimension regressed. A model that gains 2 points on MMLU and loses 5 on GPQA-Diamond should not be reported as "+0 average".

**Operational consequence**: ship the metric *vector*, not a scalar. Aggregation is a downstream policy choice (per-product), not an eval-system choice.

**Source**: multi-criteria decision theory; Pareto-optimality literature. For ML-specific failure mode, see how "average benchmark score" leaderboards (early Open LLM Leaderboard v1) were gamed by overfitting to easy benchmarks while regressing on hard ones — addressed in OpenLLM Leaderboard v2 by reporting per-benchmark scores with CIs.

### A4 — Cross-entropy loss → KL divergence is the most-information-per-FLOP signal

Models are trained to minimize cross-entropy `H(p, q) = H(p) + KL(p || q)`. The optimization-induced delta between baseline and modified model is therefore most directly captured by `KL(p_baseline || p_optimized)` on the logit distribution over a held-out corpus.

KL is sensitive to distributional changes that argmax-based metrics (accuracy, exact-match) provably miss:

- Quantization that preserves top-1 token but shifts distribution → silently breaks temperature-sampling, RLHF-trained reward shape, agentic exploration.
- Pruning that keeps multiple-choice accuracy but flattens entropy → kills creative generation.
- Speculative decoding labeled "exact" that nonetheless shifts draft-model distribution.

**Sample efficiency**: detecting a given distributional change via KL on logits typically needs ~10× fewer samples than detecting equivalent task-accuracy regression, because KL uses the full probability vector instead of a single argmax.

**Operational consequence**: KL on a held-out distribution-matched corpus is the cheapest high-signal quality metric. Always run it.

**Source**: information theory (Cover & Thomas, *Elements of Information Theory*). For empirical demonstration on quantization: Dettmers et al. 2022 (LLM.int8(), NeurIPS) and Lin et al. 2023 (AWQ, MLSys 2024) both show KL/PPL detect regressions before downstream task accuracy does.

### A5 — Throughput and latency are adversarial (Little's Law)

By Little's Law (Little 1961, *Operations Research*): `L = λ × W`, where L = mean concurrency, λ = throughput, W = mean response time. For a system at fixed L (saturated), increasing λ requires proportionally increasing W. Throughput and latency are *not* independent metrics — optimizing one degrades the other near saturation.

**Operational consequence**: never report throughput without the corresponding latency distribution. The user-relevant single number is **goodput@SLO** = throughput × P(latency ≤ SLO), defined formally in DistServe (Zhong et al. 2024, OSDI).

### A6 — User experience is dominated by the tail, not the mean

Mean latency hides a 1%-of-users-see-10×-latency failure mode. For real-time control (VLA), p999 latency violations cause control-loop failures — *safety-critical*. Median is no better; both are central-tendency estimators.

**Operational consequence**: serving evals report p50, p95, p99, p999 *and* jitter (IQR or std-dev). Means are decorative.

**Source**: queuing theory (Kleinrock); production SRE practice (Beyer et al., *Site Reliability Engineering*, O'Reilly 2016). For VLA: RT-2 (Brohan et al. 2023), OpenVLA (Kim et al. 2024), π0 (Black et al. 2024) all report jitter + p999 because robotics requires bounded worst-case.

### A7 — Benchmarks decay (saturation + contamination)

Two mechanisms make benchmarks lose discriminating power over time:

- **Saturation**: when top models score >90%, all signal is in the top-quintile noise band. MMLU is saturated for frontier models. HumanEval is saturated.
- **Contamination**: training corpora include benchmark text; gains reflect memorization, not capability. Heavily contaminated: HumanEval, MBPP, GSM8K (all on the public web pre-2023 cutoffs).

**Operational consequence**: benchmark choice has a half-life. The 2026 frontier replacements are listed in §6; expect to revisit annually.

**Source**: contamination evidence — Brown et al. 2020 (GPT-3, original n-gram contamination analysis); Sainz et al. 2023 ("NLP Evaluation in trouble", EMNLP); Schaeffer et al. 2023. Saturation evidence — every modern leaderboard.

### A8 — The agent is itself an artifact-under-eval

The system's *deliverable* is the agent's recommendation, not the underlying model. Capability composition is non-monotonic: strong MMLU + strong HumanEval ≠ strong agentic tool-use. (BFCL-v3 vs MMLU correlation across open models is r ≈ 0.4–0.6 — they measure different things.)

**Operational consequence**: a separate eval surface for the agent's *decisions* (diagnosis, tool routing, recommendation quality, loop efficiency) is mandatory. Model-only eval is necessary but not sufficient.

**Source**: τ-bench (Yao et al. 2024, Sierra), BFCL-v3 (Patil et al. 2024, Berkeley), GAIA (Mialon et al. 2023, Meta), MLE-bench (OpenAI 2024), SWE-bench (Jimenez et al. 2024, ICLR) — the entire agentic-eval literature exists because model evals don't predict agent performance.

### A9 — Reproducibility is binary, not gradient

Without seed control, dependency pinning, and hardware fingerprinting, an eval is not reproducible — it cannot be re-run to defend a claim. cuBLAS version, CUDA version, GPU SKU (even within "H100"), driver version, and kernel nondeterminism shift PPL by >0.5% in published case studies.

**Operational consequence**: every eval run produces an *envelope* (seed, dep hashes, GPU fingerprint, data hashes, env vars). Runs without envelopes are advisory only; they cannot gate.

**Source**: NeurIPS Reproducibility Checklist (mandatory since 2019); MLPerf benchmark rules (specifies compiler flags); Pineau et al. 2021 ("Improving reproducibility in machine learning research", JMLR).

### A10 — The eval system itself must be evaluated (eval-of-eval)

Without measuring eval quality (test-retest correlation, false-positive rate on null changes, false-negative rate on planted regressions), you cannot trust the gates. An eval system that has never failed has either never been stressed or is silently biased.

**Operational consequence**: every release of the eval system runs against (a) a *null fixture* (identical model A/B comparison — should never gate-fail) and (b) *planted-regression fixtures* (known bad models — should always gate-fail). Track FPR/FNR over time.

**Source**: classical software-testing principle (Beizer, *Software Testing Techniques*). For ML-specific application, see how leaderboard noise was diagnosed via test-retest in Dehghani et al. 2021 ("The benchmark lottery").

---

## 3. The 5-tier eval architecture

### 3.1 Why tiers (re-derivation)

Three forces:

1. **Cost asymmetry**: full eval (all benchmarks × all seeds × full corpora × real-traffic shadow) costs $1k–$10k per change. Cheap eval (smoke test) costs $0.10. Doing the expensive one on every change is infeasible.
2. **Signal asymmetry**: different failure modes have different *base rates* and different *detection costs*. Catastrophic breakage is rare but cheap to detect; subtle distribution shift is common but expensive to detect.
3. **Decision asymmetry**: different decisions have different FN/FP cost ratios. A merge gate can be permissive (cheap to revert); a production deploy gate must be strict (expensive to revert).

Tiering aligns the three: spend each marginal dollar where it most reduces P(undetected regression reaches users).

### 3.2 Tier definitions

Each tier specifies: **what failure mode it catches**, **gate decision it informs**, **cost budget**, **cadence**, **acceptance criterion**.

#### T0 — Smoke (catch catastrophic breakage)

- **Catches**: model fails to load, tokenizer mismatch, output shape wrong, NaN/Inf, runaway generation length, immediate OOM.
- **Gate**: pre-merge to feature branch (every commit touching optimization code).
- **Cost budget**: <$0.10 per run, <2 min wall-clock.
- **Cadence**: every CI run.
- **Method**: 50 fixed prompts → check output exists, no NaN, output length < 4× input, KL(baseline‖optimized) < 10 (wide).
- **Acceptance**: 100% pass.
- **Statistical rigor required**: none — these are deterministic correctness checks.

#### T1 — Calibrated quality (catch distributional shift)

- **Catches**: silent quality regression — KL divergence shift, perplexity regression, downstream-task accuracy degradation outside CI.
- **Gate**: pre-merge to main branch.
- **Cost budget**: $5–$50 per run, 10–60 min wall-clock.
- **Cadence**: every PR to main; every nightly on main.
- **Method**: KL-on-logits (held-out, 5k tokens) + 2–4 capability benchmarks at sample sizes that meet pre-registered MDE; bootstrap CIs; Holm-Bonferroni correction.
- **Acceptance**: see §4.6 (the conjunctive verdict rule).
- **Statistical rigor**: full — CIs, MDE pre-registered, multiple-comparisons corrected.

#### T2 — Serving (catch production-physics regression)

- **Catches**: tail latency regression, OOM under load, throughput collapse near saturation, scheduling pathology, KV-cache churn.
- **Gate**: pre-merge if optimization touches the serving path (kernel selection, batching, scheduler, KV cache); otherwise advisory.
- **Cost budget**: $20–$100 per run, 30–90 min wall-clock.
- **Cadence**: nightly on main; on-demand for serving-path PRs.
- **Method**: load test against fixed traffic shape (Poisson arrivals, length distribution matched to production sample) on real serving framework (vLLM/SGLang/TRT-LLM); measure goodput@SLO, p50/p95/p99/p999 TTFT/TBT/ITL, GPU utilization, OOM count.
- **Acceptance**: goodput@SLO not worse than baseline by more than the lower 95% CI; no p999 violation worse than 1.5× baseline; zero OOM.
- **Statistical rigor**: full — bootstrap CIs on percentiles (which are non-Gaussian).

#### T3 — Shadow (catch bench-vs-prod distribution shift)

- **Catches**: optimizations that look good on bench but degrade on real traffic (length distribution mismatch, prompt-style mismatch, multi-turn-context behavior).
- **Gate**: pre-deploy to canary.
- **Cost budget**: $100–$500 per run, hours-to-1-day wall-clock.
- **Cadence**: per release-candidate; weekly on main.
- **Method**: replay a captured slice of real production traffic (anonymized, sampled) through baseline and optimized side-by-side; compare KL on outputs, paired comparisons via McNemar test on agreement, latency distributions on the real shape.
- **Acceptance**: paired KL within pre-registered band; latency distributions within tier-2-style envelope on real (not synthetic) traffic shape.
- **Statistical rigor**: full — paired tests (McNemar for binary outcomes, paired bootstrap for continuous).

#### T4 — Canary (catch what only real users surface)

- **Catches**: cosmetic regressions, alignment edge cases, long-tail user complaint patterns, unforeseen interactions with downstream consumers.
- **Gate**: pre-full-rollout.
- **Cost budget**: $200–$1000 + user-exposure risk; hours-to-days wall-clock.
- **Cadence**: per release.
- **Method**: route 1–5% of real traffic to optimized variant; monitor SLO compliance, error rate, user-feedback signals (thumbs-down, regenerate-rate, session-abandon-rate); pre-registered guardrail metrics with auto-rollback thresholds.
- **Acceptance**: no guardrail violation over a pre-registered observation window (typically ≥ 24h or ≥ N requests for power).
- **Statistical rigor**: sequential testing (e.g., always-valid p-values via Howard et al. 2021 "Time-uniform Chernoff bounds") to enable safe early stopping.

### 3.3 Meta layers (cross-tier)

These are not tiers — they apply *across* tiers.

- **M1 — Reproducibility envelope** (§8): every result tagged with seeds, hashes, hardware fingerprint.
- **M2 — Eval-of-eval** (§10.4): null fixtures and planted-regression fixtures run continuously; FPR/FNR tracked.
- **M3 — Cost telemetry**: every run reports compute cost; total eval spend monitored to prevent runaway.

### 3.4 Tier summary table

| Tier | Catches | Gate | Cost/run | Cadence | Stats rigor |
|------|---------|------|----------|---------|-------------|
| T0 | Catastrophic breakage | Pre-merge to branch | <$0.10 | Every commit | None (deterministic) |
| T1 | Distributional/quality shift | Pre-merge to main | $5–$50 | Every PR + nightly | Full (CIs + MDE + Holm) |
| T2 | Serving-physics regression | Pre-merge if serving-path | $20–$100 | Nightly + on-demand | Full (percentile CIs) |
| T3 | Bench-vs-prod shift | Pre-canary | $100–$500 | Per RC + weekly | Full (paired tests) |
| T4 | Real-user impact | Pre-full-rollout | $200–$1000 + risk | Per release | Sequential testing |

---

## 4. Statistical framework

### 4.1 Why bootstrap, not parametric

ML metrics are routinely non-Gaussian:

- **Perplexity**: roughly log-normal, heavy upper tail.
- **Accuracy**: bounded in [0, 1], Beta-distributed near edges.
- **Latency**: heavy-tailed (often log-normal or Pareto in serving systems).

Parametric (t-distribution) CIs assume normality and produce miscalibrated coverage on these distributions. Bootstrap (Efron 1979, *Annals of Statistics*) is distribution-free and gives correct coverage at typical sample sizes.

**Implementation**: percentile bootstrap for n ≥ 30; BCa (bias-corrected and accelerated, Efron 1987) for n ≥ 100 when the metric is potentially biased (e.g., percentile estimators).

### 4.2 Why Holm-Bonferroni for multiple comparisons

If you test k metrics each at α = 0.05 independently, the family-wise error rate is `1 − (1 − 0.05)^k`:

| k | FWER | Probability of ≥1 false positive |
|---|------|----------------------------------|
| 1 | 0.05 | 5% |
| 5 | 0.226 | 23% |
| 10 | 0.401 | 40% |
| 20 | 0.642 | 64% |

ML evals routinely involve 10–20 metric comparisons (PPL, KL, MMLU, GPQA, MATH, HumanEval, BFCL, plus latencies). Without correction, "significant" results are mostly noise.

**Holm-Bonferroni** (Holm 1979, *Scandinavian Journal of Statistics*) controls FWER at α while being uniformly more powerful than vanilla Bonferroni — it strictly dominates. Use it.

**Algorithm**: sort p-values ascending: p₁ ≤ p₂ ≤ … ≤ pₖ. Reject Hᵢ if pᵢ ≤ α / (k − i + 1). Stop at first non-rejection.

**When to use Benjamini-Hochberg (FDR) instead**: discovery contexts (you're scanning for hypotheses, willing to accept some false positives in exchange for power). Not appropriate for go/no-go gates — use Holm.

### 4.3 MDE pre-registration

Before running any eval, compute and publish the MDE the experiment can detect at α = 0.05, power = 0.8.

**Worked examples** (these are illustrative — exact numbers depend on benchmark variance, which should be measured from baseline runs):

- **Wikitext-2 perplexity, n = 100 documents**: MDE ≈ 0.11 PPL units. Any claim of "0.05 PPL improvement" is unfalsifiable — *do not gate on it*.
- **Full MMLU, n ≈ 14,000 questions**: MDE ≈ 0.36% accuracy.
- **MMLU 1k subsample**: MDE ≈ 1.4% accuracy. Sub-1.4% claims invalid.
- **GPQA-Diamond, n = 198 questions**: MDE ≈ 5–7% (small benchmark, high variance). Not appropriate for fine-grained discrimination — use as a coarse capability filter only.

**Operational rule**: every benchmark in T1+ ships with a pre-computed MDE table. If a claimed effect is below MDE, the gate refuses to consider it (auto-flagged as "below detection threshold").

**How to estimate σ**: run baseline 5–10 times (seed-varied) and take sample std-dev. This becomes the σ in the MDE formula (§A2).

### 4.4 Paired vs unpaired tests

When comparing baseline vs optimized on the *same* questions/prompts, use paired tests — they cancel question-difficulty variance and need ~2× fewer samples than unpaired for the same MDE.

- **Continuous metrics** (KL, PPL, latency on same prompt): paired bootstrap on differences.
- **Binary metrics** (correct/incorrect on same question): McNemar's test (McNemar 1947) on the discordant-pair count.

Most eval comparisons are paired by construction — exploit this.

### 4.5 Sequential testing for canary

Canary monitoring is fundamentally a sequential decision: you observe data over time and want to stop early if a problem appears. Naive repeated p-value testing inflates false-positive rate (the "peeking problem").

Use **always-valid p-values** (Howard et al. 2021, "Time-uniform Chernoff bounds for the mean of bounded variables") or mixture sequential probability ratio tests (mSPRT, Kohavi et al. industry literature). Both allow continuous monitoring without α inflation.

### 4.6 The verdict rule (conjunctive ACCEPT)

A T1+ eval ACCEPTs an optimization iff **all** of the following hold:

1. **Quality gate**: no quality metric regresses outside its CI by more than the pre-registered tolerance, with Holm-Bonferroni applied across all quality metrics.
2. **Performance gate**: the target performance metric (e.g., goodput@SLO, training MFU) improves by at least the pre-registered MDE, with CI lower bound > 0.
3. **No-surprise gate**: no non-target metric regresses by more than its pre-registered guardrail (e.g., latency p99 doesn't double when target was throughput).
4. **Reproducibility gate**: the run produced a complete envelope (§8); seed-replication on a second run confirms the result is within CI.

The conjunction is intentional — any single-axis ACCEPT is a known failure mode (you'd ship throughput-up / quality-down). RIPPLE this rule through every gating decision.

### 4.7 Effect size, not just significance

Report Cohen's *d* (for continuous) or odds ratios (for binary) alongside p-values. A statistically significant 0.1% gain on a 100k-sample benchmark is operationally meaningless. Pre-register a *minimum effect size of interest* (MEI) per metric — claims below MEI are reported as "detected but operationally negligible".

**Source**: APA Publication Manual (effect-size reporting required since 2010); for ML, Card et al. 2020 "With Little Power Comes Great Responsibility" (EMNLP).

---

## 5. Metric taxonomy

Three metric classes; each has a distinct role and statistical handling.

### 5.1 Quality metrics (model output fidelity)

**Primary** (always run in T1):

- **KL divergence** on logits, baseline vs optimized, on a held-out distribution-matched corpus. Most-information-per-FLOP signal (§A4).
  - Sample size: 5k–50k tokens depending on tier.
  - Aggregation: token-mean KL + token-max KL (catches localized blow-ups).
  - Why both: mean KL can hide a few catastrophic tokens; max KL alone is noisy.
- **Perplexity** on held-out corpus matched to model's training distribution. Cheap, well-understood, comparable across literature.

**Capability** (run subset matched to optimization_target):

- See §6 for the full benchmark table.
- Always include at least one *contamination-resistant* benchmark (GPQA-Diamond, LiveCodeBench, MMLU-Pro).

**Distributional shape** (cheap, run always):

- **Output entropy** distribution: did the optimization flatten/sharpen the output distribution?
- **Top-k token agreement rate** with baseline.
- **Length distribution** of generations: did the optimization induce length drift?

### 5.2 Performance metrics (compute/serving)

**Training**:

- **MFU** (Model FLOPs Utilization) — ratio of achieved to theoretical peak FLOPs. Reference: Chowdhery et al. 2022 (PaLM paper) for the canonical definition; PaLM achieved 46.2% MFU as a benchmark figure.
- **Tokens/sec/GPU**.
- **Memory peak** (must include activation memory under chosen recompute strategy).
- **Time-to-target-loss** (when comparing training-time optimizations).

**Inference / serving** (DistServe taxonomy, Zhong et al. 2024):

- **TTFT** (Time-To-First-Token) — prefill latency. Critical for chat UX.
- **TBT** (Time-Between-Tokens) — inter-token decode latency.
- **ITL** (Inter-Token Latency) — alias for TBT in some literature.
- **E2E latency** — full request completion time.
- **Throughput** — tokens/sec aggregate.
- **Goodput@SLO** = throughput × P(latency ≤ SLO) — the user-relevant single number (§A5).

**For all latency metrics**: report the *distribution* (p50, p95, p99, p999) plus jitter (IQR or std-dev). Means are decorative (§A6).

**VLA / real-time** (additional):

- **Control-loop frequency** (Hz) achieved.
- **p999 jitter** within control window.
- **Worst-case action latency** — for safety-critical, this matters more than any percentile.

### 5.3 Resource metrics

- **GPU memory peak** (including activation, KV cache, optimizer state).
- **GPU utilization** (sustained, not peak).
- **Power draw** (W) — increasingly relevant for cost/ton-CO₂ accounting.
- **Cost per 1M tokens** at the measured load — the deployment-relevant economic metric.

### 5.4 Safety / alignment metrics

These are *never* trade-offable against performance — they are gates, not optimization targets.

- **HarmBench** (Mazeika et al. 2024) — refusal robustness.
- **StrongREJECT** (Souly et al. 2024) — jailbreak resistance.
- **XSTest** (Röttger et al. 2024) — over-refusal detection (false positive on benign).
- **JailbreakBench** (Chao et al. 2024) — adversarial prompt suite.

**Operational rule**: any optimization that degrades safety metrics outside CI fails the gate, regardless of performance gains. No exceptions.

---

## 6. Benchmark suite

### 6.1 Why these benchmarks (2026 frontier)

Benchmark choice has a half-life. The 2024–2025 wave of benchmarks was specifically designed to address saturation + contamination on 2026-class models. Using MMLU + HumanEval in 2026 is like using ImageNet in 2020 — the leaderboard is flat.

Selection criteria:
1. **Discriminating power on current frontier models** (top model not above 90%).
2. **Contamination resistance** (recent, held-out, or by-construction novel).
3. **Construct validity** (measures what its name claims).
4. **Open and stable** (won't disappear or change between runs).

### 6.2 Benchmark table

| Benchmark | Domain | n | Why this one (2026) | Source |
|-----------|--------|---|---------------------|--------|
| **MMLU-Pro** | Knowledge + reasoning | ~12k | 10-option (vs 4 in MMLU), harder distractors, less contaminated. Discriminates 2026 frontier. | Wang et al. 2024, TIGER-Lab |
| **GPQA-Diamond** | PhD-level science | 198 | "Google-proof" — designed contamination-resistant. Current gold for hard reasoning. Small n → coarse signal only. | Rein et al. 2023 |
| **MATH-500** | Math reasoning | 500 | Subset of MATH (Hendrycks et al. 2021) with stable difficulty. AIME-2024+ is the held-out variant. | Lightman et al. 2023 (PRM800K) |
| **AIME 2024+** | Competition math | 30/yr | Yearly held-out by construction. Contamination-immune for current year. | Math Olympiad |
| **HumanEval+ / MBPP+** | Code | extended | EvalPlus (Liu et al. 2023) adds 80×+ test cases — exposes brittle code that passes original tests. | Liu et al. 2023, NeurIPS |
| **LiveCodeBench** | Code | rolling | Problems indexed by date — use only problems newer than model's cutoff. Contamination by construction impossible. | Jain et al. 2024 |
| **RULER** | Long context | configurable | Only long-context benchmark whose difficulty scales with length. Plain NIAH is trivial for 2026 models. | Hsieh et al. 2024, NVIDIA |
| **BFCL-v3** | Function calling | ~2k | Multi-turn, parallel calls, irrelevance-detection. v1/v2 saturated. | Patil et al. 2024, Berkeley |
| **τ-bench** | Agent dialogue | 165 | Customer-service flows with policy adherence. Hardest agent benchmark — top models <60% pass^4. | Yao et al. 2024, Sierra |
| **SWE-bench-Verified** | Code agent | 500 | Human-verified subset of SWE-bench (full has noise). Real GitHub issues. | OpenAI 2024 (verified subset) |
| **HarmBench** | Safety (refusal) | 510 | Standardized adversarial behaviors. Use for any safety regression check. | Mazeika et al. 2024 |
| **StrongREJECT** | Safety (jailbreak) | 313 | Calibrated adversarial prompts; correlates with human harm judgment. | Souly et al. 2024 |
| **XSTest** | Over-refusal | 450 | Detects false positives — model refusing benign prompts. | Röttger et al. 2024 |

### 6.3 Benchmark routing by optimization_target

The agent's `optimization_target` determines which benchmarks T1 runs. Pre-registered mapping:

| optimization_target | Required T1 benchmarks |
|---------------------|------------------------|
| `throughput` | KL + MMLU-Pro + GPQA-D + safety suite |
| `latency` | KL + MMLU-Pro + GPQA-D + safety suite |
| `memory` | KL + MMLU-Pro + safety suite |
| `quality` | KL + full quality matrix (MMLU-Pro, GPQA-D, MATH-500, HumanEval+, BFCL-v3) + safety |
| `multimodal` | KL + MMMU + MathVista + safety |
| `vla` | KL + simulator-rollout success rate + p999 jitter + safety |
| `agentic` | All quality + BFCL-v3 + τ-bench + SWE-bench-Verified + safety |

Justification: minimize cost while ensuring the dimension being optimized is bounded by complementary axes. E.g., throughput-targeted optimizations rarely hurt math reasoning specifically — but they can shift distributional behavior, which KL + general-knowledge MMLU-Pro catches.

### 6.4 Sample sizing rules

- Use *full* benchmark when computationally feasible (cost < tier budget).
- If subsampling, the subsample size must support the pre-registered MDE for the claim being made.
- *Stratified* subsampling (preserve category proportions) only — uniform random subsampling adds variance unnecessarily.
- Subsample seed: pre-registered and fixed across baseline / optimized to enable paired comparison.

### 6.5 Contamination defense

- **For benchmarks with public test sets**: assume contamination; treat results as upper bounds.
- **For benchmarks with held-out test sets (LiveCodeBench, AIME)**: filter to problems published *after* the model's training cutoff.
- **Active monitoring**: periodically run contamination probes (Carlini et al. 2021, "Extracting Training Data from Large Language Models" methodology) on suspect benchmarks.

---

## 7. Agent-level evaluation

### 7.1 Why a separate eval surface

Per axiom A8: the agent's deliverable is the *recommendation*, not the underlying model. Evaluating only the model leaves four agent-specific failure modes uncaught:

1. **Mis-diagnosis**: agent identifies wrong bottleneck (e.g., recommends quantization when the actual bottleneck is KV-cache size).
2. **Tool-routing error**: agent picks a tool that doesn't apply to the optimization_target.
3. **Bad recommendation**: agent's chosen optimization is inferior to alternatives on the actual Pareto frontier.
4. **Loop pathology**: agent enters doom-loop, fails to terminate, burns token budget.

### 7.2 Four agent eval surfaces

#### S1 — Diagnosis accuracy

- **Method**: gold-labeled scenario suite — synthetic profiles paired with expert-labeled root-cause bottleneck.
- **Metric**: top-1 and top-3 diagnosis accuracy with CIs.
- **Suite size**: ≥ 100 scenarios spanning (training/inference) × (compute/memory/IO/comm-bound) × (model-size buckets).
- **Adversarial subset**: ≥ 20 scenarios with red-herring signals (e.g., low GPU utilization that's actually due to data-loading, not compute).

#### S2 — Tool routing

- **Method**: given (profile, optimization_target), check that agent invokes tools in the pre-registered correct subset of TOOL_SUITES.
- **Metric**: precision (no inappropriate tools) + recall (all required tools invoked).
- **Failure modes to probe**: invoking quantization tools when target is `latency` and bottleneck is comm-bound (wrong); skipping kernel-selection when target is `throughput` and bottleneck is compute-bound (incomplete).

#### S3 — Recommendation Pareto-quality

- **Method**: for a held-out scenario set with known Pareto frontier (computed offline by exhaustive search over a small action space), check if agent's recommendation is on or near the frontier.
- **Metric**: distance from Pareto frontier in (quality, throughput, memory) space, normalized.
- **Why this matters**: an agent can be locally correct (each individual recommendation is fine) but globally suboptimal (it misses a much better Pareto point that requires combining techniques).

#### S4 — Loop efficiency

- **Method**: track per-task token spend, n_iterations, n_tool_calls, time-to-recommendation.
- **Metric**: distribution (median + p95) on a fixed task suite. Compare across agent versions.
- **Failure mode**: doom-loops, redundant tool invocations, premature termination.
- **Hard guardrail**: max token budget per task; max iterations; auto-abort.

### 7.3 Adversarial agent scenarios

A small set of intentionally-hard scenarios that probe known agent failure modes:

- **Conflicting signals**: profile shows both compute-bound (high MFU) and memory-bound (high HBM utilization) signatures. Correct response: ask for clarification or run additional diagnostic — *not* default to one.
- **Unreliable tool output**: inject corrupted profiler output. Correct response: detect and re-run, not propagate.
- **Hardware spec absent**: profile lacks hardware info. Correct response: refuse to recommend hardware-specific optimizations, not hallucinate.
- **Out-of-distribution model**: model architecture not in HARDWARE_SPECS table. Correct response: degrade to general advice + flag for human review.

Pass rate on the adversarial suite is a separate gate from S1–S4; it gates the agent release, not individual optimization decisions.

### 7.4 Eval cadence for the agent

- **S1, S2, S4**: every PR to agent code (`agent/optimization/**`).
- **S3**: nightly (more expensive — requires Pareto computation).
- **Adversarial suite**: pre-release.

---

## 8. Reproducibility envelope

### 8.1 What must be captured

Every eval run produces an envelope with:

```
{
  "run_id": "uuid",
  "timestamp": "ISO-8601 with TZ",
  "git_commit_sha": "...",
  "git_dirty": false,
  "code_dependencies_hash": "sha256 of pinned requirements.txt",
  "uv_lockfile_hash": "sha256 of uv.lock",
  "python_version": "3.x.y",
  "cuda_version": "12.x.y",
  "cudnn_version": "...",
  "driver_version": "...",
  "gpu_model": "H100-SXM5-80GB",
  "gpu_uuid": "...",
  "n_gpus": 8,
  "host_fingerprint": "sha256 of (cpu, mem, kernel, libc)",
  "data_hashes": {"benchmark_name": "sha256"},
  "seeds": {"torch": 42, "numpy": 42, "python": 42, "cuda": 42},
  "deterministic_mode": true,
  "env_vars": {"CUBLAS_WORKSPACE_CONFIG": "...", "TF32": "off"},
  "framework_versions": {"torch": "...", "transformers": "...", "vllm": "..."},
  "tier": "T1",
  "optimization_target": "throughput",
  "baseline_config": {...},
  "optimized_config": {...},
  "results": [...]
}
```

### 8.2 Determinism settings

- `torch.use_deterministic_algorithms(True)` where supported.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (required for cuBLAS determinism on CUDA ≥ 10.2).
- TF32 off for eval (tradeoff: deterministic but slower; eval should not be the bottleneck).
- For inherently nondeterministic kernels (e.g., scatter operations), document the residual variance and report it as part of σ in MDE.

### 8.3 Replay protocol

To verify a result, a third party must be able to:

1. Check out the recorded git commit.
2. `uv sync` against the recorded lockfile hash.
3. Run on hardware matching the recorded fingerprint (or equivalent).
4. Reproduce results within the recorded CI.

If steps 1–4 cannot be performed, the result is not reproducible — and per A9, not gate-eligible.

### 8.4 Source of practice

- NeurIPS Reproducibility Checklist (mandatory since 2019, see Pineau et al. 2021, JMLR).
- MLPerf benchmark rules (specifies compiler flags, kernels, batch sizes).
- Anthropic, OpenAI, DeepMind public release artifacts include env hashes for major releases.

---

## 9. Implementation specification

### 9.1 Directory layout (owned paths only)

Per `CLAUDE.md` zero-diff invariant, all eval code lives in owned paths:

```
agent/
  eval/                          # NEW — eval substrate
    __init__.py
    stat_utils.py                # bootstrap CIs, MDE, Holm-Bonferroni
    verdict.py                   # conjunctive ACCEPT rule
    envelope.py                  # reproducibility envelope capture
    tiers/
      __init__.py
      t0_smoke.py
      t1_quality.py
      t2_serving.py
      t3_shadow.py
      t4_canary.py
    metrics/
      __init__.py
      kl_divergence.py
      perplexity.py
      goodput.py
      latency.py
      mfu.py
    benchmarks/
      __init__.py
      mmlu_pro.py
      gpqa_diamond.py
      math_500.py
      humaneval_plus.py
      bfcl_v3.py
      tau_bench.py
      ruler.py
      harmbench.py
      strongreject.py
      xstest.py
    agent_eval/                  # §7
      __init__.py
      diagnosis.py               # S1
      tool_routing.py            # S2
      pareto.py                  # S3
      loop_efficiency.py         # S4
      adversarial.py
    fixtures/
      null_change.py             # eval-of-eval: null fixture
      planted_regressions.py     # eval-of-eval: known-bad fixtures
    cli.py                       # `ml-intern-eval` entrypoint
configs/
  eval/
    tier_thresholds.yaml         # MDE, MEI, gating thresholds per metric
    benchmark_suites.yaml        # routing by optimization_target
    serving_load_profiles.yaml   # T2 traffic shapes
tests/
  optimization/
    eval/
      test_stat_utils.py
      test_verdict.py
      test_envelope.py
      test_tier_t0.py
      test_tier_t1.py
      ...
```

### 9.2 Key dataclasses (extend, don't modify)

The existing `Experiment` dataclass in `agent/optimization/` is extended with eval-specific fields:

```python
# agent/optimization/experiment.py (extend, do not modify upstream)
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class MetricResult:
    name: str
    value: float
    ci_low: float
    ci_high: float
    ci_method: str          # "bootstrap_percentile" | "bootstrap_bca" | "mcnemar"
    n_samples: int
    seed: int
    mde: float              # Minimum Detectable Effect at α=0.05, power=0.8
    mei: float              # Minimum Effect of Interest (pre-registered)

@dataclass
class TierResult:
    tier: str               # "T0" | "T1" | "T2" | "T3" | "T4"
    metrics: list[MetricResult]
    p_values_raw: dict[str, float]
    p_values_holm: dict[str, float]
    verdict: str            # "ACCEPT" | "REJECT" | "INCONCLUSIVE"
    verdict_reasons: list[str]
    cost_usd: float
    wall_clock_s: float

@dataclass
class ReproEnvelope:
    run_id: str
    timestamp: datetime
    git_commit_sha: str
    git_dirty: bool
    deps_hash: str
    python_version: str
    cuda_version: str
    cudnn_version: str
    driver_version: str
    gpu_model: str
    gpu_uuids: list[str]
    n_gpus: int
    host_fingerprint: str
    data_hashes: dict[str, str]
    seeds: dict[str, int]
    deterministic_mode: bool
    env_vars: dict[str, str]
    framework_versions: dict[str, str]

@dataclass
class EvalRun:
    envelope: ReproEnvelope
    optimization_target: str
    baseline_config: dict
    optimized_config: dict
    tier_results: list[TierResult]
    overall_verdict: str
    overall_reasons: list[str]
```

### 9.3 The verdict function (conjunctive ACCEPT)

```python
# agent/eval/verdict.py
def compute_verdict(tier_result: TierResult, thresholds: dict) -> tuple[str, list[str]]:
    """Conjunctive ACCEPT rule (§4.6). Returns (verdict, reasons)."""
    reasons = []

    # Gate 1: Quality — no quality metric regresses outside CI by more than tolerance.
    for m in tier_result.metrics:
        if m.name in thresholds["quality_metrics"]:
            tol = thresholds["quality_tolerance"][m.name]
            if m.ci_high < -tol:  # CI lies entirely below tolerance band → real regression
                reasons.append(f"REJECT: {m.name} regressed: CI={m.ci_low:.4f}..{m.ci_high:.4f} < -{tol}")

    # Gate 2: Performance — target metric improves by ≥ MDE, CI lower bound > 0.
    target = thresholds["target_metric"]
    target_m = next((m for m in tier_result.metrics if m.name == target), None)
    if target_m is None:
        reasons.append(f"REJECT: target metric {target} not measured")
    elif target_m.ci_low <= 0:
        reasons.append(f"REJECT: {target} improvement CI lower bound {target_m.ci_low:.4f} ≤ 0")
    elif target_m.value < target_m.mde:
        reasons.append(f"REJECT: {target} effect {target_m.value:.4f} < MDE {target_m.mde:.4f}")

    # Gate 3: No-surprise — guardrails on non-target metrics.
    for m in tier_result.metrics:
        if m.name in thresholds["guardrails"]:
            limit = thresholds["guardrails"][m.name]
            if m.ci_high < limit:
                reasons.append(f"REJECT: guardrail violated: {m.name} CI_high={m.ci_high:.4f} < {limit}")

    # Gate 4: Reproducibility — handled at envelope-capture time, not here.

    if not reasons:
        return "ACCEPT", ["All gates passed."]
    return "REJECT", reasons
```

### 9.4 Configuration (pre-registered thresholds)

```yaml
# configs/eval/tier_thresholds.yaml — pre-registered per metric
T1:
  target_metric: "goodput_at_slo"   # or as overridden by optimization_target
  quality_metrics:
    - kl_divergence_mean
    - kl_divergence_max
    - perplexity_wikitext2
    - mmlu_pro_accuracy
    - gpqa_diamond_accuracy
  quality_tolerance:                 # max acceptable regression per metric
    kl_divergence_mean: 0.05
    kl_divergence_max: 0.5
    perplexity_wikitext2: 0.5        # PPL units
    mmlu_pro_accuracy: 0.005         # 0.5%
    gpqa_diamond_accuracy: 0.02      # 2% (tighter would be below MDE)
  guardrails:
    latency_p99_ms: 1.5              # ratio: optimized/baseline ≤ 1.5
    memory_peak_gb: 1.1              # ratio
  alpha: 0.05
  power: 0.80
  multiple_comparisons: "holm"
```

### 9.5 CLI surface

```bash
# Run a tier on a model pair
ml-intern-eval run --tier T1 \
  --baseline ./checkpoints/baseline \
  --optimized ./checkpoints/optimized \
  --target throughput \
  --output ./eval_runs/

# Inspect a result
ml-intern-eval show ./eval_runs/<run_id>/

# Replay (verify reproducibility)
ml-intern-eval replay ./eval_runs/<run_id>/

# Run agent-level eval
ml-intern-eval agent --suite diagnosis --output ./agent_evals/
```

### 9.6 Integration with PLAN.md phases

A new `Phase 8 — Eval substrate` should be added to PLAN.md, sequenced *before* Phase 4 (quantization) — because Phase 4 will use the eval substrate to validate every quantization claim.

Phase 8 deliverables (suggested):

| Step | Owned path | Acceptance |
|------|-----------|------------|
| 8.1 | `agent/eval/stat_utils.py` + tests | Bootstrap CIs, MDE, Holm. `pytest tests/optimization/eval/test_stat_utils.py` green. |
| 8.2 | `agent/eval/envelope.py` + tests | Captures full envelope; `replay` works. |
| 8.3 | `agent/eval/verdict.py` + tests | Conjunctive ACCEPT rule. Null-fixture passes; planted-regression fixtures all REJECT. |
| 8.4 | `agent/eval/tiers/t0_smoke.py` | <2 min on H100, deterministic. |
| 8.5 | `agent/eval/tiers/t1_quality.py` | KL + 2 benchmarks; CIs; verdict. |
| 8.6 | `agent/eval/tiers/t2_serving.py` | Goodput@SLO + percentile CIs against vLLM. |
| 8.7 | `agent/eval/agent_eval/diagnosis.py` | S1 ≥ 80% top-1 on scenario suite. |

Existing `MC-1` (evaluate_model_quality with lm-eval-harness MMLU) should be **replaced** by T1 — current MC-1 reports a single number with no CIs, no MDE, single benchmark. It would not survive the verdict rule.

---

## 10. Operational runbooks

### 10.1 "The eval is failing — what now?"

1. Check the verdict reasons (§9.3 produces them). The reason names which gate failed.
2. Inspect the metric CI: is the regression real or noise?
3. Check the envelope: was the run actually reproducible? Was the baseline drift-free? (Compare envelope to last known-good.)
4. Re-run the failing tier with a fresh seed. If verdict flips, you had insufficient seeds — increase n_seeds and re-run with paired analysis.
5. If the regression is real and reproducible: the optimization is bad. Revert.
6. If you suspect a false positive: check eval-of-eval FPR (§10.4). If it's spiking, the eval system itself may be drifting.

### 10.2 "I want to add a new benchmark"

1. Verify the benchmark meets §6.1 selection criteria (discriminating, contamination-resistant, valid construct).
2. Measure baseline σ across ≥ 5 seeds. Compute MDE.
3. Pre-register MDE, MEI, and tolerance in `configs/eval/tier_thresholds.yaml`.
4. Add to `agent/eval/benchmarks/<name>.py`.
5. Add to `configs/eval/benchmark_suites.yaml` routing.
6. Run on null fixture (§10.4) — it must not gate-fail.
7. Run on planted-regression fixtures relevant to the benchmark — they must gate-fail.

### 10.3 "I want to change a threshold"

Pre-registered thresholds are *append-only with rationale*. Process:

1. Open a PR that adds the new threshold *alongside* the old.
2. Run last 30 days of historical evals against both thresholds. Compare ACCEPT/REJECT decisions.
3. If decisions differ, justify the change with reasoning (new MDE measurement, new MEI from product, etc.).
4. Get review.
5. Merge with both thresholds active for one release; then remove the old.

This prevents *threshold-shopping* — silently tightening a threshold to flip a verdict.

### 10.4 Eval-of-eval (continuous quality assurance)

Two fixture types run on every eval-system release:

- **Null fixture**: identical model A vs identical model A. Eval system should ACCEPT (or report "no detectable effect"). If it REJECTs, the eval system has a false-positive bug.
- **Planted-regression fixtures**: model A vs deliberately-broken model A' (e.g., A' has a known 5% MMLU regression injected). Eval system should REJECT. If it ACCEPTs, the eval system has a false-negative bug.

Track FPR (rate of incorrect REJECTs on null) and FNR (rate of incorrect ACCEPTs on planted) over time. If either drifts, freeze eval-system releases and investigate.

**Source of practice**: chaos engineering / fault injection (Netflix Simian Army); mutation testing in software engineering (Jia & Harman 2011, IEEE TSE).

### 10.5 Cost monitoring

- Track total eval $-spend per week, per tier.
- Alert on >2× week-over-week increase (likely a runaway loop or accidentally-expensive benchmark added).
- Hard cap: per-PR eval cost ≤ $200; per-week total ≤ $5k. Above that, require human approval.

---

## 11. Risks, anti-patterns, and known limitations

### 11.1 Risks specific to this eval design

| Risk | Mitigation |
|------|------------|
| Pre-registered thresholds become stale | §10.3 process; quarterly review of historical FPR/FNR. |
| Bench-vs-prod distribution drift | T3 catches it; refresh production traffic samples monthly. |
| Benchmark contamination grows over time | Annual review of benchmark choice (§6); active contamination probes. |
| Statistical framework misuse (e.g., p-hacking via metric selection) | Pre-registration of metric set per `optimization_target` is gate-enforced; new metrics require §10.2 process. |
| Reproducibility envelope capture incomplete (missing some env var) | Eval-of-eval planted-regression fixtures should catch envelope-induced variance. |
| Cost runaway | §10.5 monitoring + hard caps. |
| Eval system itself becomes a bottleneck (slows dev velocity) | T0/T1 cost budgets are tight; long evals run async; verdict cached. |
| Agent-eval suite becomes overfit to | Adversarial subset rotation; periodic suite refresh from real failure modes. |

### 11.2 Anti-patterns explicitly forbidden

- **Single-number quality verdicts**. Per A3.
- **Throughput without latency distribution**. Per A5.
- **Mean latency as primary**. Per A6.
- **Point estimates without CIs in T1+**. Per A1.
- **Effect claims below MDE**. Per A2.
- **Metric collection-without-pre-registration** (testing what you find significant — p-hacking). Per §4.7.
- **Skipping safety gates "just for this experiment"**. Per §5.4.
- **Modifying tier thresholds mid-experiment**. Per §10.3.
- **Eval results without envelopes used for gating**. Per A9.

### 11.3 Known limitations of this design

- **LLM-as-judge is used only as tiebreaker, not primary**. This is conservative; some eval systems use judges as primary for creative tasks. We don't, because of documented bias modes (positional, length, self-preference) — see Zheng et al. 2023 (NeurIPS). When ground truth is absent, we accept eval-incompleteness rather than introduce judge bias.
- **The agent eval suite (S1–S4) requires a hand-labeled scenario set** — initial labeling is expensive (~50–100 expert hours). Without it, agent decisions are not gate-able; with it, the suite must be maintained.
- **Sequential testing in T4 (canary)** requires careful implementation — naive repeated p-value testing inflates FPR. Use Howard et al. 2021 always-valid p-values or established A/B testing framework (e.g., Eppo, Optimizely).
- **Reproducibility on cloud-shared infrastructure is imperfect** — even with full envelope, neighboring tenants can introduce performance variance. T2 should run on dedicated hardware where possible.
- **No causal inference framework** — we measure correlation between optimization and metrics; we do not formally identify causation. For optimization changes this is acceptable (the intervention is direct) but for any inference about *why* an optimization works, additional investigation is needed.

### 11.4 What this design refuses to do

- Composite quality scores → ship the vector (A3).
- "We'll add stats later" → retrofitting CIs requires re-running every baseline. Build it in from commit 1.
- LLM-as-judge as primary capability gate → bias modes too well-documented (Zheng et al. 2023).
- Optimization-target trades against safety → safety is a gate, never a parameter.

---

## 12. References

Cited inline above; consolidated here for verification. All references are real and publicly available; URLs given where stable.

### Statistical foundations
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
- Efron, B. (1979). "Bootstrap methods: another look at the jackknife." *Annals of Statistics* 7(1):1–26.
- Efron, B. (1987). "Better bootstrap confidence intervals." *JASA* 82(397):171–185.
- Holm, S. (1979). "A simple sequentially rejective multiple test procedure." *Scandinavian Journal of Statistics* 6(2):65–70.
- Howard, S.R. et al. (2021). "Time-uniform Chernoff bounds via nonnegative supermartingales." *Probability Surveys*.
- Little, J.D.C. (1961). "A proof for the queuing formula L = λW." *Operations Research* 9(3):383–387.
- McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages." *Psychometrika* 12(2):153–157.

### ML reproducibility and statistical rigor
- Card, D., Henderson, P. et al. (2020). "With Little Power Comes Great Responsibility." *EMNLP 2020*.
- Dehghani, M. et al. (2021). "The Benchmark Lottery." arXiv:2107.07002.
- Henderson, P. et al. (2018). "Deep Reinforcement Learning that Matters." *AAAI 2018*.
- Pineau, J. et al. (2021). "Improving Reproducibility in Machine Learning Research." *JMLR* 22.
- Schaeffer, R., Miranda, B., Koyejo, S. (2023). "Are Emergent Abilities of Large Language Models a Mirage?" *NeurIPS 2023* (best paper).

### Information theory and quantization
- Cover, T.M., Thomas, J.A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
- Dettmers, T. et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *NeurIPS 2022*.
- Frantar, E. et al. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." arXiv:2210.17323.
- Lin, J. et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." *MLSys 2024*.
- Xiao, G. et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." *ICML 2023*.

### Serving systems and goodput
- Beyer, B. et al. (2016). *Site Reliability Engineering*. O'Reilly.
- Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023* (vLLM).
- Williams, S., Waterman, A., Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM* 52(4):65–76.
- Zhong, Y. et al. (2024). "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." *OSDI 2024*.

### Training-time metrics
- Chowdhery, A. et al. (2022). "PaLM: Scaling Language Modeling with Pathways." arXiv:2204.02311 (MFU definition).

### 2026 frontier benchmarks
- Brohan, A. et al. (2023). "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." *CoRL 2023*.
- Black, K. et al. (2024). "π0: A Vision-Language-Action Flow Model for General Robot Control." arXiv:2410.24164.
- Chao, P. et al. (2024). "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models." *NeurIPS 2024 D&B*.
- Hendrycks, D. et al. (2021). "Measuring Massive Multitask Language Understanding." *ICLR 2021* (MMLU).
- Hendrycks, D. et al. (2021). "Measuring Mathematical Problem Solving With the MATH Dataset." *NeurIPS 2021 D&B*.
- Hsieh, C-P. et al. (2024). "RULER: What's the Real Context Size of Your Long-Context Language Models?" arXiv:2404.06654 (NVIDIA).
- Jain, N. et al. (2024). "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code." arXiv:2403.07974.
- Jimenez, C.E. et al. (2024). "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" *ICLR 2024*.
- Kim, M.J. et al. (2024). "OpenVLA: An Open-Source Vision-Language-Action Model." arXiv:2406.09246.
- Liang, P. et al. (2022). "Holistic Evaluation of Language Models." arXiv:2211.09110 (HELM).
- Lightman, H. et al. (2023). "Let's Verify Step by Step." arXiv:2305.20050 (PRM800K / MATH-500 split).
- Liu, J. et al. (2023). "Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation." *NeurIPS 2023* (HumanEval+/MBPP+ via EvalPlus).
- Mazeika, M. et al. (2024). "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal." *ICML 2024*.
- Mialon, G. et al. (2023). "GAIA: A Benchmark for General AI Assistants." arXiv:2311.12983.
- Patil, S.G. et al. (2024). "Berkeley Function Calling Leaderboard (BFCL)." (project; v3 2024).
- Rein, D. et al. (2023). "GPQA: A Graduate-Level Google-Proof Q&A Benchmark." arXiv:2311.12022.
- Röttger, P. et al. (2024). "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models." *NAACL 2024*.
- Sainz, O. et al. (2023). "NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark." *EMNLP 2023 Findings*.
- Souly, A. et al. (2024). "A StrongREJECT for Empty Jailbreaks." arXiv:2402.10260.
- Srivastava, A. et al. (2022). "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models." (BIG-bench).
- Wang, Y. et al. (2024). "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark." arXiv:2406.01574 (TIGER-Lab).
- Yao, S. et al. (2024). "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." arXiv:2406.12045 (Sierra).
- Zheng, L. et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS 2023 D&B*.

### Contamination
- Brown, T.B. et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS 2020* (GPT-3, original n-gram contamination analysis).
- Carlini, N. et al. (2021). "Extracting Training Data from Large Language Models." *USENIX Security 2021*.

### Software testing / mutation testing (basis for eval-of-eval)
- Beizer, B. (1990). *Software Testing Techniques* (2nd ed.). Van Nostrand Reinhold.
- Jia, Y., Harman, M. (2011). "An Analysis and Survey of the Development of Mutation Testing." *IEEE TSE* 37(5):649–678.

### Standards and practice
- MLPerf Inference Benchmark Rules. MLCommons. https://mlcommons.org/
- NeurIPS Reproducibility Checklist. https://neurips.cc/

---

## Appendix A — How to read a verdict

Sample verdict output from `ml-intern-eval show <run_id>`:

```
Run: 7f3a2b1c-...   Tier: T1   Target: throughput
Envelope: ✓ complete (commit a3f9..., H100-SXM5×8, deterministic)

Metric                       Value     CI(95%)               MDE      MEI    Verdict
─────────────────────────────────────────────────────────────────────────────────
goodput_at_slo (target)     +18.4%    [+15.2%, +21.6%]      ±2.0%    +5%    ✓ pass
kl_divergence_mean          +0.012    [+0.008, +0.017]      ±0.003   ±0.05  ✓ within tol
kl_divergence_max           +0.31     [+0.21, +0.42]        ±0.05    ±0.5   ✓ within tol
perplexity_wikitext2        +0.08     [-0.04, +0.20]        ±0.11    ±0.5   ⊘ below MDE (advisory)
mmlu_pro_accuracy           -0.003    [-0.007, +0.001]      ±0.002   ±0.005 ✓ within tol
gpqa_diamond_accuracy       -0.015    [-0.040, +0.010]      ±0.025   ±0.02  ⊘ below MDE
latency_p99 (guardrail)     ×1.12     [×1.08, ×1.16]                 ≤×1.5  ✓ guardrail ok
memory_peak (guardrail)     ×0.94     [×0.93, ×0.95]                 ≤×1.1  ✓ guardrail ok

Holm-Bonferroni (k=5 quality metrics, α=0.05): all rejections held after correction.

VERDICT: ACCEPT
Reasons: All gates passed. Target metric (goodput_at_slo) improvement +18.4%
[CI excludes 0; > MDE]; no quality metric regressed outside CI tolerance;
all guardrails within bounds. Below-MDE results reported as advisory only.
```

How to read it:
- `Value` is the point estimate; `CI(95%)` is the bootstrap interval; if CI crosses 0 (or 1× for ratios), the effect is not distinguishable from noise.
- `MDE` is the smallest effect this experiment could detect — values inside `[-MDE, +MDE]` are reported as "below detection threshold" regardless of point estimate.
- `MEI` is the smallest effect that would matter operationally.
- `⊘ below MDE` means the result is advisory only — it cannot inform the gate decision.
- `VERDICT: ACCEPT` requires all four gates (§4.6) to pass conjunctively.

---

## Appendix B — Glossary

- **CI**: Confidence Interval. A range that contains the true parameter value with stated frequentist probability (95% standard).
- **FPR / FNR**: False Positive Rate / False Negative Rate.
- **FWER**: Family-Wise Error Rate — probability of ≥1 false positive across a family of tests.
- **Goodput@SLO**: throughput × probability(latency ≤ SLO). The user-relevant deployment metric.
- **ITL**: Inter-Token Latency. Often used synonymously with TBT.
- **KL divergence**: Kullback-Leibler divergence. Information-theoretic distance between two probability distributions.
- **MDE**: Minimum Detectable Effect. Smallest effect detectable at given α, power, n, σ.
- **MEI**: Minimum Effect of Interest. Smallest effect with operational meaning (pre-registered, product-defined).
- **MFU**: Model FLOPs Utilization. Achieved FLOPs / theoretical peak.
- **MT-bench, Chatbot Arena**: benchmarks for instruction-following / preference comparison; cited here only re: judge bias (Zheng et al. 2023), not used as primary metrics in this design.
- **p99, p999**: 99th, 99.9th percentile latency.
- **PPL**: Perplexity. exp(cross-entropy loss). Lower is better.
- **TBT**: Time Between Tokens. Decode-phase per-token latency.
- **TTFT**: Time To First Token. Prefill-phase latency.
- **Tier**: a level in the eval stack (T0–T4), defined by cost, cadence, and gate decision.
- **VLA**: Vision-Language-Action model. Robotics policy with vision + language inputs.

---

*End of EVAL_SPEC.md.*
