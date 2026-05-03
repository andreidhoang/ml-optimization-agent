# Frontier AI Optimization Agent — Implementation Plan

> **Grounded in:** actual codebase at `/Users/danghuyhoang/Desktop/ml-intern`  
> **Strategy:** Build on top of ML Intern infrastructure. Fork, don't modify.  
> **Target:** A specialized agent that profiles, diagnoses, and optimizes training + inference for LLMs, multimodal models, and VLAs.

---

## Architecture Decision Record

**Decision:** Build on ML Intern infrastructure, not from scratch.

**Rationale (verified from codebase):**

| Infrastructure Component | Lines | Why Keep |
|---|---|---|
| `agent/core/agent_loop.py` | 1,626 | Contains 6+ hard edge cases: abandoned approvals, thinking signature healing, malformed JSON recovery, stream cut-off handling |
| `agent/core/session.py` | 487 | Atomic write, detached subprocess upload, heartbeat saves already solved |
| `agent/context_manager/manager.py` | 415 | Dangling tool call patching, compaction threshold math, system prompt reuse |
| `backend/session_manager.py` | 608 | `asyncio.to_thread` for blocking init, EventBroadcaster fan-out, sandbox cleanup retry |
| `agent/tools/jobs_tool.py` | 1,198 | Log streaming resilience, UV log filtering, GPU flavor specs, timeout enforcement |
| `agent/tools/sandbox_tool.py` | 477 | Orphan cleanup, Trackio injection, hardware tier selection |

**What we throw away:** `agent/prompts/system_prompt_v3.yaml` (replaced entirely), `configs/cli_agent_config.json` (new config), general-purpose tool descriptions.

**What we add:** 4 new tool suites, 1 new optimization context system, 1 new system prompt, 1 new config.

---

## Cross-Cutting Rules

These rules apply across every phase. Each one prevents a class of false-positive results that would otherwise survive into the agent's recommendations.

### Rule 1 — Two-Level Benchmarking *(mandatory for every optimization tool)*

Every tool that claims a speedup MUST report **both** numbers:

1. **Component speedup** — optimized op vs. baseline op, in isolation
2. **End-to-end speedup** — optimized pipeline vs. baseline pipeline, on a realistic workload

**Why:** A custom RMSNorm kernel measured at 1.88× faster in isolation yields only 1.06× end-to-end speedup if RMSNorm is 5% of the pipeline (Amdahl's law). Reporting only the isolated number is malpractice — it claims a win the user does not see.

Tool return contract for everything in `training_opt/`, `inference_opt/`, `multimodal_opt/`, `vla_opt/`, `kernel_gen/`:

```json
{
  "component_speedup": 1.88,
  "end_to_end_speedup": 1.06,
  "component_fraction_of_pipeline": 0.05,
  "amdahl_predicted_e2e": 1.05,
  "deviation_from_amdahl_pct": 1.0
}
```

If `deviation_from_amdahl_pct > 10`, the tool flags the result for investigation — the gap signals measurement error, contention, or a confounding optimization, and the experiment is recorded with `verdict="investigating"`, not `"keep"`.

### Rule 2 — Measured Peak Over Vendor Peak

Every roofline calculation uses **measured** peak throughput from `measure_peak_throughput` (Step 1.3), not the static `HARDWARE_SPECS` table. Vendor specs are upper bounds; thermal throttling, MIG partitioning, and power capping routinely deliver 60–90% of nameplate. Treating vendor numbers as truth produces "MFU = 28%, severe bottleneck" diagnoses on hardware that is already at its real ceiling.

The static table remains as the documented theoretical ceiling and as the fallback when measurement fails or is explicitly skipped (`use_measured_peak=false` for fast iteration).

### Rule 3 — One Optimization Per Experiment

Each `Experiment` recorded in `OptimizationContext` changes exactly one variable from the baseline. Stacking FP8 quantization + speculative decoding + sequence packing in a single run makes the speedup unattributable: if quality regresses, which one caused it? If throughput improves less than expected, which one underperformed? The system prompt enforces this; tool handlers reject configs that combine techniques unless an explicit `combine_with` argument is passed by the user.

---

## Repository Structure (Target State)

```text
ml-optimization-agent/          ← fork of ml-intern
│
├── agent/
│   ├── core/                   ← KEEP AS-IS (zero modifications)
│   │   ├── agent_loop.py
│   │   ├── session.py
│   │   ├── doom_loop.py
│   │   ├── prompt_caching.py
│   │   ├── telemetry.py
│   │   ├── redact.py
│   │   ├── llm_params.py
│   │   ├── model_switcher.py
│   │   └── effort_probe.py
│   │
│   ├── context_manager/        ← ONE addition: persistent_state field
│   │   └── manager.py
│   │
│   ├── messaging/              ← KEEP AS-IS
│   │
│   ├── prompts/
│   │   ├── system_prompt_v3.yaml          ← KEEP (for reference)
│   │   └── system_prompt_optimization_v1.yaml  ← NEW (Phase 1)
│   │
│   ├── tools/
│   │   ├── (existing tools)    ← KEEP AS-IS (jobs, sandbox, research, docs...)
│   │   │
│   │   ├── profiling/          ← NEW (Phase 2)
│   │   │   ├── __init__.py
│   │   │   ├── training_mfu.py
│   │   │   ├── inference_latency.py
│   │   │   ├── memory_timeline.py
│   │   │   ├── measured_peak.py        ← Step 1.3 (used by all profilers)
│   │   │   └── nsight_profile.py       ← Step 2.3 (kernel-level metrics)
│   │   │
│   │   ├── training_opt/       ← NEW (Phase 3)
│   │   │   ├── __init__.py
│   │   │   ├── parallelism_tuner.py
│   │   │   ├── sequence_packing.py
│   │   │   ├── flash_attention.py
│   │   │   └── liger_kernels.py
│   │   │
│   │   ├── inference_opt/      ← NEW (Phase 4)
│   │   │   ├── __init__.py
│   │   │   ├── quantization.py
│   │   │   ├── vllm_deployer.py
│   │   │   ├── sglang_deployer.py
│   │   │   ├── speculative_decoding.py
│   │   │   └── serving_benchmark.py
│   │   │
│   │   ├── multimodal_opt/     ← NEW (Phase 5)
│   │   │   ├── __init__.py
│   │   │   └── visual_token_compressor.py
│   │   │
│   │   ├── vla_opt/            ← NEW (Phase 5)
│   │   │   ├── __init__.py
│   │   │   ├── action_latency_profiler.py
│   │   │   └── fast_slow_splitter.py
│   │   │
│   │   └── kernel_gen/         ← NEW (Phase 7, gated)
│   │       ├── __init__.py
│   │       ├── generate_kernel.py
│   │       ├── publish_kernel.py
│   │       └── orchestrator.py
│   │
│   ├── skills/                 ← NEW (Phase 7 + optional MC-4 migration)
│   │   └── cuda-kernels/
│   │       ├── SKILL.md
│   │       ├── scripts/
│   │       └── references/     ← per-arch + per-framework knowledge files
│   │
│   ├── optimization/           ← NEW (Phase 6)
│   │   ├── __init__.py
│   │   ├── context.py          ← OptimizationContext, Experiment
│   │   ├── roofline.py         ← Roofline calculations
│   │   ├── pareto.py           ← Multi-objective Pareto analysis
│   │   └── bottleneck.py       ← Bottleneck classifier
│   │
│   └── config.py               ← EXTEND: add optimization fields
│
├── backend/                    ← KEEP AS-IS
├── frontend/                   ← KEEP AS-IS (minor: add profiling viz)
│
├── configs/
│   ├── cli_agent_config.json              ← KEEP (reference)
│   ├── frontend_agent_config.json         ← KEEP (reference)
│   └── optimization_agent_config.json     ← NEW (Phase 1)
│
└── tests/
    ├── unit/                   ← KEEP existing tests
    └── optimization/           ← NEW tests per phase
```

---

## Timeline Overview

```text
Week  1–2    Phase 0: Repository setup + environment verification
Week  3–4    Phase 1: Knowledge foundation (system prompt + hardware specs + measured peak)
Week  5–7    Phase 2: Profiling suite (MFU + inference + memory + Nsight kernel-level)
Week  8–10   Phase 3: Training optimization tools
Week 11–13   Phase 4: Inference optimization tools
Week 14–15   Phase 5: Multimodal + VLA tools
Week 16–17   Phase 6: Optimization state machine
Week 18–20   Phase 7: Custom CUDA kernel generation  ← gated; activated only if Phase 4 plateaus
Week 21–23   Phase 8: Scored ML-optimization benchmark suite (AHE Stage C — rate-limiter)
Week 24–26   Phase 9: Trajectory observability + manifest verification (AHE Stages E + F)
Week 27–30   Phase 10: Evolve Agent + Algorithm 1 orchestration (AHE Stages G + H)
Week 31      Phase 11: Cross-model transfer evaluation (AHE Stage I)
```

**Phases 8–11 implement the AHE meta-stack** (Lin et al., arXiv:2604.25850v2). See `RESEARCH_AHE_ANALYSIS.md` for the full architectural rationale. AHE Stages A (7-slot decomposition) and D (manifest discipline) are cross-cutting — A starts in Phase 0, D starts in Phase 1 and persists through all phases.

---

## Phase 0: Repository Setup & Baseline Verification
**Duration:** 1–2 weeks  
**Goal:** Working fork with verified infrastructure. All existing tests pass. New config loaded correctly.

---

### Step 0.1 — Fork the Repository

**Action:**
```bash
# Option A: GitHub fork
gh repo fork huggingface/ml-intern --clone --remote
mv ml-intern ml-optimization-agent
cd ml-optimization-agent

# Option B: Local copy
cp -r /Users/danghuyhoang/Desktop/ml-intern /Users/danghuyhoang/Desktop/ml-optimization-agent
cd /Users/danghuyhoang/Desktop/ml-optimization-agent
git remote set-url origin <your-new-remote>
```

**Verify:**
```bash
uv sync
uv run pytest tests/unit/ -x -q
# Expected: all existing tests pass
```

**Acceptance Criteria:** `pytest tests/unit/` exits 0. All 20 unit tests pass.

---

### Step 0.2 — Add Optimization Fields to Config

**File to modify:** `agent/config.py`

**Exact change** — add after `reasoning_effort` field in the `Config` class:

```python
# --- Optimization agent fields ---
# Target modality for optimization. Drives system prompt selection
# and tool availability. None = general mode (backward compatible).
optimization_target: str | None = None
# Valid: "training" | "inference" | "multimodal" | "vla" | None

# Hardware the model will run on. Used for roofline analysis.
# Keys match hardware_constants in system_prompt_optimization_v1.yaml.
target_hardware: str | None = None
# Valid: "h100_sxm" | "a100_sxm" | "a100_pcie" | "l40s" | "mi300x" | None

# Quality budget for optimization (0.0–1.0, where 1.0 = no degradation allowed).
# Agent uses this for multi-objective trade-off decisions.
quality_budget: float = 0.98
# Example: 0.98 = accept up to 2% quality drop for speed/memory gains

# Enable iterative optimization loop (profile → fix → profile → compare).
optimization_loop_enabled: bool = True
```

**Verify:**
```python
# tests/optimization/test_config_optimization.py
from agent.config import load_config

def test_optimization_fields_default(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"model_name": "moonshotai/Kimi-K2.6"}')
    cfg = load_config(str(cfg_path))
    assert cfg.optimization_target is None
    assert cfg.target_hardware is None
    assert cfg.quality_budget == 0.98
    assert cfg.optimization_loop_enabled is True

def test_optimization_fields_set(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('''{
        "model_name": "moonshotai/Kimi-K2.6",
        "optimization_target": "inference",
        "target_hardware": "h100_sxm",
        "quality_budget": 0.95
    }''')
    cfg = load_config(str(cfg_path))
    assert cfg.optimization_target == "inference"
    assert cfg.target_hardware == "h100_sxm"
    assert cfg.quality_budget == 0.95
```

**Acceptance Criteria:** Test passes. Existing `test_config.py` still passes.

---

### Step 0.3 — Create Optimization Agent Config

**File to create:** `configs/optimization_agent_config.json`

```json
{
  "model_name": "bedrock/us.anthropic.claude-opus-4-6-v1",
  "optimization_target": null,
  "target_hardware": null,
  "quality_budget": 0.98,
  "optimization_loop_enabled": true,
  "save_sessions": true,
  "session_dataset_repo": "smolagents/ml-optimization-sessions",
  "yolo_mode": false,
  "confirm_cpu_jobs": true,
  "auto_file_upload": true,
  "reasoning_effort": "max",
  "mcpServers": {
    "hf-mcp-server": {
      "transport": "http",
      "url": "https://huggingface.co/mcp?login"
    }
  }
}
```

**Verify:**
```bash
python -c "
from agent.config import load_config
cfg = load_config('configs/optimization_agent_config.json')
print('quality_budget:', cfg.quality_budget)
print('optimization_loop_enabled:', cfg.optimization_loop_enabled)
"
# Expected: prints 0.98 and True
```

---

### Step 0.4 — Add `persistent_state` to ContextManager

This is the **only modification** needed in infrastructure. It allows `OptimizationContext` to survive context compaction.

**File to modify:** `agent/context_manager/manager.py`

**Locate the `__init__` method** (line ~136) and add one field:

```python
class ContextManager:
    def __init__(
        self,
        model_max_tokens: int,
        compact_size: float = 0.1,
        untouched_messages: int = 5,
        tool_specs: list = None,
        hf_token: str | None = None,
        local_mode: bool = False,
        prompt_file_suffix: str = "system_prompt_v3.yaml",
    ):
        # ... existing code unchanged ...
        self.items: list[Message] = [Message(role="system", content=self.system_prompt)]
        
        # NEW: Structured state that survives compaction.
        # Populated by optimization/context.py. Serialized back into
        # context as a user message after each compact() call.
        self.persistent_state: dict = {}
```

**Locate the `compact()` method** (line ~350) and add state re-injection after the existing compaction logic:

```python
async def compact(self, ...):
    # ... existing compaction code unchanged ...
    
    # NEW: Re-inject persistent_state after compaction so the LLM
    # always has access to experiment history, baseline metrics, etc.
    # This must happen AFTER compaction rewrites self.items.
    #
    # CRITICAL: Do NOT insert a new Message(role="user") here.
    # Anthropic API enforces strict user/assistant alternation.
    # After compaction, self.items is [system, user_summary, assistant, ...].
    # Inserting a new user message at index 1 produces [system, user, user, ...]
    # which causes API error: "roles must alternate between user and assistant".
    # Instead, APPEND the state to the first existing user message's content.
    if self.persistent_state:
        import json
        state_suffix = (
            "\n\n[OPTIMIZATION_STATE — persisted across compaction]\n"
            + json.dumps(self.persistent_state, indent=2)
        )
        first_user_idx = next(
            (i for i, m in enumerate(self.items) if m.role == "user"), None
        )
        if first_user_idx is not None:
            # Append to existing user message — alternation invariant preserved
            if isinstance(self.items[first_user_idx].content, str):
                self.items[first_user_idx].content += state_suffix
            else:
                # Content is a list of blocks (multimodal) — append text block
                self.items[first_user_idx].content.append(
                    {"type": "text", "text": state_suffix}
                )
        else:
            # No user message yet — safe to append (session is in initial state)
            self.items.append(Message(role="user", content=state_suffix.strip()))
```

**Verify:**
```python
# tests/optimization/test_persistent_state.py
import asyncio
from unittest.mock import AsyncMock, patch
from agent.context_manager.manager import ContextManager

def test_persistent_state_survives_compaction():
    cm = ContextManager(model_max_tokens=10_000)
    cm.persistent_state = {"experiment_count": 3, "best_mfu": 0.52}
    
    # Force compaction threshold
    cm.running_context_usage = 9500
    
    # Mock summarize_messages to return a simple string
    async def run():
        with patch("agent.context_manager.manager.summarize_messages",
                   new=AsyncMock(return_value=("Summary text", 100))):
            await cm.compact(model_name="mock", hf_token=None)
        
        # Check state was re-injected
        contents = [m.content for m in cm.items if hasattr(m, 'content')]
        state_injected = any(
            "OPTIMIZATION_STATE" in str(c) for c in contents
        )
        assert state_injected, "persistent_state not found after compaction"
        assert '"experiment_count": 3' in str(cm.items)
    
    asyncio.run(run())
```

**Acceptance Criteria:** Test passes. Existing `test_dangling_tool_calls.py` still passes.

---

## Phase 1: Knowledge Foundation
**Duration:** 2 weeks  
**Goal:** Agent reasons correctly about optimization before any tools exist. System prompt drives roofline-first thinking.

---

### Step 1.1 — Write the Optimization System Prompt

**File to create:** `agent/prompts/system_prompt_optimization_v1.yaml`

The prompt is a Jinja2 template (same format as `system_prompt_v3.yaml`).

```yaml
system_prompt: |
  You are a Frontier AI Optimization Engineer. You have {{ num_tools }} tools
  for profiling, diagnosing, and optimizing training and inference for LLMs,
  multimodal models, and Vision-Language-Action (VLA) models.

  Your job is not to train models from scratch. Your job is to make existing
  models and training pipelines measurably faster, cheaper, or more memory-efficient —
  while preserving quality within a defined budget.

  # The Optimization Mandate (Non-Negotiable)

  **NEVER suggest an optimization technique without profiling data first.**

  The only acceptable workflow is:
  0. HARDWARE: Call lookup_hardware_specs(hardware=target_hardware) first.
               Never use hardware constants from memory — the tool is the authoritative
               source and may include GPUs added after your training cutoff.
  1. MEASURE: Run profile_training_mfu() or profile_inference_latency() to establish baseline
  2. IDENTIFY: Classify the bottleneck (compute / memory-bandwidth / communication / I/O)
  3. REASON: Apply the Roofline Model to understand the theoretical ceiling
  4. SELECT: Choose the technique that directly addresses the identified bottleneck
  5. IMPLEMENT: Apply the technique via the appropriate tool
  6. VERIFY: Re-profile. Quantify the delta (throughput, latency, memory, quality)
  7. DECIDE: Keep if within quality_budget, revert and try next hypothesis otherwise

  Saying "I think it might be memory-bound" and then recommending quantization
  without profiling is malpractice. Always measure first.

  # The Roofline Model

  Every computation is either compute-bound or memory-bandwidth-bound.
  The ridge point separates them.

  ```
  Achievable Performance (FLOPS/s)
        │          ╱ Compute ceiling
        │         ╱
        │        ╱
  ──────────────────── Memory BW ceiling
        │
        └──────────────────→ Arithmetic Intensity (FLOPS/byte)
  ```

  **To classify any operation:**
  ```
  arithmetic_intensity = total_flops / total_bytes_moved

  if arithmetic_intensity < hardware_ridge_point:
      bottleneck = "memory_bandwidth"
      solutions = ["quantization", "KV compression", "attention fusion",
                   "reduce activation size", "flash attention"]
  else:
      bottleneck = "compute"
      solutions = ["tensor core utilization", "larger batch size",
                   "kernel fusion", "mixed precision", "better GEMM tiling"]
  ```

  # Hardware Constants

  **Always call `lookup_hardware_specs(hardware=target_hardware)` before any profiling.**
  The tool is the single source of truth. Do NOT hard-code specs from memory — new GPUs
  are added to the tool's lookup table without updating this prompt.

  Key interpretation rules (apply after calling the tool):
  - Autoregressive decode: ~2 FLOPS/byte → always memory-bandwidth-bound on every GPU
  - Dense Transformer matmuls at bf16: ~128–512 FLOPS/byte → near or above ridge point
  - Attention with FlashAttention: fused and HBM-optimal → effectively compute-bound
  - For NVLink presence: check `nvlink_bandwidth_gbs > 0` before recommending tensor parallel

  # MFU Interpretation

  MFU (Model FLOP Utilization) = achieved_tflops / hardware_peak_tflops

  ```
  MFU < 15%  → Severe bottleneck. Likely: data starvation, optimizer overhead,
                communication not overlapped, or single-GPU when multi needed.
  MFU 15-35% → Significant room. Likely: no flash attention, large activation memory,
                suboptimal parallelism, or missing fused kernels.
  MFU 35-55% → Typical well-tuned single-node training. Normal range.
  MFU 55-65% → Excellent. Requires: FlashAttention, FSDP2 with overlap,
                sequence packing, torch.compile.
  MFU > 65%  → World-class. Megatron-LM or torchtitan level tuning.
  ```

  # Bottleneck Taxonomy

  Before selecting any technique, classify the bottleneck:

  **Training Bottlenecks:**
  - compute_bound: MFU < ridge point, forward/backward dominate profile
  - memory_bound: Activation memory exploding, gradient checkpointing needed
  - communication_bound: AllReduce time > 30% of step time
  - io_bound: GPU idle waiting for data (DataLoader bottleneck)
  - optimizer_bound: optimizer.step() > 20% of step time (large models, Adam states)

  **Inference Bottlenecks:**
  - kv_cache_memory: KV cache fills GPU, limits batch size
  - decode_bandwidth: Each token reads entire model weights (autoregressive)
  - prefill_compute: Long prompt, compute-bound on attention
  - cpu_overhead: Token sampling, Python overhead between GPU calls

  # Multi-Objective Trade-off Framework

  Every recommendation MUST include a trade-off table:

  ~~~
  Technique        | Throughput Δ | Quality Δ  | Memory Δ | Complexity
  ─────────────────┼──────────────┼────────────┼──────────┼───────────
  FP8 quantization | +1.9x        | -0.3% MMLU | -49%     | Low
  GPTQ INT4        | +2.1x        | -2.8% MMLU | -58%     | Medium
  AWQ INT4         | +2.0x        | -1.4% MMLU | -58%     | Medium
  Spec. decoding   | +3.2x        | 0%         | +15%     | High
  ~~~

  Let the user choose the Pareto-optimal point for their constraints.
  Never choose for them without asking.

  # Parallelism Selection (Distributed Training)

  Use this decision tree for multi-GPU training:

  ~~~
  Model fits on 1 GPU?
    Yes → Data Parallel (DDP or FSDP2 with no sharding)
    No  → Does model fit across 1 node (8 GPUs × 80GB)?
           Yes → Tensor Parallel (TP=8, same node, NVLink required)
                 OR FSDP2 ZeRO-3 (simpler, slightly slower)
           No  → Pipeline Parallel (PP) across nodes
                 + Tensor Parallel within nodes
                 + FSDP2 for optimizer states

  For sequence length > 32k tokens:
    Add Context Parallel (CP) = ring attention across GPUs
  ~~~

  # Optimization by Model Architecture

  **Dense LLM (Llama, Mistral, Qwen):**
    Training: FSDP2 + FlashAttention + sequence packing + torch.compile
    Inference: vLLM + PagedAttention + FP8/AWQ + speculative decoding

  **MoE (Mixtral, DeepSeek-V3):**
    Training: Expert Parallel (EP) + careful load balancing loss tuning
    Inference: Expert routing cache + expert parallelism per GPU
    Note: MoE inference has lower arithmetic intensity → more memory-bandwidth-bound

  **Multimodal (LLaVA, Qwen-VL, InternVL):**
    Training: Freeze vision encoder early, visual token compression, mixed packing
    Inference: Cache visual prefix KV, dynamic resolution batching

  **VLA (π0, OpenVLA):**
    HARD CONSTRAINT: action inference must be < 50ms for manipulation
    Use CUDA graphs + static shapes. No dynamic batching in hot path.
    Fast/slow split: LLM for planning (can be slow), MLP for reactive control (must be fast)
    Do NOT use speculative decoding for real-time control (timing variance)

  # Common Mistakes to Avoid

  DO NOT recommend technique without profiling: "I think flash attention will help" is wrong.
  DO NOT change model architecture to fix an optimization problem without user approval.
  DO NOT apply multiple optimizations simultaneously (can't attribute which helped).
  DO NOT compare results across different hardware.
  DO NOT mistake training throughput for inference throughput (they optimize differently).
  DO NOT assume quantization degrades quality without measuring — FP8 is often lossless.

  # Research Integration

  When you don't know the current state of an optimization technique:
  1. Use search_mlsys_papers() to find the most recent papers
  2. Use github_find_examples() to find working implementations
  3. Use fetch_hf_docs() for HF library-specific APIs

  Optimization papers move fast. Your internal knowledge of specific numbers
  (benchmark results, speedup claims) may be outdated. Always ground claims in
  a specific paper or measurement.
```

**Verify prompt loads correctly:**

```python
# tests/optimization/test_system_prompt.py
from pathlib import Path
import yaml
from jinja2 import Template

def test_optimization_prompt_valid_yaml():
    path = Path("agent/prompts/system_prompt_optimization_v1.yaml")
    assert path.exists()
    data = yaml.safe_load(path.read_text())
    assert "system_prompt" in data
    template_str = data["system_prompt"]
    # Should render without error when num_tools is provided
    rendered = Template(template_str).render(num_tools=20)
    assert "Roofline" in rendered
    assert "MFU" in rendered
    assert "h100_sxm" in rendered
    assert len(rendered) > 3000
```

**Wire prompt into ContextManager:**

In `agent/context_manager/manager.py`, find `_load_system_prompt()`. The method takes `prompt_file_suffix`. The new config needs to pass `"system_prompt_optimization_v1.yaml"` when optimization mode is active. This is done via the config path passed to `Session.__init__`.

**Acceptance Criteria:** Prompt file exists. YAML is valid. Template renders. `num_tools` variable resolves. All hardware constants present.

---

### Step 1.2 — Hardware Specs Lookup Tool

**File to create:** `agent/tools/hardware_specs.py`

```python
"""
Hardware specifications for roofline analysis.
No network calls — pure lookup table from vendor specs.
"""
from agent.tools.types import ToolSpec

HARDWARE_SPECS: dict[str, dict] = {
    "h100_sxm": {
        "peak_bf16_tflops": 989,
        "peak_fp8_tflops": 1979,
        "memory_bandwidth_gbs": 3350,
        "ridge_point_bf16": 295,
        "nvlink_bandwidth_gbs": 900,
        "hbm_capacity_gb": 80,
        "sm_count": 132,
        "chip": "Hopper GH100",
        "interconnect": "NVLink 4.0 (900 GB/s bidirectional)",
    },
    "a100_sxm": {
        "peak_bf16_tflops": 312,
        "peak_fp8_tflops": None,   # No FP8 native support
        "memory_bandwidth_gbs": 2000,
        "ridge_point_bf16": 156,
        "nvlink_bandwidth_gbs": 600,
        "hbm_capacity_gb": 80,
        "sm_count": 108,
        "chip": "Ampere GA100",
        "interconnect": "NVLink 3.0 (600 GB/s bidirectional)",
    },
    "a100_pcie": {
        "peak_bf16_tflops": 250,
        "peak_fp8_tflops": None,
        "memory_bandwidth_gbs": 1935,
        "ridge_point_bf16": 129,
        "nvlink_bandwidth_gbs": 0,
        "hbm_capacity_gb": 80,
        "sm_count": 108,
        "chip": "Ampere GA100 (PCIe)",
        "interconnect": "PCIe 4.0 (64 GB/s)",
        "note": "No NVLink — tensor parallelism across nodes not recommended",
    },
    "l40s": {
        "peak_bf16_tflops": 362,
        "peak_fp8_tflops": 724,
        "memory_bandwidth_gbs": 864,
        "ridge_point_bf16": 419,
        "nvlink_bandwidth_gbs": 0,
        "hbm_capacity_gb": 48,
        "sm_count": 142,
        "chip": "Ada Lovelace AD102",
        "interconnect": "PCIe 4.0 (64 GB/s)",
        "note": "High ridge point = more operations are compute-bound vs A100",
    },
    "mi300x": {
        "peak_bf16_tflops": 1307,
        "peak_fp8_tflops": 2614,
        "memory_bandwidth_gbs": 5300,
        "ridge_point_bf16": 247,
        "nvlink_bandwidth_gbs": 0,
        "hbm_capacity_gb": 192,
        "sm_count": 304,
        "chip": "AMD CDNA3",
        "interconnect": "AMD Infinity Fabric",
        "note": "192GB HBM enables large models without tensor parallelism",
    },
    "t4": {
        "peak_bf16_tflops": 65,
        "peak_fp8_tflops": None,
        "memory_bandwidth_gbs": 320,
        "ridge_point_bf16": 203,
        "nvlink_bandwidth_gbs": 0,
        "hbm_capacity_gb": 16,
        "sm_count": 40,
        "chip": "Turing TU104",
        "note": "Small model inference only. Not suitable for training > 1B params",
    },
}

# HF flavor → hardware mapping (verified from jobs_tool.py)
HF_FLAVOR_TO_HARDWARE = {
    "t4-small": "t4",
    "t4-medium": "t4",
    "a10g-small": None,        # Not in table — similar to A100 PCIe at lower scale
    "a10g-large": None,
    "a10g-largex2": None,
    "a10g-largex4": None,
    "a100-large": "a100_sxm",
    "a100x4": "a100_sxm",
    "a100x8": "a100_sxm",
    "l40sx1": "l40s",
    "l40sx4": "l40s",
    "l40sx8": "l40s",
}


async def hardware_specs_handler(args: dict) -> tuple[str, bool]:
    import json

    hardware = args.get("hardware")

    if hardware == "list":
        return json.dumps(list(HARDWARE_SPECS.keys()), indent=2), True

    if hardware not in HARDWARE_SPECS:
        # Try HF flavor lookup
        mapped = HF_FLAVOR_TO_HARDWARE.get(hardware)
        if mapped and mapped in HARDWARE_SPECS:
            hardware = mapped
        else:
            return (
                f"Unknown hardware: '{hardware}'. "
                f"Valid options: {list(HARDWARE_SPECS.keys())} "
                f"or HF flavors: {list(HF_FLAVOR_TO_HARDWARE.keys())}",
                False,
            )

    specs = HARDWARE_SPECS[hardware]
    result = {
        "hardware": hardware,
        "specs": specs,
        "roofline_guidance": {
            "memory_bound_threshold_flops_per_byte": specs["ridge_point_bf16"],
            "interpretation": (
                f"Operations with arithmetic intensity < {specs['ridge_point_bf16']} FLOPS/byte "
                f"are memory-bandwidth-bound on {hardware}. "
                f"Autoregressive LLM decode (~2 FLOPS/byte) is always memory-bound here."
            ),
        },
    }
    return json.dumps(result, indent=2), True


HARDWARE_SPECS_TOOL_SPEC = ToolSpec(
    name="lookup_hardware_specs",
    description=(
        "Look up hardware specifications for roofline analysis. "
        "Returns peak TFLOPS, memory bandwidth, ridge point, NVLink bandwidth, and HBM capacity. "
        "Use this BEFORE running profile_training_mfu to interpret results. "
        "Pass hardware='list' to see all available options."
    ),
    parameters={
        "type": "object",
        "properties": {
            "hardware": {
                "type": "string",
                "description": (
                    "Hardware identifier. Options: h100_sxm, a100_sxm, a100_pcie, l40s, mi300x, t4. "
                    "Also accepts HF job flavors: a100-large, l40sx8, etc. "
                    "Pass 'list' to enumerate all options."
                ),
            }
        },
        "required": ["hardware"],
    },
    handler=hardware_specs_handler,
)
```

**Wire into ToolRouter** in `agent/core/tools.py`:

Find the import block and add:
```python
from agent.tools.hardware_specs import HARDWARE_SPECS_TOOL_SPEC, hardware_specs_handler
```

Find `create_builtin_tools()` and add:
```python
HARDWARE_SPECS_TOOL_SPEC,
```

**Verify:**
```python
# tests/optimization/test_hardware_specs.py
import asyncio
from agent.tools.hardware_specs import hardware_specs_handler

def test_h100_lookup():
    result, ok = asyncio.run(hardware_specs_handler({"hardware": "h100_sxm"}))
    import json
    data = json.loads(result)
    assert ok
    assert data["specs"]["peak_bf16_tflops"] == 989
    assert data["specs"]["ridge_point_bf16"] == 295

def test_hf_flavor_lookup():
    result, ok = asyncio.run(hardware_specs_handler({"hardware": "a100-large"}))
    import json
    data = json.loads(result)
    assert ok
    assert data["hardware"] == "a100_sxm"

def test_unknown_hardware():
    result, ok = asyncio.run(hardware_specs_handler({"hardware": "rtx4090"}))
    assert not ok
    assert "Unknown hardware" in result
```

**Acceptance Criteria:** All 3 tests pass. Tool registered in ToolRouter. `lookup_hardware_specs` callable from agent.

---

### Step 1.3 — Measured Peak Throughput Tool

The static `HARDWARE_SPECS` table from Step 1.2 reports **theoretical** peak TFLOPS and HBM bandwidth from vendor datasheets. Real silicon delivers less:

- Thermal throttling under sustained load: −5 to −20%
- Power cap (e.g., 350W H100 SXM5 vs. nameplate 700W): −40 to −50%
- MIG partition: 1/2 or 1/7 of full SM count
- Defective or older die: silent variance up to 10%

A roofline diagnosis built on theoretical peaks falsely flags well-tuned code as "underperforming." This tool measures actual peak so the agent reasons against ground truth (Cross-Cutting Rule 2).

**File to create:** `agent/tools/profiling/measured_peak.py`

```python
"""
Measure achievable peak HBM bandwidth and dense bf16/fp8 TFLOPS on the live GPU.
Single-GPU only — distributed peaks come from a separate communication benchmark.
Inspired by cfregly/ai-performance-engineering's benchmark_peak.py — measure first,
trust vendor specs second.
"""
from agent.tools.types import ToolSpec

_PEAK_BENCH_SCRIPT = '''
import torch, time, json

# 1) HBM bandwidth: large device-to-device copy, measure GB/s.
N = 1 << 28                     # 256M float32 = 1 GiB
a = torch.empty(N, dtype=torch.float32, device="cuda")
b = torch.empty(N, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
for _ in range(3): b.copy_(a)   # warmup
torch.cuda.synchronize()
t0 = time.perf_counter()
ITERS = 50
for _ in range(ITERS): b.copy_(a)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
# Each copy reads N*4 bytes and writes N*4 bytes
measured_bw_gbs = (2 * N * 4 * ITERS) / elapsed / 1e9

# 2) Dense bf16 GEMM peak: large square matmul, derive TFLOPS.
M = 8192
A = torch.randn(M, M, dtype=torch.bfloat16, device="cuda")
B = torch.randn(M, M, dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()
for _ in range(3): C = A @ B    # warmup
torch.cuda.synchronize()
t0 = time.perf_counter()
ITERS_GEMM = 20
for _ in range(ITERS_GEMM): C = A @ B
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
measured_bf16_tflops = (2 * M**3 * ITERS_GEMM) / elapsed / 1e12

# 3) Optional FP8 GEMM for Hopper+ / MI300X.
measured_fp8_tflops = None
try:
    if torch.cuda.get_device_capability()[0] >= 9:
        Af = torch.randn(M, M, device="cuda").to(torch.float8_e4m3fn)
        Bf = torch.randn(M, M, device="cuda").to(torch.float8_e4m3fn).t().contiguous().t()
        scale = torch.tensor(1.0, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS_GEMM):
            torch._scaled_mm(Af, Bf, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        measured_fp8_tflops = (2 * M**3 * ITERS_GEMM) / elapsed / 1e12
except Exception:
    pass

result = {
    "device_name": torch.cuda.get_device_name(0),
    "measured_hbm_bandwidth_gbs": round(measured_bw_gbs, 1),
    "measured_bf16_tflops": round(measured_bf16_tflops, 1),
    "measured_fp8_tflops": round(measured_fp8_tflops, 1) if measured_fp8_tflops else None,
    "measured_ridge_point_bf16": round(
        (measured_bf16_tflops * 1e12) / (measured_bw_gbs * 1e9), 1
    ),
}
print("PEAK_RESULT:" + json.dumps(result))
'''


async def measure_peak_throughput_handler(args: dict) -> tuple[str, bool]:
    import json
    from agent.tools.sandbox_tool import sandbox_exec_handler
    from agent.tools.hardware_specs import HARDWARE_SPECS

    exec_args = {"command": f"python -c '{_PEAK_BENCH_SCRIPT}'", "timeout": 180}
    result, ok = await sandbox_exec_handler(exec_args)
    if not ok:
        return f"Peak measurement failed: {result}", False

    for line in result.split("\n"):
        if not line.startswith("PEAK_RESULT:"):
            continue
        try:
            data = json.loads(line[len("PEAK_RESULT:"):])
        except json.JSONDecodeError:
            continue

        expected = args.get("expected_hardware")
        if expected and expected in HARDWARE_SPECS:
            spec = HARDWARE_SPECS[expected]
            bw_ratio = data["measured_hbm_bandwidth_gbs"] / spec["memory_bandwidth_gbs"]
            tflops_ratio = data["measured_bf16_tflops"] / spec["peak_bf16_tflops"]
            data["bandwidth_efficiency_vs_vendor"] = round(bw_ratio, 2)
            data["bf16_efficiency_vs_vendor"] = round(tflops_ratio, 2)
            if min(bw_ratio, tflops_ratio) < 0.7:
                data["warning"] = (
                    f"Measured peak is {min(bw_ratio, tflops_ratio)*100:.0f}% of vendor spec. "
                    "Likely thermal throttling, power cap, or MIG partition. "
                    "Use measured_ridge_point_bf16 for roofline analysis, NOT the vendor spec."
                )
        return json.dumps(data, indent=2), True
    return f"Could not parse peak result.\nRaw:\n{result}", False


MEASURE_PEAK_TOOL_SPEC = ToolSpec(
    name="measure_peak_throughput",
    description=(
        "Measure ACTUAL peak HBM bandwidth and bf16/fp8 TFLOPS on the live GPU. "
        "Use this output (not lookup_hardware_specs alone) for roofline ridge points "
        "when diagnosing whether code is memory- or compute-bound. "
        "Vendor specs are upper bounds; thermal throttling, power caps, and MIG partitions "
        "routinely deliver 60-90% of nameplate. "
        "Result is cached in OptimizationContext.persistent_state['measured_peak'] — call once per session."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expected_hardware": {
                "type": "string",
                "description": "Optional. Hardware key from HARDWARE_SPECS. If provided, the tool compares measured to vendor spec and emits a warning when measured is < 70% of vendor.",
            }
        },
        "required": [],
    },
    handler=measure_peak_throughput_handler,
)
```

**Wire into Phase 2 profilers:** `profile_training_mfu` and `profile_inference_latency` accept `use_measured_peak: bool = True`. On first call, they invoke `measure_peak_throughput` and cache the result in `OptimizationContext.persistent_state["measured_peak"]`. All subsequent MFU calculations divide by `measured_bf16_tflops`, not the static value.

**Acceptance Criteria:**
- On a healthy unthrottled GPU, measured bandwidth lands within 5% of vendor spec (sanity check)
- Returns `warning` field when measured is < 70% of vendor (catches MIG / power-cap / thermal cases)
- Result cached in `OptimizationContext.persistent_state["measured_peak"]` after first call
- Downstream profilers default to `measured_ridge_point_bf16` for bottleneck classification

---

### Step 1.4 — MLSys Papers Search Tool

**File to create:** `agent/tools/mlsys_papers.py`

This wraps the existing `hf_papers` and `web_search` infrastructure with optimization-specific context.

```python
"""
Specialized paper search for ML systems optimization literature.
Wraps hf_papers + web_search with optimization-specific defaults.
"""
from agent.tools.types import ToolSpec

# Curated anchor papers by domain. These are starting points for
# citation graph traversal via the research tool.
ANCHOR_PAPERS = {
    "flash_attention": [
        "2205.14135",   # FlashAttention
        "2307.08691",   # FlashAttention-2
        "2407.08608",   # FlashAttention-3
    ],
    "quantization": [
        "2210.17323",   # GPTQ
        "2306.00978",   # AWQ
        "2211.10438",   # SmoothQuant
        "2209.05433",   # LLM.int8()
    ],
    "inference_serving": [
        "2309.06180",   # PagedAttention (vLLM)
        "2312.07104",   # SGLang / RadixAttention
        "2302.01318",   # Orca / continuous batching
    ],
    "speculative_decoding": [
        "2211.17192",   # Speculative Decoding (original)
        "2401.15077",   # Eagle
        "2406.16858",   # Eagle-2
    ],
    "distributed_training": [
        "1909.08053",   # Megatron-LM
        "2205.05198",   # Megatron-LM v3 (sequence parallelism)
        "1910.02054",   # ZeRO-3 / DeepSpeed (Rajbhandari et al. 2019) — NOT 2101.03961 which is Switch Transformer
    ],
    "efficient_attention": [
        "2112.05682",   # Sparse Attention
        "2004.05150",   # Longformer
        "2310.01558",   # Ring Attention
    ],
    "training_efficiency": [
        "2403.03507",   # GaLore (Memory-Efficient LLM Training by Gradient Low-Rank Projection) — NOT 2302.13971
        "2310.05914",   # Liger Kernel
        "2407.21783",   # Sequence packing survey
    ],
    "vla": [
        "2410.24164",   # π0 (Physical Intelligence)
        "2406.09246",   # OpenVLA
        "2212.06817",   # RT-2
        "2307.15818",   # RT-X / Open-X Embodiment
    ],
    "moe": [
        "2101.03961",   # Switch Transformer
        "2401.04088",   # DeepSeek-MoE
        "2412.19437",   # DeepSeek-V3 technical report
    ],
}


async def mlsys_papers_handler(args: dict) -> tuple[str, bool]:
    import json

    domain = args.get("domain")
    query = args.get("query", "")

    if domain == "list":
        return json.dumps({
            "available_domains": list(ANCHOR_PAPERS.keys()),
            "usage": "Pass domain='flash_attention' to get anchor paper IDs for citation graph traversal"
        }, indent=2), True

    result = {
        "query": query,
        "domain": domain,
        "recommended_workflow": (
            "1. Use hf_papers(task='find_papers', query=query) to find recent papers\n"
            "2. Use hf_papers(task='citation_graph', arxiv_id=anchor_id) to find downstream work\n"
            "3. Use hf_papers(task='read_paper', arxiv_id=id) to read methodology sections 3-5\n"
            "4. Extract: technique + conditions where it works + benchmark numbers"
        ),
    }

    if domain and domain in ANCHOR_PAPERS:
        result["anchor_papers"] = {
            "arxiv_ids": ANCHOR_PAPERS[domain],
            "note": "These are landmark papers. Use citation_graph to find 2024-2025 work that cites them."
        }
    elif domain:
        result["warning"] = f"Domain '{domain}' not in curated list. Using web search instead."
        result["web_search_query"] = f"{query or domain} optimization arxiv 2024 2025"

    return json.dumps(result, indent=2), True


MLSYS_PAPERS_TOOL_SPEC = ToolSpec(
    name="search_mlsys_papers",
    description=(
        "Get anchor paper IDs and search guidance for ML systems optimization literature. "
        "Domains: flash_attention, quantization, inference_serving, speculative_decoding, "
        "distributed_training, efficient_attention, training_efficiency, vla, moe. "
        "Returns arxiv IDs to feed into hf_papers(task='citation_graph') for up-to-date research. "
        "Pass domain='list' to see all available domains."
    ),
    parameters={
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": "Optimization domain. Pass 'list' to enumerate options.",
            },
            "query": {
                "type": "string",
                "description": "Free-text search query within the domain.",
            },
        },
        "required": ["domain"],
    },
    handler=mlsys_papers_handler,
)
```

**Wire into ToolRouter** in `agent/core/tools.py`:
```python
from agent.tools.mlsys_papers import MLSYS_PAPERS_TOOL_SPEC, mlsys_papers_handler
# Add to create_builtin_tools(): MLSYS_PAPERS_TOOL_SPEC,
```

**Acceptance Criteria:** Tool callable. Returns anchor paper IDs for each domain. `domain='list'` returns all domains.

---

## Phase 2: Profiling Suite
**Duration:** 2 weeks  
**Goal:** Agent can measure MFU, inference latency, and memory usage. "Profile first" becomes possible, not just a principle.

---

### Step 2.1 — Training MFU Profiler

**File to create:** `agent/tools/profiling/training_mfu.py`

This submits a profiling job to HF Jobs (uses existing `hf_jobs_handler` pattern) and parses structured output.

```python
"""
Profile a training step and compute Model FLOP Utilization (MFU).
Uses torch.profiler with flops counting. Submits to HF Jobs.
"""
from agent.tools.types import ToolSpec
from agent.tools.jobs_tool import hf_jobs_handler  # Reuse existing infrastructure

_PROFILING_SCRIPT_TEMPLATE = '''
import torch
import torch.profiler
from transformers import AutoModelForCausalLM, AutoConfig
import json, time, sys

MODEL = "{model_name}"
BATCH_SIZE = {batch_size}
SEQ_LEN = {seq_len}
WARMUP = {warmup_steps}
PROFILE_STEPS = {profile_steps}
HARDWARE = "{hardware}"
DTYPE = torch.bfloat16 if "{dtype}" == "bfloat16" else torch.float16

HARDWARE_PEAK_TFLOPS = {{
    "h100_sxm": 989, "a100_sxm": 312, "a100_pcie": 250,
    "l40s": 362, "mi300x": 1307, "t4": 65,
}}.get(HARDWARE, 312)

print(f"Loading model {{MODEL}}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE, device_map="auto",
        attn_implementation="flash_attention_2" if {use_flash_attention} else "eager"
    )
except Exception as e:
    print(f"flash_attention_2 failed, falling back: {{e}}")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, device_map="auto")

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
input_ids = torch.randint(0, 50000, (BATCH_SIZE, SEQ_LEN), device="cuda")

print(f"Warming up {{WARMUP}} steps...")
for _ in range(WARMUP):
    loss = model(input_ids, labels=input_ids).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()

print(f"Profiling {{PROFILE_STEPS}} steps...")
torch.cuda.reset_peak_memory_stats()
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
    profile_memory=True,
) as prof:
    t_start = time.perf_counter()
    for _ in range(PROFILE_STEPS):
        loss = model(input_ids, labels=input_ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
    t_end = time.perf_counter()

events = prof.key_averages()
# IMPORTANT: Do NOT use profiler flop counts.
# FlashAttention / fused kernels bypass PyTorch's flop registry → they report flops=0.
# For a typical Transformer, attention is 30-60% of compute → profiler-based MFU
# would be 30-60% lower than actual, causing false "severe bottleneck" diagnoses.
#
# Use the standard Chinchilla theoretical estimate instead:
#   6 * N * B * S  (forward + backward = 6× single-pass FLOPs per token)
# where N = model parameters, B = batch_size, S = seq_len.
# This is how Chinchilla, PaLM, LLaMA training reports, and Megatron-LM compute MFU.
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
theoretical_flops_per_step = 6 * num_params * BATCH_SIZE * SEQ_LEN
total_cuda_us = sum(e.cuda_time_total for e in events)
total_cuda_s = total_cuda_us / 1e6 / PROFILE_STEPS

achieved_tflops = theoretical_flops_per_step / total_cuda_s / 1e12
mfu = achieved_tflops / HARDWARE_PEAK_TFLOPS
peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
wall_time_ms = (t_end - t_start) * 1000 / PROFILE_STEPS
tokens_per_sec = (BATCH_SIZE * SEQ_LEN) / (total_cuda_s)

top_ops = sorted(events, key=lambda e: e.cuda_time_total, reverse=True)[:10]
breakdown = [{{
    "name": e.key,
    "cuda_time_pct": round(e.cuda_time_total / total_cuda_us * 100, 1),
    "flops": getattr(e, "flops", 0),
}} for e in top_ops]

# Classify bottleneck from profile
forward_time = sum(e.cuda_time_total for e in events if "forward" in e.key.lower() or "matmul" in e.key.lower())
backward_time = sum(e.cuda_time_total for e in events if "backward" in e.key.lower())
optim_time = sum(e.cuda_time_total for e in events if "adam" in e.key.lower() or "optim" in e.key.lower())
comm_time = sum(e.cuda_time_total for e in events if "allreduce" in e.key.lower() or "all_reduce" in e.key.lower())

def bottleneck(fwd, bwd, opt, comm, total):
    fracs = {{"compute": (fwd+bwd)/total, "optimizer": opt/total, "communication": comm/total}}
    return max(fracs, key=fracs.get)

result = {{
    "model": MODEL,
    "hardware": HARDWARE,
    "config": {{"batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "dtype": "{dtype}"}},
    "mfu": round(mfu, 4),
    "achieved_tflops": round(achieved_tflops, 2),
    "hardware_peak_tflops": HARDWARE_PEAK_TFLOPS,
    "tokens_per_sec": round(tokens_per_sec),
    "wall_time_ms": round(wall_time_ms, 1),
    "peak_memory_gb": round(peak_memory_gb, 2),
    "bottleneck": bottleneck(forward_time, backward_time, optim_time, comm_time, total_cuda_us),
    "time_breakdown_pct": {{
        "compute_fwd_bwd": round((forward_time + backward_time) / total_cuda_us * 100, 1),
        "optimizer": round(optim_time / total_cuda_us * 100, 1),
        "communication": round(comm_time / total_cuda_us * 100, 1),
    }},
    "top_ops": breakdown,
    "mfu_interpretation": (
        "severe bottleneck" if mfu < 0.15 else
        "significant room" if mfu < 0.35 else
        "typical well-tuned" if mfu < 0.55 else
        "excellent"
    ),
}}
print("PROFILE_RESULT:" + json.dumps(result))
'''


async def profile_training_mfu_handler(args: dict) -> tuple[str, bool]:
    import json

    model_name = args["model_name"]
    batch_size = args.get("batch_size", 4)
    seq_len = args.get("seq_len", 512)
    hardware = args.get("hardware", "a100_sxm")
    dtype = args.get("dtype", "bfloat16")
    warmup = args.get("warmup_steps", 3)
    profile_steps = args.get("profile_steps", 5)
    hf_flavor = args.get("hf_flavor", "a100-large")
    use_flash = str(args.get("use_flash_attention", True)).lower() == "true"

    script = _PROFILING_SCRIPT_TEMPLATE.format(
        model_name=model_name, batch_size=batch_size, seq_len=seq_len,
        hardware=hardware, dtype=dtype, warmup_steps=warmup,
        profile_steps=profile_steps, use_flash_attention=use_flash,
    )

    # Estimate timeout: ~5 min per profile step for 7B model
    timeout = "30m"

    job_args = {
        "action": "run",
        "command": script,
        "hardware_flavor": hf_flavor,
        "timeout": timeout,
        "python_dependencies": [
            "transformers>=4.40.0",
            "torch>=2.3.0",
            "flash-attn>=2.5.0; platform_machine=='x86_64'",
        ],
    }

    result, ok = await hf_jobs_handler(job_args)

    if not ok:
        return f"Job submission failed: {result}", False

    # Parse PROFILE_RESULT from job logs
    for line in result.split("\\n"):
        if line.startswith("PROFILE_RESULT:"):
            try:
                profile_data = json.loads(line[len("PROFILE_RESULT:"):])
                return json.dumps(profile_data, indent=2), True
            except json.JSONDecodeError:
                pass

    return f"Job completed but could not parse profiling output.\\n\\nRaw output:\\n{result}", False


PROFILE_TRAINING_MFU_TOOL_SPEC = ToolSpec(
    name="profile_training_mfu",
    description=(
        "Profile a single training step and compute Model FLOP Utilization (MFU). "
        "MFU = achieved_TFLOPS / hardware_peak_TFLOPS. "
        "Returns: MFU score, tokens/sec, peak memory, bottleneck classification "
        "(compute/optimizer/communication), and top-10 slowest operations. "
        "Use lookup_hardware_specs first to understand the target hardware. "
        "Run this BEFORE any training optimization to establish baseline."
    ),
    parameters={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "HF model ID or local path"},
            "batch_size": {"type": "integer", "default": 4},
            "seq_len": {"type": "integer", "default": 512},
            "hardware": {
                "type": "string",
                "description": "Hardware for roofline calculation. Use lookup_hardware_specs to get valid values.",
                "default": "a100_sxm",
            },
            "hf_flavor": {
                "type": "string",
                "description": "HF Jobs flavor to run profiling on",
                "default": "a100-large",
            },
            "dtype": {"type": "string", "enum": ["bfloat16", "float16"], "default": "bfloat16"},
            "use_flash_attention": {"type": "boolean", "default": True},
            "warmup_steps": {"type": "integer", "default": 3},
            "profile_steps": {"type": "integer", "default": 5},
        },
        "required": ["model_name"],
    },
    handler=profile_training_mfu_handler,
)
```

**Acceptance Criteria:**
- Script template renders without syntax errors
- Tool spec JSON schema validates
- `profile_training_mfu` appears in `ToolRouter.get_tool_specs_for_llm()` output
- Unit test mocking `hf_jobs_handler` parses `PROFILE_RESULT:` lines correctly

---

### Step 2.2 — Inference Latency Profiler

**File to create:** `agent/tools/profiling/inference_latency.py`

Runs in **sandbox** (faster iteration) rather than HF Jobs. Tests TTFT, TBT, and throughput.

```python
"""
Benchmark inference latency: TTFT, TBT, throughput vs batch size.
Runs in sandbox for faster iteration (no job queue wait).
"""
from agent.tools.types import ToolSpec

_INFERENCE_BENCHMARK_SCRIPT = '''
import torch, time, json, statistics
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "{model_name}"
BATCH_SIZES = {batch_sizes}
INPUT_LEN = {input_length}
OUTPUT_LEN = {output_length}
WARMUP = {warmup_runs}
MEASURE = {measure_runs}

print(f"Loading {{MODEL}}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()

results = []
for bs in BATCH_SIZES:
    input_ids = torch.randint(1000, 50000, (bs, INPUT_LEN), device="cuda")
    
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model.generate(input_ids, max_new_tokens=OUTPUT_LEN, do_sample=False)
    torch.cuda.synchronize()

    # TTFT = prefill latency (one forward pass on input_ids, no generation).
    # This is the correct approximation for batch benchmarking.
    # The naive `total_time / new_tokens` is WRONG — it computes average token
    # latency, not time-to-first-token.  True per-request TTFT via
    # TextIteratorStreamer is incompatible with batch_size > 1.
    prefill_times = []
    with torch.no_grad():
        for _ in range(MEASURE):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids)   # prefill only — no generation
            torch.cuda.synchronize()
            prefill_times.append((time.perf_counter() - t0) * 1000)

    # Full generation for TBT and throughput
    tbts, throughputs = [], []
    with torch.no_grad():
        for _ in range(MEASURE):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(input_ids, max_new_tokens=OUTPUT_LEN, do_sample=False)
            torch.cuda.synchronize()
            total_time = time.perf_counter() - t0
            new_tokens = out.shape[1] - INPUT_LEN
            ttft_s = statistics.mean(prefill_times) / 1000   # prefill as TTFT proxy
            tbt = (total_time - ttft_s) / max(new_tokens - 1, 1) * 1000
            tput = (bs * new_tokens) / total_time
            tbts.append(tbt)
            throughputs.append(tput)
    ttfts = prefill_times   # TTFT ≈ prefill time

    results.append({{
        "batch_size": bs,
        "ttft_ms_mean": round(statistics.mean(ttfts), 1),
        "ttft_ms_p95": round(sorted(ttfts)[int(len(ttfts) * 0.95)], 1),
        "tbt_ms_mean": round(statistics.mean(tbts), 2),
        "throughput_tokens_per_sec": round(statistics.mean(throughputs)),
        "memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }})

summary = {{
    "model": MODEL,
    "input_length": INPUT_LEN,
    "output_length": OUTPUT_LEN,
    "backend": "transformers (naive autoregressive)",
    "results_by_batch_size": results,
    "bottleneck_note": (
        "Low throughput at batch_size=1 = memory-bandwidth-bound (typical). "
        "Throughput scales linearly with batch = good GPU utilization. "
        "Compare against vLLM results to measure serving overhead."
    ),
}}
print("INFERENCE_RESULT:" + json.dumps(summary))
'''


async def profile_inference_latency_handler(args: dict) -> tuple[str, bool]:
    import json
    from agent.tools.sandbox_tool import sandbox_exec_handler

    script = _INFERENCE_BENCHMARK_SCRIPT.format(
        model_name=args["model_name"],
        batch_sizes=args.get("batch_sizes", [1, 4, 16]),
        input_length=args.get("input_length", 512),
        output_length=args.get("output_length", 128),
        warmup_runs=args.get("warmup_runs", 2),
        measure_runs=args.get("measure_runs", 3),
    )

    exec_args = {
        "command": f"pip install transformers torch -q && python -c '{script}'",
        "timeout": 300,
    }

    result, ok = await sandbox_exec_handler(exec_args)
    if not ok:
        return f"Execution failed: {result}", False

    for line in result.split("\\n"):
        if line.startswith("INFERENCE_RESULT:"):
            try:
                data = json.loads(line[len("INFERENCE_RESULT:"):])
                return json.dumps(data, indent=2), True
            except json.JSONDecodeError:
                pass

    return f"Could not parse inference result.\\nRaw:\\n{result}", False


PROFILE_INFERENCE_LATENCY_TOOL_SPEC = ToolSpec(
    name="profile_inference_latency",
    description=(
        "Benchmark LLM inference: TTFT (time-to-first-token), TBT (time-between-tokens), "
        "and throughput (tokens/sec) across multiple batch sizes. "
        "Runs in sandbox — faster than HF Jobs. "
        "Always run this first before recommending quantization or serving optimization. "
        "Returns baseline numbers to compare against after applying optimizations."
    ),
    parameters={
        "type": "object",
        "properties": {
            "model_name": {"type": "string"},
            "batch_sizes": {
                "type": "array",
                "items": {"type": "integer"},
                "default": [1, 4, 16],
                "description": "Batch sizes to benchmark. Keep small for large models.",
            },
            "input_length": {"type": "integer", "default": 512},
            "output_length": {"type": "integer", "default": 128},
            "warmup_runs": {"type": "integer", "default": 2},
            "measure_runs": {"type": "integer", "default": 3},
        },
        "required": ["model_name"],
    },
    handler=profile_inference_latency_handler,
)
```

**Acceptance Criteria:**
- Script template renders without syntax errors
- Tool registered in ToolRouter
- Parses `INFERENCE_RESULT:` lines from sandbox output

---

### Step 2.3 — Kernel-Level Profiling with Nsight

`torch.profiler` from Steps 2.1 and 2.2 reports op-level timing but cannot answer:

- **Why** is this kernel slow? (achieved occupancy, register spill, L1/L2 hit rate, achieved bandwidth)
- Are kernel launches gapped? (CPU↔GPU sync stalls, stream serialization, NCCL not overlapped)
- Which memory tier is the bottleneck? (HBM vs. L2 vs. shared vs. registers)

Without these, "MFU = 28%, compute-bound" is the end of the diagnosis. With them, the agent can recommend specific fixes: bump block size for occupancy, eliminate register spill via shared memory, fuse two kernels to remove launch overhead, hoist NCCL out of the critical path.

This tool is the dividing line between "the agent can identify there is a bottleneck" (Steps 2.1–2.2) and "the agent can identify *why* the bottleneck exists" — a prerequisite for Phase 7 custom kernel work.

**File to create:** `agent/tools/profiling/nsight_profile.py`

Two complementary profilers; the agent picks one based on the question:

| Tool | Captures | Use when |
|---|---|---|
| `nsys` (Nsight Systems) | Cross-stream timeline, kernel launch gaps, NCCL overlap, CPU↔GPU sync | "Why is the GPU idle X% of the time?" |
| `ncu` (Nsight Compute) | Per-kernel SM occupancy, achieved bandwidth, L1/L2 hit rate, register usage, shared mem banking | "Why is *this specific* kernel slow?" |

```python
async def profile_with_nsight_handler(args: dict) -> tuple[str, bool]:
    """
    Run a target script under Nsight Systems (timeline) or Nsight Compute (kernel metrics).
    Returns parsed metrics, not raw .nsys-rep / .ncu-rep blobs.
    """
    profiler = args["profiler"]               # "nsys" | "ncu"
    target_script = args["target_script"]
    kernel_filter = args.get("kernel_filter") # ncu only — regex on kernel names
    duration_s = args.get("duration_s", 30)

    if profiler == "nsys":
        cmd = (
            f"nsys profile --output=/tmp/profile.nsys-rep "
            f"--trace=cuda,nvtx,osrt --sample=cpu --duration={duration_s} "
            f"python {target_script} && "
            f"nsys stats --report gputrace,gpukernsum /tmp/profile.nsys-rep --format json"
        )
    elif profiler == "ncu":
        # ncu --set full slows execution ~10x; cap launches and skip warmup.
        kernel_arg = f"--kernel-regex {kernel_filter}" if kernel_filter else ""
        cmd = (
            f"ncu --set full --launch-count 5 --launch-skip 10 {kernel_arg} "
            f"--export /tmp/profile.ncu-rep --force-overwrite "
            f"python {target_script} && "
            f"ncu --import /tmp/profile.ncu-rep --csv --print-summary per-kernel"
        )
    else:
        return f"Unknown profiler '{profiler}'. Use 'nsys' or 'ncu'.", False

    from agent.tools.jobs_tool import hf_jobs_handler
    job_args = {
        "action": "run",
        "command": cmd,
        "hardware_flavor": args.get("hf_flavor", "a100-large"),
        "timeout": "20m",
        "python_dependencies": ["torch>=2.3.0"],
    }
    result, ok = await hf_jobs_handler(job_args)
    if not ok:
        return result, False

    import json
    return json.dumps(_parse_nsight_output(result, profiler), indent=2), True


def _parse_nsight_output(raw: str, profiler: str) -> dict:
    """
    Extract a unified schema regardless of which profiler ran:
    - top-10 kernels by GPU time
    - achieved occupancy, achieved bandwidth, register spill flag
    - launch gap percentage (nsys only)
    - bottleneck hint based on metric thresholds
    """
    return {
        "profiler": profiler,
        "top_kernels": [
            # Example row:
            # {"name": "ampere_bf16_s16816gemm", "time_pct": 42.3,
            #  "occupancy_pct": 38, "achieved_bw_gbs": 1820,
            #  "register_spill_bytes": 0, "l2_hit_rate_pct": 64}
        ],
        "launch_gap_pct": None,                  # nsys only — fraction of timeline GPU is idle
        "memory_throughput_pct_of_peak": None,
        "bottleneck_hint": None,                 # one of:
        # "memory_bandwidth" | "compute" | "low_occupancy" |
        # "register_spill" | "launch_overhead" | "sync_stall"
    }


PROFILE_NSIGHT_TOOL_SPEC = ToolSpec(
    name="profile_with_nsight",
    description=(
        "Run kernel-level profiling with Nsight Systems (timeline) or Nsight Compute (per-kernel). "
        "Use 'nsys' to find launch gaps and stream serialization. "
        "Use 'ncu' to find low-occupancy or register-spilling kernels. "
        "Run AFTER profile_training_mfu identifies a compute-bound bottleneck — "
        "Nsight tells you WHY the kernel is slow, not just THAT it is. "
        "Required before recommending any custom kernel work in Phase 7."
    ),
    parameters={
        "type": "object",
        "properties": {
            "profiler": {"type": "string", "enum": ["nsys", "ncu"]},
            "target_script": {"type": "string", "description": "Python script path to profile (uploaded as part of the job)"},
            "kernel_filter": {"type": "string", "description": "ncu only — regex matching kernel names to keep output focused"},
            "duration_s": {"type": "integer", "default": 30, "description": "nsys only"},
            "hf_flavor": {"type": "string", "default": "a100-large"},
        },
        "required": ["profiler", "target_script"],
    },
    handler=profile_with_nsight_handler,
)
```

**Acceptance Criteria:**
- `nsys` run on a small training script returns top-10 kernels by GPU time and a launch-gap percentage
- `ncu --kernel-regex flash_attn` returns occupancy, achieved bandwidth, and register usage limited to matching kernels
- Tool registered in ToolRouter and added to the `training`, `inference`, and (later) `kernel_dev` suites in MC-2
- Job-flavor allowlist confirmed to grant the privileges Nsight needs (`--cap-add=SYS_ADMIN` for `ncu`); document the working flavor in the tool description

---

## Phase 3: Training Optimization Tools
**Duration:** 3 weeks  
**Goal:** Agent can apply sequence packing, parallelism tuning, FlashAttention, and Liger Kernels to a training script and measure the delta.

### Key Tools to Implement

| Tool name | File | What it does |
|---|---|---|
| `tune_parallelism_topology` | `training_opt/parallelism_tuner.py` | Given model size + GPU count, return optimal TP×PP×DP |
| `apply_sequence_packing` | `training_opt/sequence_packing.py` | Transform dataset + rewrite DataCollator to remove padding |
| `install_flash_attention` | `training_opt/flash_attention.py` | Install FA3 (H100) or FA2 (others), patch model config |
| `setup_liger_kernels` | `training_opt/liger_kernels.py` | Apply Liger Triton kernels to model, measure memory Δ |

### Step 3.1 — Parallelism Topology Tuner (Pure Logic, No Job Required)

**File to create:** `agent/tools/training_opt/parallelism_tuner.py`

This tool is pure calculation — no GPU needed. Given model size, GPU count, and cluster topology, returns optimal parallelism config.

```python
"""
Pure-logic parallelism topology recommender.
No network calls. Based on established guidelines from Megatron-LM and FSDP2 docs.
"""

def recommend_parallelism(
    model_params_b: float,     # billions of parameters
    gpu_count: int,
    gpu_vram_gb: int,
    has_nvlink: bool,
    seq_len: int = 2048,
    batch_size_target: int = 1024,
) -> dict:
    """
    Return recommended TP×PP×DP topology.
    
    Rules (verified against Megatron-LM paper Sec 4 and FSDP2 docs):
    - TP requires NVLink (same node). Without NVLink, TP=1.
    - PP has pipeline bubble = (PP-1)/PP waste. Keep PP low.
    - DP is always beneficial for throughput.
    - Context Parallel (CP) for seq_len > 32k.
    """
    model_bytes = model_params_b * 1e9 * 2   # bf16 working weights
    # Mixed-precision training (AMP, the standard):
    #   bf16 param (2) + fp32 master weight (4) + fp32 grad (4) + fp32 m (4) + fp32 v (4) = 18 bytes/param
    # Pure bf16 training (less common, less stable):
    #   bf16 param (2) + bf16 grad (2) + fp32 m (4) + fp32 v (4) = 12 bytes/param
    # Using 18 bytes (mixed precision default) — using 12 would underestimate by 50%
    # and cause FSDP2 recommendations that silently OOM.
    optimizer_bytes = model_params_b * 1e9 * 18
    total_bytes = model_bytes + optimizer_bytes

    # Single GPU capacity
    single_gpu_bytes = gpu_vram_gb * 1e9 * 0.85  # 85% usable

    recommendations = []

    # Strategy 1: FSDP2 (simplest, good for ≤ 8 GPUs)
    if total_bytes <= single_gpu_bytes * gpu_count * 0.9:
        recommendations.append({
            "strategy": "FSDP2_ZeRO3",
            "tp": 1, "pp": 1, "dp": gpu_count, "cp": 1,
            "estimated_gpu_memory_gb": round(total_bytes / gpu_count / 1e9, 1),
            "pros": ["Simple config", "Good for ≤ 8 GPU", "torch.compile compatible"],
            "cons": ["More communication than TP for very large models"],
            "framework": "PyTorch FSDP2",
        })

    # Strategy 2: Tensor Parallel (requires NVLink)
    if has_nvlink and gpu_count >= 4:
        tp = min(8, gpu_count)  # TP across NVLink domain (1 node)
        dp = gpu_count // tp
        recommendations.append({
            "strategy": "TensorParallel_DataParallel",
            "tp": tp, "pp": 1, "dp": max(1, dp), "cp": 1,
            "estimated_gpu_memory_gb": round(model_bytes / tp / 1e9, 1),
            "pros": ["Best throughput for large models", "No pipeline bubble"],
            "cons": ["Requires NVLink", "Complex implementation (Megatron-LM or torchtitan)"],
            "framework": "Megatron-LM / torchtitan",
        })

    # Strategy 3: Pipeline + Tensor (for models spanning multiple nodes)
    if total_bytes > single_gpu_bytes * 8 and gpu_count > 8:
        tp = 8 if has_nvlink else 1
        remaining = gpu_count // tp
        pp = min(4, remaining)  # Limit PP to reduce bubble
        dp = remaining // pp
        recommendations.append({
            "strategy": "Hybrid_TP_PP_DP",
            "tp": tp, "pp": pp, "dp": max(1, dp), "cp": 1,
            "estimated_gpu_memory_gb": round(model_bytes / (tp * pp) / 1e9, 1),
            "pros": ["Handles models larger than single-node memory"],
            "cons": [
                f"Pipeline bubble wastes {round((pp-1)/pp*100, 0)}% of pipeline cycles",
                "Most complex to tune"
            ],
            "framework": "Megatron-LM",
        })

    # Context Parallel for long sequences
    cp = 1
    if seq_len > 32768:
        cp = min(8, gpu_count // 2)
        for r in recommendations:
            r["cp"] = cp
            r["note"] = f"Added CP={cp} for seq_len={seq_len} (ring attention)"

    return {
        "model_params_b": model_params_b,
        "gpu_count": gpu_count,
        "gpu_vram_gb": gpu_vram_gb,
        "has_nvlink": has_nvlink,
        "recommendations": recommendations,
        "selection_guide": (
            "Start with FSDP2_ZeRO3 — it's simpler and nearly as fast for most cases. "
            "Only move to TensorParallel if profiling shows FSDP2 communication > 30% of step time."
        ),
    }
```

**Acceptance Criteria:**
```python
# tests/optimization/test_parallelism_tuner.py
from agent.tools.training_opt.parallelism_tuner import recommend_parallelism

def test_7b_single_node():
    result = recommend_parallelism(7.0, gpu_count=8, gpu_vram_gb=80, has_nvlink=True)
    strategies = [r["strategy"] for r in result["recommendations"]]
    assert "FSDP2_ZeRO3" in strategies

def test_70b_no_nvlink():
    result = recommend_parallelism(70.0, gpu_count=8, gpu_vram_gb=80, has_nvlink=False)
    tp_recs = [r for r in result["recommendations"] if r.get("tp", 1) > 1]
    assert len(tp_recs) == 0, "Should not recommend TP without NVLink"

def test_context_parallel_triggered():
    result = recommend_parallelism(7.0, gpu_count=8, gpu_vram_gb=80, 
                                    has_nvlink=True, seq_len=65536)
    assert any(r.get("cp", 1) > 1 for r in result["recommendations"])
```

---

## Phase 4: Inference Optimization Tools
**Duration:** 3 weeks  
**Goal:** Agent can quantize a model (GPTQ/AWQ/FP8), deploy vLLM, and measure the speedup delta against baseline from Phase 2.

### Key Tools

| Tool | File | What it does |
|---|---|---|
| `quantize_model` | `inference_opt/quantization.py` | Run GPTQ/AWQ/FP8 pipeline, push quantized model to HF, measure quality delta |
| `deploy_vllm` | `inference_opt/vllm_deployer.py` | Start vLLM server in sandbox, run benchmark, compare to baseline |
| `setup_speculative_decoding` | `inference_opt/speculative_decoding.py` | Configure Eagle-2/Medusa, benchmark speedup |
| `benchmark_serving` | `inference_opt/serving_benchmark.py` | Multi-backend comparison table |

### Step 4.1 — Quantization Pipeline

**File to create:** `agent/tools/inference_opt/quantization.py`

Critical design: always measures quality delta after quantization. Never recommend without quality report.

```python
_QUANTIZATION_SCRIPT_TEMPLATE = '''
# method: {method}  (gptq | awq | fp8_dynamic | gguf)
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "{model_name}"
METHOD = "{method}"
OUTPUT_REPO = "{output_repo}"
QUALITY_EVAL = {quality_eval}

print(f"Quantizing {{MODEL}} with {{METHOD}}...")

if METHOD == "awq":
    from awq import AutoAWQForCausalLM
    model = AutoAWQForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    quant_config = {{"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}}
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(OUTPUT_REPO)
    tokenizer.save_pretrained(OUTPUT_REPO)

elif METHOD == "gptq":
    from optimum.gptq import GPTQQuantizer
    from transformers import AutoModelForCausalLM
    quantizer = GPTQQuantizer(bits=4, dataset="c4", block_name_to_quantize="model.layers")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    quantized_model = quantizer.quantize_model(model, tokenizer)
    quantized_model.save_pretrained(OUTPUT_REPO)
    tokenizer.save_pretrained(OUTPUT_REPO)

elif METHOD == "fp8_dynamic":
    # FP8 quantization via llm-compressor (H100/MI300X hardware FP8 required).
    # IMPORTANT: torch_dtype=torch.float8_e4m3fn does NOT quantize — it silently
    # casts without calibration, producing wrong activations. Always use
    # llm-compressor which runs a calibration pass to determine per-tensor scales.
    from llmcompressor.transformers import SparseAutoModelForCausalLM
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # 256 calibration samples is the standard for FP8 (matches vLLM recipe)
    cal_data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    calibration = [
        tokenizer(row["messages"][0]["content"], return_tensors="pt",
                  max_length=512, truncation=True)
        for _, row in zip(range(256), cal_data)
    ]

    recipe = QuantizationModifier(
        targets="Linear",
        scheme=QuantizationScheme(weights=QuantizationArgs(num_bits=8, type="float")),
        ignore=["lm_head"],
    )
    model = SparseAutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.apply_compression(recipe=recipe, calibration_data=calibration)
    model.save_pretrained(OUTPUT_REPO, save_compressed=True)
    tokenizer.save_pretrained(OUTPUT_REPO)

# Quick perplexity eval on wikitext-2 (proxy for quality)
if QUALITY_EVAL:
    import math
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model_q = AutoModelForCausalLM.from_pretrained(OUTPUT_REPO, device_map="auto")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(dataset["text"][:100])
    encodings = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    stride = 512
    nlls = []
    for i in range(0, encodings.size(1), stride):
        chunk = encodings[:, i:i+stride]
        with torch.no_grad():
            outputs = model_q(chunk, labels=chunk)
            nlls.append(outputs.loss.item() * chunk.shape[1])
    ppl = math.exp(sum(nlls) / encodings.size(1))
    print(f"QUANT_RESULT:{json.dumps({{'method': METHOD, 'perplexity_wikitext2': round(ppl, 2), 'output_repo': OUTPUT_REPO}})}")
'''
```

**Acceptance Criteria:**
- Script template renders for all 3 methods (awq, gptq, fp8_dynamic) without syntax errors
- Tool always returns perplexity delta alongside throughput delta
- `output_repo` is pushed to HF Hub (uses existing HF token from session)

---

## Phase 5: Multimodal + VLA
**Duration:** 2 weeks  
**Goal:** Agent handles visual token compression for multimodal models and fast/slow split for VLA real-time inference.

### Key Tools

| Tool | File | What it does |
|---|---|---|
| `compress_visual_tokens` | `multimodal_opt/visual_token_compressor.py` | Apply pooling/Q-Former to reduce visual tokens, measure quality Δ |
| `profile_action_latency` | `vla_opt/action_latency_profiler.py` | Measure VLA action head end-to-end latency with P99 |
| `setup_fast_slow_system` | `vla_opt/fast_slow_splitter.py` | Split LLM planning from reactive MLP control |

### VLA Constraint — Hard Real-Time Check

Every VLA optimization tool must enforce this check:

```python
def validate_realtime_constraint(latency_ms: float, robot_type: str) -> dict:
    CONSTRAINTS = {
        "manipulation": 50,   # robot arm: 50ms max
        "locomotion": 20,     # walking robot: 20ms max
        "general": 100,       # default
    }
    limit = CONSTRAINTS.get(robot_type, 100)
    return {
        "passes": latency_ms <= limit,
        "latency_ms": latency_ms,
        "limit_ms": limit,
        "margin_ms": limit - latency_ms,
        "warning": None if latency_ms <= limit else (
            f"Latency {latency_ms}ms EXCEEDS {robot_type} limit of {limit}ms. "
            f"Must use quantization + CUDA graphs to reduce by {latency_ms - limit:.0f}ms."
        ),
    }
```

---

## Phase 6: Optimization State Machine
**Duration:** 2 weeks  
**Goal:** Agent tracks experiments across turns, builds Pareto frontier, persists state through context compaction.

### Step 6.1 — OptimizationContext

**File to create:** `agent/optimization/context.py`

```python
"""
Persistent optimization state that survives context compaction.
Stored in ContextManager.persistent_state under key 'optimization'.
"""
from dataclasses import dataclass, field, asdict
from typing import Literal
import json


@dataclass
class Experiment:
    id: str
    technique: str
    config: dict
    baseline_metric: float
    achieved_metric: float
    metric_name: str          # "mfu" | "ttft_ms" | "throughput_tps" | "memory_gb"
    quality_delta_pct: float  # Negative = quality loss. Must be within quality_budget.
    verdict: Literal["keep", "revert", "investigating"]
    notes: str = ""

    @property
    def improvement_pct(self) -> float:
        if self.baseline_metric == 0:
            return 0.0
        return (self.achieved_metric - self.baseline_metric) / abs(self.baseline_metric) * 100


@dataclass
class OptimizationContext:
    target: Literal["training_throughput", "inference_latency", "inference_throughput", "memory"]
    model_name: str
    hardware: str
    quality_budget: float = 0.98   # From config
    baseline: dict = field(default_factory=dict)
    current_best: dict = field(default_factory=dict)
    experiments: list[Experiment] = field(default_factory=list)

    def add_experiment(self, experiment: Experiment) -> None:
        self.experiments.append(experiment)
        if experiment.verdict == "keep":
            self._update_best(experiment)

    # Metrics where lower is better — latency, memory.
    # All other metrics (throughput, MFU) are higher-is-better.
    _MINIMIZE_METRICS: frozenset = frozenset(
        {"ttft_ms", "tbt_ms", "memory_gb", "latency_ms", "inference_latency"}
    )

    def _update_best(self, exp: Experiment) -> None:
        current_metric = self.current_best.get("metric")
        if current_metric is None:
            is_better = True
        elif exp.metric_name in self._MINIMIZE_METRICS:
            is_better = exp.achieved_metric < current_metric   # lower latency = better
        else:
            is_better = exp.achieved_metric > current_metric   # higher throughput = better

        if is_better:
            self.current_best = {
                "technique": exp.technique,
                "config": exp.config,
                "metric": exp.achieved_metric,
                "metric_name": exp.metric_name,
                "improvement_over_baseline_pct": exp.improvement_pct,
            }

    def pareto_frontier(self) -> list[dict]:
        """Return experiments on the throughput vs quality Pareto frontier."""
        kept = [e for e in self.experiments if e.verdict == "keep"]
        if not kept:
            return []
        frontier = []
        for exp in sorted(kept, key=lambda e: e.achieved_metric, reverse=True):
            dominated = any(
                e.achieved_metric >= exp.achieved_metric and
                e.quality_delta_pct >= exp.quality_delta_pct
                for e in frontier
            )
            if not dominated:
                frontier.append(exp)
        return [asdict(e) for e in frontier]

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "model_name": self.model_name,
            "hardware": self.hardware,
            "quality_budget": self.quality_budget,
            "baseline": self.baseline,
            "current_best": self.current_best,
            "experiment_count": len(self.experiments),
            "experiments_summary": [
                {
                    "id": e.id,
                    "technique": e.technique,
                    "improvement_pct": round(e.improvement_pct, 1),
                    "quality_delta_pct": round(e.quality_delta_pct, 2),
                    "verdict": e.verdict,
                }
                for e in self.experiments
            ],
            "pareto_frontier": self.pareto_frontier(),
        }

    def save_to_context_manager(self, context_manager) -> None:
        """Persist to ContextManager.persistent_state (survives compaction)."""
        context_manager.persistent_state["optimization"] = self.to_dict()

    @classmethod
    def load_from_context_manager(cls, context_manager) -> "OptimizationContext | None":
        state = context_manager.persistent_state.get("optimization")
        if not state:
            return None
        ctx = cls(
            target=state["target"],
            model_name=state["model_name"],
            hardware=state["hardware"],
            quality_budget=state["quality_budget"],
            baseline=state["baseline"],
            current_best=state["current_best"],
        )
        return ctx
```

**Acceptance Criteria:**
```python
# tests/optimization/test_optimization_context.py
from agent.optimization.context import OptimizationContext, Experiment

def test_pareto_frontier():
    ctx = OptimizationContext("inference_throughput", "llama-7b", "a100_sxm")
    ctx.add_experiment(Experiment("e1", "fp8", {}, 100, 190, "throughput_tps", -0.3, "keep"))
    ctx.add_experiment(Experiment("e2", "gptq_int4", {}, 100, 210, "throughput_tps", -2.8, "keep"))
    ctx.add_experiment(Experiment("e3", "awq_int4", {}, 100, 200, "throughput_tps", -1.4, "keep"))
    
    frontier = ctx.pareto_frontier()
    # Pareto dominance requires being worse on ALL axes simultaneously.
    # fp8  (190 tps, -0.3% quality): best quality; not dominated by awq (190 < 200 throughput)
    # gptq (210 tps, -2.8% quality): best throughput; never dominated
    # awq  (200 tps, -1.4% quality): better throughput than fp8 (200 > 190)
    #                                  better quality than gptq (-1.4 > -2.8)
    #                                  → awq IS Pareto-optimal (middle-ground point)
    technique_names = [f["technique"] for f in frontier]
    assert "fp8" in technique_names        # best quality loss
    assert "gptq_int4" in technique_names  # best throughput
    assert "awq_int4" in technique_names   # Pareto-optimal middle ground — NOT dominated

def test_survives_compaction(tmp_path):
    from unittest.mock import MagicMock
    ctx = OptimizationContext("training_throughput", "llama-7b", "h100_sxm")
    ctx.baseline = {"mfu": 0.23, "tokens_per_sec": 8500}
    
    mock_cm = MagicMock()
    mock_cm.persistent_state = {}
    ctx.save_to_context_manager(mock_cm)
    
    assert "optimization" in mock_cm.persistent_state
    assert mock_cm.persistent_state["optimization"]["baseline"]["mfu"] == 0.23
```

---

## Phase 7: Custom CUDA Kernel Generation
**Duration:** 3 weeks
**Goal:** When quantization, vLLM, and speculative decoding are exhausted, the agent can write, compile, validate, benchmark, and publish custom fused CUDA kernels — closing the gap between "use existing fast kernels" and "be the source of fast kernels."

**Gating:** This phase is conditional. It activates only when the agent has a *measured* hot kernel (from Step 2.3 Nsight profiling) that is (a) > 5% of pipeline time and (b) lacks an existing optimized implementation in flash-attention, Liger Kernel, or torch-native. Without the gate, the agent will be tempted to write kernels for ops where speedup × pipeline-fraction yields negligible end-to-end improvement (Cross-Cutting Rule 1).

This phase borrows the workflow pattern from HuggingFace's `kernels` skill (https://huggingface.co/blog/custom-cuda-kernels-agent-skills): generate a project skeleton, compile via `kernel-builder`, validate against a PyTorch reference, benchmark at two levels, publish to Kernel Hub for reuse.

---

### Step 7.1 — Kernel Skill Pack

**Pattern:** instead of one giant system-prompt section, kernel-development knowledge is split into a small `SKILL.md` (~500 tokens, just rules and workflow) plus on-demand reference files loaded via a `read_reference` tool. The `SKILL.md` is loaded when `optimization_target` includes `"kernel_dev"`; references are read only when the agent needs them.

**Directory to create:** `agent/skills/cuda-kernels/`

```text
agent/skills/cuda-kernels/
├── SKILL.md                              # Always loaded when kernel_dev active (~500 tokens)
├── scripts/
│   ├── benchmark_kernel.py               # Standard isolated-kernel benchmark
│   ├── correctness_test.py               # Compare custom kernel vs. PyTorch reference
│   └── pipeline_benchmark.py             # End-to-end pipeline speedup measurement
└── references/                           # Loaded on demand, not always-on
    ├── h100-optimization-guide.md        # SM count, shared mem, BF16 tensor cores, FP8, NVLink
    ├── a100-optimization-guide.md        # SM count, shared mem, BF16 tensor cores
    ├── l40s-optimization-guide.md
    ├── kernel-templates.md               # Vectorized loads, warp shuffle reductions, swizzle
    ├── memory-patterns.md                # Coalesced loads, shared mem banking, async copy
    ├── transformers-integration.md       # Registering ops; attn_implementation plumbing
    ├── diffusers-integration.md          # Pipeline injection patterns
    └── troubleshooting.md                # Compile errors, illegal memory access, race bugs
```

`SKILL.md` mandates (each enforced by tool-side checks where possible):

- Always start from a kernel template; never from a blank file
- Always benchmark against a `torch.nn.functional.<reference>` op for correctness AND speedup
- Always include both isolated AND end-to-end benchmarks (Cross-Cutting Rule 1)
- Always target a single `(architecture, dtype)` pair per kernel — no `#ifdef` sprawl
- Always emit `build.toml` with `cuda-capabilities` set to a single value (e.g., `"9.0"` for H100), not a range

---

### Step 7.2 — `generate_cuda_kernel` Tool

**File to create:** `agent/tools/kernel_gen/generate_kernel.py`

Given a target operation (e.g., "fused RMSNorm + residual add"), the tool:

1. Loads the relevant references from the skill pack (`<arch>-optimization-guide.md`, `kernel-templates.md`, integration guide for the consuming framework)
2. Generates a kernel project skeleton:
   ```text
   generated_kernels/<op_name>_<arch>/
   ├── build.toml                  # cuda-capabilities = ["9.0"] for H100
   ├── kernel_src/<op_name>.cu     # The kernel itself
   ├── torch-ext/torch_binding.cpp # Registers the op so torch.compile sees it
   └── benchmarks/
       ├── isolated.py             # Component speedup
       └── pipeline.py             # End-to-end speedup
   ```
3. Compiles via `kernel-builder` (HF tool, pip-installable)
4. Runs correctness test against the PyTorch reference (max abs diff < 1e-3 for bf16; user-tunable per op)
5. Runs isolated benchmark + pipeline benchmark and returns BOTH per Rule 1, with the Amdahl-predicted end-to-end vs. measured end-to-end

```python
async def generate_cuda_kernel_handler(args: dict) -> tuple[str, bool]:
    op_spec = args["op_spec"]              # e.g., "fused_rmsnorm_residual"
    target_arch = args["target_arch"]      # "h100" → cuda-capabilities = ["9.0"]
    dtype = args.get("dtype", "bfloat16")
    reference_op = args["reference_op"]    # PyTorch ground truth, e.g., "F.rms_norm"
    pipeline_script = args.get("pipeline_script")      # for end-to-end measurement
    component_fraction = args.get("component_fraction") # baseline % time spent in this op
    tolerance = args.get("tolerance", 1e-3)

    # Steps orchestrated by kernel_gen/orchestrator.py:
    # 1. Load skill references (SKILL.md + arch guide + kernel-templates.md)
    # 2. LLM generates kernel source + binding + build.toml (per skill mandates)
    # 3. Compile with kernel-builder; surface build errors verbatim for repair
    # 4. correctness_test.py — abort if max_abs_diff > tolerance
    # 5. isolated benchmark — component_speedup
    # 6. pipeline benchmark — end_to_end_speedup
    # 7. Compute Amdahl prediction; flag if deviation > 10% (Rule 1)
    # 8. Return unified result schema (Rule 1) + path to compiled artifact
    ...
```

---

### Step 7.3 — Kernel Hub Publisher

**File to create:** `agent/tools/kernel_gen/publish_kernel.py`

Pushes the compiled kernel to HF Hub so future sessions (and other users) load it via `kernels.get_kernel("user/op-name")` with **no recompilation** — the Hub stores pre-built variants for the (Python, PyTorch, CUDA) matrix. Uses the HF token already on the session.

```python
PUBLISH_KERNEL_TOOL_SPEC = ToolSpec(
    name="publish_kernel_to_hub",
    description=(
        "Upload a compiled and benchmark-validated custom kernel to HF Hub. "
        "Future sessions load it via kernels.get_kernel(repo_id) with no recompilation. "
        "Only call this AFTER generate_cuda_kernel reports passing correctness AND "
        "Amdahl-consistent end-to-end speedup (deviation_from_amdahl_pct < 10)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "kernel_dir": {"type": "string", "description": "Output dir from generate_cuda_kernel"},
            "repo_id": {"type": "string", "description": "Target HF Hub repo, e.g., 'user/llama3-rmsnorm-h100'"},
            "private": {"type": "boolean", "default": True},
        },
        "required": ["kernel_dir", "repo_id"],
    },
    ...
)
```

---

### Step 7.4 — Acceptance Workflow

End-to-end test for the phase:

```text
User: "RMSNorm is 7% of my Llama-3-8B inference pipeline on H100. Can we write a custom kernel?"

Agent:
1. profile_with_nsight(profiler="ncu", kernel_filter="rms_norm")
   → existing kernel: occupancy 38%, register spill, 1.2 TB/s achieved (36% of H100 measured peak)
   → diagnosis: memory-bandwidth-bound, ample headroom — custom kernel is justified
2. read_reference("h100-optimization-guide.md", "kernel-templates.md")
3. generate_cuda_kernel(
       op_spec="rmsnorm",
       target_arch="h100",
       dtype="bfloat16",
       reference_op="F.rms_norm",
       pipeline_script="bench_llama.py",
       component_fraction=0.07,
   )
   → correctness: max_abs_diff = 4.2e-4 ✓ (under tolerance)
   → isolated: 1.94x speedup, 76% of H100 measured bandwidth peak
   → Amdahl predicts: 1 / (0.93 + 0.07/1.94) = 1.045x e2e
   → measured pipeline: 1.05x e2e (deviation_from_amdahl_pct = 0.5%) ✓
4. publish_kernel_to_hub(repo_id="user/llama3-rmsnorm-h100")
5. OptimizationContext.add_experiment(verdict="keep", improvement_pct=5.0)
```

**Acceptance Criteria:**
- Generated kernel passes correctness within tolerance vs. PyTorch reference
- Both isolated AND pipeline speedup reported (Cross-Cutting Rule 1)
- Pipeline speedup within 10% of Amdahl prediction; otherwise flagged for investigation
- Kernel uploaded to Hub and re-importable via `kernels.get_kernel(repo_id)` in a fresh session with no recompilation
- Tool refuses to run when component_fraction × (component_speedup − 1) < 0.02 (gates against negligible-payoff kernel work)

---

## Phase 8: Scored ML-Optimization Benchmark Suite (AHE Stage C)
**Duration:** 3 weeks
**Goal:** A deterministic ≥50-task benchmark that scores end-to-end ML optimization runs. This is the **rate-limiter** — without stable, reproducible scoring, the AHE meta-loop in Phase 10 has no signal to evolve against. Per the AHE paper, the entire "evolve" mechanism collapses to noise without per-task pass/fail determinism.

**Why this comes after Phase 7:** until the Code Agent (Phases 1–5, plus Phase 7 if activated) can plausibly handle a meaningful fraction of these tasks, building the suite is premature. Phase 6 (state machine) gives us per-experiment tracking *within* a session; Phase 8 gives us a stable scoreboard *across* sessions.

---

### Step 8.1 — Task Schema

**File to create:** `tests/optimization/benchmarks/schema.py`

Each task is a YAML file under `tests/optimization/benchmarks/tasks/`:

```yaml
id: opt-014
name: "Llama-3-8B QLoRA fits A100-40GB"
hardware: a100_sxm
modality: training
input:
  base_model: "meta-llama/Llama-3-8B"
  starter_script: "tests/optimization/benchmarks/scripts/llama3_qlora_starter.py"  # intentionally non-fitting baseline
  hardware_budget: {vram_gb: 40, time_minutes: 60}
success_criteria:
  - {kind: oom_free, value: true}
  - {kind: training_loss_decreasing, window_steps: 100}
  - {kind: wall_time_under, value_minutes: 60}
quality_floor:
  kind: mmlu_drop_under
  value_pct: 1.0
  eval_dataset: "mmlu-stem-200"  # 200-question subset for speed
scoring:
  oracle: tests/optimization/benchmarks/oracles/qlora_fit.py
  pass_threshold: "all success_criteria met AND quality_floor met"
```

Schema enforced via Pydantic at load time. Invalid tasks fail-fast.

---

### Step 8.2 — Task Inventory (≥50 tasks)

Target distribution by category:

| Category | Count | Examples |
|---|---|---|
| Training fit | 12 | "Make Llama-3-8B QLoRA fit A100-40GB", "Fit Mixtral-8x7B on 4×L40S" |
| Training speed | 10 | "Reduce wall-time of GPT2-medium pretrain by 30%" |
| Inference latency | 12 | "Reduce p99 latency of Llama-3-70B serving by 25%" |
| Inference cost | 8 | "Fit Llama-3-70B on single H100 with MMLU drop <2%" |
| Multimodal | 5 | "Reduce LLaVA-1.5 inference latency by 40%" |
| VLA | 3 | "Achieve 30Hz inference for OpenVLA-7B on Jetson AGX" |

Each task ships with: starter script (intentionally suboptimal), hardware target, scoring oracle, quality floor. Starter scripts deliberately violate at least one optimization heuristic to give the agent meaningful work.

---

### Step 8.3 — Scoring Harness

**File to create:** `tests/optimization/benchmarks/runner.py`

Runs an agent against a task; captures structured result:

```python
@dataclass
class TaskResult:
    task_id: str
    passed: bool
    criteria_results: dict[str, bool]
    quality_delta: float | None      # signed: negative = quality improved
    tokens_used: int
    wall_time_s: float
    trace_path: Path                  # consumed by Phase 9 Debugger
    workspace_diff: str               # final agent edits

def run_task(agent_harness: Path, task: TaskSpec, run_id: str) -> TaskResult:
    # 1. Spawn isolated sandbox (E2B or similar)
    # 2. Mount harness; copy starter script
    # 3. Invoke agent with task description
    # 4. On agent termination: run scoring oracle
    # 5. Persist trace to runs/<run_id>/<task_id>/trace.jsonl
    # 6. Return TaskResult
```

Trace format (per-step JSONL): `{step_id, action_type, tool_name, tool_input, tool_output, llm_thought, timestamp}`. **This format is the contract with Phase 9's Agent Debugger** — do not change it after Phase 9 ships without a coordinated migration.

---

### Step 8.4 — Determinism Verification

Re-running the seed harness on the suite **3 times** must yield aggregate pass-rate variance <2pp. Per-task variance reported separately; flaky tasks (variance >5pp across 3 runs) are quarantined into `tasks/_quarantine/` until fixed.

```bash
# Determinism check command
uv run python -m tests.optimization.benchmarks.runner \
    --suite tasks/ --runs 3 --report determinism_report.json
```

**Acceptance Criteria:**
- ≥50 tasks defined, scored, committed
- 3 baseline runs of seed harness yield <2pp aggregate variance
- Per-task variance reported; <10% of tasks quarantined
- Trace files written in format consumable by Phase 9 Agent Debugger
- Scoring oracles deterministic: same `(task, agent_output)` → same verdict, always

---

## Phase 9: Trajectory Observability + Manifest Verification (AHE Stages E + F)
**Duration:** 3 weeks
**Goal:** Build the "eyes" of AHE. The **Agent Debugger** turns raw trajectories into structured root-cause reports. The **Manifest Verifier** grades change-manifest predictions against actual task-level deltas. Together they make Phase 10's evolve-loop falsifiable instead of vibes-based.

**Hard prerequisite:** Phase 8 stable. If Phase 8 pass-rate variance >2pp, do not start Phase 9 — the Debugger will train on noise.

---

### Step 9.1 — Trace Format (frozen contract with Phase 8)

**File to create:** `agent/optimization/meta/trace_format.py`

Pydantic models for trace schema. Phase 9 reads `runs/<run_id>/<task_id>/trace.jsonl` and parses into these models. Schema is the boundary between Phase 8 (writer) and Phase 9 (reader); breaking changes require coordinated migration.

---

### Step 9.2 — Agent Debugger

**Directory to create:** `agent/optimization/meta/debugger/`

```text
agent/optimization/meta/debugger/
├── prompt.yaml                    # Debugger's system prompt (separate slot from Code Agent)
├── tools/
│   ├── list_failed_tasks.py       # (benchmark_id, round) → list[task_id]
│   ├── read_trace.py              # (task_id, step_range) → trace fragment
│   ├── compare_traces.py          # (task_id, round_a, round_b) → structured diff
│   └── summarize_failure.py       # (task_id) → per-task report
├── debugger_agent.py              # Agent loop (separate from Code Agent's loop)
└── report_schema.py               # Pydantic models for output
```

**Same base model as Code Agent**, different prompt + tools — per AHE paper, all role agents share one base model.

Per-task report schema:

```yaml
task_id: opt-014
verdict: failed
failure_class: oom_at_step          # one of: oom | quality_below_floor | timeout | tool_error | logic_error
proximate_cause: "QLoRA bnb_4bit_compute_dtype=fp32 instead of bf16"
root_cause: "Agent unaware of bnb compute_dtype effect on memory"
evidence_steps: [12, 18, 23]        # step IDs that justify diagnosis
suggested_slot: skill                # which NexAU slot the fix likely belongs in
confidence: 0.7
```

Benchmark-level overview rolls up per-task reports into a failure-class histogram + top-N root causes (paper's "progressive disclosure" pattern).

**Acceptance Criteria:**
- On a synthetic 20-task failure suite with known seeded root causes, Debugger correctly classifies failure_class for ≥70% of cases
- Output token count per task ≤5% of input trace token count (compression target from paper §3.2)
- Benchmark-level overview ≤2K tokens regardless of suite size

---

### Step 9.3 — Manifest Verifier

**File to create:** `agent/optimization/meta/verifier.py`

Pure code module (NO LLM). Given commit-N's manifest `M_N` and round-(N+1) results, computes:

```python
@dataclass
class VerifierMetrics:
    fix_precision: float       # |fixed ∩ predicted_fix|  / |predicted_fix|
    fix_recall: float          # |fixed ∩ predicted_fix|  / |fixed|
    regression_precision: float
    regression_recall: float
    rollback_recommended: bool   # True if regressions exceed threshold
```

Tracked round-over-round. These are the metrics that tell us how reliable Phase 10 Evolve Agent contracts are. **Paper baseline: fix 33.7%/51.4%, regression 11.8%/11.1%.** Our numbers being better or worse is the most important data point about whether Phase 10's autonomy can be trusted.

---

### Step 9.4 — Manifest Discipline Audit

Cross-cutting: run verifier retroactively on all manifests authored under the AHE Stage D discipline (Phase 1 onward). Report baseline metrics on human-authored manifests before Phase 10 launches — this gives us a calibration target.

**Acceptance Criteria:**
- Verifier runs on ≥10 human-authored manifests from Phase 1+
- Baseline fix/regression precision-recall metrics committed to repo
- Debugger 70% root-cause accuracy on synthetic suite
- Per-task report compression ≤5% of input trace size

---

## Phase 10: Evolve Agent + Algorithm 1 Orchestration (AHE Stages G + H)
**Duration:** 4 weeks
**Goal:** The "hands" and "controller" of AHE. **Evolve Agent** reads Debugger reports and proposes harness edits with falsifiable manifests. **Orchestrator** runs the full `Rollout → Clean → Attribute/Rollback → Distill → Evolve → Commit` loop on the Phase 8 benchmark.

**Hard prerequisites:**
- Phase 8: scored suite stable (variance <2pp)
- Phase 9: Debugger root-cause accuracy ≥70%; Verifier baseline metrics committed
- Phase 1+ manifest discipline: ≥10 human-authored manifests on file

Without these, Phase 10 evolves against noise. Per the AHE paper, sub-additive component interactions and regression blindness are real failure modes; our preconditions are designed to keep them in scope.

---

### Step 10.1 — Workspace Isolation (Controllability Invariant)

**Directory layout:**

```text
workspace/                                 # writable by Evolve Agent ONLY
└── agent/optimization/
    ├── prompts/
    ├── tools/
    ├── middleware/
    ├── skills/
    └── memory/

workspace_snapshots/                       # read-only after write
├── round-000/                             # initial seed
├── round-001/
└── ...

runs/                                      # read-only after write (Phase 8 trace output)
└── <run_id>/<task_id>/trace.jsonl
```

Filesystem-enforced invariants per AHE paper:
- Evolve Agent writes ONLY to `workspace/agent/optimization/...`
- Code Agent runs FROM `workspace/`; everything outside is mounted read-only
- Each round commits a snapshot to `workspace_snapshots/round-<N>/`
- Any attempt to write outside `workspace/` → tool-level rejection with audit log

---

### Step 10.2 — Evolve Agent

**Directory to create:** `agent/optimization/meta/evolver/`

```text
agent/optimization/meta/evolver/
├── prompt.yaml                        # Evolve Agent's system prompt
├── tools/
│   ├── read_debugger_report.py        # (round) → DebuggerReport
│   ├── read_workspace_file.py         # (path) → str
│   ├── propose_edit.py                # (slot, file_path, diff, manifest) — must include manifest
│   └── commit_round.py                # finalize and trigger next rollout
├── evolve_agent.py                    # Agent loop
└── manifest_schema.py                 # Pydantic models for manifests
```

**Same base model**, different prompt + tools. Per AHE paper.

Tool-level constraints (enforced before LLM call returns):
- `propose_edit` REJECTS if manifest missing or schema-invalid
- `propose_edit` to `system_prompt` slot triggers held-out validation gate (per paper's −2.3pp warning) — edit goes into a staging area, validated against held-out subset before promotion
- `propose_edit` REJECTS if target path is outside `workspace/`
- `propose_edit` REJECTS if same file is edited >1x per round (forces atomic, attributable changes)

---

### Step 10.3 — Orchestrator (Algorithm 1)

**File to create:** `agent/optimization/meta/loop.py`

```python
def run_round(round_id: int, harness_path: Path, benchmark: Benchmark) -> RoundResult:
    # 1. Rollout — run Code Agent on benchmark
    rollout = benchmark.run(harness_path, round_id)

    # 2. Clean — strip non-deterministic noise from traces
    cleaned = clean_traces(rollout.traces)

    # 3. Attribute / Rollback — verify previous round's manifest predictions
    if round_id > 0:
        prev_manifest = load_manifest(round_id - 1)
        metrics = verifier.score(prev_manifest, rollout)
        log_verifier_metrics(round_id - 1, metrics)
        if metrics.rollback_recommended:
            rollback_to_snapshot(round_id - 1)
            return RoundResult.rolled_back(metrics)

    # 4. Distill — Agent Debugger produces structured reports
    debugger_report = agent_debugger.run(cleaned, round_id)

    # 5. Evolve — Evolve Agent proposes edits with manifests
    proposed_edits = evolve_agent.run(debugger_report, harness_path)

    # 6. Commit — snapshot harness + manifest
    commit_snapshot(harness_path, round_id, proposed_edits)
    return RoundResult.committed(rollout, proposed_edits)


def run_evolution(n_rounds: int = 5, compute_budget_usd: float = ...) -> EvolutionReport:
    for round_id in range(n_rounds):
        if projected_round_cost(round_id) > 1.5 * remaining_budget():
            return EvolutionReport.budget_exhausted(round_id)
        result = run_round(round_id, harness_path, benchmark)
        if result.is_rolled_back:
            log_rollback(round_id, result.metrics)
    return EvolutionReport.completed(rounds=n_rounds)
```

---

### Step 10.4 — Compute Budget Guardrails

**Paper baseline (verified):** The AHE paper reports **~32 hours wall-time for 10 iterations on the 89-task Terminal-Bench 2 benchmark** — i.e., ~3.2 hr/round on 89 tasks, all three agents sharing GPT-5.4 high-reasoning. This is the only concrete cost data point the paper provides; it does not break down per-agent or per-token.

**Our projection (50-task suite, scaled):** linear scaling by task count gives `~3.2 hr × (50/89) ≈ 1.8 hr/round` as an order-of-magnitude estimate. **However**, ML-optimization tasks have longer per-task rollouts than terminal coding tasks because real workloads (training, inference benchmarking) have non-trivial wall-time floors regardless of agent efficiency. **Expect higher than 1.8 hr/round in practice.** Treat this estimate as a lower bound until calibrated by Round 1.

**Hard rule:** measure Round 1 wall-time and token spend before launching subsequent rounds. Abort if projected total cost exceeds 1.5× remaining budget. The paper's 32-hour figure is for one benchmark, one run — our compute model must include re-runs (failed attempts, calibration, transfer evaluation in Phase 11).

```python
@dataclass
class RoundCostProfile:
    wall_time_s: float
    rollout_tokens: int      # tokens consumed across all 50 task rollouts
    debugger_tokens: int     # tokens consumed by Agent Debugger (input + output)
    evolve_tokens: int       # tokens consumed by Evolve Agent
    rounds_remaining: int

def project_remaining_cost(round1: RoundCostProfile) -> float:
    # Linear projection; revisit if rounds 2+ diverge significantly
    return round1.wall_time_s * round1.rounds_remaining
```

**Why this matters:** the paper does not amortize evolution cost across user sessions. Our project follows the same convention (per Operating Mode A in `RESEARCH_AHE_ANALYSIS.md` §1.5) — evolution is a build-time expense, not a per-session expense. Budget accordingly.

---

### Step 10.5 — Acceptance Workflow

**Calibration note (our choice, not paper-mandated):** the paper used **N=10 rounds** and achieved **+7.3pp** on Terminal-Bench 2 (89 tasks, ~32 hours total). Our acceptance bar uses **N=5 rounds** and **≥+5pp** — lower on both axes. Rationale:

- N=5 vs paper's N=10 → ~50% lower compute on the first attempt. If Round 5 result is still trending upward (positive slope across rounds 3-5), extend to N=10 in a second campaign. Better to spend half the compute, learn, then decide.
- +5pp vs paper's +7.3pp accounts for: narrower domain (ML optimization vs general coding), likely lower attribution precision in our first iteration, smaller suite (50 vs 89 tasks → less statistical power), and our tasks having higher wall-time per rollout.

If Round 5 plateaus below +5pp, **investigate before extending** — the failure mode may be elsewhere:
- Scoring noise → revisit Phase 8 determinism (variance >2pp would mask gains)
- Attribution failure → revisit Phase 9 verifier metrics (regression precision/recall trending below paper's already-weak baseline of 11.8%/11.1%)
- Sub-additive interactions → stage component additions one at a time per AHE Table 3

End-to-end test:

```text
1. Seed harness H₀ runs Phase 8 suite       → baseline pass-rate P₀
2. Run Phase 10 loop for N=5 rounds         → harness H₁, H₂, ..., H₅
3. Final harness H₅ pass-rate P₅            → measure
4. Verify P₅ ≥ P₀ + 5pp                     ← Acceptance (calibrated, see above)
5. Verify rollback rate over 5 rounds < 30% ← Acceptance
6. Verify all system_prompt edits gated      ← Acceptance
7. Audit trail: round-N manifest → verifier metrics → next-round delta
```

**Acceptance Criteria:**
- 5-round loop completes within compute budget (Round 1 calibration per Step 10.4)
- Aggregate pass-rate gain ≥+5pp over seed (calibrated below paper's +7.3pp; see rationale above)
- Rollback rate <30%
- 100% of system-prompt slot edits validated against held-out subset (per paper's −2.3pp ablation warning)
- Full audit trail per round: rollout result → debugger report → evolve manifest → next-round verifier metrics
- Compute spend stays within 1.5× of pre-round projection
- If P₅ < P₀ + 5pp: investigation report before any extension, not blind continuation

---

## Phase 11: Cross-Model Transfer Evaluation (AHE Stage I)
**Duration:** 1 week
**Goal:** Verify the auto-evolved harness from Phase 10 transfers to alternate base models. Per the AHE paper, weaker models often gain MORE from a well-evolved harness — validating that the harness encodes general engineering knowledge, not model-specific tricks. Our pass criterion is calibrated below the paper's because our domain (ML optimization) is narrower than terminal-bench.

**Hard prerequisite:** Phase 10 complete with H_final pass-rate ≥+5pp over seed.

---

### Step 11.1 — Transfer Test Harness

**File to create:** `tests/optimization/transfer/run_transfer.py`

Run final harness `H_final` on Phase 8 benchmark with:
- Original base model (control — should reproduce Phase 10 result within 1pp)
- ≥3 alternate models — at minimum: one stronger, one peer, one weaker

For each alternate model, also run the **seed harness** as that model's baseline. Transfer gain = `H_final pass-rate − seed pass-rate` for THAT model.

```python
def run_transfer_eval(
    final_harness: Path,
    seed_harness: Path,
    benchmark: Benchmark,
    models: list[str],
) -> TransferReport:
    results = {}
    for model in models:
        seed_pr   = benchmark.run_with_model(seed_harness, model).pass_rate
        final_pr  = benchmark.run_with_model(final_harness, model).pass_rate
        results[model] = TransferDelta(seed=seed_pr, final=final_pr, delta=final_pr - seed_pr)
    return TransferReport(results=results)
```

---

### Step 11.2 — Failure Mode Documentation

For any model where transfer gain is negative or below threshold, document:
- Which task classes regressed
- Whether regression correlates with a specific NexAU slot (system prompt? tool description?)
- Whether it suggests a slot that was over-fit to the original base model

This is empirical input for future evolve-loop tuning — not blocking for Phase 11 acceptance.

**Acceptance Criteria:**
- ≥3 alternate models tested
- ≥2 of 3 show transfer gain ≥+3pp over their own seed-harness baseline
- Negative transfer cases documented with task-class breakdown
- Control re-run of original model reproduces Phase 10 result within 1pp

---

## Missing Components — Implement Before MVP

These were absent from the original phases but are required for the Definition of Done to be achievable.

---

### MC-1 — Model Quality Evaluation Tool *(required: DoD specifies MMLU budget)*

Perplexity (wikitext-2) is a poor proxy — the GPTQ paper (2210.17323 §4.3) documents cases where perplexity is unchanged but task accuracy drops 5%. The Definition of Done says "no more than 1% MMLU degradation" but no tool exists to measure it.

**File to create:** `agent/tools/inference_opt/quality_eval.py`

```python
_QUALITY_EVAL_SCRIPT = '''
import subprocess, json, sys, glob

MODEL = "{model_name}"
BENCHMARKS = {benchmarks}
NUM_FEWSHOT = {num_fewshot}

subprocess.run([
    sys.executable, "-m", "lm_eval",
    "--model", "hf",
    "--model_args", f"pretrained={{MODEL}}",
    "--tasks", ",".join(BENCHMARKS),
    "--num_fewshot", str(NUM_FEWSHOT),
    "--output_path", "/tmp/eval_results",
], check=True)

results = {{}}
for f in glob.glob("/tmp/eval_results/**/*.json", recursive=True):
    with open(f) as fp:
        data = json.load(fp)
        if "results" in data:
            results.update(data["results"])
print("EVAL_RESULT:" + json.dumps(results))
'''
```

Tool spec: `evaluate_model_quality(model_name, benchmarks=["mmlu"], num_fewshot=5, hf_flavor="a100-large")`

**Wire into Phase 4** after `quantize_model` so every quantization produces a quality delta against the baseline run.

---

### MC-2 — Tool Suite Routing by `optimization_target` *(required before Phase 3)*

With ~15 tools across all phases loaded simultaneously, the LLM's effective tool selection degrades. Route by `optimization_target` to expose only the relevant suite.

**File to modify:** `agent/core/tools.py`

```python
TOOL_SUITES: dict[str | None, list[str] | None] = {
    "training":   ["lookup_hardware_specs", "search_mlsys_papers",
                   "profile_training_mfu", "tune_parallelism_topology",
                   "apply_sequence_packing", "install_flash_attention", "setup_liger_kernels"],
    "inference":  ["lookup_hardware_specs", "search_mlsys_papers",
                   "profile_inference_latency", "quantize_model", "deploy_vllm",
                   "setup_speculative_decoding", "benchmark_serving", "evaluate_model_quality"],
    "multimodal": ["lookup_hardware_specs", "search_mlsys_papers",
                   "profile_inference_latency", "compress_visual_tokens",
                   "quantize_model", "evaluate_model_quality"],
    "vla":        ["lookup_hardware_specs", "profile_action_latency", "setup_fast_slow_system"],
    None:         None,  # None = load all tools (general mode, backward compatible)
}

def create_builtin_tools(optimization_target: str | None = None) -> list:
    all_tools = [...]   # existing logic unchanged
    suite = TOOL_SUITES.get(optimization_target)
    if suite is None:
        return all_tools
    return [t for t in all_tools if t.name in suite]
```

---

### MC-3 — `model_name` Input Sanitization *(required before Phase 2)*

Every profiling and quantization tool injects `model_name` into a Python script via `.format()`. A model name containing `"`, `;`, or newlines allows script injection into the HF Jobs sandbox.

**Add to all tools that template model names** (`training_mfu.py`, `inference_latency.py`, `quantization.py`, `speculative_decoding.py`):

```python
import re

def _sanitize_model_name(model_name: str) -> str:
    if not re.match(r'^[a-zA-Z0-9_\-\./]+$', model_name):
        raise ValueError(
            f"Invalid model name '{model_name}'. "
            "Must contain only alphanumeric characters, hyphens, underscores, slashes, and dots."
        )
    return model_name

# Call at top of each handler:
model_name = _sanitize_model_name(args["model_name"])
```

---

### MC-4 — Skill-Pack Knowledge Restructuring *(structural alternative to current Step 1.1)*

The current `system_prompt_optimization_v1.yaml` is a ~200-line monolith loaded into every turn. As Phases 4–7 add architecture-specific guidance (H100 FP8 vs. MI300X, vLLM vs. SGLang, Triton vs. CUDA, transformers vs. diffusers integration), this prompt will balloon past 1500 lines — eating context budget and burying the workflow rules under reference data.

**Alternative: skill-pack pattern** (proven by HuggingFace `kernels` skill, ~550-token core)

```text
agent/skills/optimization/
├── SKILL.md                         # Always loaded — workflow + mandates only (~600 tokens)
└── references/                      # Loaded on demand via read_reference tool
    ├── roofline-analysis.md         # The detailed roofline section currently in the prompt
    ├── mfu-interpretation.md
    ├── parallelism-decision-tree.md
    ├── bottleneck-taxonomy.md
    ├── h100-guide.md                # Per-architecture
    ├── a100-guide.md
    ├── mi300x-guide.md
    ├── llm-architectures.md         # Dense / MoE / multimodal / VLA decision logic
    └── common-mistakes.md
```

A `read_reference(topic)` tool exposes references on demand. Estimated reduction: ~70% of always-on prompt tokens.

**Trade-off:**
- **Pro:** smaller always-on context; one reference can be updated without retesting the whole prompt; clean composition with the Phase 7 `cuda-kernels` skill pack
- **Con:** extra tool-call hop; agent may forget to read a reference before deciding (mitigated by `SKILL.md` mandates that name the reference per workflow step)

**Decision pending:** evaluate after Phase 1 lands. Migration trigger: if `system_prompt_optimization_v1.yaml` exceeds 800 lines OR if context budget pressure forces compaction more than once per non-trivial session.

---

## Verification Checklist (Run After Each Phase)

```bash
# After Phase 0
uv run pytest tests/unit/ tests/optimization/test_config_optimization.py -q

# After Phase 1
uv run pytest tests/optimization/test_hardware_specs.py -q
python -c "from agent.core.tools import ToolRouter; \
           tr = ToolRouter({}); \
           names = [t.name for t in tr.tools.values()]; \
           assert 'lookup_hardware_specs' in names; \
           assert 'search_mlsys_papers' in names; \
           print('Phase 1 tools OK:', names)"

# After Phase 2
uv run pytest tests/optimization/test_profiling.py -q
python -c "from agent.core.tools import ToolRouter; \
           tr = ToolRouter({}); \
           names = [t.name for t in tr.tools.values()]; \
           assert 'profile_training_mfu' in names; \
           assert 'profile_inference_latency' in names; \
           print('Phase 2 tools OK')"

# After Phase 3
uv run pytest tests/optimization/test_parallelism_tuner.py -q

# After Phase 6
uv run pytest tests/optimization/ -q
# All optimization tests pass

# Full regression
uv run pytest tests/ -q
# All tests (unit + optimization) pass
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| HF Jobs profiling output parsing fails | Medium | Phase 2 blocked | Add fallback: parse raw logs with regex if JSON marker absent |
| Flash Attention install fails in HF Jobs | High | Phase 3 partial | Make FA optional; tool falls back to PyTorch native |
| Sandbox GPU memory too small for 7B profiling | High | Phase 2 partial | Use `meta` device for model load, only move to GPU for profiling step |
| Context compaction deletes experiment history | Low (mitigated) | Phase 6 critical | Covered by `persistent_state` in Step 0.4 |
| Tool count exceeds LLM context (too many tools) | Medium | Quality degradation | Group tools into suites; only load relevant suite based on `optimization_target` in config |
| Agent recommends multiple optimizations simultaneously | Medium | Unattributable results | Enforce in system prompt: "One technique per experiment" |
| MFU severely underestimated if profiler flops used | High | False "severe bottleneck" diagnoses | Fixed: use 6\*N\*B\*S theoretical estimate — see Step 2.1 |
| `model_name` user input injected into script template | Medium | Code injection in HF sandbox | Add `re.match(r'^[a-zA-Z0-9_\-\./]+$', model_name)` before `format()` call |
| FP8 via `torch_dtype=float8_e4m3fn` is a silent no-op | High | False quality pass, zero actual speedup | Fixed: use llm-compressor with calibration pass — see Step 4.1 |
| TTFT measured as `total_time / n_tokens` — wrong metric | High | Misleading latency baselines | Fixed: use prefill approximation (forward pass only) — see Step 2.2 |
| Optimizer memory underestimated (12 vs 18 bytes/param) | Medium | FSDP2 topology recommendations that OOM | Fixed: use 18 bytes for mixed-precision (default) — see Step 3.1 |
| No quality benchmark tool beyond perplexity | High | "1% MMLU budget" in DoD is unmeasurable | Add `evaluate_model_quality` tool — see Missing Components |
| `optimization_target=None` falls through to v3 general prompt | Medium | Wrong system prompt loaded in general mode | Wire config → ContextManager prompt selection explicitly |
| Vendor peak ≠ measured peak (thermal/MIG/power-cap) | High | False "severe bottleneck" diagnoses on hardware already at its real ceiling | Fixed: Cross-Cutting Rule 2 + Step 1.3 `measure_peak_throughput` cached in OptimizationContext |
| Component speedup reported without end-to-end verification | High | Agent claims wins (1.88× kernel) the user does not see (1.06× e2e) | Fixed: Cross-Cutting Rule 1 — every optimization tool returns BOTH numbers + Amdahl deviation check |
| Nsight requires `cap_sys_admin` not granted in default container | Medium | Step 2.3 silently fails or returns empty profiles | Document working HF Jobs flavor in tool description; surface privilege error verbatim, do not fall back silently |
| Custom kernel passes µbench but breaks at boundary cases | Medium (Phase 7) | Hub-published kernel corrupts inference at edge shapes | Tolerance check on a shape grid (small/medium/large + non-power-of-2) before publish; reject on any failure |
| Phase 7 invoked when Amdahl payoff < 2% e2e | Medium (Phase 7) | Wasted effort writing kernels that move no metric | Fixed: gate in Step 7.4 — `component_fraction × (component_speedup − 1) ≥ 0.02` required |

---

## Definition of Done

The agent is complete when it can execute this workflow end-to-end without human intervention:

```text
User: "Optimize inference latency for Llama-3-8B on H100. 
       Quality budget: no more than 1% MMLU degradation."

Agent:
1. lookup_hardware_specs("h100_sxm") → ridge_point=295 (theoretical), bandwidth=3350GB/s
2. measure_peak_throughput(expected_hardware="h100_sxm")
   → measured: 3180 GB/s, 920 bf16 TFLOPS, ridge_point_measured=289
   → bandwidth_efficiency_vs_vendor = 0.95 (healthy, no warning)
3. profile_inference_latency("meta-llama/Llama-3-8B", batch_sizes=[1,4,16], use_measured_peak=True) → baseline
4. Classify: TTFT=340ms, TBT=18ms → memory-bandwidth-bound at batch_size=1 ✓
5. Roofline: ~2 FLOPS/byte for decode << 289 measured ridge point → quantization is the correct lever
6. quantize_model("meta-llama/Llama-3-8B", method="fp8_dynamic") → perplexity Δ = -0.2%
7. evaluate_model_quality(quantized_model, benchmarks=["mmlu"]) → MMLU Δ = -0.4% < 1% budget ✓
8. profile_inference_latency(quantized_model)
   → component (TTFT): -47%, end_to_end (TTFT): -47%, deviation_from_amdahl_pct = 0.8% ✓
   → component (TBT): -50%, end_to_end (TBT): -50%, deviation_from_amdahl_pct = 0.6% ✓
9. OptimizationContext.add_experiment(Experiment("e1", "fp8", ..., verdict="keep"))
10. Hypothesis: speculative decoding could further reduce TTFT
11. setup_speculative_decoding(target="quantized_llama3_8b", draft="llama3_1b") → +2.8x TTFT
12. Final report: Pareto frontier with 2 solutions, recommendation based on quality budget
```

**Optional Phase 7 extension** — fired only when Phase 4 plateaus (Pareto frontier saturated) and Nsight flags a hot kernel with headroom:

```text
13. profile_with_nsight(profiler="ncu", kernel_filter="rms_norm")
    → existing kernel: occupancy 38%, register spill, 36% of measured bandwidth peak
14. read_reference("h100-optimization-guide.md", "kernel-templates.md")
15. generate_cuda_kernel(op_spec="rmsnorm", target_arch="h100", dtype="bfloat16",
                        reference_op="F.rms_norm", pipeline_script="bench.py",
                        component_fraction=0.07)
    → correctness ✓, isolated 1.94×, end_to_end 1.05×, Amdahl-consistent ✓
16. publish_kernel_to_hub("user/llama3-rmsnorm-h100")
17. Updated Pareto frontier with custom-kernel solution
```

This workflow requires Phases 0–6 to be complete; Phase 7 is gated on measured kernel-level headroom and Amdahl-justified payoff.
