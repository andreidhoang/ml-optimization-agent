# Development Workflow

This repo is a fork of `huggingface/ml-intern` extended with an AI optimization agent.
The upstream team ships to `huggingface/ml-intern` daily. This document explains how to
build on top of that without conflicts — for both human engineers and AI agents.

---

## Remote Setup

```text
upstream  →  https://github.com/huggingface/ml-intern               (source of truth, read-only)
origin    →  https://github.com/andreidhoang/ml-optimization-agent  (your fork, push here)
```

Verify at any time:

```bash
git remote -v
```

Expected output:

```text
origin    https://github.com/andreidhoang/ml-optimization-agent.git (fetch)
origin    https://github.com/andreidhoang/ml-optimization-agent.git (push)
upstream  https://github.com/huggingface/ml-intern (fetch)
upstream  https://github.com/huggingface/ml-intern (push)
```

If `upstream` is missing, add it:

```bash
git remote add upstream https://github.com/huggingface/ml-intern.git
```

---

## Syncing Upstream Changes

Run this whenever the upstream team ships new commits (daily or before starting work):

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

**This will never overwrite your work.** See the "Why merges are always clean" section below.

To check what upstream shipped before merging:

```bash
git fetch upstream
git log upstream/main --oneline --not main  # commits in upstream not yet in your branch
git diff main upstream/main --stat          # which files changed
```

---

## The Zero-Diff Rule

> **Never modify any file that already exists in the upstream repo.**

This is the single rule that makes `git merge upstream/main` conflict-free forever.

All new code lives in paths that do not exist in upstream:

| Your path | Upstream has it? |
| --- | --- |
| `agent/optimization/` | No — safe to create |
| `agent/tools/hardware_specs.py` | No — safe to create |
| `agent/tools/profiling/` | No — safe to create |
| `agent/tools/training_opt/` | No — safe to create |
| `agent/tools/inference_opt/` | No — safe to create |
| `configs/optimization_agent_config.json` | No — safe to create |
| `agent/prompts/system_prompt_optimization_v1.yaml` | No — safe to create |
| `tests/optimization/` | No — safe to create |
| `agent/core/agent_loop.py` | **Yes — do not touch** |
| `agent/core/session.py` | **Yes — do not touch** |
| `agent/core/tools.py` | **Yes — do not touch** |
| `agent/config.py` | **Yes — do not touch** |
| `agent/context_manager/manager.py` | **Yes — do not touch** |
| `backend/` | **Yes — do not touch** |
| `frontend/` | **Yes — do not touch** |

---

## Extending Upstream Classes (Without Modifying Them)

Three upstream classes need extension. Use Python subclassing — no source changes.

### Config → OptimizationConfig

Upstream file: `agent/config.py` — **do not edit**

Your extension: `agent/optimization/config_ext.py`

```python
from agent.config import Config

class OptimizationConfig(Config):
    optimization_target: str | None = None
    target_hardware: str | None = None
    quality_budget: float = 0.98
    optimization_loop_enabled: bool = True
```

When upstream adds a new field to `Config`, you get it automatically via inheritance.

### ContextManager → OptimizationContextManager

Upstream file: `agent/context_manager/manager.py` — **do not edit**

Your extension: `agent/optimization/context_manager_ext.py`

```python
from agent.context_manager.manager import ContextManager

class OptimizationContextManager(ContextManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persistent_state: dict = {}  # survives compaction

    async def compact(self, *args, **kwargs):
        await super().compact(*args, **kwargs)
        # re-inject persistent_state after compaction
        ...
```

When upstream improves `compact()`, your subclass picks up the fix via `super().compact()`.

### ToolRouter — no subclass needed

Upstream file: `agent/core/tools.py` — **do not edit**

`ToolRouter` already has a public `register_tool()` method. Add tools after construction:

```python
from agent.core.tools import ToolRouter

router = ToolRouter(mcp_servers=config.mcpServers)
router.register_tool(HARDWARE_SPECS_TOOL_SPEC)  # your tool
router.register_tool(MLSYS_PAPERS_TOOL_SPEC)    # your tool
```

When upstream adds new tools to `create_builtin_tools()`, they appear in `router` automatically.

---

## Adding New Code

1. Create your file under one of the owned paths listed above.
2. Import from upstream freely — imports are not modifications.
3. Run existing tests to make sure nothing broke.

```bash
uv run pytest tests/unit/ -q  # upstream tests must always pass
```

---

## Why Merges Are Always Clean

`git merge` applies a **diff**, not a folder copy. It only changes lines that upstream changed.

Three cases:

**Untracked files** (e.g. `PLAN.md`, `WORKFLOW.md`) — git has never heard of them.
No merge command can touch them. They are invisible to git until you `git add` them.

**Your new files** (e.g. `agent/optimization/`) — upstream has no history for these paths.
Upstream's diff says nothing about them. Merge leaves them alone.

**Upstream files you didn't touch** (e.g. `agent/config.py`) — git does a 3-way merge:

- Base: the commit where both branches last agreed
- Upstream: changed line X → X'
- You: never changed line X
- Result: apply X → X' cleanly, no conflict

A conflict only occurs when **both you and upstream changed the same line**.
The zero-diff rule makes that impossible.

---

## What To Do When Upstream Changes a File You Extend

Example: upstream refactors `ContextManager.compact()` and renames a parameter.

1. The merge still succeeds — no conflict, because you didn't touch `manager.py`.
2. Run tests: `uv run pytest tests/ -q`
3. If a test fails, inspect what changed: `git diff upstream/main HEAD -- agent/context_manager/manager.py`
4. Fix the affected `super()` call in your subclass (`agent/optimization/context_manager_ext.py`).
5. Tests pass again.

This is the only maintenance cost of this architecture. It happens rarely and takes minutes.

---

## File Ownership Reference

| Owner | Paths | Rule |
| --- | --- | --- |
| **Upstream** | `agent/core/`, `agent/config.py`, `agent/context_manager/manager.py`, `agent/tools/*.py` (existing), `backend/`, `frontend/`, `tests/unit/` | Never modify. Pull freely. |
| **This fork** | `agent/optimization/`, `agent/tools/hardware_specs.py`, `agent/tools/profiling/`, `agent/tools/training_opt/`, `agent/tools/inference_opt/`, `agent/tools/multimodal_opt/`, `agent/tools/vla_opt/`, `agent/prompts/system_prompt_optimization_v1.yaml`, `configs/optimization_agent_config.json`, `tests/optimization/` | Full ownership. Never exists in upstream. |

---

## Quick Reference

```bash
# Sync upstream (run daily)
git fetch upstream && git merge upstream/main && git push origin main

# Check what upstream shipped
git log upstream/main --oneline --not main

# Verify tests still pass after sync
uv run pytest tests/unit/ -q

# See which files you own vs upstream
git diff upstream/main --name-only  # should only list files in "This fork" table above
```
