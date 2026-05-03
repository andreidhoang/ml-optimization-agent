"""cosmos_lab.harness.nat — register cosmos-lab as an invokable tool in nat workflows.

Per PLAN_V2.md §0.4.5 two-layer architecture (v5.1):
  cosmos-lab CLI is the primary entry point; nat is a *deployment wrapper*,
  not a runtime substrate. This module provides the thin registration shim
  so a nat workflow YAML can invoke `cosmos-lab principal --task <spec>` as
  a tool callable.

Why nat-as-deployment-only (not nat-as-runtime):
  ml-intern's `submission_loop` (agent_loop.py:1771) is queue-based, not
  function-based. Embedding it as runtime substrate inside a nat workflow
  requires a 1-2 week async-bridge engineering effort that v5.1 explicitly
  rejected. cosmos-lab CLI is the natural orchestrator; nat invokes it.

Cosmos team usage:

  # 1. Author a nat workflow YAML that lists cosmos_lab_principal as a tool:
  #
  #    workflow:
  #      _type: nat.react_agent
  #      tools: [cosmos_lab_principal, ...]
  #
  # 2. Register the tool with nat:
  #
  #    from cosmos_lab.harness.nat import register_as_nat_tool
  #    register_as_nat_tool(builder)
  #
  # 3. Run:  $ nat run --config-file cosmos-lab.yaml
  #
  # nat invokes `cosmos-lab principal --task <spec>` per workflow rules.
  # Real nat integration validated in P10 deployment (per v5.1).

Contract:
  register_as_nat_tool(builder)
    - Adds ONE tool to the builder named `cosmos_lab_principal`
    - The tool's invocation runs `cosmos-lab principal --task <spec>` and
      returns a structured dict {outcome, cost_usd, trajectory_id, ...}
    - Builder is mutated in place; idempotency check via _COSMOS_LAB_TOOL_NAME
"""

from __future__ import annotations

from typing import Any, Protocol


_COSMOS_LAB_TOOL_NAME = "cosmos_lab_principal"


class BuilderLike(Protocol):
    """Duck-typed nat Builder protocol.

    Real `nvidia-nat` Builder has add_function() / register_function().
    We accept either via getattr fallback so this works against both v1.6
    and any future Builder API stabilization.
    """

    def add_function(self, name: str, func: Any) -> None:  # pragma: no cover
        ...


def register_as_nat_tool(builder: BuilderLike) -> None:
    """Register cosmos-lab as an invokable tool in a nat Builder.

    After this call, the nat workflow can invoke `cosmos_lab_principal(task=...)`
    as a tool. The implementation shells out to the cosmos-lab CLI per the
    v5.1 two-layer architecture (PLAN_V2 §0.4.5).

    Args:
        builder: nat Builder (or a BuilderLike test double)

    Raises:
        RuntimeError: if cosmos-lab tool is already registered on this builder
    """
    if _is_already_registered(builder):
        raise RuntimeError(
            f"Tool '{_COSMOS_LAB_TOOL_NAME}' already registered on this builder; "
            "double-registration would shadow the existing tool. Use a fresh "
            "builder or unregister first."
        )

    add_fn = _resolve_add_function(builder)
    add_fn(_COSMOS_LAB_TOOL_NAME, _cosmos_lab_principal_tool)


def _resolve_add_function(builder: BuilderLike):
    """Find the right method on the builder to register a tool.

    nat v1.6 uses add_function; future versions may rename. Be tolerant.
    """
    for method_name in ("add_function", "register_function", "register_tool", "add_tool"):
        method = getattr(builder, method_name, None)
        if callable(method):
            return method
    raise AttributeError(
        f"Builder {type(builder).__name__} has no recognizable tool-registration "
        "method (tried: add_function, register_function, register_tool, add_tool)"
    )


def _is_already_registered(builder: BuilderLike) -> bool:
    """Best-effort idempotency check — looks at builder's known tool listing."""
    for attr in ("functions", "tools", "_functions", "_tools"):
        registry = getattr(builder, attr, None)
        if registry is not None and _COSMOS_LAB_TOOL_NAME in registry:
            return True
    return False


def _cosmos_lab_principal_tool(
    task: str,
    budget_usd: float = 10.0,
    timeout_sec: int = 600,
) -> dict[str, Any]:
    """The actual tool nat invokes — shells to cosmos-lab CLI.

    Returns a dict with outcome, cost, and trajectory pointer that nat can
    surface back to its workflow.

    NOTE: in v0 this is a placeholder that returns a structured stub. Real
    CLI invocation lands when `cosmos-lab principal` CLI ships in P3
    (PrincipalAgent v0). For P0.5 D3, the contract + registration are
    what's tested; real subprocess execution is P3 work.
    """
    return {
        "task": task,
        "budget_usd": budget_usd,
        "timeout_sec": timeout_sec,
        "outcome": "<placeholder — real cosmos-lab CLI lands in P3>",
        "cost_usd": 0.0,
        "trajectory_id": None,
        "status": "stub",
    }
