"""P0.5 D3 — smoke tests for cosmos_lab.harness.nat lightweight wrapper.

Per PLAN_V2.md §0.4.5 v5.1 architecture: nat is a deployment wrapper, not
a runtime substrate. These tests verify the registration contract works
against a duck-typed MockBuilder. Real `nvidia-nat` integration is
validated in P10 deployment.

Pattern matches D2's MockSession approach (workflow LEARN #3): the
adapter's contract is "I register one tool" — smoke test verifies that
without pulling in the full nat package.
"""

from __future__ import annotations

from typing import Any

import pytest

from cosmos_lab.harness import register_as_nat_tool
from cosmos_lab.harness.nat import (
    _COSMOS_LAB_TOOL_NAME,
    _cosmos_lab_principal_tool,
)


# ---------------------------------------------------------------------------
# Test doubles — minimal nat Builder duck types
# ---------------------------------------------------------------------------


class MockBuilder:
    """nat Builder shape for register_as_nat_tool() tests.

    Real Builder has add_function (or register_function in newer APIs).
    Provides .functions registry for idempotency check.
    """

    def __init__(self) -> None:
        self.functions: dict[str, Any] = {}

    def add_function(self, name: str, func: Any) -> None:
        self.functions[name] = func


class MockBuilderAlt:
    """nat Builder using register_function naming (future-proofing test)."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def register_function(self, name: str, func: Any) -> None:
        self.tools[name] = func


class EmptyBuilder:
    """Builder with no recognizable registration method."""


# ---------------------------------------------------------------------------
# Contract tests — registration mechanics
# ---------------------------------------------------------------------------


def test_register_adds_cosmos_lab_principal_tool() -> None:
    builder = MockBuilder()
    register_as_nat_tool(builder)
    assert _COSMOS_LAB_TOOL_NAME in builder.functions


def test_register_works_with_alternative_method_name() -> None:
    builder = MockBuilderAlt()
    register_as_nat_tool(builder)
    assert _COSMOS_LAB_TOOL_NAME in builder.tools


def test_register_raises_on_double_registration() -> None:
    builder = MockBuilder()
    register_as_nat_tool(builder)

    with pytest.raises(RuntimeError, match="already registered"):
        register_as_nat_tool(builder)


def test_register_raises_when_builder_has_no_known_method() -> None:
    builder = EmptyBuilder()
    with pytest.raises(AttributeError, match="no recognizable tool-registration"):
        register_as_nat_tool(builder)


def test_register_only_adds_one_tool() -> None:
    builder = MockBuilder()
    register_as_nat_tool(builder)
    assert len(builder.functions) == 1


# ---------------------------------------------------------------------------
# Tool callable contract — what nat will actually invoke
# ---------------------------------------------------------------------------


def test_registered_tool_is_callable() -> None:
    builder = MockBuilder()
    register_as_nat_tool(builder)
    tool = builder.functions[_COSMOS_LAB_TOOL_NAME]
    assert callable(tool)


def test_tool_returns_structured_dict_with_required_keys() -> None:
    result = _cosmos_lab_principal_tool(task="dummy task")
    required_keys = {"task", "budget_usd", "timeout_sec", "outcome", "cost_usd", "trajectory_id", "status"}
    assert required_keys.issubset(result.keys())


def test_tool_passes_task_through() -> None:
    result = _cosmos_lab_principal_tool(task="improve cosmos-reason-2 by 3pp")
    assert result["task"] == "improve cosmos-reason-2 by 3pp"


def test_tool_uses_default_budget_and_timeout() -> None:
    result = _cosmos_lab_principal_tool(task="x")
    assert result["budget_usd"] == 10.0
    assert result["timeout_sec"] == 600


def test_tool_accepts_explicit_budget_and_timeout() -> None:
    result = _cosmos_lab_principal_tool(task="x", budget_usd=400.0, timeout_sec=86400)
    assert result["budget_usd"] == 400.0
    assert result["timeout_sec"] == 86400


def test_v0_tool_returns_stub_status() -> None:
    """In P0.5 D3, the tool is a registered placeholder; real CLI invocation
    lands in P3 (PrincipalAgent v0). The stub status communicates this to
    any caller."""
    result = _cosmos_lab_principal_tool(task="x")
    assert result["status"] == "stub"
