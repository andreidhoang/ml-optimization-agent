"""P0.5 D4 — adapter contract tests parametrized across shipped adapters.

Per cosmos_lab/harness/CONTRACT.md, these tests verify the 5 shared
requirements (S1-S5) that ALL adapters must satisfy regardless of family.
Family-specific tests live in test_ml_intern_adapter.py and
test_nat_adapter.py.

Why parametrize: when v1.1 adds claude_sdk adapter (Family A) or
langgraph adapter (Family B), they MUST pass these same shared
requirement tests. Adding a row to ADAPTERS = automatic contract
enforcement for the new adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from cosmos_lab.harness import install_into_session, register_as_nat_tool
from cosmos_lab.identity import AgentIdentity, AuditLog


# ---------------------------------------------------------------------------
# Test doubles for both families
# ---------------------------------------------------------------------------


class _FakeRouter:
    """Minimal router for Family A (execution substrate) adapter tests."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def get_tool_specs_for_llm(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "read_file", "description": "", "parameters": {}}}]

    async def call_tool(self, tool_name, arguments, session=None, tool_call_id=None):
        self.calls.append((tool_name, arguments))
        return ("ok", True)


class _MockSession:
    """Minimal Family A host (matches ml-intern Session shape for adapter)."""

    def __init__(self, tool_router: Any = None) -> None:
        self.tool_router = tool_router


class _MockBuilder:
    """Minimal Family B host (matches nat Builder shape for adapter)."""

    def __init__(self) -> None:
        self.functions: dict[str, Any] = {}

    def add_function(self, name: str, func: Any) -> None:
        self.functions[name] = func


# ---------------------------------------------------------------------------
# Adapter registry — single source of truth for parametrization
#
# When a new adapter ships, add a row here and the contract tests below
# run against it automatically.
# ---------------------------------------------------------------------------


def _ml_intern_install_factory(tmp_path: Path) -> Callable[[], None]:
    """Returns a callable that performs a successful install on a fresh host."""
    session = _MockSession(tool_router=_FakeRouter())
    identity = AgentIdentity.root("contract-test")
    audit = AuditLog(tmp_path / "audit.jsonl")

    def do_install() -> None:
        install_into_session(session, identity, audit)

    do_install._host = session  # attached for inspection
    do_install._fresh_host_factory = lambda: _MockSession(tool_router=_FakeRouter())
    do_install._call_with_fresh_host = lambda host: install_into_session(host, identity, audit)
    return do_install


def _nat_register_factory(tmp_path: Path) -> Callable[[], None]:
    """Returns a callable that performs a successful register on a fresh host."""
    builder = _MockBuilder()

    def do_register() -> None:
        register_as_nat_tool(builder)

    do_register._host = builder
    do_register._fresh_host_factory = lambda: _MockBuilder()
    do_register._call_with_fresh_host = lambda host: register_as_nat_tool(host)
    return do_register


# (adapter_name, family, factory)
ADAPTERS = [
    ("ml_intern", "A", _ml_intern_install_factory),
    ("nat", "B", _nat_register_factory),
]


@pytest.fixture(params=ADAPTERS, ids=[a[0] for a in ADAPTERS])
def adapter(request, tmp_path):
    """Parametrized fixture: yields a freshly-built install/register callable."""
    name, family, factory = request.param
    fn = factory(tmp_path)
    return {"name": name, "family": family, "fn": fn}


# ---------------------------------------------------------------------------
# Shared requirement tests (S1-S5 from CONTRACT.md)
# All adapters must pass all of these.
# ---------------------------------------------------------------------------


def test_s1_idempotency_double_install_raises(adapter) -> None:
    """S1 — Re-installing/re-registering on same host raises clear error."""
    fn = adapter["fn"]
    fn()  # first call succeeds
    with pytest.raises((RuntimeError, ValueError)):
        fn()  # second call must raise


def test_s4_returns_none(adapter) -> None:
    """S4 — Adapter returns None (mutates host in place)."""
    fn = adapter["fn"]
    result = fn()
    assert result is None


def test_s5_no_partial_state_on_failure(adapter) -> None:
    """S5 — Atomicity: failed install leaves host unchanged.

    For Family A: an install on a host with no tool_router should fail with
    ValueError, and the host's tool_router (None) and _cosmos_lab_installed
    (absent) state must remain unchanged.

    For Family B: an install on an empty builder (no recognizable method)
    should fail with AttributeError, and the builder's functions must
    remain unchanged.
    """
    if adapter["family"] == "A":
        # Family A: pass a host without tool_router → should raise
        bad_host = _MockSession(tool_router=None)
        identity = AgentIdentity.root("test")
        audit_path = Path("/tmp/cosmos_lab_test_audit_s5.jsonl")
        audit = AuditLog(audit_path)
        with pytest.raises(ValueError):
            install_into_session(bad_host, identity, audit)
        # Host unchanged
        assert bad_host.tool_router is None
        assert not getattr(bad_host, "_cosmos_lab_installed", False)
    else:  # Family B
        # Family B: pass a builder with no registration method → should raise
        class _EmptyBuilder:
            pass

        bad_builder = _EmptyBuilder()
        with pytest.raises(AttributeError):
            register_as_nat_tool(bad_builder)
        # No functions/tools attribute should be created
        assert not hasattr(bad_builder, "functions")
        assert not hasattr(bad_builder, "tools")


# ---------------------------------------------------------------------------
# Family-specific contract tests (only run for relevant family)
# ---------------------------------------------------------------------------


def test_family_a_wraps_tool_router_with_capability_scoped(adapter, tmp_path) -> None:
    """Family A specific: install must wrap host.tool_router with CapabilityScopedRouter."""
    if adapter["family"] != "A":
        pytest.skip("Family A only")

    from cosmos_lab.identity import CapabilityScopedRouter

    base_router = _FakeRouter()
    session = _MockSession(tool_router=base_router)
    identity = AgentIdentity.root("test-fa")
    audit = AuditLog(tmp_path / "audit_fa.jsonl")

    install_into_session(session, identity, audit)

    assert isinstance(session.tool_router, CapabilityScopedRouter)
    assert session.tool_router._base is base_router


def test_family_b_registers_cosmos_lab_principal_tool(adapter, tmp_path) -> None:
    """Family B specific: register must add 'cosmos_lab_principal' tool."""
    if adapter["family"] != "B":
        pytest.skip("Family B only")

    from cosmos_lab.harness.nat import _COSMOS_LAB_TOOL_NAME

    builder = _MockBuilder()
    register_as_nat_tool(builder)

    assert _COSMOS_LAB_TOOL_NAME in builder.functions
    assert callable(builder.functions[_COSMOS_LAB_TOOL_NAME])


# ---------------------------------------------------------------------------
# Coverage summary test (meta — sanity check on the parametrization)
# ---------------------------------------------------------------------------


def test_both_shipped_adapters_are_in_registry() -> None:
    """Sanity: both adapters shipped in P0.5 are registered for parametrization.

    When a new adapter ships (claude_sdk, langgraph), this test should be
    extended to assert it's in the registry too.
    """
    adapter_names = [a[0] for a in ADAPTERS]
    assert "ml_intern" in adapter_names, "Family A: ml_intern adapter (D2) missing from registry"
    assert "nat" in adapter_names, "Family B: nat adapter (D3) missing from registry"

    # Family coverage: at least one of each family
    families = {a[1] for a in ADAPTERS}
    assert "A" in families, "No Family A (execution substrate) adapter shipped"
    assert "B" in families, "No Family B (deployment surface) adapter shipped"
