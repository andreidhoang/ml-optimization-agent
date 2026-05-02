"""P0.5 D2 — smoke tests for cosmos_lab.harness.ml_intern adapter.

Tests verify the adapter's contract: it wraps Session.tool_router with
CapabilityScopedRouter, refuses to install on a router-less Session, and
refuses to double-install.

Uses a duck-typed `MockSession` rather than constructing a real ml-intern
Session — the adapter only touches `.tool_router`, so the smoke test
should verify that one thing without pulling in the full Session
construction stack (which requires Config, ContextManager, event_queue,
sandbox, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cosmos_lab.harness import install_into_session
from cosmos_lab.harness.ml_intern import _INSTALLED_MARKER
from cosmos_lab.identity import (
    AgentIdentity,
    AuditLog,
    CapabilityDenied,
    CapabilityScopedRouter,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeRouter:
    """Duck-typed router matching the Protocol CapabilityScopedRouter wraps."""

    def __init__(self, tools: dict[str, tuple[str, bool]] | None = None) -> None:
        self._tools = tools or {
            "read_file": ("file contents", True),
            "delete_repo": ("deleted", True),
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_tool_specs_for_llm(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}
            for name in self._tools
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session: Any = None,
        tool_call_id: str | None = None,
    ) -> tuple[str, bool]:
        self.calls.append((tool_name, arguments))
        if tool_name not in self._tools:
            return f"unknown tool {tool_name}", False
        return self._tools[tool_name]


class MockSession:
    """Duck-typed Session for adapter tests.

    Adapter only touches `.tool_router` (per its contract). A real
    `agent.core.session.Session` requires Config, ContextManager,
    event_queue, sandbox, and more — overkill for verifying the
    one-method adapter contract.
    """

    def __init__(self, tool_router: Any = None) -> None:
        self.tool_router = tool_router


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_install_wraps_router_with_capability_scoped(tmp_path: Path) -> None:
    base = FakeRouter()
    session = MockSession(tool_router=base)
    identity = AgentIdentity.scoped("a1", "A1", capabilities=["read_file"])
    audit = AuditLog(tmp_path / "audit.jsonl")

    install_into_session(session, identity, audit)

    assert isinstance(session.tool_router, CapabilityScopedRouter)
    assert session.tool_router._base is base
    assert getattr(session, _INSTALLED_MARKER) is True


def test_install_raises_when_session_has_no_router(tmp_path: Path) -> None:
    session = MockSession(tool_router=None)
    identity = AgentIdentity.root()
    audit = AuditLog(tmp_path / "audit.jsonl")

    with pytest.raises(ValueError, match="no tool_router to wrap"):
        install_into_session(session, identity, audit)


def test_install_is_not_re_installable_on_same_session(tmp_path: Path) -> None:
    session = MockSession(tool_router=FakeRouter())
    identity = AgentIdentity.root()
    audit = AuditLog(tmp_path / "audit.jsonl")

    install_into_session(session, identity, audit)

    with pytest.raises(RuntimeError, match="already installed"):
        install_into_session(session, identity, audit)


# ---------------------------------------------------------------------------
# End-to-end behavior tests (governance actually works after install)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_after_install_unauthorized_call_is_denied_and_audited(
    tmp_path: Path,
) -> None:
    base = FakeRouter()
    session = MockSession(tool_router=base)
    identity = AgentIdentity.scoped("a2", "A2", capabilities=["read_file"])
    audit = AuditLog(tmp_path / "audit.jsonl")
    install_into_session(session, identity, audit)

    with pytest.raises(CapabilityDenied):
        await session.tool_router.call_tool("delete_repo", {"repo": "x/y"})

    rows = audit.read_all()
    assert len(rows) == 1
    assert rows[0]["phase"] == "denied"
    assert rows[0]["tool"] == "delete_repo"
    assert rows[0]["agent_id"] == "a2"

    # Base router was never reached.
    assert base.calls == []


@pytest.mark.asyncio
async def test_after_install_authorized_call_passes_through_and_audits(
    tmp_path: Path,
) -> None:
    base = FakeRouter()
    session = MockSession(tool_router=base)
    identity = AgentIdentity.scoped("a3", "A3", capabilities=["read_file"])
    audit = AuditLog(tmp_path / "audit.jsonl")
    install_into_session(session, identity, audit)

    out, ok = await session.tool_router.call_tool("read_file", {"path": "/x"})

    assert ok is True
    assert out == "file contents"
    assert base.calls == [("read_file", {"path": "/x"})]

    rows = audit.read_all()
    phases = [r["phase"] for r in rows]
    assert phases == ["before", "after"]


def test_after_install_tool_specs_are_filtered_by_capability(tmp_path: Path) -> None:
    base = FakeRouter()
    session = MockSession(tool_router=base)
    identity = AgentIdentity.scoped("a4", "A4", capabilities=["read_file"])
    audit = AuditLog(tmp_path / "audit.jsonl")
    install_into_session(session, identity, audit)

    specs = session.tool_router.get_tool_specs_for_llm()
    names = {s["function"]["name"] for s in specs}

    assert names == {"read_file"}  # delete_repo filtered out
