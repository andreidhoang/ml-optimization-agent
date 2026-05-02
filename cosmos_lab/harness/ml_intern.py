"""cosmos_lab.harness.ml_intern — install cosmos-lab governance into ml-intern Session.

Per PLAN_V2.md §0.4 library architecture: cosmos-lab is a library that plugs
into agent harnesses via thin adapters; this is the v1 compat adapter for
ml-intern's `agent.core.session.Session`. Composition only — no upstream
files are modified (Invariant 1 zero-diff).

Contract (install_into_session):
  Wraps the host Session's tool_router with `CapabilityScopedRouter` so that
  every tool call routed through the Session is governed by cosmos-lab
  identity + audit. The host Session is mutated in place; no other state
  changes.

Scope (P0.5 D2):
  This adapter only wraps the tool router. OTel span emission and lifecycle
  hook wiring land in P1 (TrajectorySink + OTelGenAIEmitter).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cosmos_lab.identity import (
    AgentIdentity,
    AuditLog,
    CapabilityScopedRouter,
)

if TYPE_CHECKING:
    # Session is imported only for type-checkers to avoid pulling in the
    # full ml-intern session module (which transitively imports backends,
    # context manager, sandbox, etc.) at adapter import time.
    from agent.core.session import Session


_INSTALLED_MARKER = "_cosmos_lab_installed"


def install_into_session(
    session: "Session",
    identity: AgentIdentity,
    audit_log: AuditLog,
) -> None:
    """Install cosmos-lab governance into an existing ml-intern Session.

    After this call, every tool invocation through `session.tool_router`
    is filtered by `identity.can_call(...)` and audited via `audit_log`.

    Args:
        session: an ml-intern `agent.core.session.Session` instance with a
            `tool_router` attribute already initialized
        identity: the `AgentIdentity` whose capabilities scope this Session
        audit_log: the `AuditLog` to which tool-call events are recorded

    Raises:
        ValueError: if the Session has no `tool_router` (cannot wrap nothing)
        RuntimeError: if cosmos-lab governance is already installed on this
            Session (use a fresh Session or detach first)

    Idempotency:
        Re-installing on the same Session raises `RuntimeError`. This is
        intentional — silent re-wrapping would shadow audit history and
        confuse capability scope reasoning. Spawn a fresh Session instead.
    """
    if getattr(session, "tool_router", None) is None:
        raise ValueError(
            "Session has no tool_router to wrap; cosmos-lab requires an "
            "initialized router (Session must be constructed with "
            "tool_router=ToolRouter(...) or equivalent)"
        )

    if getattr(session, _INSTALLED_MARKER, False):
        raise RuntimeError(
            "cosmos-lab governance is already installed on this Session; "
            "re-installation would shadow audit history. Spawn a fresh "
            "Session or detach first."
        )

    session.tool_router = CapabilityScopedRouter(
        base_router=session.tool_router,
        identity=identity,
        audit_log=audit_log,
    )
    setattr(session, _INSTALLED_MARKER, True)
