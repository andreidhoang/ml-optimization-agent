"""Identity, audit, and capability-scoped tool routing for cosmos-lab agents.

Phase 0 deliverable. Standalone in Phase 0 — wired into agent_loop via
TracedSession in Phase 1+.
"""

from agent.optimization.identity.audit import AuditLog
from agent.optimization.identity.identity import AgentIdentity, CapabilityDenied
from agent.optimization.identity.router import CapabilityScopedRouter

__all__ = [
    "AgentIdentity",
    "AuditLog",
    "CapabilityDenied",
    "CapabilityScopedRouter",
]
