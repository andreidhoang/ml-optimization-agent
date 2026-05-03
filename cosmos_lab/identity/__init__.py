"""cosmos_lab.identity — AgentIdentity, AuditLog, CapabilityScopedRouter.

Re-export shim. Implementation lives at `agent.optimization.identity` per the
zero-diff fork strategy (see PLAN_V2.md §0.4 library architecture).

P0 ships AuthZ + audit (unsigned identity, JSONL log). P4b graduates to
MCP OAuth 2.1 + RFC 8707/8693 + Ed25519 signed log.
"""

from agent.optimization.identity import (
    AgentIdentity,
    AuditLog,
    CapabilityDenied,
    CapabilityScopedRouter,
)

__all__ = [
    "AgentIdentity",
    "AuditLog",
    "CapabilityDenied",
    "CapabilityScopedRouter",
]
