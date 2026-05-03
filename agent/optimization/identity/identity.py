"""AgentIdentity — frozen dataclass identifying an agent and its capabilities.

Capability semantics (Phase 0): exact tool-name match plus a single wildcard
``"*"`` that grants all tools. No glob/prefix matching — keeps the allow check
trivially auditable. Richer policies can layer on top in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

WILDCARD = "*"


class CapabilityDenied(PermissionError):
    """Raised when an agent attempts to call a tool outside its capability set."""


@dataclass(frozen=True)
class AgentIdentity:
    agent_id: str
    display_name: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    parent_id: Optional[str] = None

    def can_call(self, tool_name: str) -> bool:
        return WILDCARD in self.capabilities or tool_name in self.capabilities

    @classmethod
    def root(cls, agent_id: str = "root", display_name: str = "Root") -> "AgentIdentity":
        return cls(
            agent_id=agent_id,
            display_name=display_name,
            capabilities=frozenset({WILDCARD}),
        )

    @classmethod
    def scoped(
        cls,
        agent_id: str,
        display_name: str,
        capabilities: list[str] | set[str] | frozenset[str],
        parent_id: Optional[str] = None,
    ) -> "AgentIdentity":
        return cls(
            agent_id=agent_id,
            display_name=display_name,
            capabilities=frozenset(capabilities),
            parent_id=parent_id,
        )
