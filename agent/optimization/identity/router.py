"""CapabilityScopedRouter — wraps an upstream ToolRouter to enforce per-agent
capability allowlists and audit every tool invocation.

Composition (not subclass-init) chosen because ToolRouter.__init__ instantiates
the full builtin tool stack (sandbox tools require HF auth), which makes pure
unit tests hard. Wrapping any object that exposes
``call_tool(name, args, session, tool_call_id) -> (str, bool)`` and
``get_tool_specs_for_llm() -> list[dict]`` keeps tests fast and decoupled.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from agent.optimization.identity.audit import AuditLog, hash_args, iso_now
from agent.optimization.identity.identity import AgentIdentity, CapabilityDenied

logger = logging.getLogger(__name__)


class _RouterLike(Protocol):
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session: Any = None,
        tool_call_id: str | None = None,
    ) -> tuple[str, bool]: ...

    def get_tool_specs_for_llm(self) -> list[dict[str, Any]]: ...


class CapabilityScopedRouter:
    def __init__(
        self,
        base_router: _RouterLike,
        identity: AgentIdentity,
        audit_log: AuditLog,
    ) -> None:
        self._base = base_router
        self.identity = identity
        self.audit_log = audit_log

    def get_tool_specs_for_llm(self) -> list[dict[str, Any]]:
        """Filter tool specs to only those this identity can invoke."""
        all_specs = self._base.get_tool_specs_for_llm()
        return [
            spec
            for spec in all_specs
            if self.identity.can_call(spec["function"]["name"])
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session: Any = None,
        tool_call_id: str | None = None,
    ) -> tuple[str, bool]:
        started = time.monotonic()
        args_h = hash_args(arguments or {})
        base_event = {
            "ts": iso_now(),
            "agent_id": self.identity.agent_id,
            "tool": tool_name,
            "args_hash": args_h,
            "tool_call_id": tool_call_id,
        }

        if not self.identity.can_call(tool_name):
            self.audit_log.record(
                {**base_event, "phase": "denied", "allowed": False, "reason": "capability_not_granted"}
            )
            logger.warning(
                "CapabilityDenied: agent=%s tool=%s", self.identity.agent_id, tool_name
            )
            raise CapabilityDenied(
                f"agent '{self.identity.agent_id}' is not granted capability '{tool_name}'"
            )

        self.audit_log.record({**base_event, "phase": "before", "allowed": True})

        try:
            result, success = await self._base.call_tool(
                tool_name, arguments, session=session, tool_call_id=tool_call_id
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            self.audit_log.record(
                {
                    **base_event,
                    "phase": "exception",
                    "duration_ms": duration_ms,
                    "exception_type": type(exc).__name__,
                    "exception_msg": str(exc)[:500],
                }
            )
            raise

        duration_ms = int((time.monotonic() - started) * 1000)
        result_summary = (
            result[:200] if isinstance(result, str) else f"<{type(result).__name__}>"
        )
        self.audit_log.record(
            {
                **base_event,
                "phase": "after",
                "duration_ms": duration_ms,
                "success": success,
                "result_summary": result_summary,
            }
        )
        return result, success
