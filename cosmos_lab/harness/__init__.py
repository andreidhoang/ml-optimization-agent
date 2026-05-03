"""cosmos_lab.harness — adapters that install cosmos-lab governance into agent harnesses.

Each adapter implements the contract:
  install(host, identity, audit_log) -> None
  - wraps host's tool router with CapabilityScopedRouter
  - validates host meets the contract before mutation
  - is idempotent under re-install on a fresh host (raises on double-install
    of the same wrapped router)

Adapters in this package:
  - ml_intern: install into agent.core.session.Session (HF stack, P0.5 D2)
  - nat:       install into nvidia-nat Builder (Cosmos stack, P0.5 D3 — pending)
  - claude_sdk: install into Claude Agent SDK (v1.1 — pending)

See PLAN_V2.md §0.4 for the library architecture rationale and
docs/02_current_phase.md for the current adapter being built.
"""

from cosmos_lab.harness.ml_intern import install_into_session
from cosmos_lab.harness.nat import register_as_nat_tool

__all__ = ["install_into_session", "register_as_nat_tool"]
