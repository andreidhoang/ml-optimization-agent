"""cosmos-lab — production-grade agentic ML lifecycle library.

Six reference agents (Data, Eval, Train, Optimize, Video, Code) on a shared
governance runtime: sentinel-gated judging, MCP-OAuth identity with sub-agent
scope-down, GEPA promotion contracts, quality-budget invariants. Plugs into
`nvidia-nat` (primary harness), `ml-intern` (compat), Claude SDK (v1.1).

This is the v1 importable surface. Code physically lives under
`agent/optimization/` per the zero-diff fork strategy; `cosmos_lab.*` re-exports
so library consumers can `from cosmos_lab import ...` without depending on
upstream ml-intern import paths.

See PLAN_V2.md §0.4 for the library architecture rationale.
"""

from agent.optimization.config_ext import (
    OptimizationConfig,
    load_optimization_config,
)
from cosmos_lab.identity import (
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
    "OptimizationConfig",
    "load_optimization_config",
]

__version__ = "0.1.0.dev0"
