"""cosmos_lab — agentic ML lifecycle platform built atop ml-intern.

Owned package per WORKFLOW.md. Never import-shadows or modifies upstream
agent/* modules; only extends them via subclassing or composition.
"""

from agent.optimization.config_ext import (
    OptimizationConfig,
    load_optimization_config,
)

__all__ = ["OptimizationConfig", "load_optimization_config"]
