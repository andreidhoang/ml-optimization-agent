"""OptimizationConfig — extends upstream Config with cosmos-lab fields.

Subclassing Config (not modifying it) preserves the zero-diff invariant and
lets upstream evolve config defaults independently.

Use ``load_optimization_config()`` instead of upstream ``load_config()`` —
the upstream loader returns a ``Config``, which silently drops the owned
fields below.
"""

from pathlib import Path

from agent.config import (
    Config,
    _load_json_config,
    apply_slack_user_defaults,
    substitute_env_vars,
)
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class OptimizationConfig(Config):
    """cosmos-lab configuration. Inherits all upstream Config fields."""

    optimization_target: str | None = None
    target_hardware: str | None = None
    quality_budget: float = 0.98
    optimization_loop_enabled: bool = True

    audit_log_path: str = "~/.cosmos_lab/audit.jsonl"
    trajectory_db_path: str = "~/.cosmos_lab/trajectories.duckdb"


def load_optimization_config(
    config_path: str = "configs/optimization_agent_config.json",
    include_user_defaults: bool = False,
) -> OptimizationConfig:
    """Mirror of ``agent.config.load_config`` that validates as OptimizationConfig.

    Upstream's ``load_config`` returns a base ``Config``, which Pydantic builds
    by silently dropping unknown keys — owned fields like ``quality_budget``
    and ``trajectory_db_path`` would be lost. This loader applies the same
    .env + env-var substitution + Slack-defaults pipeline but validates the
    result against the subclass.
    """
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(override=False)

    raw = _load_json_config(Path(config_path))
    if include_user_defaults:
        raw = apply_slack_user_defaults(raw)
    return OptimizationConfig.model_validate(substitute_env_vars(raw))
