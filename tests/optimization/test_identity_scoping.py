"""Phase 0 — identity scoping + audit log + capability-scoped router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent.optimization import OptimizationConfig, load_optimization_config
from agent.optimization.identity import (
    AgentIdentity,
    AuditLog,
    CapabilityDenied,
    CapabilityScopedRouter,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeRouter:
    """Duck-typed stand-in for ToolRouter so tests don't need HF auth or sandbox."""

    def __init__(self, tools: dict[str, tuple[str, bool]] | None = None) -> None:
        # tool_name -> (return_string, success_bool)
        self._tools = tools or {
            "read_file": ("file contents", True),
            "write_file": ("ok", True),
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


class ExplodingRouter(FakeRouter):
    async def call_tool(self, tool_name, arguments, session=None, tool_call_id=None):  # type: ignore[override]
        raise RuntimeError("kaboom")


# ---------------------------------------------------------------------------
# OptimizationConfig
# ---------------------------------------------------------------------------


def test_optimization_config_inherits_upstream_fields_with_defaults():
    cfg = OptimizationConfig(model_name="anthropic/claude-sonnet-4-6")
    # Upstream fields preserved
    assert cfg.max_iterations == 300
    assert cfg.save_sessions is True
    # Owned fields with defaults
    assert cfg.quality_budget == 0.98
    assert cfg.optimization_loop_enabled is True
    assert cfg.optimization_target is None
    assert cfg.audit_log_path.endswith("audit.jsonl")


def test_optimization_config_round_trips_via_pydantic():
    raw = {
        "model_name": "anthropic/claude-sonnet-4-6",
        "optimization_target": "throughput",
        "quality_budget": 0.95,
    }
    cfg = OptimizationConfig.model_validate(raw)
    assert cfg.optimization_target == "throughput"
    assert cfg.quality_budget == 0.95


def test_load_optimization_config_preserves_owned_fields(tmp_path):
    """Regression: upstream load_config() returns a base Config and silently
    drops owned fields. load_optimization_config must preserve them."""
    import json

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_name": "anthropic/claude-sonnet-4-6",
                "optimization_target": "latency",
                "quality_budget": 0.95,
                "trajectory_db_path": "/tmp/foo.duckdb",
            }
        )
    )
    cfg = load_optimization_config(str(cfg_path))
    assert isinstance(cfg, OptimizationConfig)
    assert cfg.optimization_target == "latency"
    assert cfg.quality_budget == 0.95
    assert cfg.trajectory_db_path == "/tmp/foo.duckdb"


def test_shipped_config_loads_with_owned_fields():
    """The actual shipped config file must round-trip through the loader."""
    cfg = load_optimization_config("configs/optimization_agent_config.json")
    assert isinstance(cfg, OptimizationConfig)
    assert cfg.quality_budget == 0.98
    assert cfg.optimization_loop_enabled is True
    assert cfg.audit_log_path.endswith("audit.jsonl")
    assert cfg.trajectory_db_path.endswith("trajectories.duckdb")


# ---------------------------------------------------------------------------
# AgentIdentity
# ---------------------------------------------------------------------------


def test_root_identity_can_call_anything():
    root = AgentIdentity.root()
    assert root.can_call("read_file")
    assert root.can_call("delete_repo")
    assert root.can_call("any_future_tool_name")


def test_scoped_identity_allows_listed_only():
    ident = AgentIdentity.scoped(
        agent_id="reader",
        display_name="ReadOnlyAgent",
        capabilities=["read_file"],
    )
    assert ident.can_call("read_file")
    assert not ident.can_call("write_file")
    assert not ident.can_call("delete_repo")


def test_scoped_identity_records_parent_chain():
    parent = AgentIdentity.root("parent")
    child = AgentIdentity.scoped(
        agent_id="child", display_name="Child", capabilities=["read_file"], parent_id=parent.agent_id
    )
    assert child.parent_id == "parent"


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------


def test_audit_log_writes_jsonl_round_trip(tmp_path: Path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record({"event": "start", "n": 1})
    log.record({"event": "end", "n": 2})
    rows = log.read_all()
    assert len(rows) == 2
    assert rows[0]["event"] == "start"
    assert rows[1]["n"] == 2


def test_audit_log_creates_parent_dirs(tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c" / "audit.jsonl"
    log = AuditLog(deep)
    log.record({"x": 1})
    assert deep.exists()
    assert deep.read_text().strip().startswith("{")


def test_audit_log_lines_are_independently_parseable(tmp_path: Path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.record({"i": i})
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 5
    for i, line in enumerate(lines):
        assert json.loads(line)["i"] == i


# ---------------------------------------------------------------------------
# CapabilityScopedRouter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_denies_unauthorized_call_and_audits(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    ident = AgentIdentity.scoped("a1", "A1", capabilities=["read_file"])
    router = CapabilityScopedRouter(FakeRouter(), ident, audit)

    with pytest.raises(CapabilityDenied):
        await router.call_tool("delete_repo", {"repo": "x/y"})

    rows = audit.read_all()
    assert len(rows) == 1
    assert rows[0]["phase"] == "denied"
    assert rows[0]["allowed"] is False
    assert rows[0]["tool"] == "delete_repo"
    assert rows[0]["agent_id"] == "a1"


@pytest.mark.asyncio
async def test_router_allows_authorized_call_and_audits_before_after(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    ident = AgentIdentity.scoped("a2", "A2", capabilities=["read_file"])
    base = FakeRouter()
    router = CapabilityScopedRouter(base, ident, audit)

    out, ok = await router.call_tool("read_file", {"path": "/x"})
    assert ok is True
    assert out == "file contents"
    assert base.calls == [("read_file", {"path": "/x"})]

    rows = audit.read_all()
    phases = [r["phase"] for r in rows]
    assert phases == ["before", "after"]
    assert rows[1]["success"] is True
    assert rows[1]["duration_ms"] >= 0
    assert rows[1]["result_summary"].startswith("file contents")


@pytest.mark.asyncio
async def test_router_audits_exception_and_reraises(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    ident = AgentIdentity.root("root")
    router = CapabilityScopedRouter(ExplodingRouter(), ident, audit)

    with pytest.raises(RuntimeError, match="kaboom"):
        await router.call_tool("read_file", {})

    rows = audit.read_all()
    assert [r["phase"] for r in rows] == ["before", "exception"]
    assert rows[1]["exception_type"] == "RuntimeError"
    assert "kaboom" in rows[1]["exception_msg"]


def test_router_filters_tool_specs_by_capability(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    ident = AgentIdentity.scoped("a3", "A3", capabilities=["read_file"])
    router = CapabilityScopedRouter(FakeRouter(), ident, audit)

    specs = router.get_tool_specs_for_llm()
    names = {s["function"]["name"] for s in specs}
    assert names == {"read_file"}


def test_router_root_sees_all_specs(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    router = CapabilityScopedRouter(FakeRouter(), AgentIdentity.root(), audit)
    names = {s["function"]["name"] for s in router.get_tool_specs_for_llm()}
    assert names == {"read_file", "write_file", "delete_repo"}


@pytest.mark.asyncio
async def test_router_args_hash_is_canonical(tmp_path: Path):
    """Reordered keys with same values must produce same args_hash."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    router = CapabilityScopedRouter(FakeRouter(), AgentIdentity.root(), audit)

    await router.call_tool("read_file", {"a": 1, "b": 2})
    await router.call_tool("read_file", {"b": 2, "a": 1})
    rows = audit.read_all()
    before_rows = [r for r in rows if r["phase"] == "before"]
    assert len(before_rows) == 2
    assert before_rows[0]["args_hash"] == before_rows[1]["args_hash"]
