"""AuditLog — append-only JSONL audit trail for tool invocations.

JSONL chosen over a DB so:
- audit is human-readable and `tail -f`-able
- no schema migrations needed in Phase 0
- easy to ship to S3 / Phoenix / OpenTelemetry collector later

Thread-safe via a process-local lock. For multi-process writers, prefer one
audit file per process and aggregate offline.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_args(arguments: dict[str, Any]) -> str:
    """Canonical sha256 over arguments — matches doom_loop.py's normalization."""
    canonical = json.dumps(arguments or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out
