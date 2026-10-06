"""Append-only JSONL event log (one per instance): the observability stream.

The ``.traj.json`` remains the authoritative full trajectory for the evaluation
pipeline. This file records the lifecycle events M1's observability DoD requires
(token waterlines, compactions, recoveries, hooks, costs) and that M2's data
tooling consumes directly.
"""

import json
import time
from pathlib import Path


class EventLog:
    """Appends one JSON object per line; never rewrites history."""

    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields) -> None:
        if not self.path:
            return
        record = {"ts": round(time.time(), 3), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def read_all(self) -> list[dict]:
        """Load the events back (for tooling/tests)."""
        if not self.path or not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
