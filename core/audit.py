"""
core/audit.py
AuditLogger - append-only log of phase start/complete/fail, file paths
produced, and the decision rationale at each step. This is the compliance /
"what happened" trail, separate from AgentTrace (which is the "how the
agent reasoned" trail in observability.py).
"""
import json
import os
import time

AUDIT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "audit")
os.makedirs(AUDIT_DIR, exist_ok=True)


class AuditLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.path = os.path.join(AUDIT_DIR, f"{run_id}.jsonl")

    def log(self, phase: str, event: str, detail: dict | str = "", files: list[str] | None = None):
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": self.run_id,
            "phase": phase,
            "event": event,  # e.g. "phase_start", "phase_complete", "phase_failed", "human_approval"
            "detail": detail,
            "files": files or [],
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def read_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]
