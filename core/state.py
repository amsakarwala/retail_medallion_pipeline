"""
core/state.py
PipelineState: the single shared object the Orchestrator creates, passes to
the Supervisor/agents, and persists to disk so a Streamlit rerun (or a
browser refresh) doesn't lose pipeline progress.
"""
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

STATE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "state")
os.makedirs(STATE_DIR, exist_ok=True)


def new_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]


@dataclass
class PipelineState:
    run_id: str
    business_intent: str = ""
    raw_paths: list[str] = field(default_factory=list)
    phase: str = "profiling"  # profiling -> bronze -> silver -> gold -> done
    profile: dict[str, Any] = field(default_factory=dict)
    sttm_bronze: list[dict] = field(default_factory=list)
    sttm_silver: list[dict] = field(default_factory=list)
    sttm_gold: list[dict] = field(default_factory=list)
    approvals: dict[str, bool] = field(default_factory=dict)  # {"bronze_sttm": True, ...}
    bronze_paths: list[str] = field(default_factory=list)
    silver_paths: list[str] = field(default_factory=list)
    gold_paths: list[str] = field(default_factory=list)
    report_path: str | None = None
    report_summary: dict[str, Any] = field(default_factory=dict)
    scratchpad: dict[str, Any] = field(default_factory=dict)
    status: str = "created"  # created / running / waiting_approval / error / complete
    error: str | None = None
    llm_usage: list[dict] = field(default_factory=list)  # audit of which steps used real LLM vs fallback

    def save(self):
        path = os.path.join(STATE_DIR, f"{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return path

    @staticmethod
    def load(run_id: str) -> "PipelineState":
        path = os.path.join(STATE_DIR, f"{run_id}.json")
        with open(path) as f:
            data = json.load(f)
        return PipelineState(**data)

    @staticmethod
    def list_runs() -> list[str]:
        if not os.path.isdir(STATE_DIR):
            return []
        runs = [f[:-5] for f in os.listdir(STATE_DIR) if f.endswith(".json")]
        return sorted(runs, reverse=True)
