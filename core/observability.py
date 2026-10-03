"""
core/observability.py
AgentTrace - per-agent, per-run structured trace: input, plan, tool_calls,
reasoning_steps, output, duration, status. Written to
data/traces/trace_{agent}_{run_id}.json, matching the slide deck spec.
"""
import json
import os
import time
from contextlib import contextmanager

TRACE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "traces")
os.makedirs(TRACE_DIR, exist_ok=True)


class AgentTrace:
    def __init__(self, agent: str, run_id: str):
        self.agent = agent
        self.run_id = run_id
        self.path = os.path.join(TRACE_DIR, f"trace_{agent}_{run_id}.json")
        self._entries = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return []

    def record(self, input_: dict, plan: str, tool_calls: list[dict], output: dict,
               duration_seconds: float, status: str, reasoning_steps: list[str] | None = None):
        entry = {
            "agent": self.agent,
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input": input_,
            "plan": plan,
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps or [],
            "output": output,
            "duration_seconds": round(duration_seconds, 3),
            "status": status,  # success / failed
        }
        self._entries.append(entry)
        with open(self.path, "w") as f:
            json.dump(self._entries, f, indent=2, default=str)
        return entry

    @contextmanager
    def timed(self, input_: dict, plan: str):
        start = time.time()
        tool_calls = []
        state = {"tool_calls": tool_calls}
        try:
            yield state
            self.record(input_, plan, tool_calls, state.get("output", {}),
                        time.time() - start, "success", state.get("reasoning_steps"))
        except Exception as e:  # noqa: BLE001
            self.record(input_, plan, tool_calls, {"error": str(e)},
                        time.time() - start, "failed")
            raise

    @staticmethod
    def all_traces_for_run(run_id: str) -> list[dict]:
        out = []
        if not os.path.isdir(TRACE_DIR):
            return out
        for fname in os.listdir(TRACE_DIR):
            if fname.endswith(f"_{run_id}.json"):
                with open(os.path.join(TRACE_DIR, fname)) as f:
                    out.extend(json.load(f))
        return sorted(out, key=lambda e: e["timestamp"])
