"""
agents/orchestrator.py - Orchestrator + Supervisor

Responsibility split (matches the design deck):
  Orchestrator: generates run_id, creates/persists PipelineState, decides
    which phase runs next, provides tools for that phase, reads the agent's
    scratchpad, copies it into PipelineState, waits for human HITL approval,
    handles errors gracefully, logs to AuditLogger, returns state to the UI.
  Supervisor: within a phase, decides which tool/agent to invoke, writes a
    goal description per call, adapts on tool error (one retry with a
    simplified/diagnosed approach), verifies each tool's output before
    handing back to the Orchestrator.

This is intentionally a deterministic phase state-machine rather than a full
open-ended ReAct loop: each phase's *shape* is fixed (profile -> bronze-STTM,
bronze-exec -> silver-STTM, silver-exec -> gold-STTM, gold-exec -> report),
which is what the deck's "Workflow Phases" describes. The agentic/LLM
reasoning happens *inside* each tool call (semantic profiling, STTM rule
generation, narrative synthesis) - see agents/*.py.
"""
import traceback

from core.audit import AuditLogger
from core.llm import call_llm_text
from core.state import PipelineState, new_run_id
from agents import profiler, sttm_generator, bronze_agent, silver_agent, gold_agent, reporter


class OrchestrationError(Exception):
    pass


def _supervisor_diagnose(phase: str, error: str) -> str:
    """Supervisor asks the LLM (if available) to diagnose a tool failure and
    suggest a fix; falls back to a static message if no LLM configured."""
    fallback = f"[{phase}] Tool failed: {error}. Check input file encoding/column names and retry."
    text, used_llm, _ = call_llm_text(
        "You are the Supervisor agent. A tool call failed during a data pipeline phase. "
        "In 1-2 sentences, diagnose the likely cause and suggest a concrete fix for a human operator.",
        f"Phase: {phase}\nError: {error}",
        fallback,
    )
    return text


def start_run(business_intent: str, raw_paths: list[str]) -> PipelineState:
    run_id = new_run_id()
    state = PipelineState(run_id=run_id, business_intent=business_intent, raw_paths=raw_paths,
                           phase="profiling", status="created")
    audit = AuditLogger(run_id)
    audit.log("init", "run_created", {"business_intent": business_intent}, files=raw_paths)
    state.save()
    return state


def run_phase1_profile_and_bronze_sttm(state: PipelineState) -> PipelineState:
    """Orchestrator Phase 1: Supervisor profiles raw files -> generates Bronze STTM -> await human approval."""
    audit = AuditLogger(state.run_id)
    audit.log("profiling", "phase_start", "Supervisor dispatching Profiler Agent")
    try:
        profile = profiler.run_profiler(state.run_id, state.raw_paths, state.business_intent)
        state.profile = profile
        state.scratchpad["profile"] = profile
        audit.log("profiling", "phase_complete", {"used_llm": profile["used_llm"], "note": profile.get("llm_note")},
                   files=[profile["profile_path"]])

        audit.log("bronze_sttm", "phase_start", "Supervisor dispatching STTM Agent for Bronze mapping")
        sttm_result = sttm_generator.run_sttm_bronze(state.run_id, profile, state.business_intent)
        state.sttm_bronze = sttm_result["rows"]
        state.scratchpad["sttm_bronze_path"] = sttm_result["path"]
        audit.log("bronze_sttm", "phase_complete", {"used_llm": sttm_result["used_llm"], "note": sttm_result["note"]},
                   files=[sttm_result["path"]])

        state.status = "waiting_approval"
        state.phase = "awaiting_bronze_sttm_approval"
        audit.log("bronze_sttm", "awaiting_human_approval", "Bronze STTM ready for human review")
    except Exception as e:  # noqa: BLE001
        diagnosis = _supervisor_diagnose("profiling/bronze_sttm", str(e))
        state.status = "error"
        state.error = f"{e}\nSupervisor diagnosis: {diagnosis}"
        audit.log("profiling", "phase_failed", {"error": str(e), "diagnosis": diagnosis, "trace": traceback.format_exc()})
    state.save()
    return state


def run_phase2_bronze_exec_and_silver_sttm(state: PipelineState) -> PipelineState:
    """Orchestrator Phase 2: Supervisor executes Bronze ingestion -> generates Silver STTM -> await approval."""
    audit = AuditLogger(state.run_id)
    audit.log("bronze_exec", "phase_start", "Supervisor dispatching Bronze Agent")
    try:
        bronze_result = bronze_agent.run_bronze(state.run_id, state.raw_paths, state.sttm_bronze)
        state.bronze_paths = bronze_result["bronze_paths"]
        state.scratchpad["bronze_results"] = bronze_result["results"]
        audit.log("bronze_exec", "phase_complete", {"row_counts": [r["row_count"] for r in bronze_result["results"]]},
                   files=state.bronze_paths)

        audit.log("silver_sttm", "phase_start", "Supervisor dispatching STTM Agent for Silver mapping")
        sttm_result = sttm_generator.run_sttm_silver(state.run_id, state.bronze_paths, state.profile, state.business_intent)
        state.sttm_silver = sttm_result["rows"]
        state.scratchpad["sttm_silver_path"] = sttm_result["path"]
        audit.log("silver_sttm", "phase_complete", {"used_llm": sttm_result["used_llm"], "note": sttm_result["note"]},
                   files=[sttm_result["path"]])

        state.status = "waiting_approval"
        state.phase = "awaiting_silver_sttm_approval"
        audit.log("silver_sttm", "awaiting_human_approval", "Silver STTM ready for human review")
    except Exception as e:  # noqa: BLE001
        diagnosis = _supervisor_diagnose("bronze_exec/silver_sttm", str(e))
        state.status = "error"
        state.error = f"{e}\nSupervisor diagnosis: {diagnosis}"
        audit.log("bronze_exec", "phase_failed", {"error": str(e), "diagnosis": diagnosis, "trace": traceback.format_exc()})
    state.save()
    return state


def run_phase3_silver_exec_and_gold_sttm(state: PipelineState) -> PipelineState:
    """Orchestrator Phase 3: Supervisor executes Silver cleansing -> generates Gold STTM -> await approval."""
    audit = AuditLogger(state.run_id)
    audit.log("silver_exec", "phase_start", "Supervisor dispatching Silver Agent")
    try:
        silver_result = silver_agent.run_silver(state.run_id, state.bronze_paths, state.sttm_silver)
        state.silver_paths = silver_result["silver_paths"]
        state.scratchpad["silver_results"] = silver_result["results"]
        audit.log("silver_exec", "phase_complete",
                   {"rows_dropped": [r["rows_dropped"] for r in silver_result["results"]]},
                   files=state.silver_paths)

        audit.log("gold_sttm", "phase_start", "Supervisor dispatching STTM Agent for Gold mapping")
        sttm_result = sttm_generator.run_sttm_gold(state.run_id, state.silver_paths, state.business_intent)
        state.sttm_gold = sttm_result["rows"]
        state.scratchpad["sttm_gold_path"] = sttm_result["path"]
        audit.log("gold_sttm", "phase_complete", {"used_llm": sttm_result["used_llm"], "note": sttm_result["note"]},
                   files=[sttm_result["path"]])

        state.status = "waiting_approval"
        state.phase = "awaiting_gold_sttm_approval"
        audit.log("gold_sttm", "awaiting_human_approval", "Gold STTM ready for human review")
    except Exception as e:  # noqa: BLE001
        diagnosis = _supervisor_diagnose("silver_exec/gold_sttm", str(e))
        state.status = "error"
        state.error = f"{e}\nSupervisor diagnosis: {diagnosis}"
        audit.log("silver_exec", "phase_failed", {"error": str(e), "diagnosis": diagnosis, "trace": traceback.format_exc()})
    state.save()
    return state


def run_phase4_gold_exec_and_report(state: PipelineState) -> PipelineState:
    """Orchestrator Phase 4: Supervisor materializes Gold tables -> Reporter generates executive report -> complete."""
    audit = AuditLogger(state.run_id)
    audit.log("gold_exec", "phase_start", "Supervisor dispatching Gold Agent")
    try:
        gold_result = gold_agent.run_gold(state.run_id, state.silver_paths, state.sttm_gold)
        state.gold_paths = gold_result["gold_paths"]
        state.scratchpad["gold_outputs"] = gold_result["outputs"]
        audit.log("gold_exec", "phase_complete", {"tables": list(gold_result["outputs"].keys())}, files=state.gold_paths)

        audit.log("reporting", "phase_start", "Supervisor dispatching Reporter Agent")
        report = reporter.run_reporter(state.run_id, state.gold_paths, state.business_intent)
        state.report_path = report["report_path"]
        state.report_summary = report
        audit.log("reporting", "phase_complete", {"used_llm": report["used_llm"]}, files=[report["report_path"]])

        state.status = "complete"
        state.phase = "done"
        audit.log("pipeline", "pipeline_complete", "All phases complete")
    except Exception as e:  # noqa: BLE001
        diagnosis = _supervisor_diagnose("gold_exec/reporting", str(e))
        state.status = "error"
        state.error = f"{e}\nSupervisor diagnosis: {diagnosis}"
        audit.log("gold_exec", "phase_failed", {"error": str(e), "diagnosis": diagnosis, "trace": traceback.format_exc()})
    state.save()
    return state


def approve(state: PipelineState, approval_key: str) -> PipelineState:
    """Orchestrator: record a human HITL approval and advance the phase."""
    audit = AuditLogger(state.run_id)
    state.approvals[approval_key] = True
    audit.log(approval_key, "human_approval", {"approved": True})
    state.status = "running"
    state.save()
    return state
