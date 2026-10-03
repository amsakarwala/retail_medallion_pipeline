"""
app.py - Streamlit UI for the Retail @Medallion Pipeline & Reporting agentic workflow.

Run: streamlit run app.py
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from core.state import PipelineState, new_run_id  # noqa: E402
from core.audit import AuditLogger  # noqa: E402
from core.observability import AgentTrace  # noqa: E402
from core.llm import llm_available  # noqa: E402
from agents import orchestrator  # noqa: E402

st.set_page_config(page_title="Retail @Medallion Pipeline", page_icon="🧱", layout="wide")

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Session state bootstrap
# ----------------------------------------------------------------------------
if "run_id" not in st.session_state:
    st.session_state.run_id = None


def get_state() -> PipelineState | None:
    if st.session_state.run_id:
        try:
            return PipelineState.load(st.session_state.run_id)
        except FileNotFoundError:
            return None
    return None


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧱 Retail @Medallion Pipeline")
    st.caption("Agentic data engineering: raw CSV → Bronze → Silver → Gold → Report")

    if llm_available():
        st.success("Groq LLM connected — agents reasoning with live LLM calls.")
    else:
        st.warning("No GROQ_API_KEY set — agents run on deterministic heuristic fallbacks. "
                   "Add a key to your `.env` for real LLM-driven profiling/STTM/reporting.")

    st.divider()
    st.markdown("### Runs")
    runs = PipelineState.list_runs()
    options = ["➕ New run"] + runs
    default_idx = 0
    if st.session_state.run_id and st.session_state.run_id in runs:
        default_idx = options.index(st.session_state.run_id)
    choice = st.selectbox("Select a run", options, index=default_idx)
    if choice == "➕ New run":
        st.session_state.run_id = None
    else:
        st.session_state.run_id = choice

    st.divider()
    st.markdown("### Tech stack")
    st.caption("Python · Pandas · PyArrow · DuckDB · Plotly · Streamlit · LangChain (Groq) · ChromaDB")

# ----------------------------------------------------------------------------
# New run setup
# ----------------------------------------------------------------------------
state = get_state()

if state is None:
    st.title("Start a new pipeline run")
    st.write(
        "Upload your retail CSV file(s) (or use the bundled synthetic messy demo data), "
        "describe the business question, then let the agents profile, map, transform, and report."
    )

    col1, col2 = st.columns(2)
    with col1:
        use_demo = st.checkbox("Use bundled synthetic demo data (sales_transactions, stores, products)", value=True)
    with col2:
        uploaded = st.file_uploader("...or upload your own CSV file(s)", type="csv", accept_multiple_files=True)

    business_intent = st.text_area(
        "Business question / intent",
        value="Understand sales performance by store, category, and time, and identify top products.",
        height=80,
    )

    if st.button("🚀 Start Pipeline", type="primary"):
        raw_paths = []
        if uploaded:
            for f in uploaded:
                path = os.path.join(RAW_DIR, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                raw_paths.append(path)
        elif use_demo:
            import glob
            demo_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
            if not demo_paths:
                from scripts.generate_sample_data import gen_sales_transactions, gen_stores, gen_products
                gen_sales_transactions()
                gen_stores()
                gen_products()
                demo_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
            raw_paths = demo_paths

        if not raw_paths:
            st.error("Please upload at least one CSV or enable the demo data option.")
        else:
            new_state = orchestrator.start_run(business_intent, raw_paths)
            st.session_state.run_id = new_state.run_id
            with st.spinner("Supervisor: profiling raw data & drafting Bronze STTM..."):
                orchestrator.run_phase1_profile_and_bronze_sttm(new_state)
            st.rerun()

    st.stop()

# ----------------------------------------------------------------------------
# Existing run - header
# ----------------------------------------------------------------------------
st.title("🧱 Retail @Medallion Pipeline & Reporting")
c1, c2, c3 = st.columns([2, 1, 1])
c1.markdown(f"**Run:** `{state.run_id}`")
c2.markdown(f"**Phase:** `{state.phase}`")
c3.markdown(f"**Status:** `{state.status}`")
st.caption(f"Business intent: {state.business_intent}")

phase_order = ["profiling", "awaiting_bronze_sttm_approval", "awaiting_silver_sttm_approval",
               "awaiting_gold_sttm_approval", "done"]
phase_labels = {"profiling": "1. Profile + Bronze STTM", "awaiting_bronze_sttm_approval": "1. Profile + Bronze STTM",
                "awaiting_silver_sttm_approval": "2. Bronze exec + Silver STTM",
                "awaiting_gold_sttm_approval": "3. Silver exec + Gold STTM", "done": "4. Gold exec + Report"}
try:
    step_idx = phase_order.index(state.phase)
except ValueError:
    step_idx = 0
st.progress((step_idx + 1) / len(phase_order), text=phase_labels.get(state.phase, state.phase))

if state.status == "error":
    st.error(f"Pipeline error: {state.error}")
    if st.button("🔁 Retry current phase"):
        state.status = "running"
        state.save()
        st.rerun()

tab_pipeline, tab_data, tab_report, tab_audit, tab_trace = st.tabs(
    ["🔧 Pipeline / Approvals", "📊 Data preview", "📄 Executive Report", "📜 Audit Log", "🕵️ Agent Traces"]
)

# ----------------------------------------------------------------------------
# TAB: Pipeline / Approvals (the HITL flow)
# ----------------------------------------------------------------------------
with tab_pipeline:
    # --- Phase 1: Profile + Bronze STTM approval ---
    if state.profile:
        with st.expander("📋 Data Profiler Agent output", expanded=(state.phase == "awaiting_bronze_sttm_approval")):
            st.caption(f"Semantic interpretation via {'LLM' if state.profile.get('used_llm') else 'heuristic fallback'}"
                       + (f" — {state.profile.get('llm_note')}" if not state.profile.get('used_llm') else ""))
            for table, info in state.profile.get("stats", {}).items():
                st.markdown(f"**{table}** — {info['row_count']} rows, {info['duplicate_row_count']} exact duplicates")
                rows = []
                for col, cstat in info["columns"].items():
                    rows.append({"column": col, "null_%": cstat["null_pct"], "unique": cstat["unique_count"],
                                 "quality_notes": ", ".join(cstat.get("quality_notes", [])),
                                 "meaning": state.profile.get("semantic", {}).get(table, {}).get("column_meanings", {}).get(col, "")})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if state.profile.get("join_key_candidates"):
                st.markdown("**🔗 Discovered join-key candidates**")
                st.dataframe(pd.DataFrame(state.profile["join_key_candidates"]), use_container_width=True, hide_index=True)

    if state.phase == "awaiting_bronze_sttm_approval":
        st.markdown("### ✅ Human review: Bronze STTM (raw → typed passthrough)")
        st.caption("Edit rules inline if needed, then approve to execute Bronze ingestion.")
        edited = st.data_editor(pd.DataFrame(state.sttm_bronze), num_rows="dynamic", use_container_width=True, key="edit_bronze")
        if st.button("✅ Approve Bronze STTM & continue", type="primary"):
            state.sttm_bronze = edited.to_dict(orient="records")
            state.save()
            state = orchestrator.approve(state, "bronze_sttm")
            with st.spinner("Bronze Agent executing ingestion; STTM Agent drafting Silver rules..."):
                orchestrator.run_phase2_bronze_exec_and_silver_sttm(state)
            st.rerun()

    # --- Phase 2: Bronze exec + Silver STTM approval ---
    if state.bronze_paths:
        with st.expander("🟫 Bronze Agent output", expanded=False):
            for res in state.scratchpad.get("bronze_results", []):
                st.write(f"**{res['table_name']}**: {res['row_count']} rows → `{os.path.basename(res['out_path'])}`")

    if state.phase == "awaiting_silver_sttm_approval":
        st.markdown("### ✅ Human review: Silver STTM (cleansing, dedup, standardization)")
        st.caption("Edit rules inline if needed, then approve to execute Silver cleansing.")
        edited = st.data_editor(pd.DataFrame(state.sttm_silver), num_rows="dynamic", use_container_width=True, key="edit_silver")
        if st.button("✅ Approve Silver STTM & continue", type="primary"):
            state.sttm_silver = edited.to_dict(orient="records")
            state.save()
            state = orchestrator.approve(state, "silver_sttm")
            with st.spinner("Silver Agent cleansing data; STTM Agent drafting Gold rules..."):
                orchestrator.run_phase3_silver_exec_and_gold_sttm(state)
            st.rerun()

    # --- Phase 3: Silver exec + Gold STTM approval ---
    if state.silver_paths:
        with st.expander("🥈 Silver Agent output", expanded=False):
            for res in state.scratchpad.get("silver_results", []):
                st.write(f"**{res['table_name']}**: {res['row_count_before']} → {res['row_count_after']} rows "
                         f"({res['rows_dropped']} dropped as duplicates/blank) → `{os.path.basename(res['out_path'])}`")

    if state.phase == "awaiting_gold_sttm_approval":
        st.markdown("### ✅ Human review: Gold STTM (business aggregations)")
        st.caption("Edit rules inline if needed, then approve to materialize Gold tables & generate the report.")
        edited = st.data_editor(pd.DataFrame(state.sttm_gold), num_rows="dynamic", use_container_width=True, key="edit_gold")
        if st.button("✅ Approve Gold STTM & generate report", type="primary"):
            state.sttm_gold = edited.to_dict(orient="records")
            state.save()
            state = orchestrator.approve(state, "gold_sttm")
            with st.spinner("Gold Agent materializing tables; Reporter Agent synthesizing insights..."):
                orchestrator.run_phase4_gold_exec_and_report(state)
            st.rerun()

    if state.status == "complete":
        st.success("🎉 Pipeline complete! See the Executive Report tab.")

# ----------------------------------------------------------------------------
# TAB: Data preview across layers
# ----------------------------------------------------------------------------
with tab_data:
    layer = st.radio("Layer", ["Raw", "Bronze", "Silver", "Gold"], horizontal=True)
    path_map = {"Raw": state.raw_paths, "Bronze": state.bronze_paths, "Silver": state.silver_paths, "Gold": state.gold_paths}
    paths = path_map[layer]
    if not paths:
        st.info(f"No {layer} data yet.")
    else:
        for p in paths:
            st.markdown(f"**{os.path.basename(p)}**")
            try:
                df = pd.read_csv(p) if p.endswith(".csv") else pd.read_parquet(p)
                st.dataframe(df.head(50), use_container_width=True)
                st.caption(f"{len(df)} rows total")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read {p}: {e}")

# ----------------------------------------------------------------------------
# TAB: Executive report
# ----------------------------------------------------------------------------
with tab_report:
    if state.report_path and os.path.exists(state.report_path):
        with open(state.report_path) as f:
            html = f.read()
        st.download_button("⬇️ Download report (HTML)", data=html, file_name=f"report_{state.run_id}.html", mime="text/html")
        st.components.v1.html(html, height=1200, scrolling=True)
    else:
        st.info("Report will appear here once the pipeline completes (Phase 4).")

# ----------------------------------------------------------------------------
# TAB: Audit log
# ----------------------------------------------------------------------------
with tab_audit:
    audit = AuditLogger(state.run_id)
    entries = audit.read_all()
    if entries:
        st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)
    else:
        st.info("No audit entries yet.")

# ----------------------------------------------------------------------------
# TAB: Agent traces (observability)
# ----------------------------------------------------------------------------
with tab_trace:
    traces = AgentTrace.all_traces_for_run(state.run_id)
    if not traces:
        st.info("No agent traces yet.")
    for t in traces:
        with st.expander(f"{t['timestamp']} — {t['agent']} ({t['status']}, {t['duration_seconds']}s)"):
            st.markdown(f"**Plan:** {t['plan']}")
            if t.get("reasoning_steps"):
                st.markdown("**Reasoning steps:**")
                for step in t["reasoning_steps"]:
                    st.markdown(f"- {step}")
            st.markdown("**Tool calls:**")
            st.json(t["tool_calls"])
            st.markdown("**Output:**")
            st.json(t["output"])
