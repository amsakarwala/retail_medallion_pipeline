# Retail @Medallion Pipeline & Reporting

An agentic data engineering workflow that ingests messy retail CSVs and
takes them through a **Medallion Architecture** (Bronze → Silver → Gold)
with human-in-the-loop (HITL) approval at every transformation step, ending
in an auto-generated executive report.

```
Raw CSVs → [Profiler Agent] → [STTM Agent: Bronze] → 🧑 approve
         → [Bronze Agent]   → [STTM Agent: Silver] → 🧑 approve
         → [Silver Agent]   → [STTM Agent: Gold]   → 🧑 approve
         → [Gold Agent]     → [Reporter Agent]     → 📄 Executive report
```

## Quick start

```bash
cd retail_medallion_pipeline
pip install -r requirements.txt
cp .env.example .env          # optional - add your Groq key, or skip and run on heuristics
python scripts/generate_sample_data.py   # optional - pipeline will auto-generate this if skipped
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

**No API key required to try it.** Every LLM-driven agent (Profiler,
STTM Agent, Reporter) has a deterministic heuristic fallback, so the whole
pipeline runs and produces a real report with zero configuration. Add a
free [Groq API key](https://console.groq.com/keys) to `.env` to upgrade
those three agents to live LLM reasoning — no code changes needed.

## Project layout

```
app.py                     Streamlit UI — the Orchestrator's human interface
core/
  state.py                 PipelineState schema + JSON persistence per run
  audit.py                 AuditLogger — compliance trail (what happened)
  observability.py         AgentTrace — reasoning trail (how each agent decided)
  memory.py                ChromaDB-backed semantic join-key discovery (+ fallback)
  llm.py                   Groq client wrapper with graceful no-key fallback
agents/
  orchestrator.py          Orchestrator + Supervisor: phase state machine, HITL, error diagnosis
  profiler.py              Data Profiler Agent
  sttm_generator.py         STTM Agent (unified Bronze/Silver/Gold rule generation)
  bronze_agent.py           Bronze Agent (raw → typed Parquet)
  silver_agent.py           Silver Agent (cleanse, dedup, standardize)
  gold_agent.py              Gold Agent (DuckDB joins/aggregations)
  reporter.py                Reporter Agent (narrative + Plotly charts + HTML report)
scripts/generate_sample_data.py   Synthetic "messy" retail data generator
data/                        raw / bronze / silver / gold / profiles / traces / audit / reports / state
```

## Design notes

- **Where the "agentic-ness" lives:** the Profiler, STTM Agent, and Reporter
  do open-ended LLM reasoning (semantic column interpretation, transformation
  rule design, insight synthesis). The Bronze/Silver/Gold execution agents
  are deterministic — they faithfully execute the human-approved STTM rules.
  This mirrors the deck's tool tables and keeps the pipeline reliable and
  auditable rather than letting an LLM improvise on production data.
- **HITL is real, not cosmetic:** the Streamlit `st.data_editor` lets you
  edit any STTM rule (add/remove/change transformation logic) before
  approving each phase — your edits are what actually get executed.
- **Every phase is traced twice:** `AuditLogger` (what ran, what files were
  produced, human approval events) and `AgentTrace` (the agent's plan, tool
  calls, reasoning steps, and duration) — both viewable in the app.

## Suggested enhancements

See the accompanying chat response for a full list of ideas (data quality
scoring, lineage graph, Slack/email alerts on pipeline failure, Great
Expectations-style validation rules, scheduling via Airflow/Prefect,
multi-file incremental loads, PDF export of the report, and more).
