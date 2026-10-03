"""
agents/bronze_agent.py - Bronze Agent
Purpose: execute the approved Bronze layer STTM rules to transform data
(raw CSV -> typed Parquet, no cleansing). Deterministic executor - no LLM.

Tools:
  inspect_task_tool(csv_path, sttm_df) - preview csv + relevant STTM rules
  bronze_ingestion_tool(csv_path, sttm_df, run_id) - execute ingestion
"""
import os
import time

import pandas as pd

from core.observability import AgentTrace

BRONZE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "bronze")
os.makedirs(BRONZE_DIR, exist_ok=True)


def inspect_task_tool(csv_path: str, sttm_df: pd.DataFrame) -> dict:
    table_name = os.path.splitext(os.path.basename(csv_path))[0]
    df_preview = pd.read_csv(csv_path, nrows=5)
    rules = sttm_df[sttm_df["source_table"] == table_name].to_dict(orient="records")
    return {"table_name": table_name, "columns": list(df_preview.columns), "applicable_rules": rules}


def bronze_ingestion_tool(csv_path: str, sttm_df: pd.DataFrame, run_id: str) -> dict:
    table_name = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path, dtype=str)  # bronze = preserve raw as strings, minimal casting
    df["_source_file"] = os.path.basename(csv_path)
    df["_ingested_at"] = pd.Timestamp.utcnow().isoformat()
    df["_run_id"] = run_id

    out_path = os.path.join(BRONZE_DIR, f"{table_name}_bronze.parquet")
    df.to_parquet(out_path, index=False)
    return {"table_name": table_name, "out_path": out_path, "row_count": len(df), "columns": list(df.columns)}


def run_bronze(run_id: str, raw_paths: list[str], sttm_bronze_rows: list[dict]) -> dict:
    trace = AgentTrace("bronze_agent", run_id)
    start = time.time()
    sttm_df = pd.DataFrame(sttm_bronze_rows) if sttm_bronze_rows else pd.DataFrame(
        columns=["source_table", "source_col", "target_table", "target_col", "transformation_logic", "notes"])

    results = []
    tool_calls = []
    for csv_path in raw_paths:
        task = inspect_task_tool(csv_path, sttm_df)
        tool_calls.append({"tool": "inspect_task_tool", "table": task["table_name"]})
        res = bronze_ingestion_tool(csv_path, sttm_df, run_id)
        tool_calls.append({"tool": "bronze_ingestion_tool", "table": res["table_name"], "rows": res["row_count"]})
        results.append(res)

    out_paths = [r["out_path"] for r in results]
    trace.record(
        input_={"raw_paths": raw_paths},
        plan="For each raw CSV: inspect_task_tool -> bronze_ingestion_tool (typed passthrough to Parquet)",
        tool_calls=tool_calls,
        output={"bronze_paths": out_paths, "row_counts": {r["table_name"]: r["row_count"] for r in results}},
        duration_seconds=time.time() - start, status="success",
        reasoning_steps=[f"Ingested {len(raw_paths)} raw file(s) to Bronze parquet with lineage columns (_source_file, _ingested_at, _run_id)"],
    )
    return {"bronze_paths": out_paths, "results": results}
