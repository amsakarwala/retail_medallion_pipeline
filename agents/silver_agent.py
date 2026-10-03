"""
agents/silver_agent.py - Silver Agent
Purpose: execute the approved Silver layer STTM rules to transform data:
null handling, dedup, type casting, text normalization. Deterministic
executor driven by the STTM rules generated (and possibly hand-edited by
the human) in the HITL approval step.

Tools:
  inspect_task_tool(bronze_path, sttm_df) - preview bronze schema + rules
  silver_ingestion_tool(bronze_path, sttm_df, run_id) - execute cleansing
"""
import os
import re
import time
import uuid

import pandas as pd

from core.observability import AgentTrace

SILVER_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "silver")
os.makedirs(SILVER_DIR, exist_ok=True)

LINEAGE_COLS = {"_source_file", "_ingested_at", "_run_id"}
CURRENCY_RE = re.compile(r"[^\d.\-]")


def inspect_task_tool(bronze_path: str, sttm_df: pd.DataFrame) -> dict:
    table_name = os.path.splitext(os.path.basename(bronze_path))[0]
    df_preview = pd.read_parquet(bronze_path).head(5)
    base = table_name.replace("_bronze", "")
    rules = sttm_df[sttm_df["source_table"].str.contains(base, na=False) | sttm_df["source_table"].eq(table_name)].to_dict(orient="records")
    return {"table_name": table_name, "columns": list(df_preview.columns), "applicable_rules": rules}


def _apply_column_rule(series: pd.Series, logic: str) -> pd.Series:
    logic = (logic or "").lower()
    out = series.copy()

    na_mask = series.isna()
    if "trim" in logic:
        out = out.astype(str).str.strip()
        out = out.mask(na_mask | out.str.lower().isin(["nan", "none", ""]))
    if "lowercase" in logic:
        out = out.astype(str).str.lower()
        out = out.mask(na_mask)
    if "title_case" in logic or "standardize_categories" in logic:
        out = out.astype(str).str.strip().str.title()
        out = out.mask(na_mask | out.str.lower().isin(["nan", "none", ""]))
    if "strip_currency_symbols" in logic or "cast_to_float" in logic:
        def _clean_currency(v):
            if not isinstance(v, str) or v.strip().lower() in ("", "nan", "none"):
                return None
            return CURRENCY_RE.sub("", v)
        out = out.astype(str).apply(_clean_currency)
        out = pd.to_numeric(out, errors="coerce")
    if "cast_to_int" in logic:
        out = pd.to_numeric(out, errors="coerce")
    if "fillna(median)" in logic:
        med = pd.to_numeric(out, errors="coerce").median()
        out = pd.to_numeric(out, errors="coerce").fillna(med)
    if "parse_to_datetime" in logic:
        out = pd.to_datetime(out, errors="coerce", format=None)
    if "validate_email_format" in logic:
        out = out.astype(str).str.strip().str.lower()
        valid_mask = out.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False)
        out = out.where(valid_mask & ~na_mask, None)
    return out


def silver_ingestion_tool(bronze_path: str, sttm_df: pd.DataFrame, run_id: str) -> dict:
    table_name = os.path.splitext(os.path.basename(bronze_path))[0]
    base = table_name.replace("_bronze", "")
    df = pd.read_parquet(bronze_path)
    original_row_count = len(df)

    rules = sttm_df[sttm_df["source_table"].astype(str).str.contains(base, na=False) | sttm_df["source_table"].eq(table_name)]

    approved_cols = [c for c in df.columns if c not in LINEAGE_COLS]
    col_rules = rules[rules["source_col"] != "*"]
    for _, rule in col_rules.iterrows():
        col = rule["source_col"]
        if col in df.columns:
            df[col] = _apply_column_rule(df[col], rule.get("transformation_logic", ""))

    # drop fully-blank rows and exact duplicates (table-level rule)
    df = df.dropna(how="all", subset=approved_cols)
    df = df.drop_duplicates(subset=approved_cols, keep="first")

    df = df[approved_cols].copy()
    df.insert(0, "_silver_id", [str(uuid.uuid4()) for _ in range(len(df))])
    df["_source_bronze"] = table_name
    df["_run_id"] = run_id

    out_path = os.path.join(SILVER_DIR, f"{base}_silver.parquet")
    df.to_parquet(out_path, index=False)

    return {
        "table_name": base, "out_path": out_path,
        "row_count_before": original_row_count, "row_count_after": len(df),
        "rows_dropped": original_row_count - len(df),
        "columns": list(df.columns),
    }


def run_silver(run_id: str, bronze_paths: list[str], sttm_silver_rows: list[dict]) -> dict:
    trace = AgentTrace("silver_agent", run_id)
    start = time.time()
    sttm_df = pd.DataFrame(sttm_silver_rows) if sttm_silver_rows else pd.DataFrame(
        columns=["source_table", "source_col", "target_table", "target_col", "transformation_logic", "notes"])

    results = []
    tool_calls = []
    for bronze_path in bronze_paths:
        task = inspect_task_tool(bronze_path, sttm_df)
        tool_calls.append({"tool": "inspect_task_tool", "table": task["table_name"]})
        res = silver_ingestion_tool(bronze_path, sttm_df, run_id)
        tool_calls.append({"tool": "silver_ingestion_tool", "table": res["table_name"],
                            "rows_dropped": res["rows_dropped"]})
        results.append(res)

    out_paths = [r["out_path"] for r in results]
    trace.record(
        input_={"bronze_paths": bronze_paths},
        plan="For each Bronze table: inspect_task_tool -> silver_ingestion_tool (null handling, dedup, casting, standardization)",
        tool_calls=tool_calls,
        output={"silver_paths": out_paths,
                "rows_dropped": {r["table_name"]: r["rows_dropped"] for r in results}},
        duration_seconds=time.time() - start, status="success",
        reasoning_steps=[f"Cleansed {len(bronze_paths)} table(s); dropped exact duplicates and blank rows per approved STTM rules"],
    )
    return {"silver_paths": out_paths, "results": results}
