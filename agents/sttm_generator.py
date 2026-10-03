"""
agents/sttm_generator.py - STTM Agent (Unified)
Purpose: design the data transformation logic by translating the user's
intent and data profile into a multi-layer plan (Bronze -> Silver -> Gold).

Tools:
  inspect_context_tool(phase, paths)     - preview source context for the requested layer
  generate_bronze_sttm_tool(profile)     - column mapping + light typing, no cleansing
  generate_silver_sttm_tool(bronze_info) - null handling, dedup, type cast, standardization
  generate_gold_sttm_tool(silver_info)   - join/aggregation rules across silver tables

Each STTM row: {source_table, source_col, target_table, target_col, transformation_logic, notes}
Output: sttm_bronze_{run_id}.csv / sttm_silver_{run_id}.csv / sttm_gold_{run_id}.csv
"""
import json
import os
import time

import pandas as pd
import pyarrow.parquet as pq

from core.llm import call_llm_json
from core.observability import AgentTrace

STTM_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "profiles")
os.makedirs(STTM_DIR, exist_ok=True)


def inspect_context_tool(phase: str, context: dict) -> dict:
    """Preview source context for the requested layer (profile.json / bronze parquet / silver parquet)."""
    if phase == "bronze":
        return context.get("profile", {})
    elif phase == "silver":
        info = {}
        for path in context.get("bronze_paths", []):
            t = pq.read_table(path)
            info[os.path.basename(path)] = {"columns": t.column_names, "num_rows": t.num_rows}
        return info
    elif phase == "gold":
        info = {}
        for path in context.get("silver_paths", []):
            t = pq.read_table(path)
            info[os.path.basename(path)] = {"columns": t.column_names, "num_rows": t.num_rows}
        return info
    return {}


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    cols = ["source_table", "source_col", "target_table", "target_col", "transformation_logic", "notes"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def _heuristic_bronze_sttm(profile: dict) -> list[dict]:
    rows = []
    for table, info in profile.get("preview", {}).items():
        for col in info["columns"]:
            rows.append({
                "source_table": table, "source_col": col,
                "target_table": f"{table}_bronze", "target_col": col,
                "transformation_logic": "cast_to_string; passthrough (no cleansing at bronze)",
                "notes": "heuristic default - raw ingestion",
            })
    return rows


def _heuristic_silver_sttm(bronze_info: dict, profile: dict) -> list[dict]:
    rows = []
    stats_by_table = {t.replace("_bronze", ""): v for t, v in profile.get("stats", {}).items()}
    for table, info in bronze_info.items():
        base_table = table.replace("_bronze.parquet", "").replace(".parquet", "")
        stats = None
        for k, v in stats_by_table.items():
            if k in base_table or base_table in k:
                stats = v
                break
        for col in info["columns"]:
            cl = col.lower()
            if col.startswith("_"):
                continue
            if "date" in cl:
                logic = "parse_to_datetime (multi-format)"
            elif "price" in cl or "amount" in cl or "cost" in cl:
                logic = "strip_currency_symbols; cast_to_float"
            elif "email" in cl:
                logic = "trim; lowercase; validate_email_format"
            elif "quantity" in cl or "qty" in cl:
                logic = "cast_to_int; fillna(median)"
            elif "name" in cl or "method" in cl:
                logic = "trim; title_case; standardize_categories"
            else:
                logic = "trim_whitespace"
            null_pct = 0
            if stats:
                col_stats = stats.get("columns", {}).get(col, {})
                null_pct = col_stats.get("null_pct", 0)
            notes = f"null_pct={null_pct}%" + ("; fillna required" if null_pct > 0 else "")
            rows.append({
                "source_table": table, "source_col": col,
                "target_table": base_table + "_silver", "target_col": col,
                "transformation_logic": logic, "notes": notes,
            })
    # dedup + pk note as a table-level row
    for table in bronze_info:
        base_table = table.replace("_bronze.parquet", "").replace(".parquet", "")
        rows.append({
            "source_table": table, "source_col": "*",
            "target_table": base_table + "_silver", "target_col": "_silver_id",
            "transformation_logic": "drop_exact_duplicates; inject surrogate key (uuid4) as first column",
            "notes": "table-level rule",
        })
    return rows


def _heuristic_gold_sttm(silver_info: dict, business_intent: str) -> list[dict]:
    rows = [
        {
            "source_table": "sales_transactions_silver, stores_silver, products_silver",
            "source_col": "StoreID/store_id, ProductName/product_name",
            "target_table": "sales_by_store_category_gold",
            "target_col": "store_name, category, region, total_quantity, total_revenue, avg_unit_price",
            "transformation_logic": "JOIN sales_transactions_silver to stores_silver ON StoreID=store_id; "
                                     "JOIN to products_silver ON ProductName=product_name; "
                                     "GROUP BY store_name, category, region; "
                                     "SUM(Quantity) AS total_quantity, SUM(Quantity*UnitPrice) AS total_revenue, "
                                     "AVG(UnitPrice) AS avg_unit_price",
            "notes": f"aggregation grain chosen to answer: {business_intent}",
        },
        {
            "source_table": "sales_transactions_silver",
            "source_col": "TransactionDate, Quantity, UnitPrice",
            "target_table": "sales_trend_by_month_gold",
            "target_col": "month, total_revenue, total_transactions",
            "transformation_logic": "GROUP BY date_trunc('month', TransactionDate); "
                                     "SUM(Quantity*UnitPrice) AS total_revenue; COUNT(*) AS total_transactions",
            "notes": "monthly trend for reporting",
        },
    ]
    return rows


def generate_bronze_sttm_tool(profile: dict, business_intent: str) -> tuple[list[dict], bool, str | None]:
    system_prompt = (
        "You are the STTM (Source-to-Target Mapping) Agent for a retail medallion pipeline, generating the "
        "BRONZE layer mapping. Bronze = raw ingestion: map every source column to a bronze target column with "
        "MINIMAL transformation (only safe type casting, no cleansing/dedup/business logic). "
        "Return JSON list of rows: [{\"source_table\":.., \"source_col\":.., \"target_table\":.., "
        "\"target_col\":.., \"transformation_logic\":.., \"notes\":..}]"
    )
    user_prompt = json.dumps({"business_intent": business_intent, "profile_preview": profile.get("preview", {})}, default=str)[:12000]
    fallback = _heuristic_bronze_sttm(profile)
    rows, used_llm, err = call_llm_json(system_prompt, user_prompt, fallback)
    if not isinstance(rows, list) or not rows:
        rows, used_llm, err = fallback, False, "LLM returned empty/invalid list; used fallback"
    return rows, used_llm, err


def generate_silver_sttm_tool(bronze_info: dict, profile: dict, business_intent: str) -> tuple[list[dict], bool, str | None]:
    system_prompt = (
        "You are the STTM Agent generating the SILVER layer mapping. Silver = cleansed, standardized, "
        "deduplicated data. For each bronze column define: null handling strategy (mean/median/mode/drop/fillna), "
        "deduplication keys, type casting, text normalization/standardization rules. Also add one table-level row "
        "with source_col='*' and target_col='_silver_id' describing the dedup + surrogate key rule. "
        "Return JSON list of rows: [{\"source_table\":.., \"source_col\":.., \"target_table\":.., "
        "\"target_col\":.., \"transformation_logic\":.., \"notes\":..}]"
    )
    user_prompt = json.dumps({
        "business_intent": business_intent, "bronze_schemas": bronze_info,
        "quality_stats": profile.get("stats", {}),
    }, default=str)[:12000]
    fallback = _heuristic_silver_sttm(bronze_info, profile)
    rows, used_llm, err = call_llm_json(system_prompt, user_prompt, fallback)
    if not isinstance(rows, list) or not rows:
        rows, used_llm, err = fallback, False, "LLM returned empty/invalid list; used fallback"
    return rows, used_llm, err


def generate_gold_sttm_tool(silver_info: dict, business_intent: str) -> tuple[list[dict], bool, str | None]:
    system_prompt = (
        "You are the STTM Agent generating the GOLD layer mapping: business-ready aggregated/joined tables "
        "that directly answer the stated business intent. Define join keys across silver tables, group-by "
        "grains, and aggregation formulas (write transformation_logic as pseudo-SQL). "
        "Return JSON list of rows: [{\"source_table\":.., \"source_col\":.., \"target_table\":.., "
        "\"target_col\":.., \"transformation_logic\":.., \"notes\":..}]"
    )
    user_prompt = json.dumps({"business_intent": business_intent, "silver_schemas": silver_info}, default=str)[:12000]
    fallback = _heuristic_gold_sttm(silver_info, business_intent)
    rows, used_llm, err = call_llm_json(system_prompt, user_prompt, fallback)
    if not isinstance(rows, list) or not rows:
        rows, used_llm, err = fallback, False, "LLM returned empty/invalid list; used fallback"
    return rows, used_llm, err


def run_sttm_bronze(run_id: str, profile: dict, business_intent: str) -> dict:
    trace = AgentTrace("sttm_generator", run_id)
    start = time.time()
    ctx = inspect_context_tool("bronze", {"profile": profile})
    rows, used_llm, err = generate_bronze_sttm_tool(profile, business_intent)
    df = _rows_to_df(rows)
    path = os.path.join(STTM_DIR, f"sttm_bronze_{run_id}.csv")
    df.to_csv(path, index=False)
    trace.record(
        input_={"phase": "bronze", "business_intent": business_intent},
        plan="inspect_context_tool -> generate_bronze_sttm_tool",
        tool_calls=[{"tool": "inspect_context_tool", "phase": "bronze"},
                    {"tool": "generate_bronze_sttm_tool", "used_llm": used_llm, "note": err}],
        output={"sttm_path": path, "rule_count": len(df)},
        duration_seconds=time.time() - start, status="success",
        reasoning_steps=[f"Generated {len(df)} bronze mapping rules {'via LLM' if used_llm else 'via heuristic fallback'}"],
    )
    return {"rows": rows, "path": path, "used_llm": used_llm, "note": err}


def run_sttm_silver(run_id: str, bronze_paths: list[str], profile: dict, business_intent: str) -> dict:
    trace = AgentTrace("sttm_generator", run_id)
    start = time.time()
    ctx = inspect_context_tool("silver", {"bronze_paths": bronze_paths})
    rows, used_llm, err = generate_silver_sttm_tool(ctx, profile, business_intent)
    df = _rows_to_df(rows)
    path = os.path.join(STTM_DIR, f"sttm_silver_{run_id}.csv")
    df.to_csv(path, index=False)
    trace.record(
        input_={"phase": "silver", "bronze_paths": bronze_paths},
        plan="inspect_context_tool -> generate_silver_sttm_tool",
        tool_calls=[{"tool": "inspect_context_tool", "phase": "silver"},
                    {"tool": "generate_silver_sttm_tool", "used_llm": used_llm, "note": err}],
        output={"sttm_path": path, "rule_count": len(df)},
        duration_seconds=time.time() - start, status="success",
        reasoning_steps=[f"Generated {len(df)} silver cleansing rules {'via LLM' if used_llm else 'via heuristic fallback'}"],
    )
    return {"rows": rows, "path": path, "used_llm": used_llm, "note": err}


def run_sttm_gold(run_id: str, silver_paths: list[str], business_intent: str) -> dict:
    trace = AgentTrace("sttm_generator", run_id)
    start = time.time()
    ctx = inspect_context_tool("gold", {"silver_paths": silver_paths})
    rows, used_llm, err = generate_gold_sttm_tool(ctx, business_intent)
    df = _rows_to_df(rows)
    path = os.path.join(STTM_DIR, f"sttm_gold_{run_id}.csv")
    df.to_csv(path, index=False)
    trace.record(
        input_={"phase": "gold", "silver_paths": silver_paths, "business_intent": business_intent},
        plan="inspect_context_tool -> generate_gold_sttm_tool",
        tool_calls=[{"tool": "inspect_context_tool", "phase": "gold"},
                    {"tool": "generate_gold_sttm_tool", "used_llm": used_llm, "note": err}],
        output={"sttm_path": path, "rule_count": len(df)},
        duration_seconds=time.time() - start, status="success",
        reasoning_steps=[f"Generated {len(df)} gold aggregation rules {'via LLM' if used_llm else 'via heuristic fallback'}"],
    )
    return {"rows": rows, "path": path, "used_llm": used_llm, "note": err}
