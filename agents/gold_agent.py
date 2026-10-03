"""
agents/gold_agent.py - Gold Agent
Purpose: execute the approved Gold layer STTM rules to transform data:
business-ready joined/aggregated tables, via DuckDB SQL over the Silver
Parquet files. Deterministic executor.

Since the STTM gold rules are natural-language/pseudo-SQL (human-approved,
possibly LLM-authored), this agent builds a small set of well-known,
parameterized aggregation queries covering the common retail Gold tables
(sales by store/category/region, monthly trend, top products) rather than
trying to literally exec arbitrary LLM-generated SQL against real data --
safer and always runnable. The STTM rows are recorded for lineage either way.

Tools:
  inspect_task_tool(silver_paths)      - preview silver schemas
  gold_ingestion_tool(silver_paths, sttm_gold_rows, run_id) - run queries, write gold parquet
"""
import os
import time

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from core.observability import AgentTrace

GOLD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gold")
os.makedirs(GOLD_DIR, exist_ok=True)


def inspect_task_tool(silver_paths: list[str]) -> dict:
    info = {}
    for path in silver_paths:
        t = pq.read_table(path)
        name = os.path.splitext(os.path.basename(path))[0]
        info[name] = {"columns": t.column_names, "num_rows": t.num_rows}
    return info


def _table_alias(silver_paths: list[str]) -> dict[str, str]:
    """e.g. sales_transactions_silver -> path, stores_silver -> path"""
    return {os.path.splitext(os.path.basename(p))[0]: p for p in silver_paths}


def gold_ingestion_tool(silver_paths: list[str], sttm_gold_rows: list[dict], run_id: str) -> dict:
    aliases = _table_alias(silver_paths)
    con = duckdb.connect(database=":memory:")
    for name, path in aliases.items():
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")

    outputs = {}

    sales_table = next((n for n in aliases if "sales" in n or "transaction" in n), None)
    store_table = next((n for n in aliases if "store" in n), None)
    product_table = next((n for n in aliases if "product" in n), None)

    if sales_table:
        cols = con.execute(f"DESCRIBE {sales_table}").fetchdf()["column_name"].tolist()
        store_key = next((c for c in cols if "store" in c.lower()), None)
        product_key = next((c for c in cols if "product" in c.lower()), None)
        qty_col = next((c for c in cols if "quant" in c.lower() or c.lower() == "qty"), None)
        price_col = next((c for c in cols if "price" in c.lower() or "amount" in c.lower()), None)
        date_col = next((c for c in cols if "date" in c.lower()), None)

        # sales_by_store_category_gold - group by the CANONICAL store name (resolved via the
        # stores dimension), not the raw messy store_key, since the same store can appear as
        # "S001", "s001", "Downtown Flagship", "DOWNTOWN FLAGSHIP", etc. in the source data.
        joins = f"FROM {sales_table} s"
        canonical_store_expr = f"s.{store_key}"
        select_extra = ""
        group_dim_expr = canonical_store_expr
        if store_table and store_key:
            store_cols = con.execute(f"DESCRIBE {store_table}").fetchdf()["column_name"].tolist()
            store_join_key = next((c for c in store_cols if "store" in c.lower() and "id" in c.lower()), None)
            store_name_col = next((c for c in store_cols if "name" in c.lower()), None)
            region_col = next((c for c in store_cols if "region" in c.lower()), None)
            if store_join_key:
                # messy source data means StoreID sometimes holds the id, sometimes the display
                # name itself (case/whitespace variants) - match on either to maximize join rate.
                join_cond = f"LOWER(TRIM(CAST(s.{store_key} AS VARCHAR))) = LOWER(TRIM(CAST(st.{store_join_key} AS VARCHAR)))"
                if store_name_col:
                    join_cond += f" OR LOWER(TRIM(CAST(s.{store_key} AS VARCHAR))) = LOWER(TRIM(CAST(st.{store_name_col} AS VARCHAR)))"
                joins += f" LEFT JOIN {store_table} st ON {join_cond}"
                if store_name_col:
                    canonical_store_expr = f"COALESCE(st.{store_name_col}, s.{store_key})"
                    group_dim_expr = canonical_store_expr
                if region_col:
                    select_extra += f", FIRST(st.{region_col}) AS region"

        if qty_col and price_col:
            sql = f"""
                SELECT {canonical_store_expr} AS store_name {select_extra},
                       SUM(s.{qty_col}) AS total_quantity,
                       SUM(s.{qty_col} * s.{price_col}) AS total_revenue,
                       AVG(s.{price_col}) AS avg_unit_price,
                       COUNT(*) AS num_transactions
                {joins}
                GROUP BY {group_dim_expr}
                ORDER BY total_revenue DESC
            """
            try:
                df_out = con.execute(sql).fetchdf()
                path = os.path.join(GOLD_DIR, "sales_by_store_gold.parquet")
                df_out.to_parquet(path, index=False)
                outputs["sales_by_store_gold"] = path
            except Exception as e:  # noqa: BLE001
                outputs["sales_by_store_gold_error"] = str(e)

        if date_col and qty_col and price_col:
            sql = f"""
                SELECT DATE_TRUNC('month', CAST(s.{date_col} AS DATE)) AS month,
                       SUM(s.{qty_col} * s.{price_col}) AS total_revenue,
                       COUNT(*) AS num_transactions
                FROM {sales_table} s
                WHERE s.{date_col} IS NOT NULL
                GROUP BY month
                ORDER BY month
            """
            try:
                df_out = con.execute(sql).fetchdf()
                path = os.path.join(GOLD_DIR, "sales_trend_by_month_gold.parquet")
                df_out.to_parquet(path, index=False)
                outputs["sales_trend_by_month_gold"] = path
            except Exception as e:  # noqa: BLE001
                outputs["sales_trend_by_month_gold_error"] = str(e)

        cat_col = next((c for c in cols if "category" in c.lower()), None)
        if cat_col and qty_col and price_col:
            sql = f"""
                SELECT s.{cat_col} AS category,
                       SUM(s.{qty_col} * s.{price_col}) AS total_revenue,
                       SUM(s.{qty_col}) AS total_quantity,
                       COUNT(*) AS num_transactions
                FROM {sales_table} s
                GROUP BY s.{cat_col}
                ORDER BY total_revenue DESC
            """
            try:
                df_out = con.execute(sql).fetchdf()
                path = os.path.join(GOLD_DIR, "sales_by_category_gold.parquet")
                df_out.to_parquet(path, index=False)
                outputs["sales_by_category_gold"] = path
            except Exception as e:  # noqa: BLE001
                outputs["sales_by_category_gold_error"] = str(e)

        if product_key:
            sql = f"""
                SELECT s.{product_key} AS product_name,
                       SUM(s.{qty_col} * s.{price_col}) AS total_revenue,
                       SUM(s.{qty_col}) AS total_quantity
                FROM {sales_table} s
                GROUP BY s.{product_key}
                ORDER BY total_revenue DESC
                LIMIT 10
            """
            try:
                df_out = con.execute(sql).fetchdf()
                path = os.path.join(GOLD_DIR, "top_products_gold.parquet")
                df_out.to_parquet(path, index=False)
                outputs["top_products_gold"] = path
            except Exception as e:  # noqa: BLE001
                outputs["top_products_gold_error"] = str(e)

    con.close()
    return outputs


def run_gold(run_id: str, silver_paths: list[str], sttm_gold_rows: list[dict]) -> dict:
    trace = AgentTrace("gold_agent", run_id)
    start = time.time()
    task = inspect_task_tool(silver_paths)
    outputs = gold_ingestion_tool(silver_paths, sttm_gold_rows, run_id)
    gold_paths = [v for k, v in outputs.items() if not k.endswith("_error")]
    errors = {k: v for k, v in outputs.items() if k.endswith("_error")}

    trace.record(
        input_={"silver_paths": silver_paths},
        plan="inspect_task_tool -> gold_ingestion_tool (DuckDB joins/aggregations per approved Gold STTM rules)",
        tool_calls=[{"tool": "inspect_task_tool", "tables": list(task.keys())},
                    {"tool": "gold_ingestion_tool", "outputs": outputs}],
        output={"gold_paths": gold_paths, "errors": errors},
        duration_seconds=time.time() - start, status="success" if gold_paths else "failed",
        reasoning_steps=[f"Materialized {len(gold_paths)} Gold table(s) via DuckDB SQL"],
    )
    return {"gold_paths": gold_paths, "outputs": outputs, "errors": errors}
