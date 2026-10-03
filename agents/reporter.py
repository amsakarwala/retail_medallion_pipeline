"""
agents/reporter.py - Reporter Agent
Purpose: analyze the final Gold layer data, synthesize insights, and
generate a human-readable executive report answering the business question.

Tools:
  inspect_gold_tool(gold_paths)     - preview gold tables
  load_data_tool(gold_paths)        - load gold parquet into DuckDB in-memory
  execute_query_tool(con, sql)      - run SQL against DuckDB, return df
"""
import json
import os
import time

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from core.llm import call_llm_text
from core.observability import AgentTrace

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def inspect_gold_tool(gold_paths: list[str]) -> dict:
    info = {}
    for path in gold_paths:
        df = pd.read_parquet(path)
        name = os.path.splitext(os.path.basename(path))[0]
        info[name] = {"columns": list(df.columns), "row_count": len(df), "sample": df.head(5).to_dict(orient="records")}
    return info


def load_data_tool(gold_paths: list[str]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    for path in gold_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    return con


def execute_query_tool(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def _build_charts(gold_data: dict[str, pd.DataFrame]) -> list[str]:
    """Returns a list of HTML div strings (plotly figures) to embed in the report."""
    chart_htmls = []

    df = gold_data.get("sales_by_store_gold")
    if df is not None and not df.empty:
        name_col = "store_name" if "store_name" in df.columns else "store_id"
        fig = go.Figure(go.Bar(x=df[name_col], y=df["total_revenue"], marker_color="#2E86AB"))
        fig.update_layout(title="Revenue by Store", xaxis_title="Store", yaxis_title="Total Revenue ($)",
                           template="plotly_white", height=400)
        chart_htmls.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    df = gold_data.get("sales_by_category_gold")
    if df is not None and not df.empty:
        fig = go.Figure(go.Pie(labels=df["category"], values=df["total_revenue"], hole=0.4))
        fig.update_layout(title="Revenue Share by Category", template="plotly_white", height=400)
        chart_htmls.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    df = gold_data.get("sales_trend_by_month_gold")
    if df is not None and not df.empty:
        df = df.sort_values("month")
        fig = go.Figure(go.Scatter(x=df["month"].astype(str), y=df["total_revenue"], mode="lines+markers",
                                    line=dict(color="#A23B72", width=3)))
        fig.update_layout(title="Monthly Revenue Trend", xaxis_title="Month", yaxis_title="Total Revenue ($)",
                           template="plotly_white", height=400)
        chart_htmls.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    df = gold_data.get("top_products_gold")
    if df is not None and not df.empty:
        fig = go.Figure(go.Bar(x=df["total_revenue"][::-1], y=df["product_name"][::-1], orientation="h",
                                marker_color="#F18F01"))
        fig.update_layout(title="Top 10 Products by Revenue", xaxis_title="Total Revenue ($)",
                           template="plotly_white", height=450)
        chart_htmls.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    return chart_htmls


def _heuristic_narrative(gold_data: dict[str, pd.DataFrame], business_intent: str) -> str:
    lines = [f"<h3>Summary</h3><p>This report answers: <em>{business_intent}</em></p>"]
    df = gold_data.get("sales_by_store_gold")
    if df is not None and not df.empty:
        top = df.iloc[0]
        name_col = "store_name" if "store_name" in df.columns else "store_id"
        lines.append(f"<p>The top-performing store is <b>{top[name_col]}</b> with total revenue of "
                      f"<b>${top['total_revenue']:,.2f}</b> across {int(top['num_transactions'])} transactions.</p>")
    df = gold_data.get("sales_by_category_gold")
    if df is not None and not df.empty:
        top = df.iloc[0]
        lines.append(f"<p>The leading category is <b>{top['category']}</b>, generating "
                      f"<b>${top['total_revenue']:,.2f}</b> in revenue.</p>")
    df = gold_data.get("sales_trend_by_month_gold")
    if df is not None and len(df) > 1:
        first, last = df.iloc[0], df.iloc[-1]
        change = ((last["total_revenue"] - first["total_revenue"]) / first["total_revenue"] * 100) if first["total_revenue"] else 0
        direction = "grew" if change >= 0 else "declined"
        lines.append(f"<p>Monthly revenue {direction} by approximately <b>{abs(change):.1f}%</b> "
                      f"from the first to the last month in the dataset.</p>")
    lines.append("<p><em>(Narrative generated via heuristic fallback - no LLM key configured.)</em></p>")
    return "\n".join(lines)


def run_reporter(run_id: str, gold_paths: list[str], business_intent: str) -> dict:
    trace = AgentTrace("reporter", run_id)
    start = time.time()

    gold_info = inspect_gold_tool(gold_paths)
    con = load_data_tool(gold_paths)
    gold_data = {name: execute_query_tool(con, f"SELECT * FROM {name}") for name in gold_info}
    con.close()

    system_prompt = (
        "You are the Reporter Agent. You are given aggregated Gold-layer retail sales tables as JSON and the "
        "business question. Write a concise executive summary (HTML, using <h3>/<p>/<b> tags only, no <html> wrapper) "
        "with 3-5 sentences of concrete, numbers-backed insight that directly answers the business question. "
        "Call out standout stores/categories/products/trends, and one actionable recommendation."
    )
    sample_json = {name: df.head(10).to_dict(orient="records") for name, df in gold_data.items()}
    user_prompt = json.dumps({"business_intent": business_intent, "gold_tables": sample_json}, default=str)[:12000]
    fallback = _heuristic_narrative(gold_data, business_intent)
    narrative, used_llm, err = call_llm_text(system_prompt, user_prompt, fallback)

    charts = _build_charts(gold_data)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Retail Medallion Pipeline Report</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; max-width: 1000px; margin: 40px auto; color: #1a1a1a; }}
h1 {{ color: #2E86AB; }} h3 {{ color: #A23B72; }}
.meta {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
.chart {{ margin: 24px 0; border: 1px solid #eee; border-radius: 8px; padding: 12px; }}
.badge {{ display:inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75em; background:#eef6ff; color:#2E86AB;}}
</style></head><body>
<h1>Retail Medallion Pipeline - Executive Report</h1>
<div class="meta">Run ID: {run_id} &middot; Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} &middot;
<span class="badge">{'LLM-generated narrative' if used_llm else 'heuristic narrative (no LLM key)'}</span></div>
{narrative}
{''.join(f'<div class="chart">{c}</div>' for c in charts)}
</body></html>"""

    report_path = os.path.join(REPORT_DIR, f"report_{run_id}.html")
    with open(report_path, "w") as f:
        f.write(html)

    summary = {
        "business_intent": business_intent,
        "narrative_html": narrative,
        "used_llm": used_llm,
        "llm_note": err,
        "gold_tables": {name: df.to_dict(orient="records") for name, df in gold_data.items()},
        "report_path": report_path,
    }

    trace.record(
        input_={"gold_paths": gold_paths, "business_intent": business_intent},
        plan="inspect_gold_tool -> load_data_tool -> execute_query_tool (per table) -> LLM synthesis -> build charts -> write HTML report",
        tool_calls=[{"tool": "inspect_gold_tool"}, {"tool": "load_data_tool"},
                    {"tool": "execute_query_tool", "tables": list(gold_data.keys())},
                    {"tool": "llm_narrative_synthesis", "used_llm": used_llm, "note": err}],
        output={"report_path": report_path, "chart_count": len(charts)},
        duration_seconds=time.time() - start, status="success",
        reasoning_steps=[f"Synthesized executive narrative {'via LLM' if used_llm else 'via heuristic fallback'}",
                          f"Rendered {len(charts)} Plotly charts from Gold aggregates"],
    )
    return summary
