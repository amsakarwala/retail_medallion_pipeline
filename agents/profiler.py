"""
agents/profiler.py - Data Profiler Agent
Purpose: analyze raw data and create a comprehensive, machine-readable
profile of its structure and quality.

Tools:
  inspect_files_tool(paths)  - lightweight preview: shape, columns, dtypes, 3 samples
  profiler_tool(paths)       - full stats: null count/%, unique count, sample
                                values, min/max/mean for numerics
run_profiler(...)             - orchestrates the two tools + LLM semantic pass,
                                writes profile_combined_{run_id}.json
"""
import json
import os
import time

import pandas as pd

from core import memory
from core.llm import call_llm_json
from core.observability import AgentTrace

PROFILE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)


def inspect_files_tool(paths: list[str]) -> dict:
    """Lightweight preview per file: shape, columns, dtypes, 3 sample rows."""
    preview = {}
    for path in paths:
        df = pd.read_csv(path, nrows=200)  # cheap peek, not full load
        full_shape_df = pd.read_csv(path)
        name = os.path.splitext(os.path.basename(path))[0]
        preview[name] = {
            "path": path,
            "shape": list(full_shape_df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample_rows": df.head(3).fillna("").astype(str).to_dict(orient="records"),
        }
    return preview


def profiler_tool(paths: list[str]) -> dict:
    """Full column statistics per file."""
    stats = {}
    for path in paths:
        df = pd.read_csv(path)
        name = os.path.splitext(os.path.basename(path))[0]
        col_stats = {}
        for c in df.columns:
            s = df[c]
            entry = {
                "null_count": int(s.isna().sum()),
                "null_pct": round(float(s.isna().mean()) * 100, 2),
                "unique_count": int(s.nunique(dropna=True)),
                "sample_values": [str(v) for v in s.dropna().unique()[:5].tolist()],
            }
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().mean() > 0.6:  # mostly numeric-ish
                entry["min"] = float(numeric.min()) if numeric.notna().any() else None
                entry["max"] = float(numeric.max()) if numeric.notna().any() else None
                entry["mean"] = round(float(numeric.mean()), 2) if numeric.notna().any() else None
            quality_notes = []
            if entry["null_pct"] > 20:
                quality_notes.append("high_nulls")
            if entry["unique_count"] <= 1:
                quality_notes.append("low_cardinality")
            if s.dropna().astype(str).str.contains(r"[A-Z]").any() and s.dropna().astype(str).str.contains(r"[a-z]").any():
                quality_notes.append("mixed_case")
            entry["quality_notes"] = quality_notes
            col_stats[c] = entry
        dup_count = int(df.duplicated().sum())
        stats[name] = {"columns": col_stats, "duplicate_row_count": dup_count, "row_count": len(df)}
    return stats


def _heuristic_semantic_profile(preview: dict, stats: dict) -> dict:
    """Fallback semantic pass used when no LLM key is configured."""
    semantic = {}
    for table, info in preview.items():
        col_meanings = {}
        for c in info["columns"]:
            cl = c.lower()
            if "date" in cl or "time" in cl:
                col_meanings[c] = "date/time field"
            elif "id" in cl:
                col_meanings[c] = "identifier / likely join key"
            elif "price" in cl or "amount" in cl or "cost" in cl:
                col_meanings[c] = "monetary value (may contain currency symbols)"
            elif "email" in cl:
                col_meanings[c] = "contact / PII field"
            elif "qty" in cl or "quantity" in cl:
                col_meanings[c] = "numeric quantity"
            elif "name" in cl:
                col_meanings[c] = "descriptive text field"
            else:
                col_meanings[c] = "categorical/text field"
        semantic[table] = {"column_meanings": col_meanings, "business_intent_notes": "heuristic pass (no LLM)"}
    return semantic


def run_profiler(run_id: str, raw_paths: list[str], business_intent: str) -> dict:
    trace = AgentTrace("profiler", run_id)
    start = time.time()
    plan = "1) inspect_files_tool for cheap preview  2) profiler_tool for full stats  3) LLM semantic interpretation  4) index columns in memory for join-key discovery"
    tool_calls = []

    preview = inspect_files_tool(raw_paths)
    tool_calls.append({"tool": "inspect_files_tool", "input": raw_paths, "output_summary": {k: v["shape"] for k, v in preview.items()}})

    stats = profiler_tool(raw_paths)
    tool_calls.append({"tool": "profiler_tool", "input": raw_paths, "output_summary": {k: v["row_count"] for k, v in stats.items()}})

    # index columns for semantic join-key discovery (ChromaDB or fallback)
    for table, info in preview.items():
        sample_vals = {c: [r.get(c, "") for r in info["sample_rows"]] for c in info["columns"]}
        memory.index_columns(run_id, table, info["columns"], sample_vals)

    join_keys = memory.find_join_key_candidates(run_id, {t: v["columns"] for t, v in preview.items()})

    system_prompt = (
        "You are a Data Profiler Agent for a retail analytics medallion pipeline. "
        "Given file previews and column statistics, infer the semantic meaning of each column, "
        "flag data quality issues, and note the business intent. "
        "Return JSON: {\"<table>\": {\"column_meanings\": {\"<col>\": \"<meaning>\"}, "
        "\"business_intent_notes\": \"<short note>\"}}"
    )
    user_prompt = json.dumps({"business_intent": business_intent, "preview": preview, "stats": stats}, default=str)[:12000]
    fallback = _heuristic_semantic_profile(preview, stats)
    semantic, used_llm, err = call_llm_json(system_prompt, user_prompt, fallback)
    tool_calls.append({"tool": "llm_semantic_interpretation", "used_llm": used_llm, "note": err})

    profile = {
        "run_id": run_id,
        "business_intent": business_intent,
        "preview": preview,
        "stats": stats,
        "semantic": semantic,
        "join_key_candidates": join_keys,
        "used_llm": used_llm,
        "llm_note": err,
    }

    out_path = os.path.join(PROFILE_DIR, f"profile_combined_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2, default=str)

    trace.record(
        input_={"raw_paths": raw_paths, "business_intent": business_intent},
        plan=plan,
        tool_calls=tool_calls,
        output={"profile_path": out_path, "join_key_candidates": join_keys},
        duration_seconds=time.time() - start,
        status="success",
        reasoning_steps=[
            "Previewed files cheaply before full stats to avoid wasted work on bad files",
            "Computed full null/unique/min/max stats per column",
            f"Indexed columns for semantic join discovery; found {len(join_keys)} candidate join key pairs",
            f"Interpreted column semantics {'via LLM' if used_llm else 'via heuristic fallback (' + str(err) + ')'}",
        ],
    )

    profile["profile_path"] = out_path
    return profile
