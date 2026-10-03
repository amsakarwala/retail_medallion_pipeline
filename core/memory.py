"""
core/memory.py
MemoryStore - stores intent + profiles + reports for agent context, and
gives the Profiler Agent a semantic "join key discovery" helper: it embeds
column names (+ sample values) per dataset in ChromaDB and finds likely
matches across files by embedding similarity. If chromadb isn't installed
or fails to init, falls back to a simple normalized-name/token overlap
heuristic so nothing breaks.
"""
import json
import os
import re

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "state", "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

_chroma_client = None
_collection = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb

        _chroma_client = chromadb.PersistentClient(path=os.path.join(MEMORY_DIR, "chroma"))
        _collection = _chroma_client.get_or_create_collection("column_semantics")
        return _collection
    except Exception:  # noqa: BLE001
        return None


def store_document(run_id: str, doc_type: str, content: dict):
    """Persist intent / profile / report JSON for later agent context lookups."""
    path = os.path.join(MEMORY_DIR, f"{run_id}_{doc_type}.json")
    with open(path, "w") as f:
        json.dump(content, f, indent=2, default=str)


def load_document(run_id: str, doc_type: str) -> dict | None:
    path = os.path.join(MEMORY_DIR, f"{run_id}_{doc_type}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def index_columns(run_id: str, table_name: str, columns: list[str], sample_values: dict[str, list]):
    """Index each column (as 'colname: sample1, sample2, sample3') for semantic search."""
    col = _get_collection()
    docs = [f"{c}: {', '.join(str(v) for v in sample_values.get(c, [])[:3])}" for c in columns]
    ids = [f"{run_id}::{table_name}::{c}" for c in columns]
    metadatas = [{"run_id": run_id, "table": table_name, "column": c} for c in columns]
    if col is not None:
        try:
            col.upsert(documents=docs, ids=ids, metadatas=metadatas)
            return
        except Exception:  # noqa: BLE001
            pass
    # fallback: keep a flat json index we can token-match against
    idx_path = os.path.join(MEMORY_DIR, "fallback_column_index.json")
    idx = {}
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
    idx.setdefault(run_id, []).extend(
        [{"table": table_name, "column": c, "doc": d} for c, d in zip(columns, docs)]
    )
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)


def find_join_key_candidates(run_id: str, tables: dict[str, list[str]]) -> list[dict]:
    """
    Given {table_name: [columns]}, propose likely join-key pairs across
    tables using semantic similarity (ChromaDB) or normalized-name overlap
    (fallback). Returns list of {table_a, col_a, table_b, col_b, confidence}.
    """
    col = _get_collection()
    candidates = []
    table_names = list(tables.keys())

    if col is not None:
        try:
            for i, ta in enumerate(table_names):
                for ca in tables[ta]:
                    results = col.query(
                        query_texts=[ca], n_results=5,
                        where={"run_id": run_id},
                    )
                    for meta, dist in zip(results.get("metadatas", [[]])[0], results.get("distances", [[0]])[0]):
                        if meta["table"] != ta and dist < 0.6:
                            candidates.append({
                                "table_a": ta, "col_a": ca,
                                "table_b": meta["table"], "col_b": meta["column"],
                                "confidence": round(max(0, 1 - dist), 2),
                                "method": "chromadb_embedding",
                            })
            return _dedupe_candidates(candidates)
        except Exception:  # noqa: BLE001
            pass

    # fallback heuristic: normalized name exact/substring match
    for i, ta in enumerate(table_names):
        for tb in table_names[i + 1:]:
            for ca in tables[ta]:
                for cb in tables[tb]:
                    na, nb = _normalize(ca), _normalize(cb)
                    if na == nb or na in nb or nb in na:
                        candidates.append({
                            "table_a": ta, "col_a": ca, "table_b": tb, "col_b": cb,
                            "confidence": 0.9 if na == nb else 0.6,
                            "method": "name_overlap_heuristic",
                        })
    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for c in sorted(candidates, key=lambda x: -x["confidence"]):
        key = tuple(sorted([f"{c['table_a']}.{c['col_a']}", f"{c['table_b']}.{c['col_b']}"]))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out
