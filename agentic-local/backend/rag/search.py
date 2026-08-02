from __future__ import annotations

import math
import json
import re
from collections import defaultdict
from typing import Any

import httpx

from backend.config import EMBEDDINGS_MODEL, RAG_CANDIDATE_POOL, RAG_CONTEXT_TOKENS, RAG_MIN_VECTOR_SCORE, RAG_TOP_K
from backend.rag.embeddings import EmbeddingClient
from backend.rag.store import RagStore


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


def vector_search(query: str, store: RagStore, client: EmbeddingClient, limit: int = RAG_CANDIDATE_POOL) -> list[dict[str, Any]]:
    query_vector = client.embed([query], query=True)[0]
    rows = store.chunks_with_embeddings(client.model)
    for row in rows:
        row["score"] = _cosine(query_vector, row.pop("vector"))
        row["reasons"] = ["vector"]
    return sorted(rows, key=lambda row: float(row["score"]), reverse=True)[:limit]


def reciprocal_rank_fusion(rankings: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, dict[str, Any]] = {}
    reasons: dict[str, set[str]] = defaultdict(set)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            chunk_id = str(item["id"])
            scores[chunk_id] += 1.0 / (k + rank)
            items[chunk_id] = item
            reasons[chunk_id].update(item.get("reasons", []))
    result = []
    for chunk_id in sorted(scores, key=scores.get, reverse=True):
        result.append(items[chunk_id] | {"score": scores[chunk_id], "reasons": sorted(reasons[chunk_id])})
    return result


def _passes_filters(item: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    if filters.get("path") and not str(item["path"]).startswith(str(filters["path"])):
        return False
    if filters.get("type") and item.get("source_type") != filters["type"]:
        return False
    if filters.get("source") and filters["source"] not in {item.get("source_type"), item.get("path")}:
        return False
    updated = float(item.get("document_updated_at", item.get("updated_at", 0)))
    if filters.get("updated_after") is not None and updated < float(filters["updated_after"]):
        return False
    if filters.get("updated_before") is not None and updated > float(filters["updated_before"]):
        return False
    if filters.get("tag"):
        tags = json.loads(str(item.get("tags_json", "[]")))
        if str(filters["tag"]) not in tags:
            return False
    return True


def _diverse_budget(items: list[dict[str, Any]], top_k: int, token_budget: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_section: dict[tuple[str, str], int] = defaultdict(int)
    tokens = 0
    for item in items:
        key = (str(item["path"]), str(item.get("section", "")))
        if per_section[key] >= 2:
            continue
        cost = int(item.get("token_count", 0))
        if selected and tokens + cost > token_budget:
            continue
        selected.append(item)
        per_section[key] += 1
        tokens += cost
        if len(selected) >= top_k:
            break
    return selected


def query_type(query: str) -> str:
    if re.search(r"[/\\]|\.[a-zA-Z0-9]{1,5}\b|[A-Z_]{3,}", query):
        return "exact"
    if any(word in query.lower() for word in ("compara", "diferencia", "versus")):
        return "comparison"
    if any(word in query.lower() for word in ("resume", "todo", "global")):
        return "global"
    return "conceptual"


def hybrid_search(query: str, store: RagStore | None = None, client: EmbeddingClient | None = None, top_k: int = RAG_TOP_K, filters: dict[str, Any] | None = None, token_budget: int = RAG_CONTEXT_TOKENS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    store = store or RagStore()
    lexical = store.keyword_search(query, RAG_CANDIDATE_POOL)
    semantic: list[dict[str, Any]] = []
    vector_error = None
    if store.status()["embeddings"]:
        try:
            semantic = vector_search(query, store, client or EmbeddingClient(model=EMBEDDINGS_MODEL), RAG_CANDIDATE_POOL)
        except (httpx.HTTPError, ValueError, KeyError) as exc:  # type: ignore[name-defined]
            vector_error = f"{type(exc).__name__}: {exc}"
    fused = reciprocal_rank_fusion([lexical, semantic]) if semantic else lexical
    semantic_confidence = float(semantic[0]["score"]) if semantic else None
    if not lexical and (semantic_confidence is None or semantic_confidence < RAG_MIN_VECTOR_SCORE):
        fused = []
    fused = [item for item in fused if _passes_filters(item, filters)]
    requested_top_k = top_k
    if query_type(query) == "exact" and lexical:
        top_k = min(top_k, 4)
    elif not lexical and semantic and float(semantic[0].get("score", 0)) < 0.45:
        top_k = min(max(top_k, 8), 12)
    selected = _diverse_budget(fused, top_k, token_budget)
    trace = {
        "query_type": query_type(query),
        "keyword_ranking": [item["id"] for item in lexical],
        "vector_ranking": [item["id"] for item in semantic],
        "fused_ranking": [item["id"] for item in fused],
        "selected": [item["id"] for item in selected],
        "requested_top_k": requested_top_k,
        "adaptive_top_k": top_k,
        "vector_error": vector_error,
        "semantic_confidence": semantic_confidence,
        "minimum_vector_score": RAG_MIN_VECTOR_SCORE,
    }
    return selected, trace
