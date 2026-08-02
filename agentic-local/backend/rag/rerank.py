from __future__ import annotations

import hashlib
import json
from typing import Any, Awaitable, Callable

from backend.rag.store import RagStore


Completion = Callable[[str], Awaitable[str]]
RERANK_VERSION = "2-evidence-tier"


def _evidence_tier(item: dict[str, Any]) -> int:
    reasons = set(item.get("reasons", []))
    if {"keyword", "vector"} <= reasons:
        return 2
    if "keyword" in reasons:
        return 1
    return 0


async def rerank(query: str, candidates: list[dict[str, Any]], completion: Completion | None = None, store: RagStore | None = None, top_k: int = 4) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(candidates) <= top_k or completion is None:
        return candidates[:top_k], {"strategy": "retrieval_order", "ranked_chunk_ids": [item["id"] for item in candidates[:top_k]]}
    store = store or RagStore()
    cache_key = hashlib.sha256((RERANK_VERSION + "|" + query + "|" + "|".join(str(item["id"]) for item in candidates)).encode()).hexdigest()
    cached = store.cache_get(cache_key)
    if cached:
        by_id = {str(item["id"]): item for item in candidates}
        return [by_id[item] for item in cached if item in by_id][:top_k], {"strategy": "llm_cache", "ranked_chunk_ids": cached}
    compact = [{"id": item["id"], "text": str(item.get("content", ""))[:900]} for item in candidates[:12]]
    prompt = "Ordena los chunks por relevancia. Devuelve solo JSON {\"ranked_chunk_ids\":[...]}.\nPregunta: " + query + "\nChunks: " + json.dumps(compact, ensure_ascii=False)
    raw = await completion(prompt)
    try:
        parsed = json.loads(raw)
        ids = [str(value) for value in parsed["ranked_chunk_ids"]]
    except (json.JSONDecodeError, KeyError, TypeError):
        ids = [str(item["id"]) for item in candidates]
    allowed = {str(item["id"]) for item in candidates}
    ids = [item for item in ids if item in allowed]
    ids.extend(str(item["id"]) for item in candidates if str(item["id"]) not in ids)
    by_id = {str(item["id"]): item for item in candidates}
    model_position = {chunk_id: position for position, chunk_id in enumerate(ids)}
    ids.sort(key=lambda chunk_id: (-_evidence_tier(by_id[chunk_id]), model_position[chunk_id]))
    store.cache_put(cache_key, ids)
    return [by_id[item] for item in ids[:top_k]], {"strategy": "llm", "ranked_chunk_ids": ids}
