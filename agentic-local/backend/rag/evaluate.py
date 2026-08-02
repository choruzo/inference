from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from backend.config import RAG_CACHE_DIR, WORKSPACE_ROOT
from backend.rag.embeddings import EmbeddingClient
from backend.rag.search import hybrid_search
from backend.rag.store import RagStore


def evaluate(golden_path: Path | None = None, store: RagStore | None = None, client: EmbeddingClient | None = None, top_k: int = 5) -> dict[str, Any]:
    golden_path = golden_path or WORKSPACE_ROOT / "evals" / "rag_golden.jsonl"
    store = store or RagStore()
    cases = [json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    started = time.perf_counter()
    for case in cases:
        before = time.perf_counter()
        chunks, trace = hybrid_search(case["question"], store, client, top_k=top_k)
        paths = {str(item["path"]) for item in chunks}
        expected = set(case.get("expected_sources", []))
        hit = bool(paths & expected) if expected else not chunks
        top_source_correct = bool(chunks and str(chunks[0]["path"]) in expected) if expected else not chunks
        relevant = sum(1 for item in chunks if str(item["path"]) in expected)
        expected_answer = str(case.get("expected_answer", "")).lower()
        grounded = not expected_answer or any(expected_answer in str(item["content"]).lower() for item in chunks)
        results.append({"question": case["question"], "hit": hit, "top_source_correct": top_source_correct, "grounded": grounded, "context_precision": relevant / max(1, len(chunks)), "paths": sorted(paths), "latency_ms": round((time.perf_counter() - before) * 1000, 2), "trace": trace})
    count = max(1, len(results))
    output = {"cases": len(cases), "hit_at_k": sum(item["hit"] for item in results) / count, "top_source_accuracy": sum(item["top_source_correct"] for item in results) / count, "citation_accuracy": sum(item["hit"] for item in results) / count, "answer_groundedness": sum(item["grounded"] for item in results) / count, "context_precision": sum(item["context_precision"] for item in results) / count, "latency_ms": round((time.perf_counter() - started) * 1000, 2), "results": results}
    RAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (RAG_CACHE_DIR / "last_evaluation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
