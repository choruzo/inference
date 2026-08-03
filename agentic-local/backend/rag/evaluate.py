from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from backend.agent import LocalAgent
from backend.modes import ChatModes
from backend.config import RAG_CACHE_DIR, WORKSPACE_ROOT
from backend.rag.embeddings import EmbeddingClient
from backend.rag.search import hybrid_search
from backend.rag.store import RagStore


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


async def _evaluate_cases(cases: list[dict[str, Any]], store: RagStore, client: EmbeddingClient | None, top_k: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    agent = LocalAgent(rag_store=store, embedding_client=client)
    try:
        for case in cases:
            before = time.perf_counter()
            chunks, trace = hybrid_search(case["question"], store, client, top_k=top_k)
            paths = {str(item["path"]) for item in chunks}
            expected = set(case.get("expected_sources", []))
            hit = bool(paths & expected) if expected else not chunks
            top_source_correct = bool(chunks and str(chunks[0]["path"]) in expected) if expected else not chunks
            relevant = sum(1 for item in chunks if str(item["path"]) in expected)

            response = await agent.chat(
                str(case["question"]),
                [],
                ChatModes(chat=True, rag=True, web=False, reasoning_panel=False),
            )
            answer = str(response.get("answer", ""))
            citations = list(response.get("citations", []))
            cited_paths = {str(item.get("path", "")) for item in citations}
            referenced = {int(value) for value in re.findall(re.escape("[") + "([0-9]+)" + re.escape("]"), answer)}
            citation_ids = {int(item.get("id", -1)) for item in citations}
            if expected:
                citation_correct = bool(cited_paths & expected) and bool(referenced) and referenced == citation_ids
            else:
                citation_correct = not citations and not referenced

            expected_answer = _normalize(str(case.get("expected_answer", "")))
            answer_matches = not expected_answer or expected_answer in _normalize(answer)
            grounded = bool(answer) and answer_matches and citation_correct and response.get("stopped") in {"final", "no_evidence"}
            results.append(
                {
                    "question": case["question"],
                    "answer": answer,
                    "stopped": response.get("stopped"),
                    "hit": hit,
                    "top_source_correct": top_source_correct,
                    "citation_correct": citation_correct,
                    "grounded": grounded,
                    "context_precision": relevant / max(1, len(chunks)),
                    "paths": sorted(paths),
                    "cited_paths": sorted(cited_paths),
                    "latency_ms": round((time.perf_counter() - before) * 1000, 2),
                    "trace": trace,
                }
            )
    finally:
        await agent.close()
    return results


def evaluate(golden_path: Path | None = None, store: RagStore | None = None, client: EmbeddingClient | None = None, top_k: int = 5) -> dict[str, Any]:
    golden_path = golden_path or WORKSPACE_ROOT / "evals" / "rag_golden.jsonl"
    store = store or RagStore()
    cases = [json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.perf_counter()
    results = asyncio.run(_evaluate_cases(cases, store, client, top_k))
    count = max(1, len(results))
    output = {
        "cases": len(cases),
        "hit_at_k": sum(item["hit"] for item in results) / count,
        "top_source_accuracy": sum(item["top_source_correct"] for item in results) / count,
        "citation_accuracy": sum(item["citation_correct"] for item in results) / count,
        "answer_groundedness": sum(item["grounded"] for item in results) / count,
        "context_precision": sum(item["context_precision"] for item in results) / count,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "results": results,
    }
    RAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (RAG_CACHE_DIR / "last_evaluation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
