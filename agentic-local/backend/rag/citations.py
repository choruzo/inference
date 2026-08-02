from __future__ import annotations

from typing import Any

from backend.contracts import Citation
from backend.rag.store import RagStore


def build_citations(chunks: list[dict[str, Any]]) -> list[Citation]:
    return [Citation(id=index, chunk_id=str(chunk["id"]), path=str(chunk["path"]), title=str(chunk.get("title", "")), section=str(chunk.get("section", "")), start_line=int(chunk["start_line"]), end_line=int(chunk["end_line"]), quote=str(chunk["content"])[:240]) for index, chunk in enumerate(chunks, 1)]


def validate_citations(citations: list[Citation], store: RagStore | None = None) -> bool:
    store = store or RagStore()
    ids = {citation.chunk_id for citation in citations}
    return ids == store.validate_chunk_ids(ids) and len({citation.id for citation in citations}) == len(citations)


def context_with_citations(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, 1):
        parts.append(f"[FUENTE {index}] {chunk['path']}:{chunk['start_line']}-{chunk['end_line']} | {chunk.get('section', '')}\n{chunk['content']}")
    return "\n\n".join(parts)
