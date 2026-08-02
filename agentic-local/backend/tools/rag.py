from __future__ import annotations

from typing import Any

from backend.rag.ingest import reindex_all
from backend.rag.search import hybrid_search
from backend.rag.store import RagStore
from backend.tools.base import Tool, ToolRegistry


def register_rag_tools(registry: ToolRegistry) -> None:
    registry.register(Tool("rag_status", "Show local RAG index counts and models.", {}, lambda args: RagStore().status()))
    registry.register(Tool("rag_reindex", "Incrementally reindex workspace/docs and canonical sources.", {}, lambda args: reindex_all()))
    registry.register(
        Tool(
            "rag_search",
            "Search indexed local documentation with hybrid retrieval.",
            {"query": "required search text", "limit": "maximum chunks"},
            lambda args: {"chunks": hybrid_search(str(args.get("query") or ""), top_k=int(args.get("limit") or 8))[0]},
        )
    )
