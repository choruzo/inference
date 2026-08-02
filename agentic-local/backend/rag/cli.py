from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.config import WORKSPACE_ROOT
from backend.rag.documents import convert_to_markdown
from backend.rag.embeddings import index_embeddings
from backend.rag.evaluate import evaluate
from backend.rag.ingest import reindex_all
from backend.rag.search import hybrid_search
from backend.rag.store import RagStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the local RAG index")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("reindex")
    sub.add_parser("embed")
    search = sub.add_parser("search")
    search.add_argument("query")
    convert = sub.add_parser("convert")
    convert.add_argument("path")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--golden", type=Path)
    args = parser.parse_args()
    store = RagStore()
    if args.command == "status":
        result = store.status()
    elif args.command == "reindex":
        result = reindex_all(store)
    elif args.command == "embed":
        result = index_embeddings(store=store)
    elif args.command == "search":
        chunks, trace = hybrid_search(args.query, store=store)
        result = {"chunks": chunks, "trace": trace}
    elif args.command == "convert":
        result = convert_to_markdown((WORKSPACE_ROOT / args.path).resolve(), store)
    else:
        result = evaluate(args.golden, store)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
