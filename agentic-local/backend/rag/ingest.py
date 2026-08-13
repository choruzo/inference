from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from backend.config import RAG_CHUNK_OVERLAP, RAG_CHUNK_TOKENS, RAG_DOCS_DIR, RAG_SOURCES_DIR, WORKSPACE_ROOT
from backend.rag.chunking import chunk_text
from backend.rag.store import RagStore


SUPPORTED_SUFFIXES = {".md", ".mdx", ".txt", ".py", ".js", ".ts", ".tsx", ".html", ".css", ".json", ".yaml", ".yml"}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tags(text: str) -> list[str]:
    frontmatter = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not frontmatter:
        return []
    match = re.search(r"^tags:\s*(.+)$", frontmatter.group(1), re.MULTILINE)
    if not match:
        return []
    return [item.strip().strip("'\"") for item in match.group(1).strip("[]").split(",") if item.strip()]


def ingest_documents(docs_dir: Path = RAG_DOCS_DIR, store: RagStore | None = None) -> dict[str, int]:
    store = store or RagStore()
    docs_dir.mkdir(parents=True, exist_ok=True)
    stats = {"discovered": 0, "indexed": 0, "unchanged": 0, "deleted": 0, "chunks": 0, "errors": 0}
    now = time.time()
    with store.connect() as db:
        job_id = db.execute("INSERT INTO ingest_jobs(started_at,status) VALUES(?,?)", (now, "running")).lastrowid
        try:
            root_prefix = docs_dir.resolve().relative_to(WORKSPACE_ROOT).as_posix().rstrip("/") + "/"
        except ValueError as exc:
            raise ValueError("Ingestion directory escapes workspace") from exc
        known = {row["path"]: dict(row) for row in db.execute("SELECT * FROM documents WHERE path LIKE ?", (root_prefix + "%",))}
        seen: set[str] = set()
        try:
            for path in sorted(docs_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                stats["discovered"] += 1
                relative = path.relative_to(WORKSPACE_ROOT).as_posix()
                seen.add(relative)
                try:
                    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
                except (UnicodeDecodeError, OSError):
                    stats["errors"] += 1
                    continue
                document_hash = _digest(text)
                if relative in known and known[relative]["content_hash"] == document_hash:
                    stats["unchanged"] += 1
                    continue
                doc_id = _digest(relative)
                title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.strip()), path.stem)[:300]
                db.execute(
                    """INSERT INTO documents(id,path,source_type,normalized_path,title,content_hash,indexed_at,updated_at,tags_json)
                       VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,
                       content_hash=excluded.content_hash,indexed_at=excluded.indexed_at,updated_at=excluded.updated_at,tags_json=excluded.tags_json""",
                    (doc_id, relative, path.suffix.lower().lstrip("."), relative, title, document_hash, now, now, json.dumps(_tags(text))),
                )
                db.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
                chunks = chunk_text(path, text, RAG_CHUNK_TOKENS, RAG_CHUNK_OVERLAP)
                for item in chunks:
                    chunk_id = _digest(f"{relative}:{item.start_line}:{item.end_line}:{item.content_hash}")
                    db.execute(
                        """INSERT INTO chunks(id,document_id,content,start_line,end_line,section,symbol,token_count,content_hash,created_at,updated_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (chunk_id, doc_id, item.content, item.start_line, item.end_line, item.section, item.symbol, item.token_count, item.content_hash, now, now),
                    )
                    db.execute("INSERT INTO chunk_fts(content,chunk_id) VALUES(?,?)", (item.content, chunk_id))
                stats["indexed"] += 1
                stats["chunks"] += len(chunks)
            for relative, document in known.items():
                if relative not in seen:
                    db.execute("DELETE FROM documents WHERE id=?", (document["id"],))
                    stats["deleted"] += 1
            db.execute("UPDATE ingest_jobs SET completed_at=?, status='complete', stats_json=? WHERE id=?", (time.time(), str(stats), job_id))
        except Exception as exc:
            db.execute("UPDATE ingest_jobs SET completed_at=?, status='failed', error=? WHERE id=?", (time.time(), str(exc), job_id))
            raise
    return stats


def reindex_all(store: RagStore | None = None) -> dict[str, int]:
    store = store or RagStore()
    results = [ingest_documents(RAG_DOCS_DIR, store)]
    if RAG_SOURCES_DIR.exists():
        results.append(ingest_documents(RAG_SOURCES_DIR, store))
    return {key: sum(result[key] for result in results) for key in results[0]}
