from __future__ import annotations

import json
import re
import sqlite3
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from backend.config import RAG_INDEX_DIR


SCHEMA_VERSION = 2
STOPWORDS = {
    "a", "al", "and", "como", "con", "cual", "de", "del", "el", "en", "es", "esta", "for", "how", "la", "las", "lo", "los", "of", "on", "or", "para", "por", "que", "se", "the", "to", "un", "una", "y",
}


class RagStore:
    def __init__(self, index_dir: Path = RAG_INDEX_DIR) -> None:
        self.index_dir = index_dir
        self.path = index_dir / "index.sqlite"

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        self._migrate(connection)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migrate(self, db: sqlite3.Connection) -> None:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS documents(
              id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, source_type TEXT NOT NULL,
              normalized_path TEXT NOT NULL, title TEXT NOT NULL, content_hash TEXT NOT NULL,
              indexed_at REAL NOT NULL, updated_at REAL NOT NULL, tags_json TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
            CREATE TABLE IF NOT EXISTS chunks(
              id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
              content TEXT NOT NULL, start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
              section TEXT NOT NULL, symbol TEXT NOT NULL DEFAULT '', token_count INTEGER NOT NULL,
              content_hash TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(content, chunk_id UNINDEXED, tokenize='unicode61');
            CREATE TABLE IF NOT EXISTS embeddings(
              chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE, model TEXT NOT NULL,
              dimensions INTEGER NOT NULL, vector_blob BLOB NOT NULL, content_hash TEXT NOT NULL,
              created_at REAL NOT NULL, PRIMARY KEY(chunk_id, model)
            );
            CREATE TABLE IF NOT EXISTS sources(
              id TEXT PRIMARY KEY, original_path TEXT NOT NULL, normalized_markdown_path TEXT NOT NULL,
              converter TEXT NOT NULL, converter_version TEXT NOT NULL, ocr_model TEXT,
              metadata_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ingest_jobs(
              id INTEGER PRIMARY KEY AUTOINCREMENT, started_at REAL NOT NULL, completed_at REAL,
              status TEXT NOT NULL, stats_json TEXT NOT NULL DEFAULT '{}', error TEXT
            );
            CREATE TABLE IF NOT EXISTS rerank_cache(
              cache_key TEXT PRIMARY KEY, result_json TEXT NOT NULL, created_at REAL NOT NULL
            );
            """
        )
        columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
        if "tags_json" not in columns:
            db.execute("ALTER TABLE documents ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]'")
        for version in range(1, SCHEMA_VERSION + 1):
            db.execute("INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)", (version, time.time()))

    def status(self) -> dict[str, object]:
        with self.connect() as db:
            counts = {name: db.execute(f"SELECT count(*) FROM {name}").fetchone()[0] for name in ("documents", "chunks", "embeddings", "sources")}
            model_rows = db.execute("SELECT model, count(*) count FROM embeddings GROUP BY model").fetchall()
            return {**counts, "embedding_models": {row["model"]: row["count"] for row in model_rows}, "database": str(self.path), "schema_version": SCHEMA_VERSION}

    def keyword_search(self, query: str, limit: int = 20) -> list[dict[str, object]]:
        raw_terms = re.findall(r"[\w./:=_-]+", query, flags=re.UNICODE)
        terms = [term for term in raw_terms if term.lower() not in STOPWORDS and (len(term) >= 3 or term.isdigit() or term.isupper())]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms[:20])
        with self.connect() as db:
            rows = db.execute(
                """SELECT c.*, d.path, d.title, d.source_type, d.updated_at AS document_updated_at, d.tags_json, bm25(chunk_fts) AS rank
                   FROM chunk_fts JOIN chunks c ON c.id=chunk_fts.chunk_id
                   JOIN documents d ON d.id=c.document_id
                   WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
        return [dict(row) | {"score": 1.0 / (1.0 + abs(float(row["rank"]))), "reasons": ["keyword"]} for row in rows]

    def chunks_with_embeddings(self, model: str) -> list[dict[str, object]]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT c.*, d.path, d.title, d.source_type, d.updated_at AS document_updated_at, d.tags_json, e.dimensions, e.vector_blob
                   FROM embeddings e JOIN chunks c ON c.id=e.chunk_id JOIN documents d ON d.id=c.document_id
                   WHERE e.model=?""",
                (model,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["vector"] = list(struct.unpack(f"<{row['dimensions']}f", row["vector_blob"]))
            del item["vector_blob"]
            result.append(item)
        return result

    def chunks_missing_embeddings(self, model: str, limit: int = 1000) -> list[dict[str, object]]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT c.id, c.content, c.content_hash FROM chunks c
                   LEFT JOIN embeddings e ON e.chunk_id=c.id AND e.model=?
                   WHERE e.chunk_id IS NULL OR e.content_hash != c.content_hash LIMIT ?""",
                (model, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_embeddings(self, model: str, vectors: Sequence[tuple[str, str, Sequence[float]]]) -> None:
        now = time.time()
        with self.connect() as db:
            for chunk_id, content_hash, vector in vectors:
                blob = struct.pack(f"<{len(vector)}f", *vector)
                db.execute(
                    "INSERT OR REPLACE INTO embeddings(chunk_id,model,dimensions,vector_blob,content_hash,created_at) VALUES(?,?,?,?,?,?)",
                    (chunk_id, model, len(vector), blob, content_hash, now),
                )

    def validate_chunk_ids(self, ids: Iterable[str]) -> set[str]:
        ids = list(ids)
        if not ids:
            return set()
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as db:
            return {row[0] for row in db.execute(f"SELECT id FROM chunks WHERE id IN ({placeholders})", ids)}

    def cache_get(self, key: str) -> list[str] | None:
        with self.connect() as db:
            row = db.execute("SELECT result_json FROM rerank_cache WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, key: str, ids: list[str]) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO rerank_cache VALUES(?,?,?)", (key, json.dumps(ids), time.time()))
