from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import httpx

from backend.config import EMBEDDINGS_BASE_URL, EMBEDDINGS_BATCH_SIZE, EMBEDDINGS_DIMENSIONS, EMBEDDINGS_MODEL, EMBEDDINGS_PROVIDER, REQUEST_TIMEOUT
from backend.rag.store import RagStore


class EmbeddingClient:
    def __init__(self, provider: str = EMBEDDINGS_PROVIDER, model: str = EMBEDDINGS_MODEL, base_url: str = EMBEDDINGS_BASE_URL, dimensions: int = EMBEDDINGS_DIMENSIONS) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        if self.provider == "deterministic":
            return [self._deterministic(text, self.dimensions) for text in texts]
        prefixed = [f"Represent this sentence for searching relevant passages: {text}" if query else text for text in texts]
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(f"{self.base_url}/embeddings", json={"model": self.model, "input": prefixed})
            response.raise_for_status()
            payload = response.json()
        ordered = sorted(payload["data"], key=lambda item: item.get("index", 0))
        vectors = [list(map(float, item["embedding"])) for item in ordered]
        if vectors:
            self.dimensions = len(vectors[0])
        return vectors

    @staticmethod
    def _deterministic(text: str, dimensions: int) -> list[float]:
        vector = [0.0] * dimensions
        words = text.lower().split()
        for word in words:
            digest = hashlib.sha256(word.encode()).digest()
            for offset in range(0, len(digest), 4):
                index = int.from_bytes(digest[offset : offset + 2], "little") % dimensions
                vector[index] += 1.0 if digest[offset + 2] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def index_embeddings(store: RagStore | None = None, client: EmbeddingClient | None = None, batch_size: int = EMBEDDINGS_BATCH_SIZE) -> dict[str, int | str]:
    store = store or RagStore()
    client = client or EmbeddingClient()
    missing = store.chunks_missing_embeddings(client.model)
    indexed = 0
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = client.embed([str(item["content"]) for item in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding endpoint returned an unexpected vector count")
        store.save_embeddings(client.model, [(str(item["id"]), str(item["content_hash"]), vector) for item, vector in zip(batch, vectors)])
        indexed += len(batch)
    return {"model": client.model, "indexed": indexed, "unchanged": max(0, store.status()["chunks"] - indexed)}
