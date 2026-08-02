from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.modes import ChatModes


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    modes: ChatModes | None = None


class Citation(BaseModel):
    id: int
    chunk_id: str
    path: str
    title: str = ""
    section: str = ""
    start_line: int
    end_line: int
    quote: str = ""
    source_type: Literal["local", "web"] = "local"


class RetrievalChunk(BaseModel):
    chunk_id: str
    path: str
    title: str = ""
    section: str = ""
    start_line: int
    end_line: int
    content: str
    score: float
    reasons: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievalChunk] = Field(default_factory=list)
    reranked: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    type: Literal["tool", "rag", "web", "rerank", "model"]
    event: str
    data: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    reasoning: str | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    retrieval: RetrievalResult | None = None
    workspace: str
    stopped: str
