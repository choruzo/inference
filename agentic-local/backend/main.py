from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agent import LocalAgent
from backend.config import APP_ROOT, LLM_BASE_URL, RAG_UPLOAD_MAX_BYTES, WORKSPACE_ROOT
from backend.contracts import ChatRequest, ChatResponse
from backend.rag.documents import convert_to_markdown
from backend.rag.embeddings import index_embeddings
from backend.rag.ingest import reindex_all
from backend.rag.store import RagStore
from backend.tools import registry


RAG_UPLOAD_SUFFIXES = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


class ToolRequest(BaseModel):
    name: str = Field(min_length=1)
    args: dict[str, object] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("docs", ".rag_index", ".rag_sources", ".rag_cache", "evals"):
        (WORKSPACE_ROOT / name).mkdir(parents=True, exist_ok=True)
    agent = LocalAgent()
    app.state.agent = agent
    yield
    await agent.close()


app = FastAPI(title="Local Agentic LLM", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "llm_base_url": LLM_BASE_URL, "workspace": str(WORKSPACE_ROOT)}


@app.get("/api/tools")
async def tools() -> dict[str, str]:
    return {"tools": registry.describe_for_prompt()}


@app.post("/api/tool")
async def run_tool(request: ToolRequest):
    try:
        return {"ok": True, "result": registry.run(request.name, request.args)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    history = [item.model_dump() for item in request.history]
    return ChatResponse.model_validate(await app.state.agent.chat(request.message, history, request.modes))


@app.get("/api/rag/status")
async def rag_status() -> dict[str, object]:
    return RagStore().status()


@app.post("/api/rag/reindex")
async def rag_reindex() -> dict[str, int]:
    return reindex_all()


@app.post("/api/rag/convert")
async def rag_convert(file: UploadFile = File(...)) -> dict[str, object]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in RAG_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Tipo de archivo no soportado: {suffix or '(sin extension)'}")
    safe_name = Path(file.filename or "").name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Nombre de archivo invalido")
    data = await file.read()
    if len(data) > RAG_UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande")
    dest_dir = WORKSPACE_ROOT / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe_name
    dest.write_bytes(data)
    try:
        result = convert_to_markdown(dest)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
    if "embedding_index" not in result:
        try:
            result["embedding_index"] = index_embeddings()
        except Exception as exc:
            result["embedding_index"] = {"skipped": f"{type(exc).__name__}: {exc}"}
    return result


@app.get("/api/source", response_class=PlainTextResponse)
async def source(path: str, line: int = 1) -> str:
    candidate = (WORKSPACE_ROOT / path).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path escapes workspace") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Source not found")
    lines = candidate.read_text(encoding="utf-8").splitlines()
    start = max(0, line - 4)
    return "\n".join(f"{number:>6}  {value}" for number, value in enumerate(lines[start : start + 200], start + 1))


FRONTEND_DIR = APP_ROOT / "frontend"
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(FRONTEND_DIR / "index.html"))
