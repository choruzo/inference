from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agent import LocalAgent
from backend.config import APP_ROOT, LLM_BASE_URL, WORKSPACE_ROOT
from backend.tools import registry


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class ToolRequest(BaseModel):
    name: str = Field(min_length=1)
    args: dict[str, object] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
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
async def chat(request: ChatRequest):
    history = [item.model_dump() for item in request.history]
    return await app.state.agent.chat(request.message, history)


FRONTEND_DIR = APP_ROOT / "frontend"
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(FRONTEND_DIR / "index.html"))
