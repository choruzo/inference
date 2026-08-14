import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.getenv("AGENT_WORKSPACE", APP_ROOT / "workspace")).resolve()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://llm:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "local-gguf")
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))
RESPONSE_TIMEOUT = float(os.getenv("RESPONSE_TIMEOUT", "0"))

MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "-1"))
MAX_STRUCTURED_TOKENS = int(os.getenv("MAX_STRUCTURED_TOKENS", "2048"))
MAX_CHAT_HISTORY_MESSAGES = int(os.getenv("MAX_CHAT_HISTORY_MESSAGES", "40"))
MAX_TOOL_OUTPUT_CHARS = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "12000"))
MAX_FILE_READ_CHARS = int(os.getenv("MAX_FILE_READ_CHARS", "20000"))

# RAG capabilities are opt-in at request time. These flags allow operators to
# disable a capability globally without changing the API contract.
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
WEB_ENABLED = os.getenv("WEB_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
RAG_DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", WORKSPACE_ROOT / "docs")).resolve()
RAG_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", WORKSPACE_ROOT / ".rag_index")).resolve()
RAG_SOURCES_DIR = Path(os.getenv("RAG_SOURCES_DIR", WORKSPACE_ROOT / ".rag_sources")).resolve()
RAG_CACHE_DIR = Path(os.getenv("RAG_CACHE_DIR", WORKSPACE_ROOT / ".rag_cache")).resolve()
RAG_CHUNK_TOKENS = int(os.getenv("RAG_CHUNK_TOKENS", "450"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
RAG_RERANK_TOP_K = int(os.getenv("RAG_RERANK_TOP_K", "4"))
RAG_CANDIDATE_POOL = int(os.getenv("RAG_CANDIDATE_POOL", "40"))
RAG_CONTEXT_TOKENS = int(os.getenv("RAG_CONTEXT_TOKENS", "2400"))
RAG_MIN_VECTOR_SCORE = float(os.getenv("RAG_MIN_VECTOR_SCORE", "0.62"))

EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "bge-small-en-v1.5")
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "http://host.docker.internal:8091/v1")
EMBEDDINGS_DIMENSIONS = int(os.getenv("EMBEDDINGS_DIMENSIONS", "384"))
EMBEDDINGS_BATCH_SIZE = int(os.getenv("EMBEDDINGS_BATCH_SIZE", "16"))

RAG_UPLOAD_MAX_BYTES = int(os.getenv("RAG_UPLOAD_MAX_BYTES", str(50 * 1024 * 1024)))

OCR_PROVIDER = os.getenv("OCR_PROVIDER", "rapidocr")
OCR_MODEL = os.getenv("OCR_MODEL", "RapidOCR/PP-OCRv4")
OCR_MODEL_DIR = Path(os.getenv("OCR_MODEL_DIR", APP_ROOT.parent / "model" / "GOT-OCR2_0")).resolve()
MODEL_ROUTER_ENABLED = os.getenv("MODEL_ROUTER_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
MODEL_ROUTER_BASE_URL = os.getenv("MODEL_ROUTER_BASE_URL", "http://localhost:8080")
MODEL_ROUTER_MAX_MODELS = int(os.getenv("MODEL_ROUTER_MAX_MODELS", "2"))
MODEL_ROUTER_RESTORE_CHAT = os.getenv("MODEL_ROUTER_RESTORE_CHAT", "true").lower() in {"1", "true", "yes", "on"}

WEB_SEARCH_URL = os.getenv("WEB_SEARCH_URL", "https://html.duckduckgo.com/html/")
WEB_TIMEOUT = float(os.getenv("WEB_TIMEOUT", "15"))
WEB_MAX_BYTES = int(os.getenv("WEB_MAX_BYTES", "2000000"))
