import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.getenv("AGENT_WORKSPACE", APP_ROOT / "workspace")).resolve()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://llm:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "local-gguf")
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))

MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "512"))
MAX_TOOL_OUTPUT_CHARS = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "12000"))
MAX_FILE_READ_CHARS = int(os.getenv("MAX_FILE_READ_CHARS", "20000"))
