# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project scope

This workspace contains multiple sibling directories, but **`agentic-local/` is the primary project**. Do not make changes in `llama.cpp/`, `model/`, or `venv/` unless explicitly asked:

- `llama.cpp/` — vendored upstream llama.cpp checkout, built locally to produce `build-vulkan/bin/llama-server`.
- `model/` — GGUF weights and the GOT-OCR2_0 model directory (large binary artifacts, not source).
- `venv/` — Python virtualenv used to run `agentic-local/backend` and its RAG CLI outside Docker.

All commands below assume `cd agentic-local` first unless noted otherwise.

## Commands

Run the app (host GPU llama-server + Dockerized FastAPI app):

```bash
./start-host-gpu.sh     # starts llama-server on :8080, embeddings on :8091, app on :8000
./stop-host-gpu.sh
```

If Docker needs elevated permissions, run `sudo -v` once first or run the script from an interactive terminal.

Python syntax check:

```bash
python3 -m py_compile backend/*.py backend/tools/*.py
```

Docker config validation:

```bash
docker compose config      # use `sudo docker compose ...` if Docker needs elevation on this machine
```

Tests (pytest config lives in `agentic-local/pytest.ini`, `testpaths = tests`):

```bash
../venv/bin/pytest -q
../venv/bin/pytest tests/test_rag.py::test_chunking_preserves_sections_symbols_and_lines   # single test
```

RAG CLI (reindex / embed / status / convert / evaluate), run from `agentic-local/`:

```bash
../venv/bin/python -m backend.rag.cli reindex
EMBEDDINGS_BASE_URL=http://127.0.0.1:8091/v1 ../venv/bin/python -m backend.rag.cli embed
../venv/bin/python -m backend.rag.cli status
../venv/bin/python -m backend.rag.cli convert documento.pdf   # offline PDF/DOCX -> Markdown
LLM_BASE_URL=http://127.0.0.1:8080/v1 EMBEDDINGS_BASE_URL=http://127.0.0.1:8091/v1 \
  ../venv/bin/python -m backend.rag.cli evaluate
```

The OCR worker is a separate optional install: `../venv/bin/pip install -r backend/requirements-ocr.txt` (add `DOWNLOAD_OCR_MODEL=1 ./download-rag-models.sh` for the GOT-OCR2_0 weights).

## Architecture

`backend/` is a Python 3.12 FastAPI app (`backend/main.py`) that serves both the JSON API and the static `frontend/` (plain HTML/CSS/JS, mounted at `/assets`, no build step). The chat endpoint (`POST /api/chat`) is the single entry point; requests carry a `ChatRequest` (`backend/contracts.py`) with a `modes: ChatModes` field (`backend/modes.py`) selecting **chat**, **rag**, and/or **web**, plus a `reasoning_panel` toggle. `LocalAgent.chat()` in `backend/agent.py` dispatches on those modes to one of three code paths:

- **Legacy/tool-agent path** (`modes is None`): the original ReAct-style loop. It sends the system prompt with the tool registry description, expects the model to respond with strict JSON (`{"tool": ..., "args": ...}` or `{"final": ...}`) via `response_format` JSON Schema, executes the tool through `backend/tools/registry`, feeds the observation back, and loops up to `MAX_AGENT_STEPS`.
- **`_plain_chat`**: free-form chat with no tools and no JSON schema constraint, so `Thinking` models can emit `<think>...</think>` freely (stripped by `_strip_thinking`/`_thinking` before being shown/returned as `reasoning`).
- **`_rag_chat`**: hybrid retrieval (`backend/rag/search.py`: BM25/FTS5 + vector cosine search, fused with reciprocal rank fusion) → LLM-based rerank (`backend/rag/rerank.py`, structured JSON output) → an answer prompt that requires every factual paragraph to carry a `[n]` citation → citation validation against the store (`backend/rag/citations.py`). If citations can't be validated or paragraphs aren't cited, the answer is replaced with a fixed "not found" response rather than let ungrounded text through.

Tools live in `backend/tools/`: `base.py` defines a simple `Tool`/`ToolRegistry` pattern; `filesystem.py` registers workspace-scoped tools (`list_dir`, `glob`, `find`, `read_file`, `write_file`, `edit_file`, `file_info`) all funneled through `_safe_path()`, which resolves paths against `WORKSPACE_ROOT` and rejects anything that escapes it; `rag.py` registers `rag_status`/`rag_reindex`/`rag_search`. `backend/tools/__init__.py` merges both into one `registry` instance. **Any tool added to the registry must also be added to the JSON Schema `enum` in `backend/agent.py`** (`AGENT_RESPONSE_FORMAT`/`TOOL_RESPONSE_FORMAT`) or the structured-output-constrained model can never call it.

`workspace/` (mounted as `/workspace` in Docker) is the only filesystem area tools may read or write — this boundary is enforced by `_safe_path()`, not by convention, so preserve it when adding tools.

### RAG subsystem (`backend/rag/`)

- `documents.py` — offline document conversion (PDF via `pypdf`, DOCX via `python-docx`) to canonical Markdown stored under `workspace/.rag_sources`; pages/images without extractable text fall back to RapidOCR/ONNX on CPU with a hash-based cache in `workspace/.rag_cache`. `ModelRouterSession` here coordinates with the optional model router (below) so OCR/embeddings can borrow VRAM/CPU without permanently evicting the chat model.
- `chunking.py` — splits Markdown/text/code into token-bounded, section-aware chunks (`RAG_CHUNK_TOKENS`/`RAG_CHUNK_OVERLAP`) preserving line ranges for citation.
- `store.py` — `RagStore`: SQLite + FTS5 index at `workspace/.rag_index/index.sqlite`, with a `SCHEMA_VERSION`-gated migration path; each chunk row keeps path, section, line range, content hash, and embedding version.
- `embeddings.py` — `EmbeddingClient` against an OpenAI-compatible embeddings endpoint (default `bge-small-en-v1.5` served by `start-embeddings.sh` on `:8091`).
- `search.py` — `vector_search` (cosine over stored embeddings) fused with FTS5 keyword search via `reciprocal_rank_fusion`; `RAG_MIN_VECTOR_SCORE` filters weak matches.
- `rerank.py` — LLM-driven reranking of the fused candidate pool down to `RAG_RERANK_TOP_K`.
- `citations.py` — builds numbered citations from ranked chunks and validates them still exist in the store before an answer can cite them.
- `ingest.py` — `reindex_all()`/`ingest_documents()`: incremental reindex of `workspace/docs` plus canonical sources, used by both the CLI and the `rag_reindex` tool/endpoint.
- `evaluate.py` / `cli.py` — reproducible evaluation harness against `workspace/evals/rag_golden.jsonl`, writing the last run to `workspace/.rag_cache/last_evaluation.json`; scores retrieval, citation emission, and answer grounding separately.

### Model router

When `MODEL_ROUTER_ENABLED=true`, `llama-server` runs with `--models-preset rag-models.ini --models-max 1`, keeping only one model resident at a time (tuned for a 4GB GTX 1050). The app then swaps models in/out via `/models/unload` and `/models/load` around indexing work, restoring the chat model afterward even if the job fails (`MODEL_ROUTER_RESTORE_CHAT`). This mode is for testing true sequential loading — stop any normal run first before starting it.

### Configuration

All environment variables are centralized in `backend/config.py` (workspace paths, LLM connection, RAG tuning, embeddings, OCR, model router, web search). `docker-compose.yml` sets container defaults and mounts `./workspace` as `/workspace`; `start-host-gpu.sh` overrides `LLM_BASE_URL`/`EMBEDDINGS_BASE_URL` to reach the host-run `llama-server`/embeddings server via `host.docker.internal`. Chat and RAG responses default to `MAX_RESPONSE_TOKENS=-1` (generate until the model stops or context is exhausted) with `RESPONSE_TIMEOUT=0` (no timeout); tool calls and reranking instead use `MAX_STRUCTURED_TOKENS` and always run under a JSON Schema `response_format` — chat intentionally does not, so `Thinking` models can emit `<think>` freely.

## Change guidance

- Keep backend changes compatible with FastAPI and the existing `Tool`/`ToolRegistry` pattern; don't introduce a new DI or plugin framework for tools.
- When adding a tool: implement it in `backend/tools/filesystem.py` (or a new focused module merged into `backend/tools/__init__.py`), register it with `ToolRegistry`, **and** add it to the schema enum in `backend/agent.py`.
- Never bypass `_safe_path()` / the `WORKSPACE_ROOT` boundary in new tools.
- Keep `frontend/` framework-free (plain JS/CSS/HTML) unless the user explicitly asks for a build system.
- UI copy is Spanish; match the existing app language in user-facing strings.
- RAG answers must stay grounded: any change to `_rag_chat`, `citations.py`, or the answer prompt should preserve the invariant that every factual paragraph carries a `[n]` citation validated against the store, falling back to "No lo encuentro en la documentacion indexada." otherwise.
