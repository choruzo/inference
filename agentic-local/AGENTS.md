# Repository Guidelines

## Project Structure & Module Organization

`backend/` contains the Python 3.12 FastAPI application. `backend/main.py` defines the HTTP API and serves the static UI, while `backend/agent.py` implements the local model/tool loop. RAG indexing, retrieval, OCR, and evaluation live under `backend/rag/`; tool implementations live under `backend/tools/`. The framework-free frontend is in `frontend/` (`index.html`, `app.js`, and `styles.css`). Tests are in `tests/`, and runtime documents, indexes, and evaluation fixtures belong under `workspace/`. Treat sibling directories outside `agentic-local/` as supporting dependencies unless a task explicitly includes them.

## Build, Test, and Development Commands

- `./start-host-gpu.sh` starts the host Vulkan `llama-server` and the Dockerized app. Open `http://localhost:8000`.
- `./stop-host-gpu.sh` stops the local stack.
- `docker compose config` validates Compose configuration without starting services.
- `../venv/bin/pytest -q` runs the complete pytest suite.
- `python3 -m py_compile backend/*.py backend/tools/*.py backend/rag/*.py` performs a quick syntax check.
- `../venv/bin/python -m backend.rag.cli reindex` rebuilds the local document index.

## Coding Style & Naming Conventions

Use four-space indentation and standard PEP 8 naming in Python: `snake_case` for functions and modules, `PascalCase` for classes, and uppercase names for configuration constants. Add type annotations where they clarify public interfaces. Keep JavaScript framework-free and follow the existing `camelCase` style. Match the concise Spanish wording already used in the UI. No formatter or linter is currently enforced, so keep diffs focused and consistent with nearby code.

## Testing Guidelines

Pytest is configured through `pytest.ini`; test files belong in `tests/` and functions must start with `test_`. Add regression coverage for changes to API contracts, retrieval ranking, citations, OCR conversion, or model routing. Prefer deterministic clients, temporary paths, and monkeypatching over live model or network dependencies. Run the full suite before submitting changes; no numeric coverage threshold is currently defined.

## Commit & Pull Request Guidelines

Recent history mostly uses short imperative subjects, often Conventional Commit prefixes such as `feat:`. Prefer `feat: add upload validation` or `fix: preserve citation order`; keep each commit narrowly scoped. Pull requests should explain the motivation and behavior change, list validation commands, and link relevant issues. Include screenshots for frontend changes and call out configuration, model, VRAM, or migration impacts.

## Security & Configuration Tips

Keep filesystem access confined to `workspace/`; preserve the path checks that enforce `WORKSPACE_ROOT`. Never commit model weights, generated indexes, logs, credentials, or machine-specific secrets. Configure endpoints and limits through environment variables documented in `README.md` and `backend/config.py`.
