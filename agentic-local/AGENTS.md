# Agentic Local Notes

## Project Scope

- Treat `agentic-local/` as the primary project in this workspace.
- Do not make changes in sibling directories such as `llama.cpp/`, `model/`, or `venv/` unless the user explicitly asks.
- There is no git repository rooted at `agentic-local/` in the current workspace, so preserve unrelated local files and avoid assuming git history is available.

## Architecture

- `backend/` is a Python 3.12 FastAPI app.
- `backend/main.py` exposes the HTTP API and serves the static frontend.
- `backend/agent.py` contains the local agent loop. It calls an OpenAI-compatible `llama-server` chat completions endpoint and expects JSON output for either one tool call or a final answer.
- `backend/tools/` contains the tool registry and filesystem tools.
- `frontend/` is plain HTML, CSS, and JavaScript served from FastAPI under `/assets`; there is no frontend build step.
- `workspace/` is the only filesystem area the in-app agent is allowed to read or write.

## Runtime

- Recommended startup:

```bash
cd agentic-local
./start-host-gpu.sh
```

- Recommended shutdown:

```bash
cd agentic-local
./stop-host-gpu.sh
```

- UI runs at `http://localhost:8000`.
- Host `llama-server` runs at `http://localhost:8080`.
- The Docker app service talks to the host LLM through `LLM_BASE_URL`, defaulting to `http://host.docker.internal:8080/v1`.
- The optional `container-llm` compose profile exists, but the README notes Docker was not seeing the GPU on this machine.

## Configuration

- Main config lives in `backend/config.py`.
- Important environment variables:
  - `AGENT_WORKSPACE`
  - `LLM_BASE_URL`
  - `LLM_MODEL`
  - `LLM_TIMEOUT`
  - `MAX_AGENT_STEPS`
  - `MAX_RESPONSE_TOKENS`
  - `MAX_TOOL_OUTPUT_CHARS`
  - `MAX_FILE_READ_CHARS`
- `docker-compose.yml` sets the app defaults and mounts `./workspace` as `/workspace`.

## Development Checks

- For Python syntax checks, run:

```bash
python3 -m py_compile backend/*.py backend/tools/*.py
```

- For Docker configuration checks, run:

```bash
docker compose config
```

- Use `sudo docker compose ...` only when Docker requires elevated permissions on this machine.

## Change Guidance

- Keep backend changes compatible with FastAPI and the existing simple registry pattern.
- When adding a tool, register it in `backend/tools/filesystem.py` or add a focused module and export/merge the registry through `backend/tools/__init__.py`.
- Any new tool exposed to the LLM must also be reflected in the JSON schema enum in `backend/agent.py`.
- Preserve the workspace boundary enforced by `_safe_path`; tools must not escape `WORKSPACE_ROOT`.
- Keep frontend changes framework-free unless the user explicitly requests a frontend build system.
- Prefer concise Spanish UI copy, matching the existing app language.
