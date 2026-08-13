#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${APP_DIR}/.." && pwd)"
LLAMA_BIN="${ROOT_DIR}/llama.cpp/build-vulkan/bin"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/model/LFM2.5-1.2B-Thinking-Q4_K_M.gguf}"
LOG_DIR="${APP_DIR}/logs"
PID_FILE="${LOG_DIR}/llama-server.pid"
LLAMA_BIND_HOST="${LLAMA_BIND_HOST:-0.0.0.0}"
LLAMA_HEALTH_HOST="${LLAMA_HEALTH_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
EMBEDDINGS_PORT="${EMBEDDINGS_PORT:-8091}"
MODEL_ROUTER_ENABLED="${MODEL_ROUTER_ENABLED:-false}"
MODEL_ROUTER_PRESET="${MODEL_ROUTER_PRESET:-${APP_DIR}/rag-models.ini}"
if [[ "${MODEL_ROUTER_ENABLED}" == "1" || "${MODEL_ROUTER_ENABLED}" == "true" ]]; then
  EMBEDDINGS_EFFECTIVE_PORT="${LLAMA_PORT}"
else
  EMBEDDINGS_EFFECTIVE_PORT="${EMBEDDINGS_PORT}"
fi

mkdir -p "${LOG_DIR}"
cd "${APP_DIR}"

docker_compose() {
  if docker compose version >/dev/null 2>&1 && docker ps >/dev/null 2>&1; then
    env MODEL_ROUTER_ENABLED="${MODEL_ROUTER_ENABLED}" MODEL_ROUTER_BASE_URL="http://host.docker.internal:${LLAMA_PORT}" LLM_BASE_URL="http://host.docker.internal:${LLAMA_PORT}/v1" EMBEDDINGS_BASE_URL="http://host.docker.internal:${EMBEDDINGS_EFFECTIVE_PORT}/v1" docker compose "$@"
    return
  fi

  if sudo -n true >/dev/null 2>&1; then
    if sudo -n env MODEL_ROUTER_ENABLED="${MODEL_ROUTER_ENABLED}" MODEL_ROUTER_BASE_URL="http://host.docker.internal:${LLAMA_PORT}" LLM_BASE_URL="http://host.docker.internal:${LLAMA_PORT}/v1" EMBEDDINGS_BASE_URL="http://host.docker.internal:${EMBEDDINGS_EFFECTIVE_PORT}/v1" docker compose "$@"; then
      return
    fi
  fi

  if [[ -t 0 ]]; then
    sudo env MODEL_ROUTER_ENABLED="${MODEL_ROUTER_ENABLED}" MODEL_ROUTER_BASE_URL="http://host.docker.internal:${LLAMA_PORT}" LLM_BASE_URL="http://host.docker.internal:${LLAMA_PORT}/v1" EMBEDDINGS_BASE_URL="http://host.docker.internal:${EMBEDDINGS_EFFECTIVE_PORT}/v1" docker compose "$@"
    return
  fi

  echo "Docker requiere sudo y esta sesion no tiene sudo autenticado." >&2
  echo "Ejecuta primero: sudo -v" >&2
  echo "Despues repite: ./start-host-gpu.sh" >&2
  exit 1
}

if curl -fsS "http://${LLAMA_HEALTH_HOST}:${LLAMA_PORT}/health" >/dev/null 2>&1; then
  echo "llama-server ya esta respondiendo en http://${LLAMA_HEALTH_HOST}:${LLAMA_PORT}"
else
  if [[ ! -x "${LLAMA_BIN}/llama-server" ]]; then
    echo "No encuentro llama-server ejecutable en ${LLAMA_BIN}/llama-server" >&2
    exit 1
  fi

  if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "No encuentro el modelo en ${MODEL_PATH}" >&2
    exit 1
  fi

  if [[ "${MODEL_ROUTER_ENABLED}" == "1" || "${MODEL_ROUTER_ENABLED}" == "true" ]]; then
    [[ -f "${MODEL_ROUTER_PRESET}" ]] || { echo "No encuentro el preset ${MODEL_ROUTER_PRESET}" >&2; exit 1; }
    echo "Arrancando llama-server en modo router secuencial..."
    LLAMA_ARGS=(--models-preset "${MODEL_ROUTER_PRESET}" --models-max "${MODEL_ROUTER_MAX_MODELS:-2}" --no-models-autoload --host "${LLAMA_BIND_HOST}" --port "${LLAMA_PORT}")
  else
    echo "Arrancando llama-server en host con build Vulkan..."
    LLAMA_ARGS=(--model "${MODEL_PATH}" --host "${LLAMA_BIND_HOST}" --port "${LLAMA_PORT}" --ctx-size "${LLAMA_CTX_SIZE:-128000}" --parallel "${LLAMA_PARALLEL:-1}" --threads "${LLAMA_THREADS:-8}" --n-gpu-layers "${LLAMA_GPU_LAYERS:-99}")
  fi
  setsid env LD_LIBRARY_PATH="${LLAMA_BIN}:${LD_LIBRARY_PATH:-}" "${LLAMA_BIN}/llama-server" "${LLAMA_ARGS[@]}" >"${LOG_DIR}/llama-server.log" 2>&1 &
  echo "$!" > "${PID_FILE}"

  echo "Esperando a que llama-server cargue el modelo..."
  for _ in $(seq 1 120); do
    if curl -fsS "http://${LLAMA_HEALTH_HOST}:${LLAMA_PORT}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if ! curl -fsS "http://${LLAMA_HEALTH_HOST}:${LLAMA_PORT}/health" >/dev/null 2>&1; then
    echo "llama-server no respondio. Revisa ${LOG_DIR}/llama-server.log" >&2
    exit 1
  fi
fi

if [[ "${START_EMBEDDINGS:-1}" == "1" && "${EMBEDDINGS_EFFECTIVE_PORT}" != "${LLAMA_PORT}" ]]; then
  EMBEDDINGS_PORT="${EMBEDDINGS_PORT}" "${APP_DIR}/start-embeddings.sh"
fi

echo "Arrancando app Docker..."
cd "${APP_DIR}"
docker_compose up -d --build app

echo "Listo:"
echo "  UI:  http://localhost:8000"
echo "  LLM: http://${LLAMA_HEALTH_HOST}:${LLAMA_PORT}"
echo "  Embeddings: http://127.0.0.1:${EMBEDDINGS_EFFECTIVE_PORT}"
echo "  Logs llama-server: ${LOG_DIR}/llama-server.log"
