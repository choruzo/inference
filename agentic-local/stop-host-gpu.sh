#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${APP_DIR}/logs/llama-server.pid"
EMBEDDINGS_PID_FILE="${APP_DIR}/logs/embeddings-server.pid"

docker_compose_stop_services() {
  if docker compose version >/dev/null 2>&1 && docker ps >/dev/null 2>&1; then
    docker compose --profile web-search stop app searxng >/dev/null
    return
  fi

  if sudo -n true >/dev/null 2>&1 && sudo -n docker ps >/dev/null 2>&1; then
    sudo -n docker compose --profile web-search stop app searxng >/dev/null
    return
  fi

  if [[ -t 0 ]]; then
    sudo docker compose --profile web-search stop app searxng >/dev/null
    return
  fi

  echo "No pude parar Docker porque requiere sudo interactivo." >&2
}

cd "${APP_DIR}"
docker_compose_stop_services

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" >/dev/null 2>&1; then
    kill "${PID}"
    echo "llama-server parado: ${PID}"
  fi
  rm -f "${PID_FILE}"
fi

if [[ -f "${EMBEDDINGS_PID_FILE}" ]]; then
  EMBEDDINGS_PID="$(cat "${EMBEDDINGS_PID_FILE}")"
  kill "${EMBEDDINGS_PID}" >/dev/null 2>&1 || true
  rm -f "${EMBEDDINGS_PID_FILE}"
fi

echo "app y SearXNG parados"
