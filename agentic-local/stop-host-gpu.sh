#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${APP_DIR}/logs/llama-server.pid"

docker_compose_stop_app() {
  if docker compose version >/dev/null 2>&1 && docker ps >/dev/null 2>&1; then
    docker compose stop app >/dev/null
    return
  fi

  if sudo -n true >/dev/null 2>&1 && sudo -n docker ps >/dev/null 2>&1; then
    sudo -n docker compose stop app >/dev/null
    return
  fi

  if [[ -t 0 ]]; then
    sudo docker compose stop app >/dev/null
    return
  fi

  echo "No pude parar Docker porque requiere sudo interactivo." >&2
}

cd "${APP_DIR}"
docker_compose_stop_app

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" >/dev/null 2>&1; then
    kill "${PID}"
    echo "llama-server parado: ${PID}"
  fi
  rm -f "${PID_FILE}"
fi

echo "app parada"
