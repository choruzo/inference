#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${APP_DIR}/.." && pwd)"
[[ $# -eq 1 ]] || { echo "Uso: ./run-ocr-ingest.sh ruta-relativa-al-workspace" >&2; exit 2; }

CHAT_WAS_RUNNING=0
curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && CHAT_WAS_RUNNING=1
"${APP_DIR}/stop-host-gpu.sh"

restart() {
  if [[ "${CHAT_WAS_RUNNING}" == "1" ]]; then
    "${APP_DIR}/start-host-gpu.sh"
  else
    "${APP_DIR}/start-embeddings.sh"
  fi
}
trap restart EXIT

cd "${APP_DIR}"
"${ROOT_DIR}/venv/bin/python" -m backend.rag.cli convert "$1"
"${ROOT_DIR}/venv/bin/python" -m backend.rag.cli reindex
"${APP_DIR}/start-embeddings.sh"
EMBEDDINGS_BASE_URL=http://127.0.0.1:8091/v1 "${ROOT_DIR}/venv/bin/python" -m backend.rag.cli embed
