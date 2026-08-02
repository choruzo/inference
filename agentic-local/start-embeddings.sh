#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${APP_DIR}/.." && pwd)"
BIN="${ROOT_DIR}/llama.cpp/build-vulkan/bin"
MODEL="${EMBEDDINGS_MODEL_PATH:-${ROOT_DIR}/model/bge-small-en-v1.5-Q8_0.gguf}"
PORT="${EMBEDDINGS_PORT:-8091}"
PID_FILE="${APP_DIR}/logs/embeddings-server.pid"

mkdir -p "${APP_DIR}/logs"
if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Embeddings ya disponibles en http://127.0.0.1:${PORT}"
  exit 0
fi
if (echo >/dev/tcp/127.0.0.1/"${PORT}") >/dev/null 2>&1; then
  echo "El puerto ${PORT} esta ocupado por otro proceso; usa EMBEDDINGS_PORT con otro puerto." >&2
  exit 1
fi
[[ -x "${BIN}/llama-server" ]] || { echo "No encuentro llama-server" >&2; exit 1; }
[[ -s "${MODEL}" ]] || { echo "No encuentro ${MODEL}; ejecuta ./download-rag-models.sh" >&2; exit 1; }
setsid env LD_LIBRARY_PATH="${BIN}:${LD_LIBRARY_PATH:-}" "${BIN}/llama-server" \
  --model "${MODEL}" --embedding --pooling cls --host 0.0.0.0 --port "${PORT}" \
  --ctx-size 512 --batch-size 512 --threads "${EMBEDDINGS_THREADS:-4}" \
  --n-gpu-layers "${EMBEDDINGS_GPU_LAYERS:-0}" >"${APP_DIR}/logs/embeddings-server.log" 2>&1 &
echo "$!" > "${PID_FILE}"
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && exit 0
  sleep 1
done
echo "El servidor de embeddings no arranco" >&2
exit 1
