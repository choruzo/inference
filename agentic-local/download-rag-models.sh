#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${APP_DIR}/.." && pwd)/model"
VENV="$(cd "${APP_DIR}/.." && pwd)/venv"
mkdir -p "${MODEL_DIR}"

EMBEDDING_FILE="${MODEL_DIR}/bge-small-en-v1.5-Q8_0.gguf"
EMBEDDING_SHA256="cb23bbfa9bc2f2e2adf32fee567cc72bed2a4250bed09a42dc12bac81dc58bef"
if [[ ! -s "${EMBEDDING_FILE}" ]]; then
  curl -fL --retry 3 --continue-at - \
    -o "${EMBEDDING_FILE}" \
    https://huggingface.co/smarttasks/bge-small-en-v1.5-GGUF/resolve/main/bge-small-en-v1.5-Q8_0.gguf
fi
echo "${EMBEDDING_SHA256}  ${EMBEDDING_FILE}" | sha256sum --check --status || { echo "Checksum invalido para ${EMBEDDING_FILE}" >&2; exit 1; }

if [[ "${DOWNLOAD_OCR_MODEL:-0}" == "1" ]]; then
  if [[ ! -x "${VENV}/bin/hf" ]]; then
    echo "Falta ${VENV}/bin/hf; instala huggingface_hub en el entorno virtual." >&2
    exit 1
  fi
  "${VENV}/bin/hf" download stepfun-ai/GOT-OCR2_0 --local-dir "${MODEL_DIR}/GOT-OCR2_0"
  echo "77d6144039548b14253176b6eb264896bc39eba532f8894700f210a7fd2a5956  ${MODEL_DIR}/GOT-OCR2_0/model.safetensors" | sha256sum --check --status || { echo "Checksum invalido para GOT-OCR2_0" >&2; exit 1; }
fi

echo "Modelos RAG disponibles:"
du -sh "${EMBEDDING_FILE}"
if [[ "${DOWNLOAD_OCR_MODEL:-0}" == "1" ]]; then
  du -sh "${MODEL_DIR}/GOT-OCR2_0"
fi
