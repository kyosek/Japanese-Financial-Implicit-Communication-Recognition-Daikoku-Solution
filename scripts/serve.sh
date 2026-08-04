#!/usr/bin/env bash
# Starts a local llama.cpp server (native arm64, Metal-accelerated) on an
# OpenAI-compatible endpoint, hosting a GGUF model from models/.
#
# Usage:
#   scripts/serve.sh [model-filename-in-models-dir]
#   scripts/serve.sh gpt-oss-20b-F16.gguf
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BIN="$ROOT/runtime/llama.cpp/llama-b10244"
MODEL_FILE="${1:-shisa-v2-qwen2.5-32b.Q4_K_M.gguf}"
MODEL="$ROOT/models/$MODEL_FILE"
ALIAS="${MODEL_FILE%.gguf}"

if [ ! -f "$MODEL" ]; then
  echo "Model not found at $MODEL — download it into models/ first." >&2
  exit 1
fi

export DYLD_LIBRARY_PATH="$LLAMA_BIN"
exec "$LLAMA_BIN/llama-server" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  -c 8192 \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8080
