#!/usr/bin/env bash
# Downloads a GGUF model file from a Hugging Face repo into models/.
#
# Usage:
#   scripts/download_model.sh <hf-repo> <filename>
#   scripts/download_model.sh mradermacher/shisa-v2-qwen2.5-32b-GGUF shisa-v2-qwen2.5-32b.Q4_K_M.gguf
set -euo pipefail

REPO="${1:?usage: download_model.sh <hf-repo> <filename>}"
FILE="${2:?usage: download_model.sh <hf-repo> <filename>}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/models"
curl -L -C - --retry 3 --retry-delay 5 \
  -o "$ROOT/models/$FILE" \
  "https://huggingface.co/${REPO}/resolve/main/${FILE}"
