#!/usr/bin/env bash
# Downloads a native arm64 (Metal-accelerated) llama.cpp release into runtime/.
# Needed because Homebrew on this machine is the x86_64/Rosetta build, which
# has no Metal support and is much slower on Apple Silicon.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${LLAMA_CPP_TAG:-b10244}"
ASSET="llama-${TAG}-bin-macos-arm64.tar.gz"

mkdir -p "$ROOT/runtime"
curl -L -o "$ROOT/runtime/$ASSET" \
  "https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/${ASSET}"
mkdir -p "$ROOT/runtime/llama.cpp"
tar -xzf "$ROOT/runtime/$ASSET" -C "$ROOT/runtime/llama.cpp"
rm "$ROOT/runtime/$ASSET"

echo "Extracted to $ROOT/runtime/llama.cpp/llama-${TAG}"
