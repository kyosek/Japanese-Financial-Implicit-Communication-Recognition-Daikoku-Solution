#!/usr/bin/env bash
# Chains run_bench.sh across multiple models, sequentially (only one model
# fits in 48GB RAM at a time). Meant to be run under `nohup caffeinate -i ... &`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/scripts/run_bench.sh" shisa-v2-qwen2.5-32b.Q4_K_M.gguf shisa
"$ROOT/scripts/run_bench.sh" Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf qwen3
"$ROOT/scripts/run_bench.sh" gpt-oss-20b-F16.gguf gptoss --reasoning off

echo "[run_all_bench] all done"
