#!/usr/bin/env bash
# Chains run_participant_bench.sh across qwen3.6 and gemma4 (only one model
# fits in 48GB RAM at a time). Meant to be run under `nohup caffeinate -i ... &`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/scripts/run_participant_bench.sh" Qwen3.6-35B-A3B-UD-Q4_K_M.gguf qwen36 --reasoning off
"$ROOT/scripts/run_participant_bench.sh" gemma-4-12b-it-Q8_0.gguf gemma4 --reasoning off

echo "[run_all_participant_bench] all done"
