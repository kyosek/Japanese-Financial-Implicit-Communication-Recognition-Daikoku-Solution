#!/usr/bin/env bash
# Runs bench/decompose_experiment.py unattended against a local llama.cpp
# server: stops any running server, starts the requested model, waits for
# health, runs the experiment, then stops the server.
#
# Meant to be run in the background, e.g.
#   nohup caffeinate -i scripts/run_decompose_experiment.sh > outputs/decompose.log 2>&1
#
# Usage:
#   scripts/run_decompose_experiment.sh [model-file] [alias] [extra llama-server args...]
#   scripts/run_decompose_experiment.sh gemma-4-12b-it-Q8_0.gguf gemma4 --reasoning off
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FILE="${1:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
ALIAS="${2:-qwen36}"
if [ $# -gt 2 ]; then shift 2; else shift $#; fi
EXTRA_ARGS=("$@")
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
  # Both local hybrids burn the whole token budget on an unfinished thinking
  # preamble without this -- see README.
  EXTRA_ARGS=(--reasoning off)
fi

PY="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs"
mkdir -p "$OUT"

echo "[decompose $ALIAS] stopping any existing server"
"$ROOT/scripts/stop_server.sh" || true
sleep 2

echo "[decompose $ALIAS] starting server for $MODEL_FILE extra_args=${EXTRA_ARGS[*]}"
"$ROOT/scripts/serve.sh" "$MODEL_FILE" "${EXTRA_ARGS[@]}" > "$OUT/serve_decompose_${ALIAS}.log" 2>&1 &
SERVER_PID=$!

echo "[decompose $ALIAS] waiting for server health"
for i in $(seq 1 180); do
  if curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "[decompose $ALIAS] server ready (~$((i * 5))s)"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[decompose $ALIAS] server process died, see $OUT/serve_decompose_${ALIAS}.log" >&2
    exit 1
  fi
  sleep 5
done

echo "[decompose $ALIAS] running decomposition experiment"
"$PY" "$ROOT/bench/decompose_experiment.py" \
  --data "$ROOT/JF-ICR_public_set.parquet" \
  --model "$ALIAS" \
  --n-multi 40 --n-single 20 \
  --out "$OUT/decompose_experiment_${ALIAS}.jsonl" \
  --report "$OUT/report_decompose_experiment_${ALIAS}.json"

echo "[decompose $ALIAS] stopping server"
"$ROOT/scripts/stop_server.sh" || true

echo "[decompose $ALIAS] done"
