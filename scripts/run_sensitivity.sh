#!/usr/bin/env bash
# Runs both E4 prompt-sensitivity probes (bench/prompt_sensitivity.py) for
# one model, unattended: stops any running server, starts the requested
# model, waits for it to become healthy, runs --probe rules then
# --probe label-order, then stops the server.
#
# Meant to be chained across models and run in the background (e.g. under
# `nohup caffeinate -i ... &`) since the rules probe alone is N paraphrases
# x M seeds full passes over the eval set.
#
# Usage:
#   scripts/run_sensitivity.sh <model-filename-in-models-dir> <alias> [extra llama-server args...]
#   scripts/run_sensitivity.sh gemma-4-12b-it-Q8_0.gguf gemma4 --reasoning off
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FILE="${1:?usage: run_sensitivity.sh <model-file> <alias> [extra llama-server args...]}"
ALIAS="${2:?usage: run_sensitivity.sh <model-file> <alias> [extra llama-server args...]}"
shift 2

PY="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs"
mkdir -p "$OUT"

echo "[run_sensitivity $ALIAS] stopping any existing server"
"$ROOT/scripts/stop_server.sh" || true
sleep 2

echo "[run_sensitivity $ALIAS] starting server for $MODEL_FILE extra_args=$*"
"$ROOT/scripts/serve.sh" "$MODEL_FILE" "$@" > "$OUT/serve_${ALIAS}_sensitivity.log" 2>&1 &
SERVER_PID=$!

echo "[run_sensitivity $ALIAS] waiting for server health"
for i in $(seq 1 180); do
  if curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "[run_sensitivity $ALIAS] server ready (~$((i * 5))s)"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run_sensitivity $ALIAS] server process died, see $OUT/serve_${ALIAS}_sensitivity.log" >&2
    exit 1
  fi
  sleep 5
done

echo "[run_sensitivity $ALIAS] rule-paraphrase probe (5 variants x 3 seeds)"
"$PY" "$ROOT/bench/prompt_sensitivity.py" --probe rules --model "$ALIAS" \
  --out-prefix "$OUT/sensitivity_${ALIAS}"

echo "[run_sensitivity $ALIAS] label-order probe (5 orderings)"
"$PY" "$ROOT/bench/prompt_sensitivity.py" --probe label-order --model "$ALIAS" \
  --out-prefix "$OUT/sensitivity_${ALIAS}"

echo "[run_sensitivity $ALIAS] stopping server"
"$ROOT/scripts/stop_server.sh" || true

echo "[run_sensitivity $ALIAS] done"
