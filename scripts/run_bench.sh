#!/usr/bin/env bash
# Runs the full solve+evaluate pipeline (zero-shot, then few-shot) for one
# model, unattended: stops any running server, starts the requested model,
# waits for it to become healthy, solves, evaluates, then stops the server.
#
# Meant to be chained across models and run in the background (e.g. under
# `nohup caffeinate -i ... &`) since a full run can take hours.
#
# Usage:
#   scripts/run_bench.sh <model-filename-in-models-dir> <alias> [extra llama-server args...]
#   scripts/run_bench.sh gemma-4-12b-it-Q8_0.gguf gemma4 --reasoning off
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FILE="${1:?usage: run_bench.sh <model-file> <alias> [extra llama-server args...]}"
ALIAS="${2:?usage: run_bench.sh <model-file> <alias> [extra llama-server args...]}"
shift 2

PY="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs"
mkdir -p "$OUT"

echo "[run_bench $ALIAS] stopping any existing server"
"$ROOT/scripts/stop_server.sh" || true
sleep 2

echo "[run_bench $ALIAS] starting server for $MODEL_FILE extra_args=$*"
"$ROOT/scripts/serve.sh" "$MODEL_FILE" "$@" > "$OUT/serve_${ALIAS}.log" 2>&1 &
SERVER_PID=$!

echo "[run_bench $ALIAS] waiting for server health"
for i in $(seq 1 180); do
  if curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "[run_bench $ALIAS] server ready (~$((i * 5))s)"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run_bench $ALIAS] server process died, see $OUT/serve_${ALIAS}.log" >&2
    exit 1
  fi
  sleep 5
done

echo "[run_bench $ALIAS] zero-shot solve"
"$PY" "$ROOT/bench/solve.py" --model "$ALIAS" \
  --out "$OUT/predictions_${ALIAS}_zeroshot.jsonl"
"$PY" "$ROOT/bench/evaluate.py" \
  --predictions "$OUT/predictions_${ALIAS}_zeroshot.jsonl" \
  --report "$OUT/report_${ALIAS}_zeroshot.json"

echo "[run_bench $ALIAS] few-shot solve"
"$PY" "$ROOT/bench/solve.py" --model "$ALIAS" --few-shot-k 1 \
  --out "$OUT/predictions_${ALIAS}_fewshot.jsonl"
"$PY" "$ROOT/bench/evaluate.py" \
  --predictions "$OUT/predictions_${ALIAS}_fewshot.jsonl" \
  --report "$OUT/report_${ALIAS}_fewshot.json"

echo "[run_bench $ALIAS] stopping server"
"$ROOT/scripts/stop_server.sh" || true

echo "[run_bench $ALIAS] done"
