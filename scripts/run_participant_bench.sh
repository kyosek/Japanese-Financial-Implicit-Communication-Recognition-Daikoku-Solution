#!/usr/bin/env bash
# Zero-shot-only solve over the unlabeled JF-ICR_test_participant.parquet set
# (no `answer` column, so there's nothing for evaluate.py to score -- this
# just produces predictions for submission). Otherwise mirrors run_bench.sh:
# stops any running server, starts the requested model, waits for health,
# solves, stops the server.
#
# Usage:
#   scripts/run_participant_bench.sh <model-filename-in-models-dir> <alias> [extra llama-server args...]
#   scripts/run_participant_bench.sh gemma-4-12b-it-Q8_0.gguf gemma4 --reasoning off
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FILE="${1:?usage: run_participant_bench.sh <model-file> <alias> [extra llama-server args...]}"
ALIAS="${2:?usage: run_participant_bench.sh <model-file> <alias> [extra llama-server args...]}"
shift 2

PY="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs"
mkdir -p "$OUT"

echo "[run_participant_bench $ALIAS] stopping any existing server"
"$ROOT/scripts/stop_server.sh" || true
sleep 2

echo "[run_participant_bench $ALIAS] starting server for $MODEL_FILE extra_args=$*"
"$ROOT/scripts/serve.sh" "$MODEL_FILE" "$@" > "$OUT/serve_${ALIAS}_participant.log" 2>&1 &
SERVER_PID=$!

echo "[run_participant_bench $ALIAS] waiting for server health"
for i in $(seq 1 180); do
  if curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "[run_participant_bench $ALIAS] server ready (~$((i * 5))s)"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run_participant_bench $ALIAS] server process died, see $OUT/serve_${ALIAS}_participant.log" >&2
    exit 1
  fi
  sleep 5
done

echo "[run_participant_bench $ALIAS] zero-shot solve on test_participant set"
"$PY" "$ROOT/bench/solve.py" --model "$ALIAS" \
  --data "$ROOT/JF-ICR_test_participant.parquet" \
  --out "$OUT/predictions_${ALIAS}_test_participant_zeroshot.jsonl"

echo "[run_participant_bench $ALIAS] stopping server"
"$ROOT/scripts/stop_server.sh" || true

echo "[run_participant_bench $ALIAS] done"
