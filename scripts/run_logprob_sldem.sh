#!/usr/bin/env bash
# Runs the log-prob verbalizer pipeline (solve_logprob -> calibrate sld-em ->
# evaluate) for one model, unattended: stops any running server, starts the
# requested model, waits for it to become healthy, solves, calibrates,
# evaluates, then stops the server.
#
# Meant to be chained in the background (e.g. under `nohup caffeinate -i ...
# &`) since a full run can take a while.
#
# Usage:
#   scripts/run_logprob_sldem.sh <model-filename-in-models-dir> <alias> [extra llama-server args...]
#   scripts/run_logprob_sldem.sh gemma-4-12b-it-Q8_0.gguf gemma4 --reasoning off
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FILE="${1:?usage: run_logprob_sldem.sh <model-file> <alias> [extra llama-server args...]}"
ALIAS="${2:?usage: run_logprob_sldem.sh <model-file> <alias> [extra llama-server args...]}"
shift 2

PY="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs"
mkdir -p "$OUT"

echo "[run_logprob_sldem $ALIAS] stopping any existing server"
"$ROOT/scripts/stop_server.sh" || true
sleep 2

echo "[run_logprob_sldem $ALIAS] starting server for $MODEL_FILE extra_args=$*"
"$ROOT/scripts/serve.sh" "$MODEL_FILE" "$@" > "$OUT/serve_${ALIAS}_logprob.log" 2>&1 &
SERVER_PID=$!

echo "[run_logprob_sldem $ALIAS] waiting for server health"
for i in $(seq 1 180); do
  if curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "[run_logprob_sldem $ALIAS] server ready (~$((i * 5))s)"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run_logprob_sldem $ALIAS] server process died, see $OUT/serve_${ALIAS}_logprob.log" >&2
    exit 1
  fi
  sleep 5
done

echo "[run_logprob_sldem $ALIAS] zero-shot logprob solve"
"$PY" "$ROOT/bench/solve_logprob.py" --model "$ALIAS" \
  --out "$OUT/predictions_${ALIAS}_zeroshot_logprob.jsonl"

echo "[run_logprob_sldem $ALIAS] sld-em calibration"
"$PY" "$ROOT/bench/calibrate_logprob.py" --method sld-em \
  --predictions "$OUT/predictions_${ALIAS}_zeroshot_logprob.jsonl" \
  --out "$OUT/predictions_${ALIAS}_zeroshot_logprob_sldem.jsonl"

echo "[run_logprob_sldem $ALIAS] evaluate calibrated predictions"
"$PY" "$ROOT/bench/evaluate_logprob.py" \
  --predictions "$OUT/predictions_${ALIAS}_zeroshot_logprob_sldem.jsonl" \
  --report "$OUT/report_${ALIAS}_zeroshot_logprob_sldem.json"

echo "[run_logprob_sldem $ALIAS] stopping server"
"$ROOT/scripts/stop_server.sh" || true

echo "[run_logprob_sldem $ALIAS] done"
