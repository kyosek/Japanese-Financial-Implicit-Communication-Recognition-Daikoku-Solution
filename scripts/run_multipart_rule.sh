#!/usr/bin/env bash
# A/B/C for solve.py's MULTIPART_RULE (the bundled-question aggregation rule),
# over both the public set and the labelled participant test set.
#
#   off     never state the rule -- the pre-rule control
#   always  state it in every prompt
#   gated   state it only for questions bench/multipart.py flags as bundling
#           several asks (76/253 public, 0/50 test)
#
# All arms run back-to-back against the same server process, so the only
# difference between them is the rule block. (The stored predictions_qwen36_*
# baselines predate the last edit to solve.py, so they can't serve as the
# control.)
#
# Note: the test set contains no multi-part questions, so "gated" is a no-op
# there and must reproduce "off" exactly -- that arm is a correctness check on
# the gating, not evidence about the rule.
#
# Meant to be run in the background, e.g.
#   nohup caffeinate -i scripts/run_multipart_rule.sh > outputs/multipart_rule.log 2>&1
#
# Usage:
#   scripts/run_multipart_rule.sh [model-file] [alias] [extra llama-server args...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FILE="${1:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
ALIAS="${2:-qwen36}"
if [ $# -gt 2 ]; then shift 2; else shift $#; fi
EXTRA_ARGS=("$@")
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
  EXTRA_ARGS=(--reasoning off)
fi

PY="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs"
mkdir -p "$OUT"

echo "[multipart $ALIAS] stopping any existing server"
"$ROOT/scripts/stop_server.sh" || true
sleep 2

echo "[multipart $ALIAS] starting server for $MODEL_FILE extra_args=${EXTRA_ARGS[*]}"
"$ROOT/scripts/serve.sh" "$MODEL_FILE" "${EXTRA_ARGS[@]}" > "$OUT/serve_multipart_${ALIAS}.log" 2>&1 &
SERVER_PID=$!

echo "[multipart $ALIAS] waiting for server health"
for i in $(seq 1 180); do
  if curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "[multipart $ALIAS] server ready (~$((i * 5))s)"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[multipart $ALIAS] server process died, see $OUT/serve_multipart_${ALIAS}.log" >&2
    exit 1
  fi
  sleep 5
done

# solve+evaluate one dataset under one rule mode.
#   $1 dataset path   $2 set tag   $3 rule mode
run_arm() {
  local data="$1" set_tag="$2" mode="$3"
  local stem="${ALIAS}_${set_tag}_zeroshot_rule-${mode}"
  echo "[multipart $ALIAS] $set_tag / rule=$mode"
  "$PY" "$ROOT/bench/solve.py" --model "$ALIAS" \
    --data "$data" --multipart-rule "$mode" \
    --out "$OUT/predictions_${stem}.jsonl"
  "$PY" "$ROOT/bench/evaluate.py" \
    --predictions "$OUT/predictions_${stem}.jsonl" \
    --report "$OUT/report_${stem}.json"
}

PUBLIC="$ROOT/JF-ICR_public_set.parquet"
TEST="$ROOT/JF-ICR_test_participant_labelled.parquet"

for mode in off always gated; do
  run_arm "$PUBLIC" public "$mode"
done
for mode in off always gated; do
  run_arm "$TEST" test "$mode"
done

echo "[multipart $ALIAS] stopping server"
"$ROOT/scripts/stop_server.sh" || true

echo
echo "================ MULTIPART RULE A/B/C ================"
# compare.py splits each arg on its FIRST "=", so labels must not contain one.
"$PY" "$ROOT/bench/compare.py" \
  "public  rule:off"="$OUT/report_${ALIAS}_public_zeroshot_rule-off.json" \
  "public  rule:always"="$OUT/report_${ALIAS}_public_zeroshot_rule-always.json" \
  "public  rule:gated"="$OUT/report_${ALIAS}_public_zeroshot_rule-gated.json" \
  "test    rule:off"="$OUT/report_${ALIAS}_test_zeroshot_rule-off.json" \
  "test    rule:always"="$OUT/report_${ALIAS}_test_zeroshot_rule-always.json" \
  "test    rule:gated"="$OUT/report_${ALIAS}_test_zeroshot_rule-gated.json"

echo "[multipart $ALIAS] done"
