#!/usr/bin/env bash
# Zero-shot solve + evaluate for one model over BOTH evaluation sets (the
# 253-row public set and the 50-row labelled participant test set), against a
# single server process. No few-shot arm -- see run_bench.sh for that.
#
# Uses solve.py's current defaults (annotation rules on, --multipart-rule
# always), so results line up with the *_rule-always arms of
# run_multipart_rule.sh rather than the older predictions_*_zeroshot baselines,
# which predate the last solve.py edit.
#
# Meant to be run in the background, e.g.
#   nohup caffeinate -i scripts/run_zeroshot.sh Qwen3.8-27B-UD-Q4_K_XL.gguf qwen38 > outputs/zeroshot_qwen38.log 2>&1
#
# Usage:
#   scripts/run_zeroshot.sh <model-filename-in-models-dir> <alias> [extra llama-server args...]
# Extra args default to `--reasoning off` (hybrid-reasoning models must have
# thinking disabled -- solve.py caps generation at 32 tokens).
#
# Env overrides, for running a second arm without clobbering the first:
#   TAG=_thinking   suffix on every output filename (default: none)
#   MAX_TOKENS=2048 solve.py --max-tokens   (default 32; a thinking run needs
#                   room for the thought AND the answer, or content comes back
#                   empty with finish_reason "length")
#   TIMEOUT=600     solve.py --timeout in seconds (default 120)
#   SETS="test"     which eval sets to run (default: "public test")
#   BASELINE_PUBLIC=... / BASELINE_TEST=...
#                   reports to compare against in the closing table
#                   (default: qwen36's rule-always arms)
#
# Reasoning arm, e.g.
#   TAG=_thinking MAX_TOKENS=2048 TIMEOUT=600 \
#     scripts/run_zeroshot.sh Qwen3.6-35B-A3B-UD-Q4_K_M.gguf qwen36 \
#     --reasoning on --reasoning-format deepseek
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FILE="${1:?usage: run_zeroshot.sh <model-file> <alias> [extra llama-server args...]}"
ALIAS="${2:?usage: run_zeroshot.sh <model-file> <alias> [extra llama-server args...]}"
shift 2
EXTRA_ARGS=("$@")
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
  EXTRA_ARGS=(--reasoning off)
fi

PY="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs"
mkdir -p "$OUT"

TAG="${TAG:-}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TIMEOUT="${TIMEOUT:-120}"
SETS="${SETS:-public test}"
RUN="${ALIAS}${TAG}"

echo "[zeroshot $RUN] stopping any existing server"
"$ROOT/scripts/stop_server.sh" || true
sleep 2

echo "[zeroshot $RUN] starting server for $MODEL_FILE extra_args=${EXTRA_ARGS[*]}"
"$ROOT/scripts/serve.sh" "$MODEL_FILE" "${EXTRA_ARGS[@]}" > "$OUT/serve_zeroshot_${RUN}.log" 2>&1 &
SERVER_PID=$!

echo "[zeroshot $RUN] waiting for server health"
for i in $(seq 1 180); do
  if curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "[zeroshot $RUN] server ready (~$((i * 5))s)"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[zeroshot $RUN] server process died, see $OUT/serve_zeroshot_${RUN}.log" >&2
    exit 1
  fi
  sleep 5
done

# solve+evaluate one dataset.  $1 dataset path   $2 set tag
run_set() {
  local data="$1" set_tag="$2"
  local stem="${ALIAS}_${set_tag}_zeroshot${TAG}"
  echo "[zeroshot $RUN] $set_tag (max_tokens=$MAX_TOKENS timeout=$TIMEOUT)"
  "$PY" "$ROOT/bench/solve.py" --model "$ALIAS" \
    --data "$data" \
    --max-tokens "$MAX_TOKENS" --timeout "$TIMEOUT" \
    --out "$OUT/predictions_${stem}.jsonl"
  "$PY" "$ROOT/bench/evaluate.py" \
    --predictions "$OUT/predictions_${stem}.jsonl" \
    --report "$OUT/report_${stem}.json"
}

for set_tag in $SETS; do
  case "$set_tag" in
    public) run_set "$ROOT/JF-ICR_public_set.parquet" public ;;
    test)   run_set "$ROOT/JF-ICR_test_participant_labelled.parquet" test ;;
    *)      echo "[zeroshot $RUN] unknown set '$set_tag' (want: public test)" >&2; exit 1 ;;
  esac
done

echo "[zeroshot $RUN] stopping server"
"$ROOT/scripts/stop_server.sh" || true

BASELINE_PUBLIC="${BASELINE_PUBLIC:-$OUT/report_qwen36_public_zeroshot_rule-always.json}"
BASELINE_TEST="${BASELINE_TEST:-$OUT/report_qwen36_test_zeroshot_rule-always.json}"

# Only compare the sets we actually ran -- naming a missing report here would
# fail the whole run after the expensive part already succeeded.
# compare.py splits each arg on its FIRST "=", so labels must not contain one.
CMP=()
for set_tag in $SETS; do
  case "$set_tag" in
    public) CMP+=( "public  baseline=$BASELINE_PUBLIC"
                   "public  $RUN=$OUT/report_${ALIAS}_public_zeroshot${TAG}.json" ) ;;
    test)   CMP+=( "test    baseline=$BASELINE_TEST"
                   "test    $RUN=$OUT/report_${ALIAS}_test_zeroshot${TAG}.json" ) ;;
  esac
done

echo
echo "================ ZERO-SHOT: $RUN vs baseline ================"
"$PY" "$ROOT/bench/compare.py" "${CMP[@]}"

echo "[zeroshot $RUN] done"
