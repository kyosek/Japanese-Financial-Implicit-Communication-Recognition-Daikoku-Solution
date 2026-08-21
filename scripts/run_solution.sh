#!/usr/bin/env bash
# The JF-ICR submission pipeline, end to end, on the 50-item test set.
#
# Five stages, no gold labels and no human judgement anywhere:
#
#   1. solve_logprob.py   score the five label tokens per item -> P(y|x), argmax
#   2. quota_audit.py     compare class counts to the published 10-per-label
#                         quota. Yields a PROVABLE error count and names the
#                         over-full buckets those errors live in. Pure
#                         arithmetic, no model call.
#   3. adjudicate.py      within each flagged bucket, ask the model the
#                         comparative question the quota licenses -- "these N
#                         were labelled +2, exactly K are really 0, which?" --
#                         voted over rotated candidate orderings
#   4. quota_audit.py     re-audit. The corrected counts must now meet the
#                         quota exactly, i.e. a provable error floor of 0
#   5. make_submission.py write and validate the CSV
#
# Stage 3 exists because stage 2 cannot finish the job: the quota proves how
# many predictions are wrong and where, but not which. Majority voting does not
# close that gap -- on the one genuinely wrong test item the 16 available runs
# split 12 `+2` / 3 `0` / 1 `-1`, so the ensemble majority is wrong and matches
# plain argmax on all 50 items. What works is either a comparative question
# (stage 3) or the vote *spread* rather than the majority (stage 3b).
#
# Set RUNS to a glob of other prediction files to enable stage 3b, which
# localises the same error from run disagreement alone and cross-checks the
# adjudicator. Two independent detectors agreeing is worth more than either.
#
# Scope limit, stated because it bounds what stage 4 means: the quota can only
# see errors that change the class counts. A compensating pair (one item
# labelled A that is really B, another labelled B that is really A) leaves the
# counts untouched and is invisible. "Error floor 0" means the counts are
# CONSISTENT with a perfect labelling, not that the labelling is perfect.
#
# Usage:
#   scripts/run_solution.sh                                  # full run, default model
#   scripts/run_solution.sh Qwen3.8-27B-UD-Q4_K_XL.gguf qwen38
#   REUSE_PREDICTIONS=outputs/predictions_qwen36_test_participant_zeroshot_logprob.jsonl \
#     scripts/run_solution.sh                                # skip stage 1
#
# REUSE_PREDICTIONS skips inference and runs stages 2-5 over an existing
# predictions file. Worth knowing why it exists: llama.cpp is not bit-exact at
# temperature 0, so a fresh stage 1 can differ from a previous one by an item
# or two. Reusing a stored predictions file makes stages 2-5 reproducible.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FILE="${1:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
ALIAS="${2:-qwen36}"
DATA="${DATA:-$ROOT/JF-ICR_test_participant.parquet}"
QUOTA="${QUOTA:-uniform}"
REPEATS="${REPEATS:-5}"
# Deliberately not submission.csv: that file is committed and holds the raw
# argmax labelling, which stage 2 needs as a reference point. Override with
# SUBMISSION= to write elsewhere.
SUBMISSION="${SUBMISSION:-$ROOT/submission_solution.csv}"

PY="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs"
mkdir -p "$OUT"

RAW="$OUT/predictions_${ALIAS}_solution_raw.jsonl"
FINAL="$OUT/predictions_${ALIAS}_solution_final.jsonl"
SERVER_PID=""

start_server() {
  echo "[solution] stopping any existing server"
  "$ROOT/scripts/stop_server.sh" || true
  sleep 2
  echo "[solution] starting $MODEL_FILE (reasoning off, 16k context)"
  "$ROOT/scripts/serve.sh" "$MODEL_FILE" -c 16384 --reasoning off > "$OUT/serve_${ALIAS}_solution.log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 180); do
    if curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"'; then
      echo "[solution] server ready (~$((i * 5))s)"
      return
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[solution] server died, see $OUT/serve_${ALIAS}_solution.log" >&2
      exit 1
    fi
    sleep 5
  done
  echo "[solution] server did not become healthy in 15 minutes" >&2
  exit 1
}

start_server

if [ -n "${REUSE_PREDICTIONS:-}" ]; then
  echo "[solution] stage 1 SKIPPED -- reusing $REUSE_PREDICTIONS"
  RAW="$REUSE_PREDICTIONS"
else
  echo "[solution] stage 1: log-prob solve over $(basename "$DATA")"
  "$PY" "$ROOT/bench/solve_logprob.py" --model "$ALIAS" --data "$DATA" --out "$RAW"
fi

echo
echo "[solution] stage 2: quota audit of the raw predictions"
"$PY" "$ROOT/bench/quota_audit.py" --predictions "$RAW" --quota "$QUOTA" \
  | tee "$OUT/quota_audit_${ALIAS}_before.log"

echo
echo "[solution] stage 3: adjudicate each flagged bucket ($REPEATS orderings)"
"$PY" "$ROOT/bench/adjudicate.py" \
  --predictions "$RAW" --data "$DATA" --quota "$QUOTA" \
  --repeats "$REPEATS" --max-tokens 64 \
  --out "$OUT/adjudicate_${ALIAS}_solution.json" \
  --write-predictions "$FINAL" \
  | tee "$OUT/adjudicate_${ALIAS}_solution.log"

if [ -n "${RUNS:-}" ]; then
  echo
  echo "[solution] stage 3b: cross-check the flips against run disagreement"
  "$PY" "$ROOT/bench/consensus.py" --predictions "$RAW" --runs "$RUNS" --quota "$QUOTA" \
    --out "$OUT/consensus_${ALIAS}_solution.json" | tee "$OUT/consensus_${ALIAS}_solution.log"
  if ! "$PY" - "$OUT/adjudicate_${ALIAS}_solution.json" "$OUT/consensus_${ALIAS}_solution.json" <<'EOF'; then
import json, sys
key = lambda p: sorted((f["id"], f["to"]) for f in json.load(open(p))["flips"])
a, c = key(sys.argv[1]), key(sys.argv[2])
if a == c:
    print(f"[solution] detectors agree on {len(a)} flip(s)")
else:
    print(f"[solution] DETECTORS DISAGREE -- adjudicator {a}, consensus {c}", file=sys.stderr)
    sys.exit(1)
EOF
    echo "[solution] the two detectors picked different items; review before submitting" >&2
    "$ROOT/scripts/stop_server.sh" || true
    exit 1
  fi
fi

echo
echo "[solution] stage 4: re-audit -- the floor must now be 0"
"$PY" "$ROOT/bench/quota_audit.py" --predictions "$FINAL" --quota "$QUOTA" \
  | tee "$OUT/quota_audit_${ALIAS}_after.log"
if ! grep -q "at least 0 of" "$OUT/quota_audit_${ALIAS}_after.log"; then
  echo "[solution] adjudication did not clear the quota -- refusing to write a submission" >&2
  "$ROOT/scripts/stop_server.sh" || true
  exit 1
fi

echo
echo "[solution] stage 5: write and validate the submission"
"$PY" "$ROOT/bench/make_submission.py" --predictions "$FINAL" --data "$DATA" --out "$SUBMISSION"

echo "[solution] stopping server"
"$ROOT/scripts/stop_server.sh" || true
echo "[solution] done -- $SUBMISSION"
