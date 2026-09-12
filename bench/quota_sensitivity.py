"""What the quota mechanism is worth when the published marginal is only approximate.

Stages 2 and 3 consume a published class marginal as if it were exact. A reviewer
asked the obvious question: what happens when it is not -- when the organisers say
"balanced" and the split is 11/10/10/10/9, or when the marginal is a historical
estimate rather than a design guarantee.

The answer is that Eq. 1 degrades linearly and by a knowable amount, so the
certificate survives with a correction term that needs no knowledge of the true
marginal beyond a bound on how wrong it is.

Write q for the true class counts, q~ for the published ones, n for the predicted
counts, all summing to N, and

    F(q) = sum_y max(0, n_y - q_y).

Since b -> max(0, a - b) is non-increasing and 1-Lipschitz,

    F(q~) - F(q) <= sum_y max(0, q_y - q~_y) = D,
    F(q)  - F(q~) <= sum_y max(0, q~_y - q_y) = D,

where the two sums are equal because q and q~ both total N, and their common value
is D = ||q~ - q||_1 / 2 = N * d_TV(q~/N, q/N). Hence

    |F(q~) - F(q)| <= D          and          F(q~) - D  is still a valid floor.

Stage 3 moves exactly F(q~) items, so at most D of those moves are in excess of
what the true marginal licenses. The exposure is bounded by the error in the
marginal, not by the size of the split.

This script checks both claims exhaustively rather than asserting them: it
enumerates every integer marginal at a given distance D from the truth and reports
the worst and typical degradation. No model calls -- it reads a stored stage-1
predictions file and is pure arithmetic.

Usage:
    python3 bench/quota_sensitivity.py \
        --predictions outputs/predictions_qwen36_zeroshot_logprob.jsonl \
        --max-distance 6
"""

import argparse
import itertools
import json
import statistics
from collections import Counter
from pathlib import Path

LABELS = ["+2", "+1", "0", "-1", "-2"]


def floor_at(n: Counter, quota: dict[str, int]) -> int:
    """Eq. 1: the provable number of count-changing errors."""
    return sum(max(0, n[y] - quota[y]) for y in LABELS)


def marginals_at_distance(q: dict[str, int], d: int):
    """Every non-negative integer marginal q~ with ||q~ - q||_1 / 2 == d and sum q~ == sum q.

    A perturbation is a pair of non-negative vectors (add, sub) each summing to d
    with disjoint support; disjointness is what makes d the L1/2 distance rather
    than an over-count of it.
    """
    k = len(LABELS)
    parts = [c for c in itertools.product(range(d + 1), repeat=k) if sum(c) == d]
    for add in parts:
        for sub in parts:
            if any(a and s for a, s in zip(add, sub)):
                continue  # overlapping support: the true distance is below d
            cand = {y: q[y] + a - s for y, a, s in zip(LABELS, add, sub)}
            if all(v >= 0 for v in cand.values()):
                yield cand


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--max-distance", type=int, default=6)
    ap.add_argument("--gold-key", default="gold")
    ap.add_argument("--assume-quota", default=None,
                    help="published marginal when the file carries no gold, e.g. 'uniform'")
    ap.add_argument("--out", help="write the sweep as JSON")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.predictions).read_text().splitlines() if l.strip()]
    n = Counter(r["prediction"] for r in rows)
    N = len(rows)

    if args.assume_quota == "uniform":
        q = {y: N // len(LABELS) for y in LABELS}
        true_errors = None
    else:
        q = Counter(r[args.gold_key] for r in rows)
        q = {y: q[y] for y in LABELS}
        true_errors = sum(r[args.gold_key] != r["prediction"] for r in rows)

    f_true = floor_at(n, q)
    print(f"n = {N} items")
    print(f"predicted counts  {[(y, n[y]) for y in LABELS]}")
    print(f"reference marginal {[(y, q[y]) for y in LABELS]}")
    print(f"F at the reference marginal: {f_true}"
          + (f"   (true errors {true_errors}; the floor sees {f_true} of them)" if true_errors is not None else ""))
    print()
    print(f"{'D':>3}  {'#q~':>7}  {'F(q~) range':>13}  {'worst |F(q~)-F(q)|':>19}  "
          f"{'median dev':>11}  {'corrected floor':>16}  {'valid':>6}")

    sweep = []
    for d in range(1, args.max_distance + 1):
        cands = list(marginals_at_distance(q, d))
        if not cands:
            continue
        fs = [floor_at(n, c) for c in cands]
        devs = [abs(f - f_true) for f in fs]
        worst = max(devs)
        corrected = [f - d for f in fs]
        # the theorem: F(q~) - D never exceeds the floor at the true marginal
        valid = all(c <= f_true for c in corrected)
        row = {
            "distance": d, "n_marginals": len(cands),
            "f_min": min(fs), "f_max": max(fs),
            "worst_deviation": worst, "median_deviation": statistics.median(devs),
            "bound_attained": worst == d,
            "corrected_floor_worst": max(0, min(corrected)),
            "corrected_floor_best": max(0, max(corrected)),
            "certificate_valid": valid,
        }
        sweep.append(row)
        print(f"{d:>3}  {len(cands):>7}  {min(fs):>5} - {max(fs):<5}  {worst:>19}  "
              f"{statistics.median(devs):>11.1f}  {max(0, max(corrected)):>16}  {str(valid):>6}")

    vacuous = next((r["distance"] for r in sweep if r["corrected_floor_best"] == 0), None)
    print()
    print(f"bound |F(q~) - F(q)| <= D holds in every case: "
          f"{all(r['worst_deviation'] <= r['distance'] for r in sweep)}")
    print(f"bound attained (so it cannot be tightened): "
          f"{[r['distance'] for r in sweep if r['bound_attained']]}")
    if vacuous:
        print(f"certificate becomes vacuous at D = {vacuous} (F(q~) - D <= 0 for some marginal at that distance)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n": N, "predicted": {y: n[y] for y in LABELS}, "reference": q,
             "f_reference": f_true, "true_errors": true_errors, "sweep": sweep}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
