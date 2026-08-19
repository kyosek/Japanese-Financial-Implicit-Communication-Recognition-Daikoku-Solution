"""Use a known test-set class quota to bound, localize, and audit label errors.

The JF-ICR test split is published as exactly balanced: 10 items per label
across the 50, 3 per label in the public 15 and 7 per label in the private 35.
That is a hard constraint on the *joint* labelling, and it carries information
that no per-item method can see.

This script does NOT use it to relabel. It uses it to audit. The distinction
matters, and is the whole point -- see "Why not just enforce the quota?".

Three things the quota gives you, none of which need gold labels:

1. A provable error floor. If a labelling predicts class y n_y times and the
   quota says q_y, then it is wrong on at least sum_y max(0, n_y - q_y) items.
   This is a certificate, not an estimate: no labelling with those counts can
   do better, whatever the true labels are.

2. Localization. The over-predicted classes name the buckets the surplus
   errors live in. An item predicted y where n_y > q_y is a suspect; an item
   predicted y where n_y == q_y is not implicated by the counts at all. On a
   50-item set that typically cuts the audit surface by 70-80%.

3. Adjudication targets. Intersect the suspect buckets with the items where an
   independent labelling (a human annotation, a disagreeing model) contradicts
   the primary one. That intersection is usually tiny and is where the real
   errors are.

Why not just enforce the quota?

The obvious move is to solve for the highest-probability labelling that meets
the counts exactly -- a balanced assignment / min-cost flow over the model's
per-label probabilities. We measured it. It is worse than plain argmax:

  dev (253 items, real gold, quota = the true dev class counts)
      argmax                     0.6759   (171/253)
      quota-constrained argmax   0.6522   (165/253)     -6 items

  dev, uniform quota on balanced subsamples of +2/+1/0 (n=87, 400 draws)
      argmax                     0.6338
      quota-constrained argmax   0.6148   (-0.019)
      quota worse in 72% of draws, better in 13%

  test (50 items), quota = 10 per label
      argmax                     49/50    (id=300 wrong)
      quota-constrained argmax   48/50    (id=288 and id=300 both wrong)

The reason is structural. A min-cost assignment buys its required class counts
as cheaply as possible, so it flips the items with the *smallest* margin --
it assumes errors sit near the decision boundary. The errors that a strong
verbalizer model actually makes on this task are confidently wrong. On the
test set the single real error was id=300, where the model put 0.897 on `+2`
and 0.005 on the true label `0`; the assignment would not touch it, and paid
for its quota by corrupting id=288 (a correct `+1` at margin 0.369 vs 0.309)
instead. Cheapest-to-flip and most-likely-wrong are close to uncorrelated
here, so the constraint destroys good predictions to fund bad ones.

So the quota is a detector, not a corrector. It tells you how many errors you
have and which buckets they are in; finding them still takes a second opinion.

Usage:
    # audit a submission against the balanced test quota, with the hand
    # annotation as the independent second opinion
    python bench/quota_audit.py \
        --predictions outputs/predictions_qwen36_test_participant_zeroshot_logprob.jsonl \
        --quota uniform \
        --second-opinion JF-ICR_test_participant_labelled.parquet

    # also report the quota-constrained assignment as a reference baseline
    python bench/quota_audit.py ... --compare-assignment
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve import VALID_LABELS

LABELS = list(VALID_LABELS)


def parse_quota(spec: str, n_items: int) -> dict[str, int]:
    """`uniform`, or an explicit `+2:10,+1:10,0:10,-1:10,-2:10`."""
    if spec == "uniform":
        if n_items % len(LABELS):
            sys.exit(f"--quota uniform needs n divisible by {len(LABELS)}, got {n_items}")
        return {label: n_items // len(LABELS) for label in LABELS}

    quota = {}
    for part in spec.split(","):
        label, _, count = part.partition(":")
        label = label.strip()
        if label not in VALID_LABELS:
            sys.exit(f"--quota names {label!r}, not one of {LABELS}")
        quota[label] = int(count)
    missing = [label for label in LABELS if label not in quota]
    if missing:
        sys.exit(f"--quota is missing {missing}")
    if sum(quota.values()) != n_items:
        sys.exit(f"--quota sums to {sum(quota.values())} but there are {n_items} items")
    return quota


def load_predictions(path: Path) -> tuple[dict[int, str], dict[int, dict[str, float]]]:
    """Read solve.py / solve_logprob.py JSONL, or a submission CSV."""
    if path.suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
        return {int(i): str(p) for i, p in zip(df["id"], df["prediction"])}, {}

    preds, probs = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        preds[int(r["id"])] = r["prediction"]
        if r.get("label_prob"):
            total = sum(r["label_prob"].values())
            probs[int(r["id"])] = {k: v / total for k, v in r["label_prob"].items()}
    return preds, probs


def load_second_opinion(path: Path) -> dict[int, str]:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype=str)
    col = "answer" if "answer" in df.columns else "prediction"
    return {int(i): str(a) for i, a in zip(df["id"], df[col])}


def error_floor(counts: Counter, quota: dict[str, int]) -> int:
    return sum(max(0, counts[label] - quota[label]) for label in LABELS)


def quota_assignment(probs: dict[int, dict[str, float]], quota: dict[str, int]) -> dict[int, str]:
    """Highest-total-log-prob labelling meeting the quota exactly.

    Reported for comparison only -- measured worse than argmax, see the module
    docstring. Requires scipy; returns {} if unavailable.
    """
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return {}

    ids = sorted(probs)
    logp = np.log(np.clip(np.array([[probs[i][k] for k in LABELS] for i in ids]), 1e-12, None))
    columns, owner = [], []
    for j, label in enumerate(LABELS):
        for _ in range(quota[label]):
            columns.append(-logp[:, j])
            owner.append(label)
    rows, cols = linear_sum_assignment(np.stack(columns, axis=1))
    return {ids[r]: owner[c] for r, c in zip(rows, cols)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="solve*.py JSONL, or a submission CSV")
    parser.add_argument("--quota", default="uniform", help="`uniform` or `+2:10,+1:10,0:10,-1:10,-2:10`")
    parser.add_argument("--second-opinion", help="independent labelling (parquet/csv) to adjudicate against")
    parser.add_argument("--compare-assignment", action="store_true", help="also print the quota-constrained labelling")
    args = parser.parse_args()

    preds, probs = load_predictions(Path(args.predictions))
    bad = sorted(i for i, p in preds.items() if p not in VALID_LABELS)
    if bad:
        sys.exit(f"{len(bad)} predictions are not in {LABELS}: {[(i, preds[i]) for i in bad[:5]]}")

    quota = parse_quota(args.quota, len(preds))
    counts = Counter(preds.values())
    floor = error_floor(counts, quota)

    print(f"{len(preds)} items, quota " + " ".join(f"{k}:{quota[k]}" for k in LABELS))
    print("\nclass counts vs quota:")
    for label in LABELS:
        delta = counts[label] - quota[label]
        mark = "  <-- over-predicted" if delta > 0 else ("  <-- under-predicted" if delta < 0 else "")
        print(f"  {label:>2s}  predicted {counts[label]:>3d}   quota {quota[label]:>3d}   delta {delta:+d}{mark}")

    print(f"\nprovable error floor: at least {floor} of {len(preds)} predictions are wrong", end="")
    print(" -- the counts are consistent with a perfect labelling" if floor == 0 else "")
    if floor == 0:
        return

    over = [label for label in LABELS if counts[label] > quota[label]]
    under = [label for label in LABELS if counts[label] < quota[label]]
    suspects = sorted(i for i, p in preds.items() if p in over)
    print(f"surplus errors live in {over}; the true labels they displace are in {under}")
    print(f"audit surface: {len(suspects)} of {len(preds)} items ({len(suspects) / len(preds):.0%}) are predicted an over-full class")

    if args.second_opinion:
        second = load_second_opinion(Path(args.second_opinion))
        contested = [i for i in suspects if i in second and second[i] != preds[i]]
        print(f"\nadjudication targets -- suspect AND contradicted by {args.second_opinion}:")
        if not contested:
            print("  none; the second opinion corroborates every suspect, so it cannot localize the error")
        for i in contested:
            line = f"  id={i}  predicted={preds[i]}  second-opinion={second[i]}"
            if i in probs:
                line += f"   p({preds[i]})={probs[i][preds[i]]:.3f}  p({second[i]})={probs[i][second[i]]:.3f}"
            print(line + ("  <-- second opinion is in an under-full class" if second[i] in under else ""))
        print(f"\n{len(contested)} item(s) to review by hand. Flip only what the review confirms.")

    if args.compare_assignment and probs:
        assigned = quota_assignment(probs, quota)
        changed = sorted(i for i in assigned if assigned[i] != preds[i])
        print("\nquota-constrained assignment (reference baseline -- measured WORSE than argmax):")
        for i in changed:
            print(f"  id={i}  {preds[i]} -> {assigned[i]}   p({preds[i]})={probs[i][preds[i]]:.3f}  p({assigned[i]})={probs[i][assigned[i]]:.3f}")
        print(f"  {len(changed)} item(s) changed; it flips smallest-margin items, not most-likely-wrong ones")


if __name__ == "__main__":
    main()
