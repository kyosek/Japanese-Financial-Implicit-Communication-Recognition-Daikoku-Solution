"""Localize a quota-flagged error by run disagreement, with no extra model call.

bench/quota_audit.py proves how many predictions are wrong and which class
buckets hold them, but not which item. bench/adjudicate.py closes that with a
forced-choice model call. This script closes it a cheaper way when several
prediction runs already exist: inside the flagged bucket, look for the item the
runs disagree about.

The distinction that makes this work is between the *majority* and the
*spread*. Majority voting fails here -- on the one genuinely wrong test item,
id=300, the 16 available runs split 12 `+2` / 3 `0` / 1 `-1`, so the ensemble's
majority label is `+2`, wrong, and identical to plain argmax on all 50 items.
Ensembling by vote buys nothing.

The spread is a different signal. Within the quota-flagged `+2` bucket, ten of
the eleven candidates are unanimous at 16/16 and id=300 is the only one that is
not. Ranking the bucket by consensus puts the real error first, alone, with a
clean margin to the next candidate. A wrong prediction that every run agrees on
is invisible here; one that any run doubts is not, and the quota has already
guaranteed that an error is present in this bucket, which is what makes a weak
disagreement signal decisive rather than merely suggestive.

Runs must be genuinely different (model, prompt, decoding) for the spread to
mean anything -- N reruns of one config at temperature 0 measure sampling
noise, not disagreement.

Usage:
    python bench/consensus.py \
        --predictions outputs/predictions_qwen36_test_participant_zeroshot_logprob.jsonl \
        --runs 'outputs/predictions_*test*.jsonl' \
        --quota uniform
"""

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quota_audit import LABELS, load_predictions, parse_quota
from solve import VALID_LABELS


def load_runs(pattern: str, n_expected: int, exclude: Path) -> dict[str, dict[int, str]]:
    """Every prediction file matching the glob that covers the full item set.

    The audited file is excluded even when the glob matches it: letting a
    labelling vote on itself inflates the consensus of exactly the items it got
    wrong, which is the opposite of what this measures.
    """
    runs = {}
    exclude = exclude.resolve()
    for path in sorted(glob.glob(pattern)):
        if Path(path).resolve() == exclude:
            continue
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != n_expected:
            continue
        if not all(r.get("prediction") in VALID_LABELS for r in rows):
            continue
        runs[Path(path).name] = {int(r["id"]): r["prediction"] for r in rows}
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="the labelling being audited")
    parser.add_argument("--runs", required=True, help="glob of prediction JSONLs to measure consensus over")
    parser.add_argument("--quota", default="uniform")
    parser.add_argument("--out", help="write the ranked candidates as JSON")
    args = parser.parse_args()

    preds, _ = load_predictions(Path(args.predictions))
    quota = parse_quota(args.quota, len(preds))
    counts = Counter(preds.values())
    over = [lab for lab in LABELS if counts[lab] > quota[lab]]
    under = [lab for lab in LABELS if counts[lab] < quota[lab]]
    if not over:
        print("counts already meet the quota -- nothing flagged")
        return

    runs = load_runs(args.runs, len(preds), Path(args.predictions))
    if len(runs) < 2:
        sys.exit(f"--runs matched {len(runs)} usable run(s); consensus needs at least 2")
    print(f"{len(runs)} runs over {len(preds)} items; destinations {under}\n")

    flips = {}
    for label in over:
        surplus = counts[label] - quota[label]
        bucket = sorted(i for i, p in preds.items() if p == label)
        scored = []
        for i in bucket:
            votes = Counter(r[i] for r in runs.values())
            scored.append((votes[label] / len(runs), i, votes))
        scored.sort()

        print(f"bucket {label!r}: {len(bucket)} candidates, surplus {surplus}")
        for rank, (frac, i, votes) in enumerate(scored, 1):
            mark = "  <-- flagged" if rank <= surplus else ""
            spread = " ".join(f"{k}:{v}" for k, v in votes.most_common())
            print(f"  {frac:.2f} agreement  id={i}  [{spread}]{mark}")
        if scored[0][0] == 1.0:
            print("  every candidate is unanimous -- disagreement cannot localize this bucket")
            continue
        if surplus < len(scored) and scored[surplus - 1][0] == scored[surplus][0]:
            print(f"  tie at the cut ({scored[surplus - 1][0]:.2f}) -- use bench/adjudicate.py instead")
            continue

        for _, i, votes in scored[:surplus]:
            # Destination: the label the dissenting runs actually named, if it is
            # an under-full class; otherwise fall back to the sole destination.
            named = [lab for lab, _ in votes.most_common() if lab != label and lab in under]
            if named:
                flips[i] = named[0]
            elif len(under) == 1:
                flips[i] = under[0]
            else:
                print(f"  id={i}: dissent names no under-full class and {len(under)} destinations exist -- unresolved")
        print()

    print("consensus flips: " + (", ".join(f"{i}: {preds[i]} -> {lab}" for i, lab in flips.items()) or "none"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {"n_runs": len(runs), "runs": sorted(runs), "flips": [{"id": i, "from": preds[i], "to": lab} for i, lab in flips.items()]}
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
