"""Turn a solve.py predictions JSONL into a JF-ICR submission CSV.

Matches submission.csv byte-for-byte in format: an `id,prediction` header,
one row per test item ordered by id, labels written bare (no quoting), LF
line endings, no BOM, trailing newline.

Refuses to write anything that would be silently rejected or mis-scored: a
missing id, an unparsed (null) prediction, a label outside the five valid
ones, or a prediction set that doesn't exactly match the dataset's ids. A
submission is one-shot, so these fail loudly rather than warn.

Usage:
    python bench/make_submission.py \
        --predictions outputs/predictions_qwen36_test_zeroshot_thinking.jsonl \
        --data JF-ICR_test_participant.parquet \
        --out outputs/submission_qwen36_thinking.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve import VALID_LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data", default="JF-ICR_test_participant.parquet", help="the set being submitted on")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preds = {}
    for line in Path(args.predictions).read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            preds[int(r["id"])] = r["prediction"]

    df = pd.read_parquet(args.data)
    want = {int(i) for i in df["id"]}

    problems = []
    missing = sorted(want - preds.keys())
    if missing:
        problems.append(f"{len(missing)} dataset ids have no prediction: {missing[:10]}")
    extra = sorted(preds.keys() - want)
    if extra:
        problems.append(f"{len(extra)} predicted ids are not in the dataset: {extra[:10]}")
    null = sorted(i for i in want & preds.keys() if preds[i] is None)
    if null:
        problems.append(
            f"{len(null)} predictions are null (unparsed, or truncated at the token cap): {null[:10]}"
        )
    bad = sorted(i for i in want & preds.keys() if preds[i] is not None and preds[i] not in VALID_LABELS)
    if bad:
        problems.append(f"{len(bad)} predictions are not one of {VALID_LABELS}: {[(i, preds[i]) for i in bad[:10]]}")

    if problems:
        print(f"refusing to write {args.out}:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    ids = sorted(want)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" plus an explicit LF terminator: csv defaults to CRLF, which
    # would not match submission.csv.
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "prediction"])
        for i in ids:
            w.writerow([i, preds[i]])

    dist = {lab: sum(1 for i in ids if preds[i] == lab) for lab in VALID_LABELS}
    print(f"wrote {out_path} -- {len(ids)} rows, ids {ids[0]}..{ids[-1]}")
    print("label distribution: " + "  ".join(f"{lab}:{n}" for lab, n in dist.items()))


if __name__ == "__main__":
    main()
