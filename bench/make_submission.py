"""Turn a solve.py predictions.jsonl into the shared-task submission CSV.

Usage:
    python bench/make_submission.py \
        --predictions outputs/predictions_qwen36_test_zeroshot_rule-off.jsonl \
        --test-set JF-ICR_test_participant.parquet \
        --out submission.csv

Validates the submission contract before writing: exactly one row per test-set
id, in test-set order, each prediction one of the five ICR labels. Anything
short of that is an error rather than a silently malformed upload.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

LABELS = {"+2", "+1", "0", "-1", "-2"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, type=Path, help="predictions.jsonl from bench/solve.py")
    parser.add_argument("--test-set", required=True, type=Path, help="official test parquet (supplies id order)")
    parser.add_argument("--out", required=True, type=Path, help="destination CSV")
    args = parser.parse_args()

    expected_ids = [int(i) for i in pd.read_parquet(args.test_set)["id"]]

    by_id: dict[int, str | None] = {}
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_id[int(row["id"])] = row["prediction"]

    problems = []
    missing = [i for i in expected_ids if i not in by_id]
    if missing:
        problems.append(f"missing predictions for ids: {missing}")
    extra = sorted(set(by_id) - set(expected_ids))
    if extra:
        problems.append(f"predictions for ids not in the test set: {extra}")
    bad = {i: by_id[i] for i in expected_ids if i in by_id and by_id[i] not in LABELS}
    if bad:
        problems.append(f"predictions outside {sorted(LABELS)}: {bad}")
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        raise SystemExit(1)

    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["id", "prediction"])
        for test_id in expected_ids:
            writer.writerow([test_id, by_id[test_id]])

    print(f"wrote {args.out} ({len(expected_ids)} rows) from {args.predictions}")


if __name__ == "__main__":
    main()
