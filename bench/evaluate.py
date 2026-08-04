"""Score predictions produced by solve.py against gold labels.

Usage:
    python bench/evaluate.py --predictions outputs/predictions.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

LABELS = ["+2", "+1", "0", "-1", "-2"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="outputs/predictions.jsonl")
    parser.add_argument("--report", default="outputs/report.json")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.predictions).open(encoding="utf-8")]

    total = len(records)
    correct = sum(1 for r in records if r["prediction"] == r["gold"])
    unparsed = sum(1 for r in records if r["prediction"] is None)
    accuracy = correct / total if total else 0.0

    confusion = Counter((r["gold"], r["prediction"]) for r in records)

    print(f"n examples      : {total}")
    print(f"correct         : {correct}")
    print(f"accuracy        : {accuracy:.4f}")
    print(f"unparsed labels : {unparsed}")
    print()
    print("confusion matrix (rows=gold, cols=predicted; '?' = unparsed)")
    header = ["gold\\pred", *LABELS, "?"]
    print("  ".join(f"{h:>6}" for h in header))
    for gold in LABELS:
        row = [confusion[(gold, pred)] for pred in LABELS]
        row.append(confusion[(gold, None)])
        print("  ".join(f"{v:>6}" for v in [gold, *row]))

    report = {
        "n": total,
        "correct": correct,
        "accuracy": accuracy,
        "unparsed": unparsed,
        "confusion": {f"{g}|{p}": c for (g, p), c in confusion.items()},
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote report to {report_path}")


if __name__ == "__main__":
    main()
