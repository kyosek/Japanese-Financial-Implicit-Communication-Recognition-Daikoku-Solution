"""Score predictions produced by solve.py against gold labels.

Usage:
    python bench/evaluate.py --predictions outputs/predictions.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import LABELS, ordinal_block, print_ordinal_block


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

    # Per-label precision/recall/F1 from the confusion counts, then macro-averaged.
    # Matters here because gold labels are heavily skewed (+1 is 55% of the set) --
    # accuracy alone rewards a classifier that just leans on the majority class.
    f1_per_label = {}
    for label in LABELS:
        tp = confusion[(label, label)]
        fp = sum(confusion[(g, label)] for g in LABELS if g != label)
        fn = sum(confusion[(label, p)] for p in LABELS if p != label) + confusion[(label, None)]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_per_label[label] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    macro_f1 = sum(f1_per_label.values()) / len(LABELS)

    ordinal = ordinal_block([r["gold"] for r in records], [r["prediction"] for r in records])

    print(f"n examples      : {total}")
    print(f"correct         : {correct}")
    print(f"accuracy        : {accuracy:.4f}")
    print(f"macro F1        : {macro_f1:.4f}")
    print_ordinal_block(ordinal)
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
        "macro_f1": macro_f1,
        "f1_per_label": f1_per_label,
        **ordinal,
        "unparsed": unparsed,
        "confusion": {f"{g}|{p}": c for (g, p), c in confusion.items()},
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote report to {report_path}")


if __name__ == "__main__":
    main()
