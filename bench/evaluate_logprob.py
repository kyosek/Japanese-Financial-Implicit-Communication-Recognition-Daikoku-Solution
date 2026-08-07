"""Score predictions produced by solve_logprob.py against gold labels.

Same accuracy/confusion-matrix report as evaluate.py (argmax vs gold), plus
two diagnostics free generation can't give you:

- rank of the gold label within the model's own 5-way probability ranking.
  If gold is consistently ranked #2 but rarely argmax, that's a
  prior/calibration problem (the model favors an adjacent label by a hair),
  not a comprehension failure -- a very different fix than if gold is
  consistently ranked #4-5.
- mean P(gold) per gold label, a quick calibration signal alongside the
  rank histogram.

Usage:
    python bench/evaluate_logprob.py --predictions outputs/predictions_logprob.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["+2", "+1", "0", "-1", "-2"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", default="outputs/predictions_logprob.jsonl")
    parser.add_argument("--report", default="outputs/report_logprob.json")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.predictions).open(encoding="utf-8")]

    total = len(records)
    correct = sum(1 for r in records if r["prediction"] == r["gold"])
    unparsed = sum(1 for r in records if r["prediction"] is None)
    accuracy = correct / total if total else 0.0

    confusion = Counter((r["gold"], r["prediction"]) for r in records)

    f1_per_label = {}
    for label in LABELS:
        tp = confusion[(label, label)]
        fp = sum(confusion[(g, label)] for g in LABELS if g != label)
        fn = sum(confusion[(label, p)] for p in LABELS if p != label) + confusion[(label, None)]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_per_label[label] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    macro_f1 = sum(f1_per_label.values()) / len(LABELS)

    rank_hist = {label: [0] * len(LABELS) for label in LABELS}
    prob_sum = defaultdict(float)
    prob_n = defaultdict(int)
    for r in records:
        if r["label_prob"] is None:
            continue
        gold = r["gold"]
        ranked = sorted(LABELS, key=lambda label: r["label_prob"][label], reverse=True)
        rank = ranked.index(gold) + 1  # 1 = argmax
        rank_hist[gold][rank - 1] += 1
        prob_sum[gold] += r["label_prob"][gold]
        prob_n[gold] += 1

    print(f"n examples      : {total}")
    print(f"correct (argmax): {correct}")
    print(f"accuracy        : {accuracy:.4f}")
    print(f"macro F1        : {macro_f1:.4f}")
    print(f"failed to score : {unparsed}")
    print()
    print("confusion matrix (rows=gold, cols=argmax pred; '?' = failed to score)")
    header = ["gold\\pred", *LABELS, "?"]
    print("  ".join(f"{h:>6}" for h in header))
    for gold in LABELS:
        row = [confusion[(gold, pred)] for pred in LABELS]
        row.append(confusion[(gold, None)])
        print("  ".join(f"{v:>6}" for v in [gold, *row]))
    print()
    print("gold-label rank in the model's own 5-way probability ranking (1 = argmax)")
    print("  ".join(f"{h:>6}" for h in ["gold", "n", "P(gold)", *[f"rank{k}" for k in range(1, len(LABELS) + 1)]]))
    for gold in LABELS:
        n = prob_n[gold]
        mean_p = prob_sum[gold] / n if n else 0.0
        print("  ".join(f"{v:>6}" for v in [gold, n, f"{mean_p:.3f}", *rank_hist[gold]]))

    report = {
        "n": total,
        "correct": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "f1_per_label": f1_per_label,
        "unparsed": unparsed,
        "confusion": {f"{g}|{p}": c for (g, p), c in confusion.items()},
        "rank_histogram": rank_hist,
        "mean_prob_gold": {label: (prob_sum[label] / prob_n[label] if prob_n[label] else None) for label in LABELS},
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote report to {report_path}")


if __name__ == "__main__":
    main()
