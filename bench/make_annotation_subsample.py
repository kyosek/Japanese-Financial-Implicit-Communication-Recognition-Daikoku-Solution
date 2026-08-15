"""Draw a stratified subsample of JF-ICR for blind multi-annotator re-labeling.

Why this exists: every model we have scores ~0.65 accuracy / ~0.50 QWK on the
public set and ~0.93 / ~0.97 on the participant set. We cannot currently tell
how much of that gap is model weakness and how much is irreducible label
ambiguity, because the public set has exactly one label per item and no
measured human agreement. This script produces the sheets needed to measure
the human ceiling; bench/agreement.py scores them once they come back.

Two strata, recorded per item so estimates can be reweighted:

- "random": proportionally stratified over the gold labels. Gives an unbiased
  estimate of overall human agreement on the set as a whole.
- "boundary": drawn from the adjacent label pairs that account for most of the
  model errors (+1/0 and +1/+2 are 74 of gemma4's 94 public-set errors). Buys
  statistical power exactly where the ambiguity is suspected, at the cost of
  needing reweighting for population-level claims.

When --predictions files are supplied, boundary items are ranked by model
disagreement (mean normalized entropy over the supplied label distributions,
plus argmax spread across files) rather than drawn at random from the
boundary classes. Note the resulting circularity: those items are selected
for being model-hard, so "humans also disagree here" is a claim about
model-hard items specifically, not about the set overall -- which is what the
random stratum is for.

Usage:
    python bench/make_annotation_subsample.py \
        --data JF-ICR_public_set.parquet \
        --predictions outputs/predictions_gemma4_zeroshot_logprob.jsonl \
        --predictions outputs/predictions_qwen36_zeroshot_logprob.jsonl \
        --n-random 40 --n-boundary 40 --annotators 3 \
        --out-dir annotation/round1
"""

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import LABELS, LABEL_RANK

QUESTION_MARKER = "Financial Question:"
RESPONSE_MARKER = "Company Response:"
REMINDER_MARKER = "Directly output the chosen label"

# The adjacent pairs that dominate the public-set confusion matrix.
BOUNDARY_PAIRS = [("+1", "0"), ("+1", "+2")]
BOUNDARY_LABELS = sorted({lbl for pair in BOUNDARY_PAIRS for lbl in pair}, key=lambda l: LABEL_RANK[l])

# Shown at the top of every sheet so annotators work from the same definitions
# without having to hold the paper open. Deliberately the bare label
# definitions, not the Appendix A.3 rules -- if we hand annotators the same
# fine-grained cues we hand the models, we measure agreement with the
# guideline rather than the difficulty of the judgment.
LABEL_GUIDE = """JF-ICR intent labels (choose exactly one per row):

  +2  Strong Commitment              明確・確定的なコミットメント
  +1  Weak or Qualified Commitment   留保つき/方向性のみのコミットメント
   0  Neutral or Hedged Intent       中立、ヘッジ、説明のみ
  -1  Weak Refusal                   弱い拒否・消極的な否定
  -2  Strong Refusal                 明確な拒否・否定

Judge the company's underlying intent toward the action the question asks
about. Label what the response commits to, not whether the answer is good
news. If a response addresses several points, label the intent toward the
point the question actually asked about.

Fill in the `label` column. If you genuinely cannot decide between two
labels, still pick one, and note the runner-up in `second_choice` -- the
disagreement itself is the measurement, so please do not confer with the
other annotators.
"""


def split_query(query):
    """Pull the question and response text back out of a built prompt."""
    question = re.search(
        rf"{re.escape(QUESTION_MARKER)}\s*(.*?)\n{re.escape(RESPONSE_MARKER)}", query, re.S
    )
    response = re.search(
        rf"{re.escape(RESPONSE_MARKER)}\s*(.*?)\n\s*{re.escape(REMINDER_MARKER)}", query, re.S
    )
    return (
        question.group(1).strip() if question else "",
        response.group(1).strip() if response else "",
    )


def load_dataset(path):
    path = Path(path)
    if path.suffix == ".parquet":
        records = pd.read_parquet(path).to_dict("records")
    else:
        records = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for record in records:
        gold = (record.get("annotation_answer") or record.get("answer") or "").strip()
        question, response = split_query(record["query"])
        rows.append(
            {
                "id": int(record["id"]),
                "gold": gold or None,
                "question": question,
                "response": response,
            }
        )
    return rows


def load_uncertainty(prediction_paths):
    """Mean normalized entropy and argmax spread per id, across prediction files.

    Files without label_prob contribute only their argmax. Returns {} when no
    usable prediction files were supplied.
    """
    entropies = defaultdict(list)
    argmaxes = defaultdict(list)

    for path in prediction_paths:
        for line in Path(path).open(encoding="utf-8"):
            record = json.loads(line)
            item_id = int(record["id"])
            if record.get("prediction") is not None:
                argmaxes[item_id].append(record["prediction"])
            probs = record.get("label_prob")
            if not probs:
                continue
            values = [max(float(probs[label]), 1e-12) for label in LABELS]
            total = sum(values)
            entropy = -sum((v / total) * math.log(v / total) for v in values)
            entropies[item_id].append(entropy / math.log(len(LABELS)))

    uncertainty = {}
    for item_id in set(entropies) | set(argmaxes):
        ent = entropies.get(item_id, [])
        preds = argmaxes.get(item_id, [])
        uncertainty[item_id] = {
            "mean_entropy": sum(ent) / len(ent) if ent else 0.0,
            "n_distinct_preds": len(set(preds)),
            "model_preds": preds,
        }
    return uncertainty


def stratified_random(rows, n, rng):
    """Proportional allocation over gold labels, largest-remainder rounded."""
    pool = [r for r in rows if r["gold"]]
    by_label = defaultdict(list)
    for row in pool:
        by_label[row["gold"]].append(row)

    n = min(n, len(pool))
    exact = {label: len(items) * n / len(pool) for label, items in by_label.items()}
    quota = {label: int(value) for label, value in exact.items()}
    # Hand out the rounding shortfall to the largest fractional parts.
    shortfall = n - sum(quota.values())
    for label in sorted(exact, key=lambda l: exact[l] - quota[l], reverse=True)[:shortfall]:
        quota[label] += 1

    picked = []
    for label, items in by_label.items():
        take = min(quota[label], len(items))
        picked.extend(rng.sample(items, take))
    return picked


def boundary_sample(rows, n, rng, uncertainty, exclude):
    """Items on the +1/0 and +1/+2 boundaries, hardest-first when we can rank."""
    pool = [r for r in rows if r["gold"] in BOUNDARY_LABELS and r["id"] not in exclude]
    n = min(n, len(pool))
    if not n:
        return []

    if uncertainty:
        # Rank by model disagreement; break ties randomly so equal-entropy
        # items don't get ordered by id.
        jitter = {row["id"]: rng.random() for row in pool}
        pool.sort(
            key=lambda r: (
                uncertainty.get(r["id"], {}).get("n_distinct_preds", 1),
                uncertainty.get(r["id"], {}).get("mean_entropy", 0.0),
                jitter[r["id"]],
            ),
            reverse=True,
        )
        return pool[:n]

    return rng.sample(pool, n)


def write_sheets(picked, out_dir, annotators, rng, include_second_choice=True):
    """One blind CSV + JSONL per annotator, each independently shuffled.

    Independent shuffles keep order/fatigue effects from correlating across
    annotators, which would otherwise inflate apparent agreement. Neither the
    gold label nor any model prediction appears in the sheets.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "GUIDELINE.txt").write_text(LABEL_GUIDE, encoding="utf-8")

    columns = ["row", "id", "question", "response", "label"]
    if include_second_choice:
        columns.append("second_choice")
    columns.append("notes")

    for index in range(annotators):
        name = chr(ord("A") + index) if index < 26 else str(index + 1)
        order = picked[:]
        rng.shuffle(order)

        csv_path = out_dir / f"sheet_{name}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for position, row in enumerate(order, start=1):
                entry = {
                    "row": position,
                    "id": row["id"],
                    "question": row["question"],
                    "response": row["response"],
                    "label": "",
                    "notes": "",
                }
                if include_second_choice:
                    entry["second_choice"] = ""
                writer.writerow(entry)

        jsonl_path = out_dir / f"sheet_{name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for position, row in enumerate(order, start=1):
                handle.write(
                    json.dumps(
                        {
                            "row": position,
                            "id": row["id"],
                            "question": row["question"],
                            "response": row["response"],
                            "label": None,
                            "second_choice": None,
                            "notes": "",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"  wrote {csv_path} and {jsonl_path.name} ({len(order)} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="JF-ICR_public_set.parquet")
    parser.add_argument("--predictions", action="append", default=[],
                        help="logprob prediction jsonl; repeatable. Ranks the boundary stratum by model disagreement.")
    parser.add_argument("--n-random", type=int, default=40)
    parser.add_argument("--n-boundary", type=int, default=40)
    parser.add_argument("--annotators", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--out-dir", default="annotation/round1")
    parser.add_argument("--no-second-choice", action="store_true",
                        help="drop the runner-up column from the sheets")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_dataset(args.data)
    labeled = [r for r in rows if r["gold"]]
    print(f"loaded {len(rows)} rows from {args.data} ({len(labeled)} with a gold label)")

    uncertainty = load_uncertainty(args.predictions)
    if args.predictions:
        print(f"loaded model uncertainty for {len(uncertainty)} ids from {len(args.predictions)} file(s)")
    else:
        print("no --predictions given; boundary stratum will be drawn at random")

    random_rows = stratified_random(labeled, args.n_random, rng)
    chosen_ids = {r["id"] for r in random_rows}
    boundary_rows = boundary_sample(labeled, args.n_boundary, rng, uncertainty, chosen_ids)

    for row in random_rows:
        row["stratum"] = "random"
    for row in boundary_rows:
        row["stratum"] = "boundary"
    picked = random_rows + boundary_rows

    # Inclusion probability per stratum, so agreement.py can reweight the
    # boundary-enriched sample back to a population estimate.
    n_boundary_pool = len([r for r in labeled if r["gold"] in BOUNDARY_LABELS])
    weights = {
        "random": len(labeled) / len(random_rows) if random_rows else None,
        "boundary": n_boundary_pool / len(boundary_rows) if boundary_rows else None,
    }

    out_dir = Path(args.out_dir)
    print(f"\nsampled {len(picked)} items "
          f"({len(random_rows)} random, {len(boundary_rows)} boundary)")
    print("  random stratum gold distribution  :",
          dict(Counter(r["gold"] for r in random_rows)))
    print("  boundary stratum gold distribution:",
          dict(Counter(r["gold"] for r in boundary_rows)))
    print(f"\nwriting {args.annotators} blind sheet(s) to {out_dir}/")
    write_sheets(picked, out_dir, args.annotators, rng, include_second_choice=not args.no_second_choice)

    key_path = out_dir / "key.json"
    key_path.write_text(
        json.dumps(
            {
                "data": str(args.data),
                "seed": args.seed,
                "predictions_used": args.predictions,
                "stratum_weights": weights,
                "items": [
                    {
                        "id": r["id"],
                        "stratum": r["stratum"],
                        "original_gold": r["gold"],
                        "mean_entropy": uncertainty.get(r["id"], {}).get("mean_entropy"),
                        "n_distinct_model_preds": uncertainty.get(r["id"], {}).get("n_distinct_preds"),
                        "model_preds": uncertainty.get(r["id"], {}).get("model_preds", []),
                    }
                    for r in picked
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote key (gold labels, strata, weights) to {key_path}")
    print("Do not share key.json with the annotators.")


if __name__ == "__main__":
    main()
