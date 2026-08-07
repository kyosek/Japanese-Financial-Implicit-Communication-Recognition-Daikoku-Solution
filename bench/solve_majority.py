"""Trivial majority-class baseline: always predict the most common gold label.

The README already noted this number (139/253 = 0.5494, always predicting
"+1") as a sanity floor for the LLM results; this script just makes it a
reproducible predictions.jsonl/report.json like every other baseline here,
so bench/compare.py can line it up next to the rest instead of a hardcoded
comment.

Usage:
    python bench/solve_majority.py --out outputs/predictions_majority.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="JF-ICR_public_set.parquet")
    parser.add_argument("--out", default="outputs/predictions_majority.jsonl")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    majority_label = df["answer"].value_counts().idxmax()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in df.itertuples():
            f.write(json.dumps({"id": int(row.id), "gold": row.answer, "prediction": majority_label}, ensure_ascii=False) + "\n")
    print(f"majority label: {majority_label!r} ({(df['answer'] == majority_label).sum()}/{len(df)})")
    print(f"wrote {len(df)} predictions to {out_path}")


if __name__ == "__main__":
    main()
