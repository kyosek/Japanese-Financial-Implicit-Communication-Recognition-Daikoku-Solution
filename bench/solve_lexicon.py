"""Non-LLM baseline (a): hedging/modality cue-lexicon features + logistic regression.

Rather than asking an LLM to read the Q&A pair, this counts occurrences of a
small, hand-curated set of Japanese hedging/modality/commitment cue phrases
in the company's response text -- sentence-final forms (差し控えさせていた
だきます, 検討してまいります, 努めてまいります, 注視してまいります, 〜と考え
ております), the benefactive 〜させていただく, and conditional/epistemic
markers -- and fits a logistic regression on those counts.

If this simple, transparent baseline rivals the LLMs in bench/solve.py, that
says the JF-ICR label is largely recoverable from surface lexical form (which
is exactly what the annotation guideline's own per-label "linguistic signal"
rules, spliced into the LLM prompts by inject_annotation_rules(), describe)
rather than requiring deeper reasoning about the response's content.

Cross-validated rather than train/test split: the public set is only 253
rows and one class (-2) has just 2 examples, so a single held-out split would
be too noisy to trust. K-fold (plain, not stratified -- stratification isn't
feasible with a 2-member class) gives an out-of-fold prediction for every
row, comparable to the LLM runs via evaluate.py.

Usage:
    python bench/solve_lexicon.py --out outputs/predictions_lexicon.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from solve import VALID_LABELS, split_query

RESPONSE_MARKER = "Company Response:"

# Each group is a linguistic phenomenon; a row's feature value is the total
# number of substring hits across all phrases in the group, found in the
# company response text only (not the analyst's question).
CUE_GROUPS = {
    "sentence_final_hedge": [
        "差し控えさせていただきます", "差し控えます", "差し控えたいと思います",
        "検討してまいります", "努めてまいります", "注視してまいります",
        "と考えております", "と考えています",
    ],
    "benefactive": [
        "させていただきます", "させていただきたい", "させていただいて", "させていただき",
    ],
    "conditional": [
        "であれば", "次第で", "次第です", "場合には", "場合は", "たら、",
    ],
    "epistemic": [
        "と思います", "と考えます", "と見ています", "と見込んでいます", "と見込んでおります",
        "かもしれません", "可能性があります", "可能性がある", "想定しています", "想定しております",
        "と存じます",
    ],
    "strong_commitment": [
        "を実施します", "を達成します", "決定しています", "決定しております",
        "確定しています", "確定しております", "確実に",
    ],
    "strong_refusal": [
        "予定はありません", "予定はございません", "は行いません", "を否定します",
        "考えておりません", "想定しておりません",
    ],
    "directional_commitment": [
        "していきたい", "を目指しています", "を目指してまいります", "取り組んでまいります",
    ],
    "neutral_explanatory": [
        "断定できません", "明確な見通しは示せません", "検討中です", "状況を見極める",
    ],
}
FEATURE_NAMES = [*CUE_GROUPS.keys(), "response_length"]


def extract_response(query: str) -> str:
    """Pull just the company's response text out of a dataset prompt."""
    _, middle, _ = split_query(query)
    i = middle.index(RESPONSE_MARKER) + len(RESPONSE_MARKER)
    return middle[i:].strip()


def featurize(text: str) -> list[float]:
    counts = [float(sum(text.count(phrase) for phrase in phrases)) for phrases in CUE_GROUPS.values()]
    return [*counts, len(text) / 100.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="JF-ICR_public_set.parquet")
    parser.add_argument("--out", default="outputs/predictions_lexicon.jsonl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0, help="logistic regression inverse regularization strength")
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="'balanced' re-weights the severely skewed label distribution (trades accuracy for recall on rare labels)",
    )
    args = parser.parse_args()
    class_weight = None if args.class_weight == "none" else args.class_weight

    df = pd.read_parquet(args.data)
    texts = [extract_response(q) for q in df["query"]]
    X = np.array([featurize(t) for t in texts])
    y = df["answer"].to_numpy()
    ids = df["id"].to_numpy()

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_pred = np.full(len(df), None, dtype=object)
    for train_idx, test_idx in kf.split(X):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=args.C, class_weight=class_weight))
        clf.fit(X[train_idx], y[train_idx])
        oof_pred[test_idx] = clf.predict(X[test_idx])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(len(df)):
            record = {
                "id": int(ids[idx]),
                "gold": y[idx],
                "prediction": oof_pred[idx],
                "features": dict(zip(FEATURE_NAMES, X[idx].tolist())),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(df)} out-of-fold predictions to {out_path}")

    # Fit once on all data purely for interpretability -- which cues the model
    # actually leans on per label. Not used for the reported predictions above.
    full_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=args.C, class_weight=class_weight))
    full_clf.fit(X, y)
    lr = full_clf.named_steps["logisticregression"]
    print("\nlogistic regression coefficients (fit on all data, for interpretation only):")
    header = f"{'feature':<24}" + "".join(f"{label:>10}" for label in lr.classes_)
    print(header)
    for j, name in enumerate(FEATURE_NAMES):
        row = "".join(f"{lr.coef_[k, j]:>10.2f}" for k in range(len(lr.classes_)))
        print(f"{name:<24}{row}")


if __name__ == "__main__":
    main()
