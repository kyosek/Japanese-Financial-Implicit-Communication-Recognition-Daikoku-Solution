"""Shared scoring helpers for the JF-ICR label set.

Accuracy and macro F1 both treat the five labels as unordered, which
misreads this task twice over:

- the labels are an ordinal scale, so confusing +1 with +2 is a smaller
  error than confusing +1 with -2, and neither metric knows that.
- the public set is heavily skewed (+1 is 55% of it), so a classifier that
  just answers "+1" every time already scores 0.549 accuracy and 0.949
  "within one level" -- numbers that look respectable but carry no signal.

Quadratic weighted kappa fixes both: errors are penalised by the squared
distance along the scale, and the score is chance-corrected against the
marginals, so the always-majority baseline lands at exactly 0.0.
"""

from collections import Counter

LABELS = ["+2", "+1", "0", "-1", "-2"]

# Ordinal position on the -2..+2 scale, used for the distance weights.
LABEL_RANK = {"-2": 0, "-1": 1, "0": 2, "+1": 3, "+2": 4}


def quadratic_weighted_kappa(gold, pred):
    """QWK over LABELS. Unparsed predictions (None) count as maximally wrong.

    Returns 0.0 when the expected-disagreement matrix is degenerate (a single
    label in both gold and pred), matching the "no better than chance" reading.
    """
    n = len(LABEL_RANK)
    max_dist_sq = (n - 1) ** 2

    observed = 0.0
    for g, p in zip(gold, pred):
        if p is None:
            observed += 1.0  # no credit; weight of a maximally distant miss
        else:
            observed += ((LABEL_RANK[g] - LABEL_RANK[p]) ** 2) / max_dist_sq

    total = len(gold)
    if not total:
        return 0.0

    gold_marginal = Counter(LABEL_RANK[g] for g in gold)
    # Unparsed predictions get spread across the scale rather than dropped, so
    # a model that fails to parse half its outputs cannot win on the marginals.
    pred_marginal = Counter(LABEL_RANK[p] for p in pred if p is not None)
    n_unparsed = sum(1 for p in pred if p is None)

    expected = 0.0
    for i, gi in gold_marginal.items():
        for j, pj in pred_marginal.items():
            expected += ((i - j) ** 2) / max_dist_sq * gi * pj / total
    expected += n_unparsed * (sum(gold_marginal.values()) / total)

    return 1.0 - observed / expected if expected else 0.0


def majority_baseline(gold):
    """Scores for the always-answer-the-majority-label classifier.

    Reported alongside every run so a metric that merely tracks the label
    prior is visibly distinguishable from one that tracks the task.
    """
    if not gold:
        return {"label": None, "accuracy": 0.0, "within_one": 0.0, "qwk": 0.0}

    counts = Counter(gold)
    label = counts.most_common(1)[0][0]
    within_one = sum(
        c for lbl, c in counts.items() if abs(LABEL_RANK[lbl] - LABEL_RANK[label]) <= 1
    )
    return {
        "label": label,
        "accuracy": counts[label] / len(gold),
        "within_one": within_one / len(gold),
        "qwk": quadratic_weighted_kappa(gold, [label] * len(gold)),
    }


def within_one(gold, pred):
    """Fraction of predictions landing within one step on the ordinal scale."""
    if not gold:
        return 0.0
    hits = sum(
        1
        for g, p in zip(gold, pred)
        if p is not None and abs(LABEL_RANK[g] - LABEL_RANK[p]) <= 1
    )
    return hits / len(gold)


def ordinal_block(gold, pred):
    """The ordinal metrics plus their majority baselines, for report dicts."""
    baseline = majority_baseline(gold)
    return {
        "qwk": quadratic_weighted_kappa(gold, pred),
        "within_one": within_one(gold, pred),
        "majority_baseline": baseline,
    }


def print_ordinal_block(block):
    baseline = block["majority_baseline"]
    print(f"QWK             : {block['qwk']:.4f}   (always-'{baseline['label']}' baseline: {baseline['qwk']:.4f})")
    print(f"within one step : {block['within_one']:.4f}   (always-'{baseline['label']}' baseline: {baseline['within_one']:.4f})")
