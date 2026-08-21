"""Score returned annotation sheets: human agreement, and the ceiling it implies.

Answers the question the single-annotator public set cannot: when models score
~0.50 QWK there, is that model weakness or an ambiguous task? Compare the
model QWK against the human ceiling this reports. A model at or above the mean
annotator-vs-annotator QWK is not "failing" -- it is at the noise floor of the
label, and the remaining headroom is a guideline problem, not a modelling one.

Reports:
  - pairwise QWK between every annotator pair, and their mean (the ceiling)
  - Krippendorff's alpha with the ordinal difference metric (chance-corrected,
    handles >2 annotators and missing cells in one number)
  - the same, split by sampling stratum, since the boundary stratum is
    deliberately enriched for hard items and will read lower
  - agreement with the dataset's existing gold label
  - "plural gold" coverage: how often a model's single answer matches ANY
    annotator. This is the defensible version of scoring against a model's own
    top-2 -- the acceptable set is defined by observed human variation rather
    than by the model's own probability ranking, so a model cannot earn credit
    by hedging towards the majority class.

Usage:
    python bench/agreement.py --sheets annotation/round1 \
        --predictions "gemma4"=outputs/predictions_gemma4_zeroshot.jsonl \
        --predictions "qwen36"=outputs/predictions_qwen36_zeroshot.jsonl
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import LABELS, LABEL_RANK, quadratic_weighted_kappa

ORDERED = sorted(LABELS, key=lambda l: LABEL_RANK[l])


def normalise_label(raw):
    """Accept '+1', '1', '＋１', ' 0 ' etc.; return None for blanks."""
    if raw is None:
        return None
    text = str(raw).strip().translate(str.maketrans("＋－０１２", "+-012"))
    if not text:
        return None
    if text in LABEL_RANK:
        return text
    if text.lstrip("+-").isdigit():
        value = int(text)
        candidate = f"{value:+d}" if value else "0"
        if candidate in LABEL_RANK:
            return candidate
    return None


def load_sheets(sheet_dir):
    """{annotator: {id: label}} from sheet_*.csv / sheet_*.jsonl."""
    sheets = {}
    for path in sorted(Path(sheet_dir).glob("sheet_*.*")):
        if path.suffix not in {".csv", ".jsonl"}:
            continue
        name = path.stem.replace("sheet_", "")
        # Prefer a filled CSV over the JSONL twin if both are present.
        if name in sheets and path.suffix == ".jsonl":
            continue

        labels = {}
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    label = normalise_label(row.get("label"))
                    if label:
                        labels[int(row["id"])] = label
        else:
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                label = normalise_label(row.get("label"))
                if label:
                    labels[int(row["id"])] = label

        if labels:
            sheets[name] = labels
        else:
            print(f"  (skipping {path.name}: no labels filled in)")
    return sheets


def krippendorff_alpha_ordinal(units):
    """Alpha with the ordinal difference metric. `units` = list of label lists."""
    usable = [u for u in units if len(u) >= 2]
    if not usable:
        return None

    # Coincidence matrix over ordered label values.
    coincidence = defaultdict(float)
    for unit in usable:
        m = len(unit)
        for a, b in combinations(range(m), 2):
            for c, k in ((unit[a], unit[b]), (unit[b], unit[a])):
                coincidence[(c, k)] += 1.0 / (m - 1)

    counts = defaultdict(float)
    for (c, _), value in coincidence.items():
        counts[c] += value
    # Freeze to a plain dict: delta_sq reads labels that may be absent from the
    # marginals, and a defaultdict would insert them mid-iteration below.
    marginal = dict(counts)
    n = sum(marginal.values())
    if n <= 1:
        return None

    def delta_sq(c, k):
        i, j = LABEL_RANK[c], LABEL_RANK[k]
        lo, hi = min(i, j), max(i, j)
        between = sum(marginal.get(ORDERED[g], 0.0) for g in range(lo, hi + 1))
        return (between - (marginal.get(c, 0.0) + marginal.get(k, 0.0)) / 2.0) ** 2

    observed = sum(value * delta_sq(c, k) for (c, k), value in coincidence.items())
    expected = sum(
        marginal[c] * marginal[k] * delta_sq(c, k)
        for c in marginal
        for k in marginal
    ) / (n - 1)
    return 1.0 - observed / expected if expected else None


def bootstrap_ci(units, n_boot, seed):
    """Percentile CI for ordinal alpha, resampling units. Small n needs this."""
    if not units:
        return None, None
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        draw = [units[rng.randrange(len(units))] for _ in units]
        alpha = krippendorff_alpha_ordinal(draw)
        if alpha is not None:
            samples.append(alpha)
    if not samples:
        return None, None
    samples.sort()
    return samples[int(0.025 * len(samples))], samples[min(int(0.975 * len(samples)), len(samples) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheets", default="annotation/round1",
                        help="directory holding the filled sheet_*.csv and key.json")
    parser.add_argument("--predictions", action="append", default=[],
                        help='"label"=path/to/predictions.jsonl; repeatable')
    parser.add_argument("--report", default=None, help="optional path for a JSON report")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    sheet_dir = Path(args.sheets)
    sheets = load_sheets(sheet_dir)
    if len(sheets) < 2:
        sys.exit(f"need at least 2 filled sheets in {sheet_dir}, found {len(sheets)}")

    key_path = sheet_dir / "key.json"
    key = json.loads(key_path.read_text(encoding="utf-8")) if key_path.exists() else {"items": []}
    stratum = {int(i["id"]): i["stratum"] for i in key["items"]}
    original_gold = {int(i["id"]): i["original_gold"] for i in key["items"]}

    names = sorted(sheets)
    all_ids = sorted(set().union(*(set(s) for s in sheets.values())))
    complete = [i for i in all_ids if all(i in sheets[n] for n in names)]

    print(f"annotators      : {', '.join(names)}")
    print(f"items labelled  : {len(all_ids)} ({len(complete)} labelled by all {len(names)})")
    coverage = {n: len(sheets[n]) for n in names}
    if len(set(coverage.values())) > 1:
        print(f"  per-annotator counts: {coverage}")

    # --- pairwise QWK: the human ceiling -------------------------------------
    print("\npairwise QWK between annotators")
    pairwise = {}
    for a, b in combinations(names, 2):
        shared = [i for i in all_ids if i in sheets[a] and i in sheets[b]]
        if not shared:
            continue
        score = quadratic_weighted_kappa([sheets[a][i] for i in shared], [sheets[b][i] for i in shared])
        pairwise[f"{a}-{b}"] = score
        exact = sum(sheets[a][i] == sheets[b][i] for i in shared) / len(shared)
        print(f"  {a} vs {b}   n={len(shared):3d}  QWK={score:.4f}  exact={exact:.4f}")
    ceiling = sum(pairwise.values()) / len(pairwise) if pairwise else None
    if ceiling is not None:
        print(f"  mean pairwise QWK (HUMAN CEILING) = {ceiling:.4f}")

    # --- Krippendorff alpha, overall and by stratum --------------------------
    units = [[sheets[n][i] for n in names if i in sheets[n]] for i in all_ids]
    alpha = krippendorff_alpha_ordinal(units)
    low, high = bootstrap_ci(units, args.bootstrap, args.seed)
    print("\nKrippendorff's alpha (ordinal)")
    if alpha is None:
        print("  n/a (not enough pairable judgements)")
    else:
        ci = f"  95% CI [{low:.4f}, {high:.4f}]" if low is not None else ""
        print(f"  overall  n={len(units):3d}  alpha={alpha:.4f}{ci}")

    by_stratum = {}
    for name in sorted(set(stratum.values())):
        ids = [i for i in all_ids if stratum.get(i) == name]
        sub = [[sheets[n][i] for n in names if i in sheets[n]] for i in ids]
        value = krippendorff_alpha_ordinal(sub)
        by_stratum[name] = value
        if value is not None:
            print(f"  {name:8s} n={len(sub):3d}  alpha={value:.4f}")

    # --- agreement with the dataset's existing label -------------------------
    scored_ids = [i for i in all_ids if original_gold.get(i)]
    vs_gold = {}
    if scored_ids:
        print("\nannotator vs. dataset's existing gold label")
        for name in names:
            shared = [i for i in scored_ids if i in sheets[name]]
            if not shared:
                continue
            score = quadratic_weighted_kappa([original_gold[i] for i in shared], [sheets[name][i] for i in shared])
            exact = sum(original_gold[i] == sheets[name][i] for i in shared) / len(shared)
            vs_gold[name] = score
            print(f"  {name}  n={len(shared):3d}  QWK={score:.4f}  exact={exact:.4f}")

    # --- models against the same items and the same ceiling ------------------
    model_rows = []
    for item in args.predictions:
        label, path = item.split("=", 1)
        preds = {}
        for line in Path(path).open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("prediction") is not None:
                preds[int(record["id"])] = record["prediction"]

        shared = [i for i in complete if i in preds]
        if not shared:
            print(f"\n  ({label}: no overlap with the annotated subsample, skipping)")
            continue

        # Mean QWK against each annotator individually -- directly comparable
        # to the annotator-vs-annotator numbers above.
        per_annotator = [
            quadratic_weighted_kappa([sheets[n][i] for i in shared], [preds[i] for i in shared])
            for n in names
        ]
        mean_qwk = sum(per_annotator) / len(per_annotator)
        any_match = sum(1 for i in shared if preds[i] in {sheets[n][i] for n in names}) / len(shared)
        unanimous = [i for i in shared if len({sheets[n][i] for n in names}) == 1]
        acc_unanimous = (
            sum(1 for i in unanimous if preds[i] == sheets[names[0]][i]) / len(unanimous)
            if unanimous else None
        )
        model_rows.append((label, len(shared), mean_qwk, any_match, acc_unanimous, len(unanimous)))

    if model_rows:
        print("\nmodels on the annotated subsample")
        print(f"  {'model':<14} {'n':>4} {'mean QWK':>9} {'matches any':>12} {'acc on unanimous':>17}")
        for label, n, mean_qwk, any_match, acc_unanimous, n_unanimous in model_rows:
            unanimous_str = f"{acc_unanimous:.4f} (n={n_unanimous})" if acc_unanimous is not None else "n/a"
            print(f"  {label:<14} {n:>4} {mean_qwk:>9.4f} {any_match:>12.4f} {unanimous_str:>17}")
        if ceiling is not None:
            print(f"\n  human ceiling (mean pairwise QWK) = {ceiling:.4f}")
            best = max(model_rows, key=lambda r: r[2])
            gap = ceiling - best[2]
            verdict = (
                f"best model ({best[0]}) is {gap:.4f} QWK below the ceiling"
                if gap > 0 else
                f"best model ({best[0]}) is AT or ABOVE the human ceiling"
            )
            print(f"  {verdict}")

    if args.report:
        report = {
            "annotators": names,
            "n_items": len(all_ids),
            "n_complete": len(complete),
            "pairwise_qwk": pairwise,
            "human_ceiling_qwk": ceiling,
            "krippendorff_alpha_ordinal": alpha,
            "krippendorff_alpha_ci95": [low, high],
            "krippendorff_alpha_by_stratum": by_stratum,
            "annotator_vs_original_gold_qwk": vs_gold,
            "label_distribution": {n: dict(Counter(sheets[n].values())) for n in names},
            "models": [
                {
                    "label": label,
                    "n": n,
                    "mean_qwk_vs_annotators": mean_qwk,
                    "matches_any_annotator": any_match,
                    "accuracy_on_unanimous": acc_unanimous,
                    "n_unanimous": n_unanimous,
                }
                for label, n, mean_qwk, any_match, acc_unanimous, n_unanimous in model_rows
            ],
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote report to {report_path}")


if __name__ == "__main__":
    main()
