"""E4: prompt-sensitivity and label-order robustness probes.

The headline results table (README) reports one accuracy number per
model/setting from one fixed prompt. Without checking whether that number
moves under harmless rewording or reordering of the *same* prompt, a delta
between two rows in that table is unfalsifiable -- it could be a real model
difference, or it could just be noise in how the prompt happens to be
written. Two probes here, run independently:

--probe rules: reruns the zero-shot free-generation eval (solve.py) under
bench/rule_paraphrases.py's paraphrases of the Appendix A.3 annotation
rules -- same linguistic-signal cue phrases and worked examples as
solve.py's ANNOTATION_RULES, byte-identical across variants, only the
connective prose/structure differs -- at --seeds different llama.cpp
sampler seeds each. The headline table runs at temperature 0 (greedy),
which is *by construction* insensitive to seed, so this probe samples at
--temperature > 0 instead: that gives a seed-driven noise floor to compare
paraphrase-driven deltas against. Reports, per paraphrase, mean +/- sd
accuracy/macro-F1 across seeds, plus a check of whether the spread across
paraphrases exceeds the largest seed-noise observed.

--probe label-order: reruns the zero-shot **logprob** eval (solve_logprob.py
verbalizer scoring -- deterministic, no seed axis needed) under
bench/label_orders.py's orderings of how the 5 labels are *listed* in the
fixed instruction block (the rules text and the label->meaning mapping are
unchanged; only the order they're presented in changes). The label tokens
score_label() teacher-forces against are identical across orderings, so any
prediction shift is attributable to list position (primacy/recency-style
bias) rather than content. Reports, per ordering vs. the original order:
accuracy/macro-F1, predicted-label marginal frequency, mean row-wise
total-variation distance between softmax P(y|x) vectors, and the argmax
flip rate.

Both probes are zero-shot only (few-shot would add a third confound axis).

Usage:
    python bench/prompt_sensitivity.py --probe rules \\
        --model gemma4 --out-prefix outputs/sensitivity_gemma4

    python bench/prompt_sensitivity.py --probe label-order \\
        --model gemma4 --out-prefix outputs/sensitivity_gemma4
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from label_orders import LABEL_ORDERS, permute_label_order
from rule_paraphrases import RULE_PARAPHRASES
from solve import QUESTION_MARKER, VALID_LABELS, call_model, extract_label, inject_annotation_rules
from solve_logprob import apply_template, score_label, softmax_over_labels, tokenize

LABEL_SCORE = {"+2": 2, "+1": 1, "0": 0, "-1": -1, "-2": -2}
BASELINE_ORDER = "original"


def inject_rules(query: str, rule_text: str) -> str:
    i = query.index(QUESTION_MARKER)
    return query[:i] + rule_text + query[i:]


def score_predictions(records: list[dict]) -> dict:
    """Same accuracy/macro-F1/confusion definitions as evaluate.py, so report_*.json here is compare.py-compatible."""
    total = len(records)
    correct = sum(1 for r in records if r["prediction"] == r["gold"])
    unparsed = sum(1 for r in records if r["prediction"] is None)
    confusion = Counter((r["gold"], r["prediction"]) for r in records)
    f1_per_label = {}
    for label in VALID_LABELS:
        tp = confusion[(label, label)]
        fp = sum(confusion[(g, label)] for g in VALID_LABELS if g != label)
        fn = sum(confusion[(label, p)] for p in VALID_LABELS if p != label) + confusion[(label, None)]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_per_label[label] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_per_label.values()) / len(VALID_LABELS),
        "f1_per_label": f1_per_label,
        "unparsed": unparsed,
        "confusion": {f"{g}|{p}": c for (g, p), c in confusion.items()},
    }


def run_rules_probe(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    seeds = [int(s) for s in args.seeds.split(",")]
    all_metrics: dict[str, list[dict]] = {}

    for variant_name, rule_text in RULE_PARAPHRASES.items():
        variant_metrics = []
        for seed in seeds:
            records = []
            out_path = Path(f"{args.out_prefix}_predictions_rules_{variant_name}_seed{seed}.jsonl")
            with out_path.open("w", encoding="utf-8") as f:
                desc = f"rules:{variant_name} seed={seed}"
                for row in tqdm(df.itertuples(), total=len(df), desc=desc):
                    query = inject_rules(row.query, rule_text)
                    start = time.monotonic()
                    try:
                        raw = call_model(args.endpoint, query, args.model, args.temperature, args.max_tokens, seed=seed)
                        label = extract_label(raw)
                    except requests.RequestException as exc:
                        print(f"\nid={row.id}: request failed: {exc}", file=sys.stderr)
                        raw, label = None, None
                    elapsed = time.monotonic() - start

                    record = {
                        "id": int(row.id),
                        "gold": row.answer,
                        "raw_response": raw,
                        "prediction": label,
                        "elapsed_s": round(elapsed, 2),
                    }
                    records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

            metrics = score_predictions(records)
            metrics["variant"], metrics["seed"] = variant_name, seed
            report_path = Path(f"{args.out_prefix}_report_rules_{variant_name}_seed{seed}.json")
            report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{variant_name} seed={seed}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} unparsed={metrics['unparsed']}")
            variant_metrics.append(metrics)
        all_metrics[variant_name] = variant_metrics

    return all_metrics


def summarize_rules_probe(all_metrics: dict[str, list[dict]]) -> dict:
    summary = {}
    for variant, metrics_list in all_metrics.items():
        accs = [m["accuracy"] for m in metrics_list]
        f1s = [m["macro_f1"] for m in metrics_list]
        summary[variant] = {
            "seeds": [m["seed"] for m in metrics_list],
            "accuracy_mean": statistics.mean(accs),
            "accuracy_sd": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "macro_f1_mean": statistics.mean(f1s),
            "macro_f1_sd": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
            "per_seed": metrics_list,
        }

    cross_variant_accs = [s["accuracy_mean"] for s in summary.values()]
    cross_spread = max(cross_variant_accs) - min(cross_variant_accs)
    max_seed_sd = max(s["accuracy_sd"] for s in summary.values())
    summary["_falsifiability_check"] = {
        "cross_paraphrase_accuracy_spread": cross_spread,
        "largest_within_paraphrase_seed_sd": max_seed_sd,
        "spread_exceeds_seed_noise": cross_spread > max_seed_sd,
    }
    return summary


def print_rules_summary(summary: dict) -> None:
    check = summary["_falsifiability_check"]
    print("\nrule-paraphrase sensitivity (mean +/- sd over seeds):")
    print(f"{'variant':<14}{'n_seeds':>8}   {'accuracy':<18}{'macro_f1':<18}")
    for variant, s in summary.items():
        if variant == "_falsifiability_check":
            continue
        acc = f"{s['accuracy_mean']:.4f} +/- {s['accuracy_sd']:.4f}"
        f1 = f"{s['macro_f1_mean']:.4f} +/- {s['macro_f1_sd']:.4f}"
        print(f"{variant:<14}{len(s['seeds']):>8}   {acc:<18}{f1:<18}")
    print(f"\ncross-paraphrase spread in mean accuracy : {check['cross_paraphrase_accuracy_spread']:.4f}")
    print(f"largest within-paraphrase seed sd        : {check['largest_within_paraphrase_seed_sd']:.4f}")
    if check["spread_exceeds_seed_noise"]:
        print("-> paraphrase-to-paraphrase spread EXCEEDS the largest seed noise observed: there is a wording effect beyond sampling noise.")
    else:
        print("-> paraphrase-to-paraphrase spread does NOT exceed seed noise: accuracy deltas between paraphrases are not distinguishable from sampling noise at this seed count.")


def run_label_order_probe(df: pd.DataFrame, session: requests.Session, args: argparse.Namespace) -> dict:
    metrics_by_order = {}
    records_by_order: dict[str, dict[int, dict]] = {}

    for order_name, order in LABEL_ORDERS.items():
        records = []
        out_path = Path(f"{args.out_prefix}_predictions_labelorder_{order_name}.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for row in tqdm(df.itertuples(), total=len(df), desc=f"label-order={order_name}"):
                query = inject_annotation_rules(permute_label_order(row.query, order))
                start = time.monotonic()
                try:
                    rendered = apply_template(session, args.endpoint, query)
                    label_logprob = {
                        label: score_label(session, args.endpoint, rendered, tokenize(session, args.endpoint, label))
                        for label in VALID_LABELS
                    }
                    label_prob = softmax_over_labels(label_logprob)
                    argmax_label = max(label_logprob, key=label_logprob.get)
                    expected_score = sum(LABEL_SCORE[label] * p for label, p in label_prob.items())
                except requests.RequestException as exc:
                    print(f"\nid={row.id}: request failed: {exc}", file=sys.stderr)
                    label_prob, argmax_label, expected_score = None, None, None
                elapsed = time.monotonic() - start

                record = {
                    "id": int(row.id),
                    "gold": row.answer,
                    "prediction": argmax_label,
                    "expected_score": expected_score,
                    "label_prob": label_prob,
                    "elapsed_s": round(elapsed, 2),
                }
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

        metrics = score_predictions(records)
        metrics["order"] = order_name
        report_path = Path(f"{args.out_prefix}_report_labelorder_{order_name}.json")
        report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"label-order={order_name}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")

        metrics_by_order[order_name] = metrics
        records_by_order[order_name] = {r["id"]: r for r in records}

    baseline_records = records_by_order[BASELINE_ORDER]
    summary = {}
    for order_name, records_by_id in records_by_order.items():
        n = len(records_by_id)
        pred_counts = Counter(r["prediction"] for r in records_by_id.values())
        marginal = {label: pred_counts.get(label, 0) / n for label in VALID_LABELS}
        entry = {
            "accuracy": metrics_by_order[order_name]["accuracy"],
            "macro_f1": metrics_by_order[order_name]["macro_f1"],
            "predicted_label_marginal": marginal,
        }
        if order_name != BASELINE_ORDER:
            tvds, flips, compared = [], 0, 0
            for id_, rec in records_by_id.items():
                base_rec = baseline_records.get(id_)
                if base_rec is None or rec["label_prob"] is None or base_rec["label_prob"] is None:
                    continue
                tvd = 0.5 * sum(abs(rec["label_prob"][label] - base_rec["label_prob"][label]) for label in VALID_LABELS)
                tvds.append(tvd)
                flips += int(rec["prediction"] != base_rec["prediction"])
                compared += 1
            entry["mean_tvd_vs_baseline"] = statistics.mean(tvds) if tvds else None
            entry["argmax_flip_rate_vs_baseline"] = flips / compared if compared else None
        summary[order_name] = entry

    return summary


def print_label_order_summary(summary: dict) -> None:
    print(f"\nlabel-order sensitivity (baseline = {BASELINE_ORDER!r}):")
    print(f"{'order':<16}{'accuracy':>10}{'macro_f1':>10}{'mean_tvd':>12}{'flip_rate':>12}")
    for order_name, e in summary.items():
        tvd = f"{e['mean_tvd_vs_baseline']:.4f}" if e.get("mean_tvd_vs_baseline") is not None else "--"
        flip = f"{e['argmax_flip_rate_vs_baseline']:.4f}" if e.get("argmax_flip_rate_vs_baseline") is not None else "--"
        print(f"{order_name:<16}{e['accuracy']:>10.4f}{e['macro_f1']:>10.4f}{tvd:>12}{flip:>12}")

    print("\npredicted-label marginal frequency per order:")
    print(f"{'order':<16}" + "".join(f"{label:>8}" for label in VALID_LABELS))
    for order_name, e in summary.items():
        print(f"{order_name:<16}" + "".join(f"{e['predicted_label_marginal'][label]:>8.3f}" for label in VALID_LABELS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="JF-ICR_public_set.parquet")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="local")
    parser.add_argument(
        "--out-prefix", required=True, help="prefix for predictions/report/summary files, e.g. outputs/sensitivity_gemma4"
    )
    parser.add_argument("--probe", choices=["rules", "label-order"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N eval rows (smoke test)")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated llama.cpp sampler seeds (--probe rules only)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="--probe rules only -- must be >0 for --seeds to have any effect (temperature 0 is greedy and seed-invariant)",
    )
    parser.add_argument("--max-tokens", type=int, default=32, help="--probe rules only")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)

    out_dir = Path(args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.probe == "rules":
        all_metrics = run_rules_probe(df, args)
        summary = summarize_rules_probe(all_metrics)
        print_rules_summary(summary)
        summary_path = Path(f"{args.out_prefix}_summary_rules.json")
    else:
        session = requests.Session()
        summary = run_label_order_probe(df, session, args)
        print_label_order_summary(summary)
        summary_path = Path(f"{args.out_prefix}_summary_labelorder.json")

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote summary to {summary_path}")


if __name__ == "__main__":
    main()
