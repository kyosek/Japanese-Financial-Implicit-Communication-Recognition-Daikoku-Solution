"""Does splitting multi-part questions into pieces make JF-ICR easier to label?

A chunk of the public set bundles several asks into one `Financial Question:`
(「〜の背景を教えてほしい。また、今後の方針は。」). The label is defined over the
whole (question, response) pair, so the annotator has to collapse a response
that may commit to one sub-ask and hedge on another into a single point on the
-2..+2 scale -- with no aggregation rule in the guideline. The obvious fix is
to decompose, label each piece, then aggregate.

This measures whether that actually helps, by testing the three places the
ambiguity could be hiding rather than assuming it lives in the question:

  1. SEGMENTATION -- is "how many questions is this" even well-defined?
     Recorded as the LLM's sub-question count vs. the regex heuristic's, plus
     whether each returned sub-question is a verbatim span of the original
     (a paraphrase means the splitter had to interpret, not just cut).

  2. ATTRIBUTION -- an IR response is one block that does not map 1:1 onto the
     sub-asks, and the characteristic move is to answer A at length while
     declining B by never mentioning it. The coverage probe asks, per
     sub-question, whether the response addresses it at all; unaddressed
     pieces are where decomposition invents a labeling decision that the
     joint framing never had to make.

  3. AGGREGATION -- if the sub-labels diverge, the single benchmark label
     depends entirely on which rule you pick. We score first / last /
     strongest / weakest / modal / mean-round against gold, so "which rule
     reproduces the annotator" becomes an answerable question instead of a
     guess.

The decisive comparison is DECOMPOSE-THEN-AGGREGATE vs. JOINT labeling by the
same model, at the same settings, on the same items: if no aggregation rule
beats the joint label, decomposition is relocating the ambiguity, not removing
it. SINGLE-part control items are run through the identical pipeline -- if
their sub-labels diverge about as often as the multi-part ones', the noise is
coming from chopping up discourse-level hedging, not from multi-part-ness.

Sub-labels reuse the dataset's own prompt template (and, by default, the same
Appendix A.3 annotation rules solve.py injects) with only the question text
swapped, so a sub-label is directly comparable to a main-task label.

Usage:
    python bench/decompose_experiment.py \
        --data JF-ICR_public_set.parquet \
        --endpoint http://127.0.0.1:8080 --model qwen36 \
        --out outputs/decompose_experiment.jsonl \
        --report outputs/report_decompose_experiment.json
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import LABEL_RANK, quadratic_weighted_kappa
from multipart import heuristic_multipart
from solve import (
    QUESTION_MARKER,
    REMINDER_MARKER,
    RESPONSE_MARKER,
    call_model,
    extract_label,
    inject_annotation_rules,
)

RANK_TO_LABEL = {rank: label for label, rank in LABEL_RANK.items()}


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


def substitute_question(query, new_question):
    """Rebuild a dataset prompt with the question text replaced, response intact."""
    i = query.index(QUESTION_MARKER) + len(QUESTION_MARKER)
    j = query.index(RESPONSE_MARKER)
    return query[:i] + " " + new_question.strip() + "\n" + query[j:]


# --- LLM stages -------------------------------------------------------------
DECOMPOSE_PROMPT = """You are a Japanese financial expert reading questions asked by analysts at Japanese investor relations meetings.

Below is one question entry from an IR transcript. It may contain a single ask, or several distinct asks bundled together.

Split it into the distinct asks it actually contains. Rules:
- Copy each ask VERBATIM from the original text wherever possible; do not paraphrase or translate.
- Background/premise statements that carry no ask of their own are NOT separate asks -- attach them to the ask they set up, or drop them.
- A restatement of the same ask from a different angle is ONE ask, not two.
- If the entry contains only one ask, return a list with exactly one element.

Question entry:
{question}

Output ONLY a JSON array of strings, no explanation, no markdown fence.
Example format: ["...", "..."]
Answer:
"""

COVERAGE_PROMPT = """You are a Japanese financial expert reading a Japanese investor relations Q&A.

Sub-question asked by the analyst:
{sub_question}

Full company response:
{response}

Does the company response actually address this specific sub-question -- that is, does any part of the response speak to it, whether by answering, hedging, or explicitly declining?
Answer NO if the response simply never engages with this sub-question at all.

Output exactly one word, YES or NO, with no explanation.
Answer:
"""


def parse_json_list(raw):
    """Best-effort JSON array extraction from a model response."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    items, seen = [], set()
    for element in parsed:
        if not isinstance(element, str):
            continue
        cleaned = element.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            items.append(cleaned)
    return items or None


def ask(endpoint, prompt, model, max_tokens, retries=2):
    """One chat call with retries; returns the raw string or None."""
    for attempt in range(retries):
        try:
            return call_model(endpoint, prompt, model, 0.0, max_tokens)
        except requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"  request failed: {exc}", file=sys.stderr)
                return None
            time.sleep(2)
    return None


# --- aggregation rules ------------------------------------------------------
def aggregate(labels):
    """Collapse sub-labels to one label under each candidate rule."""
    usable = [l for l in labels if l in LABEL_RANK]
    if not usable:
        return {}
    ranks = [LABEL_RANK[l] for l in usable]
    counts = Counter(usable)
    top = max(counts.values())
    modal_candidates = [l for l, c in counts.items() if c == top]
    mean_rank = sum(ranks) / len(ranks)
    # Tie-break the mode toward the piece closest to the item's centre of mass,
    # so a 2-way tie doesn't resolve on dict insertion order.
    modal = min(modal_candidates, key=lambda l: (abs(LABEL_RANK[l] - mean_rank), -LABEL_RANK[l]))
    return {
        "first": usable[0],
        "last": usable[-1],
        "strongest": RANK_TO_LABEL[max(ranks)],   # most committal piece wins
        "weakest": RANK_TO_LABEL[min(ranks)],     # most refusing/hedged piece wins
        "modal": modal,
        "mean_round": RANK_TO_LABEL[int(round(mean_rank))],
    }


AGG_RULES = ["first", "last", "strongest", "weakest", "modal", "mean_round"]


def score(gold, pred):
    """Accuracy / within-one / QWK for a list of (gold, pred) pairs."""
    pairs = [(g, p) for g, p in zip(gold, pred) if g in LABEL_RANK]
    if not pairs:
        return {"n": 0, "accuracy": None, "within_one": None, "qwk": None}
    golds = [g for g, _ in pairs]
    preds = [p for _, p in pairs]
    correct = sum(1 for g, p in pairs if g == p)
    near = sum(
        1 for g, p in pairs if p in LABEL_RANK and abs(LABEL_RANK[g] - LABEL_RANK[p]) <= 1
    )
    return {
        "n": len(pairs),
        "accuracy": correct / len(pairs),
        "within_one": near / len(pairs),
        "qwk": quadratic_weighted_kappa(golds, preds),
    }


def load_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        records = pd.read_parquet(path).to_dict("records")
    else:
        records = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for record in records:
        # Prefer a re-annotated label when the JSON carries one, matching
        # make_annotation_subsample.py's precedence.
        gold = (record.get("annotation_answer") or record.get("answer") or "").strip()
        question, response = split_query(record["query"])
        is_multi, n_ask, connectives = heuristic_multipart(question)
        rows.append(
            {
                "id": int(record["id"]),
                "query": record["query"],
                "gold": gold or None,
                "question": question,
                "response": response,
                "heuristic_multi": is_multi,
                "n_ask_sentences": n_ask,
                "connectives": connectives,
            }
        )
    return rows


def run_item(row, args):
    """Decompose -> joint label -> per-sub label + coverage, for one item."""
    # multipart_rule="off" on purpose: this experiment is what produced that
    # rule, and a sub-question prompt has no bundled asks to aggregate. Keeping
    # it off also keeps the joint arm comparable to the pre-rule baseline.
    prep = (
        (lambda q: q)
        if args.no_annotation_rules
        else (lambda q: inject_annotation_rules(q, multipart_rule="off"))
    )

    joint_raw = ask(args.endpoint, prep(row["query"]), args.model, args.max_tokens)
    joint = extract_label(joint_raw or "")

    decomp_raw = ask(
        args.endpoint,
        DECOMPOSE_PROMPT.format(question=row["question"]),
        args.model,
        args.decompose_max_tokens,
    )
    subs = parse_json_list(decomp_raw or "")
    if subs is None:
        subs = [row["question"]]
        decompose_ok = False
    else:
        decompose_ok = True
    subs = subs[: args.max_subquestions]

    pieces = []
    for sub in subs:
        sub_prompt = prep(substitute_question(row["query"], sub))
        sub_label = extract_label(ask(args.endpoint, sub_prompt, args.model, args.max_tokens) or "")

        covered_raw = ask(
            args.endpoint,
            COVERAGE_PROMPT.format(sub_question=sub, response=row["response"]),
            args.model,
            8,
        )
        covered = None
        if covered_raw:
            upper = covered_raw.strip().upper()
            if "NO" in upper and "YES" not in upper:
                covered = False
            elif "YES" in upper:
                covered = True

        pieces.append(
            {
                "sub_question": sub,
                "verbatim": sub in row["question"],
                "label": sub_label,
                "addressed": covered,
            }
        )

    labels = [p["label"] for p in pieces if p["label"] in LABEL_RANK]
    ranks = [LABEL_RANK[l] for l in labels]
    return {
        "id": row["id"],
        "gold": row["gold"],
        "stratum": "multi" if row["heuristic_multi"] else "single",
        "question": row["question"],
        "n_ask_sentences": row["n_ask_sentences"],
        "connectives": row["connectives"],
        "decompose_ok": decompose_ok,
        "n_sub": len(pieces),
        "pieces": pieces,
        "joint_label": joint,
        "joint_raw": joint_raw,
        "sub_labels": [p["label"] for p in pieces],
        "diverged": len(set(labels)) > 1,
        "spread": (max(ranks) - min(ranks)) if ranks else None,
        "aggregated": aggregate([p["label"] for p in pieces]),
    }


def summarize(results, rows_all):
    """Everything the experiment is meant to answer, as a report dict."""
    n_all = len(rows_all)
    n_multi_all = sum(1 for r in rows_all if r["heuristic_multi"])

    report = {
        "prevalence": {
            "n_dataset": n_all,
            "n_heuristic_multi": n_multi_all,
            "share_heuristic_multi": n_multi_all / n_all if n_all else 0.0,
            "n_ask_sentence_distribution": dict(
                sorted(Counter(r["n_ask_sentences"] for r in rows_all).items())
            ),
        },
        "strata": {},
    }

    for stratum in ("multi", "single"):
        group = [r for r in results if r["stratum"] == stratum]
        if not group:
            continue

        gold = [r["gold"] for r in group]
        block = {
            "n": len(group),
            "decompose_parse_failures": sum(1 for r in group if not r["decompose_ok"]),
            # --- segmentation ---
            "n_sub_distribution": dict(sorted(Counter(r["n_sub"] for r in group).items())),
            "mean_n_sub": sum(r["n_sub"] for r in group) / len(group),
            "llm_says_multi": sum(1 for r in group if r["n_sub"] >= 2),
            "verbatim_rate": (
                sum(1 for r in group for p in r["pieces"] if p["verbatim"])
                / max(1, sum(len(r["pieces"]) for r in group))
            ),
            # --- divergence ---
            "n_splittable": sum(1 for r in group if r["n_sub"] >= 2),
            "n_diverged": sum(1 for r in group if r["n_sub"] >= 2 and r["diverged"]),
            "spread_distribution": dict(
                sorted(Counter(r["spread"] for r in group if r["n_sub"] >= 2 and r["spread"] is not None).items())
            ),
            # --- attribution ---
            "n_subquestions": sum(len(r["pieces"]) for r in group),
            "n_unaddressed": sum(
                1 for r in group for p in r["pieces"] if p["addressed"] is False
            ),
            # --- the decisive comparison ---
            "joint_vs_gold": score(gold, [r["joint_label"] for r in group]),
            "aggregation_vs_gold": {
                rule: score(gold, [r["aggregated"].get(rule) for r in group])
                for rule in AGG_RULES
            },
        }
        splittable = block["n_splittable"]
        block["divergence_rate"] = block["n_diverged"] / splittable if splittable else None

        # Divergence conditioned on whether every piece was actually addressed:
        # if it concentrates on items with an unaddressed piece, the extra
        # disagreement is attribution noise decomposition created.
        for name, keep in (
            ("all_addressed", lambda r: all(p["addressed"] is not False for p in r["pieces"])),
            ("has_unaddressed", lambda r: any(p["addressed"] is False for p in r["pieces"])),
        ):
            subset = [r for r in group if r["n_sub"] >= 2 and keep(r)]
            block[f"divergence_{name}"] = {
                "n": len(subset),
                "rate": (sum(1 for r in subset if r["diverged"]) / len(subset)) if subset else None,
            }

        # Restricted to items the LLM actually split, where the joint-vs-
        # decomposed comparison is a real contrast rather than the same call twice.
        split_only = [r for r in group if r["n_sub"] >= 2]
        if split_only:
            split_gold = [r["gold"] for r in split_only]
            block["split_only"] = {
                "n": len(split_only),
                "joint_vs_gold": score(split_gold, [r["joint_label"] for r in split_only]),
                "aggregation_vs_gold": {
                    rule: score(split_gold, [r["aggregated"].get(rule) for r in split_only])
                    for rule in AGG_RULES
                },
            }

        report["strata"][stratum] = block

    return report


def print_report(report):
    prevalence = report["prevalence"]
    print("\n" + "=" * 72)
    print("MULTI-PART PREVALENCE (regex heuristic, whole dataset)")
    print("=" * 72)
    print(f"  {prevalence['n_heuristic_multi']}/{prevalence['n_dataset']} items "
          f"({prevalence['share_heuristic_multi']:.1%}) look multi-part")
    print(f"  ask-sentences per question: {prevalence['n_ask_sentence_distribution']}")

    for stratum, block in report["strata"].items():
        print("\n" + "=" * 72)
        print(f"STRATUM: {stratum}  (n={block['n']})")
        print("=" * 72)

        print("\n-- 1. segmentation: is 'how many questions' well-defined? --")
        print(f"  LLM split into >=2 pieces : {block['llm_says_multi']}/{block['n']}")
        print(f"  sub-question count dist   : {block['n_sub_distribution']}")
        print(f"  mean pieces per item      : {block['mean_n_sub']:.2f}")
        print(f"  verbatim (not paraphrased): {block['verbatim_rate']:.1%}")
        if block["decompose_parse_failures"]:
            print(f"  decompose parse failures  : {block['decompose_parse_failures']}")

        print("\n-- 2. attribution: does the response cover each piece? --")
        unaddressed_share = block["n_unaddressed"] / max(1, block["n_subquestions"])
        print(f"  sub-questions the response never engages: "
              f"{block['n_unaddressed']}/{block['n_subquestions']} ({unaddressed_share:.1%})")

        print("\n-- 3. divergence: do the pieces disagree with each other? --")
        rate = block["divergence_rate"]
        print(f"  items whose sub-labels disagree: {block['n_diverged']}/{block['n_splittable']}"
              + (f" ({rate:.1%})" if rate is not None else ""))
        print(f"  ordinal spread distribution    : {block['spread_distribution']}")
        for name in ("all_addressed", "has_unaddressed"):
            entry = block[f"divergence_{name}"]
            shown = f"{entry['rate']:.1%}" if entry["rate"] is not None else "n/a"
            print(f"    {name:16s} n={entry['n']:3d}  divergence={shown}")

        print("\n-- 4. does decompose-then-aggregate beat labeling jointly? --")
        for scope in ("all items", "items the LLM actually split"):
            data = block if scope == "all items" else block.get("split_only")
            if not data:
                continue
            print(f"  [{scope}]")
            joint = data["joint_vs_gold"]
            if joint["accuracy"] is None:
                print("    joint: no gold-labeled items")
                continue
            print(f"    {'joint (no split)':<18} n={joint['n']:3d}  "
                  f"acc={joint['accuracy']:.4f}  within1={joint['within_one']:.4f}  qwk={joint['qwk']:.4f}")
            for rule in AGG_RULES:
                entry = data["aggregation_vs_gold"][rule]
                if entry["accuracy"] is None:
                    continue
                delta = entry["accuracy"] - joint["accuracy"]
                print(f"    {'agg:' + rule:<18} n={entry['n']:3d}  "
                      f"acc={entry['accuracy']:.4f}  within1={entry['within_one']:.4f}  "
                      f"qwk={entry['qwk']:.4f}  (acc {delta:+.4f} vs joint)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="JF-ICR_public_set.parquet")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="local")
    parser.add_argument("--out", default="outputs/decompose_experiment.jsonl")
    parser.add_argument("--report", default="outputs/report_decompose_experiment.json")
    parser.add_argument("--n-multi", type=int, default=40,
                        help="multi-part items to sample (0 = all detected)")
    parser.add_argument("--n-single", type=int, default=20,
                        help="single-part control items to sample (0 = all)")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--decompose-max-tokens", type=int, default=512)
    parser.add_argument("--max-subquestions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--no-annotation-rules", action="store_true",
                        help="omit the Appendix A.3 rules from the label prompts")
    args = parser.parse_args()

    rows = load_rows(args.data)
    multi = [r for r in rows if r["heuristic_multi"]]
    single = [r for r in rows if not r["heuristic_multi"]]
    print(f"dataset: {len(rows)} items -- {len(multi)} multi-part, {len(single)} single-part (heuristic)")

    rng = random.Random(args.seed)
    picked_multi = sorted(rng.sample(multi, min(args.n_multi, len(multi))) if args.n_multi else multi,
                          key=lambda r: r["id"])
    picked_single = sorted(rng.sample(single, min(args.n_single, len(single))) if args.n_single else single,
                           key=lambda r: r["id"])
    sample = picked_multi + picked_single
    print(f"sampling {len(picked_multi)} multi + {len(picked_single)} single = {len(sample)} items")
    print(f"~{sum(2 + 2 * 2 for _ in sample)} model calls expected (varies with piece count)\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with out_path.open("w", encoding="utf-8") as handle:
        for row in tqdm(sample, desc="decomposing"):
            result = run_item(row, args)
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()

    report = summarize(results, rows)
    print_report(report)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {len(results)} records to {out_path}")
    print(f"wrote report to {report_path}")


if __name__ == "__main__":
    main()
