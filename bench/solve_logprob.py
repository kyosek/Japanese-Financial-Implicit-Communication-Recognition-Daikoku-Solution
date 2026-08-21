"""Score every JF-ICR prompt by verbaliser log-prob instead of free generation.

Rather than letting the model generate freely and parsing a label out of the
text (solve.py), this computes P(label | prompt) directly for each of the
five label strings and takes the argmax -- no parsing, no unparsed-label
failure mode, and a continuous score (Sigma k*P(k)) as a bonus.

How: llama.cpp's raw /completion endpoint returns the top-N token
log-probs at a position without sampling anything. We render the exact
chat-templated prompt via /apply-template (so this sees precisely what
/v1/chat/completions would feed the model), tokenize each candidate label
(they're 1-2 tokens depending on the model's vocab -- "+2" is usually
["+", "2"]), and teacher-force through the label's tokens one at a time,
summing log-probs along the way. cache_prompt reuses the KV cache across
the 5 label branches for a given row since they share the same prefix.

Caveat -- reasoning/harmony-format models (e.g. gpt-oss-20b): if the
model's chat template mandates an analysis-channel preamble before any
content can appear, the label does NOT immediately follow the prompt, and
scoring it there measures the wrong thing (every label will look equally
unlikely). Run with --sanity-check-n first and check the agreement rate
against free generation before trusting a full run on a new model.

Usage:
    python bench/solve_logprob.py --model shisa --out outputs/predictions_logprob.jsonl
    python bench/solve_logprob.py --model shisa --few-shot-k 1 --out outputs/predictions_logprob_fewshot.jsonl
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from solve import VALID_LABELS, build_prompt, extract_label, inject_annotation_rules, select_exemplars

LABEL_SCORE = {"+2": 2, "+1": 1, "0": 0, "-1": -1, "-2": -2}
N_PROBS_LADDER = [40, 500, 5000]


def apply_template(session: requests.Session, endpoint: str, prompt: str) -> str:
    resp = session.post(
        f"{endpoint}/apply-template",
        json={"messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["prompt"]


def tokenize(session: requests.Session, endpoint: str, text: str) -> list[dict]:
    resp = session.post(
        f"{endpoint}/tokenize",
        json={"content": text, "with_pieces": True},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["tokens"]


def next_token_logprob(session: requests.Session, endpoint: str, ctx: str, target_id: int) -> float:
    """P(target_id | ctx) under the model's raw (unsampled) next-token distribution."""
    for n_probs in N_PROBS_LADDER:
        resp = session.post(
            f"{endpoint}/completion",
            json={
                "prompt": ctx,
                "n_predict": 1,
                "n_probs": n_probs,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "cache_prompt": True,
            },
            timeout=120,
        )
        resp.raise_for_status()
        candidates = resp.json()["completion_probabilities"][0]["top_logprobs"]
        match = next((c for c in candidates if c["id"] == target_id), None)
        if match is not None:
            return match["logprob"]
    # target token wasn't in the top N_PROBS_LADDER[-1] candidates -- effectively negligible
    return float("-inf")


def score_label(session: requests.Session, endpoint: str, rendered_prompt: str, label_tokens: list[dict]) -> float:
    """Teacher-forced joint log P(label | rendered_prompt), one token at a time."""
    total = 0.0
    ctx = rendered_prompt
    for tok in label_tokens:
        lp = next_token_logprob(session, endpoint, ctx, tok["id"])
        if lp == float("-inf"):
            return float("-inf")
        total += lp
        ctx += tok["piece"]
    return total


def softmax_over_labels(logprobs: dict[str, float]) -> dict[str, float]:
    m = max(logprobs.values())
    exps = {label: math.exp(lp - m) for label, lp in logprobs.items()}
    z = sum(exps.values())
    return {label: v / z for label, v in exps.items()}


def sanity_check(
    session: requests.Session,
    endpoint: str,
    model: str,
    df: pd.DataFrame,
    n: int,
    no_annotation_rules: bool,
    multipart_rule: str = "always",
) -> None:
    """Cross-check log-prob argmax against free generation on a few rows before a full run."""
    sample = df.head(n)
    agree = 0
    for row in sample.itertuples():
        query = row.query if no_annotation_rules else inject_annotation_rules(row.query, multipart_rule)
        gen_resp = session.post(
            f"{endpoint}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": query}],
                "temperature": 0.0,
                "max_tokens": 32,
            },
            timeout=120,
        )
        gen_resp.raise_for_status()
        gen_label = extract_label(gen_resp.json()["choices"][0]["message"]["content"])

        rendered = apply_template(session, endpoint, query)
        label_logprobs = {
            label: score_label(session, endpoint, rendered, tokenize(session, endpoint, label)) for label in VALID_LABELS
        }
        lp_label = max(label_logprobs, key=label_logprobs.get)

        agree += int(gen_label == lp_label)
        print(f"  id={row.id}: free-gen={gen_label!r}  logprob-argmax={lp_label!r}", file=sys.stderr)

    rate = agree / len(sample) if len(sample) else 0.0
    print(f"sanity check: {agree}/{len(sample)} ({rate:.0%}) agreement between free generation and logprob argmax", file=sys.stderr)
    if rate < 0.8:
        print(
            "warning: low agreement -- this model's chat template may require a reasoning/channel "
            "preamble before content (see gpt-oss caveat in the module docstring). Inspect before trusting a full run.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="JF-ICR_public_set.parquet")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="local")
    parser.add_argument("--out", default="outputs/predictions_logprob.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N eval rows (smoke test)")
    parser.add_argument(
        "--few-shot-k",
        type=int,
        default=0,
        help="exemplars per label to prepend (0 = zero-shot). Exemplars are held out of evaluation.",
    )
    parser.add_argument(
        "--no-annotation-rules",
        action="store_true",
        help="omit the Appendix A.3 label-specific linguistic-signal rules (included by default)",
    )
    parser.add_argument(
        "--multipart-rule",
        choices=["off", "always", "gated"],
        default="always",
        help="when to state the multi-part aggregation rule, mirroring solve.py. Default 'always' matches "
        "the historical behaviour; 'gated' adds it only to questions bench/multipart.py flags as bundling "
        "several asks, and is the right choice on a set with no multi-part questions",
    )
    parser.add_argument(
        "--sanity-check-n",
        type=int,
        default=5,
        help="cross-check logprob argmax vs free generation on N rows before the full run (0 = skip)",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    exemplars = []
    if args.few_shot_k > 0:
        exemplars = select_exemplars(df, args.few_shot_k)
        exemplar_ids = {ex["id"] for ex in exemplars}
        df = df[~df["id"].isin(exemplar_ids)]
        print(f"few-shot: {len(exemplars)} exemplars held out, {len(df)} rows left to evaluate")

    session = requests.Session()

    if args.sanity_check_n > 0:
        sanity_check(
            session, args.endpoint, args.model, df, args.sanity_check_n, args.no_annotation_rules, args.multipart_rule
        )

    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(df.itertuples(), total=len(df), desc="scoring"):
            query = row.query if args.no_annotation_rules else inject_annotation_rules(row.query, args.multipart_rule)
            prompt = build_prompt(query, exemplars) if exemplars else query
            start = time.monotonic()
            try:
                rendered = apply_template(session, args.endpoint, prompt)
                label_logprob = {
                    label: score_label(session, args.endpoint, rendered, tokenize(session, args.endpoint, label))
                    for label in VALID_LABELS
                }
                label_prob = softmax_over_labels(label_logprob)
                argmax_label = max(label_logprob, key=label_logprob.get)
                expected_score = sum(LABEL_SCORE[label] * p for label, p in label_prob.items())
            except requests.RequestException as exc:
                print(f"\nid={row.id}: request failed: {exc}", file=sys.stderr)
                label_logprob, label_prob, argmax_label, expected_score = None, None, None, None
            elapsed = time.monotonic() - start

            record = {
                "id": int(row.id),
                # The competition test parquet ships without an `answer` column,
                # so gold is absent when scoring the real submission set.
                "gold": getattr(row, "answer", None),
                "prediction": argmax_label,
                "expected_score": expected_score,
                "label_logprob": label_logprob,
                "label_prob": label_prob,
                "elapsed_s": round(elapsed, 2),
            }
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    n_failed = sum(1 for r in results if r["prediction"] is None)
    if n_failed:
        print(f"warning: {n_failed}/{len(results)} rows failed to score", file=sys.stderr)
    print(f"wrote {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()
