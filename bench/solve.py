"""Run every JF-ICR prompt through a local llama.cpp server and save predictions.

Each row's `query` is already a complete zero-shot prompt (fixed instructions
+ the Q&A pair + an output reminder). For few-shot, we splice a few labeled
exemplars' Q&A blocks in between the instructions and the target question,
reusing the dataset's own template rather than inventing new wording.

Usage:
    python bench/solve.py \
        --data JF-ICR_public_set.parquet \
        --endpoint http://127.0.0.1:8080 \
        --out outputs/predictions.jsonl

    # few-shot: 1 exemplar per label (5 shots), held out from evaluation
    python bench/solve.py --few-shot-k 1 --out outputs/predictions_fewshot.jsonl
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

VALID_LABELS = ["+2", "+1", "0", "-1", "-2"]
# Longest-first so "+2"/"-2" aren't swallowed by a bare "2" match, etc.
LABEL_PATTERN = re.compile(r"\+2|\+1|-2|-1|(?<![\d.+-])0(?![\d.])")

QUESTION_MARKER = "Financial Question:"
REMINDER_MARKER = "Directly output the chosen label"


def extract_label(raw_text: str) -> str | None:
    """Pull the last valid ICR label out of a model response.

    Models are asked to output only the label, but instruction-following
    isn't perfect (extra whitespace, stray punctuation, occasional reasoning
    leakage) -- taking the *last* match favors a trailing final answer over
    incidental +N/-N-shaped text earlier in any reasoning.
    """
    matches = LABEL_PATTERN.findall(raw_text.strip())
    return matches[-1] if matches else None


def split_query(query: str) -> tuple[str, str, str]:
    """Split a dataset prompt into (shared instructions, QA block, output reminder)."""
    i = query.index(QUESTION_MARKER)
    j = query.index(REMINDER_MARKER)
    return query[:i], query[i:j], query[j:]


def select_exemplars(df: pd.DataFrame, k: int) -> list[dict]:
    """Pick the first k rows (by id) for each label, in +2..-2 order."""
    exemplars = []
    for label in VALID_LABELS:
        subset = df[df["answer"] == label].sort_values("id").head(k)
        exemplars.extend(subset.to_dict("records"))
    return exemplars


def build_prompt(target_query: str, exemplars: list[dict]) -> str:
    prefix, middle, suffix = split_query(target_query)
    parts = [prefix]
    for ex in exemplars:
        _, ex_middle, _ = split_query(ex["query"])
        parts.append(ex_middle)
        parts.append(f"Answer: {ex['answer']}\n\n")
    parts.append(middle)
    parts.append(suffix)
    return "".join(parts)


def call_model(endpoint: str, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    resp = requests.post(
        f"{endpoint}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="JF-ICR_public_set.parquet")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="local")
    parser.add_argument("--out", default="outputs/predictions.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N eval rows (smoke test)")
    parser.add_argument(
        "--few-shot-k",
        type=int,
        default=0,
        help="exemplars per label to prepend (0 = zero-shot). Exemplars are held out of evaluation.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    exemplars = []
    if args.few_shot_k > 0:
        exemplars = select_exemplars(df, args.few_shot_k)
        exemplar_ids = {ex["id"] for ex in exemplars}
        df = df[~df["id"].isin(exemplar_ids)]
        print(f"few-shot: {len(exemplars)} exemplars held out, {len(df)} rows left to evaluate")

    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(df.itertuples(), total=len(df), desc="solving"):
            prompt = build_prompt(row.query, exemplars) if exemplars else row.query
            start = time.monotonic()
            try:
                raw = call_model(args.endpoint, prompt, args.model, args.temperature, args.max_tokens)
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
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    n_unparsed = sum(1 for r in results if r["prediction"] is None)
    if n_unparsed:
        print(f"warning: {n_unparsed}/{len(results)} responses had no extractable label", file=sys.stderr)
    print(f"wrote {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()
