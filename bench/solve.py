"""Run every JF-ICR prompt through a local llama.cpp server and save predictions.

Each row's `query` is already a complete zero-shot prompt (fixed instructions
+ the Q&A pair + an output reminder). For few-shot, we splice a few labeled
exemplars' Q&A blocks in between the instructions and the target question,
reusing the dataset's own template rather than inventing new wording.

By default we also inject the EBISU paper's Appendix A.3 "Specific Annotation
Rules" (the per-label linguistic-signal cues and representative examples used
by the human annotators) right before the Q&A block, giving the model the
same fine-grained guidance the annotators had. Disable with
--no-annotation-rules to reproduce the bare label-definition prompt.

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

# EBISU paper, Appendix A.3 "Specific Annotation Rules" -- the linguistic
# signals and representative examples annotators used to tell adjacent
# labels apart (e.g. +1 vs +2, -1 vs -2). Spliced into the prompt right
# before the Q&A block by inject_annotation_rules().
ANNOTATION_RULES = """Detailed classification rules (from the annotation guideline), \
use these linguistic signals and examples to decide the label:

"+2" (Strong Commitment): a clear, decisive statement indicating a firm commitment.
  Signals: 「〜します」「〜を実施します」「〜を達成します」 / 「〜を決定しています」「〜は確定しています」 / 「方針を変更する考えはありません」(when directly answering a decision or policy question)
  Examples: 「来期は増配を実施します。」 / 「中長期でROE12%を達成します。」 / 「この施策により利益成長は実現できると確信しています。」 / 「次期計画の具体的な検討に着手する方針です。」 / 「変更はありません。」

"+1" (Weak or Qualified Commitment): a positive or leaning commitment with qualifications, caution, or limited specificity -- directional intent rather than a finalized decision.
  Signals: 「〜していきたい」「〜を目指しています」「〜している」 / 「〜と考えています」「〜を見込んでいます」 / conditional or hypothetical expressions (e.g., 「〜であれば」「〜次第で」)
  Examples: 「成長投資を進めていきたいと考えています。」 / 「今後も収益は拡大していくと見ています。」 / 「環境が整えば、検討を進める考えです。」  / 「収益効果を得られるかを試算する。」

"0" (Neutral or Hedged Intent): genuine ambiguity, or clarification/explanation/background without commitment or refusal toward any future action.
  Signals: 「〜断定できません」「明確な見通しは示せない」 / 「検討中」「状況を見極める必要がある」 / purely descriptive or explanatory statements providing facts or background
  Examples: 「現時点では明確な見通しは示せません。」 / 「様々な見方があり、コメントは差し控えます。」 / 「過去にはこのような取り組みを行ってきました。」(background explanation only)

"-1" (Weak Refusal): a negative stance that is qualified, conditional, or time-bound, leaving room for future reconsideration.
  Signals: 「現時点では〜しない」「直ちには考えていない」 / 「今後検討の余地はあるが」 / refusals framed as temporary, conditional, or dependent on future circumstances
  Examples: 「現時点では配当方針を変更する考えはありません。」 / 「今中計期間中に見直すことは想定していません。」 / 「足元では難しいと考えていますが、今後は検討します。」

"-2" (Strong Refusal): a clear and definitive rejection, leaving no visible room for reconsideration.
  Signals: 「〜する予定はありません」 / 「〜は行いません」「〜を否定します」
  Examples: 「株式分割を行う予定はありません。」 / 「当該事業への投資は実施しません。」 / 「その想定は当社の方針ではありません。」

"""


def inject_annotation_rules(query: str) -> str:
    """Splice the Appendix A.3 rules into a dataset prompt, right before the Q&A block."""
    i = query.index(QUESTION_MARKER)
    return query[:i] + ANNOTATION_RULES + query[i:]


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


def call_model(
    endpoint: str, prompt: str, model: str, temperature: float, max_tokens: int, seed: int | None = None
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    resp = requests.post(f"{endpoint}/v1/chat/completions", json=payload, timeout=120)
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
    parser.add_argument(
        "--no-annotation-rules",
        action="store_true",
        help="omit the Appendix A.3 label-specific linguistic-signal rules (included by default)",
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
            query = row.query if args.no_annotation_rules else inject_annotation_rules(row.query)
            prompt = build_prompt(query, exemplars) if exemplars else query
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
                "gold": getattr(row, "answer", None),
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
