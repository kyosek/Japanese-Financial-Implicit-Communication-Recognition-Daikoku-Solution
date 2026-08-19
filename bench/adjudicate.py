"""Adjudicate a quota-flagged bucket by forced choice, instead of by hand.

bench/quota_audit.py proves how many predictions are wrong and which class
buckets they are in, but it cannot say *which item*. Closing that gap needed a
human second opinion. This script tries to close it with a model instead.

The obvious approach is to run a second model and look for disagreement. On
this dataset *majority voting* does not work: across 16 full test-set runs
(claude, gemma4, qwen36, qwen38, over zeroshot/thinking/rule-gated/multipart/
logprob variants) the majority label on the one genuinely wrong item, id=300,
is `+2` by 12 votes to 3 -- the ensemble is confidently wrong, and its majority
label is identical to plain argmax on all 50 items.

But the vote *spread* is informative even where the majority is not. Within
the quota-flagged `+2` bucket, 10 of the 11 candidates are unanimous at 16/16;
id=300 is the sole exception at 12/16. So minimum-consensus inside the flagged
bucket localizes the error exactly, and bench/consensus.py does that with no
extra model call. Prefer it when several runs already exist.

This script covers the case where they do not, or where consensus leaves a tie.
Rather than "classify this item" -- which the ensemble answers wrongly by
majority -- it consumes the quota certificate and asks a comparative question:

    These N items were all labelled `+2`. Exactly one of them is really `0`.
    Which one?

Three things make this easier than the original task. It is forced choice
over a shortlist rather than open 5-way classification; it is comparative, so
the items calibrate each other instead of each being judged against an
internalized threshold; and it is told the answer exists, which blocks the
"they all look like commitments" failure that produced the error in the first
place. A model that is confidently wrong item-by-item can still rank the odd
one out correctly, because ranking does not require the absolute threshold
that item-by-item classification gets wrong.

The bucket and the surplus count are not guesses -- they come from the quota
arithmetic, which is exact. This script only chooses within them.

Usage:
    scripts/serve.sh Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -c 16384

    python bench/adjudicate.py \
        --predictions outputs/predictions_qwen36_test_participant_zeroshot_logprob.jsonl \
        --data JF-ICR_test_participant.parquet \
        --quota uniform \
        --out outputs/adjudicate_qwen36.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quota_audit import LABELS, load_predictions, parse_quota
from solve import RESPONSE_MARKER, VALID_LABELS, call_model, extract_question

RUBRIC = """    "+2" : "Strong Commitment"
    "+1" : "Weak or Qualified Commitment"
     "0" : "Neutral or Hedged Intent"
    "-1" : "Weak Refusal"
    "-2" : "Strong Refusal\""""


def extract_response(query: str) -> str:
    """The company's reply text alone, out of a built prompt."""
    i = query.index(RESPONSE_MARKER) + len(RESPONSE_MARKER)
    tail = query[i:]
    for stop in ("\n\nDirectly output", "\nDirectly output"):
        if stop in tail:
            tail = tail[: tail.index(stop)]
    return tail.strip()


def build_prompt(items: list[dict], over: str, under: list[str], surplus: int, top_k: int) -> str:
    targets = " or ".join(f'"{u}"' for u in under)
    plural = "s" if surplus > 1 else ""
    head = (
        "You are a Japanese financial expert fluent in Japanese business communication.\n"
        "The intent labels are:\n" + RUBRIC + "\n\n"
        f"The following {len(items)} question-and-response pairs were all labelled "
        f'"{over}". Exactly {surplus} of them {"are" if surplus > 1 else "is"} mislabelled: '
        f"the true label of {'those' if surplus > 1 else 'that'} item{plural} is {targets}.\n\n"
    )
    blocks = []
    for n, it in enumerate(items, 1):
        blocks.append(f"[{n}]\nFinancial Question: {it['question']}\nCompany Response: {it['response']}\n")
    tail = (
        f"\nWhich item{plural} {'are' if surplus > 1 else 'is'} mislabelled? "
        f"Rank the {top_k} most likely, most likely first.\n"
        "Output only the item numbers, comma-separated. Do not explain.\n"
        "Answer:"
    )
    return head + "\n".join(blocks) + tail


def parse_ranking(text: str, n_items: int) -> list[int]:
    """Item numbers in the order named, deduplicated, in-range only."""
    seen, out = set(), []
    for tok in re.findall(r"\d+", text):
        v = int(tok)
        if 1 <= v <= n_items and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data", default="JF-ICR_test_participant.parquet")
    parser.add_argument("--quota", default="uniform")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="local")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64, help="raise well above 64 for a thinking run")
    parser.add_argument("--timeout", type=float, default=600, help="per-request seconds; thinking runs need more")
    parser.add_argument("--top-k", type=int, default=3, help="how many ranked candidates to ask for")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="ask this many times under different candidate orderings and vote -- guards against "
        "position bias, which a single forced-choice call cannot rule out",
    )
    parser.add_argument("--out", help="adjudication record (rounds, votes, flips) as JSON")
    parser.add_argument(
        "--write-predictions",
        help="copy of --predictions with the adjudicated flips applied, ready for make_submission.py; "
        "the pre-flip label is retained on each changed row as prediction_before_adjudication",
    )
    args = parser.parse_args()
    if args.write_predictions and Path(args.predictions).suffix == ".csv":
        sys.exit("--write-predictions needs a JSONL --predictions input, not a CSV")

    preds, probs = load_predictions(Path(args.predictions))
    quota = parse_quota(args.quota, len(preds))
    counts = Counter(preds.values())
    over = [lab for lab in LABELS if counts[lab] > quota[lab]]
    under = [lab for lab in LABELS if counts[lab] < quota[lab]]
    if not over:
        print("counts already meet the quota -- nothing flagged, nothing to adjudicate")
        return

    df = pd.read_parquet(args.data)
    text = {int(r.id): r.query for r in df.itertuples()}

    results = []
    for label in over:
        surplus = counts[label] - quota[label]
        ids = sorted(i for i, p in preds.items() if p == label)
        items = [
            {"id": i, "question": extract_question(text[i]), "response": extract_response(text[i])} for i in ids
        ]
        top_k = min(args.top_k, len(items))
        print(f"\nbucket {label!r}: {len(items)} candidates, surplus {surplus}, true label in {under}")

        rounds, first_choice = [], Counter()
        for rep in range(args.repeats):
            # rotate rather than shuffle: deterministic, reproducible without a
            # seed, and it moves every candidate through a different slot.
            order = items[rep:] + items[:rep] if rep else list(items)
            prompt = build_prompt(order, label, under, surplus, top_k)
            content, finish = call_model(
                args.endpoint, prompt, args.model, args.temperature, args.max_tokens, timeout=args.timeout
            )
            picked = [order[n - 1]["id"] for n in parse_ranking(content, len(order))]
            note = f"  (finish={finish})" if finish != "stop" else ""
            slot = order.index(next(o for o in order if o["id"] == picked[0])) + 1 if picked else None
            print(
                f"  round {rep + 1}: reply={content.strip()[:60]!r}{note}"
                + (f" -> top id={picked[0]} (slot {slot} of {len(order)})" if picked else " -> unparseable")
            )
            if picked:
                first_choice[picked[0]] += 1
            rounds.append({"rotation": rep, "ranking": picked, "raw": content, "finish": finish})

        if not first_choice:
            print("  no parseable item number in any round")
            results.append(
                {"bucket": label, "under": under, "surplus": surplus, "candidates": ids, "rounds": rounds, "ranking": []}
            )
            continue

        ranking = [i for i, _ in first_choice.most_common()]
        print(f"  vote over {args.repeats} ordering(s):")
        for rank, (i, votes) in enumerate(first_choice.most_common(top_k), 1):
            mark = "  <-- adjudicated" if rank <= surplus else ""
            p = f"   model p({label})={probs[i][label]:.3f}" if i in probs else ""
            print(f"    {rank}. id={i}  {votes}/{args.repeats} first-place{p}{mark}")
        results.append(
            {"bucket": label, "under": under, "surplus": surplus, "candidates": ids, "rounds": rounds, "ranking": ranking}
        )

    # Each adjudicated item moves out of its over-full bucket into an under-full
    # one. With a single under-full class the destination is forced; with several
    # it is not, so defer to the model's own ranking among just those classes.
    flips = {}
    for r in results:
        for i in r["ranking"][: r["surplus"]]:
            if len(r["under"]) == 1:
                flips[i] = r["under"][0]
            elif i in probs:
                flips[i] = max(r["under"], key=lambda lab: probs[i][lab])
            else:
                sys.exit(f"id={i} has {len(r['under'])} candidate destinations {r['under']} and no probabilities to rank them")

    print("\nadjudicated flips: " + (", ".join(f"{i}: {preds[i]} -> {lab}" for i, lab in flips.items()) or "none"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {"results": results, "flips": [{"id": i, "from": preds[i], "to": lab} for i, lab in flips.items()]}
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"wrote {args.out}")

    if args.write_predictions:
        out = Path(args.write_predictions)
        out.parent.mkdir(parents=True, exist_ok=True)
        kept = []
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if int(rec["id"]) in flips:
                rec["prediction_before_adjudication"] = rec["prediction"]
                rec["prediction"] = flips[int(rec["id"])]
            kept.append(json.dumps(rec, ensure_ascii=False))
        out.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"wrote {out} -- {len(kept)} rows, {len(flips)} adjudicated")


if __name__ == "__main__":
    main()
