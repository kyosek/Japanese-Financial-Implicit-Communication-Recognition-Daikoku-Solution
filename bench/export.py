"""Export a run's predictions + metrics to CSV and Markdown.

The JSONL that solve.py writes is keyed by row id only, so reading it means
cross-referencing the dataset by hand. This joins the two: one CSV row per
item with the analyst question and company response inline, next to gold,
prediction, and whether the model got it right -- which is the form you want
for eyeballing what a run actually got wrong.

Alongside it writes a Markdown summary (headline metrics + confusion matrix +
the error list) suitable for pasting into notes or a paper appendix.

Usage:
    python bench/export.py \
        --predictions outputs/predictions_qwen36_public_zeroshot_thinking.jsonl \
        --report outputs/report_qwen36_public_zeroshot_thinking.json \
        --data JF-ICR_public_set.parquet \
        --out outputs/export_qwen36_thinking_public
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve import QUESTION_MARKER, REMINDER_MARKER, RESPONSE_MARKER, VALID_LABELS


def extract_qa(query: str) -> tuple[str, str]:
    """The analyst question and the company response, out of a built prompt."""
    i = query.index(QUESTION_MARKER) + len(QUESTION_MARKER)
    j = query.index(RESPONSE_MARKER)
    k = j + len(RESPONSE_MARKER)
    m = query.index(REMINDER_MARKER)
    return query[i:j].strip(), query[k:m].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--data", default="JF-ICR_public_set.parquet")
    parser.add_argument("--out", required=True, help="output path prefix; .csv and .md are appended")
    parser.add_argument("--label", default=None, help="run name for the Markdown heading (default: --out basename)")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line]
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    df = pd.read_parquet(args.data)
    qa = {int(r.id): extract_qa(r.query) for r in df.itertuples()}

    label = args.label or Path(args.out).name
    out_csv = Path(f"{args.out}.csv")
    out_md = Path(f"{args.out}.md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- per-item CSV -----------------------------------------------------
    fields = ["id", "gold", "prediction", "correct", "finish_reason", "elapsed_s", "question", "response"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            question, response = qa.get(r["id"], ("", ""))
            w.writerow(
                {
                    "id": r["id"],
                    "gold": r["gold"],
                    "prediction": r["prediction"],
                    "correct": int(r["prediction"] == r["gold"]),
                    # absent in runs predating the finish_reason capture
                    "finish_reason": r.get("finish_reason", ""),
                    "elapsed_s": r["elapsed_s"],
                    "question": question,
                    "response": response,
                }
            )

    # --- Markdown summary -------------------------------------------------
    errors = [r for r in rows if r["prediction"] != r["gold"]]
    lines = [
        f"# {label}",
        "",
        f"- n: {report['n']}",
        f"- correct: {report['correct']}",
        f"- accuracy: {report['accuracy']:.4f}",
        f"- macro F1: {report['macro_f1']:.4f}",
        f"- QWK: {report['qwk']:.4f}",
        f"- within one step: {report['within_one']:.4f}",
        f"- unparsed: {report['unparsed']}",
    ]
    truncated = sum(1 for r in rows if r.get("finish_reason") == "length")
    if truncated:
        lines.append(f"- **truncated (hit the token cap): {truncated}** -- these return empty content and score as unparsed")
    total_s = sum(r["elapsed_s"] for r in rows)
    lines += [
        f"- wall clock: {total_s / 60:.0f} min ({total_s / len(rows):.1f}s per item)",
        "",
        "## Per-label F1",
        "",
        "| label | F1 |",
        "| --- | --- |",
    ]
    lines += [f"| `{lab}` | {report['f1_per_label'][lab]:.4f} |" for lab in VALID_LABELS]
    lines += [
        "",
        "## Confusion matrix",
        "",
        "Rows are gold, columns are predicted.",
        "",
        "| gold \\ pred | " + " | ".join(f"`{p}`" for p in VALID_LABELS) + " |",
        "| --- " * (len(VALID_LABELS) + 1) + "|",
    ]
    for g in VALID_LABELS:
        cells = [str(report["confusion"].get(f"{g}|{p}", 0)) for p in VALID_LABELS]
        lines.append(f"| `{g}` | " + " | ".join(cells) + " |")

    lines += ["", f"## Errors ({len(errors)})", "", "| id | gold | pred | question |", "| --- | --- | --- | --- |"]
    for r in errors:
        question = qa.get(r["id"], ("", ""))[0].replace("|", "\\|").replace("\n", " ")
        if len(question) > 120:
            question = question[:117] + "..."
        lines.append(f"| {r['id']} | `{r['gold']}` | `{r['prediction']}` | {question} |")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_csv} ({len(rows)} rows) and {out_md}")


if __name__ == "__main__":
    main()
