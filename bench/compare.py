"""Summarize accuracy across multiple evaluate.py report.json files.

Usage:
    python bench/compare.py \
        "shisa-v2-32b zero-shot"=outputs/report_shisa_zeroshot.json \
        "shisa-v2-32b few-shot"=outputs/report_shisa_fewshot.json \
        ...
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help='"label"=path/to/report.json')
    args = parser.parse_args()

    rows = []
    for item in args.runs:
        label, path = item.split("=", 1)
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append(
            (label, report["n"], report["correct"], report["accuracy"], report.get("macro_f1"), report["unparsed"])
        )

    rows.sort(key=lambda r: r[3], reverse=True)

    name_w = max(len(r[0]) for r in rows)
    print(f"{'model / setting':<{name_w}}  {'n':>4}  {'correct':>7}  {'accuracy':>8}  {'macro_f1':>8}  {'unparsed':>8}")
    for label, n, correct, accuracy, macro_f1, unparsed in rows:
        f1_str = f"{macro_f1:.4f}" if macro_f1 is not None else "n/a"
        print(f"{label:<{name_w}}  {n:>4}  {correct:>7}  {accuracy:>8.4f}  {f1_str:>8}  {unparsed:>8}")


if __name__ == "__main__":
    main()
