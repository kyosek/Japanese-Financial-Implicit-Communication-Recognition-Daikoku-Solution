"""Summarise accuracy across multiple evaluate.py report.json files.

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
    baselines = {}
    for item in args.runs:
        label, path = item.split("=", 1)
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append(
            (
                label,
                report["n"],
                report["correct"],
                report["accuracy"],
                report.get("macro_f1"),
                report.get("qwk"),
                report["unparsed"],
            )
        )
        # Reports scored on the same set share a baseline; keyed by n so runs
        # over the public set and the participant set don't get conflated.
        if report.get("majority_baseline"):
            baselines[report["n"]] = report["majority_baseline"]

    # Sort by QWK when every report has it -- ranking by raw accuracy on a set
    # where +1 is 55% of the labels mostly ranks prior-following.
    if all(r[5] is not None for r in rows):
        rows.sort(key=lambda r: r[5], reverse=True)
    else:
        rows.sort(key=lambda r: r[3], reverse=True)

    name_w = max([len(r[0]) for r in rows] + [len("always-'+1' baseline")])
    header = f"{'model / setting':<{name_w}}  {'n':>4}  {'correct':>7}  {'accuracy':>8}  {'macro_f1':>8}  {'qwk':>8}  {'unparsed':>8}"
    print(header)
    for label, n, correct, accuracy, macro_f1, qwk, unparsed in rows:
        f1_str = f"{macro_f1:.4f}" if macro_f1 is not None else "n/a"
        qwk_str = f"{qwk:.4f}" if qwk is not None else "n/a"
        print(f"{label:<{name_w}}  {n:>4}  {correct:>7}  {accuracy:>8.4f}  {f1_str:>8}  {qwk_str:>8}  {unparsed:>8}")

    for n, baseline in sorted(baselines.items()):
        name = f"always-'{baseline['label']}' baseline"
        print(
            f"{name:<{name_w}}  {n:>4}  {'-':>7}  {baseline['accuracy']:>8.4f}  {'-':>8}  {baseline['qwk']:>8.4f}  {'-':>8}"
        )


if __name__ == "__main__":
    main()
