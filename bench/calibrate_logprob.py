"""Post-hoc prior correction for verbaliser log-prob predictions.

Two label-free calibration methods on solve_logprob.py's output -- both pure
post-processing of the stored P(y|x), no extra LLM calls, no gold labels used:

--method batch (default): Batch calibration (Zhou et al. 2023). Estimates the
model's contextual prior as the mean predicted distribution across the test
set itself, then divides each row's P(y|x) by that estimated marginal and
renormalises over the 5 labels:

    p_calibrated(y|x) proportional to p(y|x) / q(y),   q(y) = mean_x p(y|x)

--method sld-em: Saerens-Latinne-Decock (2002) EM prior adaptation. Treats
the raw P(y|x) as calibrated under an assumed prior pi_train(y) (--train-prior,
default: uniform over the 5 labels -- the standard simplifying assumption when
no canonical content-free baseline is available) and alternates:

    E-step: p_test(y|x) proportional to [pi_test(y) / pi_train(y)] * p_train(y|x), renormalised
    M-step: pi_test(y) = mean_x p_test(y|x)

starting from pi_test = pi_train, until pi_test stops moving. Unlike batch
calibration's one-shot division, this iterates the correction to a
self-consistent fixed point.

Both target the same failure mode: a model over- or under-predicting one
label irrespective of instance content.

SLD-EM instability on rare classes: if a class's raw posterior mass is
already small and skewed, the M-step can drive its estimated pi_test towards
0. Once at (or near) 0, the E-step's pi_test(y)/pi_train(y) ratio zeroes that
label out of every row's adjusted posterior, which zeroes the next M-step
estimate too -- an absorbing state a class can fall into but never leave.
Confirmed on this dataset: gold `-1` is 11/253 rows (4.3%) and gemma4's raw
posteriors already gave it little mass; unconstrained SLD-EM drove
pi_test(-1) to 0.000, and the resulting predictions never picked `-1` even
where a row's raw posterior favoured it. Two standard stabilisers, both
applied after each M-step, before the next E-step:

- --prior-floor F: clamp every class's estimated share to at least F, then
  renormalise. Keeps pi_test(y)/pi_train(y) bounded away from 0, so a rare
  class stays reachable in every row's adjusted posterior instead of being
  multiplied out entirely.
- --damping D: blend the new M-step estimate with the previous iterate
  (pi_test_new <- D * pi_test_old + (1-D) * pi_test_new), D in [0, 1). Slows
  the walk towards any fixed point, including a collapsing one -- on its own
  it delays collapse rather than preventing it, so pair it with a floor
  rather than relying on damping alone.

Usage:
    python bench/calibrate_logprob.py --method batch \
        --predictions outputs/predictions_gemma4_zeroshot_logprob.jsonl \
        --out outputs/predictions_gemma4_zeroshot_logprob_calibrated.jsonl

    python bench/calibrate_logprob.py --method sld-em \
        --predictions outputs/predictions_gemma4_zeroshot_logprob.jsonl \
        --out outputs/predictions_gemma4_zeroshot_logprob_sldem.jsonl
"""

import argparse
import json
from pathlib import Path

LABELS = ["+2", "+1", "0", "-1", "-2"]
LABEL_SCORE = {"+2": 2, "+1": 1, "0": 0, "-1": -1, "-2": -2}


def batch_prior(records: list[dict]) -> dict[str, float]:
    sums = {label: 0.0 for label in LABELS}
    n = 0
    for r in records:
        if r["label_prob"] is None:
            continue
        for label in LABELS:
            sums[label] += r["label_prob"][label]
        n += 1
    return {label: sums[label] / n for label in LABELS}


def reweight(label_prob: dict[str, float], ratio: dict[str, float]) -> dict[str, float]:
    adjusted = {label: label_prob[label] * ratio[label] for label in LABELS}
    z = sum(adjusted.values())
    if z == 0:
        return {label: 1 / len(LABELS) for label in LABELS}
    return {label: v / z for label, v in adjusted.items()}


def batch_calibrate(records: list[dict]) -> tuple[list[dict | None], dict]:
    prior = batch_prior(records)
    ratio = {label: 1 / prior[label] for label in LABELS}
    calibrated = [reweight(r["label_prob"], ratio) if r["label_prob"] is not None else None for r in records]
    return calibrated, {"q": prior}


def clamp_and_renormalise(prior: dict[str, float], floor: float) -> dict[str, float]:
    if floor <= 0:
        return prior
    clamped = {label: max(v, floor) for label, v in prior.items()}
    z = sum(clamped.values())
    return {label: v / z for label, v in clamped.items()}


def sld_em(
    records: list[dict],
    train_prior: dict[str, float],
    max_iter: int,
    tol: float,
    prior_floor: float = 0.0,
    damping: float = 0.0,
) -> tuple[list[dict | None], dict]:
    valid_idx = [i for i, r in enumerate(records) if r["label_prob"] is not None]
    test_prior = clamp_and_renormalise(dict(train_prior), prior_floor)
    adjusted = [None] * len(records)

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        ratio = {label: test_prior[label] / train_prior[label] for label in LABELS}
        for i in valid_idx:
            adjusted[i] = reweight(records[i]["label_prob"], ratio)
        new_prior = {label: sum(adjusted[i][label] for i in valid_idx) / len(valid_idx) for label in LABELS}
        if damping > 0:
            new_prior = {label: damping * test_prior[label] + (1 - damping) * new_prior[label] for label in LABELS}
        new_prior = clamp_and_renormalise(new_prior, prior_floor)
        delta = max(abs(new_prior[label] - test_prior[label]) for label in LABELS)
        test_prior = new_prior
        if delta < tol:
            break

    return adjusted, {"train_prior": train_prior, "test_prior": test_prior, "n_iter": n_iter}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="predictions_*_logprob.jsonl from solve_logprob.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["batch", "sld-em"], default="batch")
    parser.add_argument(
        "--train-prior",
        choices=["uniform"],
        default="uniform",
        help="assumed prior the raw posteriors were estimated under (sld-em only)",
    )
    parser.add_argument("--max-iter", type=int, default=50, help="sld-em only")
    parser.add_argument("--tol", type=float, default=1e-6, help="sld-em convergence tolerance on max prior delta")
    parser.add_argument(
        "--prior-floor",
        type=float,
        default=0.01,
        help="sld-em only: minimum share for any class's estimated prior after each M-step, then renormalised "
        "-- prevents a rare class's prior from being absorbed at exactly 0 (0 disables, reproducing unconstrained SLD-EM)",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.0,
        help="sld-em only: blend weight in [0, 1) towards the previous iterate's prior each M-step "
        "(0 = standard EM; use alongside --prior-floor, not as a substitute for it)",
    )
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.predictions).open(encoding="utf-8")]

    if args.method == "batch":
        calibrated, info = batch_calibrate(records)
        print("estimated batch prior q(y) = mean predicted P(y) across the test set:")
        for label in LABELS:
            print(f"  {label}: {info['q'][label]:.4f}")
    else:
        train_prior = {label: 1 / len(LABELS) for label in LABELS}  # uniform
        calibrated, info = sld_em(
            records, train_prior, args.max_iter, args.tol, prior_floor=args.prior_floor, damping=args.damping
        )
        print(
            f"SLD-EM converged after {info['n_iter']} iterations "
            f"(train prior: uniform, prior-floor={args.prior_floor}, damping={args.damping})"
        )
        print("estimated test prior pi_test(y):")
        for label in LABELS:
            print(f"  {label}: {info['test_prior'][label]:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_changed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r, calibrated_prob in zip(records, calibrated):
            if calibrated_prob is None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
            new_pred = max(calibrated_prob, key=calibrated_prob.get)
            n_changed += int(new_pred != r["prediction"])
            out = dict(r)
            out["label_prob"] = calibrated_prob
            out["prediction"] = new_pred
            out["expected_score"] = sum(LABEL_SCORE[label] * p for label, p in calibrated_prob.items())
            out["raw_prediction"] = r["prediction"]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\n{n_changed}/{len(records)} predictions changed by calibration")
    print(f"wrote {len(records)} calibrated predictions to {out_path}")


if __name__ == "__main__":
    main()
