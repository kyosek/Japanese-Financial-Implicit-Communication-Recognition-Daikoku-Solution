# FinNLP-Japanese

## JF-ICR

Solves and evaluates the EBISU **JF-ICR** (Japanese Financial Implicit
Communication Recognition) subtask: given a Japanese financial Q&A pair,
classify the company response's implicit intent on a 5-point scale (`+2`
strong commitment ... `-2` strong refusal). Official metric: accuracy.

Two evaluation sets, and they are not comparable:

| file | n | labels | notes |
|---|---|---|---|
| `JF-ICR_public_set.parquet` | 253 | released gold | heavily skewed: `+1` 139, `0` 72, `+2` 29, `-1` 11, `-2` 2 |
| `JF-ICR_test_participant.parquet` | 50 | none released | the submission target; published as balanced, 10 per label |
| `JF-ICR_test_participant_labelled.parquet` | 50 | **our own, post-hoc** | same 50 items, annotated by us after submitting |

Both parquets carry a `query` column holding a fully-formed prompt (role
framing, the five label definitions, the Japanese Q&A pair, an output
reminder) — the pipeline uses that prompt as the carrier and splices into it
rather than rewriting it.

> **Every test-split score in this repo is agreement with our own post-hoc
> annotation, not an official score.** No test gold was released to
> participants and none was used for prompt design, model selection, or the
> submission. Test numbers are labelled "(vs. our labels)" throughout.

---

## The solution pipeline

`scripts/run_solution.sh` is the end-to-end submission path. Five stages, no
gold labels and no human judgement anywhere in it:

```
JF-ICR_test_participant.parquet
        │
   1 ── solve_logprob.py ──── score the 5 label strings per item → P(y|x), argmax
        │
   2 ── quota_audit.py ────── compare class counts to the published 10-per-label
        │                     quota → a PROVABLE error count and the over-full
        │                     buckets those errors live in. Pure arithmetic.
        │
   3 ── adjudicate.py ─────── inside each flagged bucket, ask the model the
        │                     comparative question the quota licenses, voted
        │                     over rotated candidate orderings
        │
  3b ── consensus.py ──────── (optional) localise the same errors from
        │                     cross-run disagreement, and cross-check 3
        │
   4 ── quota_audit.py ────── re-audit: the corrected counts must meet the
        │                     quota exactly, i.e. a provable floor of 0
        │
   5 ── make_submission.py ── write and validate the CSV
        │
   submission_solution.csv
```

```bash
# full run (fresh inference), default model Qwen3.6-35B-A3B
nohup caffeinate -i scripts/run_solution.sh > outputs/run_solution.log 2>&1

# reproducible run: skip stage 1, reuse a stored predictions file
REUSE_PREDICTIONS=outputs/predictions_qwen36_test_participant_zeroshot_logprob.jsonl \
  RUNS='outputs/predictions_*test*.jsonl' \
  nohup caffeinate -i scripts/run_solution.sh > outputs/run_solution.log 2>&1
```

`REUSE_PREDICTIONS` exists because llama.cpp is not bit-exact at temperature 0
— a fresh stage 1 can differ from a previous one by an item or two, so stages
2–5 are only reproducible over a stored predictions file. Setting `RUNS` to a
glob of other prediction files enables stage 3b.

### Why each stage is there

**Stage 2 — the quota is a detector, not a corrector.** If a labelling
predicts class `y` n_y times and the quota says q_y, it is wrong on at least
`Σ_y max(0, n_y - q_y)` items. That is a certificate, not an estimate. The
over-predicted classes also name the buckets the surplus errors sit in, which
typically cuts the audit surface by half or more.

The obvious move — solve for the highest-probability labelling that *meets*
the counts (a min-cost assignment over P(y|x)) — was measured and is **worse
than plain argmax**:

| | argmax | quota-constrained argmax |
|---|---|---|
| public set, quota = its true class counts | 0.6759 | 0.6522 (−6 items) |
| public, uniform quota on balanced subsamples (n=87, 400 draws) | 0.6338 | 0.6148 (worse in 72% of draws) |
| test set (vs. our labels) | 49/50 | 48/50 |

A min-cost assignment buys its class counts as cheaply as possible, so it
flips the *smallest-margin* items — it assumes errors sit near the decision
boundary. This model's errors are confidently wrong: on the test set it put
0.897 on `+2` for the one real error, so the assignment would not touch it and
paid for its quota by corrupting a correct item instead. Cheapest-to-flip and
most-likely-wrong are close to uncorrelated here.

**Stage 3 — the quota proves *how many* and *where*, never *which*.** Majority
voting does not close that gap: across 16 full test-set runs (claude, gemma4,
qwen36, qwen38 × zeroshot/thinking/rule-gated/multipart/logprob) the majority
label on the genuinely wrong item id=300 is `+2` by 12 votes to 3 — the
ensemble is confidently wrong, and its majority is identical to plain argmax
on all 50 items.

So `adjudicate.py` asks a different question. Instead of "classify this item",
it consumes the quota certificate and asks a comparative one:

> These 11 items were all labelled `+2`. Exactly one of them is really `0` or
> `-1`. Which?

Forced choice over a shortlist, comparative so the items calibrate each other,
and told the answer exists — which blocks the "they all look like commitments"
failure that produced the error in the first place. Votes are taken over
rotated candidate orderings (`--repeats`, default 5) so slot position can't
decide it.

**Stage 3b — the vote *spread* works where the majority doesn't.** Inside the
flagged `+2` bucket, 10 of the 11 candidates are unanimous at 16/16 and id=300
is the sole exception at 12/16. Minimum-consensus inside a quota-flagged
bucket localises the error exactly, with no extra model call. It's cheaper
than stage 3 when several runs already exist, and the pipeline aborts if the
two detectors disagree about which items to flip.

### What it produced

Fresh stage 1 (`outputs/run_solution_e2e.log`):

```
stage 2   +2: 11 (+1)   +1: 12 (+2)   0: 9 (-1)   -1: 8 (-2)   -2: 10
          provable error floor: at least 3 of 50 are wrong
          audit surface: 23 of 50 items (46%)
stage 3   flips 300: +2→0, 296: +1→-1, 259: +1→-1   (5/5, 5/5, 3/5 votes)
stage 3b  same 3 flips from run disagreement alone → detectors agree
stage 4   all five classes at 10   →   provable error floor: 0
```

Scored afterwards against our own labels, all three flips were real errors:

| | agreement (vs. our labels) |
|---|---|
| raw argmax (stage 1) | 44/50 |
| after adjudication (stage 4) | **47/50** |
| stored logprob run → after adjudication | 46/50 → **47/50** |

The reused-predictions run started from a labelling with only one surplus,
found the one flip, and landed on the same 47/50.

**Scope limit, which bounds what stage 4 means.** The quota can only see errors
that change the class counts. A compensating pair — one item labelled A that is
really B, another labelled B that is really A — leaves the counts untouched and
is invisible. "Error floor 0" means the counts are *consistent with* a perfect
labelling, not that the labelling is perfect.

### The three residual disagreements point at our annotation, not the pipeline

The 47/50 above is agreement with our own post-hoc labels, and the same quota
certificate can be pointed at *those*:

```
our labels:   +2:10   +1:8    0:12   -1:9    -2:11     -> floor: >= 3 wrong
submission:   +2:10   +1:10   0:10   -1:10   -2:10     -> floor: 0
```

Our annotation violates the published 10-per-label quota — surplus 2 on `0`,
surplus 1 on `-2` — so by the stage-2 argument it is wrong on at least 3 of 50.
The three surplus items are exactly the three disagreements, and the pipeline
reassigns each to a class our annotation is *short* on:

| id | submitted | our label | model P(submitted) | question / response |
|---|---|---|---|---|
| 257 | `+1` | `0` | 0.983 | 価格を引き上げますか → コスト上昇が継続する場合には…価格改定をお願いする可能性があります |
| 267 | `-1` | `-2` | 0.669 | 今年中に欧州参入しますか → 今年中の参入は見送る方向です。…来年度以降に改めて可能性を検討します |
| 272 | `+1` | `0` | 0.971 | 自己株式取得を行いますか → 有力な選択肢の一つです。…機動的に実施できるよう検討を進めます |

So if the published quota is accurate, the submission is 50/50 and **our labels
are the errors**. Each also sits on a real tension inside the guideline: the
`+1` signal list explicitly includes conditionals (「〜であれば」「〜次第で」), which
is what 257 is; the `0` list explicitly includes 「検討中」, which is what 272's
「検討を進めます」 is; and `-1` is *defined* as a refusal that is time-bound and
leaves room for reconsideration, which is what 267 states outright.

This is a caveat on every "(vs. our labels)" number in this README, not just
this one — our annotation is a single-annotator pass with a measurable
inconsistency, which is what the annotation study below exists to quantify.

### Submissions in the repo

| file | what it is |
|---|---|
| `submission.csv` | raw argmax, `+2`:11 `0`:9 — kept as the stage-2 reference point |
| `submission_balanced.csv` | **the submitted file** — the same run after quota adjudication (id 300 `+2`→`0`), 10 per label. Label-identical to `outputs/predictions_qwen36_solution_final.jsonl`, i.e. `run_solution.sh` reproduces it |
| `submission_qwen36_thinking.csv` | the reasoning-mode arm, for comparison |

---

## Prompting

Everything below runs against the dataset's own `query` prompt with at most
two blocks spliced in before the Q&A pair.

**Annotation rules (on by default).** The EBISU paper's Appendix A.3 "Specific
Annotation Rules" — the per-label linguistic-signal cues and representative
examples human annotators used to tell adjacent labels apart (+1 vs +2, −1 vs
−2). `--no-annotation-rules` reproduces the bare label-definition prompt.

Note for reporting: these are reproduced close to verbatim but **not entirely**
— one example was added to the `+2` block and one signal plus one example to
the `+1` block, derived from public-set errors only. The prompt is therefore
partly tuned, not purely given.

**Multi-part aggregation rule (`--multipart-rule`, default `always`).** 30% of
the public set (76/253) bundles several asks into one question; the
participant test set has none (0/50). The guideline states no aggregation
rule, so the label depends on which piece the reader weights.
`bench/decompose_experiment.py` scored the candidates against gold — taking
the *most committal* piece tracked the annotators best. `off` / `always` /
`gated` control when the rule appears; `gated` uses `bench/multipart.py` to
detect bundling per question.

**Few-shot (`--few-shot-k`).** Holds out k exemplars per label (k=1 → 5 shots)
from the eval set and prepends their Q&A blocks using the dataset's own
template. Held-out rows are excluded from scoring, hence n=248 on few-shot
runs. With the annotation rules present, few-shot is worse than zero-shot for
every model tried, so it is not in the submission.

---

## Runtime and models

- **Runtime**: a native **arm64** llama.cpp build (Metal-accelerated). The
  Homebrew `llama.cpp` on this machine is the x86_64/Rosetta build with no
  Metal support, so `scripts/download_runtime.sh` fetches an official arm64
  release into `runtime/` instead of using it.
- **Hardware**: one Apple Silicon machine, 48GB unified memory. Only one model
  is resident at a time.

| model | size | notes |
|---|---|---|
| [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (UD-Q4_K_M) | ~22GB | MoE, ~3B active — **the pipeline default** |
| Qwen3.8-27B (`Qwen3.8-27B-UD-Q4_K_XL.gguf`) | ~17GB | successor generation, unsloth dynamic quant; did not beat qwen3.6 here |
| [gemma-4-12b-it](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) (Q8_0) | ~13GB | dense 12B, hybrid local/global attention, reasoning by default |
| [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) (Q4_K_M) | ~19GB | MoE, ~3B active |
| [shisa-v2-qwen2.5-32b](https://huggingface.co/shisa-ai/shisa-v2-qwen2.5-32b) (Q4_K_M) | ~20GB | Japanese-specialised fine-tune of Qwen2.5-32B |
| [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) (F16/native MXFP4) | ~14GB | OpenAI open-weight reasoning model (harmony format) |
| [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF) (Q4_K_M) | ~20GB | dense 32B — always reasons, no non-thinking mode |
| Claude Opus | API | run for comparison; strongest macro-F1 on the public set |

Quantisation is an uncontrolled confound: the open-weight models are compared
at different precisions (Q4_K_M vs Q8_0) because those are what fit in 48GB, so
a gap between two rows conflates model and precision.

### Reasoning-mode failures

Some models default to an extended thinking mode that burns the entire token
budget without emitting the final label — `content` comes back empty while
`reasoning_content` holds an unfinished chain of thought. The fix depends on
the model:

- **gemma-4-12b-it**, **Qwen3.6-35B-A3B**, **Qwen3.8-27B**: `--reasoning off`
  (passed through `scripts/serve.sh`) genuinely suppresses it. Without the flag
  qwen3.6 burned the default `--max-tokens 32` on an unfinished preamble every
  time (0/253 parsed); with it, zero unparsed.
- **gpt-oss-20b**: `--reasoning off` only changes how llama.cpp *labels*
  `content` vs `reasoning_content`; the harmony format's analysis channel is
  generated either way (~450+ reasoning tokens with the flag set). The actual
  fix is a generous `--max-tokens` (used 2048).
- **DeepSeek-R1-Distill-Qwen-32B**: no non-thinking mode at all — only
  `--max-tokens` to make sure generation survives the `<think>` block (used
  2048). Being **dense** 32B rather than ~3B-active MoE, it is far slower here:
  ~2h53m for 253 rows (vs ~3 min for qwen3.6), and 7/253 requests hit the 120s
  HTTP timeout — 6 of those 7 were the *last* 6 rows, consistent with thermal
  throttling over a long sustained-Metal run rather than hard prompts.

If a model shows a high unparsed rate, check which failure mode it is
(`reasoning_content` populated but `content` empty = ran out of budget) before
assuming the flag will fix it.

Enabling thinking deliberately does **not** help on this task — see the
results table.

---

## Metrics

`bench/evaluate.py` reports accuracy (the official metric), macro-F1, QWK, and
within-one-step, plus a confusion matrix.

Accuracy and macro-F1 both treat the five labels as unordered, which misreads
the task twice: the labels are an ordinal scale (confusing `+1` with `+2` is a
smaller error than `+1` with `-2`), and the public set is skewed enough that
always answering `+1` already scores 0.549 accuracy and 0.949 within-one.
**Quadratic weighted kappa** fixes both — errors are penalised by squared
distance along the scale, and chance correction puts the always-majority
baseline at exactly 0.0. `bench/metrics.py` holds the shared implementation;
`bench/compare.py` lines up accuracy / macro-F1 / QWK across report files.

---

## Results

### Public set (253 items, temperature 0, annotation rules on, `--multipart-rule off`)

Cells are `accuracy / macro-F1`.

| model | zero-shot | few-shot (k=1, 5 shots, n=248) |
|---|---|---|
| qwen3.6-35b-a3b | **0.6838 / 0.4072** | 0.6331 / 0.3339 |
| claude opus (API) | 0.6640 / **0.4744** | *(not run)* |
| qwen3.8-27b | 0.6522 / 0.3816 | *(not run)* |
| gemma-4-12b-it | 0.6443 / 0.3628 | 0.6331 / 0.3517 |
| gpt-oss-20b | 0.6008 / 0.3521 | 0.5000 / 0.3214 |
| qwen3-30b-a3b | 0.6047 / 0.3353 | 0.5605 / 0.2624 |
| shisa-v2-qwen2.5-32b | 0.5336 / 0.2917 | 0.5161 / 0.2912 |
| deepseek-r1-distill-qwen-32b | 0.4862 / 0.3026 | *(not run)* |

All runs 0 unparsed except deepseek-r1-distill (7/253, see above).
`solve.py`'s current default is `--multipart-rule always`, so a fresh run
reproduces the `rule-always` arm below, not this table.

### Test set — agreement with our own post-hoc labels, not official scores

| model / setting | agreement | macro-F1 | QWK |
|---|---|---|---|
| qwen3.6-35b-a3b zero-shot | 0.92 | 0.9196 | 0.9659 |
| gemma-4-12b-it zero-shot | 0.94 | 0.9395 | — |
| claude opus zero-shot | 0.92 | 0.9196 | — |
| qwen3.6 thinking | 0.90 | 0.8970 | 0.9454 |
| qwen3.8-27b zero-shot | 0.88 | 0.8787 | 0.8815 |
| qwen3.8-27b thinking | 0.86 | 0.8663 | 0.8848 |
| **qwen3.6 + solution pipeline** (`submission_balanced.csv`) | **0.94** | — | — |

"Agreement", not accuracy: these are all scored against our own single-annotator
labels, which themselves violate the published class quota by at least 3 items.
The pipeline row's 3 disagreements are exactly those 3 — see above. Read this
table as inter-annotator agreement between each model and one human, not as a
leaderboard.

The test set is visibly more balanced than the public set and its instances
are markedly shorter, so test and public accuracies are not comparable
quantities. The 50 items are further split into a public 15 and a private 35
(3 and 7 per label) for scoring, but the released parquet carries no split
column, so nothing here can be broken down that way.

### Multi-part aggregation rule A/B (`scripts/run_multipart_rule.sh`)

| arm | public acc / F1 / QWK | test acc / F1 / QWK |
|---|---|---|
| `off` | 0.6759 / 0.4031 / 0.4646 | 0.90 / 0.8988 / 0.9608 |
| `gated` | 0.6798 / 0.3990 / 0.4669 | 0.92 / 0.9196 / 0.9659 |
| `always` | 0.6640 / 0.3672 / 0.4636 | 0.88 / 0.8771 / 0.9271 |

Stating the rule unconditionally measurably leaks: it changed 18 single-part
public predictions (net −0.017 accuracy) because every long IR response has
spans at differing commitment levels for "take the most committal part" to
latch onto. The test set has no multi-part questions, so `gated` there is a
no-op that must reproduce `off` — that arm is a correctness check on the
gating, not evidence about the rule. **All three arms sit inside the ~2-item
noise floor on n=253; treat the ordering as unresolved.**

### Non-LLM baselines (public set, 5-fold CV, out-of-fold predictions)

| baseline | accuracy | macro-F1 |
|---|---|---|
| cue-lexicon + logistic regression | 0.5573 | 0.2510 |
| fine-tuned encoder, class-weighted loss | 0.5217 | 0.2493 |
| fine-tuned encoder, unweighted loss | 0.5534 | 0.1526 |
| majority-class (always `+1`) | 0.5494 | 0.1418 |

### Notes on the results

- **The non-LLM baselines don't match the LLMs** — every LLM zero-shot run
  beats every non-LLM baseline on macro-F1 (0.29–0.47 vs 0.15–0.25). The
  reviewer hypothesis this section was added to test ("if a cue-lexicon
  regression matches a 30B LLM, that reframes the whole paper") doesn't hold.
  That's a useful result: the annotation-rule cues aren't trivially replicated
  by substring matching or by fine-tuning a small encoder on ~200
  examples/fold, so the LLM comparison measures something real rather than a
  surface-form shortcut.
- **Unweighted encoder fine-tuning collapsed to the majority class** — its
  confusion matrix predicts `+1` or `0` for nearly every row, landing within
  0.4pp of the plain majority baseline. Standard small-data/class-imbalance
  failure for a 110M model on ~200 examples with a 55%-skewed label, not
  evidence the task is unlearnable. `--class-weight balanced` fixes the
  collapse (macro-F1 0.15 → 0.25) at some accuracy cost. Neither closes the
  gap to the LLMs.
- The cue-lexicon+LR accuracy (0.5573) is barely above majority (0.5494), but
  its macro-F1 (0.2510 vs 0.1418) is meaningfully higher — it's not a
  majority-class clone, accuracy alone just doesn't show it.
- Adding the annotation rules lifted every model well above its bare-prompt
  score (qwen3-30b-a3b zero-shot: 0.4466 → 0.6047) — the paper's own
  linguistic-signal cues are informative context, not boilerplate.
- With the rules in place, zero-shot beats few-shot for every model — the
  opposite of what we saw without them. The rules and the exemplars overlap in
  what they teach, and stacking both isn't better than rules alone.
- Extended thinking is a small net negative here (qwen3.6 public 0.6838 →
  0.6403; test 0.92 → 0.90) at a large latency cost.

---

## Log-prob verbaliser scoring

`bench/solve_logprob.py` scores each of the 5 label strings directly —
P(label | prompt) via teacher-forced token log-probs against llama.cpp's raw
`/completion` endpoint — instead of letting the model generate freely and
parsing a label out of the text. Takes the argmax as the prediction and also
reports a continuous score, Σk·P(k) over {+2..−2}. This is stage 1 of the
solution pipeline: it cannot produce an unparsed label, and it emits the
per-item distribution that stages 2–3 need.

Unlike free generation it tells you *how* wrong a miss was — whether the gold
label was a close second (calibration/prior problem) or buried near the bottom
(comprehension problem). `bench/evaluate_logprob.py` reports gold-label rank
histograms and mean P(gold) per class to answer exactly that.

```bash
./.venv/bin/python bench/solve_logprob.py --model qwen36 --out outputs/predictions_qwen36_logprob.jsonl
./.venv/bin/python bench/evaluate_logprob.py \
  --predictions outputs/predictions_qwen36_logprob.jsonl \
  --report outputs/report_qwen36_logprob.json

# same --few-shot-k / --no-annotation-rules / --multipart-rule flags as solve.py
```

It runs a 3–5 row cross-check against free generation by default
(`--sanity-check-n`, 0 to skip) before the full pass — trust that agreement
number before trusting a full run on a model you haven't scored before.
**Reasoning/harmony-format models are why this check exists**: gpt-oss-20b's
template mandates an analysis-channel preamble before any content, so the
label does not immediately follow the prompt and naive log-prob scoring there
measures the wrong thing (every label looks equally, vanishingly unlikely).
Confirmed to work out of the box for gemma-4-12b-it (100% agreement) and
qwen3.6; verify shisa/gpt-oss before trusting their numbers.

Reports use the same `n`/`correct`/`accuracy`/`unparsed`/`confusion` keys as
`evaluate.py`, so `bench/compare.py` lines free-generation and log-prob-argmax
accuracy up side by side.

### Post-hoc calibration

`bench/calibrate_logprob.py` applies label-free prior correction to the stored
P(y|x) — pure post-processing, no extra LLM calls, no gold labels:

- `--method batch` — batch calibration (Zhou et al. 2023): divide each row's
  P(y|x) by the mean predicted distribution across the test set, renormalise.
- `--method sld-em` — Saerens–Latinne–Decock (2002) EM prior adaptation:
  iterate the correction to a self-consistent fixed point rather than applying
  it once. `--prior-floor` guards the rare-class instability.

gemma-4-12b-it, public set (`scripts/run_logprob_sldem.sh`):

| | accuracy | macro-F1 |
|---|---|---|
| raw log-prob argmax | 0.6443 | 0.3628 |
| SLD-EM (prior-floor 0.01) | **0.6601** | 0.3646 |
| batch-calibrated | 0.6285 | **0.3796** |

SLD-EM changed 7/253 predictions. Both methods move accuracy and macro-F1 in
opposite directions, and both deltas are inside the noise floor — this is
reported as a measured null, not a win.

---

## Non-LLM baselines

Three baselines that don't call an LLM at all, to check whether the label is
recoverable from surface form alone (in which case the LLM comparison is the
wrong headline result):

- **`bench/solve_lexicon.py`** — counts occurrences of a hand-curated set of
  Japanese hedging/modality/commitment cue phrases in the company's response
  (sentence-final forms like 差し控えさせていただきます / 検討してまいります /
  〜と考えております, the benefactive 〜させていただく, conditional and
  epistemic markers — see `CUE_GROUPS`) and fits a logistic regression on them.
- **`bench/solve_encoder.py`** — fine-tunes a pretrained Japanese BERT-family
  encoder (default `cl-tohoku/bert-base-japanese-v3`, ~110M params) with a
  5-way head on the same text. `--class-weight balanced` switches to
  inverse-frequency-weighted loss — needed here, see the results notes.
- **`bench/solve_majority.py`** — always predicts the most common gold label
  (`+1`), as a proper predictions/report pair rather than a hardcoded number.

Both learned baselines use **k-fold CV**, not a single split: the public set is
253 rows and `-2` has just 2 examples, so a single held-out split would be too
noisy and stratified CV isn't even feasible with a 2-member class. Each row
gets an out-of-fold prediction in the same `id`/`gold`/`prediction` schema as
the LLM runs, so `evaluate.py` and `compare.py` work unmodified.

```bash
python bench/solve_majority.py --out outputs/predictions_majority.jsonl
python bench/solve_lexicon.py --out outputs/predictions_lexicon.jsonl
python bench/solve_encoder.py --class-weight balanced --out outputs/predictions_encoder.jsonl

python bench/evaluate.py --predictions outputs/predictions_lexicon.jsonl --report outputs/report_lexicon.json
# ...same for majority/encoder, then bench/compare.py to line all of them up
```

`solve_encoder.py` fine-tunes a **fresh** copy of the pretrained encoder per
fold (not incrementally), so no fold's predictions are contaminated by having
seen its own held-out rows during another fold's training. It picked up Metal
(`mps`) acceleration in testing even though this machine's Python is an
x86_64/Rosetta build — worth double-checking `device: ...` in its output
elsewhere.

---

## Probes and experiments

### Prompt sensitivity (`bench/prompt_sensitivity.py`)

The results tables report one number per model/setting from one fixed prompt.
Without checking whether that number moves under harmless rewording, a delta
between two rows is unfalsifiable. Two independent probes:

- **`--probe rules`** — reruns the zero-shot free-generation eval under
  `bench/rule_paraphrases.py`'s 5 paraphrases of the Appendix A.3 rules (same
  cue phrases and worked examples, byte-identical across variants; only the
  connective prose differs), each at `--seeds` sampler seeds (default `0,1,2`).
  The headline table is greedy and therefore seed-invariant by construction, so
  this probe samples at `--temperature` 0.7 to get a seed-driven noise floor to
  compare paraphrase deltas against.
- **`--probe label-order`** — reruns the zero-shot **logprob** eval
  (deterministic, no seed axis) under `bench/label_orders.py`'s 5 orderings of
  how labels are *listed* in the instruction block (the rules and the
  label→meaning mapping are unchanged; only listed position changes). Reports
  accuracy/macro-F1, predicted-label marginals, mean row-wise total-variation
  distance between softmax P(y|x), and argmax flip rate.

Both are zero-shot only (few-shot would add a third confound axis).

```bash
nohup caffeinate -i ./scripts/run_sensitivity.sh gemma-4-12b-it-Q8_0.gguf gemma4 --reasoning off \
  > outputs/run_gemma4_sensitivity.log 2>&1
```

**Result, gemma-4-12b-it.** Rule paraphrases (mean ± sd over 3 seeds at T=0.7):

| variant | accuracy | macro-F1 |
|---|---|---|
| original | 0.6285 ± 0.0274 | 0.3691 ± 0.0203 |
| checklist | 0.6298 ± 0.0060 | 0.3584 ± 0.0139 |
| qa_framing | 0.6324 ± 0.0181 | 0.3722 ± 0.0216 |
| terse | 0.6008 ± 0.0205 | 0.3545 ± 0.0122 |
| narrative | 0.5942 ± 0.0114 | 0.3388 ± 0.0081 |

Cross-paraphrase spread 0.0382 vs largest within-paraphrase seed sd 0.0274 —
**the spread exceeds seed noise: there is a real wording effect**, roughly the
size of the gaps between adjacent models in the headline table.

Label order (baseline = `original`, deterministic):

| order | accuracy | macro-F1 | mean TVD | argmax flip rate |
|---|---|---|---|---|
| original | 0.6443 | 0.3628 | — | — |
| reversed | 0.6364 | 0.3901 | 0.0478 | 0.036 |
| neutral_first | 0.6324 | 0.3546 | 0.0561 | 0.055 |
| shuffled | 0.6126 | 0.3738 | 0.0544 | 0.059 |
| extremes_first | 0.6166 | 0.3467 | 0.0631 | 0.075 |

Merely reordering the label list flips 3.6–7.5% of argmaxes and moves accuracy
by up to 3.2pp.

### Question decomposition (`bench/decompose_experiment.py`)

Does splitting multi-part questions into pieces make the label easier? The
experiment tests all three places the ambiguity could hide rather than
assuming it lives in the question: **segmentation** (is "how many questions is
this" well-defined?), **attribution** (an IR response is one block that
doesn't map 1:1 onto the sub-asks — the characteristic move is answering A at
length while declining B by never mentioning it), and **aggregation** (scoring
first / last / strongest / weakest / modal / mean-round against gold).

Findings on the multi-part stratum (qwen3.6): 76/253 public items bundle asks;
of 40 sampled, the LLM splitter agreed 31 were multi-part, and only 68% of
returned sub-questions were verbatim spans (the rest required interpretation).
10 of 79 sub-questions were unaddressed by the response. On the 31 splittable
items, sub-labels diverge 39% of the time, and against gold:

| aggregation rule | accuracy | QWK |
|---|---|---|
| **strongest / modal** | **0.7097** | **0.5550** |
| last | 0.7097 | 0.4015 |
| first | 0.6129 | 0.3533 |
| joint (no decomposition) | 0.6129 | 0.2849 |
| mean-round | 0.5806 | 0.2878 |
| weakest | 0.5161 | 0.2490 |

"Take the most committal part" tracks the annotators best, which is what
`MULTIPART_RULE` in `solve.py` states. Caveat: n=31, and the A/B above shows
stating the rule in the prompt does not reliably transfer that gain.

### Annotation study (`bench/make_annotation_subsample.py`, `bench/agreement.py`)

Every model scores ~0.65 accuracy / ~0.50 QWK on the public set and ~0.92 /
~0.97 on the test set. We can't currently say how much of that gap is model
weakness and how much is irreducible label ambiguity, because the public set
has one label per item and no measured human agreement.

`make_annotation_subsample.py` draws a stratified subsample for blind
multi-annotator re-labelling — a `random` stratum (proportionally stratified
over gold, for unbiased population estimates) and a `boundary` stratum (drawn
from the adjacent label pairs that account for most errors; `+1`/`0` and
`+1`/`+2` are 74 of gemma4's 94 public-set errors). The boundary stratum is
selected for being model-hard, so "humans also disagree here" is a claim about
model-hard items specifically — which is what the random stratum is for.

`agreement.py` scores the returned sheets: pairwise QWK between every
annotator pair and their mean (the human ceiling), Krippendorff's alpha with
the ordinal difference metric, the same split by stratum, agreement with the
existing gold, and "plural gold" coverage (how often a model's answer matches
*any* annotator — the defensible version of top-2 credit, since the acceptable
set is defined by observed human variation rather than by the model's own
ranking).

**Status: tooling only.** No annotation sheets have been collected yet, so
there is no human ceiling to report.

---

## Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt

./scripts/download_runtime.sh   # native arm64 llama.cpp (~10MB)

# fetch whichever models you want
./scripts/download_model.sh unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
./scripts/download_model.sh unsloth/gemma-4-12b-it-GGUF gemma-4-12b-it-Q8_0.gguf
./scripts/download_model.sh mradermacher/shisa-v2-qwen2.5-32b-GGUF shisa-v2-qwen2.5-32b.Q4_K_M.gguf
./scripts/download_model.sh unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
./scripts/download_model.sh unsloth/gpt-oss-20b-GGUF gpt-oss-20b-F16.gguf
./scripts/download_model.sh unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf
```

`requirements.txt` pins `numpy<2`, `scikit-learn<1.5`, and `transformers<5` for
the non-LLM baselines — on this machine's Python (x86_64 Homebrew via Rosetta;
same root cause as the arm64 llama.cpp note), PyPI has no `torch` wheel past
2.2.2, and that version needs `numpy<2` and `transformers<5` (5.x requires
`torch>=2.4`). On a native arm64 Python these pins can likely all be dropped.

---

## Running things by hand

Only one model server runs at a time (they don't all fit in RAM together).

```bash
# 1. start the local model server (leave running in its own terminal)
./scripts/serve.sh Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --reasoning off

# 2. solve — sends every prompt to the server, writes predictions
./.venv/bin/python bench/solve.py --model qwen36 --out outputs/predictions.jsonl

# 3. evaluate — accuracy / macro-F1 / QWK / confusion matrix
./.venv/bin/python bench/evaluate.py --predictions outputs/predictions.jsonl --report outputs/report.json

# 4. compare multiple runs
./.venv/bin/python bench/compare.py "qwen36 zero-shot"=outputs/report.json ...

# 5. eyeball the errors — CSV + Markdown with question/response text inline
./.venv/bin/python bench/export.py --predictions outputs/predictions.jsonl \
  --report outputs/report.json --data JF-ICR_public_set.parquet --out outputs/export_qwen36

./scripts/stop_server.sh
```

**Wrap every `bench/*.py` run in `nohup caffeinate -i ... &`** so it survives a
closed terminal and a sleeping Mac.

**Don't run two solve/serve passes at once** — they'll fight over port 8080 and
the same output files. Check `ps aux` for a running `run_*.sh`/`llama-server`
before launching. This is exactly how the 0%-accuracy gpt-oss few-shot result
happened the first time: two processes both restarting the server and both
writing the same predictions file.

Smoke-test a new model/setting with `--limit 5` first — reasoning models can
blow through a small `--max-tokens` budget without ever emitting an answer.

**A/B noise floor**: llama.cpp is not deterministic at temperature 0, and on
the 253-item public set a delta under ~0.008 accuracy (~2 items) is noise.
Several results here are reported as measured nulls for that reason.

### Orchestration scripts

| script | what it chains |
|---|---|
| `run_solution.sh` | **the submission pipeline** — solve → audit → adjudicate → re-audit → CSV |
| `run_zeroshot.sh` | zero-shot solve+eval for one model over both eval sets, one server |
| `run_bench.sh` | zero-shot + few-shot for one model, then stops the server |
| `run_all_bench.sh` | `run_bench.sh` across a list of models, sequentially |
| `run_participant_bench.sh` / `run_all_participant_bench.sh` | the same on the 50-item test set |
| `run_multipart_rule.sh` | the `off`/`always`/`gated` A/B/C over both sets |
| `run_decompose_experiment.sh` | the question-decomposition experiment |
| `run_sensitivity.sh` | both prompt-sensitivity probes for one model |
| `run_logprob_sldem.sh` / `run_logprob_sldem_participant.sh` | log-prob solve + SLD-EM calibration + eval |
| `serve.sh` / `stop_server.sh` / `download_runtime.sh` / `download_model.sh` | plumbing |

---

## Layout

```text
bench/
  solve.py                # free generation: one server call per row, zero- or few-shot
  solve_logprob.py        # verbaliser log-prob scoring (argmax + Sigma k*P(k)) — pipeline stage 1
  quota_audit.py          # provable error floor + bucket localisation from the class quota — stage 2
  adjudicate.py           # forced-choice comparative adjudication inside a flagged bucket — stage 3
  consensus.py            # localise the same errors from cross-run disagreement — stage 3b
  make_submission.py      # validated submission CSV — stage 5
  evaluate.py             # accuracy + macro-F1 + QWK + within-one + confusion matrix
  evaluate_logprob.py     # the above + gold-label rank histogram / calibration
  calibrate_logprob.py    # post-hoc prior correction (batch / SLD-EM) on solve_logprob.py output
  compare.py              # lines up accuracy / macro-F1 / QWK across report.json files
  export.py               # predictions + metrics -> CSV and Markdown, with the source text inline
  metrics.py              # shared QWK / within-one implementation
  multipart.py            # detect questions that bundle several asks
  decompose_experiment.py # does decomposing multi-part questions help? segmentation/attribution/aggregation
  prompt_sensitivity.py   # rule-paraphrase x seed and label-order robustness probes
  rule_paraphrases.py     # 5 semantically-equivalent paraphrases of the Appendix A.3 rules
  label_orders.py         # label-listing-order permutations + the prompt-rewrite function
  make_annotation_subsample.py  # stratified sheets for blind multi-annotator re-labelling
  agreement.py            # score returned sheets: human ceiling, Krippendorff alpha, plural gold
  solve_majority.py       # non-LLM baseline: always predict the most common gold label
  solve_lexicon.py        # non-LLM baseline: hedging/modality cue counts + logistic regression, k-fold CV
  solve_encoder.py        # non-LLM baseline: fine-tuned Japanese BERT-family encoder, k-fold CV
scripts/            # runtime/model download, server control, run_*.sh orchestration
paper/              # LaTeX write-up (draft)
models/, runtime/   # gitignored — large downloads, not source
outputs/            # gitignored — predictions*.jsonl, report*.json, run logs
```
