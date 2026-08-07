# FinNLP-Japanese

## JF-ICR benchmarking pipeline

Solves and evaluates the EBISU **JF-ICR** (Japanese Financial Implicit
Communication Recognition) subtask: given a Japanese financial Q&A pair,
classify the company response's implicit intent on a 5-point scale (`+2`
strong commitment ... `-2` strong refusal). Metric: accuracy (plus macro F1,
given the label distribution below is skewed).

`JF-ICR_public_set.parquet` already contains fully-formed prompts (`query`)
and gold labels (`answer`) — no extra prompt engineering needed.

Besides four local LLMs (`bench/solve.py`), three **non-LLM baselines**
(`bench/solve_majority.py`, `bench/solve_lexicon.py`, `bench/solve_encoder.py`)
check whether the label is recoverable without an LLM at all -- see
"Non-LLM baselines" below.

### How it works

A local model, served via [llama.cpp](https://github.com/ggml-org/llama.cpp)
on an OpenAI-compatible endpoint, answers each prompt. Predictions are
compared against gold labels for accuracy. `bench/solve.py` also supports
few-shot: it holds out one exemplar per label (5 shots) from the eval set and
prepends them to every remaining prompt, reusing the dataset's own template.

By default, `bench/solve.py` also splices in the EBISU paper's **Appendix
A.3 "Specific Annotation Rules"** — the per-label linguistic-signal cues and
representative examples the human annotators used to tell adjacent labels
apart (e.g. +1 vs +2) — right before the Q&A block, for both zero- and
few-shot prompts. Pass `--no-annotation-rules` to reproduce the bare
label-definition prompt from the dataset's `query` column unmodified.

Some models default to an extended thinking/reasoning mode that can burn the
entire token budget without ever emitting the final label — `content` comes
back empty while `reasoning_content` holds an unfinished chain of thought.
The fix depends on the model:

- **gemma-4-12b-it**: `--reasoning off` (passed through `scripts/serve.sh`)
  actually suppresses the extended thinking — cheap and reliable.
- **gpt-oss-20b**: `--reasoning off` only changes how llama.cpp *labels*
  `content` vs `reasoning_content` in the response; the harmony format's
  analysis channel is generated either way (confirmed: still ~450+ reasoning
  tokens with the flag set). The actual fix is a generous `--max-tokens`
  (used 2048) so generation doesn't get cut off before reaching the final
  channel.

If a model shows a high unparsed rate, check which failure mode it is
(`reasoning_content` populated but `content` empty = ran out of budget) before
assuming the flag will fix it.

- **Runtime**: a native **arm64** llama.cpp build (Metal-accelerated). The
  Homebrew `llama.cpp` on this machine is the x86_64/Rosetta build with no
  Metal support, so `scripts/download_runtime.sh` fetches an official arm64
  release into `runtime/` instead of using it.
- **Models compared** (all GGUF, all fit comfortably in 48GB unified memory):

  | model | size | notes |
  |---|---|---|
  | [shisa-v2-qwen2.5-32b](https://huggingface.co/shisa-ai/shisa-v2-qwen2.5-32b) (Q4_K_M) | ~20GB | Japanese-specialized fine-tune of Qwen2.5-32B (Shisa.AI) |
  | [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) (Q4_K_M) | ~19GB | MoE, ~3B active params — multilingual generalist, fast |
  | [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) (F16/native MXFP4) | ~14GB | OpenAI open-weight reasoning model (harmony format) |
  | [gemma-4-12b-it](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) (Q8_0) | ~13GB | Dense 12B, hybrid local/global attention, reasoning by default |

> Kimi-K2/K3 (mentioned in early planning) are 1T–2.8T parameter MoE models
> whose smallest GGUF quants are 350GB–600GB+ — not feasible on a 48GB Mac.

### Non-LLM baselines

Three baselines that don't call an LLM at all, to check whether the label is
recoverable from surface form alone (in which case the LLM comparison above
is the wrong headline result):

- **`bench/solve_lexicon.py`** -- counts occurrences of a small, hand-curated
  set of Japanese hedging/modality/commitment cue phrases in the company's
  response text (sentence-final forms like 差し控えさせていただきます / 検討
  してまいります / 〜と考えております, the benefactive 〜させていただく,
  conditional and epistemic markers -- see `CUE_GROUPS` in the script) and
  fits a logistic regression on the counts.
- **`bench/solve_encoder.py`** -- fine-tunes a pretrained Japanese BERT-family
  encoder (default `cl-tohoku/bert-base-japanese-v3`, ~110M params) with a
  5-way classification head on the same response text. `--class-weight
  balanced` switches to inverse-frequency-weighted cross-entropy loss --
  needed here, see the results notes below on unweighted fine-tuning
  collapsing to the majority class.
- **`bench/solve_majority.py`** -- always predicts the most common gold label
  (`+1`), made reproducible as a proper predictions/report pair instead of a
  hardcoded number.

Both learned baselines use **k-fold CV**, not a single train/test split: the
public set is only 253 rows total and one label (`-2`) has just 2 examples,
so a single held-out split would be too noisy, and stratified CV isn't even
feasible with a 2-member class. Each row gets an out-of-fold prediction from
the fold that held it out, written in the same `id`/`gold`/`prediction`
schema as the LLM runs, so `bench/evaluate.py` and `bench/compare.py` work
unmodified on all of them.

```bash
python bench/solve_majority.py --out outputs/predictions_majority.jsonl
python bench/solve_lexicon.py --out outputs/predictions_lexicon.jsonl
python bench/solve_encoder.py --class-weight balanced --out outputs/predictions_encoder.jsonl

python bench/evaluate.py --predictions outputs/predictions_lexicon.jsonl --report outputs/report_lexicon.json
# ...same for majority/encoder, then bench/compare.py to line all of them up
```

`bench/solve_encoder.py` fine-tunes a **fresh** copy of the pretrained
encoder per fold (not incrementally across folds), so no fold's predictions
are contaminated by having seen its own held-out rows during another fold's
training. It picked up Metal (`mps`) acceleration in testing even though this
machine's Python itself is an x86_64/Rosetta build (see the caveat below) --
worth double-checking `device: ...` in its output on other machines.

### Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt

./scripts/download_runtime.sh   # native arm64 llama.cpp (~10MB)

# fetch whichever models you want to compare
./scripts/download_model.sh mradermacher/shisa-v2-qwen2.5-32b-GGUF shisa-v2-qwen2.5-32b.Q4_K_M.gguf
./scripts/download_model.sh unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
./scripts/download_model.sh unsloth/gpt-oss-20b-GGUF gpt-oss-20b-F16.gguf
```

`requirements.txt` pins `numpy<2`, `scikit-learn<1.5`, and `transformers<5`
for the non-LLM baselines above -- on this machine's Python (an x86_64
Homebrew build translated via Rosetta; same root cause as the arm64
llama.cpp note below), PyPI has no `torch` wheel past 2.2.2, and that version
needs `numpy<2` for its numpy bridge and `transformers<5` (5.x requires
`torch>=2.4`). On a native arm64 Python these pins can likely all be dropped
in favor of current versions.

### Run

Only one model server runs at a time (they don't all fit in RAM together).

```bash
# 1. start the local model server (leave running in its own terminal)
./scripts/serve.sh shisa-v2-qwen2.5-32b.Q4_K_M.gguf     # or any file in models/

# 2. solve — sends every prompt to the server, writes predictions
./.venv/bin/python bench/solve.py --model shisa --out outputs/predictions.jsonl
./.venv/bin/python bench/solve.py --model shisa --few-shot-k 1 --out outputs/predictions_fewshot.jsonl

# 3. evaluate — scores predictions against gold labels
./.venv/bin/python bench/evaluate.py --predictions outputs/predictions.jsonl --report outputs/report.json

# 4. compare multiple runs
./.venv/bin/python bench/compare.py "shisa zero-shot"=outputs/report.json ...

./scripts/stop_server.sh
```

For unattended runs, `scripts/run_bench.sh` chains steps 1–3 (zero-shot +
few-shot) for one model and stops the server when done. Wrap it in
`nohup caffeinate -i ... &` so it survives closing the terminal and the Mac
sleeping:

```bash
nohup caffeinate -i ./scripts/run_bench.sh gemma-4-12b-it-Q8_0.gguf gemma4 --reasoning off \
  > outputs/run_gemma4_bench.log 2>&1 &
```

`scripts/run_all_bench.sh` chains `run_bench.sh` across multiple models
sequentially (only one fits in RAM at a time) — edit the model list at the
top of the file, then run it the same way:

```bash
nohup caffeinate -i ./scripts/run_all_bench.sh > outputs/run_all_bench.log 2>&1 &
```

**Don't run two solve/serve passes at once** — they'll fight over port 8080
and the same output files. This is exactly how the 0%-accuracy gpt-oss
few-shot result below happened the first time around: two processes both
restarting the server and both writing to `predictions_gptoss_fewshot.jsonl`.

Smoke-test a new model/setting on a few rows first with `--limit 5` —
reasoning models in particular can blow through a small `--max-tokens`
budget without ever emitting the final answer (see reasoning-model note
above).

### Log-prob verbalizer scoring (alternative to free generation)

`bench/solve_logprob.py` scores each of the 5 label strings directly —
P(label | prompt), via teacher-forced token log-probs against llama.cpp's
raw `/completion` endpoint — instead of letting the model generate freely
and parsing a label out of the text. Takes the argmax as the prediction and
also reports a continuous score, Σk·P(k) over {+2..-2}. This can't produce
an unparsed label, and unlike free generation it tells you *how* wrong a
miss was: whether the model ranked the gold label a close second (a
calibration/prior problem) or buried it near the bottom (a comprehension
problem) — `bench/evaluate_logprob.py` reports gold-label rank histograms
and mean P(gold) per class to answer exactly that.

```bash
./.venv/bin/python bench/solve_logprob.py --model shisa --out outputs/predictions_shisa_logprob.jsonl
./.venv/bin/python bench/evaluate_logprob.py \
  --predictions outputs/predictions_shisa_logprob.jsonl \
  --report outputs/report_shisa_logprob.json

# same --few-shot-k / --no-annotation-rules flags as solve.py, so results are comparable
```

It runs a 3-5 row cross-check against free generation by default
(`--sanity-check-n`, 0 to skip) before the full pass — trust that agreement
number before trusting a full run on a model you haven't scored before.
**Reasoning/harmony-format models are the reason this check exists**:
gpt-oss-20b's template mandates an analysis-channel preamble before any
content can appear, so the label does not immediately follow the prompt and
naive log-prob scoring there measures the wrong thing (every label looks
equally, vanishingly unlikely). Confirmed to just work out of the box for
gemma-4-12b-it (100% agreement with free generation in testing); verify
qwen3/shisa/gpt-oss before trusting their numbers.

`report_logprob.json` uses the same `n`/`correct`/`accuracy`/`unparsed`/
`confusion` keys as `evaluate.py`'s report, so `bench/compare.py` can line
up free-generation and log-prob-argmax accuracy side by side.

### Results (253 examples, temperature 0, with Appendix A.3 annotation rules)

Cells are `accuracy / macro-F1`. Macro-F1 matters here because gold labels
are heavily skewed (`+1`: 139, `0`: 72, `+2`: 29, `-1`: 11, `-2`: 2 out of
253) and it doesn't reward leaning on the majority class the way accuracy
alone does.

| model | zero-shot | few-shot (k=1, 5 shots) |
|---|---|---|
| gemma-4-12b-it | **0.6443 / 0.3628** | 0.6331 / 0.3517 |
| gpt-oss-20b | 0.6008 / 0.3521 | 0.5000 / 0.3214 |
| qwen3-30b-a3b | 0.6047 / 0.3353 | 0.5605 / 0.2624 |
| shisa-v2-qwen2.5-32b | 0.5336 / 0.2917 | 0.5161 / 0.2912 |

All 8 runs: 0 unparsed labels.

**Non-LLM baselines** (5-fold CV, out-of-fold predictions; see above):

| baseline | accuracy | macro-F1 |
|---|---|---|
| cue-lexicon + logistic regression | 0.5573 | 0.2510 |
| fine-tuned encoder, class-weighted loss | 0.5217 | 0.2493 |
| fine-tuned encoder, unweighted loss | 0.5534 | 0.1526 |
| majority-class (always `+1`) | 0.5494 | 0.1418 |

Notes:

- **The non-LLM baselines don't match the LLMs here** — every LLM zero-shot
  run beats every non-LLM baseline on macro-F1 (0.29-0.36 vs. 0.15-0.25), so
  the reviewer hypothesis this section was added to test ("if a cue-lexicon
  regression matches a 30B LLM, that reframes the whole paper") doesn't hold
  for this dataset/protocol. That's still a useful result: it means the
  annotation-rule cues the LLMs are given aren't trivially replicated by
  literal substring matching or by fine-tuning a small encoder on ~200
  examples/fold, so the LLM comparison is measuring something real rather
  than a surface-form shortcut.
- **Unweighted encoder fine-tuning collapsed to the majority class**: its
  confusion matrix predicts `+1` or `0` for nearly every row regardless of
  gold label, landing within 0.4pp accuracy of the plain majority-class
  baseline. This is a standard small-data/class-imbalance failure mode for
  fine-tuning a 110M-parameter model on ~200 examples with a 55%-skewed
  label, not evidence the task is unlearnable by an encoder --
  inverse-frequency class-weighted loss (`--class-weight balanced`) fixes the
  collapse (macro-F1 0.15 → 0.25, matching the lexicon baseline) at some
  accuracy cost. Neither variant closes the gap to the LLMs.
- The cue-lexicon+LR baseline's accuracy (0.5573) is barely above
  majority-class (0.5494), but its macro-F1 (0.2510) is meaningfully higher
  (0.1418) -- it's not just a majority-class clone, accuracy alone just
  doesn't show the difference.
- Adding the annotation rules lifted every model well above its bare-prompt
  score (e.g. qwen3-30b-a3b zero-shot went 0.4466 → 0.6047) — the paper's own
  linguistic-signal cues are informative context, not just boilerplate.
- With the rules in place, zero-shot beats few-shot for every model here —
  the opposite of what we saw without rules, where few-shot helped qwen3
  substantially. The rules and the exemplars appear to overlap in what
  they teach the model, and stacking both isn't better than rules alone.
- gemma-4-12b-it and gpt-oss-20b default to extended reasoning; see the
  reasoning-model note above for why the fix differs between them.

### Layout

```text
bench/solve.py            # calls the local server for every row (zero- or few-shot), saves predictions
bench/evaluate.py         # accuracy + macro F1 + confusion matrix
bench/solve_logprob.py    # verbalizer log-prob scoring (argmax + Sigma k*P(k)) instead of free generation
bench/evaluate_logprob.py # accuracy + macro F1 + confusion matrix + gold-label rank histogram / calibration
bench/solve_majority.py   # non-LLM baseline: always predict the most common gold label
bench/solve_lexicon.py    # non-LLM baseline: hedging/modality cue-lexicon counts + logistic regression, k-fold CV
bench/solve_encoder.py    # non-LLM baseline: fine-tuned Japanese BERT-family encoder, k-fold CV
bench/compare.py          # summarizes accuracy + macro F1 across multiple report.json files
scripts/             # runtime/model download, server start/stop, run_bench.sh / run_all_bench.sh orchestration
models/, runtime/    # gitignored — large downloads, not source
outputs/             # gitignored — predictions*.jsonl, report*.json
```
