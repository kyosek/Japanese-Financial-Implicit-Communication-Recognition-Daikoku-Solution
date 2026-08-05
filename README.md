# FinNLP-Japanese

## JF-ICR benchmarking pipeline

Solves and evaluates the EBISU **JF-ICR** (Japanese Financial Implicit
Communication Recognition) subtask: given a Japanese financial Q&A pair, an
LLM classifies the company response's implicit intent on a 5-point scale
(`+2` strong commitment ... `-2` strong refusal). Metric: accuracy.

`JF-ICR_public_set.parquet` already contains fully-formed prompts (`query`)
and gold labels (`answer`) — no extra prompt engineering needed.

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

### Results (253 examples, temperature 0, with Appendix A.3 annotation rules)

| model | zero-shot | few-shot (k=1, 5 shots) |
|---|---|---|
| gemma-4-12b-it | **0.6443** | 0.6331 |
| qwen3-30b-a3b | 0.6047 | 0.5605 |
| gpt-oss-20b | 0.6008 | 0.5000 |
| shisa-v2-qwen2.5-32b | 0.5336 | 0.5161 |

All 8 runs: 0 unparsed labels.

Notes:

- Adding the annotation rules lifted every model well above its bare-prompt
  score (e.g. qwen3-30b-a3b zero-shot went 0.4466 → 0.6047) — the paper's own
  linguistic-signal cues are informative context, not just boilerplate.
- With the rules in place, zero-shot beats few-shot for every model here —
  the opposite of what we saw without rules, where few-shot helped qwen3
  substantially. The rules and the exemplars appear to overlap in what
  they teach the model, and stacking both isn't better than rules alone.
- gemma-4-12b-it and gpt-oss-20b default to extended reasoning; see the
  reasoning-model note above for why the fix differs between them.
- Gold labels are heavily skewed (`+1`: 139, `0`: 72, `+2`: 29, `-1`: 11,
  `-2`: 2 out of 253) — a majority-class baseline (always `+1`) scores
  139/253 = 0.5494. Every zero-shot result above beats it; shisa and
  gpt-oss's few-shot runs don't. Worth adding macro-F1 if this pipeline is
  used for real comparisons, since accuracy alone still rewards
  over-predicting the majority class.

### Layout

```text
bench/solve.py       # calls the local server for every row (zero- or few-shot), saves predictions
bench/evaluate.py    # accuracy + confusion matrix
bench/compare.py     # summarizes accuracy across multiple report.json files
scripts/             # runtime/model download, server start/stop, run_bench.sh / run_all_bench.sh orchestration
models/, runtime/    # gitignored — large downloads, not source
outputs/             # gitignored — predictions*.jsonl, report*.json
```
