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

Smoke-test a new model/setting on a few rows first with `--limit 5` — reasoning
models in particular can blow through a small `--max-tokens` budget without
ever emitting the final answer (see gpt-oss note below).

### Results (253 examples, temperature 0)

| model | zero-shot | few-shot (k=1, 5 shots) |
|---|---|---|
| qwen3-30b-a3b | 0.4466 | **0.5282** |
| shisa-v2-qwen2.5-32b | **0.4743** | 0.4476 |
| gpt-oss-20b | 0.4190 (1 unparsed) | 0.3710 (20 unparsed) |

Notes:

- Few-shot helped Qwen3 a lot, slightly hurt shisa, and hurt gpt-oss —
  not a universal win, worth keeping both settings when trying new models.
- gpt-oss is a reasoning model: llama.cpp separates its hidden
  `reasoning_content` from the final `content`, but generation can still get
  cut off mid-reasoning before reaching an answer if `--max-tokens` is too
  low (used 1024 here). The 20 unparsed few-shot cases are exactly this —
  longer prompts leave less room before hitting the cap.
- Gold labels are heavily skewed (`+1`: 139, `0`: 72, `+2`: 29, `-1`: 11,
  `-2`: 2 out of 253) — accuracy alone rewards over-predicting `+1`/`0`; a
  majority-class baseline (`+1`) already scores 139/253 = 0.549, which beats
  every model above. Worth adding a proper class-balanced metric
  (macro-F1) if this pipeline is used for real comparisons.

### Layout

```text
bench/solve.py       # calls the local server for every row (zero- or few-shot), saves predictions
bench/evaluate.py    # accuracy + confusion matrix
bench/compare.py     # summarizes accuracy across multiple report.json files
scripts/             # runtime/model download, server start/stop
models/, runtime/    # gitignored — large downloads, not source
outputs/             # gitignored — predictions*.jsonl, report*.json
```
