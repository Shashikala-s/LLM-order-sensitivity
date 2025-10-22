# Understanding_LLM — Measuring Order Sensitivity in In-Context Learning

This project studies how **few-shot example order** affects LLM performance. We quantify accuracy spread across permutations, test simple ordering heuristics, and provide practical guidance for **robust prompting**. Everything is **inference-only** (no training).

---

## Goals

- **Measure** accuracy variability across permutations for \(k \in \{0,1,3,5\}\).
- **Explain** drivers of sensitivity (example similarity, length, position/recency).
- **Stabilize** performance using **zero-train** heuristics (e.g., sort by similarity/length or vote over random orders).
- **Deliver** a minimal, reproducible harness + JSON summaries you can plot.

---

## Models (open weights, local)

Configure via CLI or `.env`:

- **Llama-3.1-8B-Instruct** (Meta)
- **Qwen2.5-7B-Instruct** (Alibaba)
- (Optional) **Mistral-7B-Instruct**, **Phi-3-Mini**

> Works on a single GPU (~16–24 GB) with `dtype=bfloat16`/FP16. You **do not** need the CUDA toolkit unless you plan to compile extra speedups.

---

## Datasets

Start with:

- **ARC-Easy** (science MCQ).  
-  **CommonsenseQA** — loader mirrors `arc.py`.

**JSONL format (one object per line):**
```json
{"question":"Which object ...?","choices":{"A":"rock","B":"leaf","C":"metal","D":"glass"},"answer":"B"}
```
---

## Repository Structure
```text
icl-order-sensitivity/
├─ requirements.txt
├─ .env.example
├─ README.md
├─ data/                # raw data files (JSONL)
├─ outputs/             # results & cache.sqlite
└─ src/
  ├─ __init__.py
  ├─ settings.py        # project settings (env/.env), paths, defaults
  ├─ prompts.py         # hard-coded templates + prompt builder
  ├─ cache.py           # tiny SQLite KV cache
  ├─ heuristics.py      # retrieval + orderings + label balance
  ├─ runner.py          # evaluation loop + metrics
  ├─ cli.py             # single CLI entrypoint
  ├─ models/
  │  ├─ __init__.py
  │  ├─ base.py         # minimal model protocol
  │  └─ hf_local.py     # HuggingFace local wrapper
  └─ data/
     ├─ __init__.py
     ├─ base.py         # dataset schema + load helpers
     └─ arc.py          # ARC-Easy loader (CSQA will look the same)
```
---

# How the ICL Order-Sensitivity Harness Works

For each **test** question, we:

1. **Retrieve** a *candidate pool* of demos from the **train** split by embedding similarity.
2. **Select `k` demos** with rough **label balance** (A/B/C/D).
3. **Order** those `k` demos using a chosen **heuristic** (random / similarity / length).
4. **Build the prompt** = demos (with answers) + the test MCQ (without answer).
5. **Generate** with the model, **parse** the predicted letter (A/B/C/D), and **cache** the output. (SQLite)
6. If multiple orders were evaluated, **aggregate**: accuracy, flip-rate, best–worst gap; optionally **vote** over `M` random orders.

No training, no gradients—**inference only**.

---

## Key Concepts

### `k` (shots)
Number of few-shot demonstrations included before the test question.

- `k = 0` → **zero-shot**, no demos.
- `k = 1, 3, 5, 7, ...` → few-shot with `k` labeled examples.

### Permutations (orders)
Different **orders** of the *same* `k` demos.

- #orders in theory: `k!` (e.g., `3! = 6`, `5! = 120`, `7! = 5040`).
- We typically **sample** a subset with `--n_perm` when `--ordering random`.

### Ordering heuristics
- `random` — shuffle the selected `k` demos; use `--n_perm` to sample many orders.
- `similarity` — sort `k` demos by embedding similarity to the test query (desc). (Canonical; set `--n_perm 1`.)
- `length_asc` / `length_desc` — sort by total text length. (Canonical; set `--n_perm 1`.)

---
---

## How to Run

### Option A — Run as a module (no install)
```bash
export PYTHONPATH=src
#CPU
python -m src.cli --dataset arc --k 3 --n_perm 50 --ordering similarity --model Qwen/Qwen2.5-7B-Instruct --out_dir outputs/arc_qwen_k5_sim
#GPU (smoke test)
python -m src.cli --dataset arc --k 3 --n_perm 50 --ordering random --model Qwen/Qwen2.5-7B-Instruct --device cuda --dtype float16 --max_items 100 --out_dir outputs/arc_qwen_gpu_smoke
```

### Option B — Install the package (editable)
```bash
pip install -e src
python -m src.cli --dataset arc --k 3 --n_perm 50 --ordering similarity --model Qwen/Qwen2.5-7B-Instruct --out_dir outputs/arc_qwen_k5_sim
```

### Key Arguments
- `--dataset` : dataset key (e.g., `arc`)
- `--k` : number of few-shot demos (`0/1/3/5`)
- `--n_perm` : permutations sampled when `k > 1`
- `--ordering` : `random | similarity | length_asc | length_desc`
- `--k_pool` : candidate pool size from train (default `50`)
- `--vote_m` : vote over first `M` random orders (order-robust baseline)
- `--seed` : global seed
- `--max_items` : limit eval to first `N` test items (for quick tests)
- `--model` : model key or path (e.g., `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`)
- `--device` : `cpu | cuda | mps` (default `cpu`)
- `--dtype` : `float32 | float16 | bfloat16` (default `bfloat16`) // FP16 recommended for GPU
- `--data_dir` : path to data files (default `data/`)
- `--out_dir` : path to write outputs (default `outputs/`)
- `--cache_db` : path to SQLite cache (default `outputs/cache.sqlite`)
- `--max_new_tokens` : max new tokens to generate (default `8`) // adjust for longer answers
- `--temperature` : generation temperature (default `0.0`) // adjust for diversity
- `--top_p` : nucleus sampling (default `1.0`) // adjust for diversity

[//]: # (- `--max_length` : max generation length &#40;default `256`&#41; // adjust for longer contexts)
[//]: # (- `--max_retries` : max model API retries &#40;default `3`&#41;)
[//]: # (- `--retry_delay` : seconds between retries &#40;default `5`&#41;)
[//]: # (- `--log_level` : logging level &#40;default `INFO`&#41;)
**Overrides:** `--model, --device, --dtype, --data_dir, --out_dir, --cache_db`

### Outputs & Metrics
Each run writes a `summary.json` into your `--out_dir` and uses a shared prompt cache at `outputs/cache.sqlite`.

**Example fields:**
```json
{
  "dataset": "arc",
  "k": 5,
  "ordering": "similarity",
  "n_permutations": 50,
  "accuracy_vote": 0.71,
  "flip_rate": 0.34,
  "best_worst_gap": 0.12,
  "permutation_accuracies": [0.70, 0.72, ...]
}
```
