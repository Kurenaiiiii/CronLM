# CronLM

A tiny (~0.8M / ~4.8M parameter) character-level transformer, trained **from scratch** in plain PyTorch, that turns plain-English scheduling phrases into standard 5-field cron expressions.

- "every weekday at 9:30am" → `30 9 * * 1-5`
- "first of every month at midnight" → `0 0 1 * *`
- "every 15 minutes between 9am and 5pm" → `*/15 9-17 * * *`

The point of the project is learning: tokenization, transformer internals, the training loop, evaluation and error analysis. Cron is deterministic, so a plain script could solve it; the value here is that we can *measure* exactly how close a learned system gets to perfect.

## Contents

1. [What the system is made of](#what-the-system-is-made-of)
2. [Quick start (Fedora + GTX 1650)](#quick-start-fedora--gtx-1650)
3. [How training actually works](#how-training-actually-works)
4. [The data](#the-data)
5. [Checkpoints, resume, interruptions](#checkpoints-resume-interruptions)
6. [Evaluation and the failure dump](#evaluation-and-the-failure-dump)
7. [Testing](#testing)
8. [Results](#results)
9. [Kaggle alternative](#kaggle-alternative)
10. [Honest limits](#honest-limits)

## What the system is made of

The pieces, in the order data flows through them:

| Piece | File | What it does |
|---|---|---|
| Cron model | `src/cronlm/cron_spec.py` | The 5-field cron language: canonical formatter, validator, set expander, next-run scheduler. Hand-written, no `croniter`. |
| Phrase grammar | `src/cronlm/grammar.py` | Samples a *structured spec* (e.g. minute=30, hour=9, dow={1..5}) and renders **both** the English phrase (from template tables) and the canonical cron string. Because labels never come from parsing a phrase, labels are correct by construction. |
| Data builder | `src/cronlm/dataset_gen.py` | `gen-data`: dedupes by phrase, hard-fails on label conflicts, hash-splits into train/val/test/holdouts. |
| Tokenizer | `src/cronlm/tokenizer.py` | Character-level. 48 tokens: 4 specials (`<pad> <bos> <sep> <eos>`) + sorted allowed chars (`a-z 0-9 space : , . - ' * /`). |
| Model | `src/cronlm/model.py` | Hand-written decoder-only GPT: learned positional embeddings, pre-LayerNorm blocks, causal attention (`F.scaled_dot_product_attention`), GELU MLP (4×), final LN, weight-tied head. |
| Datasets | `src/cronlm/data.py` | `FileCronDataset` (fixed JSONL) and `OnlineCronDataset` (fresh generator samples every step, same hash-split filter, no `train.jsonl` involved), with collate + prompt loss mask. |
| Training | `src/cronlm/train.py` | AdamW (no decay on norms/biases/embeddings), linear warmup → cosine decay to 1e-5, grad clip 1.0, fp32 default, checkpoint/resume, early stop, val exact-match eval every 2,000 steps with greedy decoding. |
| Evaluation | `src/cronlm/evaluate.py` | Exact match, semantic match, validity, per-level / per-template_id tables, per-field accuracy, failure dump. |
| CLI | `src/cronlm/cli.py` | `python -m cronlm.cli <command>`: `gen-data`, `train`, `eval`, `predict`, `repl`. |

Everything is deterministic given a seed (hash-based split uses `hashlib.md5`, not the randomized `hash()`).

## Quick start (Fedora + GTX 1650)

Every new terminal needs the `cd` + `source` lines first.

```bash
# 1. system packages
sudo dnf install -y python3 python3-pip git

# 2. project folder + virtual environment
cd ~/Projects/CronLM
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. confirm the NVIDIA driver works, then install PyTorch with CUDA.
#    `pip install torch` installs the default PyPI wheel, which ships with CUDA
#    support, so no extra index URL or CUDA-specific flags are needed.
#    The GTX 1650 (4 GB) has no bf16, so training defaults to fp32 (see PRD §1.1, §8.3).
nvidia-smi
pip install torch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True NVIDIA GeForce GTX 1650 ...
# (no GPU? use the CPU build instead: pip install torch --index-url https://download.pytorch.org/whl/cpu)

# 4. install the project and dev tools
pip install -e ".[dev]"

# 5. run the tests
pytest -q

# 6. generate data (pregenerated mode; default train size is 500,000)
python -m cronlm.cli gen-data --out data --train 500000 --val 20000 --test 100000 --seed 1337

# 7. smoke test (under a minute; 200 steps on 5k examples)
python -m cronlm.cli train --config configs/tiny.yaml --smoke --device cuda

# 8. real training: pregenerated data
python -m cronlm.cli train --config configs/tiny.yaml --device cuda
#    or online mode (fresh data every step; val/test stay fixed files on disk)
python -m cronlm.cli train --config configs/tiny.yaml --device cuda --online

# 9. evaluate
python -m cronlm.cli eval --ckpt runs/tiny/best.pt --split test

# 10. try it
python -m cronlm.cli predict --ckpt runs/tiny/best.pt "every weekday at 9:30am"
```

Resume an interrupted run:

```bash
python -m cronlm.cli train --config configs/tiny.yaml --device cuda --resume runs/tiny/last.pt
```

## How training actually works

This section is for learning, in plain language. No ML background assumed beyond "a neural net is a big function with adjustable numbers".

### 1. The task is next-token prediction

Each training sequence is one line of tokens:

```
<bos> every monday at 9am <sep> 0 9 * * 1 <eos>
```

The model reads this left to right and at **every position** predicts the character of the next token. It never "understands" cron. It learns a function from character sequences to character sequences that happens to be right almost every time.

### 2. Embeddings and attention in one paragraph each

**Embeddings.** Every one of the 48 possible tokens (characters and the specials) is mapped to a vector of 128 numbers (`tiny`). Similar-sounding characters get similar vectors after training. The sequence becomes a matrix: one row per token. The model also has a **learned positional embedding** so it knows *where* each token is (the `9` at position 8 means something different from the `9` at position 10).

**Attention.** Each character's vector gets updated by looking at earlier vectors and deciding how much each earlier one matters, with scores computed from the vectors themselves. When the model is writing the hour field of the cron output, attention lets it look back at the `9` and the `am` in the phrase and reuse them. Multi-head attention just does this several times in parallel with different projections and concatenates the results.

### 3. Loss

**Loss** = cross-entropy between the model's predicted probability distribution for the next character and the truth, averaged over cron tokens only (prompt tokens and padding are masked out with `-100`).

- A loss around **3.9** is random guessing (about `ln(48)`; a uniform distribution over the vocab).
- A loss around **0.1** means the model is confidently right on the cron tokens.
- A loss around **0.01** is a solved task.

### 4. Backprop and AdamW

Compute the gradient of the loss with respect to every weight (how much would the loss go down if we nudged this weight up a little), then nudge every weight the opposite way a tiny bit. **AdamW** is a smarter nudge: it keeps a running average of recent gradients and their squared sizes, so it moves steadily even on noisy or sparse gradients, and it decouples weight decay from the gradient step. We apply weight decay 0.01 to the 2D Linear weights only — not to biases, LayerNorm gains or embeddings, where it would only hurt.

The learning rate (step size) **warms up** from 0 to `1e-3` over the first 500 steps. Early on the model is random and gradients are chaotic; a big first step would scramble it. After warmup the LR **cosine-decays** to `1e-5`. The slow tail matters: it lets the model settle into precise digit-by-digit solutions instead of oscillating around them. That's why the last 1–2% of accuracy (rare combos, quarter-to phrases, 12am/12pm) is mostly won in the decay tail.

### 5. Steps, batches, epochs

- **Batch**: we never feed one example at a time; we feed 256 at a time and average the gradients. This makes gradient estimates smoother and uses the GPU well.
- **Step**: one batch = one forward + one backward + one optimizer update.
- **Epoch**: with 500,000 training examples and batch 256, one epoch is about 1,950 steps. `tiny` stops at 30,000 steps (~15 epochs); `small` at 60,000 (~30 epochs).
- In `--online` mode there are no epochs. The dataloader calls the generator every step and produces fresh random batches, so the model effectively never sees the same example twice. Val and test stay fixed files on disk, exactly as in pregenerated mode. Online sampling is filtered through the same hash-split as the files, so val/test/holdout phrases are dropped and re-sampled.

### 6. What the curves should look like

Loss falls fast in the first few hundred steps (the model learns the output alphabet, `*`, digits and field positions first). Then it improves slowly as it learns hour/minute mapping, ranges and steps. Validation exact match typically jumps into the high 90s relatively early; the last 1–2% takes the longest and benefits most from the LR-decay tail and from training longer. If loss plateaus and val exact match stalls, see the debugging playbook below.

### 7. Why greedy decoding and exact match

The task has exactly one right answer per phrase, so we never sample from the distribution; we always take the argmax (greedy). The primary metric is **exact match** (string equality with the canonical label). We also track **semantic match** (same expanded sets of allowed values, and same `*`-ness of dom and dow, so `1,2,3,4,5` equals `1-5`), **validity rate**, **per-level**, **per-template**, and **per-field** accuracies.

### 8. Debugging playbook if accuracy stalls under 100%

1. Open `runs/<name>/failures_<split>.jsonl`. Group by `template_id` and `level`. Failures almost always cluster.
2. Clustered failures: check the generator for a label bug *first* (a systematic wrong label in one template teaches the wrong thing), then raise that template's sampling weight if it's just rare.
3. Failures on unseen (hour, minute) combos: the model memorized pairs; train longer, use `--online`, or raise model size (`tiny` → `small`).
4. Failures in one field only: look at the per-field table (minute / hour / dom / month / dow) and targets the failing field.
5. If `val` is perfect but `test` is not, the splits leak or differ in distribution: re-check the hash-split logic (or your generation seed).
6. Still stuck: switch `tiny` → `small`, then train longer.

### 9. Honest limits

100% means "100% on this grammar's distribution". Off-distribution phrasing (typos, slang, other languages) is not targeted; the `test_paraphrase_holdout.jsonl` split reports exactly that, honestly, with no accuracy target attached.

## The data

No data is downloaded. Everything is generated by `gen-data` from the grammar in the PRD (Section 4). Output rows look like:

```json
{"phrase": "at 9am every monday", "cron": "0 9 * * 1", "level": 3, "template_id": "time.exact.4+day.named_days.0"}
```

| File | Default size | Purpose |
|---|---|---|
| `data/train.jsonl` | 500,000 | training (pregenerated mode) |
| `data/val.jsonl` | 20,000 | model selection during training |
| `data/test.jsonl` | 100,000 | final in-distribution score (the 100% target) |
| `data/test_combo_holdout.jsonl` | 20,000 | unseen (hour, minute) combos (~5% of all pairs reserved by hash, never in training) |
| `data/test_paraphrase_holdout.jsonl` | 10,000 | phrasings from reserved templates train never sees (reported only) |
| `data/golden.jsonl` | 200 | hand-written, hand-reviewed pairs |

Rules enforced by the builder: **no label conflicts** (two crons for the same phrase → hard failure), **dedupe** by phrase string, **hash-based split** by `md5(phrase) % 1000` with fixed buckets (0-39 val, 40-239 test, 240+ train), so a phrase can never leak between splits. `--online` uses the exact same filter.

Level shares: L1 15% · L2 30% · L3 25% · L4 20% · L5 10%. Every example is tagged with `level` and `template_id` for per-bucket error analysis.

## Checkpoints, resume, interruptions

- `last.pt` is overwritten at every eval (every 2,000 steps); `best.pt` when val exact match improves (tie-break: lower val loss).
- A checkpoint contains: model weights, optimizer state, step, full config, vocab size, RNG state (Python + torch CPU + CUDA), best metric, early-stop counter.
- Resume with `--resume runs/<name>/last.pt`. The RNG state is restored and the deterministic batch stream is fast-forwarded past the consumed batches, so the resumed run sees the same data order as an uninterrupted one.
- Checkpoints are written on eval *and* on Ctrl-C. Laptop sleeps, thermal shutdowns and Kaggle session limits are all fine.
- `--smoke` runs 200 steps on 5k examples; it exists to verify the whole pipeline in under a minute. `pytest -q` uses it.

## Evaluation and the failure dump

```bash
python -m cronlm.cli eval --ckpt runs/tiny/best.pt --split test
```

prints a summary box: exact match, semantic match, validity, per-level exact match, per-field accuracy; writes `runs/<name>/eval_test.json` and `runs/<name>/failures_test.jsonl` (phrase, expected, predicted, level, template_id, which fields differ). Splits: `test`, `val`, `combo`, `paraphrase`, `golden`.

## Testing

```bash
pytest -q
```

covers: cron canonicalization/validation/expansion/scheduler · 50k generated pairs (every cron validates, no conflicts, no dom+dow restriction, sequence fits block size) · reserved-template leak checks for `--online` · tokenizer round-trip and error paths · model shape/causality/padding/param-count/32-example overfit · gen-data schema, dedupe, byte-level determinism · end-to-end `--smoke` training, resume, predict · golden set validity.

The 200 pairs in `tests/golden.jsonl` were hand-written and machine-checked for canonical form; **you should still eyeball at least 50 of them yourself** before trusting the numbers (PRD Section 11).

## Results

Real numbers from this machine (GTX 1650 Mobile, 4 GB). Everything here is measured, not estimated; the runs not yet completed are marked *pending*.

**Data generation (measured):** 650,000 rows in ~52s (~24k candidates/s). Audit over all 650k rows: 0 invalid crons, 0 non-canonical, 0 dom+dow restrictions, 0 overlong sequences, 650,000 unique phrases. Splits filled exactly as requested: train 500,000 / val 20,000 / test 100,000 / combo holdout 20,000 / paraphrase holdout 10,000. Determinism holds: same seed → byte-identical files (tested).

**CUDA smoke run (measured):** `train --config configs/tiny.yaml --smoke --device cuda` on the GTX 1650: 200 steps in 0.7 min (~21k tokens/s), loss 4.04 → 1.08, checkpoints saved, eval runs end to end. Exact match on a 512-example val subset after 200 steps is not meaningful yet (1.56%); the full run's numbers land in the table below.

**L1 scarcity (honest deviation from PRD 4.5):** the PRD targets 15% L1 in train. L1 phrases are single-clause sentences, and after the required phrase-level dedupe the unique L1 space saturates at roughly 26–36k examples, so train ends at about **7% L1** (36,080 at time of writing), not 15%. The level shares are implemented exactly as specified in the sampler (15/30/25/20/10); the gap is the result of the dedupe requirement interacting with a small L1 phrase space, and it cannot be closed by sampling harder. L2–L5 follow the intended shares closely.

| Model | Params | Steps | Train time | Hardware | Exact match (test) | Semantic | Validity |
|---|---|---|---|---|---|---|---|
| tiny (smoke) | 819,968 | 200 | ~10 min on CPU | CPU | smoke checkpoint, not meaningful | — | — |
| tiny | 819,968 | 30,000 | pending | GTX 1650 | pending | pending | pending |
| small | 4,792,320 | 60,000 | pending | GTX 1650 | pending | pending | pending |

Per-level accuracy and timings for real runs belong in this table once the full runs complete (see PRD Section 14, M8). `tiny` is the first run; `small` is the main model — on this laptop expect overnight, and use `--resume`.

## Kaggle alternative

Yes — you can (and for `small` you probably should) train on Kaggle GPUs instead of the 1650. Notes first, honestly:

- The training code is single-GPU. It does **not** use both T4s (no DDP); selecting T4 ×2 is harmless, the second card just sits idle. A `small` model needs ~4–6 GB VRAM in fp32, so one T4 (16 GB) is plenty. Multi-GPU would be a deliberate code change, not a flag.
- T4 (Turing) supports **fp16** tensor cores but **not bf16** (bf16 needs Ampere or newer; the 1650 doesn't have it either, which is why local runs default to fp32). On Kaggle, `--amp fp16` is the right choice and is much faster than fp32 there.
- Steps in the notebook: `git clone https://github.com/Kurenaiiiii/cronlm.git`, `pip install -e .`, `python -m cronlm.cli gen-data --out data --train 500000 --val 20000 --test 100000 --seed 1337` (deterministic, so you get byte-identical splits to this machine), then `python -m cronlm.cli train --config configs/small.yaml --device cuda --amp fp16`. The repo ships a ready-to-run notebook: `kaggle_train.ipynb` at the root, written for one T4.
- Rough expectations (estimates, not yet measured): tiny in well under an hour; small in ~2–4 hours on T4 fp16. Kaggle gives you a session limit (~12 h on GPU) and a weekly GPU quota; if the session dies, resume exactly where it stopped: `python -m cronlm.cli train --config configs/small.yaml --device cuda --amp fp16 --resume runs/small/last.pt`.
- When it's done, download `runs/small/best.pt` **and** `runs/small/vocab.json` (inference loads the vocab from next to the checkpoint), then run eval locally: `python -m cronlm.cli eval --ckpt /path/to/best.pt --split test`.
- Kaggle needs internet enabled to clone the repo (or use the "Add data" button to upload a repo zip instead).

## Honest limits

- 100% means on this grammar's distribution, not on free-form English.
- The paraphrase holdout (`test_paraphrase_holdout.jsonl`) exists so you can see brittleness instead of hiding it.
- Steps in cron `*/N` start from the field minimum (minute 0, hour 0, month 1) and take every Nth value — cron semantics are authoritative, even when they differ from English intuition.
- No 6/7-field cron, no macros, no L/W/# specials, no both-dom-and-dow restriction (cron treats them as OR).
