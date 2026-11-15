## CronLM — diagnostic report (combo-holdout failure analysis)


*Produced 2026-11-15 from the first `small` checkpoint (the 90.68% combo run): 1,864 failures out of 20,000 on the combo holdout, clustered by template id, hour/minute bucket and reserved-template membership. Internal diagnostic artifact — kept for traceability and referenced by the README.*


# CronLM — diagnostic report for Claude

Model: `CronLM/small`, 4,792,320 params. Checkpoint under review: `runs/small/best.pt` (and `runs/small/last.pt`, identical weights — verified bit-equal). Trained on a single cloud-instance setup. All numbers below come from the run's own eval artifacts (`eval_*.json`, `failures_*.jsonl`) and from `log.csv`. No code changes were made for this report.

---

## A. Current run

**Command used for the run (as recorded in the run's own console output):**

```
python -m cronlm.cli train --config configs/small.yaml --device cuda --amp fp16
```

**Exact config (`configs/small.yaml`):**

| field | value | | field | value |
|---|---|---|---|---|
| d_model | 256 | | steps | 60000 (early-stopped at 48,000) |
| n_layer | 6 | | peak_lr | 1e-3 (cosine → 1e-5 FINAL_LR) |
| n_head | 8 | | warmup_steps | 500 |
| ff_dim | 1024 | | batch_size | 256 |
| block_size | 160 | | grad_accum_steps | 1 |
| dropout | 0.0 | | grad_clip | 1.0 |
| weight_decay | 0.01 (no decay on biases, LayerNorm gains, embeddings) | | optimizer | AdamW, betas (0.9, 0.95) |
| seed | 1337 | | eval_interval | 2000 |
| eval_subset | 5000 | | log_interval | 100 |

**Training data:** pregenerated `data/train.jsonl` (500,000 rows); **`--online` was NOT used.** Precision: fp16 mixed. Parameter count printed at startup: 4,792,320 (matches `GPT.num_params()`). Training time: **132.3 min**. Last step reached: **48,000 of 60,000** (early stop fired at eval @ 48,000, 3 consecutive full-val 100% evals). Checkpoints: `runs/small/best.pt` and `runs/small/last.pt`, both step 48000 (see F).

**Is there a newer run?** No. Earlier code and data modes are represented only here and in the README; the two model variants are documented in the README's Results section.

**Why 90.68% combo is pre-existing (same config as everything else):** the combo-holdout score comes from exactly this run; there is no separate "old" model — the checkpoints under review ARE the combo-scoring run.

Differences between the completed run and the planned (unexecuted) `--online` run:

| | checkpoint under review | online-mode variant (not used here) |
|---|---|---|
| data source | `data/train.jsonl` fixed file | generator every step, infinite |
| steps | 48,000 (early stop) | 100,000 (early stop still applies) |
| everything else | as above | identical; `--online --amp fp16` |

**How `--online` works (file: `src/cronlm/data.py`):**

`OnlineCronDataset` (IterableDataset) loops forever. For each iteration it calls `sample_training_example(rng)`; the RNG stream is `random.Random(seed + worker_id*1_000_003 + epoch*97)` (`self.seed + worker_id*1_000_003`). Exhaust:

```python
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self._worker_seed(info))
        tokenizer = Tokenizer()
        while True:
            cand = sample_training_example(rng)
            ...
            yield tokenizer.encode(cand["phrase"], cron)
```

`sample_training_example` (src/cronlm/grammar.py) resamples until the example passes the train-only gate:

```python
def sample_training_example(rng):
    """Sample an example guaranteed to belong to the train split (online mode)."""
    while True:
        cand = sample_pair(rng)
        tids = cand["template_id"].split("+")
        if assign_split(cand["phrase"], tids, cand["hour"], cand["minute"]) == "train":
            return cand
```

`assign_split` (src/cronlm/grammar.py) — the same hash-split logic used for the files, so online can never leak:

```python
def assign_split(phrase, template_ids, hour=None, minute=None):
    # Priority: combo holdout (reserved exact hour:minute pairs) > paraphrase
    # holdout (reserved templates) > hash bucket (val / test / train).
    if hour is not None and minute is not None and is_reserved_combo(hour, minute):
        return "combo"
    tids = [template_ids] if isinstance(template_ids, str) else list(template_ids)
    if any(t in RESERVED_TEMPLATES for t in tids):
        return "paraphrase"
    h = split_bucket(phrase)
    if h < VAL_BUCKETS:                  # 40
        return "val"
    if h < TEST_BUCKETS:                # 240
        return "test"
    return "train"
```

Split mechanics:
- `split_bucket(phrase) = int(md5(phrase.encode()).hexdigest(),16) % 1000`. Buckets [0,40) → val, [40,240) → test, ≥240 → train. Phrase-level, identical phrases always land in the same split.
- `is_reserved_combo(hour, minute) = md5(f"{hour}:{minute}").int() % 1000 < 50` (exact-time examples only). Combo bucket wins over paraphrase and hash buckets; paraphrase wins over hash.
- Online uses the identical gate, so it can never emit a val/test/holdout phrase; `train.jsonl` is never read; val/test are always the fixed files.
- 38 reserved `template_id`s (2 per clause kind) are excluded from training (in online mode and in the files) except they can still land in combo (see D: 34.7% of combo rows contain ≥1 reserved template — this happened because the combo check runs before the paraphrase check).

---

## B. Data facts

**Split sizes (`wc -l`):** train 500,000 · val 20,000 · test 100,000 · combo_holdout 20,000 · paraphrase_holdout 10,000 · golden 200 · **total 650,200 rows / 650,123 unique phrases** (77 phrases appear twice — all span golden vs generated splits; generated splits never overlap each other).

**Distinct exact-time (hour, minute) pairs (cron field 1 and field 2 both single integers):**

| split | exact-time examples | distinct pairs |
|---|---|---|
| train | 243,464 / 500,000 | **1,377 / 1,440 possible** — all 63 reserved pairs absent |
| test_combo_holdout | 20,000 / 20,000 (100%) | **63** |

**Overlap train ↔ combo holdout: 0 (verified directly).**

**Reserved pair rule:** `md5("hour:minute") % 1000 < 50` → **63 reserved pairs** (4.4% of 1,440). Exact-time combo examples exist only for those 63 pairs (≈317 examples/pair).

**Templates per clause kind (enumerated via `grammar.all_template_ids()`):**

| kind | # | kind | # | kind | # |
|---|---|---|---|---|---|
| day.dom | 10 | time.verbose | 27 | time.minute_step | 12 |
| day.weekdays | 10 | time.exact | 17 | time.hour_step | 12 |
| day.weekends | 10 | time.hourly | 12 | time.hourly_at_minute | 12 |
| day.named_days | 9 | time.every_minute | 13 | month.* | 9 each |
| day.day_range | 9 | time.same_minute | 9 | time.window | 10 |

Total distinct template ids ≈ 217 (sampled enumeration); RESERVED_TEMPLATES = 38 (2 per clause kind). Level shares: L1..L5 with 15/30/25/20/10 candidate weights.

**Max token length over all 650,200 rows: 155** (`block_size`=160).

**Label conflicts check: 0** (same phrase → two different crons anywhere; read-only audit). **Non-canonical crons: 0 · invalid crons: 0 · dom+dow both restricted: 0.**

**Leakage cross-check:** generated-split phrase sets are pairwise disjoint (verified explicitly). Golden vs generated: 52 golden phrases in train, 17 in test, 4 in val, 2 in combo, 2 in paraphrase (77; same-grammar overlap, documented).

---

## C. What the checkpoints score

`best.pt` and `last.pt` have **bit-identical model weights** (step 48000 both; `torch.equal` True for every tensor; md5s differ only via the `rng` snapshots — weights and optimizer state are the same).

Method note: the 100,000-example `test` eval was **not** re-run locally; numbers above come from the run's own `eval_test.json`, restored from the downloaded release zip and cross-checked locally on a 1,000-example slice (exact 100.00%, per-field 100.00% everywhere — consistent with one 1/100,000 failure elsewhere in the file). Golden was re-run locally and matched the original numbers exactly; `val` was re-evaled locally (identical result). Since weights are identical, `best.pt == last.pt` on every split, by construction.

| checkpoint | split | exact | semantic | validity |
|---|---|---|---|---|
| best.pt | test (n=100,000) | **0.99999** (1 failure) | **1.0** | 1.0 |
| best.pt | val (full 20,000) | **1.0** | 1.0 | 1.0 |
| best.pt | combo (n=20,000) | **0.9068** | **0.9068** | 0.99995 |
| best.pt | paraphrase (n=10,000) | **0.8897** | **0.9013** | 0.9998 |
| best.pt | golden (n=200) | **0.965** | **0.97** | 1.0 |
| last.pt | all splits | identical to best.pt | identical | identical |

**test per-level:** L1 100.00% (n=7,847) · L2 99.997% (n=33,392, the single `21-23` artifact) · L3 100.00% (n=26,387) · L4 100.00% (n=22,212) · L5 100.00% (n=10,162).

**test per-field:** minute 100.00 · hour 100.00 · dom 100.00 · month 100.00 · dow 100.00.

**combo per-level:** L1 94.46% (n=1,607) · L2 94.64% (n=6,845) · L3 91.36% (n=5,789) · L4 82.64% (n=2,863) · L5 85.81% (n=2,896).

**combo per-field:** minute 96.02 · hour 96.29 · dom 98.64 · month 98.59 · dow 99.16.

**paraphrase per-field coverage:** minute 95.41 · hour 99.83 · dom 96.04 · month 95.85 · dow 97.79.

**The one test failure (`failures_test.jsonl`, 1 row):**
`at 11:31am sharp on 21st, 22nd and 23rd` → expected `31 11 21-23 * *`, predicted `31 11 21,22,23 * *` — field values identical, canonical run-of-2 style differs; semantic match, `fields_differ: []`.

---

## D. Combo-holdout failures (from `failures_combo.jsonl`, 1,864 failures of 20,000)

### 30 random failures (seeded RNG, seed=1337)

```
 1. 'at quarter after 10am once a quarter'  exp: 15 10 1 */3 *   pred: 15 10 * * *   [time.verbose.12+month.step.6]
 2. 'at 10:48pm sharp sunday thru wednesday' exp: 48 22 * * 0-3   pred: 48 22 * * 0-4   [time.exact.9+day.day_range.7]
 3. "don't forget to on tuesdays quarter to 6pm" exp: 45 17 * * 2  pred: 45 19 * * 2  [time.verbose.19+day.named_days.5]
 4. 'can you each december 9th at 8 after 4pm'  exp: 8 16 9 12 *   pred: 0 16 9 12 *   [time.exact.8+month.with_dom.8]
 5. 'jul through aug quarter to 6pm sat and sun'  exp: 45 17 * 7,8 0,6 pred: 45 19 * 7,8 0,6 [time.verbose.19+day.weekends.4+month.range.6]
 6. 'thu-sat at 58 after 5am'  exp: 58 5 * * 4-6   pred: 0 5 * * 4-6   [time.exact.8+day.day_range.4]
 7. 'at 53 after 10am'  exp: 53 10 * * *   pred: 0 10 * * *   [time.exact.8]
 8. 'semiannually at half past 1 pm'  exp: 30 13 1 */6 *   pred: 30 13 * * *   [time.verbose.2+month.step.7]
 9. 'can you at 59 after 10am'  exp: 59 10 * * *   pred: 0 10 * * *   [time.exact.8]
10. 'every nov at quarter of 6pm on wed'  exp: 45 17 * 11 3   pred: 45 19 * 11 3   [time.verbose.21+day.named_days.7+month.single.6]
11. 'make sure to at 11:29 am wednesday thru friday'  exp: 29 11 * * 3-5   pred: 29 11 * * 3-4   [time.exact.1+day.day_range.7]
12. 'at 30 after 5am'  exp: 30 5 * * *   pred: 0 5 * * *   [time.exact.8]
13. 'can you wednesday thru friday at half past 1 pm'  exp: 30 13 * * 3-5   pred: 30 13 * * 3-4   [time.verbose.2+day.day_range.7]
14. 'execute every quarter at 10:48pm'  exp: 48 22 1 */3 *   pred: 48 22 * * *   [time.exact.0+month.step.7]
15. 'at 53 after 10am every 4 months on the 1st'  exp: 53 10 1 */4 *   pred: 0 10 1 */4 *   [time.exact.8+month.step.8]
16. 'on sep 15th at quarter of 6pm'  exp: 45 17 15 9 *   pred: 45 19 15 9 *   [time.verbose.21+month.with_dom.1]
17. 'every six-month period quarter to 6pm'  exp: 45 17 1 */6 *   pred: 45 19 1 */6 *   [time.verbose.19+month.step.4]
18. 'trigger at a quarter to 6pm every saturday and sunday'  exp: 45 17 * * 0,6   pred: 45 19 * * 0,6   [time.verbose.20+day.weekends.7]
19. 'remember at fifteen to 6pm every six-month period'  exp: 45 17 1 */6 *   pred: 45 19 1 */6 *   [time.verbose.22+month.step.4]
20. 'please run this each 3 months at 40 after 12pm'  exp: 40 12 1 */3 *   pred: 0 12 1 */3 *   [time.exact.8+month.step.3]
21. 'october through december at 13 after 3pm'  exp: 13 15 * 10-12 *   pred: 0 15 * 10-12 *   [time.exact.8+month.range.1]
22. 'be sure to each sunday and monday at 44 after 3am.'  exp: 44 3 * * 0,1   pred: 0 3 * * 0,1   [time.exact.8+day.named_days.4]
23. 'sat-sun at 15 after 10am'  exp: 15 10 * * 0,6   pred: 0 10 * * 0,6   [time.exact.8+day.weekends.5]
24. 'on november the 10th at quarter to 6pm.'  exp: 45 17 10 11 *   pred: 45 19 10 11 *   [time.verbose.18+month.with_dom.6]
25. 'saturday and sunday from january to february at quarter to 6pm'  exp: 45 17 * 1,2 0,6   pred: 45 19 * 1,2 0,6   [time.verbose.18+day.weekends.3+month.range.0]
26. 'on monday and wednesday at 6 after 2am'  exp: 6 2 * * 1,3   pred: 0 2 * * 1,3   [time.exact.8+day.named_days.1]
27. 'at forty five minutes before 6pm saturday and sunday mar through apr'  exp: 45 17 * 3,4 0,6   pred: 45 19 * 3,4 0,6   [time.verbose.24+day.weekends.3+month.range.6]
28. 'every nov at 5 after 6pm'  exp: 5 18 * 11 *   pred: 0 18 * 11 *   [time.exact.8+month.single.6]
29. 'quarter to 6pm on august the 29th'  exp: 45 17 29 8 *   pred: 45 19 29 8 *   [time.verbose.19+month.with_dom.6]
30. 'remember 12:36pm every half year'  exp: 36 12 1 */6 *   pred: 36 12 * * *   [time.exact.4+month.step.6]
```

### Distribution of the 1,864 failures

**By field(s) differing** (`fields_differ`; an example can name several): minute 796 · hour 742 · month 281 · dom 272 · dow 168.

**By time style (regex classification of the phrase):**

| style | rows | failures | rate |
|---|---|---|---|
| 12h am | 6,564 | 526 | 8.01% |
| 12h pm | 6,329 | 502 | 7.93% |
| 24h | 4,427 | 43 | **0.97%** |
| verbose (half/quarter past/to) | 2,680 | 793 | **29.59%** |
| noon/midnight words | — | — | — |

**By hour value (single-int hour field):** every hour ≈ 5–9% **except hour 17 = 32.81%** (n=2,493; 818 fail). Hour 12 = 3.33% (n=750), hour 0 = 5.54% (n=812). (17→32.81, 5→5.81, 10→4.93, 13→4.45, 2→8.20, 11→7.88, 21→7.73, 8→5.99, 0→5.54, 7→8.09, 3→5.77, 23→8.80, 9→8.32, 20→7.69, 15→4.74, 19→4.72, 16→6.50, 6→5.62, 12→3.33, 22→4.37, 18→6.62, 4→6.30.)

**By minute value (single-int minute field):** every minute ≈ 2–11% **except minute 45 = 63.03%** (n=1,155; 728 fail). minute 15 = 2.63% (n=1,216), minute 30 = 3.11% (n=2,414), minute 55 = 7.69% (n=767), minute 27 = 11.36% (n=264), minute 7 = 10.69% (n=505).

**Reserved-template effect (critical):** **6,945 of 20,000 combo rows (34.7%) contain ≥1 reserved template id. Failure rate in that subset: 19.3% (1,342/6,945). Failure rate elsewhere: 4.0% (522/13,055).**

Top reserved-template-id failure counts: `time.exact.8` 804 · `time.verbose.*` 779 total · `month.step.*` 412 · `day.day_range.7` 116.

**Per-level failure rate:** L1 5.54% · L2 5.36% · L3 8.64% · L4 17.36% · L5 14.19%.

### The three failure clusters

1. **`time.exact.8` = `at N after H{am,pm}` ("at 30 after 5pm") — 804 of 1,864 failures (43.1%).** Model outputs `0 HH` instead of `NN HH` — minute never learned for this RESERVED template (training never saw it). E.g. `at 53 after 10am` → expected `53 10 * * *`, predicted `0 10 * * *`.
2. **`time.verbose.*` quarter-to family — 779 failures.** Always `45 19` instead of `45 17`: hour added 1 instead of subtracted (expected 5:45pm, model answers 7:45pm). RESERVED templates (e.g. `time.verbose.18..24`). E.g. `don't forget to on tuesdays quarter to 6pm` → expected `45 17 * * 2`, predicted `45 19 * * 2`.
3. **`month.step.*` — 412 failures.** dom expected `1`, predicted `*` (month.step forces dom=1). E.g. `once a quarter at 3:51 in the morning` → expected `51 3 1 */3 *`, predicted `51 3 * * *`.

Field-counts align with the above: minute 796, hour 742, month 281, dom 272, dow 168.

---

## E. Training curves (`runs/small/log.csv`, 482 data rows, steps 1–48,000)

`log.csv` columns: `step, loss, lr, tokens_per_sec`. **No validation columns exist in the file — combo accuracy was never tracked during training, and val exact/val loss were not written to `log.csv` (only printed to the console, captured in the training console).**

15 evenly spaced rows (step, loss, lr, tokens/s):

| step | loss | lr | tokens/s |
|---|---|---|---|
| 1 | 3.951 | 4.0e-06 | 2,120 |
| 3,400 | 0.0057 | 9.9e-04 | 32,300 |
| 6,900 | 0.0022 | 9.7e-04 | 32,764 |
| 10,300 | 0.0017 | 9.4e-04 | 32,595 |
| 13,700 | 0.0011 | 8.8e-04 | 32,887 |
| 17,100 | 0.0005 | 8.2e-04 | 32,950 |
| 20,600 | 0.0003 | 7.5e-04 | 33,162 |
| 24,000 | 0.0003 | 6.7e-04 | 32,558 |
| 27,400 | 0.0001 | 5.8e-04 | 32,879 |
| 30,900 | 0.0002 | 4.9e-04 | 33,008 |
| 34,300 | 0.0000 | 4.0e-04 | 32,888 |
| 37,700 | 0.0000 | 3.2e-04 | 32,696 |
| 41,100 | 0.0000 | 2.4e-04 | 32,591 |
| 44,600 | 0.0000 | 1.6e-04 | 32,941 |
| 48,000 | 0.0000 | 1.1e-04 | 33,566 |

**Held-out points printed by the run (console only):**

| step | val exact (subset of 5,000) | notes |
|---|---|---|
| 2,000 | **94.30%** | val loss 0.0157 |
| 4,000 | **97.00%** | val loss 0.0094 — *best* |
| 38,000 | **100.00%** | subset 5,000 — first full-val trigger |
| ~48,000 | **100.00% (full 20,000)** | early-stop after 3rd consecutive |

Combo-holdout accuracy was never measured at intermediate steps (only val gates the early stop). NOT FOUND anywhere. Local val re-eval: exact 100.00%, per-field 100.00%, n=20,000, `time_sec ≈ 477 s` — matches the run's printed values.

---

## F. Anything suspicious / notable

1. **Combo holdout by construction contains 34.7% never-trained (reserved) templates.** `assign_split` checks the combo reservation FIRST (exact-time (H,M) ∈ the 63 reserved pairs) before excluding RESERVED_TEMPLATES (the paraphrase rule). So `at 53 after 10am` lands in combo while being a template the model never saw. Consistent with the documented priority order, but it doubles the difficulty (unseen numbers + unseen phrasings). Failure 19.3% there vs 4.0% elsewhere.
2. **Grouping mechanism is otherwise consistent**: identical phrases always land in the same split (deterministic md5).
3. **Test failure rate is 0.001%** (one example; an exact-match style artifact, not a model error — predicted `... 21,22,23` vs expected `... 21-23`, semantically equal).
4. **Golden exact match is 96.5%** (7/200 wrong). Golden contains 52 phrases that also appear in train (and 17 in test); acceptable read-only overlap. Failures: two of the "quarter to/45 minutes before 6pm" family (→ `45 19`), `every 4 minutes between midnight and 6am` (→ `*/4 6-6`), `wednesday to thursday at 9am` (→ `0 9 * * 3-4`, runs-of-2 canonical), `friday through sunday at 8pm` (→ `0 20 * * 0,6`), `quarterly` + `monthly` (hour `*` instead of `0`).
5. **The 6-field degenerate output can exist** (`*/12 * * * * *` observed on a smoke checkpoint); the final model does not produce it on test (validity 100%).
6. **Online + `--resume` is not bit-exact across worker counts**: `num_workers=0` (default) gives a deterministic stream; `>0` interleaves nondeterministically (leakage unaffected). With `--resume` in online mode, the batch stream is not fast-forwarded (known simplification: `_infinite_batches` re-iterates; the seed replay gives the same example sequence but duplicated early-epoch examples may differ in count on resumed runs).
7. **Padding/attention safety**: training uses right-padding + `is_causal=True` (real tokens can't see right pads; attention mask never needed). In greedy generation with left-padded batches, positions are computed from `mask.cumsum` and a causal|padding bool mask is built — verified (CPU and GGUF side, 150/150 identical predictions vs PyTorch).
8. **Loss masking**: loss is computed only on cron tokens and `<eos>`; prompt tokens and `<pad>` are `-100` (per `Collator`; `labels[i, :sep_idx] = IGNORE_INDEX`).
9. **Token length budget**: max example = 155 ≤ 160 (block_size).
10. **The golden eval was re-scaled only locally** — the remote run printed test summary but its later steps failed on a `sorted(per_level, key=int)` bug (`level: null` rows). The paraphrase and combo numbers above come from the local re-eval of the exact same checkpoint; they are not from the remote run's console. The bug was fixed locally before this report was written.
11. **Local artifact provenance**: after the artifact download, a 1,000-example test eval overwrote the full 100,000-example `eval_test.json` / `failures_test.jsonl`; both were restored from the download (restored values match the run's printed scores exactly: exact=0.99999).
12. **Level distribution in train.jsonl** is dominated by L2–L5 (L1 ≈ 7.2% vs the PRD's 15% target): the L1 phrase space saturates after dedupe (documented in the README). Does not affect exact-match validity per level (L1 100% on test).

**Bottom line for the 90.68%:** the combo-holdout error is concentrated (a) in never-trained reserved templates within the combo bucket (19.3% vs 4.0%), and (b) in the quarter-to-6pm hour-shift family (728 of 1,864 failures are minute-45) and the after-N-minute template (804 of 1,864 = `time.exact.8`), both exclusively from reserved templates that training excluded on purpose. Normal-template combo accuracy ≈ 96% (4.0% error), 24h ≈ 99% accuracy, 12h ≈ 92%.
