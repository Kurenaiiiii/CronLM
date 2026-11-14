"""Build the train/val/test/holdout JSONL files from the grammar.

Hard requirements (Section 5.1): no label conflicts (fail loudly), dedupe by
phrase string, hash-based split by ``md5(phrase) % 1000`` with fixed buckets,
plus the reserved combo holdout and the reserved paraphrase holdout.

Split assignment lives in ``grammar.assign_split`` so that online training can
use the exact same function and never leak val/test/holdout phrases.
"""

import json
import os
import random
import time
from collections import Counter, defaultdict

from . import grammar

FILES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
    "combo": "test_combo_holdout.jsonl",
    "paraphrase": "test_paraphrase_holdout.jsonl",
}

DEFAULT_SIZES = {"train": 500_000, "val": 20_000, "test": 100_000,
                 "combo": 20_000, "paraphrase": 10_000}


def _row(cand):
    return {"phrase": cand["phrase"], "cron": cand["cron"],
            "level": cand["level"], "template_id": cand["template_id"]}


def build_splits(out_dir, train_n=DEFAULT_SIZES["train"], val_n=DEFAULT_SIZES["val"],
                 test_n=DEFAULT_SIZES["test"], combo_n=DEFAULT_SIZES["combo"],
                 paraphrase_n=DEFAULT_SIZES["paraphrase"], seed=1337,
                 golden_src="tests/golden.jsonl", verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    targets = {"train": train_n, "val": val_n, "test": test_n,
               "combo": combo_n, "paraphrase": paraphrase_n}

    rng = random.Random(seed)
    buckets = {k: [] for k in FILES}
    counts = Counter()
    level_counts = Counter()
    template_counts = Counter()
    seen = {}                      # phrase -> cron (dedupe + conflict check)
    attempts = 0
    started = time.time()

    # Cap total work so a pathological seed cannot loop forever. The factor is
    # generous: buckets get skipped mid-run (already full), so we need a lot of
    # extra samples to fill train.
    max_attempts = sum(targets.values()) * 30 + 1000

    while any(len(buckets[k]) < targets[k] for k in targets) and attempts < max_attempts:
        attempts += 1
        cand = grammar.sample_pair(rng)
        phrase, cron = cand["phrase"], cand["cron"]

        prev = seen.get(phrase)
        if prev is not None:
            if prev != cron:
                # Hard fail: two different labels for the same phrase.
                raise RuntimeError(
                    f"label conflict for phrase {phrase!r}: {prev!r} vs {cron!r}")
            continue
        seen[phrase] = cron

        tids = cand["template_id"].split("+")
        split = grammar.assign_split(phrase, tids, cand["hour"], cand["minute"])
        if len(buckets[split]) >= targets[split]:
            # This split is full; skip the phrase entirely rather than move it
            # to another split (that would leak val/test into train).
            continue

        buckets[split].append(cand)
        counts[split] += 1
        level_counts[(split, cand["level"])] += 1
        for tid in tids:
            template_counts[tid] += 1

    if verbose:
        elapsed = time.time() - started
        print(f"generated {attempts} candidates in {elapsed:.1f}s "
              f"({attempts / max(elapsed, 1e-9):.0f}/s)")
        for split in ("train", "val", "test", "combo", "paraphrase"):
            print(f"  {split:>10}: {len(buckets[split]):>7} examples")

    paths = {}
    for split, filename in FILES.items():
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for cand in buckets[split]:
                f.write(json.dumps(_row(cand), sort_keys=False) + "\n")
        paths[split] = path

    # golden.jsonl is hand-written and hand-verified; gen-data copies it next
    # to the generated splits so `eval --split golden` has a single data dir.
    golden_dst = os.path.join(out_dir, "golden.jsonl")
    if os.path.exists(golden_src):
        with open(golden_src, "r", encoding="utf-8") as src, \
                open(golden_dst, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        paths["golden"] = golden_dst

    if verbose:
        print("\nper-level counts (train):")
        for level in range(1, 6):
            print(f"  L{level}: {level_counts[('train', level)]}")
        print("\ntop templates (all splits):")
        for tid, n in template_counts.most_common(15):
            print(f"  {tid:<28} {n}")

    return paths
