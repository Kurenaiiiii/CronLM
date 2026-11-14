"""Evaluation: exact match, semantic match, validity, per-level/template/field
breakdowns, failure dump (Section 9).
"""

import json
import os
import time
from collections import Counter, defaultdict

import torch

from .cron_spec import (canonical_cron, differing_fields, is_valid_cron,
                        semantic_equal, parse_cron, FIELD_NAMES)
from .data import read_jsonl
from .model import GPT
from .tokenizer import Tokenizer

SPLIT_FILES = {
    "test": "test.jsonl",
    "val": "val.jsonl",
    "combo": "test_combo_holdout.jsonl",
    "paraphrase": "test_paraphrase_holdout.jsonl",
    "golden": "golden.jsonl",
}


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def build_model(ckpt, device):
    cfg = ckpt["config"]
    model = GPT(**{k: cfg[k] for k in
                   ("d_model", "n_layer", "n_head", "ff_dim", "block_size", "vocab_size")})
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_crons(model, tokenizer, prompts, device, batch_size=128,
                   max_new_tokens=48, show_progress=False):
    """Greedy-decode cron strings for a list of prompts."""
    out = []
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    if show_progress:
        from tqdm import tqdm
        batches = tqdm(batches)
    for batch in batches:
        prompt_ids = [tokenizer.encode_prompt(p) for p in batch]
        gen = model.generate(prompt_ids, max_new_tokens=max_new_tokens,
                             eos_id=tokenizer.eos_id)
        out.extend(tokenizer.decode(ids) for ids in gen)
    return out


def exact_match_rate(preds, refs):
    if not refs:
        return 0.0
    return sum(1 for p, r in zip(preds, refs) if p == r) / len(refs)


def _field_accuracy(preds, refs):
    correct = Counter()
    total = len(refs)
    for pred, ref in zip(preds, refs):
        try:
            pe = parse_cron(ref)
            pp = parse_cron(pred)
            fields_p = pp.canonical_fields
        except Exception:
            fields_p = None
            try:
                pe = parse_cron(ref)
            except Exception:
                continue
        fields_e = pe.canonical_fields
        for i, name in enumerate(FIELD_NAMES):
            if fields_p is not None and fields_p[i] == fields_e[i]:
                correct[name] += 1
    return {name: (correct[name] / total if total else 0.0) for name in FIELD_NAMES}


def evaluate_checkpoint(ckpt_path, split, data_dir="data", device=None,
                        batch_size=128, max_examples=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(ckpt_path, device)
    run_dir = os.path.dirname(os.path.abspath(ckpt_path))
    run_name = ckpt.get("name", os.path.basename(run_dir))
    tokenizer = Tokenizer.load(os.path.join(run_dir, "vocab.json"))

    path = os.path.join(data_dir, SPLIT_FILES[split])
    if not os.path.exists(path):
        raise FileNotFoundError(f"split file not found: {path} — run gen-data first")
    rows = list(read_jsonl(path))
    if max_examples is not None:
        rows = rows[:max_examples]

    prompts = [r["phrase"] for r in rows]
    refs = [r["cron"] for r in rows]
    levels = [r.get("level", 0) for r in rows]
    tids = [r.get("template_id", "?") for r in rows]

    model = build_model(ckpt, device)
    t0 = time.time()
    preds = generate_crons(model, tokenizer, prompts, device,
                           batch_size=batch_size, show_progress=verbose)
    elapsed = time.time() - t0

    # -- metrics
    exact = [p == r for p, r in zip(preds, refs)]
    semantic = [semantic_equal(p, r) for p, r in zip(preds, refs)]
    valid = [is_valid_cron(p) for p in preds]

    per_level = {}
    for lvl in sorted(set(levels)):
        idx = [i for i, l in enumerate(levels) if l == lvl]
        per_level[str(lvl)] = {
            "n": len(idx),
            "exact": sum(exact[i] for i in idx) / len(idx),
        }

    tid_stats = {}
    tid_n = Counter(tids)
    tid_ok = Counter(t for t, ok in zip(tids, exact) if ok)
    for tid, n in tid_n.items():
        tid_stats[tid] = {"n": n, "exact": tid_ok[tid] / n}

    per_field = _field_accuracy(preds, refs)

    failures = []
    for i in range(len(rows)):
        if not exact[i]:
            failures.append({
                "phrase": prompts[i], "expected": refs[i], "predicted": preds[i],
                "level": levels[i], "template_id": tids[i],
                "fields_differ": differing_fields(refs[i], preds[i]),
            })

    metrics = {
        "split": split,
        "n": len(rows),
        "exact_match": sum(exact) / len(exact),
        "semantic_match": sum(semantic) / len(semantic),
        "validity": sum(valid) / len(valid),
        "per_level": per_level,
        "per_template_id": tid_stats,
        "per_field": per_field,
        "time_sec": elapsed,
    }

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"failures_{split}.jsonl"), "w", encoding="utf-8") as f:
        for fail in failures:
            f.write(json.dumps(fail) + "\n")
    with open(os.path.join(run_dir, f"eval_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        box = [
            "",
            "=" * 56,
            f" CronLM eval — {run_name} — split: {split}",
            "=" * 56,
            f" examples           {metrics['n']:>10,}",
            f" exact match        {metrics['exact_match'] * 100:>9.2f}%",
            f" semantic match     {metrics['semantic_match'] * 100:>9.2f}%",
            f" validity           {metrics['validity'] * 100:>9.2f}%",
            "-" * 56,
        ]
        def _lvl_key(x):
            try:
                return (0, int(x))
            except (TypeError, ValueError):
                return (1, 0)

        for lvl in sorted(per_level, key=_lvl_key):
            d = per_level[lvl]
            box.append(f" level {lvl} exact       {d['exact'] * 100:>9.2f}%  (n={d['n']})")
        box.append("-" * 56)
        for name in FIELD_NAMES:
            box.append(f" field {name:<10}     {per_field[name] * 100:>9.2f}%")
        box.append("=" * 56)
        box.append(f" failures -> runs/{run_name}/failures_{split}.jsonl "
                   f"({len(failures)} rows)")
        print("\n".join(box))

    return metrics
