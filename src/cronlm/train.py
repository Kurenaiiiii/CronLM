"""Training loop (Section 8): AdamW with no-decay on norms/embeddings, linear
warmup then cosine decay to 1e-5, grad clip 1.0, fp32 by default, fp16/bf16
opt-in, checkpoint/resume, fixed-subset eval with greedy decoding.
"""

import csv
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
import yaml

from .data import FileCronDataset, OnlineCronDataset, Collator, read_jsonl
from .evaluate import generate_crons
from .model import GPT
from .tokenizer import Tokenizer

FINAL_LR = 1e-5


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    cfg.setdefault("seed", 1337)
    cfg.setdefault("warmup_steps", 500)
    cfg.setdefault("eval_interval", 2000)
    cfg.setdefault("log_interval", 100)
    cfg.setdefault("eval_subset", 5000)
    cfg.setdefault("grad_accum_steps", 1)
    cfg.setdefault("dropout", 0.0)
    return cfg


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return requested


def _lr_at(step, steps, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return FINAL_LR + (peak - FINAL_LR) * cosine


def _make_optimizer(model, lr, betas=(0.9, 0.95), weight_decay=0.01):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim < 2 or "emb" in name or "ln" in name or "norm" in name.lower():
            no_decay.append(p)   # biases, LayerNorm gains, embeddings
        else:
            decay.append(p)      # Linear weights (2D, non-embedding)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def _infinite_batches(loader):
    """Yield batches forever, re-entering the loader for each new epoch.

    With a seeded generator, each new pass reshuffles identically across runs,
    so recreating the stream (and fast-forwarding it on --resume) reproduces
    the exact batch order of an uninterrupted run.
    """
    while True:
        for batch in loader:
            yield batch


def run_train(args):
    cfg = load_config(args.config)
    name = cfg["name"]
    seed = cfg["seed"]
    device = resolve_device(args.device)

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"device: cuda — {props.name} (VRAM: {props.total_memory // (1024 * 1024)} MB)")
    else:
        print("device: cpu")

    amp = args.amp
    amp_dtype = {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    if amp_dtype is None and amp != "none":
        raise RuntimeError(f"unknown --amp value {amp!r} (expected none|fp16|bf16)")
    if amp == "bf16":
        if device != "cuda":
            raise RuntimeError("--amp bf16 requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "--amp bf16 requested, but this GPU has no bf16 support."
                "Use default fp32 or --amp fp16.")
    if amp == "fp16" and device != "cuda":
        raise RuntimeError("--amp fp16 requires CUDA")
    use_amp = amp_dtype is not None
    print(f"precision: {amp}")

    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    run_dir = os.path.join("runs", name)
    os.makedirs(run_dir, exist_ok=True)
    tokenizer = Tokenizer()
    tokenizer.save(os.path.join(run_dir, "vocab.json"))

    model = GPT(d_model=cfg["d_model"], n_layer=cfg["n_layer"],
                n_head=cfg["n_head"], ff_dim=cfg["ff_dim"],
                block_size=cfg.get("block_size", 160),
                vocab_size=len(tokenizer),
                dropout=cfg.get("dropout", 0.0)).to(device)
    n_params = model.num_params()
    print(f"parameters: {n_params:,}")

    optimizer = _make_optimizer(model, lr=cfg["peak_lr"])
    use_fp16 = (amp == "fp16")
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    steps = 200 if args.smoke else cfg["steps"]
    warmup = 20 if args.smoke else cfg["warmup_steps"]
    eval_interval = 50 if args.smoke else cfg["eval_interval"]
    log_interval = 20 if args.smoke else cfg["log_interval"]
    eval_subset = 512 if args.smoke else cfg["eval_subset"]
    accum = cfg.get("grad_accum_steps", 1)

    # -- data
    collate = Collator(tokenizer)
    num_workers = getattr(args, "num_workers", 0)
    if args.online:
        dataset = OnlineCronDataset(seed=seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg["batch_size"], num_workers=num_workers,
            collate_fn=collate,
            persistent_workers=num_workers > 0)
        print("data: --online (fresh generator samples every step)")
    else:
        train_path = os.path.join("data", "train.jsonl")
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"{train_path} not found — run gen-data first "
                f"(python -m cronlm.cli gen-data --out data --train 500000)")
        limit = 5000 if args.smoke else None
        dataset = FileCronDataset(train_path, tokenizer, limit=limit)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg["batch_size"], shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            drop_last=True, collate_fn=collate, num_workers=0)
        print(f"data: {train_path} ({len(dataset)} examples)"
              + (" [smoke: first 5000]" if args.smoke else ""))

    # -- resume
    start_step = 0
    best = {"exact": 0.0, "loss": float("inf")}
    consecutive_perfect = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        if scaler is not None and resume_ckpt.get("scaler") is not None:
            scaler.load_state_dict(resume_ckpt["scaler"])
        start_step = resume_ckpt["step"]
        best = resume_ckpt.get("best", best)
        consecutive_perfect = resume_ckpt.get("consecutive_perfect", 0)
        random.setstate(resume_ckpt["rng"]["random"])
        torch.set_rng_state(resume_ckpt["rng"]["torch"])
        if device == "cuda" and resume_ckpt["rng"].get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(resume_ckpt["rng"]["torch_cuda"])
        print(f"resumed from {args.resume} at step {start_step} "
              f"(best val exact so far: {best['exact'] * 100:.2f}%)")
        it = _infinite_batches(loader)
        if not args.online:
            # Fast-forward the deterministic batch stream past the consumed
            # samples so the resumed run sees the same batches as an
            # uninterrupted run.
            t_ff = time.time()
            for _ in range(start_step):
                next(it)
            print(f"fast-forwarded {start_step} batches in {time.time() - t_ff:.1f}s")
    else:
        it = _infinite_batches(loader)

    # log.csv
    log_path = os.path.join(run_dir, "log.csv")
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["step", "loss", "lr", "tokens_per_sec"])

    def save_checkpoint(step, filename):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "name": name,
            "config": {k: cfg[k] for k in ("d_model", "n_layer", "n_head", "ff_dim",
                                           "block_size", "dropout")} | {
                           "vocab_size": len(tokenizer)},
            "best": best,
            "consecutive_perfect": consecutive_perfect,
            "rng": {
                "random": random.getstate(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            },
        }
        torch.save(state, os.path.join(run_dir, filename))

    def run_eval(step):
        nonlocal consecutive_perfect, best
        model.eval()
        val_path = os.path.join("data", "val.jsonl")
        if not os.path.exists(val_path):
            print("val.jsonl not found; skipping eval")
            model.train()
            return None
        val_rows = list(read_jsonl(val_path))[:eval_subset]

        # val loss
        total_loss, total_n = 0.0, 0
        ds = FileCronDataset(val_path, tokenizer, limit=eval_subset)
        dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"],
                                         collate_fn=collate)
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, targets=y)
                n = int((y != -100).sum())
                total_loss += loss.item() * n
                total_n += n
        val_loss = total_loss / max(total_n, 1)

        # greedy exact match
        prompts = [r["phrase"] for r in val_rows]
        refs = [r["cron"] for r in val_rows]
        preds = generate_crons(model, tokenizer, prompts, device,
                               batch_size=128, max_new_tokens=48)
        exact = sum(1 for p, r in zip(preds, refs) if p == r) / len(refs)

        improved = exact > best["exact"] or (exact == best["exact"] and val_loss < best["loss"])
        is_best = improved
        if exact > best["exact"]:
            best = {"exact": exact, "loss": val_loss}
        elif exact == best["exact"] and val_loss < best["loss"]:
            best = {"exact": exact, "loss": val_loss}

        save_checkpoint(step, "last.pt")
        if is_best:
            save_checkpoint(step, "best.pt")
        print(f"eval @ step {step}: val exact {exact * 100:.2f}%  "
              f"val loss {val_loss:.4f}  (subset of {len(val_rows)})"
              + ("  *new best*" if is_best else ""))

        # early stop: full val at 100% three times in a row (not in smoke)
        if not args.smoke and exact >= 1.0:
            full_rows = list(read_jsonl(val_path))
            full_prompts = [r["phrase"] for r in full_rows]
            full_refs = [r["cron"] for r in full_rows]
            full_preds = generate_crons(model, tokenizer, full_prompts, device,
                                        batch_size=256, max_new_tokens=48)
            full_exact = sum(1 for p, r in zip(full_preds, full_refs) if p == r) / len(full_refs)
            print(f"full val exact: {full_exact * 100:.2f}% ({len(full_rows)} examples)")
            if full_exact >= 1.0:
                consecutive_perfect += 1
            else:
                consecutive_perfect = 0
        elif exact < 1.0:
            consecutive_perfect = 0

        model.train()
        return {"exact": exact, "loss": val_loss, "consecutive_perfect": consecutive_perfect}

    model.train()
    t0 = time.time()
    tokens_since_log = 0
    last_log_t = t0
    loss_accum = 0.0
    log_count = 0
    step = start_step
    try:
        while step < steps:
            step += 1
            micro_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for micro in range(accum):
                x, y = next(it)
                x, y = x.to(device), y.to(device)
                tokens_since_log += int((y != -100).sum())
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    _, loss = model(x, targets=y)
                    loss = loss / accum
                if use_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                micro_loss += loss.item()
            lr = _lr_at(step, steps, warmup, cfg["peak_lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            if use_fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            loss_accum += micro_loss
            log_count += 1

            if step % log_interval == 0 or step == 1:
                now = time.time()
                dt = max(now - last_log_t, 1e-9)
                tps = tokens_since_log / dt
                avg_loss = loss_accum / max(log_count, 1)
                print(f"step {step:>6}/{steps}  loss {avg_loss:.4f}  lr {lr:.2e}  "
                      f"tokens/s {tps:.0f}")
                log_writer.writerow([step, f"{avg_loss:.6f}", f"{lr:.3e}", f"{tps:.1f}"])
                log_file.flush()
                tokens_since_log = 0
                last_log_t = now
                loss_accum = 0.0
                log_count = 0

            if step % eval_interval == 0 or step == steps:
                result = run_eval(step)
                if result is not None and not args.smoke and \
                        result["consecutive_perfect"] >= 3:
                    print("val exact match 100% on the full val set for 3 "
                          "consecutive evals — early stop")
                    break
    except KeyboardInterrupt:
        save_checkpoint(step, "last.pt")
        log_file.close()
        print(f"\ninterrupted at step {step}; checkpoint saved to runs/{name}/last.pt")
        print(f"resume with: python -m cronlm.cli train --config {args.config} "
              f"--device {args.device} --resume runs/{name}/last.pt")
        return

    save_checkpoint(step, "last.pt")
    log_file.close()
    dt = time.time() - t0
    print(f"\ntraining done in {dt / 60:.1f} min. best val exact: {best['exact'] * 100:.2f}%")
    print(f"checkpoints: runs/{name}/last.pt, runs/{name}/best.pt")
