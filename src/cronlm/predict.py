"""Prediction helpers: greedy decoding + next-run preview (Section 10)."""

import os
from datetime import datetime

import torch

from .cron_spec import is_valid_cron, next_runs, CronError
from .evaluate import load_checkpoint, build_model
from .tokenizer import Tokenizer


@torch.no_grad()
def predict_cron(model, tokenizer, phrase, device, max_new_tokens=48):
    model.eval()
    prompt_ids = tokenizer.encode_prompt(phrase)  # raises on bad characters
    gen = model.generate([prompt_ids], max_new_tokens=max_new_tokens,
                         eos_id=tokenizer.eos_id)[0]
    return tokenizer.decode(gen)


def format_next_runs(cron, count=5):
    now = datetime.now()
    runs = next_runs(cron, now, count=count)
    return ", ".join(r.strftime("%a %Y-%m-%d %H:%M") for r in runs)


def predict(ckpt_path, phrase, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(ckpt_path, device)
    run_dir = os.path.dirname(os.path.abspath(ckpt_path))
    tokenizer = Tokenizer.load(os.path.join(run_dir, "vocab.json"))
    model = build_model(ckpt, device)

    cron = predict_cron(model, tokenizer, phrase, device)
    valid = is_valid_cron(cron)
    print(f"phrase : {phrase}")
    print(f"cron   : {cron}   ({'valid' if valid else 'INVALID'})")
    if valid:
        try:
            print(f"next 5 : {format_next_runs(cron, 5)}")
        except (CronError, RuntimeError) as exc:
            print(f"next 5 : (scheduler error: {exc})")
    return cron


def repl(ckpt_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(ckpt_path, device)
    run_dir = os.path.dirname(os.path.abspath(ckpt_path))
    tokenizer = Tokenizer.load(os.path.join(run_dir, "vocab.json"))
    model = build_model(ckpt, device)
    print(f"CronLM repl (ckpt: {ckpt_path}). Type a phrase, or 'quit'.")
    while True:
        try:
            phrase = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not phrase or phrase in {"quit", "exit"}:
            break
        try:
            cron = predict_cron(model, tokenizer, phrase, device)
        except ValueError as exc:
            print(f"error  : {exc}")
            continue
        valid = is_valid_cron(cron)
        print(f"cron   : {cron}   ({'valid' if valid else 'INVALID'})")
        if valid:
            try:
                print(f"next 5 : {format_next_runs(cron, 5)}")
            except (CronError, RuntimeError) as exc:
                print(f"next 5 : (scheduler error: {exc})")
