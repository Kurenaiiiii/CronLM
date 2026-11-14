"""Datasets and collate with loss masking (Section 8.1, 8.2).

Two ways to feed training, both required (Section 5):

1. FileCronDataset — reads a fixed JSONL file (train.jsonl / val.jsonl).
2. OnlineCronDataset — infinite IterableDataset that calls the generator every
   step for a fresh random batch. It never reads train.jsonl, excludes every
   phrase that belongs to the val/test/holdout hash buckets (same split logic
   as the files), and can be given background workers.
"""

import json
import os
import random

import torch
from torch.utils.data import Dataset, IterableDataset

from .grammar import sample_training_example, assign_split
from .tokenizer import BLOCK_SIZE as TOKENIZER_BLOCK_SIZE, Tokenizer

PAD_ID = 0
IGNORE_INDEX = -100


def encode_example(tokenizer, phrase, cron):
    """Encode one (phrase, cron) pair into token ids (<bos>..<eos>)."""
    return tokenizer.encode(phrase, cron)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class FileCronDataset(Dataset):
    """Fixed-file dataset. For val evaluation pass ``limit=5000``."""

    def __init__(self, path, tokenizer, limit=None):
        self.tokenizer = tokenizer
        self.rows = list(read_jsonl(path))
        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return self.tokenizer.encode(row["phrase"], row["cron"])


class OnlineCronDataset(IterableDataset):
    """Infinite dataset of fresh generator samples (train --online).

    Never reads train.jsonl. Leaks nothing: every candidate is passed through
    the same hash-split filter used to build the files, so phrases assigned to
    the val/test/holdout buckets (or reserved templates/combos) are dropped
    and re-sampled.
    """

    def __init__(self, seed=1337, max_phrase_len=TOKENIZER_BLOCK_SIZE):
        self.seed = seed
        self.max_phrase_len = max_phrase_len

    def _worker_seed(self, info):
        base = self.seed + (info.worker_id if info is not None else 0) * 1_000_003
        epoch = getattr(self, "_epoch", 0)
        return base + epoch * 97

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self._worker_seed(info))
        tokenizer = Tokenizer()
        while True:
            cand = sample_training_example(rng)
            cron = cand["cron"]
            if len(cand["phrase"]) + len(cron) + 4 > self.max_phrase_len:
                continue
            yield tokenizer.encode(cand["phrase"], cron)


class Collator:
    """Dynamic padding to the longest sequence in the batch + loss mask.

    labels = input shifted by one; positions whose target is the prompt
    (tokens up to and including <sep>) or padding are set to IGNORE_INDEX.
    """

    def __init__(self, tokenizer):
        self.sep_id = tokenizer.sep_id
        self.pad_id = tokenizer.pad_id

    def __call__(self, batch_ids):
        max_len = max(len(ids) for ids in batch_ids)
        width = max_len - 1
        B = len(batch_ids)
        input_ids = torch.full((B, width), self.pad_id, dtype=torch.long)
        labels = torch.full((B, width), IGNORE_INDEX, dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            sep_idx = ids.index(self.sep_id)
            seq = ids[:-1]
            tgt = ids[1:]
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            labels[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
            # y[j] has target ids[j+1]; keep it only when the target is in the
            # cron region (j+1 > sep_idx), i.e. zero out j < sep_idx.
            labels[i, :sep_idx] = IGNORE_INDEX
        return input_ids, labels


def collate_fn(tokenizer):
    return Collator(tokenizer)
