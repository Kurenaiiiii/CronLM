"""Character-level tokenizer for CronLM.

Vocab = 4 special tokens + the sorted set of allowed characters (a-z, 0-9,
space, and ``: , . - ' * /``), 48 tokens in total. Deterministic and saved
inside every checkpoint folder as ``vocab.json`` so inference never depends
on code state.
"""

import json
import os

PAD, BOS, SEP, EOS = "<pad>", "<bos>", "<sep>", "<eos>"
SPECIAL_TOKENS = [PAD, BOS, SEP, EOS]

# Allowed characters in phrases (Section 4.4) plus the cron output alphabet.
ALLOWED_CHARS = sorted(set("abcdefghijklmnopqrstuvwxyz0123456789"
                            " : , . - ' * /"))

BLOCK_SIZE = 160  # max total sequence length, Section 6

class Tokenizer:
    def __init__(self):
        self.tokens = SPECIAL_TOKENS + ALLOWED_CHARS
        self.stoi = {tok: i for i, tok in enumerate(self.tokens)}
        self.itos = {i: tok for i, tok in enumerate(self.tokens)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.sep_id = self.stoi[SEP]
        self.eos_id = self.stoi[EOS]

    def __len__(self):
        return len(self.tokens)

    @property
    def vocab_size(self):
        return len(self.tokens)

    # -- encode / decode -----------------------------------------------------

    def _encode_str(self, text):
        ids = []
        bad = sorted({c for c in text if c not in self.stoi})
        if bad:
            # Raise, never silently drop: the user should see exactly which
            # characters are not in the vocab.
            raise ValueError(
                f"text contains characters outside the vocab: {''.join(bad)!r} "
                f"(in {text!r})")
        for ch in text:
            ids.append(self.stoi[ch])
        return ids

    def encode(self, phrase, cron):
        """Encode the full training sequence: <bos> phrase <sep> cron <eos>."""
        ids = [self.bos_id] + self._encode_str(phrase) + [self.sep_id] + \
              self._encode_str(cron) + [self.eos_id]
        if len(ids) > BLOCK_SIZE:
            raise ValueError(f"sequence of {len(ids)} tokens exceeds block_size "
                             f"{BLOCK_SIZE} for {phrase!r}")
        return ids

    def encode_prompt(self, phrase):
        """Encode just the prompt, up to and including <sep> (for inference)."""
        ids = [self.bos_id] + self._encode_str(phrase) + [self.sep_id]
        if len(ids) > BLOCK_SIZE:
            raise ValueError(f"prompt of {len(ids)} tokens exceeds block_size "
                             f"{BLOCK_SIZE} for {phrase!r}")
        return ids

    def decode(self, ids):
        """Decode ids to text, stripping special tokens and stopping at <eos>."""
        out = []
        for i in ids:
            tok = self.itos.get(int(i))
            if tok is None:
                raise ValueError(f"unknown token id {i}")
            if tok == EOS:
                break
            if tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    # -- persistence ---------------------------------------------------------

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tokens": self.tokens}, f, indent=2)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            tokens = json.load(f)["tokens"]
        if tokens != tok.tokens:
            # Refuse to silently use a different vocab than the checkpoint was
            # trained with.
            raise ValueError(f"vocab in {path} does not match this build of cronlm")
        return tok
