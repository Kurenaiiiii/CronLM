"""Hand-written decoder-only GPT for CronLM. No nanoGPT copy-paste; every
piece is here on purpose so the reader can follow it.

Architecture (Section 7):
- token embedding + learned absolute positional embedding
- N pre-LayerNorm transformer blocks: x = x + Attn(LN(x)); x = x + MLP(LN(x))
- causal multi-head self-attention via scaled_dot_product_attention
- MLP: Linear(d, 4d) -> GELU -> Linear(4d, d)
- final LayerNorm, output head with weights tied to the token embedding
- normal(0, 0.02) init; residual projections scaled by 1/sqrt(2*n_layers)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import BLOCK_SIZE, Tokenizer

VOCAB_SIZE = len(Tokenizer())


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    Uses F.scaled_dot_product_attention with is_causal=True, which computes
    softmax(Q K^T / sqrt(d_head) + causal_mask) V. Right-padding is safe with
    a causal mask: a real token never attends to a padding token that sits to
    its right. The reference math, for clarity, is:

        att = (Q @ K.transpose(-2, -1)) / math.sqrt(d_head)
        att = att.masked_fill(causal_mask, float("-inf"))
        att = softmax(att, dim=-1)          # pad positions get ~0 weight
        out = att @ V

    When ``attention_mask`` (padding) is provided — eval with left-padded
    batches — we build the (causal | padding) boolean mask ourselves and pass
    it instead of is_causal=True.
    """

    def __init__(self, d_model, n_head, block_size, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout_p = dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        if attention_mask is None:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        else:
            # attention_mask: (B, T) with 1 for real tokens, 0 for padding.
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            key_ok = attention_mask.bool().view(B, 1, 1, T)
            mask = causal.view(1, 1, T, T) & key_ok  # (B, 1, T, T)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(d_model, ff_dim)
        self.proj = nn.Linear(ff_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, d_model, n_head, ff_dim, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, ff_dim, dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, d_model=128, n_layer=4, n_head=4, ff_dim=512,
                 block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model
        self.config = dict(d_model=d_model, n_layer=n_layer, n_head=n_head,
                           ff_dim=ff_dim, block_size=block_size,
                           vocab_size=vocab_size, dropout=dropout)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(d_model, n_head, ff_dim, block_size, dropout)
            for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: the output projection shares the token embedding.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Residual projections scaled by 1/sqrt(2*n_layers) (GPT-2 style).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, attention_mask=None, position_ids=None):
        B, T = idx.shape
        if attention_mask is None:
            # Training path: right-padded batch, causal attention is enough.
            pos = torch.arange(T, device=idx.device).unsqueeze(0)
        else:
            # Eval path: left-padded batch. Position of a real token is its
            # index within the non-padding part, so pad positions get 0.
            if position_ids is None:
                position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_(min=0)
            pos = position_ids
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            # Cross-entropy only on the cron tokens; prompt positions and
            # padding are already masked with -100 by the data pipeline.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    # -- generation ---------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt_ids_list, max_new_tokens=64, eos_id=None):
        """Greedy (argmax) batched decoding.

        ``prompt_ids_list`` is a list of token-id lists, left-padded inside
        the call. Returns a list of id lists (no padding), stopping at <eos>.
        """
        device = next(self.parameters()).device
        lengths = [len(p) for p in prompt_ids_list]
        max_len = max(lengths)
        pad_id = 0
        batch = torch.full((len(prompt_ids_list), max_len), pad_id,
                           dtype=torch.long, device=device)
        mask = torch.zeros((len(prompt_ids_list), max_len),
                           dtype=torch.long, device=device)
        for i, p in enumerate(prompt_ids_list):
            batch[i, max_len - len(p):] = torch.tensor(p, device=device)
            mask[i, max_len - len(p):] = 1

        finished = torch.zeros(len(prompt_ids_list), dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            position_ids = (mask.cumsum(dim=1) - 1).clamp_(min=0)
            logits, _ = self.forward(batch, targets=None,
                                     attention_mask=mask,
                                     position_ids=position_ids)
            next_id = logits[:, -1, :].argmax(dim=-1)
            if eos_id is not None:
                next_id = torch.where(finished, torch.full_like(next_id, pad_id), next_id)
                finished |= (next_id == eos_id)
            batch = torch.cat([batch, next_id.view(-1, 1)], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_id.view(-1, 1))], dim=1)
            if eos_id is not None and bool(finished.all()):
                break

        # strip left padding; stop at first <eos> per row
        results = []
        for i, p in enumerate(prompt_ids_list):
            row = batch[i, max_len:].tolist()
            if eos_id is not None and eos_id in row:
                row = row[:row.index(eos_id) + 1]
            results.append(row)
        return results
