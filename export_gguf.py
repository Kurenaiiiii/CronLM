"""Export CronLM tiny GPT -> HF GPT-2 checkpoint for GGUF conversion.

HF GPT-2 stores c_attn/c_fc/c_proj weights transposed relative to Linear, so
we transpose here. Everything else maps 1:1. lm_head is tied to wte.
"""
import json
import os
import sys

import torch

sys.path.insert(0, "src")
from cronlm.model import GPT
from cronlm.tokenizer import Tokenizer

CKPT = os.environ.get("CKPT", "runs/tiny/best.pt")
OUT = os.environ.get("OUT", "gguf_export/model")

os.makedirs(OUT, exist_ok=True)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ckpt["config"]
vocab_size = cfg["vocab_size"]
d = cfg["d_model"]

ours = GPT(d_model=cfg["d_model"], n_layer=cfg["n_layer"], n_head=cfg["n_head"],
           ff_dim=cfg["ff_dim"], block_size=cfg["block_size"],
           vocab_size=vocab_size)
ours.load_state_dict(ckpt["model"])
ours.eval()

from transformers import GPT2Config, GPT2LMHeadModel

hf_cfg = GPT2Config(
    vocab_size=vocab_size,
    n_positions=cfg["block_size"],
    n_ctx=cfg["block_size"],
    n_embd=d,
    n_layer=cfg["n_layer"],
    n_head=cfg["n_head"],
    n_inner=cfg["ff_dim"],
    activation_function="gelu",     # exact erf gelu, matches training (F.gelu)
    layer_norm_epsilon=1e-5,
    resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0,
    bos_token_id=1, eos_token_id=3, pad_token_id=0,
    tie_word_embeddings=True,
    model_type="gpt2",
)
hf = GPT2LMHeadModel(hf_cfg)

sd_ours = ours.state_dict()
sd_hf = {}

def put(hf_name, tensor):
    sd_hf[hf_name] = tensor

put("transformer.wte.weight", sd_ours["tok_emb.weight"])
put("transformer.wpe.weight", sd_ours["pos_emb.weight"])
for i in range(cfg["n_layer"]):
    p = f"blocks.{i}"
    h = f"transformer.h.{i}"
    for ln_hf, ln_ours in (("ln_1", "ln1"), ("ln_2", "ln2")):
        put(f"{h}.{ln_hf}.weight", sd_ours[f"{p}.{ln_ours}.weight"])
        put(f"{h}.{ln_hf}.bias", sd_ours[f"{p}.{ln_ours}.bias"])
    put(f"{h}.attn.c_attn.weight", sd_ours[f"{p}.attn.qkv.weight"].t())
    put(f"{h}.attn.c_attn.bias", sd_ours[f"{p}.attn.qkv.bias"])
    put(f"{h}.attn.c_proj.weight", sd_ours[f"{p}.attn.proj.weight"].t())
    put(f"{h}.attn.c_proj.bias", sd_ours[f"{p}.attn.proj.bias"])
    put(f"{h}.mlp.c_fc.weight", sd_ours[f"{p}.mlp.fc.weight"].t())
    put(f"{h}.mlp.c_fc.bias", sd_ours[f"{p}.mlp.fc.bias"])
    put(f"{h}.mlp.c_proj.weight", sd_ours[f"{p}.mlp.proj.weight"].t())
    put(f"{h}.mlp.c_proj.bias", sd_ours[f"{p}.mlp.proj.bias"])
put("transformer.ln_f.weight", sd_ours["ln_f.weight"])
put("transformer.ln_f.bias", sd_ours["ln_f.bias"])
missing, unexpected = hf.load_state_dict(sd_hf, strict=False)
print("missing:", missing)
print("unexpected:", unexpected)
# lm_head.weight is tied to wte and is expected to be absent here
assert missing == ["lm_head.weight"] and not unexpected, "state dict mismatch"

# tie lm_head (shared with wte) before saving
hf.tie_weights()
hf.save_pretrained(OUT, safe_serialization=True)

# --- tokenizer files (GPT-2 layout): space must be the byte-level 'Ġ' (U+0120)
tok = Tokenizer()
ggspace = "Ġ"  # U+0120, what GPT-2's byte-level mapping uses for byte 0x20
ids = {}
for idx, t in enumerate(tok.tokens):
    if t == " ":
        ids[ggspace] = idx
    elif t.startswith("<") and t.endswith(">"):
        ids[t] = idx          # keep special token strings as-is
    else:
        ids[t] = idx          # all other allowed chars map to themselves
vocab_json = {t: i for t, i in ids.items()}
with open(os.path.join(OUT, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(vocab_json, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT, "merges.txt"), "w", encoding="utf-8") as f:
    f.write("#version: 0.2\n")
special = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
           "unk_token": "<pad>", "sep_token": "<sep>"}
with open(os.path.join(OUT, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    json.dump({"tokenizer_class": "GPT2Tokenizer", **special,
               "model_max_length": cfg["block_size"]},
              f, ensure_ascii=False, indent=2)
print(f"saved HF checkpoint to {OUT}")
