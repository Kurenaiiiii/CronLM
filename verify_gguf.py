"""Verify the GGUF export: greedy-decode N val prompts in llama.cpp and compare
with our PyTorch model's greedy output."""
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, "src")
from cronlm.model import GPT
from cronlm.tokenizer import Tokenizer

LLAMA = os.environ.get("LLAMA_BIN", "llama-completion")
GGUF = os.environ.get("GGUF", "exports/cronlm-small-f32.gguf")
CKPT = os.environ.get("CKPT", "runs/small/best.pt")
N = 150

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = GPT(**{k: v for k, v in ckpt["config"].items() if k != "dropout"}, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()
tok = Tokenizer()

rows = [json.loads(l) for l in open("data/val.jsonl")][:N]
prompts = [r["phrase"] for r in rows]
refs = [r["cron"] for r in rows]
with torch.no_grad():
    ours = [tok.decode(g) for g in
            model.generate([tok.encode_prompt(p) for p in prompts],
                           max_new_tokens=40, eos_id=tok.eos_id)]

def llama_pred(prompt):
    out = subprocess.run(
        [LLAMA, "-m", GGUF, "-p", prompt + "<sep>", "-n", "40", "-c", "160",
         "--temp", "0"],
        capture_output=True, text=True, check=True).stdout
    i = out.find(prompt)
    if i == -1:
        return None
    after = out[i + len(prompt):]
    end = after.find("[end of text]")
    return after[:end].strip() if end != -1 else None

agree, agree_exact, agree_ref = 0, 0, 0
mismatches = []
for i, (p, ref, o) in enumerate(zip(prompts, refs, ours)):
    lp = llama_pred(p)
    ok_eq = (lp == o)
    ok_ref = (lp == ref)
    agree += ok_eq
    agree_exact += (lp == ref)
    agree_ref += ok_ref
    if not ok_eq:
        mismatches.append((p, ref, o, lp))

print(f"prompts checked      : {N}")
print(f"llama.cpp == pytorch : {agree}/{N}")
print(f"llama.cpp == label   : {agree_exact}/{N}")
print(f"pytorch  == label    : {sum(1 for o, r in zip(ours, refs) if o == r)}/{N}")
for p, ref, o, lp in mismatches[:10]:
    print(f"  phrase={p!r}\n    label={ref!r}\n    ours={o!r}\n    gguf={lp!r}")
