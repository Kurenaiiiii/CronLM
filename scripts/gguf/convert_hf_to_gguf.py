"""Run llama.cpp's HF->GGUF converter with our custom vocab's pre-tokenizer
tagged as 'gpt-2' (the GPT-2 byte-level mapping matches our 48 tokens:
all chars map to themselves, space -> U+0120). llama.cpp is used unmodified.

Env overrides:
  LLAMA_CPP  path to the llama.cpp checkout   (default /tmp/llama.cpp)
  MODEL      HF checkpoint directory to convert (default gguf_export/model)
  OUTFILE    output GGUF path                 (default cronlm-tiny-f32.gguf)
"""
import os
import runpy
import sys

LLAMA_CPP = os.environ.get("LLAMA_CPP", "/tmp/llama.cpp")
MODEL = os.environ.get("MODEL", "gguf_export/model")
OUTFILE = os.environ.get("OUTFILE", "cronlm-tiny-f32.gguf")

sys.path.insert(0, LLAMA_CPP)
from conversion import base as _base

def _get_vocab_base_pre(self, tokenizer):
    return "gpt-2"

_base.TextModel.get_vocab_base_pre = _get_vocab_base_pre

sys.argv = [
    "convert_hf_to_gguf.py",
    MODEL,
    "--outfile", OUTFILE,
    "--outtype", "f32",
]
runpy.run_path(os.path.join(LLAMA_CPP, "convert_hf_to_gguf.py"),
               run_name="__main__")
