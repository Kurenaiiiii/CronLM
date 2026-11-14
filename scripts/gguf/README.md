# GGUF export

Converts a CronLM checkpoint into a GPT-2-family GGUF that `llama.cpp` can
load greedily. Produces a **verified bit-replica** of our PyTorch model:
`llama-completion -m exports/cronlm-tiny-f32.gguf -p "every weekday at 9am<sep>" --temp 0`
prints exactly what the PyTorch model prints (150/150 val prompts match,
see `verify_gguf.py`).

Requirements: this repo venv plus the `llama.cpp` checkout and its tools
(`llama-completion`, `llama-tokenize`) built with `cmake`.

```bash
# 0. prerequisites (one time)
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build --target llama-completion llama-tokenize -j 12
pip install transformers        # conversion tooling only; not used by the project

# 1. export HF GPT-2 layout from our checkpoint (defaults: runs/tiny/best.pt -> gguf_export/model)
CKPT=runs/tiny/best.pt OUT=gguf_export/model python export_gguf.py

# 2. convert with llama.cpp's converter (bypasses its pre-tokenizer hash check;
#    our 48 tokens are GPT-2 byte-level: chars map to themselves, space -> U+0120)
python scripts/gguf/convert_hf_to_gguf.py

# 3. patch in the two keys llama.cpp requires (empty arrays are illegal in GGUF,
#    so merges carries one dead 0x7f pair that can never encode)
python scripts/gguf/patch_gguf.py cronlm-tiny-f32.gguf exports/cronlm-tiny-f32.gguf

# 4. verify (150/150 match is the acceptance test)
LLAMA_BIN=/tmp/llama.cpp/build/bin/llama-completion python verify_gguf.py
```

Notes:
- `llama-cli` (chat mode) is NOT suitable for this model; use `llama-completion`.
- Use `-c 160` to match `block_size`; otherwise you get a context-overflow warning.
- `--temp 0` is greedy, matching our inference.
- llama.cpp logs print `prompt + completion` on one line; the special `<sep>`
  is stripped from the echo. `[end of text]` marks eos.
- `verify_gguf.py` also reports `llama.cpp == label`, which exposes the model's
  own quality (not the exporter's).
