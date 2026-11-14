"""Patch a GGUF: add the two keys llama.cpp requires to load and tokenize:
  - tokenizer.ggml.merges = ["\\x7f \\x7f"]  (GGUF forbids empty arrays; this
    pair can never encode, so the merge is dead by construction)
  - tokenizer.ggml.add_bos_token = true     (llama.cpp prepends <bos> itself)
All other tensors and metadata are copied verbatim.

Env override: LLAMA_CPP  (path to the llama.cpp checkout, default /tmp/llama.cpp)

Usage: python scripts/gguf/patch_gguf.py input.gguf output.gguf
"""
import os
import sys

LLAMA_CPP = os.environ.get("LLAMA_CPP", "/tmp/llama.cpp")
sys.path.insert(0, os.path.join(LLAMA_CPP, "gguf-py"))
import gguf

inp, out = sys.argv[1], sys.argv[2]

reader = gguf.GGUFReader(inp, "r")
arch = reader.get_field("general.architecture").contents()

writer = gguf.GGUFWriter(out, arch)

for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    writer.add_key_value(field.name, val, val_type, sub_type=sub_type)

writer.add_key_value("tokenizer.ggml.merges", ["\x7f \x7f"],
                     gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
writer.add_key_value("tokenizer.ggml.add_bos_token", True,
                     gguf.GGUFValueType.BOOL)

for tensor in reader.tensors:
    writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                           tensor.data.nbytes, tensor.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for tensor in reader.tensors:
    writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
writer.close()
print(f"wrote {out}")
