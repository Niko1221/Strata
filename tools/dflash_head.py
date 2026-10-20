#!/usr/bin/env python3
"""tools/dflash_head.py - the draft GGUF for --dflash: a DFlash2 draft (convert_hf_to_gguf.py --target-model-dir) with
the target's token_embd and output head added, so llama.cpp loads it standalone (no target context).

  python tools/dflash_head.py DRAFT.gguf TARGET-00001-of-N.gguf OUT.gguf [--draft-vocab rt/draft_vocab.bin]

The head is written as Q8_0 (Metal's small-batch kernels; dequantized from the target's own head).  --draft-vocab
(int32 token ids, the MTP's rt/draft_vocab.bin from tools/draft_vocab.py) keeps only those rows plus a d2t map, so a
draft reads a fraction of the head.  The draft itself can be quantized first (llama-quantize DRAFT.gguf Q8_0).
Needs llama.cpp's gguf-py on PYTHONPATH (third_party/llama.cpp/gguf-py) and numpy.
"""
import argparse

import numpy as np
import gguf
from gguf import quants

Q8 = gguf.GGMLQuantizationType.Q8_0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("draft")
    ap.add_argument("target")
    ap.add_argument("out")
    ap.add_argument("--draft-vocab", help="int32 token ids: the head keeps these rows (with a d2t map)")
    a = ap.parse_args()

    r = gguf.GGUFReader(a.draft)
    if str(r.fields["general.architecture"].contents()) != "dflash":
        raise SystemExit(f"{a.draft}: not a DFlash draft GGUF")
    t = gguf.GGUFReader(a.target)
    head = {x.name: x for x in t.tensors if x.name in ("token_embd.weight", "output.weight")}
    if len(head) != 2:
        raise SystemExit(f"{a.target}: no token_embd/output (pass the shard that holds them)")
    have = {x.name for x in r.tensors}
    if "output.weight" in have or "token_embd.weight" in have or "d2t" in have:
        raise SystemExit(f"{a.draft}: already has a head")

    w = gguf.GGUFWriter(a.out, arch="dflash")
    for k, f in r.fields.items():
        if k.startswith("GGUF.") or k == "general.architecture":
            continue
        vt = f.types[0]
        if vt == gguf.GGUFValueType.ARRAY:
            w.add_key_value(k, f.contents(), vt, sub_type=f.types[-1])
        else:
            w.add_key_value(k, f.contents(), vt)
    for x in r.tensors:
        w.add_tensor(x.name, x.data, raw_shape=x.data.shape, raw_dtype=x.tensor_type)
    e = head["token_embd.weight"]
    w.add_tensor(e.name, e.data, raw_shape=e.data.shape, raw_dtype=e.tensor_type)

    o = head["output.weight"]
    q = o.data if o.tensor_type == Q8 else quants.quantize(quants.dequantize(o.data, o.tensor_type).astype(np.float32), Q8)
    n_vocab = q.shape[0]
    if a.draft_vocab:
        ids = np.fromfile(a.draft_vocab, dtype=np.int32)
        if ids.size == 0 or ids.min() < 0 or ids.max() >= n_vocab:
            raise SystemExit(f"{a.draft_vocab}: token ids outside 0..{n_vocab - 1}")
        q = q[ids]
        w.add_tensor("d2t", ids.astype(np.int64), raw_dtype=gguf.GGMLQuantizationType.I64)
    w.add_tensor(o.name, q, raw_shape=q.shape, raw_dtype=Q8)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{a.out}: head {q.shape[0]} of {n_vocab} tokens (Q8_0)")


if __name__ == "__main__":
    main()
