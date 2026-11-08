#!/usr/bin/env python3
"""Extract the exact per-tensor quant-type map of the reference UD-Q2_K_XL GGUF, as the
quantize target for the UNCENSORED rebuild (llama-quantize --override-tensor per name).

    python tools/extract_ref_typemap.py > /data/models/GLM-5.3-Flash-UNCENSORED-UD-Q2/ref-typemap.json
"""
import json
import re
import sys

sys.path.insert(0, "/home/simon/work/llama.cpp/gguf-py")
import gguf  # noqa: E402

REF = "/nvme/models/GLM-5.3-Flash-UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-0000{shard}-of-00004.gguf"

typemap = {}
for shard in (1, 2, 3, 4):
    r = gguf.GGUFReader(REF.format(shard=shard))
    for t in r.tensors:
        typemap[t.name] = t.tensor_type.name

# a per-name override list for llama-quantize: --override-tensor "^<escaped name>$"=<type>
overrides = []
for name in sorted(typemap):
    pat = "^" + re.escape(name) + "$"
    overrides += ["--override-tensor", f"{pat}={typemap[name]}"]

out = {
    "source": REF.format(shard=1),
    "tensor_count": len(typemap),
    "types": typemap,
}
json.dump(out, sys.stdout, indent=1)
print()
print(" ".join(overrides), file=sys.stderr)
