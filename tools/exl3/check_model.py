"""Reconstruct EXL3 experts from the multi-shard model with the numpy spec; same checksum as the C++
`exl3_model_check` tool, so the loader can be compared across shards.

    python3 tools/exl3/check_model.py <model_dir> <base_name>
    python3 tools/exl3/check_model.py <model_dir> --layer <L> --role <down_proj|gate_proj|up_proj> --count <N>
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import exl3                                                             # noqa: E402
from exl3.codebook import CB_3INST, CB_MCG, CB_MUL1                    # noqa: E402


def fnv1a(h: int, b: bytes) -> int:
    for x in b:
        h ^= x
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def main() -> int:
    from safetensors import safe_open
    d = pathlib.Path(sys.argv[1])
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    print("model: %d tensors" % len(wm))
    opened: dict[str, object] = {}

    def tensor(name):
        shard = wm[name]
        if shard not in opened:
            opened[shard] = safe_open(str(d / shard), framework="numpy")
        return opened[shard].get_tensor(name)

    def recon(base):
        tr = tensor(base + ".trellis").astype(np.int16)
        suh = tensor(base + ".suh").astype(np.float16)
        svh = tensor(base + ".svh").astype(np.float16)
        cb = CB_MUL1 if (base + ".mul1") in wm else (CB_MCG if (base + ".mcg") in wm else CB_3INST)
        w = exl3.reconstruct_weight(tr, suh, svh, cb)
        return np.ascontiguousarray(w, dtype="<f2").tobytes()

    h = 1469598103934665603
    if sys.argv[2] == "--layer":
        layer = int(sys.argv[3]); role = sys.argv[5]; count = int(sys.argv[7])
        for e in range(count):
            base = "model.language_model.layers.%d.mlp.experts.%d.%s" % (layer, e, role)
            h = fnv1a(h, recon(base))
        print("layer %d %s x%d combined fnv1a=%016x" % (layer, role, count, h))
    else:
        base = sys.argv[2]
        h = fnv1a(h, recon(base))
        print("fnv1a=%016x" % h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
