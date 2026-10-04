#!/usr/bin/env python3
"""Header-only GGUF reader (no dependencies, reads only the header, never the weights).

Usage:
  python3 tools/dump_gguf_shapes.py model-00001-of-0000N.gguf [more shards...] > shapes.txt

Prints: scalar metadata, tensor patterns (layer index collapsed to N), shapes, ggml types,
and byte sizes (total expert bytes, bytes per expert, non-expert bytes).
"""
import re
import struct
import sys
from collections import OrderedDict

# ggml type id -> (name, block_elems, block_bytes)
GT = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4),
    27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    39: ("MXFP4", 32, 17),
}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class R:
    def __init__(self, f):
        self.f = f

    def u(self, fmt):
        n = struct.calcsize(fmt)
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("truncated GGUF header")
        return struct.unpack(fmt, b)[0]

    def s(self):
        n = self.u("<Q")
        return self.f.read(n).decode("utf-8", "replace")

    def val(self, t):
        if t in SCALAR:
            return self.u(SCALAR[t])
        if t == 8:
            return self.s()
        if t == 9:
            et = self.u("<I")
            n = self.u("<Q")
            if et in SCALAR:
                sz = struct.calcsize(SCALAR[et])
                data = self.f.read(sz * n)
                head = [struct.unpack(SCALAR[et], data[i * sz:(i + 1) * sz])[0] for i in range(min(n, 8))]
                return ("array", n, head)
            if et == 8:
                head = []
                for i in range(n):
                    x = self.s()
                    if i < 4:
                        head.append(x)
                return ("array", n, head)
            raise ValueError("unsupported array elem type %d" % et)
        raise ValueError("unknown kv type %d" % t)


def read_file(path):
    kv, tensors = OrderedDict(), []
    with open(path, "rb") as f:
        r = R(f)
        if f.read(4) != b"GGUF":
            raise ValueError(path + ": not a GGUF file")
        ver = r.u("<I")
        nt = r.u("<Q")
        nkv = r.u("<Q")
        for _ in range(nkv):
            k = r.s()
            t = r.u("<I")
            kv[k] = r.val(t)
        for _ in range(nt):
            name = r.s()
            nd = r.u("<I")
            dims = [r.u("<Q") for _ in range(nd)]
            ty = r.u("<I")
            off = r.u("<Q")
            tensors.append((name, dims, ty, off))
    return ver, kv, tensors


def nbytes(dims, ty):
    if ty not in GT:
        return None
    _, be, bb = GT[ty]
    n = 1
    for d in dims:
        n *= d
    return (n // be) * bb


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    all_kv, all_t = OrderedDict(), []
    for p in sys.argv[1:]:
        ver, kv, ts = read_file(p)
        print("# %s: GGUF v%d, %d kv, %d tensors" % (p, ver, len(kv), len(ts)))
        for k, v in kv.items():
            all_kv.setdefault(k, v)
        all_t += ts

    print("\n## metadata")
    for k, v in all_kv.items():
        if isinstance(v, tuple):
            print("%s = array(len=%d) head=%s" % (k, v[1], v[2]))
        else:
            sv = str(v)
            print("%s = %s" % (k, sv if len(sv) < 160 else sv[:160] + "..."))

    print("\n## tensor patterns (layer index -> N)")
    pats = OrderedDict()
    for name, dims, ty, _ in all_t:
        key = re.sub(r"\.\d+\.", ".N.", name)
        p = pats.setdefault(key, {"n": 0, "shapes": set(), "types": set(), "ex": name})
        p["n"] += 1
        p["shapes"].add(tuple(dims))
        p["types"].add(GT.get(ty, ("type#%d" % ty,))[0])
    for key, p in pats.items():
        print("%-48s x%-3d %s %s" % (key, p["n"], sorted(p["types"]), sorted(p["shapes"])[:3]))

    exp_total, other_total, unknown = 0, 0, 0
    per_expert = {}
    for name, dims, ty, _ in all_t:
        b = nbytes(dims, ty)
        if b is None:
            unknown += 1
            continue
        if "_exps" in name and len(dims) == 3:
            exp_total += b
            m = re.search(r"\.(\d+)\.", name)
            if m:
                per_expert[m.group(1)] = per_expert.get(m.group(1), 0) + b // dims[-1]
        else:
            other_total += b
    print("\n## sizes")
    print("expert tensors (3D *_exps): %.2f GiB" % (exp_total / 2**30))
    print("everything else           : %.2f GiB" % (other_total / 2**30))
    if unknown:
        print("tensors with unknown ggml type (size not counted): %d" % unknown)
    if per_expert:
        vals = sorted(set(per_expert.values()))
        print("bytes per expert (distinct values across layers): %s" % [("%d (%.2f MiB)" % (v, v / 2**20)) for v in vals])
        print("layers with expert tensors: %d" % len(per_expert))


if __name__ == "__main__":
    main()
