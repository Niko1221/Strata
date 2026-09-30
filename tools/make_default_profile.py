"""tools/make_default_profile.py - create a default expert-profile.bin with all (layer, expert) pairs.

Usage:
    python tools/make_default_profile.py [--out data/expert-profile.bin] [--n-layer 48] [--n-expert 512]

This writes a profile that ranks every pair in layer-major order, so the engine can fill its
VRAM cache with all experts from the coldest (layer 0) to the hottest (layer N-1).
"""
import argparse
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAGIC, VERSION = b"STRP", 1


def write_default_profile(path, n_layer=48, n_expert=512):
    total = n_layer * n_expert
    ranked = [(layer, expert) for layer in range(n_layer) for expert in range(n_expert)]

    table = [[-1] * n_expert for _ in range(n_layer)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot

    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, n_layer, n_expert, total, total))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for layer in range(n_layer):
            f.write(struct.pack("<%di" % n_expert, *table[layer]))

    print(f"wrote {path}: {total} pairs ({n_layer} layers x {n_expert} experts)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--n-layer", type=int, default=48)
    ap.add_argument("--n-expert", type=int, default=512)
    a = ap.parse_args()
    write_default_profile(a.out, a.n_layer, a.n_expert)


if __name__ == "__main__":
    main()
