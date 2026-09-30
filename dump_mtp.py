#!/usr/bin/env python3
"""Mostra cosa c'e' di MTP in uno o piu' GGUF: chiavi nextn/rope/ssm e i tensori del blocco MTP.

    pip install gguf            # se manca
    python dump_mtp.py MAIN.gguf MTP.gguf

Per il file MTP (piccolo) stampa tutti i tensori; per il GGUF principale solo blk.<ultimo>.* e quelli con nextn/mtp.
"""
import sys
from gguf import GGUFReader

KEYS = ("block_count", "nextn", "mtp", "rope", "ssm.", "attention.", "full_attention", "embedding_length",
        "feed_forward_length", "context_length", "general.architecture", "general.name")


def val(f):
    try:
        d = f.contents()
        s = str(d)
        return s if len(s) < 120 else s[:117] + "..."
    except Exception:
        return "?"


for path in sys.argv[1:]:
    r = GGUFReader(path)
    print("=" * 100)
    print(path)
    print("-" * 40, "metadata")
    for name, f in r.fields.items():
        if any(k in name for k in KEYS):
            print("  %-55s %s" % (name, val(f)))
    tensors = list(r.tensors)
    small = len(tensors) < 80
    last = -1
    for t in tensors:
        if t.name.startswith("blk."):
            try:
                last = max(last, int(t.name.split(".")[1]))
            except ValueError:
                pass
    print("-" * 40, "tensors: %d in total, highest blk index %d" % (len(tensors), last))
    for t in tensors:
        n = t.name
        if small or n.startswith("blk.%d." % last) or "nextn" in n or "mtp" in n or n in ("output.weight", "output_norm.weight", "token_embd.weight"):
            print("  %-48s %-10s %s" % (n, t.tensor_type.name, list(t.shape)))
