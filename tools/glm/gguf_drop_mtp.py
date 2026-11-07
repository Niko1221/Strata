#!/usr/bin/env python3
"""In-place GGUF surgery: drop tensor info records by regex; patch the header tensor count.

Why it exists: the GLM-5.3-UNCENSORED converter (llama.cpp mainline, ModelOpt NVFP4) writes
the MTP draft layer (blk.45) including three Strata-convention sidecar tensors
(blk.45.ffn_{gate,up,down}_exps.scale, F32 per-expert scales) that mainline's glm5-next
loader does not map - they fall into TENSOR_SKIP, so n_created (1538) < n_tensors (1541)
and every load fails the count check; llama-imatrix's --nextn collector also core-dumps on
this arch.  The reference UD-Q2_K_XL pack carries no draft: Strata's engine speculates with
n-grams.  Dropping exactly the unmapped records makes the file load like the reference.

The data section never moves: dropped info records are compacted and the region is padded
back to its original size, so all kept tensors' stored offsets stay valid.  A head backup
is written beside the file before anything is written.

    python tools/glm/gguf_drop_mtp.py <gguf> --drop-regex '^blk\\.45\\..*\\.scale$'
"""
import argparse
import re
import shutil
import struct
import sys
import time

sys.path.insert(0, "/home/simon/work/llama.cpp/gguf-py")
import gguf  # noqa: E402
from gguf import gguf_reader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--drop-regex", required=True)
    a = ap.parse_args()
    t0 = time.time()
    drop_re = re.compile(a.drop_regex)

    r = gguf_reader.GGUFReader(a.gguf)                 # fields parse fine; tensors may not
    fields = list(r.fields.values())                   # file order, each with .offset

    with open(a.gguf, "rb") as fh:
        head = fh.read(24)
        _, _, n_tensors, n_kv = struct.unpack("<4sIQQ", head)

    # walk the tensor info records with the reader's own parser
    kv_end = max(f.offset + sum(p.nbytes for p in f.parts) for f in fields)
    info_recs = []
    pos = kv_end
    for _ in range(n_tensors):
        field = r._get_tensor_info_field(pos)
        size = sum(p.nbytes for p in field.parts)
        info_recs.append((field.name, pos, size))
        pos += size
    info_end = pos

    drop = [(p0, s) for name, p0, s in info_recs if drop_re.match(name)]
    if not drop:
        print("nothing to drop - file already clean")
        return 0
    info_shift = sum(s for _, s in drop)
    print(f"dropping {len(drop)} info records ({info_shift} bytes): "
          f"{[p0 for p0, _ in drop][:6]}{'...' if len(drop) > 6 else ''}")

    # head backup only (the data section is never touched; copying it would fill the disk)
    with open(a.gguf, "rb") as src, open(a.gguf + ".headbak", "wb") as dst:
        left = info_end + 64
        while left:
            chunk = src.read(min(64 * 1024 * 1024, left))
            dst.write(chunk)
            left -= len(chunk)
    print(f"head backup: {a.gguf}.headbak ({info_end + 64} bytes)", flush=True)

    with open(a.gguf, "r+b") as fh:
        fh.seek(0)
        head = bytearray(fh.read(info_end))

        # compact the info region in place: kept records slide left over the dropped ones
        wpos = kv_end
        for name, p0, s in info_recs:
            if drop_re.match(name):
                continue
            rec = bytes(head[p0:p0 + s])
            head[wpos:wpos + s] = rec
            wpos += s
        head[wpos:info_end] = bytes(info_end - wpos)   # zero-pad to the original span
        assert info_end - wpos == info_shift, (info_end - wpos, info_shift)

        fh.seek(0)
        fh.write(bytes(head[:info_end]))
        fh.seek(8)
        fh.write(struct.pack("<Q", n_tensors - len(drop)))

    print(f"done in {time.time() - t0:.1f} s: {n_tensors} -> {n_tensors - len(drop)} tensors "
          f"(n_kv {n_kv} untouched, data section untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
