"""tools/remote-vision/make_vocab_stub.py - build a metadata+vocabulary-only GGUF stub of the text model.

WHY THIS EXISTS
---------------
`strata-vision` needs the TEXT model for exactly two things: its vocabulary (mtmd walks all
248,320 tokens looking for special markers such as `<__media__>`, and it tokenizes prompt text)
and a few metadata keys.  It opens it with `mp.vocab_only = true`
(tools/vision/strata_vision.cpp), and llama.cpp returns right after the vocabulary:

    if (params.vocab_only) { ...; return {0, model_ptr.release()}; }   # src/llama.cpp

so no tensor DATA is ever read.  On the server machine the real 51 GB shard is used and that costs
nothing; on a REMOTE vision host it would mean copying 51 GB to read 11 MB.

WHY THE TENSOR TABLE IS REWRITTEN RATHER THAN JUST TRUNCATED
-----------------------------------------------------------
ggml's GGUF reader seeks to the data section before the vocab-only early-out:

    if (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment)))

`gr.seek` is bounded by the reader's size, so a file cut after the real tensor table is refused
("failed to seek to beginning of data section").  The seek target is derived from the DECLARED
tensor shapes, so this script rewrites every tensor entry to a 1x1 F32 tensor: the computed data
section is then tiny and lands inside the stub, the seek succeeds, and the empty tensor table is
never read because the vocab-only path returns first.

    python make_vocab_stub.py <full model shard1.gguf> <out stub.gguf>

Verify on the machine that has the mmproj: start strata-vision with --model <stub> and confirm it
prints "READY <n_embd>".
"""
import struct
import sys
from pathlib import Path

FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY = 8, 9
GGML_TYPE_F32, GGML_TYPE_F32_SIZE = 0, 4
ALIGNMENT = 32          # ctx->alignment used by the reader


def read_value(f, t):
    """Consume one GGUF value (used only to locate the tensor table)."""
    if t == STRING:
        n = struct.unpack("<Q", f.read(8))[0]
        f.read(n)
    elif t == ARRAY:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            read_value(f, et)
    elif t in FIXED:
        f.read(FIXED[t])
    else:
        raise ValueError(f"unknown GGUF value type {t}")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if not src.is_file():
        print(f"not a file: {src}")
        return 2

    with src.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"{src}: not a GGUF file (magic {magic!r})")
            return 2
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        print(f"{src.name}: GGUF v{version}, {n_tensors} tensors, {n_kv} metadata keys")

        vocab_bytes = 0
        patch = []                                    # (offset, replacement bytes) inside `head`
        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen)
            t = struct.unpack("<I", f.read(4))[0]
            before = f.tell()
            read_value(f, t)
            if b"token" in key:
                vocab_bytes += f.tell() - before
            # a split model makes the loader look for its siblings ("-00002-of-00002"), which a
            # remote vision host does not have and does not need: it reads the vocabulary only.
            if key == b"split.count" and t == 2:
                patch.append((before, struct.pack("<H", 1)))
        meta_end = f.tell()

        # every tensor entry: name, n_dims, dims, type, offset
        names = []
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", f.read(8))[0]
            names.append(f.read(nlen))
            nd = struct.unpack("<I", f.read(4))[0]
            f.read(8 * nd)
            f.read(4)
            f.read(8)
        table_end = f.tell()

        f.seek(0)
        head = bytearray(f.read(meta_end))            # magic..end of metadata+vocab, verbatim
        for at, repl in patch:
            head[at:at + len(repl)] = repl
        head = bytes(head)

    # Rewrite the tensor table: every tensor becomes 1x1 F32.  ggml requires the offsets to be
    # strictly consecutive, each padded to ctx->alignment:
    #     if (ti.offset != ctx->size) -> "has offset N, expected M"
    #     ctx->size += GGML_PAD(ggml_nbytes(&ti.t), ctx->alignment)
    table = bytearray()
    offset = 0
    for name in names:
        table += struct.pack("<Q", len(name)) + name
        table += struct.pack("<I", 1) + struct.pack("<Q", 1)      # n_dims=1, ne[0]=1
        table += struct.pack("<I", GGML_TYPE_F32)
        table += struct.pack("<Q", offset)                        # current, already aligned
        offset += (GGML_TYPE_F32_SIZE + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    # the reader pads the table to ctx->alignment before the data section
    pad = (-(meta_end + len(table))) % ALIGNMENT
    blob = bytes(head) + bytes(table) + b"\0" * (pad + offset)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(blob)

    kib = lambda n: f"{n / 1024:.0f} KiB" if n < 1 << 20 else f"{n / 1048576:.1f} MiB"
    print(f"  vocabulary             : {kib(vocab_bytes)}")
    print(f"  metadata+tensor table  : {kib(meta_end + len(table))}")
    print(f"  data section (1x1 each): {offset} bytes + {pad} pad")
    print(f"  dropped tensor data    : {(src.stat().st_size - table_end) / (1 << 30):.2f} GiB")
    print(f"wrote {dst}  ({dst.stat().st_size / 1048576:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
