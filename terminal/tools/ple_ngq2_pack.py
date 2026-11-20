"""terminal/tools/ple_ngq2_pack.py - the NGQ2 PLE n-gram table (dnotitia/Qwen3.8-Flash-Next-NGQ2) as a GGUF the engine reads.

    python terminal/tools/ple_ngq2_pack.py --model <NGQ2 dir> --out <ple-ngq2.gguf>

The NGQ2 release stores the table in 128 row shards, `...ngram_embedding.shard_<i>.ngq2_symbols` (U8 [rows, 40]: 80
4-bit trellis symbols a row, the high nibble first) and `...shard_<i>.ngq2_gain` (BF16 [rows]), plus one `ngq2.codebook`
(F32 [512, 2]): 13.44 GB for 320,001,536 rows of 160 values (the BF16 table is 102.4 GB, IQ4_NL 28.8 GB). Shard i holds
the rows after shard i - 1, so row r of the output is the checkpoint's row r.

The output is a two-tensor GGUF marked `strata.ple.format` = "ngq2": `per_layer_token_embd.weight`, ne = [42,
320001536], type I8 holding each row's 40 symbol bytes followed by its 2 gain bytes (little-endian BF16), at offset 0,
then `strata.ple.codebook`, ne = [2, 512] F32 (the codebook row by row). Bytes are copied as stored - nothing is
decoded or rounded.
"""
import argparse
import json
import pathlib
import re
import struct
import sys

import numpy as np

GGUF_TYPE_STRING = 8
GGML_TYPE_F32, GGML_TYPE_I8 = 0, 24
ALIGN = 32
CHUNK_ROWS = 1 << 20
SYM_BYTES, ROW_BYTES = 40, 42


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key, val):
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + gguf_string(val)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="the NGQ2 checkpoint directory (its *.safetensors)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, out = pathlib.Path(a.model), pathlib.Path(a.out)

    tensors = {}   # name -> (file, absolute offset, header entry)
    for path in sorted(model.glob("*.safetensors")):
        hdr, base = safetensors_header(path)
        for name, t in hdr.items():
            if name != "__metadata__":
                tensors[name] = (path, base + t["data_offsets"][0], t)
    shard = re.compile(r"\.ngram_embedding\.shard_(\d+)\.ngq2_(symbols|gain)$")
    parts = {}   # shard index -> {"symbols": name, "gain": name}
    for n in tensors:
        m = shard.search(n)
        if m:
            parts.setdefault(int(m.group(1)), {})[m.group(2)] = n
    if not parts or "ngq2.codebook" not in tensors:
        sys.exit("no ngram_embedding.shard_*.ngq2_symbols / ngq2_gain / ngq2.codebook in " + str(model))
    if sorted(parts) != list(range(len(parts))) or any(len(p) != 2 for p in parts.values()):
        sys.exit("the NGQ2 shards are not 0..%d, each with symbols and gain" % (len(parts) - 1))
    shards = [parts[i] for i in range(len(parts))]

    path, off, t = tensors["ngq2.codebook"]
    if t["dtype"] != "F32" or t["shape"] != [512, 2]:
        sys.exit("unexpected ngq2.codebook: %s %s" % (t["dtype"], t["shape"]))
    with open(path, "rb") as f:
        f.seek(off)
        codebook = np.frombuffer(f.read(512 * 2 * 4), dtype="<f4")

    rows = 0
    for p in shards:
        t, g = tensors[p["symbols"]][2], tensors[p["gain"]][2]
        if t["dtype"] != "U8" or len(t["shape"]) != 2 or t["shape"][1] != SYM_BYTES:
            sys.exit("%s is %s %s, not U8 [rows, 40]" % (p["symbols"], t["dtype"], t["shape"]))
        if g["dtype"] != "BF16" or g["shape"] != [t["shape"][0]]:
            sys.exit("%s is %s %s, not BF16 [%d]" % (p["gain"], g["dtype"], g["shape"], t["shape"][0]))
        rows += t["shape"][0]
    print("%d shards, %d rows, %.2f GB" % (len(shards), rows, rows * ROW_BYTES / 1e9), flush=True)

    kvs = [kv_string("general.architecture", "strata-ple"),
           kv_string("general.name", "PLE n-gram table, NGQ2 (2-bit trellis)"),
           kv_string("strata.ple.format", "ngq2"),
           kv_string("strata.ple.source", model.name)]
    table_bytes = rows * ROW_BYTES
    cb_offset = table_bytes + (-table_bytes) % ALIGN
    head = b"GGUF" + struct.pack("<IQQ", 3, 2, len(kvs)) + b"".join(kvs)
    head += gguf_string("per_layer_token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", ROW_BYTES, rows)
    head += struct.pack("<I", GGML_TYPE_I8) + struct.pack("<Q", 0)
    head += gguf_string("strata.ple.codebook") + struct.pack("<I", 2) + struct.pack("<QQ", 2, 512)
    head += struct.pack("<I", GGML_TYPE_F32) + struct.pack("<Q", cb_offset)
    head += b"\0" * ((-len(head)) % ALIGN)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    done = 0
    buf = np.empty((CHUNK_ROWS, ROW_BYTES), dtype=np.uint8)
    with open(tmp, "wb") as w:
        w.write(head)
        for i, p in enumerate(shards):
            spath, soff, t = tensors[p["symbols"]]
            gpath, goff, _ = tensors[p["gain"]]
            with open(spath, "rb") as f, open(gpath, "rb") as g:
                f.seek(soff)
                g.seek(goff)
                left = t["shape"][0]
                while left:
                    k = min(CHUNK_ROWS, left)
                    s = f.read(k * SYM_BYTES)
                    gb = g.read(k * 2)
                    if len(s) != k * SYM_BYTES or len(gb) != k * 2:
                        sys.exit("short read in shard %d" % i)
                    buf[:k, :SYM_BYTES] = np.frombuffer(s, dtype=np.uint8).reshape(k, SYM_BYTES)
                    buf[:k, SYM_BYTES:] = np.frombuffer(gb, dtype=np.uint8).reshape(k, 2)
                    w.write(buf[:k].tobytes())
                    left -= k
                    done += k
            print("  shard %d done, %d/%d rows" % (i, done, rows), flush=True)
        w.write(b"\0" * (cb_offset - table_bytes))
        w.write(codebook.astype("<f4").tobytes())
    if tmp.stat().st_size != len(head) + cb_offset + 4096:
        sys.exit("size check failed: %d != %d" % (tmp.stat().st_size, len(head) + cb_offset + 4096))
    tmp.replace(out)
    print("wrote %s (%d rows x %d B, header %d B)" % (out, rows, ROW_BYTES, len(head)))


if __name__ == "__main__":
    main()
