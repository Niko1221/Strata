"""tools/ple_table_pack.py - the PLE n-gram table from the checkpoint, as a GGUF the engine reads.

    python tools/ple_table_pack.py --model <checkpoint dir> --out <ple-table.gguf>

Qwen3.8-Flash-Next's checkpoint stores the table as 128 shards `...ngram_embedding.shard_k.weight`, and two
releases store those shards in different precisions. The tool reads whichever the checkpoint has and copies the
bytes through - nothing is decoded, rounded or re-quantized - so the table the engine gathers rows from is the
checkpoint's own, row for row.

    F8_E4M3 (51.2 GB)  one byte a value plus one BF16 `weight_scale`, as Qwen publishes it. Written as type I8
                       (GGUF has no FP8 type) with `strata.ple.format` = "f8_e4m3" and `strata.ple.scale`.
    BF16   (102.4 GB)  two bytes a value and no scale - the full-precision release. Written as type BF16, which
                       is self-describing, so it carries no extra metadata.

The engine used to take this table from ISTA-DASLab's GGUF as IQ4_NL (90 B/row), which is ~8% off per row; either
form here replaces that with the checkpoint's values.
"""
import argparse
import json
import pathlib
import struct
import sys

GGUF_TYPE_STRING, GGUF_TYPE_F32 = 8, 6
GGML_TYPE_I8 = 24
GGML_TYPE_BF16 = 30
ALIGN = 32
CHUNK = 64 << 20

# Source dtype -> (GGUF tensor type, bytes per value, whether the checkpoint carries a `weight_scale`)
SOURCES = {
    "F8_E4M3": (GGML_TYPE_I8, 1, True),
    "BF16": (GGML_TYPE_BF16, 2, False),
}


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key, val):
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + gguf_string(val)


def kv_f32(key, val):
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_F32) + struct.pack("<f", val)


def read_weight_scale(model, wmap, name):
    """The one BF16 scalar an FP8 checkpoint scales its whole table by, read as the float it encodes."""
    hdr, base = safetensors_header(model / wmap[name])
    t = hdr[name]
    if t["dtype"] != "BF16" or t["shape"] not in ([1], []):
        sys.exit("unexpected weight_scale: %s %s" % (t["dtype"], t["shape"]))
    with open(model / wmap[name], "rb") as f:
        f.seek(base + t["data_offsets"][0])
        (bits,) = struct.unpack("<H", f.read(2))
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="the checkpoint directory (model.safetensors.index.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, out = pathlib.Path(a.model), pathlib.Path(a.out)
    wmap = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]

    shards, scale_name = {}, None
    for name in wmap:
        if ".ngram_embedding.shard_" in name and name.endswith(".weight"):
            shards[int(name.rsplit(".shard_", 1)[1].split(".")[0])] = name
        elif name.endswith(".ngram_embedding.weight_scale"):
            scale_name = name
    if not shards:
        sys.exit("no ngram_embedding shards in " + str(model))
    if sorted(shards) != list(range(len(shards))):
        sys.exit("the n-gram shards are not numbered 0..%d without gaps" % (len(shards) - 1))

    # The dtype is a property of the checkpoint, not a choice: every shard must agree, because one GGUF tensor
    # has one type. The first shard decides and the rest are checked against it below.
    hdr, base = safetensors_header(model / wmap[shards[0]])
    dtype = hdr[shards[0]]["dtype"]
    if dtype not in SOURCES:
        sys.exit("%s is %s, not a 2-D %s tensor" % (shards[0], dtype, " or ".join(SOURCES)))
    gguf_type, width, needs_scale = SOURCES[dtype]
    if needs_scale and scale_name is None:
        sys.exit("this checkpoint is %s but has no ngram_embedding.weight_scale" % dtype)
    scale = read_weight_scale(model, wmap, scale_name) if needs_scale else None

    parts, rows, dim = [], 0, None
    for k in range(len(shards)):
        name = shards[k]
        hdr, base = safetensors_header(model / wmap[name])
        t = hdr[name]
        if t["dtype"] != dtype or len(t["shape"]) != 2:
            sys.exit("%s is %s %s, not a 2-D %s tensor" % (name, t["dtype"], t["shape"], dtype))
        if dim is None:
            dim = t["shape"][1]
        if t["shape"][1] != dim or t["data_offsets"][1] - t["data_offsets"][0] != t["shape"][0] * dim * width:
            sys.exit("%s: inconsistent shape %s" % (name, t["shape"]))
        parts.append((model / wmap[name], base + t["data_offsets"][0], t["shape"][0] * dim * width))
        rows += t["shape"][0]
    print("%d shards, %d rows x %d, %s, %.2f GB"
          % (len(parts), rows, dim, dtype, rows * dim * width / 1e9), flush=True)

    kvs = [kv_string("general.architecture", "strata-ple"),
           kv_string("general.name", "PLE n-gram table, %s from the checkpoint" % dtype),
           kv_string("strata.ple.source", model.name)]
    if needs_scale:
        # FP8 rows are bytes of a scaled type, so the reader needs both parts spelled out. BF16 does not.
        kvs += [kv_string("strata.ple.format", "f8_e4m3"), kv_f32("strata.ple.scale", scale)]
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs)
    head += gguf_string("per_layer_token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", dim, rows)
    head += struct.pack("<I", gguf_type) + struct.pack("<Q", 0)
    head += b"\0" * ((-len(head)) % ALIGN)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    written = 0
    with open(tmp, "wb") as w:
        w.write(head)
        for i, (path, off, n) in enumerate(parts):
            with open(path, "rb") as f:
                f.seek(off)
                left = n
                while left:
                    b = f.read(min(CHUNK, left))
                    if not b:
                        sys.exit("short read in " + str(path))
                    w.write(b)
                    left -= len(b)
            written += n
            if i % 16 == 15 or i == len(parts) - 1:
                print("  %3d/%d shards, %.1f GB" % (i + 1, len(parts), written / 1e9), flush=True)
    if tmp.stat().st_size != len(head) + written:
        sys.exit("size check failed: %d != %d" % (tmp.stat().st_size, len(head) + written))
    tmp.replace(out)
    print("wrote %s (%d rows x %d %s, header %d B)" % (out, rows, dim, dtype, len(head)))


if __name__ == "__main__":
    main()
