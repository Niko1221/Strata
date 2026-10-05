"""tools/embd_bf16_pack.py - the token embedding exactly as the checkpoint ships it (BF16), as a GGUF the engine reads.

    python tools/embd_bf16_pack.py --model <checkpoint dir> --out <token-embd-bf16.gguf>

nvfp4_convert.py stores token_embd as Q8_0 (0.56% off per value). The engine keeps the table in mapped host memory
and reads one row per token, so BF16 costs 0.6 GB of RAM more and no VRAM; `--embd-gguf` takes it from this file.

The output is a one-tensor GGUF: `token_embd.weight`, ne = [n_embd, n_vocab], type BF16, the bytes copied straight
out of the safetensors file - nothing decoded or rounded.
"""
import argparse
import json
import pathlib
import struct
import sys

GGUF_TYPE_STRING = 8
GGML_TYPE_BF16 = 30
ALIGN = 32
CHUNK = 64 << 20


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
    ap.add_argument("--model", required=True, help="the checkpoint directory (model.safetensors.index.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, out = pathlib.Path(a.model), pathlib.Path(a.out)
    wmap = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]

    names = [n for n in wmap if n.endswith("embed_tokens.weight") and "visual" not in n and "mtp" not in n]
    if len(names) != 1:
        sys.exit("expected one text embed_tokens.weight in %s, found %s" % (model, names))
    name = names[0]
    hdr, base = safetensors_header(model / wmap[name])
    t = hdr[name]
    if t["dtype"] != "BF16" or len(t["shape"]) != 2:
        sys.exit("%s is %s %s, not a 2-D BF16 tensor" % (name, t["dtype"], t["shape"]))
    vocab, dim = t["shape"]
    n = vocab * dim * 2
    if t["data_offsets"][1] - t["data_offsets"][0] != n:
        sys.exit("%s: data size does not match its shape" % name)
    print("%s: %d x %d BF16, %.2f GB" % (name, vocab, dim, n / 1e9), flush=True)

    kvs = [kv_string("general.architecture", "strata-embd"),
           kv_string("general.name", "token embedding, BF16 as shipped"),
           kv_string("strata.embd.source", model.name)]
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs)
    head += gguf_string("token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", dim, vocab)
    head += struct.pack("<I", GGML_TYPE_BF16) + struct.pack("<Q", 0)
    head += b"\0" * ((-len(head)) % ALIGN)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as w, open(model / wmap[name], "rb") as f:
        w.write(head)
        f.seek(base + t["data_offsets"][0])
        left = n
        while left:
            b = f.read(min(CHUNK, left))
            if not b:
                sys.exit("short read in " + str(model / wmap[name]))
            w.write(b)
            left -= len(b)
    if tmp.stat().st_size != len(head) + n:
        sys.exit("size check failed: %d != %d" % (tmp.stat().st_size, len(head) + n))
    tmp.replace(out)
    print("wrote %s (header %d B)" % (out, len(head)))


if __name__ == "__main__":
    main()
