"""tools/embd_bf16_pack.py - the token embedding as BF16, as a GGUF the engine reads.

    python tools/embd_bf16_pack.py --model <checkpoint dir> --out <token-embd-bf16.gguf>
    python tools/embd_bf16_pack.py --gguf <model GGUF> --out <token-embd-bf16.gguf>

nvfp4_convert.py stores token_embd as Q8_0 (0.56% off per value). The engine keeps the table in mapped host memory
and reads one row per token, so BF16 costs 0.6 GB of RAM more and no VRAM; `--embd-gguf` takes it from this file.

`--model` copies the bytes straight out of the safetensors file - nothing decoded or rounded. `--gguf` is for a
model whose GGUF stores `token_embd.weight` in a type the engine's native embedding has no kernel for (Q6_K, as in
agentionai's Signal AP-Q4_K_XL, whose every dense weight is Q6_K): the table is dequantized and rounded to BF16 a
pass of rows at a time, the same conversion `tools/iq_pack.py --compat-bf16` applies to the small projections. The
worst error of that rounding is printed and stored in the file.

The output is a one-tensor GGUF: `token_embd.weight`, ne = [n_embd, n_vocab], type BF16.
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
ROWS = 4096        # --gguf: rows per dequantize pass (160 MB of float32 at 2560 wide)


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key, val):
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + gguf_string(val)


def gguf_head(source, name_label, dim, vocab, method=None):
    """The one-tensor file's header: `token_embd.weight` as [dim, vocab] BF16 at offset 0. Without `method` this is
    the header tools/embd_bf16_pack.py has always written, byte for byte."""
    kvs = [kv_string("general.architecture", "strata-embd"),
           kv_string("general.name", name_label),
           kv_string("strata.embd.source", source)]
    if method is not None:
        kvs.append(kv_string("strata.embd.method", method))
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs)
    head += gguf_string("token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", dim, vocab)
    head += struct.pack("<I", GGML_TYPE_BF16) + struct.pack("<Q", 0)
    return head + b"\0" * ((-len(head)) % ALIGN)


def publish(out, head, total, write_body):
    """Write head + `total` bytes of body through a temporary file, then move it into place."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as w:
        w.write(head)
        write_body(w)
    if tmp.stat().st_size != len(head) + total:
        tmp.unlink()
        sys.exit("size check failed: %d != %d" % (tmp.stat().st_size, len(head) + total))
    tmp.replace(out)
    print("wrote %s (header %d B, table %d B)" % (out, len(head), total))


def from_checkpoint(model, out):
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

    head = gguf_head(model.name, "token embedding, BF16 as shipped", dim, vocab)

    def body(w):
        with open(model / wmap[name], "rb") as f:
            f.seek(base + t["data_offsets"][0])
            left = n
            while left:
                b = f.read(min(CHUNK, left))
                if not b:
                    sys.exit("short read in " + str(model / wmap[name]))
                w.write(b)
                left -= len(b)

    publish(out, head, n, body)


def from_gguf(gguf_path, tensor, out):
    """`tensor` from a model GGUF, dequantized to BF16 a pass of rows at a time."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import numpy as np
    import iq_pack as P

    model = P.Model(gguf_path)          # the shards of a split model, found by shard 1's name
    if tensor not in model.where:
        sys.exit("%s holds no %s" % (gguf_path, tensor))
    gguf, t, mm, where = model.where[tensor]
    if len(t.shape) != 2:
        sys.exit("%s: %s is %d-D; the engine reads a 2-D table" % (where.name, tensor, len(t.shape)))
    dim, vocab = t.shape
    stored = t.expected_bytes()
    if stored is None or stored % vocab:
        sys.exit("%s: %s's rows are not whole %s blocks" % (where.name, tensor, t.type_name))
    row, total = stored // vocab, vocab * dim * 2
    base = gguf.data_start + t.offset
    print("%s: %s is %d x %d %s (%.2f GB); BF16 table %.2f GB" %
          (where, tensor, dim, vocab, t.type_name, stored / 1e9, total / 1e9), flush=True)

    head = gguf_head(where.name, "token embedding, BF16 dequantized from %s" % t.type_name, dim, vocab,
                     method="dequantize->bf16 round-to-nearest-even (%s)" % t.type_name)
    worst = [0.0]

    def body(w):
        left, done = vocab, 0
        while left:
            rows = min(ROWS, left)
            raw = np.asarray(mm[base + done * row: base + (done + rows) * row], dtype=np.uint8)
            if raw.size != rows * row:
                sys.exit("short read in %s at row %d" % (where, done))
            data = P.bf16_bytes(raw, t.type_name)
            got = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            values = P.dequantize(raw, t.type_name)
            worst[0] = max(worst[0], float(np.max(np.abs(got.astype(np.float64) - values.astype(np.float64)))))
            w.write(data)
            done += rows
            left -= rows
            if not left or done % (ROWS * 16) == 0:
                print("  %d / %d rows, worst |err| so far %.5f" % (done, vocab, worst[0]), flush=True)

    publish(out, head, total, body)
    print("worst rounding error of the %s -> BF16 conversion: %.5f" % (t.type_name, worst[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="the checkpoint directory (model.safetensors.index.json)")
    src.add_argument("--gguf", help="a GGUF of the model: its embedding, dequantized to BF16")
    ap.add_argument("--tensor", default="token_embd.weight", help="--gguf: the tensor to read (default token_embd.weight)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.gguf:
        from_gguf(pathlib.Path(a.gguf), a.tensor, pathlib.Path(a.out))
    else:
        from_checkpoint(pathlib.Path(a.model), pathlib.Path(a.out))


if __name__ == "__main__":
    main()
