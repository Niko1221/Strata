"""Fetch one expert and one dense EXL3 linear from each released bitrate branch and build a mini model
directory, so `strata-exl3-gpu` can run the decode on real weights at every K.

    python3 tools/exl3/fetch_branch_tensors.py <out_dir>

Each branch of turboderp/Qwen3.8-Flash-Next-exl3 is a different K per tensor (experts vs attn/head), and
the engine's decode is K-generic: this proves it on the real files, not just synthetic fixtures.
"""
from __future__ import annotations

import json
import pathlib
import struct
import sys
import urllib.request

REPO = "turboderp/Qwen3.8-Flash-Next-exl3"
BRANCHES = ["2.05bpw_h4_ng4", "3.05bpw_h5_ng5", "4.05bpw_h6_ng6", "5.05bpw_h6_ng6", "6.05bpw_h6_ng6"]
BASES = [
    "model.language_model.layers.0.mlp.experts.0.gate_proj",       # the branch's expert K
    "model.language_model.layers.0.linear_attn.in_proj_qkv",       # the branch's attn K
]
MARKERS = [".mul1", ".mcg"]


def fetch(url: str, start: int | None = None, end: int | None = None, tries: int = 6) -> bytes:
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            if start is not None:
                req.add_header("Range", "bytes=%d-%d" % (start, end))
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if start is None or len(data) == end - start + 1:
                return data
        except Exception as e:                       # noqa: BLE001
            last = e
    raise RuntimeError("fetch failed: %s (%s)" % (url, last))


def main() -> int:
    out_root = pathlib.Path(sys.argv[1])
    for branch in BRANCHES:
        url = "https://huggingface.co/%s/resolve/%s/" % (REPO, branch)
        index = json.loads(fetch(url + "model.safetensors.index.json"))
        want = set()
        for base in BASES:
            want.update(base + s for s in (".trellis", ".suh", ".svh"))
            want.update(base + m for m in MARKERS)
        shard = index["weight_map"][BASES[0] + ".trellis"]
        head = fetch(url + shard, 0, 7)
        hlen = struct.unpack("<Q", head)[0]
        header = json.loads(fetch(url + shard, 8, 8 + hlen - 1))
        tensors = {}
        for name in sorted(want):
            if name not in header:
                continue
            t = header[name]
            a, b = t["data_offsets"]
            tensors[name] = (t["dtype"], t["shape"], fetch(url + shard, 8 + hlen + a, 8 + hlen + b - 1))
        out = out_root / branch
        out.mkdir(parents=True, exist_ok=True)
        blob, hdr, at = bytearray(), {}, 0
        for name, (dtype, shape, data) in tensors.items():
            hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(data)]}
            blob += data
            at += len(data)
        body = json.dumps(hdr).encode()
        (out / "mini.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + bytes(blob))
        (out / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"format": "pt"}, "weight_map": {n: "mini.safetensors" for n in tensors}}))
        ks = []
        for base in BASES:
            t = tensors.get(base + ".trellis")
            ks.append("%s K=%d cb=%d" % (base.rsplit(".", 1)[-1], t[1][2] * 16 // 256 if t else -1,
                                         2 if base + ".mul1" in tensors else 1))
        print("%-16s %6.1f KiB  %s" % (branch, at / 1024, "; ".join(ks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
