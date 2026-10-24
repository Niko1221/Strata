"""Fetch only layer-zero MoE tensor ranges from the pinned public Q2_0 GGUF."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

REVISION = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
FILE = "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE}"
TOTAL = 37623740192


def read_range(start, length):
    # A distinct query prevents intermediary caches reusing another Range's body.
    url = URL + f"?moe_range={start}-{start + length - 1}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=90) as response:
        expected = f"bytes {start}-{start + length - 1}/{TOTAL}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise RuntimeError(f"Invalid range response: {response.status}, {response.headers.get('Content-Range')}")
        data = response.read(length + 1)
        if len(data) != length:
            raise RuntimeError(f"Short range: {len(data)} != {length}")
        return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reader-root", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.reader_root))
    from gguf_reader import GGUFFile
    header = args.output / "header-prefix.bin"
    if not header.exists():
        header.write_bytes(read_range(0, 16 * 1024 * 1024))
    g = GGUFFile(header)
    assert g.metadata["general.architecture"] == "qwen4exp"
    assert g.metadata["qwen4exp.expert_count"] == 512
    assert g.metadata["qwen4exp.expert_used_count"] == 10
    selected = [t for t in g.tensors if t.name.startswith("blk.0.ffn_")]
    assert len(selected) == 8
    entries = []
    for t in selected:
        size = t.expected_bytes()
        assert size and t.offset >= 0 and g.data_start + t.offset + size <= TOTAL
        path = args.output / (t.name + ".bin")
        pieces = [(i, min(8 * 1024 * 1024, size - i)) for i in range(0, size, 8 * 1024 * 1024)]
        if not path.exists():
            partial = path.with_suffix(".partial")
            with partial.open("wb") as target:
                def get(piece):
                    i, n = piece
                    for attempt in range(3):
                        try:
                            return read_range(g.data_start + t.offset + i, n)
                        except Exception:
                            if attempt == 2:
                                raise
                            time.sleep(1)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    for data in pool.map(get, pieces):
                        target.write(data)
                        print(json.dumps({"tensor": t.name, "bytes": target.tell(), "total": size}), flush=True)
            os.replace(partial, path)
        assert path.stat().st_size == size
        # Independent bounded re-reads catch wrong-offset/cache/concatenation errors.
        with path.open("rb") as f:
            first = f.read(256)
            f.seek(size - 256)
            last = f.read(256)
        assert first == read_range(g.data_start + t.offset, 256)
        assert last == read_range(g.data_start + t.offset + size - 256, 256)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = {"name": t.name, "shape_gguf": t.shape, "type_id": t.type_id,
                 "type_name": t.type_name, "offset": g.data_start + t.offset,
                 "bytes": size, "file": path.name, "sha256": digest,
                 "boundary_reread_passed": True}
        entries.append(entry)
        print(json.dumps(entry), flush=True)
    manifest = {"repository": REPO, "revision": REVISION, "file": FILE,
                "url": URL, "shard_bytes": TOTAL, "layer": 0,
                "header_prefix_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
                "data_start": g.data_start, "tensors": entries,
                "provenance_limit": "TLS range reads pinned to revision; per-tensor hashes and boundary re-reads. Whole-shard LFS hash not verified."}
    out = args.output / "WEIGHTS.json"
    if out.exists():
        assert json.loads(out.read_text()) == manifest
    else:
        out.write_text(json.dumps(manifest, indent=2) + "\n")
    print("REAL_MOE_WEIGHTS_READY", flush=True)


if __name__ == "__main__":
    main()
