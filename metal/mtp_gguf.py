"""metal/mtp_gguf.py - the MTP draft head as an MTP-only GGUF for llama.cpp (`-md`, draft-mtp), from the checkpoint.

The GSQ-RCO GGUFs ship no MTP head, and Strata's own mtp-*.gguf (tools/mtp_pack.py) is in the CUDA engine's layout.
llama.cpp converts the head itself since #29761 (`convert_hf_to_gguf.py --mtp`): an MTP-only file is the MTP block
plus the embeddings and the LM head, and it drafts on top of any qwen4exp trunk (the trunk's hc-wide residual feeds it).
So this builds a sparse copy of the checkpoint holding only what that conversion reads, and converts it:

    python metal/mtp_gguf.py --mtp-dir <tools/mtp_fetch.py's --out> --llama <llama.cpp checkout> --out mtp.gguf

  - the 31 mtp.* tensors: already fetched and SHA-256 checked by tools/mtp_fetch.py (setup's step 6);
  - embed_tokens and lm_head (1.27 GB each): read here by HTTP range from the same pinned revision, under the same
    rule as mtp_fetch (#327: a 206 whose Content-Range is the range asked for; resumable);
  - config, tokenizer and chat template: the pinned revision's small files.

The conversion needs PyTorch (llama.cpp's converter imports it); `--python` names an interpreter that has it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import mtp_fetch as F  # noqa: E402  (the pinned revision, range reads with the #327 checks)

EXTRA = ("model.language_model.embed_tokens.weight", "lm_head.weight")
# sha256 over every tensor of a right conversion (name, type, data; sorted by name), for the pinned checkpoint, --outtype
# q8_0 and llama.cpp at metal/CMakeLists.txt's commit: two conversions (torch 2.11 and 2.12, numpy 2.4) gave exactly
# this.  llama.cpp's own pin, numpy 2.2.6, gave other bytes in 5 tensors and a head whose drafts were never accepted
# (0 of 354), with no error - so a conversion is checked against it, and a wrong one is not kept.
Q8_0_DIGEST = "982c44c1736c0f4589167d7435b5ba134c79c981f2d3852479d317febfd79fb1"
SMALL = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja")
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def fetch_extra(out: Path) -> list[dict]:
    """embed_tokens and lm_head as raw files beside mtp_fetch's tensors -> their rows (name, dtype, shape, file)."""
    index = json.loads(F.get(F.REPO + "model.safetensors.index.json"))["weight_map"]
    rows = []
    for name in EXTRA:
        shard = index[name]
        base, header = F.shard_header(shard)
        meta = header[name]
        a, b = meta["data_offsets"]
        start, end, size = base + a, base + b - 1, b - a
        path = out / "tensors" / (name + ".bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        have = path.stat().st_size if path.exists() else 0
        if have > size:
            path.unlink()
            have = 0
        with open(path, "ab") as f:
            pos = start + have
            while pos <= end:
                stop = min(pos + (64 << 20) - 1, end)
                f.write(F.get(F.REPO + shard, pos, stop))
                pos = stop + 1
                print(f"{name} {100 * (pos - start) / size:.0f}%", file=sys.stderr)
        if path.stat().st_size != size:
            sys.exit(f"{path}: {path.stat().st_size} bytes, not {size}")
        rows.append({"name": name, "dtype": meta["dtype"], "shape": meta["shape"], "file": str(path)})
    return rows


def write_safetensors(rows: list[dict], path: Path) -> None:
    """One safetensors file from raw tensor files: an 8-byte header length, the JSON header, the data in order."""
    header, offset = {}, 0
    for r in rows:
        n = os.path.getsize(r["file"])
        if n != DTYPE_BYTES[r["dtype"]] * math.prod(r["shape"]):
            sys.exit(f"{r['name']}: {n} bytes do not match {r['dtype']} {r['shape']}")
        header[r["name"]] = {"dtype": r["dtype"], "shape": r["shape"], "data_offsets": [offset, offset + n]}
        offset += n
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(blob)) + blob)
        for r in rows:
            with open(r["file"], "rb") as f:
                while chunk := f.read(64 << 20):
                    out.write(chunk)


def tensor_digest(path: str, gguf_py: Path) -> str:
    sys.path.insert(0, str(gguf_py))
    import gguf  # noqa: E402
    h = hashlib.sha256()
    for t in sorted(gguf.GGUFReader(path).tensors, key=lambda t: t.name):
        h.update(t.name.encode())
        h.update(t.tensor_type.name.encode())
        h.update(t.data.tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtp-dir", required=True, help="tools/mtp_fetch.py's --out (mtp-manifest.json, tensors/)")
    ap.add_argument("--llama", required=True, help="a llama.cpp checkout with qwen4exp MTP (#29761 or later)")
    ap.add_argument("--out", required=True, help="the MTP-only GGUF to write")
    ap.add_argument("--outtype", default="q8_0", help="the converter's --outtype (default q8_0)")
    ap.add_argument("--python", default=sys.executable, help="an interpreter with torch for the converter")
    a = ap.parse_args()
    mtp = Path(a.mtp_dir)
    F.resolve_repo()
    if F.verify(str(mtp)):
        sys.exit(f"{mtp}: the MTP tensors are missing or wrong; run tools/mtp_fetch.py fetch --out {mtp} first")
    manifest = json.loads((mtp / "mtp-manifest.json").read_text(encoding="utf-8"))
    rows = [{"name": r["name"], "dtype": r["dtype"], "shape": r["shape"], "file": str(mtp / r["file"])} for r in manifest]
    rows += fetch_extra(mtp)
    with tempfile.TemporaryDirectory(prefix="strata-mtp-hf-", dir=mtp) as d:
        hf = Path(d) / "Qwen3.8-Flash-Next"            # the converter names the model after its folder
        hf.mkdir()
        for name in SMALL:
            (hf / name).write_bytes(F.get(F.REPO + name))
        write_safetensors(rows, hf / "model-mtp.safetensors")
        (hf / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {}, "weight_map": {r["name"]: "model-mtp.safetensors" for r in rows}}), encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(Path(a.llama) / "gguf-py"))
        cmd = [a.python, str(Path(a.llama) / "convert_hf_to_gguf.py"), str(hf), "--mtp", "--outtype", a.outtype,
               "--outfile", a.out]
        print("> " + " ".join(cmd), file=sys.stderr)
        rc = subprocess.call(cmd, env=env)
    if rc != 0:
        return rc
    if F.pinned() and a.outtype == "q8_0":
        got = tensor_digest(a.out, Path(a.llama) / "gguf-py")
        if got != Q8_0_DIGEST:
            os.remove(a.out)
            sys.exit(f"the converted MTP head is not the expected one (tensor sha256 {got[:16]}..., expected "
                     f"{Q8_0_DIGEST[:16]}...); not kept. A converter environment with numpy 2.2.x gives a broken head: "
                     "numpy 2.4 is needed")
        print("MTP head: the expected tensors (sha256 checked)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
