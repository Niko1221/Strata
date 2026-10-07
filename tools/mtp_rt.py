"""tools/mtp_rt.py - plan v0.3 P6: the MTP draft layer's runtime files, from the packed MTP GGUF.

    python tools/mtp_rt.py --gguf <Strata>/mtp-bf16/mtp-q2_0.gguf --out <Strata>/mtp-bf16/rt
    python tools/mtp_rt.py --gguf .../mtp-q8_0.gguf --out .../rt-q8_0
    python tools/mtp_rt.py --gguf .../mtp-bf16.gguf --out .../rt-bf16

Which of the two expert layouts is written is decided by the GGUF's own `strata.mtp.expert_format`, set by
tools/mtp_pack.py.  Q2_0 gets the engine's blob; q8_0 and bf16 get the native GGUF layout the rest of the engine
already uses, and an `experts.fmt` marker telling the engine which ggml types it is looking at.

Writes
  experts.bin   At Q2_0, 512 routed experts in the engine's blob layout (`include/strata/kernels/cpu/expert.hpp`):
                gate/up rows interleaved (2r = gate r, 2r+1 = up r), then down rows; the Q2_0 codes in one plane and
                the fp16 scales in another.  A lossless relayout of the GGUF's Q2_0 blocks (same bytes,
                `cpu_expert_fixture.py`).  At q8_0/bf16, one `NativeExpertLayout` blob per expert, in the same
                [gate rows | up rows | down rows] order the engine reads through `up_off`/`down_off`.  The two
                packed tensors cannot simply be concatenated: the engine strides the file by one blob per expert
                (`grp_ptr[e] = base + e * L.bytes`), so whole-tensor order would give expert e the head of
                expert e+1.  Per expert it is gate_up[e] (1280 rows: gate 0..639 then up 640..1279) followed by
                down[e] (2560 rows), and each expert's three pieces match `blk.48.ffn_gate_exps`,
                `blk.48.ffn_up_exps` and `blk.48.ffn_down_exps` of expert e in
                `mtp-Qwen3.8-Flash-Next-BF16.gguf`, which was converted independently of these tools.
  experts.fmt   Written only for the non-Q2_0 formats: one line `<format> <gu_type> <d_type>`, e.g. `q8_0 8 8`, with
                the ggml type ids the engine's native expert path switches on.  Absent means Q2_0, which is what
                every `rt/` written before this file gained the field already is, so those are unchanged.
  dense.bin     every other tensor: the large projections quantized to Q8_0 (ggml's reference rounding) so the
                engine's multi-column MMVQ can run them; the hyper-connection and router weights kept BF16; the
                RMSNorm weights as F32 with the Gemma "+1" applied (vLLM GemmaRMSNorm scales by 1 + w).
  dense.txt     one line per tensor: name kind rows cols offset bytes   (kind = q8_0 | bf16 | f32)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402

H, FF, NE = 2560, 640, 512
BLOB = 3 * (H * FF * 18 // 64)
GU, DN = "mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"
# Expert format -> (ggml type id, values per block, bytes per block).  Ids are ggml's: 8 = Q8_0, 30 = BF16, and 42 =
# Q2_0 is this repo's own extension, which is why Q2_0 is the format the absence of a marker means.
FMTS = {"q2_0": (42, 64, 18), "q8_0": (8, 32, 34), "bf16": (30, 1, 2)}
Q8 = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
      "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
      "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"}


def q8_0(x: np.ndarray) -> bytes:
    """ggml quantize_row_q8_0_ref over rows of a 2-D float32 array: 32-value blocks, fp16 d = amax / 127."""
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def blob_of(gu: np.ndarray, dn: np.ndarray) -> bytes:
    """gu: (1280, 720) Q2_0 rows (gate rows 0..639, up rows 640..1279); dn: (2560, 180)."""
    gub = gu.reshape(2 * FF, H // 64, 18)
    inter = np.empty_like(gub)
    inter[0::2] = gub[:FF]
    inter[1::2] = gub[FF:]
    dnb = dn.reshape(H, FF // 64, 18)
    parts = [inter[:, :, 2:].tobytes(), dnb[:, :, 2:].tobytes(), inter[:, :, :2].tobytes(), dnb[:, :, :2].tobytes()]
    out = b"".join(parts)
    assert len(out) == BLOB, len(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = gguf.GGUFReader(a.gguf)
    tens = {t.name: t for t in r.tensors}
    fmt = str(r.fields["strata.mtp.expert_format"].contents()) if "strata.mtp.expert_format" in r.fields else "q2_0"
    if fmt not in FMTS:
        raise ValueError(f"unknown expert format {fmt!r}, expected one of {sorted(FMTS)}")
    type_id, block, block_bytes = FMTS[fmt]
    gu, dn = np.asarray(tens[GU].data), np.asarray(tens[DN].data)
    # gguf-py hands back the packed bytes, so the last axis is bytes-per-row rather than values-per-row.
    gu_row, dn_row = H // block * block_bytes, FF // block * block_bytes
    want = ((NE, 2 * FF, gu_row), (NE, H, dn_row))
    assert (gu.shape, dn.shape) == want, (gu.shape, dn.shape, want)
    with open(out / "experts.bin", "wb") as f:
        if fmt == "q2_0":
            for e in range(NE):
                f.write(blob_of(gu[e], dn[e]))
        else:
            # One blob per expert: gate_up[e] (gate rows then up rows) immediately followed by down[e].  The engine
            # reads expert e at base + e * L.bytes and takes down at L.down_off inside that blob, so a blob must be
            # three contiguous pieces of the SAME expert.  All 512 gate_up first, then all 512 down, would instead
            # hand every expert the head of its neighbour.
            for e in range(NE):
                f.write(gu[e].tobytes())
                f.write(dn[e].tobytes())
    if fmt != "q2_0":
        (out / "experts.fmt").write_text(f"{fmt} {type_id} {type_id}\n", encoding="utf-8")
    lines = []
    off = 0
    with open(out / "dense.bin", "wb") as f:
        for name, t in tens.items():
            if "experts." in name:
                continue
            short = name[len("mtp."):]
            short = short[len("layers.0."):] if short.startswith("layers.0.") else short
            data = np.asarray(t.data)
            if int(t.tensor_type) == 0:        # F32 norm weights: raw GemmaRMSNorm w -> 1 + w
                arr = (data.astype(np.float32) + 1.0)
                raw, kind, rows, cols = arr.tobytes(), "f32", 1, arr.size
            else:                              # BF16
                u16 = data.view(np.uint16) if data.dtype != np.uint16 else data
                u16 = u16.reshape(data.shape[0], -1)
                rows, cols = u16.shape
                if short in Q8:
                    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                    raw, kind = q8_0(f32), "q8_0"
                else:
                    raw, kind = u16.tobytes(), "bf16"
            pad = (-off) % 256
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append(f"{short} {kind} {rows} {cols} {off} {len(raw)}")
            off += len(raw)
    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"experts.bin {NE * (2 * FF * gu_row + H * dn_row)} B ({fmt}), dense.bin {off} B, "
          f"{len(lines)} tensors -> {out}")
    for l in lines:
        print("  " + l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
