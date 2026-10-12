"""tools/mtp_rt.py - plan v0.3 P6: the MTP draft layer's runtime files, from the packed MTP GGUF.

    python tools/mtp_rt.py --gguf <Strata>/mtp-bf16/mtp-q2_0.gguf --out <Strata>/mtp-bf16/rt

Writes
  experts.bin   512 routed experts.  Q2_0 keeps the engine's legacy S2 blob layout
                (`include/strata/kernels/cpu/expert.hpp`): gate/up rows interleaved, then down rows, with codes and
                scales in separate planes.  Q4_0 and Q8_0 keep native GGUF rows in [gate | up | down] order for the
                GPU native grouped-expert kernels.
  experts.txt   the expert layout, GGML types and bytes per expert.  Engines predating this file keep treating a
                runtime without it as the legacy Q2_0 S2 layout.
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
Q2_BLOB = 3 * (H * FF * 18 // 64)
QTYPE_NAMES = {2: "q4_0", 8: "q8_0", 42: "q2_0"}
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
    assert len(out) == Q2_BLOB, len(out)
    return out


def expert_plan(gu_type: int, dn_type: int, gu: np.ndarray, dn: np.ndarray) -> tuple[str, int]:
    """Return (layout, bytes per expert), validating the packed tensors the selected runtime consumes."""
    if gu_type != dn_type or gu_type not in QTYPE_NAMES:
        raise ValueError(f"unsupported MTP expert types: gate/up={gu_type}, down={dn_type}")
    if gu.shape[:2] != (NE, 2 * FF) or dn.shape[:2] != (NE, H):
        raise ValueError(f"unexpected MTP expert shapes: gate/up={gu.shape}, down={dn.shape}")
    if gu_type == 42:
        if gu.shape != (NE, 2 * FF, H // 64 * 18) or dn.shape != (NE, H, FF // 64 * 18):
            raise ValueError(f"unexpected Q2_0 expert shapes: gate/up={gu.shape}, down={dn.shape}")
        return "s2", Q2_BLOB
    block_bytes = 18 if gu_type == 2 else 34
    expected_gu = (NE, 2 * FF, H // 32 * block_bytes)
    expected_dn = (NE, H, FF // 32 * block_bytes)
    if gu.shape != expected_gu or dn.shape != expected_dn:
        raise ValueError(f"unexpected {QTYPE_NAMES[gu_type]} expert shapes: gate/up={gu.shape}, down={dn.shape}")
    return "native", int(gu[0].nbytes + dn[0].nbytes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = gguf.GGUFReader(a.gguf)
    tens = {t.name: t for t in r.tensors}
    gu_t = tens["mtp.layers.0.mlp.experts.gate_up_proj"]
    dn_t = tens["mtp.layers.0.mlp.experts.down_proj"]
    gu = np.asarray(gu_t.data)
    dn = np.asarray(dn_t.data)
    gu_type, dn_type = int(gu_t.tensor_type), int(dn_t.tensor_type)
    layout, expert_blob = expert_plan(gu_type, dn_type, gu, dn)
    with open(out / "experts.bin", "wb") as f:
        for e in range(NE):
            raw = blob_of(gu[e], dn[e]) if layout == "s2" else gu[e].tobytes() + dn[e].tobytes()
            assert len(raw) == expert_blob, (e, len(raw), expert_blob)
            f.write(raw)
    (out / "experts.txt").write_text(
        f"version 1\nlayout {layout}\ngate_up_type {gu_type}\ndown_type {dn_type}\nblob_bytes {expert_blob}\n",
        encoding="utf-8",
    )
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
    print(f"experts.bin {NE * expert_blob} B ({QTYPE_NAMES[gu_type]} {layout}), "
          f"dense.bin {off} B, {len(lines)} tensors -> {out}")
    for l in lines:
        print("  " + l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
