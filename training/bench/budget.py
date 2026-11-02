#!/usr/bin/env python3
"""How long each scheme can train on a given machine - computed, never laddered.

"Full context" is not a user statistic and not a fixed 128K. It is a **capability limit**:

    max_train_len(scheme, device, pack) = max{ T : peak(T) <= VRAM - margin }   (capped by model 128K)

The point of the stage-B probe is to produce that number for every (algorithm x platform) pair and the
resource figures at that point, so a user with a different card only changes two inputs and gets their own
limit - instead of us warm-up-running a ladder on this one box.

Two profiles, because the two stacks fail differently:

  engine (strata.exe, forward only)
      peak(T) = weights_vram + kv(T, kv_mode) + workspace
      weights_vram is what the loader keeps on the card (the rest sits in pinned host RAM), so it does
      NOT grow with T. Measured: 125B IQ3_XXS keeps 24 GiB on the card and 30 GiB pinned.
      kv per token is calibrated from two points in bench_engine_128k.json (engine-reported, not guessed):
          8K fp16  -> 0.21 GiB   => 27.5 KiB/token
          128K int8 -> 1.68 GiB  => 13.75 KiB/token   (exactly half: 1 byte vs 2)
      other modes scale by bytes per element (q4_0 4-bit, k8v4 = int8 keys + 4-bit values).

  hf (torch path, forward + backward)
      delegated to mem_guard.estimate_peak_gib / max_seq_len, which is fitted to six measured points
      (residual <= 0.6 GiB). BASE_GIB carries the resident weights plus the torch allocator's floor, so a
      different pack tier moves it - pass --base-gib (or STRATA_BASE_GIB) instead of editing code.

Why the engine's `nvidia-smi` peak is NOT the constraint: the expert cache grows to fill whatever VRAM is
left, so peak sits at ~48.1 GiB on a 48 GiB card at both 8K and 128K. Using that number as the budget would
wrongly conclude "8K is already full". The real headroom is `VRAM - weights - workspace` against the KV.

Usage:
    python budget.py                                   # the 48 GiB box this repo is developed on
    python budget.py --vram 24 --ram 32                # a 4090
    python budget.py --vram 80 --ram 128 --json budget.json
    python budget.py --matrix                          # the whole device grid in one table
    python budget.py --vram 24 --max-pack-gib          # "how big a pack fits on this card?"
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))


import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mem_guard  # noqa: E402  (the HF profile's fitted model lives there)

SAFETY = 0.95          # allocator fragmentation + the driver's own overhead
WORKSPACE_GIB = mem_guard.ENGINE_WORKSPACE_GIB   # 单一出处：mem_guard（见那里的注释）

# engine weights resident on the card (loader keeps the rest in pinned host RAM, see the docstring).
# This is the ONE number a different pack tier changes, so it is a flag, not a constant.
ENGINE_WEIGHTS_GIB = mem_guard.ENGINE_WEIGHTS_GIB
# bytes per KV element by mode; fp16/int8 are calibrated, the 4-bit modes follow from the byte count
KV_BYTES = mem_guard.KV_BYTES
# the two calibrated points: 27.5 KiB/token at 2 bytes => 13.75 KiB per byte per token
KV_KIB_PER_BYTE_TOKEN = mem_guard.KV_KIB_PER_BYTE_TOKEN

# the model's own context ceiling: no scheme can be trained past this regardless of VRAM
MODEL_CONTEXT = 131072


def kv_gib(seq_len: int, mode: str) -> float:
    return seq_len * KV_KIB_PER_BYTE_TOKEN * KV_BYTES[mode] / 2 ** 20


def engine_max(seq_len_budget_gib: float, mode: str, weights: float = ENGINE_WEIGHTS_GIB) -> int:
    avail = seq_len_budget_gib - weights - WORKSPACE_GIB
    if avail <= 0:
        return 0
    per_token = KV_KIB_PER_BYTE_TOKEN * KV_BYTES[mode] / 2 ** 20
    return min(MODEL_CONTEXT, int(avail / per_token) // 256 * 256)


# `--kv auto` 的档位定义在 mem_guard（与 generate.cpp / setup.py 的 KV_AUTO_STEPS 对齐，
# 含 256K 以上降到 q4_0 那一档；这里以前只写到 k8v4，两份已经开始漂移了）。
kv_mode_for = mem_guard.kv_mode_for


def max_engine_weights(vram: float, seq_len: int = MODEL_CONTEXT) -> float:
    """Largest resident-weight footprint that still fits `seq_len` on this card.

    This is the answer to "what pack tier should I use on a 24 GiB card?" - it is the inverse of the limit
    solve, and it is usually more actionable than the limit itself.
    """
    mode = kv_mode_for(seq_len)
    return vram * SAFETY - WORKSPACE_GIB - kv_gib(seq_len, mode)


def solve(vram: float, ram: float, weights: float = ENGINE_WEIGHTS_GIB,
          base_gib: float = mem_guard.BASE_GIB, grpo_k: int = 4) -> list[dict]:
    budget = vram * SAFETY
    rows: list[dict] = []

    # --- engine: forward only. The KV mode is a function of the length (`--kv auto`), and the length is a
    # function of the mode, so settle it as a fixed point instead of guessing which mode comes first.
    t = MODEL_CONTEXT
    for _ in range(3):
        mode = kv_mode_for(t)
        t = engine_max(budget, mode, weights)
        if t <= 0:
            break
    if t <= 0:
        rows.append({"profile": "engine-all", "vram_gib": vram, "kv_mode": "-", "max_seq": 0,
                     "peak_at_max_gib": 0.0, "capped_by": "weights do not fit"})
    else:
        mode = kv_mode_for(t)
        rows.append({"profile": "engine-all", "vram_gib": vram, "kv_mode": mode, "max_seq": t,
                     "peak_at_max_gib": round(weights + kv_gib(t, mode) + WORKSPACE_GIB, 1),
                     "capped_by": "model context" if t >= MODEL_CONTEXT else "vram",
                     "note": "ce_mezo / dpo_mezo / grpo_mezo share this (one forward at a time)"})

    # --- hf: forward + backward (ckpt) and forward twice (mezo); staging is the MoE refetch path.
    # grpo is listed twice on purpose: whether the K trajectories are resident together is an implementation
    # choice, not a property of the algorithm, and the two cases differ by a lot at long T.
    old_base = mem_guard.BASE_GIB
    mem_guard.BASE_GIB = base_gib
    try:
        grid = (
            ("mezo", 1, 0.0, "hf-ce_mezo"),
            ("mezo", grpo_k, 0.0, f"hf-grpo_mezo(batch K={grpo_k})"),
            ("bwd", 1, 0.0, "hf-ce_bwd(ckpt)"),
            ("bwd", 2, 0.0, "hf-dpo_bwd(ckpt,rows=2)"),
            ("bwd", grpo_k, 0.0, f"hf-grpo_bwd(ckpt,batch K={grpo_k})"),
            ("bwd", 1, mem_guard.MOE_STAGING_GIB, "hf-ce_bwd(refetch)"),
        )
        for mode, nrows, staging, label in grid:
            t = mem_guard.max_seq_len(mode, nrows, vram, ckpt=True, moe_staging=staging)
            t = min(MODEL_CONTEXT, t // 256 * 256)
            est = mem_guard.estimate_peak_gib(t, nrows, mode, ckpt=True, moe_staging=staging) if t else 0.0
            rows.append({"profile": label, "vram_gib": vram, "kv_mode": "n/a", "max_seq": t,
                         "peak_at_max_gib": round(est, 1),
                         "capped_by": "model context" if t >= MODEL_CONTEXT else ("budget" if t else "weights")})
    finally:
        mem_guard.BASE_GIB = old_base

    for r in rows:
        r["host_ram_gib"] = ram
        r["fits_ram"] = ram >= weights        # the rest of the pack is pinned host memory
    return rows


def print_table(rows: list[dict], vram: float, ram: float, weights: float, base_gib: float) -> None:
    print(f"device: {vram:.0f} GiB VRAM / {ram:.0f} GiB RAM   "
          f"(budget = {vram * SAFETY:.1f} GiB after the {SAFETY:.0%} safety factor)")
    print(f"pack:   engine resident {weights:.1f} GiB on card / HF BASE_GIB {base_gib:.1f} GiB\n")
    print(f"{'profile':<32}{'max seq':>9}  {'kv':<6}{'peak@max':>9}  capped by")
    print("-" * 78)
    for r in rows:
        peak = f"{r['peak_at_max_gib']:.1f}"
        print(f"{r['profile']:<32}{r['max_seq']:>9}  {r['kv_mode']:<6}{peak:>9}  {r['capped_by']}")


def _host_ram_gib() -> float:
    """Host RAM, best effort - it never enters the limit itself, it only says whether the pack's pinned half
    fits. Returns 0.0 when it cannot be read (the caller then skips that column)."""
    try:                                        # Linux / WSL
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 2 ** 20
    except Exception:
        pass
    try:                                        # Windows
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys / 2 ** 30
    except Exception:
        pass
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vram", type=float, default=None,
                    help="GPU memory in GiB (default: read from the current device - this tool answers for the "
                         "machine it runs on, not for a hypothetical one)")
    ap.add_argument("--ram", type=float, default=None, help="host RAM in GiB (default: read from this machine)")
    ap.add_argument("--weights-gib", type=float, default=ENGINE_WEIGHTS_GIB,
                    help="engine: weights resident on the card (24 for 125B IQ3_XXS)")
    ap.add_argument("--base-gib", type=float, default=mem_guard.BASE_GIB,
                    help="HF: BASE_GIB of the fitted model (moves with the pack tier)")
    ap.add_argument("--grpo-k", type=int, default=4, help="group size used for the batch-resident grpo rows")
    ap.add_argument("--max-pack-gib", action="store_true",
                    help="only answer: how large a resident pack still fits 128K on this card")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    vram = a.vram if a.vram is not None else mem_guard._total_gib()
    ram = a.ram if a.ram is not None else _host_ram_gib()

    if a.max_pack_gib:
        print(f"{vram:.0f} GiB card, target T={MODEL_CONTEXT} (KV {kv_mode_for(MODEL_CONTEXT)}):")
        w = max_engine_weights(vram)
        print(f"  engine resident weights must be <= {w:.1f} GiB"
              f"  (currently {a.weights_gib:.1f}) -> {'fits' if w >= a.weights_gib else 'does NOT fit'}")
        return 0

    rows = solve(vram, ram, a.weights_gib, a.base_gib, a.grpo_k)
    print_table(rows, vram, ram, a.weights_gib, a.base_gib)

    print("\nengine KV at the model ceiling (what a 128K step actually holds):")
    for mode in ("fp16", "int8", "k8v4"):
        print(f"  {mode:<5} 128K -> {kv_gib(MODEL_CONTEXT, mode):5.2f} GiB"
              f"   256K -> {kv_gib(2 * MODEL_CONTEXT, mode):5.2f} GiB")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"device": {"vram_gib": vram, "ram_gib": ram},
                       "pack": {"engine_resident_gib": a.weights_gib, "hf_base_gib": a.base_gib},
                       "rows": rows}, f, indent=1)
        print(f"\nwritten: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
