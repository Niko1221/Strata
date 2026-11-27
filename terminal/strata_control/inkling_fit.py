"""Which Inkling-2-Mini build fits a PC, and where its parts go: the one rule setup's menu, `strata launch`, the serve start line and the app's estimate use.

Thinking Machines Inkling-2-Mini (26.4B-A3.4B / 5.4B looped-active, 24 physical layers -> 48 virtual visits, `.strata` v1.1) is offered (0.2.23, owner 2026-11-26) as five builds, in this order everywhere:
  - SQ-20 (19.37 GB / 18.04 GiB): bit-exact 0.00000 KLD lossless native SF4+MX8 LLM (the re-adjusted 16 KiB-layout file Inkling-2-Mini-SQv2-20.strata)
  - SQ-15 (14.35 GB / 13.36 GiB): flagship 16 GiB GPU / 24 GiB Mac / 12 GiB hybrid quality build
  - SQ-10 (10.47 GB / 9.75 GiB): compact 12 GiB 100%-VRAM / 8 GiB GPU / 16 GiB Mac / 780M (gfx1103) & Strix Point (gfx1150) iGPU build
  - GSQ-RCO-IQ3_S (11.43 GB): the GSQ-RCO GGUF of ISTA-DASLab, 3.50 bits per weight
  - UD-Q4_K_XL (17.63 GB): Unsloth's Dynamic GGUF (basic support)
every build with the DFlash 2 drafter (dflash2.safetensors beside the model, 0.55 GB), plus an on-demand lossless BF16 Vision shard (`Inkling-2-Mini-Vision-v2.strata`, 0.64 GB / 0.59 GiB, loaded
when images are enabled/attached). The other nine Unsloth quants of 0.2.22 stay in BUILDS (an install that has one keeps working) but are not offered (LEGACY_ORDER).

Hardware auto-selection (docs/10-GUI-TUI.md + Track 1E):
  - >= 24 GiB discrete GPU OR >= 32 GiB UMA (Mac 32..128 GB, Strix Halo gfx1151, DGX Spark): SQ-20 (100% VRAM / UMA)
  - 16 GiB discrete GPU OR 24 GiB UMA (24 GiB Mac): SQ-15 (100% VRAM / UMA, 0 spill)
  - 12 GiB discrete GPU + >= 24 GiB RAM: SQ-15 in quality mode (default, 3x loop-pinned hybrid) or SQ-10 in speed mode (100% VRAM)
  - 8 GiB discrete GPU OR 16 GiB UMA OR AMD Radeon 780M (gfx1103) / Strix Point 880M/890M (gfx1150) iGPUs: SQ-10

Multi-request parallelism (`--parallel 1|4|8|32|64`):
  - Only 8 of the 48 visits are global KV (8 * 8 kv_heads * 128 dim * 2 * 2 B = 32 KiB/token FP16, 16 KiB/token FP8).
  - The 40 SWA-1024 visits cap at min(seq_len, 1024) tokens (~0.168 GB/slot FP16, ~0.084 GB/slot FP8) plus 3.83 MB ShortConv state/slot.
"""

from __future__ import annotations

GIB_TO_GB = 1.073741824
VRAM_DRIVER_GB = 1.30            # CUDA context + runtime + the engine's working buffers beside the weights, measured: 1.54 GB of non-expert use on a 12 GiB card (was 1.60 with the margin inside; the margin is now VRAM_SAFETY_GB)
VRAM_HYBRID_WORKSPACE_GB = 0.85  # hybrid ping-pong DMA ring + staging buffers + desktop margin (#6587, #6613)
KV_GROW_FACTOR = 1.60            # peak transient headroom for elastic/doubling KV pool growth (HIP grow_pool #6613)
HIP_KV_WINDOW_TOKENS = 32768     # HIP, discrete card: the K/V the plan keeps room for, per slot. The engine's pools start empty and grow on demand (VisitKvPool::grow_pool); when the
                                 # card is full the pool moves to host-mapped pinned memory, so a longer chat is slower, never a failure. Counting the whole context took the card from the experts (#7022)
SPLIT_GPU_OVERHEAD_GB = 1.00     # per-extra-GPU runtime/trunk/comm overhead on multi-GPU layer-split
DSPARK_BASE_GB = 0.11            # DSpark activation + compact SWA-1025 rings for layers 0 & 2
DSPARK_KV_BYTES_PER_TOK = 4096   # 1 global DSpark layer (L1) x 8 kv_heads x 128 head_dim x 2 (K+V) x 2B (FP16)
OS_RESERVE_GB = 4.50             # system RAM reserve on discrete or UMA hosts
# #6644 #6792 #6739 #6662: the plan is made from the memory that is FREE now, and keeps a margin on top of what the engine measures.
# Measured (RTX 3060 12 GB, A4000 16 GB, 4090 24 GB; strata-inkling memory ledger): after the experts, the K/V, the workspaces and the
# drafter are in, 0.4-0.5 GiB of the card was left; the margin makes that about 0.7-0.8 GiB, and a card that drives a screen keeps more.
VRAM_SAFETY_GB = 0.30            # taken off the card's free memory before anything is placed
KV_CARD_HEADROOM_GB = 0.50       # CUDA: the K/V stays in the card only with this much to spare; with less, the program that takes the memory next (browser, video) ends the growth (#6905)
DESKTOP_RESERVE_GB = 0.80        # more, when the card drives a display: the desktop (browser, video) grows after the plan is made
FREE_RAM_MARGIN_GB = 1.50        # free RAM that stays free beside the experts held in RAM
UMA_RESERVE_GB = 5.00            # unified memory OS + driver reserve
SSD_NEAR_RESIDENT_GB = 2.0
VISION_GB = 0.637                # Inkling-2-Mini-Vision-v2.strata (636,551,168 B; 0.2.22's file is 636,526,592 B; CPU / on-demand on <=12 GiB GPUs)
DSPARK_GGUF_GB = 0.636           # PLANNER memory term of a GGUF build's drafter weights, unchanged from 0.2.22 (no DSpark file is downloaded any more); the DFlash 2 reserve (about 0.44 GiB measured on an RTX 5060 Ti) is the DFLASH2 lane's to re-measure
DFLASH2_GB = 0.552              # the DFlash 2 sidecar every build downloads beside the model: dflash2.safetensors (551,600,700 B: Strata's Q8 + SF4 mix; the original is 989,975,616 B)
DEFAULT_CONTEXT = 32768          # docs/10-GUI-TUI.md default: 32K (up to 1M available)
CONTEXTS = [8192, 32768, 65536, 131072, 262144]
MAX_CONTEXT = 1048576
DEFAULT_PARALLEL = 1
PARALLEL_CHOICES = (1, 4, 8, 32, 64)
MAX_PARALLEL = 64

# 8 global visits x 8 kv_heads x 128 head_dim x 2 (K+V) x 2 bytes (FP16) = 32,768 B/token
GLOBAL_KV_BYTES_PER_TOK_FP16 = 32768
GLOBAL_KV_BYTES_PER_TOK_FP8 = 16384
# 40 SWA-1024 visits x 1024 x 8 x 128 x 2 x 2 B = 167,772,160 B (~0.168 GB) + 3.83 MB ShortConv
SWA_AND_SCONV_GB_FP16 = 0.172
SWA_AND_SCONV_GB_FP8 = 0.088
KV_GB_PER_TOKEN = GLOBAL_KV_BYTES_PER_TOK_FP16 / 1e9

FAMILY = "inkling-2-mini"
FAMILY_NAME = "inkling-2-mini"

#: AMD iGPU architectures that auto-select SQ-10 via coherent UMA/GTT allocation (even with small BIOS VRAM carve-out).
IGPU_SQ10_ARCHS = frozenset({"gfx1103", "gfx1150"})
#: High-bandwidth unified-memory GPUs (DGX Spark GB10 sm_121, Strix Halo gfx1151) that scale to SQ-20 on >= 32 GiB UMA.
BIG_UMA_ARCHS = frozenset({"121", "sm_121", "gfx1151"})

#: GGUF builds: bodies_gb is the file's routed-expert bytes (read from the GGUF headers). The loader dequantises every non-expert tensor to BF16, so
#: all ten builds load the same 3.93 GB trunk (the strata-inkling load line of UD-IQ4_XS and UD-Q4_K_XL). dense_gb = 2.45 is what it takes off the
#: card in the plan, calibrated on the experts the engine really placed with the planner's flags:
#: RTX 3060 Ti 7.66 GiB free 2.24 GB (plan 2.35), RTX 3060 11.63 GiB free 6.62 GB (plan 6.62), UD-IQ4_XS and UD-Q4_K_XL alike.
#: name -> exact Section 1 (dense + DSpark) and Section 4 (expert bodies) sizes from the verified .strata v1.1 binaries,
#: plus direct-download Unsloth Dynamic GGUF builds from unsloth/Inkling-2-Mini-GGUF.
BUILDS = {
    "SQ-20": {"file": "Inkling-2-Mini-SQv2-20.strata", "footprint_gb": 19.37, "dense_gb": 3.67, "bodies_gb": 15.68, "file_gb": 19.37, "format": "strata"},
    "SQ-15": {"file": "Inkling-2-Mini-SQ-15.strata", "footprint_gb": 14.35, "dense_gb": 2.07, "bodies_gb": 12.28, "file_gb": 14.35, "format": "strata"},
    "SQ-10": {"file": "Inkling-2-Mini-SQ-10.strata", "footprint_gb": 10.47, "dense_gb": 1.87, "bodies_gb": 8.60, "file_gb": 10.47, "format": "strata"},
    # GSQ-RCO IQ3_S (ISTA-DASLab, 3.50 bpw): bodies_gb / expert_elems_e9 are read from the file's GGUF header at the pinned commit (704 tensors, arch inkling2_loop, routed experts
    # IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S gate_up and Q2_0/IQ4_NL down, no NVFP4 scale tensors; the 0.93 GB BF16 token embedding stays in system memory). dense_gb is the
    # shared GGUF trunk figure (the loader expands every non-expert tensor to BF16, the same 3.93 GB trunk): no engine run has measured it for this file yet.
    "GSQ-RCO-IQ3_S": {"file": "Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf", "footprint_gb": 11.43, "dense_gb": 2.45, "dense_gb_q8": 1.60, "bodies_gb": 9.52, "file_gb": 11.43, "format": "gguf", "expert_elems_e9": {"IQ1_S": 2.038, "IQ2_S": 2.038, "IQ2_XS": 1.359, "IQ2_XXS": 4.077, "IQ3_S": 2.718, "IQ3_XXS": 4.077, "IQ4_NL": 7.135, "Q2_0": 1.019}},
    "UD-Q8_K_XL": {"file": "Inkling-2-Mini-UD-Q8_K_XL.gguf", "footprint_gb": 22.87, "dense_gb": 2.45, "dense_gb_q8": 1.60, "bodies_gb": 19.62, "file_gb": 22.87, "format": "gguf", "expert_elems_e9": {"BF16": 4.077, "NVFP4": 20.384}},
    "UD-Q4_K_XL": {"file": "Inkling-2-Mini-UD-Q4_K_XL.gguf", "footprint_gb": 17.63, "dense_gb": 2.45, "dense_gb_q8": 1.60, "bodies_gb": 15.80, "file_gb": 17.63, "format": "gguf", "expert_elems_e9": {"NVFP4": 20.384, "Q8_0": 4.077}},
    "UD-Q4_K_M": {"file": "Inkling-2-Mini-UD-Q4_K_M.gguf", "footprint_gb": 16.61, "dense_gb": 2.45, "dense_gb_q8": 1.60, "bodies_gb": 14.78, "file_gb": 16.61, "format": "gguf", "expert_elems_e9": {"NVFP4": 20.384, "Q5_K": 2.718, "Q8_0": 1.359}},
    "UD-IQ4_XS": {"file": "Inkling-2-Mini-UD-IQ4_XS.gguf", "footprint_gb": 15.73, "dense_gb": 2.45, "dense_gb_q8": 1.60, "bodies_gb": 14.04, "file_gb": 15.73, "format": "gguf", "expert_elems_e9": {"IQ4_NL": 0.340, "IQ4_XS": 0.679, "NVFP4": 20.384, "Q4_K": 0.679, "Q5_0": 1.019, "Q5_K": 1.359}},
    "UD-Q3_K_XL": {"file": "Inkling-2-Mini-UD-Q3_K_XL.gguf", "footprint_gb": 13.63, "dense_gb": 2.45, "dense_gb_q8": 1.60, "bodies_gb": 12.20, "file_gb": 13.63, "format": "gguf", "expert_elems_e9": {"IQ2_S": 2.038, "IQ2_XS": 0.679, "IQ3_S": 1.359, "IQ3_XXS": 3.397, "IQ4_NL": 1.359, "NVFP4": 14.948, "Q3_K": 0.679}},
    "UD-IQ3_XXS": {"file": "Inkling-2-Mini-UD-IQ3_XXS.gguf", "footprint_gb": 11.66, "dense_gb": 2.45, "dense_gb_q8": 1.60, "bodies_gb": 10.30, "file_gb": 11.66, "format": "gguf", "expert_elems_e9": {"IQ2_S": 6.115, "IQ2_XS": 0.679, "IQ2_XXS": 0.679, "IQ3_XXS": 8.833, "IQ4_NL": 1.359, "NVFP4": 6.795}},
    "UD-Q2_K_XL": {"file": "Inkling-2-Mini-UD-Q2_K_XL.gguf", "footprint_gb": 9.14, "dense_gb": 2.45, "dense_gb_q8": 1.60, "bodies_gb": 7.89, "file_gb": 9.14, "format": "gguf", "expert_elems_e9": {"IQ1_S": 4.077, "IQ2_S": 7.474, "IQ2_XS": 1.359, "IQ2_XXS": 3.397, "MXFP4": 0.340, "NVFP4": 3.737, "Q2_0": 4.077}},
    "UD-IQ2_XXS": {"file": "Inkling-2-Mini-UD-IQ2_XXS.gguf", "footprint_gb": 8.49, "dense_gb": 2.45, "dense_gb_q8": 1.60, "bodies_gb": 7.25, "file_gb": 8.49, "format": "gguf", "expert_elems_e9": {"IQ1_S": 4.077, "IQ2_S": 3.397, "IQ2_XS": 4.077, "IQ2_XXS": 4.756, "NVFP4": 2.378, "Q2_0": 5.776}},
    "UD-IQ1_M": {"file": "Inkling-2-Mini-UD-IQ1_M.gguf", "footprint_gb": 8.02, "dense_gb": 2.45, "dense_gb_q8": 1.60, "bodies_gb": 6.81, "file_gb": 8.02, "format": "gguf", "expert_elems_e9": {"IQ1_S": 4.077, "IQ2_S": 2.718, "IQ2_XS": 3.397, "IQ2_XXS": 6.115, "NVFP4": 1.019, "Q2_0": 7.135}},
    "UD-IQ1_S": {"file": "Inkling-2-Mini-UD-IQ1_S.gguf", "footprint_gb": 7.76, "dense_gb": 2.45, "dense_gb_q8": 1.60, "bodies_gb": 6.53, "file_gb": 7.76, "format": "gguf", "expert_elems_e9": {"IQ1_S": 4.077, "IQ2_S": 1.359, "IQ2_XS": 3.397, "IQ2_XXS": 7.474, "NVFP4": 0.340, "Q2_0": 7.814}},
}
STRATA_ORDER = ("SQ-20", "SQ-15", "SQ-10")       # primary recommended native .strata builds, best quality first
#: The 0.2.23 lineup (owner decision 2026-11-26): the three Strata packs, then GSQ-RCO IQ3_S (ISTA-DASLab), then Unsloth's UD-Q4_K_XL. This is the order of every list: setup,
#: the launcher, the app. Nothing else is offered.
OFFERED_GGUF_ORDER = ("GSQ-RCO-IQ3_S", "UD-Q4_K_XL")
#: The other nine Unsloth quants that 0.2.22 offered. They stay in BUILDS (an install that already has one keeps running it: `base support`) but are never offered, listed for
#: download, or recommended; the lists show one only when it is installed, as "no longer offered".
LEGACY_UNSLOTH_ORDER = (
    "UD-Q8_K_XL", "UD-Q4_K_M", "UD-IQ4_XS", "UD-Q3_K_XL",
    "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-IQ2_XXS", "UD-IQ1_M", "UD-IQ1_S",
)
GGUF_ORDER = (*OFFERED_GGUF_ORDER, *LEGACY_UNSLOTH_ORDER)       # every GGUF build the engine reads, the offered ones first
ORDER = (*STRATA_ORDER, *OFFERED_GGUF_ORDER)                      # what is offered, in display order
LEGACY_ORDER = LEGACY_UNSLOTH_ORDER


def is_gguf_build(build: str | None) -> bool:
    return BUILDS.get(str(build or ""), {}).get("format") == "gguf"


def build_names() -> list[str]:
    return list(BUILDS)


def is_igpu_arch(arch: str | None) -> bool:
    return str(arch or "").strip().lower() in IGPU_SQ10_ARCHS


def is_big_uma_arch(arch: str | None, name: str | None = None) -> bool:
    a = str(arch or "").strip().lower()
    if a in BIG_UMA_ARCHS:
        return True
    low = str(name or "").lower()
    return "gb10" in low or "dgx spark" in low or "strix halo" in low


# --parallel N: what a slot costs on each backend (read from the engines' allocation code; CUDA and SYCL numbers also from runs).
#  - CUDA (ink_runner.cuh init): the on-device global K/V is capped at 4096 tokens per slot from 2 slots on (the rest of the context goes to the
#    mapped host K/V), the SWA ring is 1056 positions from 16 slots, the workspaces grow from 768 to 1280 MiB from 8 slots.
#  - HIP: paged on-demand K/V pools sized by estimate_slot_bytes(min(ctx, 4096)) x slots (ink_batch.h): the same 4096-token active window.
#  - SYCL (ink_sy_runner.h init): every slot gets its K/V up front and in FP16: slots x max_ctx x 32 KiB for the 8 global visits (no FP8, no
#    paging, no host K/V), the drafter K/V slots x max_ctx x 12 KiB. A request of 32 slots at 32K tokens is 34 GB of K/V (Arc A750 8 GB: dalloc
#    256 MiB, out of device memory, 0 of 3072 experts in VRAM).
SLOT_WORKSPACE_EXTRA_GB = 0.54          # the extra 512 MiB of workspaces CUDA reserves from 8 slots on (applied to every backend: not measured elsewhere)
DSPARK_KV_BYTES_PER_TOK_SYCL = 12288    # 3 drafter layers x K+V x 1024 x FP16, per slot and token (ink_sy_runner.h dsp_ctx_k/v)
SLOT_BACKENDS_FULL_CTX = ("sycl",)      # backends that allocate every slot's K/V at the full context, in FP16


def _full_ctx_slots(backend: str | None) -> bool:
    return str(backend or "").strip().lower() in SLOT_BACKENDS_FULL_CTX


def dspark_gb(ctx: int | None = None, enabled: bool = True, parallel: int = 1, backend: str | None = None) -> float:
    """DSpark drafter KV + scratch footprint in GB for `ctx` tokens (0 when drafting is disabled)."""
    if not enabled:
        return 0.0
    n = DEFAULT_CONTEXT if ctx is None else max(1024, int(ctx))
    if _full_ctx_slots(backend):
        slots = max(1, min(MAX_PARALLEL, int(parallel or 1)))
        return round(DSPARK_BASE_GB + slots * n * DSPARK_KV_BYTES_PER_TOK_SYCL / 1e9, 3)
    return round(DSPARK_BASE_GB + n * DSPARK_KV_BYTES_PER_TOK / 1e9, 3)


def kv_cache_gb(ctx: int | None = None, parallel: int = 1, fp8_kv: bool = False, backend: str | None = None) -> float:
    """KV + SWA-1024 + ShortConv state in GB for `ctx` context tokens across `parallel` concurrent request slots."""
    n = DEFAULT_CONTEXT if ctx is None else max(1024, int(ctx))
    slots = max(1, min(MAX_PARALLEL, int(parallel or 1)))
    if _full_ctx_slots(backend):
        return round(slots * (n * GLOBAL_KV_BYTES_PER_TOK_FP16 / 1e9 + SWA_AND_SCONV_GB_FP16), 3)
    if fp8_kv or slots >= 16:
        per_tok = GLOBAL_KV_BYTES_PER_TOK_FP8 / 1e9
        fixed = SWA_AND_SCONV_GB_FP8
        # Continuous batching allocates paged SWA blocks & global KV on demand per slot; budget realistic active occupancy
        active_ctx = min(n, 4096) if slots >= 16 else n
    else:
        per_tok = GLOBAL_KV_BYTES_PER_TOK_FP16 / 1e9
        fixed = SWA_AND_SCONV_GB_FP16
        active_ctx = n
    return round(slots * (active_ctx * per_tok + fixed), 3)


def auto_fp8_kv(vram_gib: float | None, ctx: int | None = None, parallel: int = 1, kv_offload: str = "auto",
                backend: str | None = None) -> bool:
    """Whether FP8 KV is auto-selected. With Host KV Offload ('auto'/'on'), lossless FP16 stays default (#552)
    except at high concurrency (parallel >= 16) where 40 SWA rings dominate VRAM."""
    slots = max(1, min(MAX_PARALLEL, int(parallel or 1)))
    if _full_ctx_slots(backend):
        return False                    # the SYCL runner has no FP8 K/V
    if slots >= 16:
        return True
    if str(kv_offload or "auto").lower() == "off":
        v = float(vram_gib or 0.0)
        return 0.0 < v <= 13.0
    return False


def global_kv_gb(ctx: int | None = None, parallel: int = 1, fp8_kv: bool = False, backend: str | None = None) -> float:
    """Global KV cache size in GB across the 8 global visits (spillable to mapped pinned host RAM via --kv-offload)."""
    n = DEFAULT_CONTEXT if ctx is None else max(1024, int(ctx))
    slots = max(1, min(MAX_PARALLEL, int(parallel or 1)))
    if _full_ctx_slots(backend):
        return round(slots * n * GLOBAL_KV_BYTES_PER_TOK_FP16 / 1e9, 3)
    if fp8_kv or slots >= 16:
        per_tok = GLOBAL_KV_BYTES_PER_TOK_FP8 / 1e9
        active_ctx = min(n, 4096) if slots >= 16 else n
    else:
        per_tok = GLOBAL_KV_BYTES_PER_TOK_FP16 / 1e9
        active_ctx = n
    return round(slots * active_ctx * per_tok, 3)


def global_kv_total_gb(ctx: int | None = None, parallel: int = 1, fp8_kv: bool = False) -> float:
    """All the global K/V of `parallel` slots at `ctx` tokens: what the mapped host K/V holds when the engine offloads it. Measured (strata-inkling, RTX 3060,
    SQ-10, ctx 32768): the host share is slots x (ctx - the on-device window) x bytes per token, 3.4 GiB at 4 slots (FP16) ... 31.7 GiB at 64 slots (FP8)."""
    n = DEFAULT_CONTEXT if ctx is None else max(1024, int(ctx))
    slots = max(1, min(MAX_PARALLEL, int(parallel or 1)))
    per_tok = (GLOBAL_KV_BYTES_PER_TOK_FP8 if (fp8_kv or slots >= 16) else GLOBAL_KV_BYTES_PER_TOK_FP16) / 1e9
    return round(slots * n * per_tok, 3)


def swa_sconv_gb(parallel: int = 1, fp8_kv: bool = False, backend: str | None = None) -> float:
    """VRAM-resident 40x SWA-1024 + ShortConv state in GB."""
    slots = max(1, min(MAX_PARALLEL, int(parallel or 1)))
    fixed = SWA_AND_SCONV_GB_FP16 if _full_ctx_slots(backend) else (SWA_AND_SCONV_GB_FP8 if (fp8_kv or slots >= 16) else SWA_AND_SCONV_GB_FP16)
    return round(slots * fixed, 3)


def non_expert_gb(ctx: int | None = None, parallel: int = 1, vision: bool = False, *,
                  dspark: bool = True, fp8_kv: bool = False, hybrid: bool = False,
                  host_offload_global_kv: bool = False, backend: str | None = None) -> float:
    """Video memory kept free of expert bodies: driver/desktop + hybrid workspace + DSpark KV + KV/SWA/ShortConv (+ peak KV growth margin + Vision shard)."""
    slots = max(1, min(MAX_PARALLEL, int(parallel or 1)))
    kv_ctx = ctx
    if str(backend or "").strip().lower() == "hip":      # HIP (a card of its own): room for the K/V window; the pool grows on demand and spills to the host when the card is full
        kv_ctx = min(DEFAULT_CONTEXT if ctx is None else max(1024, int(ctx)), HIP_KV_WINDOW_TOKENS)
    if host_offload_global_kv:
        kv_vram = swa_sconv_gb(parallel, fp8_kv=fp8_kv, backend=backend)
    else:
        grow = 1.0 if _full_ctx_slots(backend) else KV_GROW_FACTOR       # SYCL allocates the whole K/V once: there is no growth to leave room for
        kv_vram = round(global_kv_gb(kv_ctx, parallel, fp8_kv=fp8_kv, backend=backend) * grow
                        + swa_sconv_gb(parallel, fp8_kv=fp8_kv, backend=backend), 3)
    return round(
        VRAM_DRIVER_GB
        + (VRAM_HYBRID_WORKSPACE_GB if hybrid else 0.0)
        + (SLOT_WORKSPACE_EXTRA_GB if slots >= 8 else 0.0)
        + dspark_gb(kv_ctx, enabled=dspark, parallel=parallel, backend=backend)
        + kv_vram
        + (VISION_GB if vision else 0.0),
        3,
    )


# Backends that cannot run every GGUF expert type natively transcode the GGUF_RAW experts at load (ink_gguf_transcode.h, choose_sk_code_for_ggml):
# types of 4 bits per weight or less (Q2_K ... IQ1_S, IQ2_*, IQ3_*, IQ4_*, MXFP4, Q2_0, Q4_*) become S4_K, Q5_* / Q6_K become S6_K, Q8_* / F16 / BF16
# become S8_K; NVFP4 stays a bit-exact SF4 copy. The converted experts take their S_K size in memory, not their file size: an IQ1_S expert grows 2.7x
# (1.56 -> 4.19 bits per weight), so a plan that counts the file size leaves part of the experts with no place (#6729). CUDA keeps GGUF_RAW as it is.
TRANSCODE_BACKENDS = ("hip", "sycl", "metal")
SK_BYTES_PER_256 = {"S4_K": 134, "S6_K": 198, "S8_K": 262}            # strata/quant/sk_codecs.hpp superblock_bytes
SK_CODE_OF_TYPE = {"S8_K": ("F32", "F16", "Q8_0", "Q8_1", "Q8_K", "BF16"), "S6_K": ("Q5_0", "Q5_1", "Q5_K", "Q6_K")}   # everything else: S4_K
GGUF_TYPE_ELEMS_BYTES = {                                              # ggml type -> (elements per block, bytes per block)
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24), "Q8_0": (32, 34),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
    "IQ1_S": (256, 50), "IQ1_M": (256, 56), "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ2_S": (256, 82), "IQ3_XXS": (256, 98),
    "IQ3_S": (256, 110), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136), "MXFP4": (32, 17), "NVFP4": (64, 36), "Q2_0": (64, 18),
}
EXPERT_BODY_OVERHEAD_BYTES = 2304 + 32          # sq_xd_dn[inter] FP16 + 64-byte alignment padding, per transcoded expert
N_EXPERTS_TOTAL = 3072


# dense_gb (2.45) is the share when the loader expands the Q8_0 matrices to BF16; dense_gb_q8 (1.60) when it keeps them as the file has them (INKSPEED's
# c60259fa6, load line trunk= 3.93 -> 2.9 GB on an sm_86 card). Both are what the engine really placed with no cap (--vram-experts-gb 50):
#   BF16: RTX 3060 Ti 7.66 GiB free 2.24 GB, RTX 3060 11.63 GiB free 6.49 GB;  Q8_0: 3060 Ti 3.19 GB (703 / 721 experts), 3060 7.49 GB (1663 / 1681).
# A GGUF build's dense share of the card depends on how the loader keeps the file's Q8_0 matrices: on ONE CUDA card of compute capability 8.0 or
# newer, without a layer split, they stay Q8_0 (int8 tensor cores; the embedding too); everywhere else they are expanded to BF16 (c60259fa6,
# STRATA_INK_Q8_0_DENSE=0 forces the BF16 way). The card's compute capability is not always known: unknown means the BF16 way (the larger share).
# Only consumer Ampere (sm 86) keeps them: the int8 dense GEMM is slower than BF16 at one token on an RTX 4090 (sm 89: 95.4 against 103.7 tok/s) and on an
# RTX 5090 (sm 120: 98 against 133), and sm 80 is unmeasured (ink_cuda_q8f_supported in the engine returns the same set).
Q8_DENSE_SMS = (86,)


def dense_gb_for(build: str, backend: str | None = None, sm: int | None = None, split: bool = False) -> float:
    info = BUILDS[build]
    if (info.get("format") == "gguf" and "dense_gb_q8" in info and str(backend or "").strip().lower() == "cuda"
            and sm is not None and int(sm) in Q8_DENSE_SMS and not split):
        return float(info["dense_gb_q8"])
    return float(info["dense_gb"])


def converted_expert_bytes_per_element(type_name: str) -> float:
    """Bytes per weight an expert tensor of this GGUF type takes in memory after the load-time conversion (NVFP4: its own, bit-exact SF4)."""
    if type_name == "NVFP4":
        el, by = GGUF_TYPE_ELEMS_BYTES["NVFP4"]
        return by / el
    for code, types in SK_CODE_OF_TYPE.items():
        if type_name in types:
            return SK_BYTES_PER_256[code] / 256.0
    return SK_BYTES_PER_256["S4_K"] / 256.0


def bodies_gb_for(build: str, backend: str | None = None) -> float:
    """GB the routed experts of `build` take on `backend`: the file's size, or, where the backend converts GGUF experts at load, their converted size."""
    info = BUILDS[build]
    native = float(info["bodies_gb"])
    elems = info.get("expert_elems_e9")
    if info.get("format") != "gguf" or not elems or str(backend or "").strip().lower() not in TRANSCODE_BACKENDS:
        return native
    src = sum(e9 * 1e9 * GGUF_TYPE_ELEMS_BYTES[t][1] / GGUF_TYPE_ELEMS_BYTES[t][0] for t, e9 in elems.items())
    dst = sum(e9 * 1e9 * converted_expert_bytes_per_element(t) for t, e9 in elems.items()) + N_EXPERTS_TOTAL * EXPERT_BODY_OVERHEAD_BYTES
    return round(native * dst / src, 2)


def compute(build: str, vram_gib: float, ram_gib: float, ctx: int | None = None, vision: bool = False,
            parallel: int = 1, dspark: bool = True, kv_offload: str = "auto", *,
            extra_vram_gib: float = 0.0, backend: str | None = None,
            desktop: bool = False, ram_free_gib: float | None = None, fp8_kv: bool | None = None, sm: int | None = None) -> dict:
    """Placement of `build` on a discrete-GPU PC (vram_gib + optional extra_vram_gib + ram_gib): GB of experts in VRAM, in RAM, and on SSD.

    Implements the unified 4-tier adaptive contract (GPU -> RAM+CPU -> SSD) with Host KV Offloading (#552),
    CPU/on-demand Vision on <= 12.5 GiB GPUs, O(1) prefill & KV growth headroom (#6613), and WDDM floor so
    4/6/8/10/12/16/24/32+ GB GPUs never OOM even at 32K–128K context with DSpark and Vision enabled.

    vram_gib is the card's FREE memory (what is left after the desktop and other programs), ram_free_gib the free RAM
    (None: not known, the total in ram_gib counts); desktop says the card drives a display (a larger reserve) (#6662 #6739).
    """
    info = dict(BUILDS[build])
    info["bodies_gb"] = bodies_gb_for(build, backend)      # GGUF experts a backend converts at load take their converted size
    info["dense_gb"] = dense_gb_for(build, backend, sm, split=bool(extra_vram_gib and extra_vram_gib > 0))
    v_primary = float(vram_gib or 0.0)
    v_extra = max(0.0, float(extra_vram_gib or 0.0))
    v_gib = v_primary + v_extra
    split_overhead = SPLIT_GPU_OVERHEAD_GB if v_extra > 0.0 else 0.0
    vram_gb = max(0.0, v_gib * GIB_TO_GB - split_overhead - (VRAM_SAFETY_GB if v_gib > 0.0 else 0.0)
                  - (DESKTOP_RESERVE_GB if (desktop and v_gib > 0.0) else 0.0))
    ram_gb = float(ram_gib or 0.0) * GIB_TO_GB
    offload_mode = str(kv_offload or "auto").lower()
    backend_low = str(backend or "").strip().lower()
    # Only CUDA implements mapped pinned-host global KV offload (and auto-enables it on <= 13 GiB cards);
    # HIP and SYCL keep global KV in VRAM (#6613).
    supports_host_kv = backend_low not in ("hip", "sycl")
    effective_offload = offload_mode if supports_host_kv else "off"
    if fp8_kv is None:      # None: the planner's rule (FP16 unless the card is small or the slots many); True / False: the caller's choice
        fp8_kv = auto_fp8_kv(v_gib, ctx, parallel, kv_offload=effective_offload, backend=backend_low)
    elif _full_ctx_slots(backend_low):
        fp8_kv = False
    kv_dtype = "fp8" if fp8_kv else "fp16"
    # The Q8_0 drafter file (0.40 GB) is NOT a further charge on a discrete card: measured on an RTX 3060 Ti 8 GB (SQ-10 and UD-Q4_K_XL, free
    # 7.66 GiB, ctx 32768) the engine places only 81 / 41 fewer experts (0.23 / 0.19 GB) with the drafter on, and its ledger shows DSpark taking 78-80 MiB;
    # the planner charged 0.25 GB for SQ and 0.65 GB for GGUF (0.40 of it for the weights), 0.45 GB too much on a tight card. Unified memory still holds it.
    dense_total = info["dense_gb"]
    # Vision stays in host RAM / CPU on <= 12.5 GiB GPUs so it never steals ~0.64 GB of hot 3x-looped expert VRAM!
    vision_in_vram = bool(vision and v_gib > 12.5)
    # the K/V that goes to the host (CUDA only): every slot's whole context, not the on-device window the card keeps (#--parallel measured, see global_kv_total_gb)
    glob_kv = global_kv_total_gb(ctx, parallel, fp8_kv=fp8_kv) if supports_host_kv else global_kv_gb(ctx, parallel, fp8_kv=fp8_kv, backend=backend_low)
    ram_avail_for_model = max(0.0, ram_gb - OS_RESERVE_GB - (VISION_GB if vision else 0.0))
    if ram_free_gib is not None:        # the free RAM, not the installed RAM: what other programs hold is not there to give
        ram_avail_for_model = max(0.0, min(ram_avail_for_model,
                                           float(ram_free_gib) * GIB_TO_GB - FREE_RAM_MARGIN_GB - (VISION_GB if vision else 0.0)))

    # 1. Try 100%-VRAM expert residency with VRAM-resident global KV first
    full_need = dense_total + info["bodies_gb"] + non_expert_gb(
        ctx, parallel, vision=vision_in_vram, dspark=dspark, fp8_kv=fp8_kv, hybrid=False, host_offload_global_kv=False, backend=backend_low
    )
    # CUDA: a K/V that fills the card to within a few hundred MB grows into whatever the desktop takes next; with that little to spare the K/V goes to host RAM
    # (the experts stay all in the card, the engine keeps the window of every slot there) when the RAM can hold it (#6905: a 16 GB card, 64K, vision, 0.4 GB to spare)
    kv_headroom = KV_CARD_HEADROOM_GB if (backend_low == "cuda" and effective_offload == "auto" and glob_kv > 1.0) else 0.0
    if vram_gb > 0.0 and effective_offload != "on" and full_need + kv_headroom <= vram_gb:
        return {
            "vram_gb": round(info["bodies_gb"], 2),
            "ram_gb": 0.0,
            "ssd_gb": 0.0,
            "host_kv_gb": 0.0,
            "kv_dtype": kv_dtype,
            "kv_offload": "off",
            "full_vram": True,
        }

    # 2. Try 100%-VRAM expert residency with Host KV Offloading (matches CUDA engine auto-offload on <= 13 GiB,
    # e.g. 12 GiB RTX 5070 + SQ-10 + lossless FP16 32K; 16 GiB cards keep global KV in VRAM so HIP never OOMs #6613).
    can_offload_kv = supports_host_kv and (effective_offload != "off") and (ram_avail_for_model >= glob_kv)
    # CUDA (named by the caller) at any card size: the experts all in the card with the K/V in RAM beats experts in RAM
    # (measured: SQ-15 on an A4000 16 GB, RESULTS-0221-INKCUDA.md); an unnamed backend keeps the older rule (<= 13 GiB or asked)
    auto_offload_full = can_offload_kv and (effective_offload == "on" or v_gib <= 13.0 or backend_low == "cuda")
    full_need_offload = dense_total + info["bodies_gb"] + non_expert_gb(
        ctx, parallel, vision=vision_in_vram, dspark=dspark, fp8_kv=fp8_kv, hybrid=False, host_offload_global_kv=True, backend=backend_low
    )
    if vram_gb > 0.0 and auto_offload_full and full_need_offload <= vram_gb:
        return {
            "vram_gb": round(info["bodies_gb"], 2),
            "ram_gb": 0.0,
            "ssd_gb": 0.0,
            "host_kv_gb": round(glob_kv, 2),
            "kv_dtype": kv_dtype,
            "kv_offload": "on",
            "full_vram": True,
        }

    # 3. Hybrid 3/4-tier placement (VRAM hot 3x-looped core -> Host RAM -> Auto-SSD)
    use_host_kv = can_offload_kv and (effective_offload == "on" or v_gib <= 13.0 or glob_kv > 1.5)
    host_kv_used = glob_kv if use_host_kv else 0.0
    gpu_room = max(0.0, vram_gb - dense_total - non_expert_gb(
        ctx, parallel, vision=vision_in_vram, dspark=dspark, fp8_kv=fp8_kv, hybrid=True, host_offload_global_kv=use_host_kv, backend=backend_low
    ))
    ram_room = max(0.0, ram_avail_for_model - host_kv_used)
    in_vram = min(info["bodies_gb"], gpu_room)
    in_ram = min(info["bodies_gb"] - in_vram, ram_room)
    on_ssd = max(0.0, info["bodies_gb"] - in_vram - in_ram)
    # A K/V that stays in the card (no room for it in RAM) must fit the card beside the dense weights even with every expert in RAM;
    # a context whose K/V fits neither is not startable, whatever the experts do.
    kv_fits = True
    if vram_gb > 0.0:
        # What the slots add to what one slot already needs (the start-up of ONE slot on a tiny card is measured: SQ-10 starts on 4 GiB with every
        # expert in RAM, the engine shrinks its workspaces): the per-slot SWA rings / ShortConv state / drafter K/V / workspaces of N slots,
        # and the global K/V when it is not moved to the host. That must fit the card beside the dense weights and the one-slot base.
        one = dict(vision=vision_in_vram, dspark=dspark, fp8_kv=fp8_kv, hybrid=True, host_offload_global_kv=True, backend=backend_low)
        base = non_expert_gb(ctx, 1, **one)
        slot_extra = max(0.0, non_expert_gb(ctx, parallel, **one) - base)
        kv_in_card = 0.0
        if not use_host_kv:
            kv_in_card = non_expert_gb(ctx, parallel, **{**one, "host_offload_global_kv": False}) - non_expert_gb(ctx, parallel, **one)
        room = vram_gb - dense_total - base
        kv_fits = (room - slot_extra - kv_in_card) >= 0.0 if (not use_host_kv or slot_extra > 0.0) else True
    return {
        "vram_gb": round(in_vram, 2),
        "ram_gb": round(in_ram, 2),
        "ssd_gb": round(on_ssd, 2),
        "host_kv_gb": round(host_kv_used, 2),
        "kv_dtype": kv_dtype,
        "kv_offload": "on" if use_host_kv else "off",
        "full_vram": False,
        "kv_fits": kv_fits,
    }


def compute_unified(build: str, pool_gib: float, ctx: int | None = None, vision: bool = False,
                    parallel: int = 1, dspark: bool = True, backend: str | None = None) -> dict:
    """Placement on a unified-memory / coherent-UMA host (Apple Silicon, Strix Halo, Strix Point 890M, Radeon 780M, DGX Spark)."""
    info = BUILDS[build]
    extra_dspark = DSPARK_GGUF_GB if (dspark and info.get("format") == "gguf") else 0.0
    need_total = info["dense_gb"] + bodies_gb_for(build, backend) + extra_dspark
    fp8_kv = auto_fp8_kv(None, ctx, parallel, backend=backend)
    room = max(
        0.0,
        float(pool_gib or 0.0) * GIB_TO_GB
        - UMA_RESERVE_GB
        - dspark_gb(ctx, enabled=dspark, parallel=parallel, backend=backend)
        - kv_cache_gb(ctx, parallel, fp8_kv=fp8_kv, backend=backend)
        - (VISION_GB if vision else 0.0),
    )
    held = min(need_total, room)
    on_ssd = max(0.0, need_total - held)
    return {
        "vram_gb": round(held, 2),
        "ram_gb": 0.0,
        "ssd_gb": round(on_ssd, 2),
        "host_kv_gb": 0.0,
        "kv_dtype": "fp8" if fp8_kv else "fp16",
        "kv_offload": "off",
        "full_vram": on_ssd <= 0.05,
    }


def place(build: str, vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None,
          vision: bool = False, parallel: int = 1, dspark: bool = True, *,
          extra_vram_gib: float = 0.0, backend: str | None = None,
          desktop: bool = False, ram_free_gib: float | None = None, fp8_kv: bool | None = None, sm: int | None = None) -> dict:
    if unified:
        return compute_unified(build, ram_gib, ctx, vision, parallel, dspark, backend=backend)
    return compute(build, vram_gib or 0.0, ram_gib, ctx, vision, parallel, dspark,
                   extra_vram_gib=extra_vram_gib, backend=backend, desktop=desktop, ram_free_gib=ram_free_gib, fp8_kv=fp8_kv, sm=sm)


def mode(placed: dict | float) -> str:
    """Placement verdict: full_vram (100% GPU/UMA), hybrid_ram_vram (zero SSD spill), near_resident_ssd, or cross_tier_ssd."""
    if isinstance(placed, dict):
        ssd_gb = float(placed.get("ssd_gb", 0.0))
        if ssd_gb <= 0.05:
            return "full_vram" if placed.get("full_vram") else "ram_vram_zero_ssd"
        if ssd_gb <= SSD_NEAR_RESIDENT_GB:
            return "near_resident_ssd"
        return "cross_tier_ssd"
    ssd_gb = float(placed)
    if ssd_gb <= 0.05:
        return "ram_vram_zero_ssd"
    if ssd_gb <= SSD_NEAR_RESIDENT_GB:
        return "near_resident_ssd"
    return "cross_tier_ssd"


def fits(build: str, vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None,
         vision: bool = False, parallel: int = 1, *, extra_vram_gib: float = 0.0, backend: str | None = None,
         desktop: bool = False, ram_free_gib: float | None = None, sm: int | None = None) -> bool:
    return place(build, vram_gib, ram_gib, unified, ctx, vision, parallel, extra_vram_gib=extra_vram_gib, backend=backend,
                 desktop=desktop, ram_free_gib=ram_free_gib, sm=sm)["ssd_gb"] <= 0.05


def recommend(vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None,
              vision: bool = False, *, preference: str = "quality", igpu: bool = False, arch: str | None = None,
              parallel: int = 1, extra_vram_gib: float = 0.0, backend: str | None = None,
              desktop: bool = False, ram_free_gib: float | None = None, sm: int | None = None) -> dict:
    """Automatic hardware auto-selection for Inkling-2-Mini (docs/10-GUI-TUI.md + Track 1E):
      - Radeon 780M (gfx1103) / Strix Point (gfx1150) iGPUs -> SQ-10 (coherent UMA/GTT, zero BIOS carve-out OOM)
      - >= 24 GiB discrete GPU (or combined multi-GPU VRAM) OR >= 32 GiB UMA -> SQ-20
      - 16 GiB discrete GPU (>= 15 GiB): SQ-15 in quality mode (with >= 20 GiB RAM) or SQ-10 in speed mode / < 20 GiB RAM
      - 24 GiB UMA (>= 22 GiB) -> SQ-15
      - 12 GiB discrete GPU (11..15 GiB): SQ-15 in quality mode (with >= 24 GiB RAM) or SQ-10 in speed mode / < 24 GiB RAM
      - 8 GiB discrete GPU (< 11 GiB) OR <= 16 GiB UMA -> SQ-10
    """
    v = float(vram_gib or 0.0) + max(0.0, float(extra_vram_gib or 0.0))
    r = float(ram_gib or 0.0)
    is_small_igpu = bool(igpu or is_igpu_arch(arch))
    if is_small_igpu:
        best = "SQ-10"
        use_unified = True
    elif unified:
        use_unified = True
        if r >= 30.0:
            best = "SQ-20"
        elif r >= 22.0:
            best = "SQ-15"
        else:
            best = "SQ-10"
    else:
        use_unified = False
        pref = str(preference).lower()
        if pref == "quality":
            # the biggest build that fits with part of its experts in RAM (slower: measured 4-8 tok/s for SQ-15 on a 12 GiB card)
            if v >= 22.0:
                best = "SQ-20"
            elif v >= 15.0:
                best = "SQ-10" if r < 20.0 else "SQ-15"
            elif v >= 11.0:
                best = "SQ-10" if r < 24.0 else "SQ-15"
            else:
                best = "SQ-10"
        else:
            # balanced (or "speed"; NOT the default: the default stays "quality", the owner prefers the bigger builds and speed on small cards
            # is for the engine to fix): the best build whose experts all sit in the card (the K/V may be in RAM), else SQ-10.
            # Experts moved to RAM cost most of the speed today (SQ-10 on a 3060: about 35 tok/s resident, 10 tok/s with 2.6 GB in RAM).
            best = "SQ-10"
            for cand in STRATA_ORDER:
                pc = place(cand, vram_gib, ram_gib, False, ctx, vision, parallel, extra_vram_gib=extra_vram_gib, backend=backend,
                           desktop=desktop, ram_free_gib=ram_free_gib, sm=sm)
                if pc.get("full_vram") and pc["ssd_gb"] <= 0.05:
                    best = cand
                    break
    placed = place(best, vram_gib, ram_gib, use_unified, ctx, vision, parallel,
                   extra_vram_gib=extra_vram_gib, backend=backend, desktop=desktop, ram_free_gib=ram_free_gib, sm=sm)
    return {
        "build": best,
        "placement": placed,
        "mode": mode(placed),
        "fits": placed["ssd_gb"] <= 0.05,
        "unified": use_unified,
    }


def rows(vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None,
         vision: bool = False, parallel: int = 1, *, desktop: bool = False, ram_free_gib: float | None = None,
         backend: str | None = None, sm: int | None = None) -> list[dict]:
    out = []
    for b in ORDER:
        placed = place(b, vram_gib, ram_gib, unified, ctx, vision, parallel, desktop=desktop, ram_free_gib=ram_free_gib, backend=backend, sm=sm)
        dl = BUILDS[b]["file_gb"] + DFLASH2_GB
        out.append({
            "build": b,
            "file": BUILDS[b]["file"],
            "format": BUILDS[b].get("format", "strata"),
            "download_gb": round(dl, 2),
            "footprint_gb": BUILDS[b]["footprint_gb"],
            "placement": placed,
            "mode": mode(placed),
        })
    return out


def max_context(build: str, vram_gib: float | None, ram_gib: float, unified: bool = False, *, vision: bool = False,
                parallel: int = 1, dspark: bool = True, backend: str | None = None, desktop: bool = False,
                ram_free_gib: float | None = None, sm: int | None = None) -> int:
    """The largest context (tokens) at which the build still starts with no expert on SSD, the experts that do not fit the card going to RAM
    (never refused, never a smaller build): 0 when even the smallest context needs the SSD.  The K/V of a longer context goes to RAM
    first (Host KV Offload), then the experts' share of the card shrinks."""
    best = 0
    for c in sorted({*CONTEXTS, 393216, 524288, MAX_CONTEXT}):
        placed = place(build, vram_gib, ram_gib, unified, c, vision, parallel, dspark, backend=backend, desktop=desktop,
                       ram_free_gib=ram_free_gib, sm=sm)
        if placed["ssd_gb"] > 0.05 or not placed.get("kv_fits", True):
            break
        best = c
    return best


def disk_gb(build: str, vision: bool = False, dspark: bool = True) -> float:
    """The download of a build: the model file, the DFlash 2 sidecar (`dspark` is the old name of the drafter switch) and the Vision shard when pictures are on."""
    extra_drafter = DFLASH2_GB if dspark else 0.0
    return round(BUILDS[build]["file_gb"] + extra_drafter + (VISION_GB if vision else 0.0), 2)


def build_of_model_name(model_name: str | None) -> str | None:
    low = str(model_name or "").lower()
    if not low.startswith(FAMILY_NAME + "-"):
        return None
    size = low[len(FAMILY_NAME) + 1:].upper()
    return size if size in BUILDS else None


def ssd_mode_flags(build: str, vram_gib: float | None, ram_gib: float, unified: bool = False,
                   ctx: int | None = None, vision: bool = False, parallel: int = 1) -> dict:
    """First-class GPU+SSD Only / CPU+SSD Only runner flags for Inkling-2-Mini (never blocks or overcommits RAM)."""
    placed = place(build, vram_gib, ram_gib, unified, ctx, vision, parallel)
    v = float(vram_gib or 0.0)
    r = float(ram_gib or 0.0)
    ssd = placed["ssd_gb"]
    if ssd <= 0.05:
        if not unified and v < 4.0:
            return {"tier": "cpu_in_memory", "runnable": True, "placement": placed, "flags": ["--cpu"]}
        return {"tier": "in_memory", "runnable": True, "placement": placed, "flags": []}
    if not unified and v >= 4.0 and (placed["ram_gb"] < 8.0 and ssd > placed["ram_gb"]):
        return {
            "tier": "gpu_ssd_only", "runnable": True, "placement": placed,
            "bounce_ring_gb": round(min(2.0, max(0.5, r * 0.15)), 1),
            "flags": ["--ssd-experts", "--ram-experts-gb", "0"],
        }
    if not unified and v < 4.0:
        lru_gb = round(max(1.0, min(placed["ram_gb"] if placed["ram_gb"] > 0 else max(1.0, r * 0.5), max(1.0, r * 0.5))), 1)
        return {
            "tier": "cpu_ssd_only", "runnable": True, "placement": placed,
            "flags": ["--cpu", "--ssd-experts", "--ram-experts-gb", f"{lru_gb:g}"],
        }
    return {
        "tier": "hybrid_ssd", "runnable": True, "placement": placed,
        "flags": ["--ssd-experts", "--ram-experts-gb", f"{max(0.0, placed['ram_gb']):g}"],
    }


def multi_gpu_flags(gpus: list[dict], *, windows: bool = False) -> list[str]:
    """Automatic multi-GPU flags for 2..4 discrete CUDA GPUs: uses --layer-split auto across 2..4 GPUs
    so both experts AND global/SWA KV span the full combined VRAM without duplicating trunk/KV (#6592, #6631)."""
    n = min(4, len(gpus or []))
    if n < 2:
        return []
    return ["--gpus", str(n), "--layer-split", "auto"]


def parallel_fits(build: str, vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None, parallel: int = 1,
                  *, vision: bool = False, dspark: bool = True, backend: str | None = None, desktop: bool = False,
                  ram_free_gib: float | None = None, extra_vram_gib: float = 0.0, sm: int | None = None) -> bool:
    """True when `parallel` request slots at `ctx` tokens can start: their K/V and per-slot buffers fit the card (or the pool) beside the
    dense weights (the experts that do not fit go to RAM or the SSD, which is the plan's business, not the slots')."""
    kw = dict(backend=backend, desktop=desktop, ram_free_gib=ram_free_gib, extra_vram_gib=extra_vram_gib, sm=sm)
    if bool(place(build, vram_gib, ram_gib, unified, ctx, vision, parallel, dspark, **kw).get("kv_fits", True)):
        return True
    # the engines that have an FP8 K/V (CUDA, HIP) pick it when FP16 does not fit; SYCL has none
    return not _full_ctx_slots(backend) and bool(
        place(build, vram_gib, ram_gib, unified, ctx, vision, parallel, dspark, fp8_kv=True, **kw).get("kv_fits", True))


def max_parallel(build: str, vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None, *,
                 requested: int | None = None, vision: bool = False, dspark: bool = True, backend: str | None = None,
                 desktop: bool = False, ram_free_gib: float | None = None, extra_vram_gib: float = 0.0, sm: int | None = None) -> dict:
    """The largest number of request slots (1..MAX_PARALLEL) that fit at `ctx` tokens, and a plain answer for the `requested` count.

    {"max_parallel": N (0: not even one slot fits at this context), "requested": R, "fits": R <= N, "context": ctx,
     "max_context_at_requested": the largest context at which R slots fit (0: none), "message": one sentence for the user}"""
    n_ctx = DEFAULT_CONTEXT if ctx is None else int(ctx)
    kw = dict(vision=vision, dspark=dspark, backend=backend, desktop=desktop, ram_free_gib=ram_free_gib, extra_vram_gib=extra_vram_gib, sm=sm)
    best = 0
    for n in range(1, MAX_PARALLEL + 1):
        if parallel_fits(build, vram_gib, ram_gib, unified, n_ctx, n, **kw):
            best = n
        else:
            break                              # the cost only grows with the slots
    want = None if requested is None else max(1, min(MAX_PARALLEL, int(requested)))
    ctx_at_want = 0
    if want is not None and want > best:
        for c in sorted(set(CONTEXTS + [1024, 2048, 4096]), reverse=True):
            if c < n_ctx and parallel_fits(build, vram_gib, ram_gib, unified, c, want, **kw):
                ctx_at_want = c
                break
    where = f"{backend} " if backend else ""
    if want is None or want <= best:
        msg = f"{best} slots fit at a context of {n_ctx} tokens" if best else f"not even one slot fits at a context of {n_ctx} tokens"
    elif best == 0:
        msg = f"--parallel {want} does not fit at a context of {n_ctx} tokens on this {where}setup: not even one slot fits; use a smaller context"
    else:
        tail = f"; {want} slots fit up to a context of {ctx_at_want} tokens" if ctx_at_want else ""
        msg = f"--parallel {want} does not fit at a context of {n_ctx} tokens on this {where}setup: the largest count that fits is {best}{tail}"
    return {"max_parallel": best, "requested": want, "fits": want is None or want <= best, "context": n_ctx,
            "max_context_at_requested": ctx_at_want, "message": msg}


def resident_parallel(build: str, vram_gib: float | None, ram_gib: float, ctx: int | None = None, *, cap: int = MAX_PARALLEL, vision: bool = False,
                      dspark: bool = True, backend: str | None = None, desktop: bool = False, ram_free_gib: float | None = None,
                      sm: int | None = None, extra_vram_gib: float = 0.0) -> int:
    """The largest number of request slots (1..cap) whose K/V stays on the card next to what one slot holds there: the plan keeps the K/V in the card
    (kv_offload off) and gives the experts the same share of the card as for one slot. More slots than that spill their K/V to host RAM: the engine keeps
    a window of every slot's positions on the card and reads the rest over PCIe (Olares, RTX 5090 Laptop 24 GB, UD-Q4_K_XL, 32K, 3 slots: 11,008 of
    21,760 + 11,008 positions on the host, decode 86 tok/s after a short prompt, 82 after 4K, 44 after 30K).  Never below 1."""
    kw = dict(vision=vision, dspark=dspark, backend=backend, desktop=desktop, ram_free_gib=ram_free_gib, sm=sm, extra_vram_gib=extra_vram_gib)
    base = place(build, vram_gib, ram_gib, False, ctx, vision, 1, dspark, **{k: v for k, v in kw.items() if k not in ("vision", "dspark")})
    best = 1
    for n in range(2, max(1, int(cap)) + 1):
        p = place(build, vram_gib, ram_gib, False, ctx, vision, n, dspark, **{k: v for k, v in kw.items() if k not in ("vision", "dspark")})
        if (p.get("kv_offload") != "off" or float(p["vram_gb"]) < float(base["vram_gb"]) - 0.05
                or float(p["ssd_gb"]) > float(base["ssd_gb"]) + 0.05):
            break
        best = n
    return best


def launch_parallel(build: str, vram_gib: float | None, ram_gib: float, ctx: int | None, table: int, *, backend: str | None = "cuda",
                    sm: int | None = None, desktop: bool = True, dspark: bool = True) -> int:
    """"Fit to your card" for Inkling-2-Mini (what `strata launch` uses when --parallel is not given): the VRAM table's number, but no more than the slots
    whose K/V stays on the card (resident_parallel); an explicit --parallel is never touched by this.  The card is planned with the desktop reserve (the
    free memory is not known here), so a headless card gets a safe, not the largest, number."""
    if build not in BUILDS:
        return table
    return max(1, min(int(table), resident_parallel(build, vram_gib, ram_gib, ctx, cap=int(table), backend=backend, desktop=desktop, sm=sm, dspark=dspark)))

