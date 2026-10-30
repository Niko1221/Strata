# Two Instinct MI50 32 GB (gfx906): decode tuning

This page describes the decode work on one machine with two MI50 32 GB cards and Qwen3.8 Flash Next IQ3_M. It covers
what was changed, which switches turn each change on, the measured effect and how each result was checked. The build
itself is described in [AMD_HIP.md](AMD_HIP.md#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source).

> **Measured on:** 2x Instinct MI50 32 GB (150 W limit each), Core i5-11600K, Z590 board, 64 GB RAM, Linux 7.0, ROCm 7.14
> (gfx906 community build), engine source 0.2.8 (870aedd9) plus the commits on this branch, re-checked on 0.2.9 (c62d5e3a,
> [below](#on-029)). Model: OrcaRouter Qwen3.8 Flash
> Next Uncensored IQ3_M, Strata pack `orca-iq3_m`. Card 0 sits on a CPU PCIe port, card 1 on PCIe 3.0 x4 behind the
> chipset. One machine, one model.

## Result

The same three requests run on each state: code (a Python function), prose, and long10k (a 9,927-token prompt). Each
generates 192 greedy tokens. All runs use `--layer-split 24 --spec 3 --kv int8 --pipeline-windows 2
--trim-stage-weights --resident-experts --spec-min-p 0.5` and an MTP drafter. The context is 10,240 tokens, except in
the last row.

| State | Code | Prose | long10k | Text |
| --- | ---: | ---: | ---: | --- |
| e0: stock 0.2.8 binary, drafter on card 0 | – | – | 51.5 tok/s | – |
| f0: drafter on the last card, `STRATA_GR_NORM_FAST=1 STRATA_IQ_STAGE_GRID_MMVQ=1` (existing switches) | 90.1 | 52.0 | 61.8 | reference |
| f7: + GEMV activations in LDS, four lanes in one (Q6_K head, Q4_K, IQ4_NL down), expert loads global | 102.5 | 57.2 | 69.0 | bit-identical |
| f10: + router GEMV without the dispatch stall (`STRATA_MMVF_ROWS=1`, also for one token) | 104.7 | 61.2 | 71.6 | bit-identical |
| f18: + German/English draft vocabulary (`--mtp-draft-vocab`, 66,829 ids) | 107.5 | 70.6 | 77.0 | bit-identical |
| f22: + expert gate/up H2, Q5_0 down via LDS | 112.2 | 72.7 | 78.5 | bit-identical |
| f25: + IQ GEMV two rows a warp, sampler loads in flight, upstream gfx906 attention switches | 112.8 | 73.4 | 81.5 | bit-identical |
| **f27: + hyper-connection down prefetch kernel, `STRATA_HIP_SWIGLU_FUSED=1`** | **115.0** | **74.7** | **82.3** | bit-identical |
| f27 at 131,072 context (the daily config) | 112.7 | 73.6 | 81.3 | bit-identical |

From e0 to f27, long10k goes from 51.5 to 82.3 tok/s (+60 %). The code request has the same draft acceptance in every
state (116 of 144), so its time shows only the kernels: 2131 ms at f0 and 1669 ms at f27 (−21.7 %). The draft
vocabulary raises prose acceptance from 65 of 128 to 80 of 130.

"Bit-identical" means each answer's text hashes the same as at f0. The long10k text also depends on the pipeline's
timing in the stock engine (see [Known issues](#known-issues)). For that reason, a kernel is called bitwise only when
its microbench hash and the code and prose texts are unchanged too.

Per-run data: [bench/results/2026-10-30-mi50-32gb-decode](../bench/results/2026-10-30-mi50-32gb-decode/README.md).

### On 0.2.9

The same branch on 0.2.9 (c62d5e3a), run on 2026-11-02 with the same requests and run config. "Existing switches" are
the gfx906 switches 0.2.9 already has: `STRATA_GR_NORM_FAST`, `STRATA_IQ_STAGE_GRID_MMVQ`, `STRATA_MMVF_ROWS`,
`STRATA_HIP_SWIGLU_FUSED`, `STRATA_GFX906_ATTN_QUERY_SWIZZLE` and `STRATA_GFX906_ATTN_REDUCE12`. Stock 0.2.9 does not
build with `-DSTRATA_HIP_GFX906=ON`, so it has the one-line `dflash2.cpp` guard from #3986 (also part of this branch).

| Build and switches | Code | Prose | long10k |
| --- | ---: | ---: | ---: |
| stock 0.2.9, no switches | 84.4 | 54.4 | 59.7 |
| this branch, no switches | 90.1 | 58.1 | 64.3 |
| stock 0.2.9, existing switches | 92.3 | 60.2 | 67.0 |
| **this branch, existing and new switches** | **115.1** | **74.7** | **82.8** |
| f27 binary (0.2.8 base), same day | 114.9 | 74.3 | 82.8 |

All five rows give the same four texts (code `d0f87379`, prose `294fcadd`, long10k `84a03701`) and the same draft
acceptance. The second row is the default path: it changes no output and is 6.8..7.7 % faster. With the switches,
the branch is 23.6..24.7 % faster than stock 0.2.9 with its own switches. `fused_gr_bench 200 1 8` gives the same
output hashes on stock 0.2.9 and on this branch, with and without `STRATA_GR_DOWN_PF=1` / `STRATA_GR_DOWN_TG=a`.

## Run it

Build as in [AMD_HIP.md](AMD_HIP.md#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source) (`-DSTRATA_HIP_GFX906=ON`;
this machine also needs `-DSTRATA_ORCA_Q4KS_MMQ=ON`, without which pipelined runs hung). Then put these in the run
config:

```json
"args": [ "...", "--layer-split", "24", "--spec", "3", "--kv", "int8", "--pipeline-windows", "2",
          "--trim-stage-weights", "--resident-experts", "--spec-min-p", "0.5",
          "--mtp-draft-vocab", "draft_vocab_de_en.bin" ],
"env": {
  "STRATA_MTP_STAGE0": "0", "STRATA_RESIDENT_PIN": "0",
  "STRATA_GR_NORM_FAST": "1", "STRATA_IQ_STAGE_GRID_MMVQ": "1",
  "STRATA_IQ_MMVQ_XLDS": "1", "STRATA_IQ_MMVQ_RB": "16", "STRATA_IQ_MMVQ_R2": "1",
  "STRATA_MMVQ_XLDS": "1", "STRATA_MMVQ_WPB": "8", "STRATA_MMVQ_Q6X4": "1", "STRATA_MMVQ_Q4X4": "1",
  "STRATA_EXP_X4": "1", "STRATA_EXP_GU_H2": "1", "STRATA_EXP_LDS_Q50": "1",
  "STRATA_GR_DOWN_TG": "a", "STRATA_GR_DOWN_PF": "1", "STRATA_MMVF_ROWS": "1",
  "STRATA_GFX906_ATTN_QUERY_SWIZZLE": "1", "STRATA_GFX906_ATTN_REDUCE12": "1", "STRATA_HIP_SWIGLU_FUSED": "1"
}
```

Each new switch is off by default. When a switch is off, that kernel runs exactly as before.

The draft vocabulary was built with `tools/draft_vocab.py` from the model's GGUF. The input was the German (`de_DE_frami`)
and English (`en_US`) hunspell dictionaries at coverage 0.999, on top of a German base subset. It is not part of this
branch.

## What changed

### 1. The router GEMV waited on a dispatch stall

On gfx906, a launch of 509..512 blocks of 256 threads (or 253..256 of 512) that does more than about 0.5 µs of work never
finishes in under about 46 µs. The blocks themselves are done after about 1.5 µs. The rest is a gap before the next
kernel, on the command-processor side. 480 or 520 blocks, and 64- or 128-thread blocks, are not affected. Per-block
timing with `s_memrealtime` in a standalone kernel showed this.

The BF16 router projection has 512 rows and one block per row, so it hit the stall on every layer. `STRATA_MMVF_ROWS=1`
already sent 2..8 tokens to the rows kernel. It now also sends one token there, with 2 rows per block (256 blocks). The
bits are the same. The router takes about 11 µs instead of about 46 µs.

To list the launch grids, use `AMD_LOG_LEVEL=3`: its `hipLaunchKernel` lines carry grid and block, also during graph
capture.

### 2. Quantized GEMVs: count address-unit instructions, not bytes

On gfx906, each vector-memory instruction costs the CU's address unit at least 16 cycles, whatever its width. Kernels
that read 4 bytes per lane per instruction are limited by that, not by HBM bandwidth.

- **Activations in LDS** (`STRATA_MMVQ_XLDS=1` for Q4_K/Q6_K, `STRATA_IQ_MMVQ_XLDS=1` for the staged IQ GEMV). The verify
  window's activation rows are copied into shared memory once per block. Before, each column added three to six
  global loads per lane and chunk.
- **Four lanes in one** (`STRATA_MMVQ_Q6X4=1`, `STRATA_MMVQ_Q4X4=1`, `STRATA_EXP_X4=1`). One lane does the work of
  four of the original kernel's lanes. It reads their 16 consecutive bytes in one load and keeps their four partial sums
  apart. Then it replays the original xor butterfly: steps 32..4 become xor 8..1, then `(p0 + p2) + (p1 + p3)`.
  That is the same additions in the same order. Q6_K output head and Q4_K qkv: −20..−27 % in the microbench. IQ4_NL
  expert down with one entry: 51 → 42 µs.
- **Rows per block** (`STRATA_IQ_MMVQ_RB=16`): the 4 KB IQ grid staging is paid once per 16 rows.
- **Two rows a warp** (`STRATA_IQ_MMVQ_R2=1`, 3..4 columns): twice the loads in flight. z/out/q at T=3 are −13..−19 % in
  the microbench, and −0.3..−0.5 ms per fresh window on stage 1 in the pipeline trace.
- **`__mul24`** in the Q4_K/Q6_K dot products (default on HIP). `v_mul_lo_u32` is quarter rate on gfx906, and the
  operands are below 2^14 and 128.

### 3. Grouped experts: fill the waves

- An expert blob's address is read from memory, so the compiler treated it as generic and issued `flat_load`. Flat
  loads also count against `lgkmcnt`, so each wait on an LDS read also waited on the HBM loads in flight. The pointer is
  now cast to the global address space (default on HIP).
- `(b << 8) - b` was folded by LLVM into `b * 255`, which is the quarter-rate `v_mul_lo_u32`. It is now a `v_perm`
  (default on HIP).
- Gate/up (mode 7) does at most two entries per pass, with `amdgpu_waves_per_eu(3)` in the gfx906 build. Before, a 4-entry pass sized
  the kernel at 112 VGPRs, which is 2 waves per SIMD for every group, though most groups hold one entry.
- **H2** (`STRATA_EXP_GU_H2=1`, `STRATA_EXP_DOWN_H2=1`). An IQ3_S gate/up row is 80 calls on 32 lanes, so the third
  round was half idle. Now a 16-lane half takes a row pair, and lane q does the work of lanes q and q + 16: the two
  sums are swapped each round, then the original warp sum is replayed. Down rows are 40 calls and get the same
  treatment.
- `STRATA_EXP_LDS_Q50=1` sends the Q5_0 down of layers 0..5 through the LDS down kernel.

GU_H2 together with LDS_Q50: −2.9 % end to end.

### 4. Hyper-connection read

The hyper-connection read runs per layer and reads two BF16 matrices. On gfx906 its kernels were latency-bound
(VALU busy about 9 %, memory unit not stalled).

- `__shfl_xor_sync` with a constant mask is a DPP or `ds_swizzle` step instead of `ds_bpermute`, which is a round trip
  through LDS per step (default on HIP, same values). That alone was about 4 µs per token.
- The up kernel keeps all 160 blocks resident: chunk-major loop, templated on T, `amdgpu_waves_per_eu(3)` (gfx906 build
  only). T=3: 23.4 → 18.4 µs.
- **`STRATA_GR_DOWN_PF=1`**: a new down kernel for 1..3 tokens that gives the same bits as the staged kernel. It runs
  6 rows per block, so 55 blocks on 55 CUs instead of 41: one CU streams only about 7..9 KB/µs, however many loads it
  has in flight. All 40 weight chunks of a row are loaded up front. A tile's activations come into registers in one
  batch, issued before the previous tile's dots. The engine's own start-up check ("checked bit for bit") passes.
- `STRATA_GR_DOWN_TG=a` picks the token grouping by count for the 41-block down.
- `STRATA_GR_SPLIT=1` (existing, opt-in) had a race: its slice sums were one array shared by every stream, so the MTP
  drafter's read raced the verifier's and the text changed from run to run. There is now one slot per stream. This
  path also quantized its output twice per read. Both are fixed.

### 5. Sampler and verify: loads in flight

`sampler_greedy_kernel` and `row_top_prob_kernel` issued one load per loop round. Over a 248,320-logit row that is 243
dependent rounds, about 110 µs. Each thread now loads eight elements before it compares or adds any of them, in the
same order, so the result is the same. This is on by default, on every backend. Effect: about −0.4 %.

### 6. Host: pipeline hangs

Host fixes for `--pipeline-windows 2` on HIP:
- The drafter's prefill was prepared for 3 rows, but the pipelined read hands it 5. The first larger window then freed
  and allocated memory while the device waited on the host. `reserve_prefill(o.spec, e)` reserves the real size up
  front.
- On HIP, `hipHostUnregister` waits for windows that still need the host. `pl_apply` now returns, and retries on the
  next round, while a stage still has a verify in flight.
- `dflash2.cpp` did not compile in the gfx906 build: it took the CUDA branch of the drafter-stream priority rule. It now
  takes the HIP branch (`STRATA_DF2_PRIO`). This is the same one-line guard as the open #3986.

## gfx906 lessons

- **Check VGPRs and the grid first.** VGPR count sets waves per SIMD. A grid that does not fit in one round on 60 CUs runs
  a second, mostly empty round. The hyper-connection up kernel (−17..−21 %) and the expert gate/up kernel (−9 %)
  lost that much this way.
- **`__syncthreads()` is `s_waitcnt vmcnt(0)`.** Every barrier waits for all loads in flight, not only the ones it
  guards. HIP's `cp_async` fallback is a blocking load, so staging loops serialize their L2 round trips.
- **One CU streams about 7..9 KB/µs.** Small grids cap bandwidth, whatever the loads in flight.
- **Pointers read from memory are generic.** Cast them to `address_space(1)`, or the loads become `flat_load` and mix
  with LDS waits.
- **Watch for `v_mul_lo_u32`.** LLVM turns shift-and-subtract into a multiply. Use `__mul24`, `v_perm`, or an empty
  `asm volatile("" : "+v"(x))` to stop it.
- **An if/else that picks the accumulator** lets LLVM hoist the identical dot out of it. The result is mul + add instead
  of `v_fmac`, and a different text. Count `v_fmac` in the ISA.
- **Hash before quantization.** An output hash after q8_1 quantization can hide a 1-ulp difference. The microbench hash
  does not prove the engine is bitwise either, so check end-to-end text hashes after every build.
- **A non-bitwise path must be T-invariant.** If a token's result depends on how many tokens share the window, or on
  the stream, the text changes from run to run.
- `rocprofv3 --kernel-trace` deadlocks the engine. Standalone benches (`fused_gr_bench`, `native_expert_bench`) work
  with it. In the engine, use `STRATA_DECODE_TIMING` and `STRATA_PIPELINE_DEBUG=1 STRATA_PIPELINE_TRACE=<file>`.
  `STRATA_VERIFY_PROFILE` turns off several fusions, so its router/shared numbers are inflated.
- A card behind the chipset PCIe port padded each kernel longer than about 7.5 µs to about 24 µs in standalone tests
  (probably link power management). In the engine its own PCIe traffic keeps the link awake, and swapping the cards
  changed nothing end to end.

## Tried and not kept

- Drafter on card 0 (`STRATA_MTP_STAGE0=1`): works with full residency, but is slower here (61 vs 71.5 tok/s long10k).
  The drafter chain then slows stage 0 in fresh windows.
- Forced HBM clock: 49.0 vs 51.6 tok/s. Power limits 130..220 W: about 1 %.
- `GR_V3` on gfx906, weight rotation, an LDS-free K split, a Q4_K wave GEMV with all loads issued before the dots, H2
  in the IQ3_S GEMV, two rows per group in the Q4_K four-lane kernel (110 VGPRs), and LO1 in the up kernel (spills):
  slower or no gain.
- IQ3_S GEMV `__mul24`, raw-byte prefetch, VALU sign masks: no gain. This GEMV is balanced between VALU, LDS and the
  texture/address unit, at about 300 GB/s.
- An external PCIe keep-alive process: breaks the pipeline (#267 hang).

## Known issues

- **long10k text depends on pipeline timing in the stock engine.** With identical kernels, about 1 in 9 long10k runs
  produces another text, with a different window structure. Inserting dummy nodes (`STRATA_DUMMY_NODES=1` / `3`) also
  changes it. Some part of the pipelined speculation path is not timing-invariant. Code and prose are not affected.
- CUDA 13.0 compiles (sm_75, sm_86, sm_120) but was not run on an NVIDIA card. SYCL and Metal sources are not touched.
  RDNA and gfx1151 HIP builds were not compiled or run; they get the DPP shuffles, the `global_load` cast, `v_perm` and
  `__mul24`, which keep the values by construction. The occupancy attributes apply to the gfx906 build only.
