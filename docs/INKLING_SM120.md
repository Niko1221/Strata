# Inkling 2 on RTX 50 cards (sm_120): the switch set and the card table

The CUDA engine has one speed mode for the RTX 50 family and the RTX PRO Blackwell cards (compute capability 12.0). **It is the default on the two cards that have a measured table row:
the RTX 5090 and the RTX 5060 Ti 16 GB.** Every other card (the 8 GB 5060 Ti, 5080, 5070 Ti, 5070, 5060, 5050, RTX PRO, and every non-RTX 50 card) runs the old path until its own row is measured.
The code is in `engine/src/inkling2/cuda` (`ink_sm120_cfg.cuh`, `ink_rtx5090_*`, `ink_rtx5060ti_opt.cuh`, `ink_5060ti_fp8.cuh`, `ink_moe_fp4_sm120.cu`);
the switches and the per-card table are in `engine/include/strata/core/ink_sm120_table.hpp` (no CUDA dependency, unit test `ink_sm120_table_test`).

## What the default is, per card

| Card (matched by name, SM count, VRAM) | Default | Fast set of the row |
|---|---|---|
| RTX 5090 (32 GB) | fast mode | FP8 trunk (cublasLt, a separate FP8 copy of the attention and controller weights, +641 MiB; it does not start when there is no room for the copy), NVFP4 tensor-core MoE with the MX8 to SF4 transcode of the four single-pass layers, FP16-accumulate prefill attention, fused dense path, deep verify tree, multi-user drafting, tensor-core split-K decode attention at long context (`TC_ATTN`, 128K decode 84 to 238 tok/s; `STRATA_INK_SM120_TC_ATTN=0` takes it out) |
| RTX 5060 Ti 16 GB | fast mode | COMPACT (all 3,072 SQ-20 experts in VRAM: MX8 to SF4 transcode of layers 20-23, FP8 KV cache, single slab; attention weights and LM head Q8_ROW, embedding in mapped host memory), TC small-M MoE, layer packing, layer-major prefill, C1 core graph, prompt-lookup drafting, multi-user drafting, L2 prefetch. No FP8 trunk. |
| RTX 5060 Ti 8 GB, 5080, 5070 Ti, 5070, 5060, 5050, RTX PRO, any other card | old path | none (placeholder rows, no measurement yet) |

When the default applies, one line is printed (`RTX 50 fast mode is the default on GPU 0 (...): STRATA_INK_SM120=0 turns it off ...`). The default is one constant, `kSm120FastDefault` in the table header, and only a card
with a measured row ever gets it.

**Falling back.** The 5060 Ti COMPACT default refuses before it converts anything when the card cannot hold it (a desktop or another program on the card, the Windows WDDM reserve, a smaller card of the
same name, a model with expert formats other than SF4/MX8): one line `COMPACT refused: ...` and the whole device runs the old path, exactly as with `STRATA_INK_SM120=0`. The 5090's FP8 cache checks its own room the
same way and runs the BF16 / Q8_ROW GEMMs when it has none.

## Switches (environment, read once per card)

| Variable | Effect |
|---|---|
| `STRATA_INK_SM120=0` | the old path, always (this also turns the default off). The old names `STRATA_INK_RTX5090=0` and `STRATA_INK_RTX5060TI=0` do the same. |
| `STRATA_INK_SM120=1` | fast mode on a card that has a table row, also on a card where it is not the default. |
| `STRATA_INK_SM120_EXACT=1` | the form that gives the same tokens as the old path: only the parts that do not change an answer (5090: none, the exact form must not change the drafting policy; 5060 Ti 16 GB: L2 prefetch only). With `STRATA_INK_SM120=1` or alone. |
| not set | the default of the table above. |
| `STRATA_INK_SM120_<PART>=0|1` | one part on or off: `PF_ATTN FP8_TRUNK MOE_OPT DENSE_OPT DSPARK_OPT C64_SPEC COMPACT COMPACT_XCODE COMPACT_KV FP8_INPLACE TC_SMALLM PLD PACK PREFILL CORE_GRAPH L2PF TC_ATTN`. `=0` takes the part out of the default set; an explicit `=1` turns it on on any cc 12.0 card. |
| `FP8_TRUNK`, `FP8_INPLACE` | `FP8_TRUNK` is ONE implementation on every card: the 5090's cublasLt FP8 with a separate FP8 copy of the attention and loop-controller weights (+641 MiB); it does not start when the card has no room for the copy, and the BF16 / Q8_ROW GEMMs run. The 5060 Ti hand-over's in-place conversion (no copy kept) failed FASTQ on 2026-11-26 (tool calls 6/26 against 23/26 off, 24/26 with FP8_TRUNK=0) and is reachable only as `STRATA_INK_SM120_FP8_INPLACE=1`; it is in no fast set. |
| `COMPACT`, `COMPACT_XCODE`, `COMPACT_KV` | `COMPACT` (all 3,072 SQ-20 experts in VRAM on a 16 GB card) is the heart of the 5060 Ti 16 GB default. FASTQ on 2026-11-26: with the in-place FP8 trunk on it failed (tool calls 6/26 vs 23/26), with attention in Q8_ROW (the default set) it is at or above the old path on every suite. It bundles lossy steps, two of which can be named apart: `COMPACT_XCODE=0` keeps the MX8 experts of layers 20-23 as MX8 (SQ-20 then does not fit and the loader falls back), `COMPACT_KV=0` keeps the KV cache FP16 instead of FP8 E4M3. Both are on whenever `COMPACT=1` and not named. What cannot be split off: the attention weights and the LM head become Q8_ROW, the embedding is mapped host memory. |
| `TC_ATTN` | tensor-core split-K decode attention of the 8 global visits at long context (fast only: FP16 tensor cores, FP16 probabilities; changes rounding, never in an exact set; in the 5090 default set, in no other row yet). One block per KV head, key slice and group of up to 16 verify rows, every K/V tile read once for the group. Kernel test against the old kernel: relative L2 4-8e-4 (FP8 KV) / 2.9e-4 (FP16 KV), 7-12x faster per visit at 128K for M = 5..16 rows on an RTX 5070. A/B knobs: `STRATA_INK_TC_ATTN_NBUF=1|2`, `STRATA_INK_TC_ATTN_DEPTH=1|2|3`. Falls back to the old kernels for K8V8 / K8V4 K/V, SWA visits and prefill chunks. |
| `STRATA_INK_SM120_ROW=<id>` | use the row `<id>` (`5090`, `5060ti-16`, ...) on a card without a measured row, at your own risk. |
| `STRATA_INK_SM120_MOE_FORCE=<sf4>[,<mx8>]` | force a MoE kernel variant (`base bk128 direct tr128 tr128s tr128bk`) for A/B runs. |
| `STRATA_INK_ELASTIC_1M=1` | the 3-tier elastic K/V cache (VRAM, host RAM, SSD) for 64K to 1M tokens. A separate feature: it is not part of the fast mode. |

Old names still work as aliases (the new name wins if both are set): `STRATA_INK_RTX5090`, `STRATA_INK_RTX5090_EXACT`, `STRATA_INK_RTX5060TI`, `STRATA_INK_RTX5060TI_EXACT`,
`STRATA_INK_5060TI_<PART>` and the numeric `STRATA_INK_5060TI_{PCIE_SHARE,CHUNK,BEAM,MAX_NODES,PLD_MAX_M,SPEC_COST,SPEC_PLAIN,C64_DRAFT_K,C64_TAIL_MODE,MACRO_BF16}`
(new form `STRATA_INK_SM120_<same>`).

## Which card gets what

The card is matched by device name, number of SMs and VRAM, never by "cc 12.0" alone. Rows: 5090 and 5060 Ti 16 GB are **measured**; 5080, 5070 Ti, 5070, 5060 Ti 8 GB,
5060, 5050 and the RTX PRO 6000 / 5000 / 4500 / 4000 / 2000 are **placeholders** (their numbers come from rentals later).

- A switch on a card that is not cc 12.0 (an RTX 3090, a 4090, a DGX Spark GB10): one line on stderr, the default path runs.
- A cc 12.0 card without a measured row: one line on stderr, the default path runs (unless `STRATA_INK_SM120_ROW` or an explicit part switch says otherwise).

Each row holds: the parts that run in fast mode and in the exact form; how the FP8 trunk is held (a cache next to the BF16 trunk on a 32 GB card, converted in place on a 16 GB card);
which multi-user drafting implementation sits behind the one hook in `ink_backend.h`; the MoE tile size and kernel variant per batch range; the C1 prefill chunk;
the workspace reserves; how big a card "compact" (all 3,072 SQ-20 experts in VRAM) is tried on; the drafting numbers. Experts that do not fit stay on the CPU/RAM path
(the hybrid engine), as before; the table decides only how many fit.

## VRAM sizing

`plan_vram_reserve()` and `plan_prefill_chunk()` at the top of `ink_cuda_backend.cu` are the one place where memory is set aside for the K/V, the ShortConv state and the workspaces
before the experts are placed, and where the prefill chunk is chosen after. With every switch off they return the shipped numbers.
