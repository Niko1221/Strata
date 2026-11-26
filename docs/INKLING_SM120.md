# Inkling 2 on RTX 50 cards (sm_120): the switch set and the card table

The CUDA engine has one opt-in speed mode for the RTX 50 family and the RTX PRO Blackwell cards (compute capability 12.0). It is **off by default**.
The code is in `engine/src/inkling2/cuda` (`ink_sm120_cfg.cuh`, `ink_rtx5090_*`, `ink_rtx5060ti_opt.cuh`, `ink_5060ti_fp8.cuh`, `ink_moe_fp4_sm120.cu`);
the switches and the per-card table are in `engine/include/strata/core/ink_sm120_table.hpp` (no CUDA dependency, unit test `ink_sm120_table_test`).

## Switches (environment, read once per card)

| Variable | Effect |
|---|---|
| `STRATA_INK_SM120=1` | fast mode: FP8 trunk, MX8 to SF4 transcode of the four single-pass layers, FP16 sums in attention prefill, NVFP4 tensor-core MoE, deeper drafting, multi-user drafting. The answers differ slightly from the default path. |
| `STRATA_INK_SM120_EXACT=1` | the form that must give the same tokens as the default path: only the parts that do not change an answer. With `STRATA_INK_SM120=1` or alone. |
| `STRATA_INK_SM120=0` | the default path, always. |
| not set | the default path today. The default is one constant (`kSm120FastDefault` in the table header); it is flipped when the owner decides that fast mode is the default on the measured cards. |
| `STRATA_INK_SM120_<PART>=0|1` | one part on or off: `PF_ATTN FP8_TRUNK MOE_OPT DENSE_OPT DSPARK_OPT C64_SPEC COMPACT TC_SMALLM PLD PACK PREFILL CORE_GRAPH L2PF`. An explicit `=1` turns that part on on any cc 12.0 card. |
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
