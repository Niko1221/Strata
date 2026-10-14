# gfx906 guarded scalar17 QSA top-k

This is a follow-up to unmerged PR #1535, based on its ca3533157d066a13b128f3daa61f56411dcf3155 head. It is independent of the attention query-swizzle/reduce12 patches.

## Change and opt-in

Set `STRATA_GFX906_TOPK_REG=2 STRATA_GFX906_TOPK_FIT17=1`. FIT17 accepts exactly `1`; unset, `0`, `2`, or other strings retain the previous route. Existing public TOPK_REG parsing is unchanged. Optional `STRATA_GFX906_TOPK_FIT17_TRACE=1` prints enabled topology once per host thread, not which arm a device window executed.

The existing uncounted gfx906 decode branch (at most8 queries, capacity above33792 and at most67584 blocks) keeps the reference256 launch. If the last query has at most24576 cells, only reference writes. Otherwise all current device rows are checked: when max(n_bid)+1<=17408 the existing kernel specialized for17 keys/thread writes; otherwise register66 writes. For chronological queries scalar17's inclusive limit is69631 cells;69632 requires18 slots even though its tail is empty. Checking all rows also handles unsorted valid steps. Guards are uniform and precede barriers, so captured graphs can grow and restore.

The selection arithmetic, radix passes, NaN/signed-zero handling, ties and ascending output order are unchanged. Counted prefill, OLD override, small/oversized capacities, nq>8, CUDA and other HIP dispatch remain unchanged. The original public CUDA SIMT scorer is preserved.

## Public component qualification

A pinned public selector TU and public headers were compiled and linked directly with the public parity harness; no private support archive. Separately labelled external captured-score and graph-transition harnesses linked against those same public objects. Both gfx906 cards passed46 correctness processes plus8 fresh-process component ABBA processes. Checks include captured consumed IDs/reference equality, canaries/input bytes, graph-changing steps, threshold neighbors, nonmonotonic batches, tails, ties, NaNs and signed zero, T1/4/6/8/9 and fallback cases.

Mean of per-process median latency, ms/call:
- GPU0:0.1907362265 →0.141653739 (25.7332% lower)
- GPU1:0.1951325085 →0.144282520 (26.0592% lower)

All candidate process medians were below every control median on their respective card. These are isolated warm-cache component timings, not whole-model gains. The earlier predeclared30% engineering screen remains negative; this reproduction did not redefine it.

Actual device-image instruction comparison retained branch operands while normalizing addresses/encoding comments and trailing padding. Scalar17, guardedwide66, default66, counted33/66 and reference instructions/ABI/resources matched the separately qualified integration selector. Default-off kernels matched the public baseline. Resource summary:
- scalar17:56B scratch/lane,104SGPR/64VGPR,44/25 SGPR/VGPR spills
- guardedwide66:1048B scratch/lane,539/468 spills
- original66:1048B scratch/lane,527/467 spills
- all three:32908B LDS/block and4 waves/SIMD

The wide path retains extra guard/launch overhead and12/1 additional SGPR/VGPR spill counts versus original66. This is not a spill-free optimization or a promised win at every context.

## Separate model evidence

A separate qualified integration binary c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049, containing other previously qualified changes, was tested OFF/ON in fresh-process ABBA:
- Suffix-disabled profile, spec6/mtp-max-t4/suffix-draft0: TG+1.41318%, PP−0.00183%; all1024 IDs and counters exact.
- Ordinary adaptive suffix profile: TG+1.29991%, PP+0.08468%; both pairs positive and all1024 IDs/topology exact. The second pair had different timing-trained draft/cache work: offered+2, lookups+960, hits+939. This is practical end-to-end evidence including policy response, not constant-workload attribution.

These are not full-public-engine measurements. An earlier historical-counter-gated attempt failed and remains failed; a later parser-only failure stopped before its candidate arm and supplied no candidate conclusion. Each reported study is separately scoped. Two observations per arm on this hardware/model are not a general performance guarantee. CUDA/non-gfx906 runtime was not measured.

## Reproduction and limits

`python3 bench/results/2026-10-09-gfx906-fit17/check_guard.py` checks the bound and unsorted partition on CPU; it does not replace GPU parity.

The existing public `qsa_topk_parity --selftest` and positional CLI can exercise OFF/ON with TOPK_REG2. In particular run contexts24576,24577,69631,69632,200000 with queries1/4/6/8/9, capacity204800, COUNT0, plus counted and OLD cases. The existing selftest alone does not cover every new-boundary dynamic graph transition; the46-process qualification additionally used an external deterministic transition fixture driver and captured real scores. Those captures are not distributed in this patch.

Minimal component build uses clang HIP `-O3 -DNDEBUG -std=c++20 --offload-arch=gfx906 -DSTRATA_HIP_GFX906=1 -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1`, platform/hip_compat before include, selector and parity objects, then HIP/C++ runtimes. No libstrata_kernels support archive. Compiler image digest:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646. Source/results/resource provenance hashes are in results.json and isa-resource-proof.json.
