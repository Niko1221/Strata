# WIP: SM75 IQ4_XS four-row activation reuse and three-warps specialization

This draft carries a narrowly scoped experiment, not a new default. `STRATA_IQ4_ACT_REUSE=1` selects four output rows per block for native/exact IQ4_XS calls at 2-4 columns, on SM75 only. `STRATA_IQ4_NW3=1` additionally uses three warps for K=2560 and output rows 6144, 10240 or 12288. Other calls retain their existing dispatch. TSUM and non-exact calls also retain their existing dispatch. HIP does not compile the candidate.

PR #1418 already proposes general activation reuse across rows for several formats. The four-row mechanism here overlaps it. The intended additional contribution is the measured three-warp specialization: ten IQ4_XS K blocks leave the fourth warp empty. This PR must be reconciled with #1418 before merging; two competing row-reuse frameworks should not be retained.

## Historical evidence (October 8, RTX 2080 Ti)

The model was Qwen3.8-Flash-Next IQ3_S with mixed dense projection formats. CPU experts were skipped, PCIe ratio was zero and dynamic expert replacement was off. These are diagnostic GPU experiments, not production throughput measurements. Three independent process pairs were combined with Student-t 95% confidence intervals (df=2), using actual T=4 windows. Prefix means layer start to publication of cold expert routes, excluding expert execution.

|Comparison|Metric|Latency reduction %, mean +/- CI95 half-width|
|---|---|---:|
|Four rows vs preceding candidate|GDN prefix, short context|1.687 +/- 0.682|
|Four rows vs preceding candidate|QSA prefix, short context|2.230 +/- 0.693|
|NW3 vs four-row NW4|Affected QSA projections|7.622 +/- 0.484|
|NW3 vs four-row NW4|Affected GDN projections|5.585 +/- 0.660|
|NW3 vs four-row NW4|Affected QSA prefix|1.481 +/- 0.450|
|NW3 vs four-row NW4|Affected GDN prefix|1.384 +/- 0.391|

Same-input historical shadow checks found zero bit differences and zero non-finite outputs for 62,089,216 four-row outputs, and 52,928,512 NW3 outputs. This does not qualify this newly rebased implementation. The intervals above come from separate incremental comparisons and must not be added together.

## Publication checks and remaining work

### October 9 construction update

Removed the unreachable single-row ablation branch from the four-row candidate. A clean Release SM75 build of the complete engine and `mmvq_multi_parity` passed with CUDA 13/MSVC 14.51 and the installed ggml source; it did not link an old Strata support archive. With both opt-in flags enabled, the full-build fixture compared 980,113 exact-layout outputs with zero bit differences and zero non-finite outputs. Its non-exact control found 290,059 finite differences, so the comparison can detect a changed reduction path. The three NW3 target shapes and an incomplete four-row tile are included.

These are synthetic correctness results, not a fresh performance result. Current-head model parity, capture, latency/throughput ablation and HIP builds still gate review. The row-reuse overlap with #1418 also needs resolution before merging.

- CUDA 13, MSVC 14.51, SM75 full engine integration build passed as recorded above.
- Expanded `mmvq_multi_parity` with all three NW3 shapes and an incomplete four-row tile. Fresh results are recorded below after running.
- Re-run same-input real model checks on this exact head.
- Re-test prefix latency and true CPU end-to-end throughput on current upstream, interleaving run order.
- Reconcile implementation with #1418; review register pressure, dispatch overhead and graph capture.
- Build HIP and verify fallback behavior; no HIP hardware qualification claimed. SYCL implementation is unchanged.
- Keep default disabled until review and architecture/shape qualification complete.

Fresh isolated harness check (October 9): expanded synthetic parity passes with 980,113 exact-layout outputs, zero bit differences and zero non-finite outputs. The non-exact control finds 290,059 differences. The current-head native_mmvq and verify_kernels translation units were compiled and linked to the existing lab support library; this is not a full clean engine build. Both opt-in flags were enabled. Test GPU: RTX 2080 Ti, driver 616.92.
