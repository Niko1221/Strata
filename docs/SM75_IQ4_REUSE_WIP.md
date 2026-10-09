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

### October 9 isolated GPU performance follow-up

Six independent process rounds used the complete current-head CUDA kernel library. Native/four-row NW4/four-row NW3 process order rotated between rounds. Each measurement times a 100-call CUDA Graph with CUDA events; weight copies rotate beyond L2 capacity. Same-process repeats were merged before paired Student-t CI95 (df=5). GPU clocks were not locked; CPU was shared with other programs. The table is T=4, isolated matrix calls, not whole-layer latency or production throughput.

|IQ4_XS shape (K,R)|Native us +/- CI95|Four-row NW4 us +/- CI95|NW3 us +/- CI95|NW3 vs NW4 reduction % +/- CI95|
|---|---:|---:|---:|---:|
|2560, 6144|44.68 +/- 0.18|26.63 +/- 0.31|23.84 +/- 0.32|10.48 +/- 1.11|
|2560, 10240|71.32 +/- 0.42|40.33 +/- 0.23|36.51 +/- 0.80|9.48 +/- 2.11|
|2560, 12288|84.83 +/- 0.33|47.46 +/- 0.20|42.52 +/- 0.15|10.39 +/- 0.40|

T=1/2/3/4 were tested, with same-input bit comparisons passing in all runs. K=6144,R=2560 is a negative specialization control: both enabled variants dispatch NW4; their measured difference was 3.68 +/- 8.77%, compatible with no change. The four-row contribution overlaps #1418, so the smaller NW3 increment is the relevant additional contribution until that overlap is resolved. Current-head full-model parity and layer/end-to-end performance remain pending.

- CUDA 13, MSVC 14.51, SM75 full engine integration build passed as recorded above.
- Expanded `mmvq_multi_parity` with all three NW3 shapes and an incomplete four-row tile. Fresh results are recorded below after running.
- Re-run same-input real model checks on this exact head.
- Re-test prefix latency and true CPU end-to-end throughput on current upstream, interleaving run order.
- Reconcile implementation with #1418; review register pressure, dispatch overhead and graph capture.
- Build HIP and verify fallback behavior; no HIP hardware qualification claimed. SYCL implementation is unchanged.
- Keep default disabled until review and architecture/shape qualification complete.

Fresh isolated harness check (October 9): expanded synthetic parity passes with 980,113 exact-layout outputs, zero bit differences and zero non-finite outputs. The non-exact control finds 290,059 differences. The current-head native_mmvq and verify_kernels translation units were compiled and linked to the existing lab support library; this is not a full clean engine build. Both opt-in flags were enabled. Test GPU: RTX 2080 Ti, driver 616.92.

### October 9 reconciliation prototype

A local patch against #1418 head 8cae814e6e819736e47c95f3b5e8b056c7528c0f parameterizes the existing B6 row-reuse kernel with NW (default unchanged) and adds only the restricted SM75 IQ4_XS four-row NW3 dispatch. This avoids retaining a second generic row-reuse framework. CUDA compilation/linking passed; direct native-vs-prototype comparisons across four IQ4 shapes and T=1/2/3/4 are bit-identical with NW3 both off and on (16 cases each, no nonfinite outputs). Thirty repetitions only establish a smoke check; no new performance interval is claimed. The prototype is not yet the published branch and has not had a complete #1418 engine/HIP build. Published performance intervals above refer to this PR's current implementation. Official code convergence must state its #1418 dependency and be requalified before review.

### October 9 complete-model numerical check

A complete-kernel diagnostic build of code head ba2fe57 (logits export only, source restored) compared native, four-row NW4 and four-row NW3 in separate processes. Six runs: a 9-token prompt and an exact 16384-token prompt, each with 128 fixed continuation positions at T=4 (initial T=1). All CPU experts retained, six participants, PCIe=0; prefill=1024, dynamic prefill CPU sharing/residency adaptation/prompt caching/suffix drafts/lookup chain disabled. For each input, the complete logits files are byte-identical across all three variants (Top-1 100%, KL=0, target PPL unchanged on these samples). Diagnostic executable SHA256: 0d574c66612f8f33f7d0224b01f267f1365d3e267a566850a00539c7207300b4. These checks apply to the current published implementation, not the B6 integration prototype; no end-to-end performance or quality-equivalence CI is claimed.
