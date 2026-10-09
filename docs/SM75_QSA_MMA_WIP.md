# WIP: optional SM75 padded QSA decode MMA

`STRATA_QSA_SM75_MMA=1` permits the padded FP16 QK/PV MMA candidate for explicitly marked main-model verification calls with 2-4 queries on SM75. FP32 accumulators are used. T=1, MTP, prompt processing, other architectures and HIP keep their native path. The query/head geometry and split-K scratch/merge layout remain the existing 256-dimension, 12-query-heads-per-KV-head, 64-cell design.

The shared tiles have padded row strides to reduce bank conflicts. K and V are read directly through selection IDs and the page table; they are not pre-expanded or permanently repacked. Failure to configure the required shared memory falls back to native attention.

## Historical performance evidence (October 8, RTX 2080 Ti)

Qwen3.8-Flash-Next IQ3_S; CPU experts skipped, pool=1, PCIe=0, adaptation/prompt cache off. Actual T=4 only. Nine independent processes, three paired rounds, with native/unpadded/padded run order reversed across rounds. Same-process repeats were merged before Student-t CI95 (df=2). GPU clocks were not locked. These results are not a current-upstream end-to-end claim.

|Prompt length|Metric|Native us +/- CI95|Padded MMA us +/- CI95|Paired latency reduction % +/- CI95|
|---|---|---:|---:|---:|
|78 tokens|QSA attention|55.172 +/- 0.294|20.798 +/- 0.216|62.303 +/- 0.420|
|78 tokens|QSA prefix|427.750 +/- 1.720|393.383 +/- 0.306|8.034 +/- 0.311|
|3038 tokens|QSA attention|131.672 +/- 0.557|67.977 +/- 0.099|48.374 +/- 0.287|
|3038 tokens|QSA prefix|534.803 +/- 3.137|471.568 +/- 0.280|11.824 +/- 0.497|

Prefix is layer start to cold-expert route publication, excluding expert execution. The 3038-token condition must not be presented as a 16K/80K validation. No GDN prefix benefit was observed.

## Numerical qualification

This candidate is **not bit-exact** to native FP32 attention. Historical main-model same-input checks over 38,486,016 outputs found no non-finite outputs, max absolute error 0.00710916519 and aggregate NRMSE 0.000151357684. Stress cases reached max absolute error 0.0702090263 and NRMSE 0.00118237193. MTP was substantially less stable, hence it is explicitly excluded. Padded and unpadded MMA were bit-identical to each other, which is a different comparison from native.

## Publication checks and remaining work

- CUDA 13, MSVC 14.51, SM75 translation unit compiled locally. Full engine integration, capture and fallback tests remain pending on this rebased head.
- Simplify the imported kernel to the single QK/PV variant, review layout/masking, and add reproducible numerical fixtures.
- Re-run every KV mode on same inputs, including partially masked/empty chunks and non-multiple-of-64 selections.
- Test quality with teacher-forced logits/KL, perplexity and long-context retrieval. Small attention error alone does not prove quality equivalence.
- Measure T=2/3/4 and longer contexts separately on latest upstream; true CPU end-to-end ablation is pending.
- Build HIP fallback; SYCL implementation is unchanged.
- Default remains disabled; this draft is not requesting merge or review yet.
