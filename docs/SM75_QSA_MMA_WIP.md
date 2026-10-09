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

### October 9 construction update

- Simplified the imported kernel to QK/PV MMA only, removing unused scalar ablation branches. Page-row address arithmetic now widens before multiplication.
- Preserved the original attention function signature and added an explicit verifier-role overload, avoiding a shared-header ABI mismatch with the existing SYCL definition. Rebuilt the complete CUDA engine after this fix. SYCL was not built locally.
- Added a compiled-image check (`cudaFuncAttributes::ptxVersion >= 75`) before enabling MMA. A lower-target PTX image must fall back even on SM75, because its MMA body was compiled out. CUDA 13 rejects SM70 compilation, so an older-toolchain runtime test remains pending.
- A clean Release SM75 engine and `qsa_mma_parity` build passed with CUDA 13/MSVC 14.51, using the installed ggml source. No old Strata support archive was used in this build.
- The new reproducible fixture passed 240 cases: all four KV formats; T=1/2/3/4/5; capacities 1/65/129/2051; reversed page mapping; fully valid, partly masked and fully masked pages; per-query widths; initial candidate use during CUDA Graph capture and repeated replay. Maximum absolute error was 0.000244141; worst per-case NRMSE was 0.000308676 against native attention. Bounds of 0.005 were specified for this bounded random fixture before execution; they are not model-quality thresholds.
- T=1/5 fallback was bit-identical within this build. With the opt-in flag set to zero, all 240 comparisons were bit-identical. This does not yet establish byte equivalence against a separately built upstream engine.
- Reproduce: build `qsa_mma_parity`, then run with `STRATA_QSA_SM75_MMA=1`. CTest sets the flag for the test. This fixture measures correctness, not speed or CI95.

The main release gates remain model-quality checks, current-head performance ablation, upstream default-path comparison, and HIP builds. Do not infer quality from the small synthetic error or infer new performance intervals from the historical table.

- Full CUDA engine integration and synthetic capture/masking checks passed as recorded above; real-model capture and runtime fallback failure injection remain pending.
- Improve kernel readability and review layout, masking and host dispatch overhead.
- Test quality with teacher-forced logits/KL, perplexity and long-context retrieval. Small attention error alone does not prove quality equivalence.
- Measure T=2/3/4 and longer contexts separately on latest upstream; true CPU end-to-end ablation is pending.
- Build HIP fallback; SYCL implementation is unchanged.
- Default remains disabled; this draft is not requesting merge or review yet.
