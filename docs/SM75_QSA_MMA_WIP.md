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

### October 9 GPU timing and model-quality follow-up

The complete current-head CUDA kernel library was used for six independent process rounds. A 100-call graph was timed with CUDA events, with native/MMA order alternating inside each process. Process-level paired Student-t CI95 uses df=5. Clocks were not locked. The same synthetic KV was reused and may reside in L2: these are isolated attention-call timings, not DRAM throughput, whole-layer prefix latency or production tok/s.

|INT8 KV capacity|T|Native us +/- CI95|MMA us +/- CI95|Paired reduction % +/- CI95|
|---|---|---:|---:|---:|
|65|2|30.29 +/- 0.13|12.74 +/- 0.06|57.96 +/- 0.09|
|65|3|30.36 +/- 0.14|12.90 +/- 0.10|57.51 +/- 0.31|
|65|4|30.36 +/- 0.14|12.88 +/- 0.12|57.59 +/- 0.23|
|2051|2|68.27 +/- 0.20|31.73 +/- 0.11|53.52 +/- 0.06|
|2051|3|95.86 +/- 0.27|43.38 +/- 0.12|54.75 +/- 0.04|
|2051|4|124.97 +/- 0.35|54.12 +/- 0.12|56.69 +/- 0.04|

Full-model fixed-token comparisons used a local logits-export diagnostic build of this head, retaining all CPU experts (six participants, PCIe=0). CPU load varied during testing. An initial comparison was confounded by upstream's default dynamic prefill CPU share: even the first native T=1 window differed. Those results were discarded. With `STRATA_PREFILL_CPU_SHARE=0`, fixed residency and adaptation off, two native prose runs were bit-identical across all 69 logits rows; every condition's initial T=1 row also matched exactly.

|Condition|Scored positions|Mean KL(native || MMA)|Top-1 agreement|Fixed-target PPL change|
|---|---:|---:|---:|---:|
|Prose, T=4, 10-token prompt|69|0.006188|95.65%|+1.35%|
|Code, T=2, 12-token prompt|93|0.007454|98.92%|-1.56%|
|Code, T=3, 12-token prompt|93|0.006822|98.92%|-2.74%|
|3872-token prompt, mixed windows|40|0.003919|100.00%|-3.31%|

The last condition scored 34 positions from T=4 windows, five from T=6 suffix-draft windows (native fallback) and one from T=1. It is not a pure T=4 test. Code text is repeated across two T settings; the 295 scored positions are not independent quality samples. PPL is exp(mean target NLL) on these small fixed continuations, not a standard corpus benchmark. No quality equivalence threshold or CI95 is claimed. Logits changed by up to approximately 1.91-2.95, despite much smaller isolated attention errors.

A separate 3656-token, single-needle free-generation smoke test with MTP returned `AMBER-7319` in both variants. It does not qualify 16K/80K retrieval or other needle placements. Larger quality datasets and true long-context checks still gate review; the candidate remains opt-in and Draft.

- Full CUDA engine integration and synthetic capture/masking checks passed as recorded above; real-model capture and runtime fallback failure injection remain pending.
- Improve kernel readability and review layout, masking and host dispatch overhead.
- Test quality with teacher-forced logits/KL, perplexity and long-context retrieval. Small attention error alone does not prove quality equivalence.
- Measure T=2/3/4 and longer contexts separately on latest upstream; true CPU end-to-end ablation is pending.
- Build HIP fallback; SYCL implementation is unchanged.
- Default remains disabled; this draft is not requesting merge or review yet.

### October 9 expanded fixed-token and true 16K checks

Same code head d115ed7 and complete-kernel diagnostic build as above; local logits export only. Full CPU experts retained, six participants, PCIe=0, fixed residency, adaptation/cache off, dynamic prefill CPU share disabled. Prefill=1024, max-context=32768. Suffix draft and lookup chain disabled: after the initial native T=1 position, every scored row uses the indicated T. External CPU load may vary; these are numerical comparisons, not end-to-end performance measurements.

|Fixed continuation|Prompt tokens|Scored positions|Top-1 agreement|Mean KL|Native PPL|MMA PPL|PPL change|
|---|---:|---:|---:|---:|---:|---:|---:|
|chinese-T3|9|300|94.67%|0.006560|17.363044|17.435742|+0.419%|
|code-T2|13|326|98.47%|0.004060|2.270584|2.255700|-0.656%|
|long16k-T4|16384|355|95.49%|0.004780|5.295901|5.312329|+0.310%|
|systems-T4|9|355|98.31%|0.004386|6.535099|6.521321|-0.211%|

The two independent 16K native runs have identical complete logits-file SHA256 (a09363dbc7b98b756588fcb45d6b5fcbf62b513af6b0e2a436567cd7cee821a7). Every condition's initial T=1 row matches between native/MMA. Total 1336 scored positions; the long condition repeats the English target, so this is three distinct continuations, not four independent corpora. PPL is exp(mean target NLL), not a standard corpus benchmark. No quality-equivalence CI95 is claimed from correlated token positions. Largest final-logit absolute difference is 7.15; this optimization is numerically different even though these PPL changes are small.

The separate free-generation smoke test used exactly 16384-token chat prompts, one needle at 10%/50%/90% of the filler, greedy MTP up to T=4, and one persistent engine per variant. All six native/MMA requests returned the exact code AMBER-7319; output token sequences match for each paired placement. Prompt cache/adaptation and suffix/lookup drafts were off. Repetitive filler and a single needle make this a limited smoke test, not a general long-context benchmark or 80K qualification.

For calibration, a separate 18-process experiment with MMA disabled compared default dynamic prefill CPU sharing (environment unset) against sharing disabled (0). On three small continuations, mean Top-1 agreement was 95.65%, 98.92%, and 95.00%. Paired process PPL changes were +1.46 +/-2.59%, -1.57 +/-3.12%, and +0.29 +/-3.08% (CI95, three pairs, df=2). Disabled-path independent repeats were bit-identical; default dynamic sharing varied across repeats. These small-sample scheduling differences are context for measurement, not an acceptance threshold for MMA. STRATA_PREFILL_CPU_SHARE=1 means a fixed 100% CPU share, not the default automatic mode.

Remaining review gates: a broader standard quality corpus and agreed numerical criteria; HIP fallback compilation; low-PTX fallback qualification with a supporting older CUDA toolchain; current-upstream layer/end-to-end ablation and final dispatch/kernel review. Keep opt-in/default-off and Draft.


# October 9: QSA MMA WikiText subset quality check

Three fixed, nonoverlapping target segments from the first 500 rows of Salesforce/wikitext, wikitext-2-raw-v1 test split. This is a subset check, not the full WikiText benchmark. Prompts are 256/256/8192 tokens; each segment has 384 teacher-forced target positions. T=2/3/4 each has native/MMA process pairs, with order alternated. CPU experts remain enabled, prefill CPU sharing, adaptive residency and speculative lookup/suffix paths are disabled. Same executable, same nominal cache byte budget; all 18 logs report 8136 initial resident experts. All initial T=1 logits are identical. CPU was shared with other programs; process wall times are not a throughput experiment.

| Segment | T | Native PPL | MMA PPL | PPL change % | Top-1 agreement % | Mean KL |
|---|---:|---:|---:|---:|---:|---:|
|wiki0|2|2.324350|2.370520|+1.986|94.271|0.025758|
|wiki0|3|2.320599|2.332855|+0.528|96.094|0.023137|
|wiki0|4|2.338502|2.359436|+0.895|93.750|0.027887|
|wiki1|2|1.472899|1.482264|+0.636|96.875|0.007000|
|wiki1|3|1.474211|1.465530|-0.589|99.219|0.007749|
|wiki1|4|1.474202|1.466795|-0.502|98.698|0.005825|
|wiki2long|2|3.188446|3.159130|-0.919|95.573|0.011450|
|wiki2long|3|3.189549|3.192524|+0.093|96.875|0.008349|
|wiki2long|4|3.199310|3.215236|+0.498|95.833|0.010810|

## Exploratory sequence-level intervals

The three target segments, not individual correlated tokens, are the units. Student-t CI95 uses df=2 on paired mean NLL differences; endpoints are transformed with exp(delta)-1 to PPL change. Only three segments and mixed prompt lengths make these exploratory intervals, not a quality-equivalence guarantee. T variants reuse the same corpus and are not additional independent samples.

| T | Pooled PPL change % | Exploratory CI95 % | Mean Top-1 agreement % |
|---|---:|---:|---:|
|2|+0.561|[-2.989, +4.240]|95.573|
|3|+0.010|[-1.380, +1.419]|97.396|
|4|+0.295|[-1.481, +2.103]|96.094|

Largest observed absolute logit difference: 7.053290605545044. Max per-position KL: 0.6280740816848684. The method changes numerical results; these data do not support bit equivalence or a general no-quality-loss claim. No acceptance threshold was prespecified. Keep the path optional and Draft pending review and broader qualification.

Source: https://huggingface.co/datasets/Salesforce/wikitext. Corpus provenance/hash: SOURCE.json; commands, fixed tokens and raw per-position statistics are in this directory. Results should be published without redistributing the corpus.
