# DFlash2 long-prompt profile and exact CUDA head/attention optimizations

Measured 2026-10-20 on RTX 4070 Ti SUPER 16 GiB, Ryzen 9 5900X, 125.7 GiB RAM, Linux, CUDA toolkit 13.4, driver 610.57.04, SM89. Source parent: `2103f3e4b7fc56cfa3231ecc001ebe0a931a7db9`; tested source and binary hashes are in `provenance.json`.

The tested engine source is committed as `aab69ad03f8c225402ff0972e5b5a4bc0147fec1`; its source-file hashes match the benchmark build. The executable was built before that commit, so embedded build metadata retains the parent.

## Result

On the existing 2425-token fixture, the selected DFlash2 Q8_0 profile reaches **56.77 versus 55.31 decode tok/s for MTP (+2.64%)**. Every measured DFlash2 sample exceeds every MTP sample in this series. These ranges describe three samples, not confidence intervals or a cross-workload guarantee.

| Drafter | Code, 27 input tokens: median [range] tok/s | Long, 2425 input tokens: median [range] tok/s |
| --- | ---: | ---: |
| MTP, up to 3 proposals, floor 0.5 | 38.46 [38.34–38.60] | 55.31 [53.50–55.42] |
| DFlash2 Q8_0, block 7, floor 0.7 | 50.86 [50.82–50.87] | 56.77 [56.76–57.13] |

One warmup per task, then three measured repeats; 96 generated tokens per request, greedy seed 1234. Both arms use the **same final executable**, target NVFP4 pack, BF16 embeddings, FP8 PLE, full shared Q8_0 output head, INT8 KV, context/resident capacity 8192, prefill 256 and `--short-read 4096`. CPU prompt sharing, PCIe sharing, adaptive swaps, suffix/lookup drafting and prompt/conversation snapshots are off. Logs confirm zero reused prompt tokens for every final request. Engines run sequentially, without concurrent builds or benchmarks.

Both arms enable `STRATA_MMVQ_IL_Q8_HEAD=1` and `STRATA_DFLASH_HEAD_IL=1`. MTP therefore gets the same improved target-head projection; the comparison does not withhold that optimization from MTP. MTP uses verifier capacity 4 and its original `rt` drafter; DFlash2 uses capacity 8 and seven possible proposals, trimmed by floor 0.7. The forward budgets differ and MTP width was not independently tuned. Its exploratory floors 0.5/0.7/0.8/0.9 favored 0.5.

Actual expert placement is pinned and asserted: **3188 target experts for Q8_0 DFlash2, 3143 for MTP**, reserve 480 MiB in both arms. These are machine-specific placements, not equal expert counts or a claim of maximum residency on every restart. The configured profile and target arithmetic placement are part of the result.

Median long prompt reading / full client request time is **50.96 / 52.64 seconds for DFlash2**, versus **61.40 / 63.14 for MTP**. These include architecture and cache-placement effects; the request-latency difference is not attributed solely to the new kernels. Decode in the main table excludes prompt reading and startup.

All warmed/measured final DFlash2 and MTP outputs match the two target-only controls exactly: **192/192 fixture IDs**, without a near-tie exemption. Controls run in the DFlash2 process at the same 3188 actual slots, with `temperature=1 top_k=1 top_p=1 min_p=0 seed=1234`, offering zero proposals. Long DFlash2 accepts 68/71 proposals in 28 windows; MTP accepts 64/68 in 32. This is fixture evidence, not universal bit equality across placements.

## Diagnosis and changes

The original fixed block 3 and per-token floor 0.5 admitted costly rejected suffixes. The selected Q8_0 block 7 / floor 0.7 accepts more tokens per verification pass. The weight format, forward width and threshold need to be measured together; one floor is not optimal for every artifact.

The target head is **Q8_0, GGML type 8, 2560 × 248320, 675430400 bytes**. The engine previously printed Q5_K unconditionally. That misleading label also reached earlier reports; their format descriptions are corrected without rewriting timing data. The interleaved projection had no Q8_0 implementation, so enabling the old DFlash head switch could not accelerate this head. The new trait preserves the two integer dot operations, scaling order and original virtual-warp FP32 reductions. Its measured dispatch is opt-in, restricted to SM89, 248320 output rows and 3–8 columns. Other defaults and unmeasured dispatch cells remain unchanged. No vocabulary rows or head weights are dropped.

Wide DFlash2 heads also need a separate quantize/interleave preparation pass: the existing fused producer supports only 2–4 columns. The first integration exposed that guard and was fixed before confirmation.

Attention now reuses each address from QK for V. Official 32Q/8KV, 64-cell pages use shifts instead of runtime divisions. For long windows, 64 threads each accumulate two adjacent FP16 values, sharing a denominator. Each component retains the original cell order, FP32 operations, mask and merge. Short windows retain four warps; other geometries use general indexing. An alignment guard keeps non-aligned V buffers on the scalar path. No global buffer is added.

## Isolated kernel measurements

Three Q8_0 head trials compare the same random finite quantized weights and activation bytes against native MMVQ, at every row count 1–8 and rows-per-warp choice 1/2/4, including odd output sizes. Every comparison is bit-identical. Head shape: 2560 × 248320. Below are medians for the selected two-rows-per-warp dispatch; one/two columns stay on the old path.

| Projected rows | Native microseconds | Interleaved microseconds |
| ---: | ---: | ---: |
| 3 | 1230.5 | 1115.1 |
| 4 | 1332.0 | 1124.4 |
| 5 | 1507.6 | 1128.3 |
| 6 | 1677.4 | 1218.4 |
| 7 | 1813.4 | 1293.3 |
| 8 | 1896.3 | 1315.3 |

Attention uses graph replay, 20 warmups and 200 launches per trial, alternating before/after order. The reference compiles the parent's unchanged kernel with the current expanded harness. All three paired dumps are byte-identical: **1146880 bytes / 286720 FP32 outputs per pair**. Tests include nonidentity page tables, 64/96-cell pages, padded capacities, unaligned V, nonfinite inputs and FP64-reference comparisons. These kernel timings are not whole-model speedups.

| Query rows | Context end | Parent median ms | Patched median ms |
| ---: | ---: | ---: | ---: |
| 2 | 32 | 0.007414 | 0.006518 |
| 2 | 128 | 0.011515 | 0.009359 |
| 2 | 512 | 0.019359 | 0.015611 |
| 2 | 1024 | 0.032440 | 0.023752 |
| 2 | 2053 | 0.060114 | 0.044442 |
| 4 | 32 | 0.008033 | 0.007071 |
| 4 | 128 | 0.013184 | 0.011878 |
| 4 | 512 | 0.032148 | 0.023562 |
| 4 | 1024 | 0.058025 | 0.042286 |
| 4 | 2053 | 0.112497 | 0.082488 |
| 8 | 32 | 0.009887 | 0.008622 |
| 8 | 128 | 0.019814 | 0.016379 |
| 8 | 512 | 0.058696 | 0.043151 |
| 8 | 1024 | 0.112579 | 0.079964 |
| 8 | 2053 | 0.240287 | 0.159862 |

## Exploratory and negative results

`exploratory/` retains every initial Q4_0 block/floor sample, the MTP floor sweep, the automatic-placement follow-up, the fixed-placement paired comparison, the head-only tie and both cached 25-cell grids. None is pooled into the final table.

The Q4_0 profile did not establish a win: the earlier final head-only comparison gave 55.11 versus 55.15 tok/s for MTP. A separate automatic-placement Q4_0 restart at 3277 slots gave only 51.87 tok/s for block 7 / floor 0.8, despite identical target IDs. Startup/placement effects were not fully isolated, so this report pins actual placements rather than claiming three extra slots caused that loss. At 3274 slots, an intermediate paired process improved 49.93→54.99 tok/s when changing block 3 / floor 0.5 to block 7 / floor 0.8, but that alone was not a clear MTP win.

Cached grids speed exploration only. The Q4 grid's `pin=2424` request was ignored; its ordinary turn checkpoint reused 2420 tokens. The Q8 grid uses valid `pin=2423`. All grid outputs match the long target fixture. Selected settings are then remeasured with caching off. An exploratory warp-maximum reduction was discarded after flat/slower microbenchmarks and is not in the patch; its local logs remain outside the checkout.

Failed: the first wide-head integration aborted before a completed request with `native_quantize_q8_1_il requires 2 <= ncols <= 4`. Its full engine log and command hash are in `failed-wide-head-prepare/`. No partial timing is included in the table.

## Validation and limits

Passed: CUDA extension-ON and extension-OFF builds; **8/8 ON CTests with the new head option disabled, 8/8 with it enabled, 3/3 OFF CTests**; full-model output controls; three head parity trials and three attention dump comparisons. Default dispatch guards explicitly check that Q8_0 remains off unless requested, with other output sizes/cards unchanged. Full CTest logs and clearly labelled build-log excerpts are included. Full compiler logs, executables and binary dumps remain local.

Not run: HIP/SYCL compilation or execution, Metal, Windows, multiple GPUs, concurrent requests, a full 128K prompt, MTP width tuning or a held-out long-context suite. Keep the PR draft pending the broader backend validation and scope agreement. The profile was selected on this fixture. No four-task aggregate, general long-context speedup or default change is inferred.

## Reproduce

From the repo root, with the existing local artifacts referenced by the preceding report's commands, build the engine and use a fresh output directory. No downloads are performed. These fixed expert counts are for the measured machine; the helper rejects a different actual placement.

```sh
cmake --build build-dflash2 --target strata dflash2_parity mmvq_il_parity -j 4
python3 bench/results/2026-10-20-dflash2-long-tuned/run.py dflash2-il \
  --draft-quant Q8_0 --block 7 --floor 0.7 --expert-cache 3188 --reserve 480 \
  --only-selected --output /tmp/strata-long-recheck
python3 bench/results/2026-10-20-dflash2-long-tuned/run.py mtp \
  --floor 0.5 --expert-cache 3143 --reserve 480 --q8-head-il \
  --only-selected --output /tmp/strata-long-recheck
python3 bench/results/2026-10-20-dflash2-long-tuned/analyze.py /tmp/strata-long-recheck
STRATA_LONG_RESULTS=/tmp/strata-long-micro bash bench/results/2026-10-20-dflash2-long-tuned/microbench.sh
```

`run.py dflash2-grid --q8-head-il --draft-quant Q8_0 --expert-cache 3188 --reserve 480 --prompt-cache 3 --pin-prompt --output <fresh-dir>` reproduces the cached exploration. Exact per-request sampling strings and all warmups are retained. Source, artifact, executable and dump hashes are in `provenance.json` and `checks.json`; model paths in copied evidence are sanitized.
