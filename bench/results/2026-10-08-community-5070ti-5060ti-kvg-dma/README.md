# Heterogeneous RTX 5070 Ti + RTX 5060 Ti: experimental PP32/16 KV-grow and pinned DMA

Measured on 2026-10-08. **Results-only submission:** every change in this PR is inside this benchmark folder.
The patch files are reproduction artifacts; this PR does not apply them to the upstream engine.

## Scope and provenance

- Ten paired rounds use a **custom fork based on source tag v0.1.40.1**, with native engine string`0.1.40`.
- `--kv-grow` with layer split and `STRATA_DMA_BOUNCE` are local extensions. These are not stock-option results.
- Separate v0.1.40.3 prototype checks are **functional only**, not a ten-round speed benchmark.
- Original benchmark engine SHA256:`ef0c87a10bb3ef5e8b324ee199b8d24da023e2103de2dbeb0d18094fa75b2ccb`.
- Original full fixture SHA256:`02a514281cadaed2279ef4302aedeff2502bfb4de1e300b9c621e07d69b97341`. The shared gzip is the four relevant synthetic cases,
  not the larger original fixture file; message hashes remain identical.

## Hardware and model

- RTX 5070 Ti 16GB + RTX 5060 Ti 16GB; GPUs0/1, PP layers0-31 /32-47. P2P was previously checked unavailable.
- Ryzen 7 9800X3D, 8 cores/16 threads; Windows/WDDM; driver 616.56; Release source build, CUDA 13.0.88, SM120.
- Recorded OS-visible RAM:66,203,865,088 bytes (61.66GiB). OS build and storage type were not recorded.
- Clocks/background load were not fully controlled; per-GPU PCIe topology/power settings are not reconstructed.
- Original Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, the00001/00002-of-00002 shards. Exact filenames are in the fixture note.
- Model revision, full GGUF and expert-profile hashes were not frozen. Same files/profile/vocabulary in both arms.
- Vision and prompt reasoning disabled; temperature0, seed42. No calibration or experimental speed projection.

## Compared settings

Both arms:262144 capacity, int8 KV, expert-cache auto, trim-stage-weights, prefill4096, MTP4, spec-min-p0.5,
KV-grow, pipeline-windows2, layer split32/16, one serving slot.

- A (`grow-pipeline2`): no batched adaptive DMA.
- B (`grow-pipeline2-dma1-bounce`): A plus`STRATA_DMA_BATCH=1` and custom`STRATA_DMA_BOUNCE=1`.
- Both GPUs actually submitted batches in B. Batch-fallback counters were zero; single-copy loops are not counted
  as batch failures. The path optimizes adaptive expert refills, not all expert/KV/prefill traffic.
- CUDA 13.0 does not compile the 13.4 overlap hint; no distinct mode2 improvement is claimed.

## Workload and method

Fresh process per arm per round. Odd rounds A/B, even rounds B/A. Fixed actual8192-input/128-output warmup.
Each arm then processes65536,131072,261000 input tokens, in that order. At each length, the first request is
a diagnostic/warmup; the second is the primary steady measurement. There are 60 steady and 60 diagnostic rows.
Prefix cache0 and native reused0 are checked. Payload hashes match across arms and actual token counts are
confirmed by the engine. **Every selected output actually generated1024 tokens**, not merely a1024 cap.

The archive is synthetic repeated historical-task comments. The task defines`summarize_events`, then generates
Python comments to keep decode running. The function is completed first; truncating a trailing comment is
expected at the output cap. Four functional cases and input-preservation checks passed for all 120 selected outputs.
Comment-heavy decode/MTP behavior does not establish prose performance, long-distance recall or general quality.

Native freshly-read tokens/prompt_ms and generated tokens/decode_ms define throughput. Client TTFT begins at
request submission and ends at first nonempty answer delta, with thinking disabled. Client wall time includes
prompt processing and decode, excludes model loading and the post-response validator. Memory peaks cover the
entire visit, including loading, warmup and all three lengths. Expert-cache warmth is distinct from prefix reuse.

## Primary results: all ten median [minimum, maximum]

| Actual input | Arm | Decode tok/s | Prefill tok/s | TTFT s | Complete request s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 65536 | A | 116.22 [54.28, 118.96] | 2788.29 [2764.33, 2795.15] | 23.69 [23.62, 23.91] | 32.56 [32.32, 42.52] |
| 65536 | B | 89.98 [52.52, 120.03] | 2781.54 [2758.09, 2800.44] | 23.75 [23.59, 23.96] | 36.02 [32.22, 43.25] |
| 131072 | A | 116.53 [53.56, 123.62] | 2765.58 [2751.39, 2795.48] | 47.74 [47.25, 47.99] | 56.48 [55.52, 66.54] |
| 131072 | B | 117.61 [61.86, 121.95] | 2787.96 [2756.33, 2797.06] | 47.36 [47.22, 47.88] | 56.16 [55.60, 64.38] |
| 261000 | A | 111.82 [77.52, 113.04] | 2603.46 [2590.33, 2615.49] | 100.91 [100.43, 101.41] | 110.10 [109.69, 114.01] |
| 261000 | B | 114.08 [71.82, 117.97] | 2603.78 [2550.50, 2608.32] | 100.89 [100.71, 103.02] | 110.01 [109.79, 115.03] |

## Operator-requested trim: independent mean of eight

Each metric independently drops one minimum and maximum. Retained round IDs can differ between metrics.
The JSON retains dropped IDs, means, sample SD and paired changes. SD is not a confidence interval.

| Actual input | A decode tok/s | B decode tok/s | Change | A complete request s | B complete request s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65536 | 103.04 | 88.96 | -13.67% | 34.37 | 36.49 |
| 131072 | 107.52 | 115.56 | +7.49% | 57.86 | 56.35 |
| 261000 | 111.15 | 113.48 | +2.10% | 110.11 | 110.23 |

DMA did not improve every length. At 128K, full-ten median decode differs by about+0.93%; the larger
+7.49% trimmed-mean difference is sensitive to slow samples. At 261K, a small decode difference did not establish
a useful complete-request latency reduction. No causal paging/transfer or statistical-significance claim is made.
Observed slow samples are retained. This is not a stock baseline or a proof that the combined settings are optimal.

The v1 pinned buffers reserve512MiB total; median whole-visit process-RSS peaks were43.551GiB (A),44.050GiB (B).
Memory counters are in GiB/MiB as named, not decimal GB.

## Failed/partial visit

One partial control round2 visit excluded after external5s validator timeout; unchanged answer passed4 cases on recheck; timeout30s, preserved3 complete visits and reran the incomplete visit. Original record preserved.

The partial visit is outside the successful throughput table. This retry was not selected by low TPS.
No additional slow samples were deleted. Successful timing boundaries, binary, fixtures and configs stayed fixed.

## Separate v0.1.40.3 functional checks

Two variants each ran64K ->128K ->261K ->8K, with actual1024 output each:8 successful outputs, no prefix reuse.
On both variants, allocated KV changed2112 ->160MiB (GPU0) and1320 ->100MiB (GPU1) after long-to-short transition,
and resident expert counts rose. KV counters exclude GDN/rope/scratch and include registered fixed drafter pools.
The bounded staging variant allocated213.312MiB per GPU,426.624MiB total,85.376MiB below the earlier512MiB.
Its long-request cumulative batch counters were300/157, fallback0/0. This is not evidence of a new speedup.
Release build, dual VMM ownership/data/stable-address checks, dual DMA data/reuse checks,26 security tests and
legacy DONE-metadata compatibility passed locally. The public patches omit local loopback-auth changes; the
26 auth tests were for the local integration, not additional tests of these stripped engine artifacts.

## Artifacts and reproduction

- `benchmark-v01401-raw.csv`:120 selected samples, source/answer paths removed; native draft/hit/PCIe counters included.
- `benchmark-v01401-stats.json`:full-ten statistics, original trim results and limitations.
- `functional-v01403.json`:eight functional samples; not a speed table.
- `synthetic-cases.json.gz`:exact four synthetic code-case messages, with per-message hashes.
- `catalog.py`, `harness.py`, `validators.py`, `experiments.py`, `reproduce.py`:portable extraction of the measured protocol.
- `prototype-v01401.patch`:engine plus numeric-metadata parsing against tag v0.1.40.1, used for the repeated results.
- `prototype-v01403.patch`:ported/bounded variant against tagv0.1.40.3, used for functional checks.
- `dma-warning-atomic.patch`:separate minimal guard suggestion, not a reported runtime failure.

Neither prototype includes credentials, loopback-auth bypass, Windows launchers, local paths, original service logs,
or original Git history. Patches contain source differences, not binaries. They are reproduction artifacts,
not a request to merge all prototype changes into the engine.

Example (substitute your own model paths; do not run while another Strata/GPU workload is active):

```sh
git worktree add --detach ../strata-report-v1 v0.1.40.1
git -C ../strata-report-v1 apply /path/to/this/report/prototype-v01401.patch
cmake -S ../strata-report-v1 -B ../strata-report-v1/build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=OFF
cmake --build ../strata-report-v1/build --target strata --parallel 2
python -m pip install -r ../strata-report-v1/requirements.txt
python -m pip install requests psutil
python reproduce.py --check-fixtures
python reproduce.py --run --rounds 10 --strata-src ../strata-report-v1 --engine ../strata-report-v1/build/strata --engine-version 0.1.40 --pack /path/to/pack --native /path/to/shard1.gguf --ple-gguf /path/to/shard2.gguf --expert-profile /path/to/expert-profile.bin --mtp /path/to/mtp
```

Windows requires a configured native compiler/CUDA shell, uses`strata.exe`, and may need`--lib-dir` for CUDA DLLs.
Use the fixed upstream ggml dependency or supply`STRATA_GGML_DIR` at configure time. Freshly building can produce
a different binary hash because of toolchain/debug paths; the measured engine hash above is provenance.
For thev0.1.40.3 variant, substitute that tag, patch and engine-version0.1.40.3. It is a new run, not the old results.

**This publication pass verified clean-tag patch application, core-file/DONE-parser equivalence to the previously tested source, syntax, synthetic fixture grammar/hashes and
offline safety checks. It did not rebuild the stripped public patches or rerun inference.** The full local
prototype previously built and ran; public stripped-patch builds and broader platform checks remain outstanding.
The runner refuses an existing Strata, occupied GPUs or insufficient RAM; model work requires`--run`.
It owns and cleans up only processes it starts. Local reproduction outputs can contain machine paths: do not
upload those without a new privacy review.

## Related upstream work

- [Community report guidance](https://github.com/Niko1221/Strata/blob/main/docs/COMMUNITY_BENCHMARKS.md).
- [VMM ownership/peer KV-grow#1223](https://github.com/Niko1221/Strata/pull/1223):scope overlap; layer split stays excluded there.
- [No-loan KV-grow safety#1231](https://github.com/Niko1221/Strata/pull/1231):small-cache cases were not tested here.
- [VRAM budgeting#765](https://github.com/Niko1221/Strata/issues/765).
- [Batched-upload design#807](https://github.com/Niko1221/Strata/pull/807):submission savings do not imply model-speed gains.

## Privacy

Public files were checked for machine user names, personal contact details, host/network identifiers, absolute
personal paths, credential values/locations, raw service/CLI logs and private prompt content. Only synthetic
messages, aggregate/per-run numeric evidence, code artifacts and necessary hardware/software facts are shared.
GitHub naturally identifies the submitting account; commit metadata uses its GitHub noreply identity.

No long-distance recall, perplexity, Linux/HIP/CUDA12, other quantizations or other splits are claimed.
