# IQ1_M fused prefill on an RTX 2080

This opt-in adds IQ1_M gate/up with Q2_0 down to Strata’s existing fused int8 prompt kernels on SM75. It targets the three remaining fallback layers (8, 13 and 37) of this mixed IQ2_XS pack. Defaults are unchanged. This is a custom A/B experiment, not a conforming report under the draft standard benchmark protocol.

## Matched results

RTX 2080 8 GB, i5-9600K, 64 GB RAM, Omarchy 4.0.6 (Linux), driver 615.78.08, CUDA 13.4.92, GCC 16.2.1, 165 W. Swift 1.5 Flash-Next IQ2_XS: 34 IQ2_S/Q2_0 layers, 11 IQ2_XXS/Q2_0, three IQ1_M/Q2_0. Same model files, expert-profile seed and settings within each comparison. Context capacity 131,072; the cold edit has 122,451 input tokens and every run produced 1,433 output tokens. Temperature 0, seed 1234, reasoning disabled. Every cold request reported zero cached tokens.

| Build / actual prefill chunk | Flag off, tok/s | Flag on, tok/s | Change | Exact-edit passes off / on |
| --- | ---: | ---: | ---: | --- |
| Current upstream + this patch / 1,024 | 353.4 (353.4–353.5) | 353.4 (353.4–353.5) | +0.0% | 2/2 / 1/2 |
| Local tuned v0.2.7 integration / 4,096 | 848.1 (845.9–850.3) | 874.9 (874.3–875.5) | +3.2% | 1/2 / 1/2 |

All HTTP runs used the same v0.2.7 Python server (Python 3.14.8); “upstream” identifies the C++ engine at the pinned main revision, not a different server/template. Two fresh processes per arm. Upstream order: off/on/on/off. Integration order: installed production/off/on/on/off/installed production. Rates above are total prompt tokens divided by total prompt time; parentheses are individual-run ranges. The integration build retains local prefill loans and broader SM75 native-IQ dispatch. It is a separate result, not an estimate of this patch’s gain over stock upstream. Upstream reduced the requested 4K chunk to 1K; compare only within each row. In the tuned build, covering all 48 layers also activates Strata’s existing fully fused buffer/ring layout (`fused_ring()`); the gain includes that planner response and is not a claim about standalone MMA throughput.

Usual speed suite on the integration build (`tools/strata_bench.py --runs 3 --long`), medians:

| Prompt size | Off, tok/s | On, tok/s |
| --- | ---: | ---: |
| 4K | 662.9 | 650.8 |
| 16K | 865.5 | 907.9 |
| 32K | 881.7 | 919.7 |

Both speed suites completed without request failures or skips. This suite measures speed, not answer correctness. The 4K case was slightly slower with the flag; this is not a claim that every prompt gets faster. Conversation-cache hit medians were 69.420% off and 69.433% on. Decode expert-cache hit rates and raw counts are in `expert-cache.json`; these are separate from conversation-prefix hits. No decode-kernel change or general decode speedup is claimed.

## Automatic chunk sizing (separate configuration)

The diagnostic used `--prefill auto:8192 STRATA_PREFILL_LEND_PCT=100` on the tuned integration build. Auto searches intermediate sizes that an explicit 8K request (halved to 4K) skips. The 100% loan ceiling matches the ceiling available to an explicit request; the existing 128-slot reserve still applies. This configuration is experimental and was not promoted.

The enabled exploratory probe selected 4,608 tokens, measured 886.6 tok/s and passed the edit. It is excluded from the following A/B aggregate.

| Auto arm | Actual chunk | Prompt tok/s (range) | Edit passes |
| --- | ---: | ---: | ---: |
| Off | 4,352 | 842.2 (842.0–842.4) | 1/2 |
| On | 4,608 | 883.1 (882.9–883.3) | 2/2 |

Matched off/on/on/off, two runs each: +4.9%. This includes the different selected chunks and buffer layout, not just kernel throughput. The enabled auto result is +0.9% versus enabled fixed-4K runs from the earlier campaign; that cross-campaign comparison was not interleaved. All outcomes and chunk-selection logs are retained in `auto-summary.json`.

## Correctness and limits

- CUDA reference tests passed on the clean upstream and integration builds, including skewed routing, unused experts, zero inputs, unrelated weight addresses and 2-byte alignment. Relative RMS error was 0.000167 against the independent model of the int8 arithmetic and 0.01595 against the FP32 reference. Activations and intermediate values are quantized to int8; this is not FP16-equivalent arithmetic.
- CUDA memcheck on the routed-layer test reported zero errors. 4K/8K routed-layer tests passed separately. These are not whole-model memcheck runs.
- Existing Q2 output hashes matched the original binary at 2K and 4K. All 46 existing device functions in the modified translation unit had unchanged disassembly after normalizing the compiler’s anonymous-namespace identifier; one new IQ1_M function was added.
- In the integration comparison, one off run and one on run missed the same two literal replacements. Both failures are retained in `edit-outputs.json`; installed production passed 2/2. The six follow-up novel-code tasks passed executable tests. This does not establish general coding-quality equivalence, and the full agentic coding harness was not rerun.
- Clean upstream exact-edit checks passed 2/2 with the flag off and 1/2 with it on. That failed answer is also retained. These small samples do not rule out a quality regression.
- First-token full-vocabulary KL is recorded in `summary.json`. Even repeated installed-production runs differed (KL 0.00394); adaptive routing/speculation were enabled. Warm follow-up KL includes differing adaptive-cache histories and is not an isolated kernel error measurement.
- Requesting an 8K chunk on the tuned integration build still fell back to 4K. The completed diagnostic is retained; redundant repeats were cancelled after it finished. No 8K full-model speedup is claimed.
- Not run: HIP compilation/hardware, Windows, other GPU architectures and a broad quality evaluation. The PR remains draft pending the shared CUDA/HIP source check; help with a HIP build and hardware regression check is requested. Production was not changed.

## Implementation and inspiration

IQ1_M weights are `d * odd_scale * (grid ± 1/8)`, with grid values -1, 0 and 1. Use the exact int8 codes `8*grid ± 1` and scale `d*odd_scale/8`, one scale per 16 values. The existing fused kernel then reads the compressed GGUF blocks directly. A fixed 128-row tile uses 53,760 bytes of shared memory; setup/occupancy failure preserves the fallback.

The investigation was prompted by [TensorFold’s CUDA recipes](https://github.com/ashhart/TensorFold/blob/d33dcfc767ee093393c6bfa00b3949bd7b4d8e88/docs/recipes/cuda.md). Its archived [tiled quantized prefill](https://github.com/ashhart/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/cuda/kernels/qmm_prefill.cu) and [expert-stage memory sizing](https://github.com/ashhart/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/cuda/experts_prefill.cu) illustrate the ideas investigated. This is an original IQ1_M adapter for Strata’s existing kernel, using GGML’s block layout/codebook; no TensorFold or Metal kernel was copied. [PR3277](https://github.com/Niko1221/Strata/pull/3277) covers the broader native-IQ dispatch on SM75, not this previously unsupported format.

## Reproduce

Base revision: `46c152a1310f77cd2f0ce2c7cd213962df14fe83`; pinned GGML `3cf03257f`. The clean build used CUDA architecture 75 with native experts, MMQ K-quants and the trellis overlay enabled. Compile and test:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_GGML_DIR=/path/to/ggml-checkout \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON \
  -DGGML_OPENMP=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=ON
cmake --build build --target strata prefill_fused_iq_test -j3
ctest --test-dir build -R prefill_fused_iq_test_iq1m --output-on-failure
STRATA_PF_FUSED_SM75=1 STRATA_PF_FUSED_IQ1_M=1 \
  build/prefill_fused_iq_test --only=IQ1_M --require-iq1 --no-ref --chunks=4096,8192
STRATA_PF_FUSED_SM75=1 STRATA_PF_FUSED_IQ1_M=1 \
  compute-sanitizer --tool memcheck --error-exitcode 1 \
  build/prefill_fused_iq_test --only=IQ1_M --require-iq1 --no-ref --chunks=256
```

Use the same binary and fresh server for each arm. Adapt `upstream-1-off.config.json` to local model/pack/profile paths; toggle only `STRATA_PF_FUSED_IQ1_M=0` / `1`. Keep `STRATA_PF_FUSED=1 STRATA_PF_FUSED_SM75=1` in both. From a v0.2.7 checkout with server dependencies installed, start `python -m serve.server --engine strata --config /path/to/config.json --host 127.0.0.1 --port 8080`, then:

```sh
python reproduce.py --url http://127.0.0.1:8080 --arm off --out runs.jsonl
```

Restart for each run and use off/on/on/off order. `reproduce.py` checks zero reused tokens, the 122,451-token fixture and exact AST equivalence. It records a failed answer before exiting with an error. Set `STRATA_API_KEY` in the environment if the local server requires authentication. The config files use `$HOME`/`$LAB` placeholders; replace those paths rather than passing them literally.

Raw per-request timings, token counts and quality outcomes are in `measurements.json`; full usual-suite records are in `speed-off.json` and `speed-on.json`. Model download checksum sidecars and the expert-profile seed hash are in `provenance.json`; the model repository revision was not recorded, and the profile itself is not included. A different profile can change timings. The local integration row requires the custom prefill-loan/cache patches and is secondary to the independently reproducible upstream comparison.
