# gfx906 guarded decode top-k — 2026-10-08

## Change and scope

Opt-in `STRATA_GFX906_TOPK_REG=2` extends the existing register-66 top-k kernel to uncounted gfx906 decode windows with at most eight queries. Capacities fitting the old register-33 route are unchanged; capacities exceeding register-66 reach retain reference fallback.

For intermediate capacities, two complementary device-side guards select one body for the whole window: the final query's current `kStepNKv <= 24576` chooses the reference256 kernel, otherwise register66. Both guards run before barriers and writes. Device steps are read on execution, including graph replay after context growth or a conversation restore.

Default dispatch, counted/prompt paths and non-gfx906 dispatch are unchanged. The implementation is three source/test/CMake files; no Q2 expert changes, CPU AVX2 changes, performance presets, or unrelated parity-test work are included.

## Base and build prerequisite

- Upstream base: `6674a0065fb96bacde33e3eb10f91a1df86f95f2` (0.1.40.4).
- Tested top-k patch SHA256: `744cdfc5be9ae9efbb7baedf49c18a474f23e62883599e6cf8a9381dac41f012`.
- Code commit: `fc23f47512191df7b5f380d7896ecccf1b0619ab`; its tree is byte-identical to the locally prepared code tree.
- Pinned image: `sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646` (community gfx906 HIP image).
- ggml/llama.cpp: `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- Baseline engine SHA256: `5629aff89bce9385e8ec893d8a1f94d59b4cc5f2ad315ab4a3a61cd5d678d236`.
- Candidate engine SHA256: `93a4ccc72895abbabfa65e1a7158b5925c80fb7fb0f3c99c6a730f301c9b6257`.

The unmodified base fails a full gfx906 engine build because its compatibility header lacks `cudaEventBlockingSync`. BOTH tested arms include the same one-line alias `#define cudaEventBlockingSync hipEventBlockingSync` from existing [PR #1396](https://github.com/Niko1221/Strata/pull/1396). This prerequisite is not part of the proposed top-k source changes. Its local patch SHA256 is `198be9ad4f6b00b4f3e7150ce6d7cb278996cfb90338548b904befad4d9594b6`. The separate Q6_K change in that PR was not needed or applied.

Both full engine and parity-target builds completed successfully. Build options are reproduced below; the original host-specific command manifests are retained privately.

## Correctness and component checks

On each of two gfx906 16-GiB cards:
- Registered `qsa_topk_parity` and `qsa_topk_gfx906_guarded` CTests passed.
- Mode unset/equivalent0 and mode2 selftests passed.
- Explicit windows1/6/8/9 at contexts4096/24575/24576/24577/65536/200000, capacity204800.
- Existing `STRATA_TOPK_OLD=1` override, capacity524288 fallback and counted/prompt regression.
- Continuous, tied, equal and adversarial score inputs, output-tail sentinels.
- Captured-graph replay at4096 →24575 →24576 →24577 →65536 →131072 →200000 →204800 →4096 where the test's graph-case conditions apply.

Sixty driver cases completed with exit0, including two CTest invocations (two registered tests each). Logs and portable test arguments are in component-checks.json. This count is driver cases, not sixty separate CTests.

Important harness detail: `STRATA_TOPK_OLD` is presence-based; setting it to0 still forces the reference. An initial manual matrix accidentally set0, so it is excluded from this evidence. The recorded final matrix leaves it UNSET except for the explicit override case. The model benchmark never set this variable. The pinned image defines neither TOPK_OLD nor GFX906_TOPK_REG.

Illustrative ordinary-launch timing, GPU0, six queries,100 repetitions, capacity204800:
- actual4096: reference0.089ms, dispatched0.088ms
- actual65536: reference0.275ms, dispatched0.189ms
- actual200000: reference0.555ms, dispatched0.462ms

These rounded component timings are not graph timings, independent repeated-trial estimates, or end-to-end speedups. Graph runs check correctness.

## Isolated model A/B

Hardware: two gfx90616-GiB cards, Xeon E5-2698B v3, about128GiB RAM. Both arms use stock HIGH performance mode with unchanged190W caps, the same pinned image/model/runtime, MTP16384, INT8 KV32768 resident tokens, target capacity204800, fixed expert placement9900+9178, later reserve623MiB, prefill4096, spec4 with suffix lookup, and identical sampling/seeds.

Model: Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0. Both arms use the same existing MMVF/SwiGLU/GDN settings and mode7 Q2 baseline path; CPU Q2 spread is off. A is upstream plus the common build alias, B adds this top-k patch with mode2. No power, MTP-window, cache-size or quantization change is mixed into the comparison.

Each workload uses fresh processes in ABBA order,1024 generated tokens per run, no prompt reuse, deterministic greedy sampling. Cells below average the two observations per arm.

| Workload / prompt tokens | A decode tok/s | B decode tok/s | Change | A prefill tok/s | B prefill tok/s |
|---|---:|---:|---:|---:|---:|
| code 4096 | 53.525 | 53.428 | -0.181% | 347.992 | 347.987 |
| code 65536 | 47.730 | 48.295 | +1.183% | 598.282 | 598.305 |
| ru 65536 | 49.747 | 50.447 | +1.408% | 597.921 | 598.007 |

All four output-ID sequences are identical within each workload (hashes in model-analysis.json). Both long-context B measurements exceed both A measurements. Four-K decode was0.181% slower, consistent with the small added guarded-launch cost; the sample is too small for a robust general regression estimate. Prefill is effectively unchanged. Native full cold-prompt-plus-output time changed+0.113% at4K,−0.195% code64K and−0.232% Russian64K.

This is a small hardware/model-specific result, not a general quality claim or a promise of uniform gains. It is deliberately opt-in. No actual200K model request was run for this isolated A/B; component tests do cover that context. CUDA and non-gfx906 HIP builds/runtime were not run in this environment. Static review found and fixed a test capture-mode portability issue by using the already-supported ThreadLocal constant.

## Reproducing focused checks

After applying the shared build prerequisite to this pinned base, use the pinned dependency and image above:

```sh
cmake -S . -B build-906 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_HIP_GFX906=ON -DSTRATA_PORTABLE=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DSTRATA_GGML_DIR=/src/third_party/llama.cpp
cmake --build build-906 --target strata qsa_topk_parity -j4
unset STRATA_TOPK_OLD
HIP_VISIBLE_DEVICES=0 ctest --test-dir build-906 --output-on-failure -R '^qsa_topk_(parity|gfx906_guarded)$'
HIP_VISIBLE_DEVICES=1 ctest --test-dir build-906 --output-on-failure -R '^qsa_topk_(parity|gfx906_guarded)$'
HIP_VISIBLE_DEVICES=0 STRATA_GFX906_TOPK_REG=2 build-906/qsa_topk_parity 65536 6 100 204800 0
```

Repeat explicit matrix commands using component-checks.json. Host-specific command manifests, model paths, source-fixture metadata and token fixtures are retained privately and are not published here. The component suite is self-contained; exact model replay requires the locally retained fixtures and model assets.

Related but excluded: existing [#1187](https://github.com/Niko1221/Strata/pull/1187) Q2 GPU foundation and [#1320](https://github.com/Niko1221/Strata/pull/1320) broader gfx906 parity work.
