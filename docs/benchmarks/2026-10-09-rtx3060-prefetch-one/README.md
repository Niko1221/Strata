# CUDA fused native prefill: one-superblock lookahead

## Change and scope

Build with **`-DSTRATA_CUDA_SM86_PREFETCH_ONE=ON -DCMAKE_CUDA_ARCHITECTURES=86`**.
The CMake option is off by default and exists only in the CUDA/native build branch.
It substitutes a CUDA-only translation unit; the shared `moe_fused_iq.cu` and
shared test file are byte-identical to upstream. The HIP CMake branch and SYCL
build are unchanged. A runtime capability check permits the optimized kernel only
on compute capability 8.6; other CUDA devices use the included original implementation
(provided the build targets their architecture). SM86 includes other Ampere cards;
the performance measurements here are specifically from the RTX 3060.

`STRATA_PF_FUSED=1 STRATA_PF_PREFETCH_ONE=1` selects a one-superblock gate/up weight
prefetch distance. The existing two-superblock kernel is the default. Activation
pipeline depth, tile selection, down-weight prefetch, expert residency and decode
are not changed. The runtime flag has no effect in default builds. HIP keeps its existing path. SYCL builds
a separate fused-prefill stub and does not compile this implementation.

This is independent of the IQ3 two-stage proposal. The two switches have not been
tested together. No setup policy or normal model configuration is changed.

## Measurement

2026-10-09, source base `fb58e0db`, engine 0.1.41, Linux, CUDA 13.4.92, SM86 Release:
RTX 3060 12 GB (28 SMs), **100 W cap**, i7-12700KF (AVX2), about 62 GiB RAM.
Native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS with the existing project control vector.

The measured baseline is the existing **opt-in fused native path**. Controls:
4096-token prefill, expert-cache budget 1625 (2148 actual slots), full GPU int8 KV
at 160K logical context, 11 workers, `--spec 4 --spec-min-p 0.70`, prompt cache and
adaptation off, PCIe expert fraction zero. Both arms used `STRATA_IQ_MT_MIN=1`,
`STRATA_PREFILL_CPU_SHARE=0`, and `STRATA_STAGE_PIN=0`.

Six alternating fresh-engine pairs per realistic prompt, greedy seed 3060, a fixed
64-token warmup, and 128 output tokens. Exact public-source prompts are in
`prompts.json`; source hashes, native timing pairs and output hashes are in
`evidence.json`. Prompt speed is the median of A/B prompt-millisecond ratios,
not a ratio of arm medians. No outliers are removed.

| Prompt | Input tokens | Median paired prompt speed change | Faster pairs | Paired time saved |
|---|---:|---:|---:|---:|
| Parser/test review | 12,247 | +0.548% | 5/6 | 86 ms |
| Runtime documentation | 20,324 | +0.813% | 6/6 | 198 ms |
| C++ prefill review | 30,757 | +0.802% | 6/6 | 297 ms |

All 18 request pairs had identical output token IDs and draft counts; each arm was
repeatable. This table is **prototype evidence** from the same prefetch policy,
originally selected with `STRATA_EXP_PF_PREFETCH=1` in an experimental worktree.
It included disabled unrelated experiments. The isolated PR instantiates PF=1 only
for gate/up; down continues to use the original PF=2 instance (its generated
prefetch operations are independent of that value). Final isolated parity and
two-pair request smoke results are separately recorded in `evidence.json`; they
are not pooled into the six-pair measurements above.

Three independent process pairs of the fixed-work native expert-layer benchmark
(512 experts, top-10 routing) reproduced gains in all 36 shape/process comparisons:

| Format pair | 2048 tokens | 4096 tokens | 8192 tokens |
|---|---:|---:|---:|
| IQ2_S / Q2_0 | +3.96% | +1.71% | +2.75% |
| IQ2_XXS / Q2_0 | +3.28% | +2.90% | +2.14% |
| IQ3_S / IQ4_NL | +6.52% | +2.91% | +4.32% |
| IQ3_XXS / IQ4_NL | +5.69% | +2.69% | +4.02% |

Output hashes matched in all comparisons. The kernel benchmark includes grouping
and preparation; internal launch repetitions are not independent process pairs.
An earlier overnight series independently found +0.33–0.36% whole-engine prompt
gains at synthetic 12K/40K lengths; realistic and synthetic results are not pooled.

## Noise and limits

A later same-binary A/A control had prompt medians -0.010%, -0.147%, -0.077% for
parser/docs/C++ respectively, with individual outliers up to about +/-1.5%. The
rebuilt-disabled numerical check also showed an unexplained +7.02% first-prompt
timing difference across executables. This is why the candidate timing comparison
uses one binary and six balanced pairs; the claim is modest and hardware-specific.

Switching normal MMQ to fused+prefetch was 8.3–10.1% faster on these inputs, but
that includes the existing fused implementation and changed two prompts' output
IDs. It is **not** the gain from this patch or proof of quality equivalence with MMQ.
No decode improvement, default-setting change, other-GPU/model benefit, full-160K
prompt benefit, or Windows performance is established here.

## CUDA-only isolation and validation

The first draft edited a shared CUDA/HIP file and was blocked on a missing ROCm
toolchain. The current revision restores that file and the shared test exactly to
upstream. It adds only CUDA-specific source selected inside the CUDA CMake branch;
there is no longer a HIP implementation change to build. No AMD speedup is claimed.
Windows and other GPU hardware have not been tested.

The specialized kernel body lives in `src/prefill/cuda/native_fused_iq_sm86.cuh`;
conversion, routing and fallback code are reused by including the unchanged
original translation unit under fallback entry-point names. The tradeoff is a
separate SM86 kernel body whose arithmetic must stay in sync with the original.
The parity wrapper compares the existing test's canonical output fingerprints;
it no longer requires any changes to the shared C++ test.

Default-off and enabled CUDA builds are checked separately. Configuration rejects
the opt-in without architecture 86, or both independent SM86 variants enabled
together. `cuda-only-evidence.json` and `cuda-only-parity.log` record validation
of this source layout; the earlier `evidence.json`/`isolated-parity.log` are retained
as history, not represented as tests of the new layout.

## Reproduction

CUDA build (use the pinned llama.cpp checkout installed by setup):

```sh
cmake -S . -B build-prefetch -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR=/path/to/llama.cpp -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_CUDA_SM86_PREFETCH_ONE=ON
cmake --build build-prefetch --target strata prefill_fused_iq_test -j 4
python tools/test_cuda_prefill_variant.py build-prefetch/prefill_fused_iq_test \
  --flag STRATA_PF_PREFETCH_ONE
```

The wrapper launches fresh off/on processes for both 64/128-row tiles, checks the
opt-in activation message, and compares complete output fingerprints for all six
format pairs. The executable also retains its FP64/MMQ numerical checks. It exits
77 without a supported SM86 CUDA device; that is a skip, not a pass. The CUDA-only
validation additionally compares disabled output to a build with the CMake option
off via `--reference-binary`; see `cuda-only-parity.log`. The earlier isolated
validation used a prior instrumented build and is retained in `isolated-parity.log`.

For fixed-work timings, run `prefill_fused_iq_test --no-ref --chunks=2048,4096,8192`
in alternating fresh processes with this flag unset/set to 1. Do not combine
reference-test runtime with CUDA event timings. For inference reproduction, use the
exact `prompts.json` entries with the controls in `evidence.json`, a fresh direct
`StrataEngine` per arm, the same 64-token bicycle-explanation warmup, then each
prompt in file order, temperature 0/seed 3060, thinking disabled and 128 output
tokens. Compare the engine's native DONE prompt timings and full output IDs.
