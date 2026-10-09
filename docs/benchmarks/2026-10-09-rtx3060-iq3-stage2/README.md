# CUDA fused native prefill: IQ3-only two-stage buffering

## Change and scope

Build with **`-DSTRATA_CUDA_SM86_IQ3_STAGE2=ON -DCMAKE_CUDA_ARCHITECTURES=86`**.
This off-by-default option is declared only in the CUDA/native build branch and
selects a CUDA-only translation unit. The shared kernel/test files, HIP CMake
branch and SYCL build remain byte-identical to upstream. The optimized path also
requires runtime compute capability 8.6; other CUDA devices take the original
implementation when supported by the build. SM86 is broader than the RTX 3060;
performance evidence here covers that card only.

`STRATA_PF_FUSED=1 STRATA_PF_IQ3_STAGE2=1` uses two activation-buffer stages only
for **IQ3_S/IQ4_NL and IQ3_XXS/IQ4_NL gate/up–down pairs**. Both gate/up and down
take the selected family. Every other pair, and the default with the flag absent
or zero, keeps four stages. Shared-memory sizing, async-copy waits and launch
attributes use the selected stage count together. The grid retains the conservative
minimum occupancy across the applicable kernels.

Weight prefetch distance, tile selection, expert residency and decode are not
changed. This CUDA opt-in is independent of the one-superblock prefetch proposal;
the switches have not been tested together. HIP uses its existing kernels, and
SYCL builds a separate fused-prefill stub. No setup defaults are changed.

## Why select formats instead of changing every kernel?

An initial global two-stage sweep made IQ2_S/Q2_0 **4.76% slower at 2048 tokens and
5.81% slower at 8192**. This patch retains the original kernel for that pair and
for formats without relevant performance evidence. IQ3_S/IQ4_NL and IQ3_XXS/IQ4_NL
improved at all three measured sizes. Two stages still allowed only one resident
block/SM on this card; an occupancy increase is not the measured explanation.

## Measurement

2026-10-09, source base `fb58e0db`, engine 0.1.41, Linux, CUDA 13.4.92, SM86 Release:
RTX 3060 12 GB (28 SMs), **100 W cap**, i7-12700KF (AVX2), about 62 GiB RAM.
Native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS with its existing project control vector.

Baseline: original **opt-in fused native prefill**. Explicit chunk 4096,
expert-cache budget 1625 (2148 actual slots), full GPU int8 KV at 160K context,
11 workers, spec 4/min-p 0.70, no adaptation or prompt reuse, PCIe fraction zero.
Both arms used `STRATA_IQ_MT_MIN=1`, CPU prefill sharing off and stage pinning off.
Six alternating fresh-engine pairs per prompt, fixed 64-token warmup, greedy seed
3060, thinking off, 128 output tokens. Public-source prompts and per-pair evidence
are retained in `prompts.json` and `evidence.json`.

| Prompt | Input tokens | Median paired prompt speed change | Faster pairs | Paired time saved |
|---|---:|---:|---:|---:|
| Parser/test review | 12,247 | +0.199% | 5/6 | 31 ms |
| Runtime documentation | 20,324 | +0.304% | 5/6 | 75 ms |
| C++ prefill review | 30,757 | +0.239% | 5/6 | 89 ms |

All 18 pairs had identical output IDs and draft counts, and each arm was repeatable.
These are **prototype measurements** of the same selective policy, originally
`STRATA_EXP_PF_STAGES=iq3` in an experimental worktree with unrelated switches off.
The isolated PR instantiates two-stage kernels only for eligible kernel types;
the prototype also registered unused two-stage types. The isolated extraction's
parity and two-pair request smoke results are separate in `evidence.json`, not
pooled with this table. Ratios are medians of paired A/B prompt milliseconds.

Three alternating process-level pairs of the synthetic native expert-layer
benchmark, 512 experts/top-10 routing:

| Format pair | 2048 tokens | 4096 tokens | 8192 tokens |
|---|---:|---:|---:|
| IQ3_S / IQ4_NL | +17.02% | +6.07% | +15.35% |
| IQ3_XXS / IQ4_NL | +11.39% | +2.65% | +10.46% |

All 18 IQ3 shape/process comparisons were faster; hashes matched. The untouched
IQ2_S/Q2_0 and IQ2_XXS/Q2_0 shape medians stayed within about 0.4% of baseline,
avoiding the earlier multi-percent IQ2_S regressions. These timings include
grouping/preparation; their internal launch repeats are not independent samples.

## Noise and limits

Whole-engine gains are small: present this as targeted kernel tuning, not a large
inference speedup. A separate four-pair same-binary A/A control had prompt medians
-0.010%, -0.147%, -0.077% for parser/docs/C++ and individual outliers up to about
+/-1.5%. The rebuilt-disabled check preserved outputs but showed an unexplained
+7.02% first-prompt timing difference across executables. No outliers were removed
and no A/A bias was subtracted. Balanced-order subsets of the candidate tests had
positive medians, but this is still limited to one card/model/profile and three prompts.

No combined-prefetch/stage test, normal-MMQ comparison specific to this selector,
other-card/model claim, Windows measurement or new decode benefit is established.
The prototype's old global stages=2 result must not be attributed to this selector.

## CUDA-only isolation and validation

The initial shared-file draft was blocked on the absent ROCm toolchain. This
revision restores the original shared kernel and test, removing that HIP change.
Only the CUDA CMake branch selects the new source. No AMD performance or new HIP
runtime validation is claimed. Windows and other GPU hardware remain untested.

The CUDA-only kernel specialization shares conversion/routing helpers and its
fallback with the unchanged original translation unit. Its separate kernel body
must preserve the original arithmetic; the Python wrapper compares the stock
test's existing canonical fingerprints without editing the shared C++ test.

Default-off and opt-in builds are checked separately; enabling without architecture
86 or enabling both independent SM86 variants together is rejected. New-layout
validation is in `cuda-only-evidence.json`/`cuda-only-parity.log`; earlier isolated
evidence is retained separately as history.

## Reproduction

```sh
cmake -S . -B build-iq3-stage -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR=/path/to/llama.cpp -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_CUDA_SM86_IQ3_STAGE2=ON
cmake --build build-iq3-stage --target strata prefill_fused_iq_test -j 4
python tools/test_cuda_prefill_variant.py build-iq3-stage/prefill_fused_iq_test \
  --flag STRATA_PF_IQ3_STAGE2
```

The wrapper compares all six supported format pairs, both tile sizes, in fresh
off/on processes and requires the activation message. It also supports a prior
build via `--reference-binary`, used to compare against a default-off build in
`cuda-only-parity.log`. The earlier instrumented-build check is preserved in
`isolated-parity.log`. The executable retains its FP64/MMQ reference checks.
Exit 77 means no supported SM86 CUDA device and must not be counted as a pass.

Run `prefill_fused_iq_test --no-ref --chunks=2048,4096,8192` with the flag unset/set
in alternating processes for the fixed-work matrix. For inference, use the exact
`prompts.json` entries and `evidence.json` controls, fresh direct `StrataEngine`
instances per arm, the 64-token bicycle-explanation warmup, then prompts in file
order with temperature 0/seed 3060, thinking disabled and 128 output tokens.
Compare native DONE prompt timings, full output IDs and draft counts.
