# Draft-independent CPU IQ dispatch, 2026-10-10

The default CPU dispatch now uses the same IQ arithmetic for an expert group of one token or several tokens.
Previously, adding another draft that routed to the same expert switched kernels and changed rounding (#152).
This is a correctness tradeoff: it promotes the existing `STRATA_IQ_MT_MIN=1` behavior to the default, changes
some output relative to older releases, and is not a speed improvement on every CPU/format. Explicit
`STRATA_IQ_MT_MIN=2` restores the old dispatch and its dependence on group occupancy.

Base: upstream `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. No changes from #1748, #1773 or #1674 were added.
The regression runs through `native_gu_rows` and `native_down_rows` with synthetic quantized weights and
activations. It compares each token's rows bit for bit against its solo rows for widths 1 through 8, rotating
which tokens occupy the group. It does not skip the default dispatch or set the consistency override.

## Correctness and builds

- Unmodified CPU implementation plus the new regression: **7 failing format/projection checks** on a Xeon
  E5-2697 v3 (AVX2), including IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS gate/up and IQ4_NL down.
  See [baseline-xeon.log](baseline-xeon.log).
- Patched default: **zero differing rows** on that CPU, Ryzen 7 2700 (AVX2), and Ryzen 5 7600X3D (AVX-512).
  Default, forced AVX2 and ggml fallback pass. The compatibility `STRATA_IQ3S_MT1=1` setting passes too.
  Explicit threshold 2 fails again: 7 checks on AVX2, 6 on AVX-512. IQ4_XS gate/up already stays on ggml
  at every width on AVX-512 CPUs, so that check passes even with the legacy setting.
- CUDA sm_61 / GCC 12 / CUDA 12.0: built `strata`, `iq_avx2_parity`, `native_expert_parity`; four registered
  CPU CTests pass. Native GPU synthetic parity passes for all six gate/up formats paired with IQ4_NL down.
- HIP gfx1100 / ROCm compiler: the same targets, four CPU CTests and six synthetic GPU format pairs pass.
- SYCL / oneAPI 2026.1 icx+icpx: built `strata` and the newly registered `iq_avx2_parity`; three CPU CTests pass.
  The AVX-512 measurements below use this native SYCL-build CPU executable, not the Xeon executable.
  No SYCL GPU numerical test was run.

Per-host logs, binary hashes, CPU details, RAM and compiler paths are in the three subdirectories. `status.json`
records the build script's steps; `ctest.log` records the later test registration/configuration. CUDA's final
build time is incremental after rebuilding the changed source following the baseline build.

Reproduce the CPU checks with `cmake --build build --target iq_avx2_parity` and
`ctest --test-dir build --output-on-failure -R '^iq_avx2_parity'` (`STRATA_BUILD_TESTS=ON`, or
`STRATA_SYCL_PARITY=ON` for SYCL). The deliberate negative control is
`STRATA_IQ_MT_MIN=2 build/iq_avx2_parity`, expected to exit 1.

## Same-build expert timing

One thread pinned to CPU 2, synthetic 2560-wide inputs and 640-wide expert hidden states, approximately 128 MiB
of expert weights, widths 1/2/4. Each process measures three passes with method order rotated. Three separate
processes per setting alternate default/legacy order; no warm-up pass is excluded. The table is the median of
those three reported medians, gate/up plus down, milliseconds per expert at width 1. Positive change means
**slower**. All samples, including the slower initial samples, widths 2/4, and ranges derivable from them are
in [timings.json](timings.json) and the `bench-*.log` files. These are expert microbenchmarks, not decode rates.

| Gate/up (IQ4_NL down) | Xeon default / legacy ms | Change | Zen 4 default / legacy ms | Change |
|---|---:|---:|---:|---:|
| iq2_s | 0.565 / 0.557 | +1.4% | 0.248 / 0.250 | -0.8% |
| iq2_xs | 0.668 / 0.658 | +1.5% | 0.376 / 0.224 | +67.9% |
| iq2_xxs | 0.570 / 0.565 | +0.9% | 0.276 / 0.234 | +17.9% |
| iq3_s | 0.782 / 0.987 | -20.8% | 0.372 / 0.336 | +10.7% |
| iq3_xxs | 0.707 / 0.725 | -2.5% | 0.334 / 0.291 | +14.8% |
| iq4_xs | 0.465 / 0.453 | +2.6% | 0.167 / 0.165 | +1.2% |

The Xeon has about 252 GiB RAM; the Ryzen 7600X3D about 30 GiB. Kernel/compiler differences mean the two
machines should not be compared as a hardware ranking. The same executable on each machine supplies its own
control. This change deliberately favors consistent arithmetic despite the measured regressions; further
single-token kernel optimization would be separate work. The earlier documented Ryzen 7600 IQ3_S whole-model
cost of 1?3% is historical evidence, not a new measurement here.

[measure.py](measure.py) reproduces the dispatch checks and timing commands beside a `build/iq_avx2_parity`
executable. Source the oneAPI environment first for a SYCL-linked executable. No model or GPU is needed.

## Model check

Fourteen greedy requests, each capped at 128 generated tokens, all produced identical full token IDs:

- Two true single-token runs: `--spec 2 --mtp-max-t 1`, **zero drafts**. Native packs require `--spec >= 2`.
- Six default runs: `--spec 4 --mtp-max-t 4`, probability floors 1, 0 and 0.95, two repetitions in reversed
  case order. They offered 2, 117 and 84 drafts per request respectively. Floor 1 was not a zero-draft baseline.
- Six otherwise identical runs with explicit `STRATA_IQ_MT_MIN=1`.

Prompt: ?Write a Python function that merges overlapping intervals, explain it, and give tests.?
It is 28 input tokens with the chat template; IDs and all output IDs are retained in the JSON files.
Model: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`, with the second shard for PLE,
local native pack and MTP runtime. Model repository revision was not recorded. Three Tesla P4 GPUs,
layer boundaries 19/37, 27 CPU workers, int8 KV, 8192 context/resident tokens, 1024 prefill chunk.
Adaptive swaps are disabled by `--adapt-every 0`, PCIe share 0, prompt/conversation caches 0, suffix drafts 0.
Profile SHA256: `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.

The model manifests preserve the settings with the home directory anonymized; replace the local model/pack,
profile, MTP and shared-arena paths to reproduce. The external runner is
[`spec_window_parity.py` at adf9795](https://github.com/CC-David-CC/Strata-a5500/blob/adf979544232c18e6d90b27adf41a5ed4e9b5afe/tools/spec_window_parity.py):
`python spec_window_parity.py --manifest model-manifest.json --output NEW_DIRECTORY`, then the solo manifest.
The full-ID cross-check is recorded in [cross-arm.json](cross-arm.json).

These model runs are serial across windows (`--pipeline-windows 0`). They do not validate remote TCP overlap,
deep speculative pipelines, rollback state, sampled decoding, changing CPU/GPU placement, or every prompt/model.
The CPU fix removes the tested group-size-dependent dispatch; those other correctness requirements remain.
