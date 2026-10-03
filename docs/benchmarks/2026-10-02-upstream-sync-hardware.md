# Targeted checks after the 0.1.34 rebase

Upstream: `1678de333d0e0711bc414ad992b640e1a37dd814` (2026-10-02).

Tested contribution source: `2d241fee2a68c1729c56952efc5c89cc768fbec8`.

These are fresh build and regression checks after replaying the contribution on upstream main.
The larger fleet and context matrices dated 2026-10-01 remain measurements of 0.1.33.
They have not been relabeled as 0.1.34 results. This update does not repeat the full matrix.

The earlier contribution history is retained in tag `archive/pre-sync-20261002-gfx1012-community`.

## Completed checks

| Host | Check | Result |
|---|---|---|
| a5500 | `hardware-configure` | Pass |
| a5500 | `hardware-build` | Pass |
| a5500 | `hip_intrinsics` | Pass |
| a5500 | `hip_prefill_gemm` | Pass |
| a5500 | `hip_q2_zero` | Pass |
| a5500 | `native-expert-parity` | Pass |
| a5500 | `hardware-8k` | Pass |
| llm-79 | `hardware-configure` | Pass |
| llm-79 | `hardware-build` | Pass |
| llm-79 | `hip_intrinsics` | Pass |
| llm-79 | `hip_prefill_gemm` | Pass |
| llm-79 | `hip_q2_zero` | Pass |
| llm-79 | `native-expert-parity` | Pass |
| llm-79 | `hardware-old-new-identity` | Pass |

Builds use the exact exported commits and verified source/archive hashes, Release mode,
and the same pinned GGML revision and toolchains as the previous reports.
All model checks use GSQ-RCO IQ3_S and Q8 KV, text only; MTP uses window 4/threshold 0.5
for timing and threshold 0 for identity checks. Startup is excluded and the OS file cache
was not reset. Engines run privately through stdio.

The CMake conflict was resolved by retaining upstream architecture-list normalization
and adding `gfx1012` to the existing unvalidated list. Upstream Windows compiler flags
and runtime guards are preserved. Linux HIP 5.7.1/gfx1012 and HIP 7.15/gfx1100 were
built and exercised. Windows HIP was not tested.

The gfx1100 identity comparison is the previous hardware contribution versus the rebased
hardware contribution, eight requests per arm, checking tokens, authoritative state,
known answers and live-prefix reuse. It is not a fresh upstream-versus-patch comparison.

## Fresh short-request observations

Up to 512 output tokens; natural EOS. Different context allocations and output lengths
make these unsuitable for a direct comparison with the older long-output matrix.

| Host / run | MTP | Task | Input / allocation | Output | Prefill s | Output tok/s | Total s | Effective tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| a5500 / hardware-8k | on | coding | 8192 / 9728 | 125 | 102.15 | 15.34 | 110.30 | 1.13 |
| a5500 / hardware-8k | on | writing | 8192 / 9728 | 225 | 99.13 | 13.02 | 116.42 | 1.93 |

[Machine-readable evidence](2026-10-02-upstream-sync-hardware.json) includes build commands, source/binary hashes,
test log summaries, request settings and full timed outputs.
