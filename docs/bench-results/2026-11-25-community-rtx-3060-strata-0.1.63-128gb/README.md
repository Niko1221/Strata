# RTX 3060 12 GB / Qwen3.8-Flash-Next Q2_0 — 128 GB RAM follow-up

Measured **2026-11-25 UTC**, same engine 0.1.63 CUDA as the [November 21 baseline](../2026-11-21-community-rtx-3060-strata-0.1.63-260k/). The 128 GB configuration processed **260,050 actual input tokens** at **1,161.7 prefill tok/s, 44.7 decode tok/s, 223.97 s TTFT**, with 2/2 dedicated synthetic recall checks. Minimum sampled available RAM at 262K rose from **10.70 to 71.91 GiB**. This is an observational follow-up, not a controlled RAM-only experiment.

## Actual long-input comparison (non-standard synthetic extension)

| Context | Actual speed input | Actual recall input | Prefill old → new tok/s | Decode old → new tok/s | TTFT old → new s | Decode change | Recall new |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64K | 64,044 | 63,262 | 1386.9 → **1387.1** | 49.6 → **52.5** | 46.23 → 46.22 | +5.85% | 2/2 |
| 128K | 128,030 | 128,030 | 1308.1 → **1309.8** | 47.8 → **48.8** | 97.95 → 97.82 | +2.09% | 2/2 |
| 262K | 260,050 | 259,889 | 1160.3 → **1161.7** | 42.8 → **44.7** | 224.25 → 223.97 | +4.44% | 2/2 |

Three speed repetitions and two dedicated recall requests per context. All 15 ended with `finish_reason=stop`, all 6 dedicated recall checks passed, no caps; all 15 synthetic markers found. The 32K actual-long-input workload was **not repeated** on November 25. Standard 32K configured context is a separate workload.

## Upstream standard protocol, 32K configured

| Metric | Nov 21 (~60 GiB usable) | Nov 25 (121.3 GiB usable) |
|---|---:|---:|
| 4K prefill tok/s | 780.3 | 779.9 |
| 16K prefill tok/s | 1378.0 | 1382.5 |
| Short decode tok/s | 47.7 | 49.2 |
| 4K decode tok/s | 46.3 | 46.2 |
| 16K decode tok/s | 46.8 | 47.6 |
| Short TTFT s | 1.63 | 1.60 |
| Sanity | 12/12 | 12/12 |
| Required requests finished or capped | 21/21 | 21/21 |

[New standard evidence](../2026-11-25-standard-rtx-3060-q2-0-32k-128gb/) · [Original standard evidence](../2026-11-21-standard-rtx-3060-q2-0-32k/). Protocol `strata-qwen38-thinking-v1-draft.1` uses prompts up to ~16K. 8/9 performance requests capped in the new run; optional concurrency had 6/12 capped and 0 failures. Concurrency aggregate: 41.28, 41.07, 40.67 tok/s (capacity 1, not four simultaneous decode slots).

## GPU and memory telemetry (new long-context runs)

| Context | Peak GPU °C | Peak W | Max VRAM MiB | Min available RAM old → new GiB | Max swap used KiB | Gen4 x4 samples |
|---|---:|---:|---:|---:|---:|---:|
| 64K | 71 | 168.66 | 11408 | 16.32 → **77.83** | 620 | 126/126 |
| 128K | 71 | 168.52 | 11406 | 10.91 → **72.37** | 620 | 253/253 |
| 262K | 71 | 169.27 | 11408 | 10.70 → **71.91** | 620 | 564/564 |

Observed sample extrema, not continuous measurements. `MemAvailable` includes reclaimable cache; this is not unused physical RAM or proof of zero disk I/O. Swap was 620 KiB in all sampled long runs, not literally zero. No sampled GPU temperature reached 75°C. PCIe link samples were all Gen4 x4.

## Configuration and reproducibility

- X1-255, Ryzen 7 255, 2 × 64 GB DDR5-5600 (operator-reported DIMM configuration); 121.3 GiB usable measured by standard runner. RTX 3060 12 GiB; NVIDIA 595.99.02 / CUDA 13.2; Ubuntu 26.04.1, kernel changed **7.0.0-43 → 7.0.0-44**. Same declared model revision and both standard GGUF file hashes as baseline. Base source revision missing in new standard run.
- Long runs use v3 synthetic methodology, thinking xhigh, requested drafting off, GPU Unlock ON, vision off, capacity 1, `kv=int8`, `kv_resident=32768`, `pool_workers=7`, `pcie_frac=0.10`. Tokenizer source and file hashes match the November 21 long manifests. [Existing v3 reproduction scripts](../2026-11-21-community-rtx-3060-strata-0.1.63-260k/reproduce/) are referenced, not re-supplied in this archive. The operator launched each matching context with tuning skipped.
- Raw manifests report the following runtime settings; cache allocation changed at 128K and 262K even though the main launch options were unchanged:

| Context | Conversation cache old → new MiB | Expert cache MiB | Prefill cap |
|---|---:|---:|---:|
| 64K | 8192 → 8192 | 6218 | 14080 |
| 128K | 7936 → 8192 | 6102 | 13568 |
| 262K | 7168 → 8192 | 5873 | 12288 |

- The manifest reports `drafting=off`; the raw timings nonetheless include nonzero `draft_n` and `draft_n_accepted`. Preserve both observations; the exact meaning of internal counters is not established here. Do not claim all speculation was disabled.
- Prefill is essentially unchanged in the long-input medians; decode is +2.1% to +5.8%. Kernel change, cache allocation, warm/cache state and sampling variation confound causal attribution. No Unlock-OFF comparison; synthetic key retrieval is not broad reasoning quality. Only three speed repetitions per context; no significance claim.
- No model weights, private host facts, raw engine logs, or private report paths published. Standard result artifacts are byte-preserved; the local launch plan alone was sanitized. No independent upstream leaderboard approval implied.

## Evidence

[64K](runs/64k/) · [128K](runs/128k/) · [262K](runs/262k/) · [performance.csv](performance.csv) · [GPU telemetry summary](gpu-telemetry-summary.csv) · [engine comparison](engine-comparison.csv) · [SHA256SUMS](SHA256SUMS).
