# RTX 3060 12 GB / i5-11400F / 31.8 GiB RAM: Strata Coder IQ1_M

**Community data: 2026-10-12, LightWarden Aether Lab.** Source code Strata **v0.1.43** (`7695859aa13afb248bdb4d36f9b2e95707db079d`), Windows 11 build 26200, NVIDIA driver 610.88, model `qwen3.8-flash-next-coder-iq1_m`. All numbers below are from this one host. **This is not a Strata engine fork.**

## A. Standard community speed suite: stock 32,768-token config
Unmodified upstream `tools/strata_bench.py`, three repeats, one running localhost Strata server, thinking off. Raw measurements in `results.json`.

| Measurement | Median | Sample |
| --- | ---: | --- |
| Short-chat decode | **26.0 tok/s** (21.9–30.7) | 9 requests |
| Fresh 4K prompt prefill | **717.0 tok/s** | 3 requests |
| Fresh 16K prompt prefill | **904.5 tok/s** | 3 requests |
| Follow-up agent-turn decode | **28.2 tok/s** | multi-turn suite |
| Agent prefix reuse | **68.759%** | as recorded in earlier authored benchmark summary |

33/33 recorded stock requests succeeded, with no skips. See `results.json`. Long option `--long` targeting 32,768 input would exceed the model's *total* 32,768-token context with answer margin; it was not run. All measured ranges and exact methods are in the original raw JSON.

## B. Optional non-stock 32K extensions (NOT merged into stock summary)
- 31,294 actual fresh input tokens, **908.4 tok/s median prompt throughput**, **34.56s median first-token latency**, three runs with zero prompt reuse (`extended-results.json`). This is *near-32K*, not a full 32,768 input.
- Three 256-generated-token short-sequence requests: **57.2 tok/s median**, 45.4–61.3 range. 192/192 draft tokens were accepted each time. This predictable sequence should **not** be presented as generic chat speed.
- Built-in needle retrieval near 30.5K: **3/3** found at 10%, 50%, 90% insertion depths, one attempt per depth (`needles-30k.json`). Does not assess detailed reasoning quality.
- Exact custom extension runner is included as `run_extended_bench.py` for transparency.

## C. Separate experimental 64K Budget12 profile (single-case diagnostics)
**NOT stock benchmark results.** A privately preserved 65,536-context configuration kept INT8 GPU-resident KV and auto expert cache, and replaced full resident experts with `--resident-budget-gib 12`. Full verified flag values, sanitized JSON and reproduction instructions are included as `budget12-sanitized-config.json`, `budget12-REPRODUCE.md`, and `64k-budget12-ADDENDUM.md`. The JSON still needs local paths before use. 2,152 observed expert slots. This was a source-unaltered configuration variant, not a fork.

| Test | Input tokens | Completion | Wall | Accuracy/qualification |
| --- | ---: | --- | ---: | --- |
| 64A full-resident NONE | 53,942 | `stop` | 120.00s | 9/10 manual review; 451 MiB free system RAM after |
| 64B full-resident LOW | not completed | safety stop | N/A | not graded; low-memory abort |
| 64C Budget12 NONE | 53,942 | `stop` | 152.25s | 8/10 manual review |
| 64D Budget12 LOW | 53,970 | `stop` | 243.26s | 9/10 manual review |
| 64E Budget12 MEDIUM | 53,940 | `stop` | 243.41s | 8/10 manual review |
| 64F Budget12 HIGH | 53,982 | `stop` | 496.54s | 10/10 manual review |
| 64G HIGH, fresh synthetic | 53,891 | `stop` | 517.44s | **12/13** strict graded; missed a canonical $39.96 refund |
| 64H HIGH, fresh synthetic | 53,983 | `length` | 531.44s | **ACCURACY NOT EVALUABLE**, no answer (`length`) |
| 64I HIGH, think cap 6144 | 53,990 | `stop` | 456.97s | **12/18**, six field misses, notably Invoice C |
| 64J HIGH, think cap 6144, provenance-first | 53,894 | `stop` | 622.42s | **24/24**, 18 primary + 6 evidence fields, one trial |

**Do not interpret test-count fractions as confidence intervals or universal accuracy.** 64I and 64J used different random cases and schema changes; the improved 64J result is not a controlled causal comparison. 64J's prompt prefill slowed to 349.2 tok/s from 64I's 833.5 without identified cause. Public aggregate metadata are provided in `64k-budget12-observed-metrics.csv`; this is not an unmodified upstream benchmark schema. Full private synthetic grading packets are not part of this contribution.

## Reproduction and disclosure notes
- RTX 3060 12GB, i5-11400F, 31.8GiB RAM, Windows 11, Python 3.14.7, NVIDIA 610.88.
- Upstream Strata v0.1.43 / original checkout `7695859aa13afb248bdb4d36f9b2e95707db079d`; Coder IQ1_M model. **Weights, executable and unsanitized local config are NOT bundled.** An exact-flags/path-redacted sample and local verification receipt ARE bundled.
- Isolated 64K trial hash `9F5DB7BC7065A18E8870DC503B7F8F60D695C24FF50883FDCF9AE2A3FF6C55EC`. The original (1,753-byte) config was SHA-256-verified locally, then exported to the included `budget12-sanitized-config.json` with paths replaced; the exported file matches the verified receipt. **This is not a ready-to-run config:** restore paths to your own upstream installation, never use the original host's paths.
- The following safety floors were enforced by **our evaluation runner**, not by Strata itself. On this host, memory-gated one-shot tests required **6,144 MiB RAM and 256 MiB free VRAM** preflight; in-run sampled abort thresholds **1,024 MiB RAM and 128 MiB VRAM**. The ~10-second sampler cannot guarantee avoiding all transient OOMs.
- Background graphical applications can consume the remaining VRAM. A closed browser/Discord was reported before one 32K benchmark; extension conditions were not independently verified across all runs.
- Pagefile size, disk class, clock/power states, and independent cross-machine replication are unknown. Upstream setup recommended pagefile changes; do not imply they were completed.
- All sensitive local paths and personal account identifiers intentionally excluded from this public PR folder. Please review manually before sharing.

Upstream guide: https://github.com/Niko1221/Strata/blob/main/docs/COMMUNITY_BENCHMARKS.md
