# Additional upgrade and NUMA comparisons

Measured 2026-10-12 on a Lenovo ThinkStation P720: dual Quadro RTX 6000
24 GiB Turing GPUs (sm_75), dual Xeon Gold 6138 (40 physical cores total),
approximately 247 GiB OS-visible RAM after adding memory to socket 1.
Linux node 0 has approximately 157 GiB, node 1 approximately 90 GiB.
Both GPUs are attached to node 0. NVLink is present; its use was not measured.

Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic, NVIDIA driver 595.91.07,
CUDA 12.4 local source builds, architecture 75, Release build, existing ggml
dependency checkout reused. GPU power limit: 260 W each. Storage is NVMe for
model shards/packs. PCIe startup probe previously measured approximately
12.2 GB/s; instantaneous idle link generation must not be read as loaded speed.

Model: Unsloth Qwen3.8-Flash-Next UD-Q4_K_XL, four GGUF shards. Quantization is
mixed; the custom benchmark kept the existing native pack, MTP runtime and expert
profile. Model repository revision was not recorded. Expert-profile SHA256:
`8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.

See `config-sanitized.json` for settings: automatic 24/24-layer split across
the GPUs, 262144 context, int8 KV, 32768 resident KV cells, RAM PLE, 700 MiB VRAM
reserve, automatic expert cache, MTP/spec 4, CPU vision enabled. Context is the
configured maximum, not the largest successfully benchmarked input.

## Method and data

The standard upstream benchmark is in the top-level README/results.json.
These are additional custom comparisons, not substitutes for that suite.

Each custom run used a freshly launched engine, three short warmup requests
plus one ~5.5K warmup request (excluded), then 12 varied short prompts, three
~5.5K prompts, two ~21.7K prompts and one ~43.4K prompt. Output cap 256,
reasoning disabled; some responses stopped before the cap. Other sampling
parameters were omitted, using each server's defaults. Prompts are in
`bench_corpus.py`; long prompts are seeded random word lists, not realistic
source documents. There is no unique prefix tag in this custom corpus; actual
processed/reused counts are included so reuse can be inspected.

Hermes was paused during each experiment. No intentional concurrent requests or
synthetic memory/CPU stress tests were run. Other OS processes were not disabled.
The PLE source shard's clean page cache was evicted after shutdown using
POSIX_FADV_DONTNEED, so cached pages did not hide the NUMA policy under test.
Loading is excluded from scored request time. Placement and GPU state were
captured before and after each run, not continuously measured as peak values.

`upgrade/` and `numa/` contain summaries and individual run directories.
`results.json` is a list of request records: API times in seconds, token counts,
engine read/generation times in milliseconds, throughput in tokens/second,
draft acceptance counts, and checkpoint counts. `engine-log-excerpts.txt` contains
the engine lines associated with each benchmark request, with request labels. `placement.json` gives
large mapping sizes and node distribution in GiB; node meminfo values are KiB.
The API-derived decode field is secondary and is not used for reported medians.

To reproduce the custom corpus against a prepared running server, from its
Strata directory:

```bash
python /path/to/this/report/bench_corpus.py --out /path/to/run-results --server-out /dev/null
```

Reset the engine between arms and use the policies below, preserving the same
model/config. The script reads `strata-unsloth-ud-q4_k_xl.log` in the working
directory. `<STRATA>` and `<DATA>` in the configuration are path placeholders.

## Version A/B

Order: 0.1.40.3 / 0.1.43 / 0.1.43 / 0.1.40.3, two launches per version.
Interleave policy `numactl --interleave=0,1` for all four. Both arena and PLE
approximately 50/50 on both nodes. v0.1.43 source is tagged release commit
`7695859` (full hash in the standard report/source checkout); old local engine
BUILD.json recorded source `3950c1f221aa6e1b`.

| Combined median | 0.1.40.3 | 0.1.43 |
| --- | ---: | ---: |
| Short decode tok/s | 66.95 | 74.80 |
| ~5.5K prefill tok/s | 1053.8 | 1059.5 |
| ~21.7K prefill tok/s | 1604.25 | 1611.4 |
| ~43.4K prefill tok/s | 1786.5 | 1790.55 |
| Sum of scored request latency, both runs (s) | 221.388 | 210.003 |

Short decode improved 11.7%; prefill changes are within noise. This is a
release-default comparison. v0.1.43 enables STRATA_ROUTE_TAIL_SKIP=7, confirmed
in the startup log, and can change answers. Output identity and quality were
not measured; changes in output lengths and draft acceptance affect timings.
The 5.1% reduction in summed latency is specific to this corpus.

## NUMA after RAM upgrade (0.1.40.3)

Order: interleave / local memory / local CPU+memory / local CPU+memory /
local memory / interleave. Two launches per policy, on the upgraded RAM layout.

| Policy | Short decode median, run 1 / 2 | Combined median |
| --- | ---: | ---: |
| --interleave=0,1 | 64.95 / 64.55 | 64.70 |
| --membind=0 | 58.20 / 55.35 | 56.80 |
| --cpunodebind=0 --membind=0 | 60.00 / 63.25 | 62.15 |

Local arms actually placed the arena and PLE 100% on node 0; interleave placed
both approximately 50/50. Prefill was essentially unchanged. Local CPU binding
was verified by per-thread affinity. The experiment supports retaining
interleave on this workload, not a universal recommendation. It did not isolate
the cause of the slowdown or benchmark hardware memory bandwidth. The NUMA
comparison has not been repeated on 0.1.43.

## Correctness scope

Post-upgrade smoke checks: exact short text reply succeeded; a required tool call
produced get_temperature with city Paris; CPU vision recognized a blue solid
shape. No recall, coding-quality, or 128K/262K prompt test was run for this report.
