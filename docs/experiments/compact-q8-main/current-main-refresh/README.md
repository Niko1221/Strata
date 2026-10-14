# Compact Q8 fill: current-main refresh

Tested source `468f01f18d9277448f12d995af7ad2c4338c6fd1`, merged with upstream
`61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`. The result JSON records each actual
engine SHA-256, environment, arguments, complete output IDs and native statistics.

RTX PRO 6000 Blackwell 96 GB, Qwen3.8-Flash-Next Q8_0, FP16 KV, 8192 fresh input
tokens and 512 output tokens, mmap PLE, 56 GiB resident-expert budget, 15,472 GPU
experts, PCIE fraction 0.55, adaptive swaps disabled. Default-off compares main
with the refreshed candidate at MTP width 1. The compact-fill comparison uses
16 secondary-cache ways and MTP width 4, with only compact-fill off/on changing.
All function checks passed. Both compared pairs returned identical full token
streams. Inputs were fully read with zero prompt reuse.

| Arm | Decode tokens/s | Effective tokens/s |
|---|---:|---:|
| Main, default-off | 55.368 | 45.393 |
| Candidate, default-off | 56.762 | 46.353 |
| Candidate, width 1, cache off | 57.002 | 46.537 |
| Candidate, width 4, cache on, compact fill off | 109.301 | 76.396 |
| Candidate, width 4, cache on, compact fill on | 118.464 | 80.752 |

Each row is one observation in fixed order, with warm filesystem caches. These
are compatibility and exact-token checks with descriptive timings, not evidence
of a general performance gain. Width 1 and width 4 rows are different drafting
policies and should not be interpreted as the effect of compact fill alone.
Secondary expert copies consume extra VRAM; this remains opt-in.

CUDA sm86 engine builds, the read-only cache fixture and byte/tail-guard compact
fill microbenchmark passed on RTX 3070. Refreshed HIP gfx1012 and standalone SYCL
engines also built. Those builds do not establish GPU cache functionality on HIP
or SYCL: this cache/fill implementation is CUDA-only. The synthetic tests verify
component bytes; the Blackwell tests verify the whole model configuration above.

Reproduce with the preserved `../reproduce/case.py`, `memory_samples.py` and
`plan.json`, replacing local source/binary/model paths and running each labelled
case on an idle GPU. Copy all helper Python files together. The initial private
refresh omitted `memory_samples.py`; that setup failure was corrected and all
five cases rerun in a fresh directory. Original historical records remain
separate and retain their source/date. No production service was changed.
