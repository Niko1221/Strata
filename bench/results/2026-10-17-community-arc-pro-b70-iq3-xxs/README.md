# Community benchmark on Intel Arc Pro B70

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges), plus the upstream needle test. Single GPU, native SYCL engine in an unprivileged Linux container on a shared Proxmox host. Speeds are the engine's own timings; first-token times are measured locally over HTTP. Recall: 6/6; this is not a general model-quality evaluation.

## Hardware and software

- GPU: one Intel Arc Pro B70, nominal 32 GB VRAM. The automatic collector did not read total VRAM on SYCL; startup cache/headroom figures are in `engine-log-excerpt.txt`.
- CPU: AMD Ryzen 9 PRO 3900, physical 12 cores / 24 threads; the container is allocated 16 logical CPUs (not a 16-thread physical processor).
- RAM: host MemTotal 65,730,400 KiB (62.69 GiB usable); container limit 40 GiB RAM / 10 GiB swap. Strata service MemoryMax 32 GiB, MemorySwapMax 0, unlimited memlock. Driver-pinned host memory is not fully reflected in service cgroup accounting.
- Storage: dedicated local NVMe for model/pack/draft files; PLE uses direct SSD reads rather than pinning the entire lookup table in RAM.
- OS: Ubuntu 26.04.1 LTS container on Proxmox VE 9.2.21, shared kernel `7.0.14-24-pve`; GPU render-device access through LXC, not a VM or Docker.
- GPU kernel driver: `xe`; `intel-opencl-icd` and `libze-intel-gpu1` 26.05.37020.3-1, Level Zero loader `libze1` 1.28.2-2. CUDA/ROCm: not applicable. Native oneAPI 2026.1.1 build, Release AOT `bmg-g31`.
- Strata: project/engine 0.1.45, commit `477ac13cbe8e582c20484871ce459d20d82df09d`; binary and profile hashes in `community-build-facts.json`.
- Background load: other homelab guests remained running; not an isolated performance host. Host load averages around the start were 0.98 / 1.56 / 2.53. GPU power limit, clocks and inference memory peaks were not recorded for this run.
- PCIe: a separate earlier H2D probe measured 12.9 GB/s; not remeasured during this suite. Endpoint link registers on this card reported an implausible Gen1 x1 despite that throughput, so no negotiated link generation/width is claimed here.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Repository: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. Both GGUF shard filenames appear in the command below. Native pack produced by upstream `tools/iq_pack.py`; no duplicate experts.bin. Shipped `data/expert-profile-sycl.bin`, not a custom calibration.
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (sycl backend); thinking off for the suite; temperature 0
- MTP draft: upstream packed draft at `rt`; startup reports 835 MiB VRAM and a 106,299-token draft head. Vision off. No experimental speed projection or local engine source changes. The launch flag is `--spec 4`; the engine reports a verify window of 6 and `mtp_max:4`, not a `--spec 6` launch.

```text
strata --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile-sycl.bin --expert-cache auto --stream-experts --prefill 4096 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --vram-reserve-mib 2048 --ple-io direct --kv-resident 32768 --prefill-borrow
```

Environment of the config: `SYCL_CACHE_PERSISTENT=0`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_VERIFY_NO_HOST=1`, `STRATA_STAGER_THREADS=12`, `STRATA_MIRROR_MIB=22528`, `STRATA_TRACE=1`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- 32K prompt: same calibrated fresh-word-list method, 24 new tokens, 3 runs. Labels are nominal sizes; actual counts are in the results table.
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- Concurrency: not tested; one parallel slot. Decode is engine generated-tokens/time, not tokens divided by total request latency. Total client latencies and token counts are retained per request in `results.json`.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not benchmarked (short input) | 42.8 (36.3-55.9) | 0.85 (0.81-0.91) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1349.3 (1291.8-1349.6) | not benchmarked (short output) | 3.01 (3.01-3.14) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1446.2 (1445.3-1447.4) | not benchmarked (short output) | 11.14 (11.13-11.15) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1406.8 (1405.9-1407.4) | not benchmarked (short output) | 22.84 (22.82-22.85) |
| agent session, follow-up turns | 14611 (14590-14618) at the last turn | see below | up to 64 | 3 | see below | 51.8 (38.7-62.0) | 3.46 (2.44-3.86) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.7-68.9); earlier-turn tokens read again per session: 3 (3-73) (earlier turns that the cache did not hold; the 13717 (13717-13717) new tokens of the files are read once); first turn (nothing cached) first token: 2.01 s (1.97-2.01).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 314.1 | 1.972 | 48.7 |
| 1 | 3775 | 655 | 3120 | 858.1 | 3.674 | 52.8 |
| 2 | 5308 | 3839 | 1469 | 582.5 | 2.559 | 49.8 |
| 3 | 8741 | 5371 | 3370 | 882.6 | 3.858 | 49.6 |
| 4 | 11655 | 8793 | 2862 | 810.0 | 3.572 | 55.7 |
| 5 | 14611 | 11712 | 2899 | 816.7 | 3.59 | 42.4 |

Per-request data: `results.json` (upstream suite 1 schema). `engine-log-excerpt.txt` is a manually selected startup-only excerpt from the matching 256K session, not the tool's historical-log extraction. The service was already running; its idle-unloaded model was reloaded by an arithmetic/tool smoke check before the suite, outside measured timings.

## Correctness and limitations

Before the speed suite, a streamed arithmetic and tool round-trip smoke test passed: 6 times 7 returned 42; a forced synthetic read_file tool call and supplied result produced the exact final answer. It accessed no real project files.

Upstream `tools/needle_bench.py`, one request at each length/depth, thinking disabled by that tool:

| Requested length | Depth | Actual prompt tokens | Found | Total client seconds |
| --- | ---: | ---: | --- | ---: |
| 32k | 10% | 32792 | yes | 32.1 |
| 32k | 50% | 32792 | yes | 31.5 |
| 32k | 90% | 32793 | yes | 31.5 |
| 128k | 10% | 127285 | yes | 139.3 |
| 128k | 50% | 127283 | yes | 139.5 |
| 128k | 90% | 127283 | yes | 74.4 |

Elapsed times (rounded to 0.1 seconds by the tool) and synthetic answers are in `needles.json`. Needle requests reuse the same repository-text haystack, so prefix caching can affect their times (the tool does not record reused tokens). These are recall checks, not fresh-prefill speed measurements. No all-256K recall or general reasoning/coding-quality claim is made by this submission.

The speed suite has no failed or skipped cases. The conversation uses the model's own replies, so token counts differ a little between runs. Short decode runs are capped at 100 tokens and depend on text/draft acceptance; do not generalize the pooled chat median to every workload. Total time of the speed suite: 224 s, excluding model loading and the separate recall checks. Automatic hardware fields in `results.json` are preserved as emitted; its 40 GiB RAM / 16-thread values describe the container, corrected above. No peak memory/power measurements were taken in this suite. See `REPRODUCE.md` for commands and privacy notes.
