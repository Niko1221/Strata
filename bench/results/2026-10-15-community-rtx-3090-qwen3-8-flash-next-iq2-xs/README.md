# Community benchmark on NVIDIA GeForce RTX 3090

Measured on 2026-10-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed, plus a six-case long-context recall check.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090, 24 GiB VRAM; PCIe Gen4 x16; power limit 420 W (381 W observed during the suite); maximum SM clock 2100 MHz
- CPU: 12th Gen Intel(R) Core(TM) i7-12700KF (12 cores, 20 threads); installed RAM: 30.4 GiB; storage: NVMe Fanxiang S500Pro 512 GB, xfs
- OS: Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic; GPU driver: 595.99.02; CUDA: toolkit 12.9 used for the build (the driver reports CUDA 13.2)
- Strata: project 0.1.44, commit ed4773eb; engine 0.1.43.1, local source build for sm_86 (`engine/BUILD.json`: `source: local`, `archs: [86]`, vision gpu; that file still records 0.1.41 for this binary)
- Background workloads and power limits: no other workloads during the run; no undervolt, the 420 W limit is the card's default

## Model and configuration

- Model repository and revision: not recorded by the installer; GGUF shards `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` (39,225,954,592 B) and `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` (28,800,138,432 B)
- Quantization: IQ2_XS; vision encoder: yes, `--vision` with `mmproj-Qwen3.8-Flash-Next-BF16.gguf`; packs: native-experts pack built by `tools/iq_pack.py` (index v3, 1079 tensors, 35,454,976,000 B, `native_experts_sha256` 48df605acb89c7e8c3176a51cec777c2ed344ea439e8f3a37e5ce4f9dcb6946c), `expert-profile.bin`, MTP draft `rt`
- Context: 231,072 tokens; KV int8; expert cache auto with resident experts; conversation cache 2048 MiB over 4 slots (min free 1024 MiB); prefill auto; low-RAM mode: not used; parallel slots: 1
- MTP: `rt`; reasoning: off for the suite; sampling: temperature 0; calibration: only the suite's token-count calibration request (about 1.08 tokens per word); experimental speed projection: `--spec 4 --spec-min-p 0.5`

```text
strata --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 231072 --kv int8 --resident-experts --conversation-cache-mib 2048 --conversation-cache-slots 4 --conversation-cache-min-free-mib 1024 --vision --vram-reserve-mib 832
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 112.5 (98.4-142.2) | 0.24 (0.20-0.28) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2145.1 (2135.3-2147.1) | not measured | 1.89 (1.89-1.90) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2632.8 (2632.2-2646.9) | not measured | 6.12 (6.09-6.12) |
| agent session, follow-up turns | 14568 (14560-14575) at the last turn | see below | up to 64 | 3 | see below | 121.8 (92.5-142.9) | 1.68 (1.21-2.17) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.8); earlier-turn tokens read again per session: 3 (3-3) (earlier turns that the cache did not hold; the 13728 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 0.82 s (0.76-1.09).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 816.2 | 0.759 | 71.8 |
| 1 | 3741 | 621 | 3120 | 1870.1 | 1.683 | 92.5 |
| 2 | 5274 | 3804 | 1470 | 1001.0 | 1.486 | 116.9 |
| 3 | 8718 | 5338 | 3380 | 1571.7 | 2.17 | 99.0 |
| 4 | 11622 | 8760 | 2862 | 1434.9 | 2.015 | 136.8 |
| 5 | 14560 | 11661 | 2899 | 1501.0 | 1.952 | 97.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

Memory: 21.62 GiB of experts resident in RAM and 785,440 exchanged with the VRAM tier since the server started (engine log); VRAM usage not measured; no paging or out-of-memory failure.

## Correctness and limitations

Long-context recall, `tools/needle_bench.py` on the same server, one run per case: 6 of 6 found.

| Prompt length | Depth | Actual prompt tokens | Found | Seconds |
| ---: | ---: | ---: | --- | ---: |
| 32k | 10 % | 32,778 | yes | 13.7 |
| 32k | 50 % | 32,778 | yes | 12.5 |
| 32k | 90 % | 32,779 | yes | 12.7 |
| 128k | 10 % | 125,041 | yes | 50.7 |
| 128k | 50 % | 125,039 | yes | 46.4 |
| 128k | 90 % | 125,039 | yes | 46.3 |

Per-case data: `needles.json`. This measures recall on these six inputs, not overall model quality; no coding or tool-call check was run.

The speed suite is single-machine and single-session: the conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi` and the OS; the power limit and PCIe link are read from the driver, and the observed draw is a single sample during the suite. Total time of the speed suite: 73 s.

The server was already running when the suite started, so model loading is not timed and no engine log excerpt was written; the memory figures above come from the engine log of that session. The model repository and revision were not recorded by the installer.
