# Community benchmark on Apple M5 MacBook Air (Metal), 32 GB

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M5 GPU (Metal), 32 GB unified memory (the tool reads "CPU only"; see the notes below)
- CPU: arm (10 cores, 10 threads); installed RAM: 32.0 GiB; storage: not measured
- OS: Darwin 27.2.0; GPU driver: not read; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1-metal
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-q2_0
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (metal backend); thinking off for the suite; temperature 0

```text
strata --pack q2_0 --native Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 --max-context 32768 --mmap-experts
```

Environment of the config: `STRATA_METAL_STREAM=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 17.7 (14.7-19.5) | 2.17 (1.93-2.40) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 165.0 (158.1-215.8) | not measured | 24.44 (18.71-25.51) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 172.1 (166.7-174.1) | not measured | 93.44 (92.36-96.46) |
| agent session, follow-up turns | 14515 (14515-14564) at the last turn | see below | up to 64 | 3 | see below | 9.7 (7.9-11.1) | 22.18 (11.43-26.10) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.7); earlier-turn tokens read again per session: 5 (5-51) (earlier turns that the cache did not hold; the 13730 (13729-13730) new tokens of the files are read once); first turn (nothing cached) first token: 5.47 s (5.47-5.73).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 108.1 | 5.733 | 9.3 |
| 1 | 3751 | 628 | 3123 | 132.2 | 23.719 | 9.2 |
| 2 | 5284 | 3814 | 1470 | 121.8 | 12.214 | 9.1 |
| 3 | 8728 | 5347 | 3381 | 130.1 | 26.097 | 10.6 |
| 4 | 11603 | 8741 | 2862 | 136.4 | 21.087 | 10.7 |
| 5 | 14515 | 11616 | 2899 | 131.3 | 22.18 | 11.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 845 s.

## Notes added by hand

- **Hardware:** MacBook Air (Mac17,3), Apple M5 (10-core CPU), 32 GB unified memory, the model on the internal SSD. The tool reports "CPU only" because it reads GPUs through `nvidia-smi`/`rocm-smi`; the engine ran on the M5's GPU through Metal (`engine 0.1.44.1-metal`).
- **Install:** `./setup.sh --yes --model Q2_0 --no-start`, built from source on this Mac; Q2_0 with the experts' copy (`.experts`, 31.64 GiB), no MTP draft layer (setup's choice for a streaming Mac).
- **This run vs. the first run the same day** (`../2026-10-16-community-macbook-air-m5-32gb-q2_0`): after a macOS update (build 26B5101f -> 26B5112a, both macOS 27.2) and a restart, with all apps except Terminal and Finder closed, and after waiting for the load average to stay under 3 with Spotlight idle for 2 minutes. The benchmark started the server itself (fresh start).
- **Expert arena:** 8.68 GB (6231 slots), against 5.41 GB in the first run (planned footprint 15.2 GB, cap 26.8 GB). Chat decode rose from 9.6 to 17.7 tok/s.
- **Prompts were not faster:** 4K 165 vs 198 tok/s, 16K 172 vs 182 tok/s. No swap was used (6.6 MB), and macOS recorded no thermal or performance warnings. The cause is not known; the 4K runs varied (215.8, 158.1, 165.0 tok/s).
- **Prompt chunks:** 1024 tokens (the engine's default on a streaming Mac under 96 GB).
- A background sync client (Nextcloud) was running.
- **Power and cooling:** on the charger (AC power), ambient temperature 23 °C, no forced cooling. The MacBook Air is fanless (passive cooling only); an M5 in an actively cooled Mac (e.g. a Mac mini) may sustain higher clocks over a long run.
