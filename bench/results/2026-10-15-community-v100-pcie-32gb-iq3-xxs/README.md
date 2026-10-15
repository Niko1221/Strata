# Community benchmark on Tesla V100-PCIE-32GB

Measured on 2026-10-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla V100-PCIE-32GB, 32 GiB VRAM
- CPU: AMD Ryzen 5 3600 (6 cores, 12 threads, 3.6 GHz base, 4.2 GHz boost, 32 MiB L3); installed RAM: 31.2 GiB plus a 23 GiB swap file; storage: ADATA LEGEND 860 NVMe SSD (465.8 GB)
- OS: Linux 6.8.0-147-generic; GPU driver: 580.178.04 (`nvidia-smi` reports CUDA 13.0); the engine was built with the CUDA 12.8 toolkit
- GPU link and limits: PCIe 3.0 x16 (the card's maximum, and what it trained at), 250 W power limit (the card's default and maximum), 1380 MHz SM clock cap
- Strata: project 0.1.42, not a git checkout (source snapshot); engine 0.1.42 locally built for sm_70 with `STRATA_EXPERIMENTAL_SM60=ON`; engine binary sha256 `7d3f93d7cefe6624f57b35b0d027792f10aefcfc962fd5b34187b5678b0a2232`
- Background workloads: none while measuring - no other engine and no GPU work (the box runs one Strata engine at a time). Power draw and throttling were not recorded.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Model files: two GGUF shards, 47.0 + 28.8 GB (IQ3_XXS)
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0
- Conversation cache: 16384 MiB, 20 slots, minimum 4096 tokens, RAM floor 2048 MiB (`--conversation-cache-*`)

```text
strata --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.70 --mtp rt --max-context 262144 --kv k8v4 --vision --vram-reserve-mib 700 --pcie-frac 0.75 --conversation-cache-mib 16384 --conversation-cache-slots 20 --conversation-cache-min-tokens 4096 --conversation-cache-min-free-mib 2048 --conversation-cache-page-out --mmap-experts
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 62.5 (44.9-88.3) | 0.46 (0.40-0.69) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 930.7 (438.0-1015.1) | not measured | 4.34 (3.98-9.20) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1180.1 (1160.6-1294.2) | not measured | 13.64 (12.44-13.87) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1206.8 (1148.8-1246.2) | not measured | 26.60 (25.76-27.95) |
| agent session, follow-up turns | 14466 (14441-14517) at the last turn | see below | up to 64 | 3 | see below | 75.4 (40.8-96.9) | 3.46 (2.82-4.31) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

The 4K row is wide because of the first run: with `--mmap-experts` the experts live in the model files and reach the
GPU through the OS file cache, so the first fresh prompt after the start reads them from disk (438.0 tok/s in run 1
against 1015.1 and 931.0 tok/s in runs 2 and 3). Every later test ran warm.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 73 (4-73) (earlier turns that the cache did not hold; the 13626 (13626-13628) new tokens of the files are read once); first turn (nothing cached) first token: 3.09 s (2.67-4.10).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 152.9 | 4.101 | 69.6 |
| 1 | 3721 | 647 | 3074 | 767.1 | 4.283 | 72.3 |
| 2 | 5254 | 3714 | 1540 | 435.7 | 3.625 | 72.6 |
| 3 | 8698 | 5318 | 3380 | 800.9 | 4.309 | 74.2 |
| 4 | 11622 | 8761 | 2861 | 747.1 | 3.909 | 82.1 |
| 5 | 14517 | 11671 | 2846 | 781.7 | 3.686 | 69.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Notes on this build

The engine binary is a local source build of v0.1.42 (CUDA 12.8, sm_70, `STRATA_EXPERIMENTAL_SM60=ON`) and it carries
one unpushed, opt-in local patch: a conversation park whose host pages would hit the RAM floor is paged out instead of
refused (`--conversation-cache-page-out`). It is off unless the flag is passed and it says nothing about the prefill or
decode paths. This run passed the flag, but its engine log has no parking or page-out line and the conversations
tested stay far below the 16 GiB / 20-slot limits, so that path was not exercised: the numbers are what the same
settings give on a build without the patch. A same-day run against a clean release build was not made.

## Correctness and limitations

Not measured: this is a speed suite; no recall or needle check, no coding task, and no concurrency run (`--clients`
was not used and the server was not started with `--batch`). Only one model size was tested in this report. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS. Total time of the suite: 243 s.
