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

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 262144 tokens; parallel slots: 1
- Model files: two GGUF shards, 39.2 + 28.8 GB (IQ2_XS)
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0
- Conversation cache: 16384 MiB, 20 slots, minimum 4096 tokens, RAM floor 2048 MiB (`--conversation-cache-*`)

```text
strata --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.70 --mtp rt --max-context 262144 --kv k8v4 --vision --vram-reserve-mib 700 --pcie-frac 0.75 --conversation-cache-mib 16384 --conversation-cache-slots 20 --conversation-cache-min-tokens 4096 --conversation-cache-min-free-mib 2048 --resident-experts --conversation-cache-page-out
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 76.1 (58.3-105.0) | 0.41 (0.35-0.46) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1251.6 (1208.9-1256.0) | not measured | 3.23 (3.22-3.35) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1373.8 (1339.0-1419.4) | not measured | 11.72 (11.35-12.02) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1344.9 (1318.6-1375.9) | not measured | 23.87 (23.34-24.35) |
| agent session, follow-up turns | 14481 (14458-14507) at the last turn | see below | up to 64 | 3 | see below | 74.4 (66.2-98.7) | 3.00 (2.00-4.25) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 74 (4-74) (earlier turns that the cache did not hold; the 13626 (13626-13626) new tokens of the files are read once); first turn (nothing cached) first token: 2.61 s (2.53-2.83).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 224.9 | 2.825 | 58.1 |
| 1 | 3713 | 640 | 3073 | 942.4 | 3.487 | 67.1 |
| 2 | 5246 | 3776 | 1470 | 694.0 | 2.198 | 70.5 |
| 3 | 8690 | 5239 | 3451 | 827.4 | 4.248 | 72.7 |
| 4 | 11610 | 8750 | 2860 | 868.9 | 3.365 | 71.5 |
| 5 | 14507 | 11661 | 2846 | 915.0 | 3.153 | 70.5 |

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
was not used and the server was not started with `--batch`). Only one model size was tested in this report. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS. Total time of the suite: 205 s.
