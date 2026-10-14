# Community benchmark — RTX 4070 in Thunderbolt 4 eGPU — 2026-10-14

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070, 12 GiB VRAM
- PCIe link to GPU: Thunderbolt 4 eGPU enclosure (~PCIe 4.0 x4 equivalent); the card is not in a motherboard slot
- CPU: 12th Gen Intel(R) Core(TM) i9-12950HX (16 cores, 24 threads); installed RAM: 62.5 GiB; storage: not measured
- OS: Linux 7.2.9-2-cachyos; GPU driver: 580.178.04; CUDA: 13.0
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 32768 --kv int8 --spec-min-p 0.5
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 33.5 (30.4-38.7) | 0.94 (0.85-1.05) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 213.7 (213.7-213.8) | not measured | 18.83 (18.83-18.84) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 413.8 (413.7-415.7) | not measured | 38.85 (38.67-38.86) |
| agent session, follow-up turns | 14540 (14534-14556) at the last turn | see below | up to 64 | 3 | see below | 35.8 (25.7-56.0) | 18.62 (18.07-18.76) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.6); earlier-turn tokens read again per session: 74 (74-74) (earlier turns that the cache did not hold; the 13728 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 7.15 s (7.09-7.22).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 85.5 | 7.148 | 25.9 |
| 1 | 3750 | 630 | 3120 | 166.7 | 18.753 | 32.4 |
| 2 | 5283 | 3813 | 1470 | 81.2 | 18.127 | 34.6 |
| 3 | 8727 | 5276 | 3451 | 184.4 | 18.748 | 33.2 |
| 4 | 11633 | 8771 | 2862 | 154.2 | 18.604 | 40.1 |
| 5 | 14534 | 11635 | 2899 | 155.7 | 18.662 | 25.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 543 s.

The GPU hangs off a Thunderbolt 4 link (~PCIe 4.0 x4), so the host-to-device path is much narrower than a desktop slot. For this run the engine's own calibration kept the PCIe share small (`pcie_frac` 0.08 in `results.json`), i.e. little expert traffic crosses the link; Strata's expert cache mostly reads what already sits in the 12 GiB of VRAM. What fraction of the prompt rate a wider slot would give was not measured.
