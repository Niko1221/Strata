# Community benchmark on Tesla P40 + Tesla P40

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla P40 + Tesla P40, 48 GiB VRAM
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (8 cores, 16 threads); installed RAM: 59.7 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: not read; CUDA: not read
- Strata: project 0.1.44.1, commit not a git checkout; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 50.3 (48.5-67.4) | 0.68 (0.64-0.75) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 350.9 (350.9-351.2) | not measured | 11.49 (11.48-11.49) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 503.0 (502.9-503.7) | not measured | 31.96 (31.92-31.97) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 638.4 (638.3-639.5) | not measured | 50.25 (50.16-50.26) |
| agent session, follow-up turns | 14573 (14559-14585) at the last turn | see below | up to 64 | 3 | see below | 50.7 (40.1-58.7) | 8.97 (5.04-10.42) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.7-68.8); earlier-turn tokens read again per session: 2 (2-3) (earlier turns that the cache did not hold; the 13729 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 2.83 s (2.81-2.84).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 219.6 | 2.805 | 36.4 |
| 1 | 3741 | 621 | 3120 | 328.1 | 9.544 | 54.3 |
| 2 | 5274 | 3804 | 1470 | 291.2 | 5.084 | 50.1 |
| 3 | 8718 | 5338 | 3380 | 325.6 | 10.418 | 46.8 |
| 4 | 11637 | 8775 | 2862 | 318.2 | 9.034 | 54.9 |
| 5 | 14585 | 11686 | 2899 | 326.9 | 8.908 | 49.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 472 s.

## Additional notes (added by hand, not measured by the tool)

- **Driver and toolkit:** NVIDIA driver 580.178.04; engine built from source with CUDA 12.8 for sm_61 and sm_70
  (`BUILD.json`: archs [61, 70], toolkit 12), in Docker (`docker-entrypoint.sh`, `GPUS=0,1`, `LAYER_SPLIT=auto`).
- **Links:** each P40 sits on a **PCIe 3.0 x4** link (cards capable of x16): one through the mini PC's OCuLink port,
  one through an M.2-to-OCuLink adapter, each in an external enclosure. Power limit 250 W per card (default).
- **Expert residency:** layer split auto across the two cards, 12,288 experts per card (16.19 GiB each):
  **24,576 / 24,576 experts resident in VRAM (33.0 GiB)**. The experts are also loaded in RAM (33.02 GiB).
- **KV:** int8, 32,768 tokens resident, the rest streamed from RAM (3.6 GB).
- **Vision encoder:** `strata-vision` from an earlier V100 (sm_70-only) build was not rebuilt by setup when the engine
  was recompiled for sm_61, and failed to start on the P40s ("the vision encoder did not start", empty message). Rebuilt
  by hand for `61;70` with the same `cmake_build` call as `Dockerfile.v100`; it then starts and serves images.
- **Background load:** an agent evaluation had just finished; nothing else used the GPUs during the suite. The host
  runs Prometheus, Grafana and small exporters (negligible CPU).
- **Same model on one Tesla V100 32 GB (same mini PC, OCuLink PCIe 4.0 x4, Strata 0.1.41 to 0.1.44.1, our own
  measurements, not this tool):** decode ~68 tok/s at short context, prompt ~1,255 tok/s at 31K tokens, ~98.6 % of
  experts resident. On a real agent task (an ESG data-collection run, same prompt and tools), the two P40s gave the
  same answers in 52.7 min against 41.7 min on the V100.
