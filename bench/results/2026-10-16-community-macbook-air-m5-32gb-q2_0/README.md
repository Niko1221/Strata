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
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 9.6 (8.8-10.2) | 3.57 (2.87-4.24) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 198.0 (187.2-201.7) | not measured | 20.38 (20.00-21.55) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 182.1 (177.8-199.2) | not measured | 88.29 (80.72-90.41) |
| agent session, follow-up turns | 14524 (14515-14527) at the last turn | see below | up to 64 | 3 | see below | 8.9 (8.0-11.6) | 19.84 (10.61-23.47) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.7); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13730 (13730-13730) new tokens of the files are read once); first turn (nothing cached) first token: 4.75 s (4.73-5.41).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 130.7 | 4.747 | 10.6 |
| 1 | 3751 | 628 | 3123 | 152.2 | 20.627 | 8.2 |
| 2 | 5284 | 3814 | 1470 | 139.7 | 10.614 | 8.9 |
| 3 | 8728 | 5347 | 3381 | 151.8 | 22.361 | 10.2 |
| 4 | 11603 | 8741 | 2862 | 159.0 | 18.094 | 11.5 |
| 5 | 14515 | 11616 | 2899 | 157.8 | 18.46 | 11.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 832 s.

## Notes added by hand

- **Hardware:** MacBook Air (Mac17,3), Apple M5 (10-core CPU), 32 GB unified memory, macOS 27.2, the model on the internal SSD. The tool reports "CPU only" because it reads GPUs through `nvidia-smi`/`rocm-smi`; the engine ran on the M5's GPU through Metal (`engine 0.1.44.1-metal`).
- **Install:** `./setup.sh --yes --model Q2_0 --no-start`, built from source on this Mac; Q2_0 with the experts' copy (`.experts`, 31.64 GiB), no MTP draft layer (setup's choice for a streaming Mac).
- **Expert arena:** 5.41 GB (3883 slots). When the engine started, macOS reported 12.9 GB available (7.0 GB kept for macOS), so the arena was much smaller than the 11.5-14.7 GB on the 32 GB M4 in [MAC.md](../../../docs/MAC.md). MAC.md reports about 8.3 tok/s decode with a 5.7 GB arena on the M4, close to the 9.6 tok/s here. A run with more memory free would likely decode faster.
- **Prompt chunks:** 1024 tokens (the engine's default on a streaming Mac under 96 GB).
- Background load during the run was not controlled (normal desktop session).
- **Power and cooling:** on the charger (AC power), ambient temperature 23 °C, no forced cooling. The MacBook Air is fanless (passive cooling only); an M5 in an actively cooled Mac (e.g. a Mac mini) may sustain higher clocks over a long run.
