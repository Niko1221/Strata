# Community benchmark on AMD Radeon RX 9070

Measured on 2026-11-07 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 9070, 16 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i5-13490F (10 cores, 16 threads); installed RAM: 63.8 GiB; storage: KIOXIA-EXCERIA G2 SSD SSD/NVMe; SanDisk SD8SB8U1T001122 SSD/SATA; Seagate ZP1000CM30021 SSD/NVMe; SAMSUNG MZALQ128HBHQ-000L2 SSD/NVMe
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.32015.2008; CUDA (driver supports up to): not read
- Strata: project ?, commit not a git checkout; engine 0.1.57
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 196608 tokens; parallel slots: 1
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 44.1 (40.1-48.9) | 0.76 (0.68-0.94) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 796.2 (790.9-796.9) | not measured | 5.09 (5.09-5.13) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 982.0 (980.9-982.3) | not measured | 16.40 (16.39-16.41) |
| agent session, follow-up turns | 9074 (9065-9100) at the last turn | see below | up to 64 | 3 | see below | 53.3 (46.8-65.4) | 2.03 (1.58-2.18) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 71.1 % (71.0-71.2); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 8208 (8208-8208) new tokens of the files are read once); first turn (nothing cached) first token: 2.22 s (1.96-2.24).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 317.5 | 1.957 | 31.7 |
| 1 | 2268 | 626 | 1642 | 1038.8 | 1.606 | 47.8 |
| 2 | 3974 | 2331 | 1643 | 910.9 | 1.845 | 46.8 |
| 3 | 5680 | 4037 | 1643 | 817.4 | 2.054 | 53.3 |
| 4 | 7386 | 5743 | 1643 | 774.4 | 2.164 | 51.3 |
| 5 | 9065 | 7423 | 1642 | 772.1 | 2.154 | 59.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 158 s.
