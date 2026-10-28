# Community benchmark on NVIDIA GeForce RTX 3070

Measured on 2026-10-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3070, 8 GiB VRAM
- CPU: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz (10 cores, 20 threads); installed RAM: 61.3 GiB; storage: nvme0n1 PM9A1 NVMe Samsung 1024GB nvme
- OS: Linux 7.0.0-38-generic; GPU driver: 595.99.02; CUDA (driver supports up to): 13.2
- Strata: project ?, commit not a git checkout; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=220.00, power_draw_w=172.18, pcie=gen3 x16, pstate=P2, temp_c=69
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
component:strata-engine --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --max-context 65536 --kv int8 --vram-reserve-mib 1024 --kv-resident 32768
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 46.0 (45.6-47.6) | 0.68 (0.56-0.73) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 411.2 (410.9-411.4) | not measured | 9.80 (9.80-9.81) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 367.9 (367.5-368.1) | not measured | 43.68 (43.66-43.73) |
| agent session, follow-up turns | 9087 (9085-9094) at the last turn | see below | up to 64 | 3 | see below | 58.5 (41.8-96.4) | 5.19 (5.02-5.24) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 71.1 % (71.1-71.1); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 8207 (8207-8208) new tokens of the files are read once); first turn (nothing cached) first token: 2.51 s (2.39-2.52).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 258.3 | 2.392 | 42.2 |
| 1 | 2274 | 632 | 1642 | 328.7 | 5.024 | 44.6 |
| 2 | 3980 | 2337 | 1643 | 324.1 | 5.104 | 44.5 |
| 3 | 5686 | 4043 | 1643 | 317.7 | 5.205 | 45.3 |
| 4 | 7392 | 5749 | 1643 | 318.2 | 5.195 | 67.3 |
| 5 | 9094 | 7452 | 1642 | 316.0 | 5.225 | 58.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 297 s.
