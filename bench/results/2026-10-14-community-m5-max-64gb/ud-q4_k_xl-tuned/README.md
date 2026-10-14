# Community benchmark on CPU only

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: arm (18 cores, 18 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Darwin 27.0.0; GPU driver: not read; CUDA: not read
- Strata: project 0.1.44, commit ed4773eb; engine 0.1.44-metal
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q4_k_xl
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (metal backend); thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-q4_k_xl --native unsloth-ud-q4_k_xl-metal.gguf --ple-gguf Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf --native-dense-gguf unsloth-ud-q4_k_xl-metal.gguf --expert-profile expert-profile.bin --expert-cache 64 --prefill auto --suffix-draft 0 --max-context 32768 --mmap-experts --mtp rt --spec 3
```

Environment of the config: `STRATA_METAL_STREAM=1`, `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_METAL_MTP_GGUF=mtp-q8_0.gguf`, `STRATA_METAL_ARENA_GB=40`, `STRATA_METAL_PROMPT_CHUNK=2048`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 42.8 (29.6-46.0) | 0.82 (0.69-1.32) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1048.4 (996.8-1076.3) | not measured | 3.87 (3.76-4.06) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1049.6 (999.3-1057.3) | not measured | 15.33 (15.22-16.10) |
| agent session, follow-up turns | 14571 (14547-14600) at the last turn | see below | up to 64 | 3 | see below | 44.8 (38.7-53.4) | 5.19 (2.64-7.56) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.7); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 13730 (13729-13731) new tokens of the files are read once); first turn (nothing cached) first token: 2.72 s (2.69-2.73).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 229.3 | 2.687 | 28.7 |
| 1 | 3741 | 619 | 3122 | 564.2 | 5.568 | 40.3 |
| 2 | 5274 | 3804 | 1470 | 539.0 | 2.761 | 41.9 |
| 3 | 8718 | 5267 | 3451 | 629.7 | 5.516 | 43.0 |
| 4 | 11644 | 8781 | 2863 | 540.1 | 5.335 | 44.6 |
| 5 | 14600 | 11701 | 2899 | 543.7 | 5.366 | 38.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 204 s.
