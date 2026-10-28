# Community benchmark on NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5080

Measured on 2026-10-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5080, 32 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 95.7 GiB; storage: HP SSD FX900 Pro 4TB SSD/NVMe; HP SSD FX900 Pro 4TB SSD/NVMe
- OS: Windows 11 (10.0.26200); GPU driver: 621.01; CUDA (driver supports up to): 13.5; engine built for CUDA 13.0
- Strata: project 0.1.52, commit 5e8a4cca; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=150.00, power_draw_w=24.61, pcie=gen5 x8, pstate=P1, temp_c=48.68
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=102.72, pcie=gen5 x8, pstate=P1, temp_c=47.54
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-nvfp4-nvfp4-nvidia-BF16
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack pack --native qwen-nvidia-nvfp4-dense.gguf --native-dense-gguf qwen-nvidia-nvfp4-dense.gguf --ple-gguf ple-fp8.gguf --embd-gguf token-embd-bf16.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:4096 --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 2048 --vram-reserve-later-mib 1500 --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 46.6 (42.7-58.1) | 0.70 (0.64-0.92) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1150.0 (820.4-1155.0) | not measured | 3.53 (3.52-4.93) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1855.6 (1650.4-1858.2) | not measured | 8.70 (8.69-9.76) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1932.0 (1925.8-1933.8) | not measured | 16.65 (16.61-16.69) |
| agent session, follow-up turns | 12807 (12798-12819) at the last turn | see below | up to 64 | 3 | see below | 58.0 (50.0-66.7) | 3.26 (3.00-5.55) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.2 % (69.1-69.5); earlier-turn tokens read again per session: 74 (3-144) (earlier turns that the cache did not hold; the 11950 (11949-11951) new tokens of the files are read once); first turn (nothing cached) first token: 2.66 s (2.66-3.09).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 200.0 | 3.086 | 35.6 |
| 1 | 3309 | 633 | 2676 | 512.4 | 5.268 | 62.1 |
| 2 | 4842 | 3372 | 1470 | 340.2 | 4.368 | 50.0 |
| 3 | 8086 | 4906 | 3180 | 578.4 | 5.547 | 57.6 |
| 4 | 10105 | 8141 | 1964 | 431.0 | 4.606 | 59.7 |
| 5 | 12819 | 10156 | 2663 | 560.8 | 4.797 | 57.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 209 s.
