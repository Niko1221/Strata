# Community benchmark on NVIDIA GeForce RTX 3060 + NVIDIA GeForce RTX 3060

Measured on 2026-10-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3060 + NVIDIA GeForce RTX 3060, 24 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2687W v4 @ 3.00GHz (12 cores, 20 threads); installed RAM: 117.2 GiB; storage: sda     WDC WD6005FFBX-6                         sata; sdb     WDC WD6005FFBX-6                         sata; sdc     ST4000VN008-2DR1                         sata; sdd     ST4000VN008-2DR1                         sata; sde     Samsung SSD 870                          sata; nvme1n1 MTFDKBA1T0QFM-1BD1AABGB                  nvme; nvme0n1 Lexar SSD NM790 2TB                      nvme
- OS: Linux 7.0.14-25-pve; GPU driver: 595.112.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.53, commit 46c152a1; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=170.00, power_draw_w=86.68, pcie=gen3 x16, pstate=P2, temp_c=62
- GPU detail (after the run, near idle): power_limit_w=170.00, power_draw_w=106.79, pcie=gen3 x16, pstate=P2, temp_c=62
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 2
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --ple-io ram --kv-resident 32768 --vision --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --prefill-pipe 3072 --conversation-cache-mib 16384 --conversation-cache-slots 8
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 63.8 (55.2-92.6) | 0.61 (0.56-0.66) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 731.8 (711.4-735.1) | not measured | 5.52 (5.50-5.68) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1207.4 (1206.9-1213.3) | not measured | 13.34 (13.28-13.35) |
| agent session, follow-up turns | 12945 (12942-12946) at the last turn | see below | up to 64 | 3 | see below | 67.7 (60.8-82.8) | 5.29 (5.09-5.89) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.1-69.3); earlier-turn tokens read again per session: 73 (71-144) (earlier turns that the cache did not hold; the 12064 (12063-12066) new tokens of the files are read once); first turn (nothing cached) first token: 4.21 s (3.94-4.21).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 155.9 | 3.936 | 54.9 |
| 1 | 3351 | 624 | 2727 | 518.9 | 5.292 | 60.8 |
| 2 | 4884 | 3344 | 1540 | 299.7 | 5.173 | 68.1 |
| 3 | 8128 | 4947 | 3181 | 577.5 | 5.546 | 62.2 |
| 4 | 10197 | 8192 | 2005 | 394.7 | 5.116 | 67.7 |
| 5 | 12942 | 10259 | 2683 | 501.8 | 5.384 | 71.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 192 s.
