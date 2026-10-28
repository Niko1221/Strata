# Community benchmark on NVIDIA GeForce RTX 4060 Ti + NVIDIA GeForce RTX 4060 Ti

Measured on 2026-10-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4060 Ti + NVIDIA GeForce RTX 4060 Ti, 32 GiB VRAM
- CPU: AMD Ryzen Threadripper PRO 3975WX 32-Cores (32 cores, 64 threads); installed RAM: 125.6 GiB; storage: sda     ST4000VN008-2DR166   usb; sdb     WDC WD20EFAX-68FB5N0 sata; zram0; nvme0n1 CT500P310SSD8        nvme
- OS: Linux 7.1.5-76070105-generic; GPU driver: 610.57.04; CUDA (driver supports up to): 13.3
- Strata: project 0.1.53, commit 46c152a1; engine 0.1.53
- GPU detail (after the run, near idle): power_limit_w=165.00, power_draw_w=47.11, pcie=gen4 x8, pstate=P2, temp_c=50
- GPU detail (after the run, near idle): power_limit_w=165.00, power_draw_w=62.97, pcie=gen4 x8, pstate=P2, temp_c=59
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072
```

Environment of the config: `STRATA_PROMPT_ATTN_IMMA=1`, `STRATA_HC_UPMIX=1`, `STRATA_ADAPT_NOWAIT=1`, `STRATA_ADAPT_LAG=2`, `STRATA_CPU_IQ_ALL_EXACT1=1`, `STRATA_SIDE_MIXER=2`, `STRATA_ATTN_DAILY=1`, `STRATA_GR_UP75=1`, `STRATA_GU_SWIGLU=1`, `STRATA_QSA_ROWS=1`, `STRATA_B6_MMVQ_ROWS=1`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 45.8 (35.1-53.8) | 0.79 (0.71-0.87) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 836.0 (832.1-848.0) | not measured | 4.84 (4.77-4.86) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1534.3 (1530.0-1560.8) | not measured | 10.51 (10.33-10.54) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1848.0 (1839.3-1862.4) | not measured | 17.40 (17.26-17.48) |
| agent session, follow-up turns | 12927 (12925-12938) at the last turn | see below | up to 64 | 3 | see below | 54.6 (50.8-61.9) | 4.29 (4.08-4.66) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.1-69.5); earlier-turn tokens read again per session: 72 (4-143) (earlier turns that the cache did not hold; the 12063 (12062-12063) new tokens of the files are read once); first turn (nothing cached) first token: 3.69 s (3.50-3.80).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 175.7 | 3.501 | 36.5 |
| 1 | 3360 | 633 | 2727 | 640.3 | 4.297 | 54.5 |
| 2 | 4893 | 3423 | 1470 | 362.7 | 4.089 | 51.5 |
| 3 | 8137 | 4956 | 3181 | 712.1 | 4.506 | 50.9 |
| 4 | 10196 | 8191 | 2005 | 485.0 | 4.171 | 54.1 |
| 5 | 12927 | 10244 | 2683 | 629.1 | 4.303 | 56.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 231 s.
