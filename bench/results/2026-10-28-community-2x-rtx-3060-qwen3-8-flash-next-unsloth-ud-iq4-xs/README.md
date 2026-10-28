# Community benchmark on NVIDIA GeForce RTX 3060 + NVIDIA GeForce RTX 3060

Measured on 2026-10-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3060 + NVIDIA GeForce RTX 3060, 24 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2687W v4 @ 3.00GHz (12 cores, 20 threads); installed RAM: 117.2 GiB; storage: sda     WDC WD6005FFBX-6                         sata; sdb     WDC WD6005FFBX-6                         sata; sdc     ST4000VN008-2DR1                         sata; sdd     ST4000VN008-2DR1                         sata; sde     Samsung SSD 870                          sata; nvme1n1 MTFDKBA1T0QFM-1BD1AABGB                  nvme; nvme0n1 Lexar SSD NM790 2TB                      nvme
- OS: Linux 7.0.14-25-pve; GPU driver: 595.112.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.53, commit 46c152a1; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=170.00, power_draw_w=55.75, pcie=gen3 x16, pstate=P2, temp_c=62
- GPU detail (after the run, near idle): power_limit_w=170.00, power_draw_w=96.62, pcie=gen3 x16, pstate=P2, temp_c=59
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --ple-io ram --kv-resident 32768 --vision --pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

Environment of the config: `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 58.3 (54.5-78.9) | 0.50 (0.45-0.57) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 642.1 (641.6-642.2) | not measured | 6.29 (6.29-6.30) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 994.6 (992.6-996.8) | not measured | 16.19 (16.15-16.22) |
| agent session, follow-up turns | 12933 (12894-12937) at the last turn | see below | up to 64 | 3 | see below | 65.9 (53.1-71.8) | 6.23 (5.79-6.56) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.2 % (69.2-69.5); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 12063 (12062-12065) new tokens of the files are read once); first turn (nothing cached) first token: 4.55 s (4.47-4.56).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 137.1 | 4.472 | 46.2 |
| 1 | 3347 | 617 | 2730 | 439.0 | 6.254 | 68.6 |
| 2 | 4880 | 3411 | 1469 | 253.1 | 5.838 | 53.1 |
| 3 | 8124 | 4944 | 3180 | 493.8 | 6.477 | 60.9 |
| 4 | 10186 | 8181 | 2005 | 335.3 | 6.016 | 60.5 |
| 5 | 12933 | 10179 | 2754 | 428.4 | 6.468 | 68.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 219 s.
