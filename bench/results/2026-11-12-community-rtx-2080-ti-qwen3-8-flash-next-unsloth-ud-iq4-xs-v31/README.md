# Community benchmark on NVIDIA GeForce RTX 2080 Ti

Measured on 2026-11-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti, 22 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2697 v2 @ 2.70GHz (12 cores, 24 threads); installed RAM: 127.9 GiB; storage: USB SanDisk 3.2Gen1 Unspecified/USB; Samsung SSD 990 EVO 1TB SSD/NVMe; ATA ST2000VX000-1CU1 HDD/SATA
- OS: Windows 11 (10.0.26300); GPU driver: 621.14; CUDA (driver supports up to): 13.5
- Strata: project 0.1.59.1, commit 4b7e478b; engine 0.1.59.1
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=128.82, pcie=gen3 x16, pstate=P2, temp_c=58.10
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs-v31
- Context: 204800 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
component:strata-engine --pack unsloth-ud-iq4_xs-v31 --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 204800 --kv int8 --kv-resident 32768 --vision --pool-workers 10 --vram-reserve-mib 512
```

Environment of the config: `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_RAM_KEEP_GIB=8`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 66.9 (54.7-92.4) | 0.56 (0.52-0.69) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 790.6 (789.5-796.8) | not measured | 5.12 (5.08-5.13) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1409.7 (1372.7-1425.4) | not measured | 11.44 (11.32-11.76) |
| agent session, follow-up turns | 13802 (13797-13826) at the last turn | see below | up to 64 | 3 | see below | 64.1 (53.1-83.4) | 4.78 (3.77-5.46) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.0 % (69.9-70.2); earlier-turn tokens read again per session: 73 (4-75) (earlier turns that the cache did not hold; the 12948 (12948-12949) new tokens of the files are read once); first turn (nothing cached) first token: 3.63 s (3.60-3.70).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 170.8 | 3.595 | 60.7 |
| 1 | 4056 | 634 | 3422 | 632.5 | 5.443 | 64.0 |
| 2 | 5589 | 4049 | 1540 | 378.1 | 4.105 | 60.2 |
| 3 | 8835 | 5653 | 3182 | 593.6 | 5.399 | 59.2 |
| 4 | 11028 | 8860 | 2168 | 513.3 | 4.269 | 62.3 |
| 5 | 13797 | 11088 | 2709 | 571.0 | 4.784 | 80.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 184 s.
