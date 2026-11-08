# Community benchmark on Tesla P40

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla P40, 22 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz (4 cores, 8 threads); installed RAM: 64.8 GiB; storage: sda  Virtual Disk; sr0  Virtual DVD-ROM
- OS: Linux 6.8.0-147-generic; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0
- Strata: project 0.1.58, commit 146df3c4; engine 0.1.58
- GPU detail (after the run, near idle): power_limit_w=150.00, power_draw_w=64.46, pcie=gen3 x16, pstate=P0, temp_c=67
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 29.0 (26.8-35.9) | 1.29 (1.21-1.40) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 351.9 (351.0-352.5) | not measured | 11.48 (11.46-11.51) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 379.1 (378.4-382.7) | not measured | 42.44 (42.03-42.51) |
| agent session, follow-up turns | 13560 (13529-13583) at the last turn | see below | up to 64 | 3 | see below | 35.2 (27.6-42.9) | 8.45 (5.09-9.60) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.8 % (69.7-69.8); earlier-turn tokens read again per session: 4 (4-73) (earlier turns that the cache did not hold; the 12723 (12722-12723) new tokens of the files are read once); first turn (nothing cached) first token: 3.88 s (3.76-4.14).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 164.6 | 3.762 | 26.0 |
| 1 | 3832 | 630 | 3202 | 345.1 | 9.345 | 32.6 |
| 2 | 5365 | 3895 | 1470 | 283.4 | 5.248 | 35.5 |
| 3 | 8609 | 5428 | 3181 | 333.7 | 9.604 | 32.1 |
| 4 | 10822 | 8658 | 2164 | 300.1 | 7.277 | 32.8 |
| 5 | 13560 | 10850 | 2710 | 313.5 | 8.714 | 35.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 370 s.

## Submitter's notes

What the tool does not record, added by hand:

- Machine: a Hyper-V guest (Windows Server 2022 host, ASUS Z10PE-D16 WS, 2x Xeon E5-2630 v3, 384 GB DDR4). The VM has 8 vCPUs and 66 GB RAM; the P40 is passed through with DDA. No other GPU or CPU workload ran in the VM; other VMs on the host kept running and their load was not recorded.
- Power: the P40 is at its 150 W limit.
- Storage: the model files sit in a VHDX on a Samsung 990 PRO NVMe SSD. Hyper-V's virtual disk reports itself as rotational to Linux, so setup warned about a rotational disk; a udev rule (`ATTR{queue/rotational}="0"` for `sd*` and `dm-*`) was added after this run. It does not change the engine's options.
- Install: `git clone --branch v0.2.12`; the CUDA 12 engine was compiled by setup with CUDA 12.9 (`STRATA_NVCC`), driver 580.178.04. Setup choices: 262K context, 8-bit KV, images on, experimental speed projection off; setup turned KV streaming on and calibrated the options shown above.
- The same P40 was measured on Strata 0.1.43 in `2026-10-13-community-tesla-p40-qwen3-8-flash-next-iq3-s` (a Windows Server 2022 VM with 98 GB RAM then): 21.5 chat / 26.2 agent tok/s against 29.0 / 35.2 here.
