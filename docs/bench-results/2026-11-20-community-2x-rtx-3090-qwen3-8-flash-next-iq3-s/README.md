# Community benchmark on NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090

Measured on 2026-11-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090, 48 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 127.6 GiB; storage: Samsung SSD 990 EVO Plus 4TB SSD/NVMe; Samsung SSD 990 EVO Plus 2TB SSD/NVMe; ST2000DM001-1ER164 HDD/SATA; Viper VP4300L 4TB SSD/NVMe
- OS: Windows 11 (10.0.26300); GPU driver: 616.92; CUDA (driver supports up to): 13.4; engine built for CUDA 13.0
- Strata: project 0.1.62, commit not a git checkout; engine 0.1.62
- GPU detail (after the run, near idle): power_limit_w=480.00, power_draw_w=178.55, pcie=gen4 x8, pstate=P2, temp_c=62
- GPU detail (after the run, near idle): power_limit_w=450.00, power_draw_w=110.99, pcie=gen4 x8, pstate=P3, temp_c=42
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 204800 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 204800 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --spec-min-p 0.5 --prefill-pipe 1536
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a <user> tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 77.1 (62.2-101.7) | 0.41 (0.39-0.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1068.1 (1048.4-1069.8) | not measured | 3.80 (3.78-3.88) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2029.3 (1999.7-2047.3) | not measured | 7.95 (7.91-8.07) |
| agent session, follow-up turns | 9075 (9044-9086) at the last turn | see below | up to 64 | 3 | see below | 87.5 (72.3-100.0) | 2.25 (2.20-2.29) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 71.0 % (71.0-71.1); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 8208 (8207-8209) new tokens of the files are read once); first turn (nothing cached) first token: 1.56 s (1.43-1.57).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 438.4 | 1.432 | 62.8 |
| 1 | 2265 | 623 | 1642 | 758.7 | 2.211 | 95.7 |
| 2 | 3971 | 2328 | 1643 | 741.4 | 2.251 | 73.7 |
| 3 | 5677 | 4034 | 1643 | 739.1 | 2.263 | 78.3 |
| 4 | 7354 | 5712 | 1642 | 736.2 | 2.269 | 87.7 |
| 5 | 9044 | 7402 | 1642 | 732.8 | 2.283 | 82.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 109 s.

## Conditions addendum (added by the author after the run)

This run was done straight after a fresh setup/calibration, **not** on an idle machine: it is a Hyper-V host and two Linux VMs (Ubuntu Docker server + Talos k8s node) were running, plus normal desktop usage — free RAM was tight while measuring. Treat these numbers as a loaded-machine cross-check only; the same machine under clean conditions was measured the same day in [2026-11-20-standard-rtx-3090-plus-rtx-3090-e2f62039](../2026-11-20-standard-rtx-3090-plus-rtx-3090-e2f62039/README.md), which supersedes this run for comparison purposes.
