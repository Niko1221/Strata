# Community benchmark on Tesla V100-PCIE-32GB

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla V100-PCIE-32GB, 32 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz (8 cores, 8 threads); installed RAM: 66.0 GiB; storage: Msft Virtual Disk Unspecified/SAS; Msft Virtual Disk Unspecified/SAS
- OS: Windows 2022Server (10.0.20348); GPU driver: 582.78; CUDA (driver supports up to): 13.0
- Strata: project 0.1.58, commit not a git checkout; engine 0.1.58
- GPU detail (after the run, near idle): power_limit_w=130.00, power_draw_w=45.64, pcie=gen3 x16, pstate=P0, temp_c=59
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --spec-min-p 0.5 --pool-workers 5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 57.2 (49.5-67.9) | 0.60 (0.54-0.64) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1082.7 (1077.9-1088.1) | not measured | 3.75 (3.75-3.78) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1299.7 (1290.6-1314.7) | not measured | 12.41 (12.28-12.50) |
| agent session, follow-up turns | 13522 (13489-13588) at the last turn | see below | up to 64 | 3 | see below | 67.3 (51.4-78.4) | 2.99 (2.06-3.85) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.5-69.7); earlier-turn tokens read again per session: 74 (73-75) (earlier turns that the cache did not hold; the 12724 (12723-12724) new tokens of the files are read once); first turn (nothing cached) first token: 2.21 s (2.09-2.25).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 276.0 | 2.251 | 37.9 |
| 1 | 3824 | 619 | 3205 | 899.4 | 3.609 | 53.3 |
| 2 | 5357 | 3817 | 1540 | 640.5 | 2.453 | 51.4 |
| 3 | 8575 | 5395 | 3180 | 884.7 | 3.648 | 76.0 |
| 4 | 10764 | 8600 | 2164 | 816.1 | 2.71 | 74.2 |
| 5 | 13489 | 10779 | 2710 | 891.9 | 3.095 | 56.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 147 s.

## Submitter's notes

What the tool does not record, added by hand:

- Machine: a Hyper-V guest on the same host as the P40 report of the same date (ASUS Z10PE-D16 WS, 2x Xeon E5-2630 v3, 384 GB DDR4). The VM runs Windows Server 2022 with 8 vCPUs and 66 GB RAM; the V100 is passed through with DDA. No other GPU or CPU workload ran in the VM; other VMs on the host kept running and their load was not recorded.
- Power: the V100 is limited to 130 W (`nvidia-smi -pl 130`) because the passively cooled card runs hot in this chassis; at 130 W it stays near 68 °C under load.
- Storage: the model files sit in a VHDX on a Samsung 990 PRO NVMe SSD.
- Install: the GitHub archive of tag v0.2.12 (no git checkout, so the report says "commit not a git checkout"; the tag is commit 146df3c4). The 0.2 releases carry no ready-made CUDA 12 engine for a source install, so setup compiled it with CUDA 12.9 (`STRATA_NVCC`) and Visual Studio 2022 Build Tools; driver 582.78. Setup choices: 262K context, 8-bit KV, images on, experimental speed projection off; setup turned KV streaming on and calibrated the options shown above.
