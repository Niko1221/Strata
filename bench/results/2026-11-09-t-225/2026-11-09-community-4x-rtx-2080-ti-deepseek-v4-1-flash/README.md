# Community benchmark on NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti

Measured on 2026-11-09 with `tools/strata_bench.py` (suite 1, 5 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti, 88 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz (14 cores, 28 threads); installed RAM: 121.4 GiB; storage: nvme0n1 Lexar SSD NM620 2TB nvme
- OS: Linux 7.0.0-42-generic; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.55, commit a4e5d2be; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=100.84, pcie=gen3 x16, pstate=P2, temp_c=66
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=130.46, pcie=gen3 x16, pstate=P2, temp_c=77
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=139.58, pcie=gen3 x16, pstate=P2, temp_c=71
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=129.48, pcie=gen3 x16, pstate=P2, temp_c=64
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-dsv41 --pack deepseek-v4.1-flash-ist-das-2.25bpw --max-context 131072 --eos 1 --vision on --dspark auto --dspark-conf 0.5 --tp 4 --experts 5 --threads on
```

## Method

Run by `python tools/strata_bench.py --long --runs 5` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 5 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 5 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 5 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 15 | not measured | 32.3 (26.0-35.1) | 0.38 (0.32-0.51) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 5 | 534.7 (475.5-561.9) | not measured | 7.73 (7.36-8.70) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 5 | 941.1 (894.2-947.3) | not measured | 17.64 (17.53-18.56) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 5 | 1060.6 (993.3-1082.5) | not measured | 31.33 (30.70-33.48) |
| agent session, follow-up turns | 12662 (12653-12688) at the last turn | see below | up to 64 | 5 | see below | 26.9 (22.8-27.4) | 12.30 (8.71-14.81) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.6-69.7); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 11797 (11796-11798) new tokens of the files are read once); first turn (nothing cached) first token: 6.01 s (5.85-7.07).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 85.3 | 7.069 | 22.3 |
| 1 | 3340 | 640 | 2700 | 196.4 | 13.778 | 22.8 |
| 2 | 4854 | 3400 | 1454 | 139.5 | 10.46 | 25.9 |
| 3 | 7957 | 4909 | 3048 | 206.5 | 14.808 | 26.0 |
| 4 | 10038 | 8017 | 2021 | 176.6 | 11.481 | 26.8 |
| 5 | 12662 | 10084 | 2578 | 185.1 | 13.986 | 26.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 759 s.

## Publication notes

**This is not a standard-protocol result.** It was produced by `tools/strata_bench.py`, not by `strata bench run` from
[BENCHMARK_PROTOCOL.md](../../../../docs/BENCHMARK_PROTOCOL.md); it is published as a community report, which the guide allows
for measurements made outside the protocol.

### Model identity

ISTA-DASLab's DeepSeek-V4.1-Flash-1017 GSQ-RCO set at **2.25 bits per weight** (the pack's own `--pack-info`: 149.1 GB of
routed experts at 2.19 bpw, 544 G parameters, 8 files).

- pack descriptor `ds4.txt`, sha256 `3996643c6700955cc4d1ac613dc761ca50590de84d1f28ec01196a7868cac1d4`;
- pack tensor index `tensors.txt`, sha256 `0f65c8c8a5b7669af1fc8ced969b1b8cba8f641d5a567c4e666e235bbaf66267`;
- DSpark drafter `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, sha256
  `6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a` (the file the installer pins);
- Engram `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (110.81 GB dense).

### Settings

Engine environment on top of the arguments in `config.json` / `env.txt`: `DS4_ASYNC_T=8`, `DS4_DEFAULTS=075`,
`DS4_ATTN_SPLIT=0`, `DS4_LANE_FUSE=1`, `DS4_GG_IQ1M=auto`. **`DS4_DEFAULTS=075` was measured on the 2.00 bpw set**, not on
this one, so it is carried as the best known setting rather than as one tuned for 2.25 bpw; it changes numerics and stays an
explicit opt-in token.

### Correctness and limitations

Not measured: answer correctness; this is a speed suite. Not tested: HIP build (no toolchain here), Windows package (no
Windows machine), any card family other than four sm_75 RTX 2080 Ti. This set's 149 GB of experts do not fit in the box's
121 GiB of RAM beside the dense weights, so the CPU lane reads them from disk instead of from RAM copies (the 2.00 bpw set,
at 71 GB of CPU-lane experts, is copied into RAM at load) - that is the structural reason its decode is the slowest of the
three sets here, and it is reported, not tuned away.
