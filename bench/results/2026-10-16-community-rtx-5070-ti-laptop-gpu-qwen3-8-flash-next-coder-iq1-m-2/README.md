# Community benchmark: Qwen3.8-Flash-Next Coder IQ1_M on a 12 GB laptop GPU

Measured on 2026-10-16 by [Aoyi-Feng](https://github.com/Aoyi-Feng). A 32 GB machine cannot hold this
quant in RAM, so the experts stream from disk through the OS file cache; the limitation is memory
capacity, not the GPU.

**Summary.** A mid-range gaming laptop (RTX 5070 Ti Laptop 12 GB, 32 GB DDR5-5600, Ryzen 9
9955HX) runs the Coder IQ1_M quant of Qwen3.8-Flash-Next through Strata in low-RAM
resident-expert mode, because the experts do not fit in 32 GB of RAM. It works, and it is
usable for chat and agent work: **20.5 tok/s decode in short chat (16.4–23.4), ~20 tok/s in a
6-turn agent session**, with the first token in about **2.3 s**. Reading long prompts is fast
(**387 tok/s at 4K, 525 tok/s at 16K**); generating is the slow half, and it is bounded by
expert streaming from RAM/SSD (53–63 % expert-cache hit during decode), not by the GPU.
GPU utilization averaged 56 % while the CPU ran at 5.1 GHz and 88 °C, so the machine's CPU/IO
side, not the GPU, is what saturates. Same-model machines whose RAM holds all the experts
report roughly 90–117 tok/s, i.e. this configuration pays about a 4–5x penalty for the
missing RAM; 2 × 32 GB is the upgrade that removes it. Nothing here beats a hosted API on
speed — it is a private, offline, zero-marginal-cost setup that is usable, not fast.

## Hardware and software

- Laptop: MECHREVO JIAOLONG Series (16-inch class, AC power, hybrid iGPU)
- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12 GiB VRAM (12227 MiB); driver 615.78.08; VBIOS 98.05.31.00.59; power limit 140 W (default 65 W)
- CPU: AMD Ryzen 9 9955HX 16-core (16 cores, 32 threads, up to 5.39 GHz, AVX-512: avx512f/dq/bw/vl/ifma/vnni/bf16)
- RAM: 2 x 16 GiB Micron CT16G56C46S5.C8D, DDR5-5600, 2 x 1-rank, dual channel, non-ECC; installed 32 GB (30.5 GiB usable)
- Storage: model and packs on Samsung SSD 990 EVO Plus 2 TB (NVMe); second drive YMTC PC411 1 TB
- OS: Arch Linux, kernel 7.2.9-arch1-1; NVIDIA driver 615.78.08; CUDA toolkit 13.4 (V13.4.92, source build of the engine)
- PCIe link width/speed: not measured
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1

## Power and thermal control

- Power, fan and profile control: **TUXEDO Control Center** (`tccd.service`) on the shared Clevo/Uniwill barebone (`clevo_wmi`, `uniwill_wmi`, `tuxedo_io`, `tuxedo_nb02_nvidia_power_ctrl` loaded).
- Active profile: **`PERFORMANCE!!!`** — CPU governor `powersave` with energy-performance preference `balance_performance`, all 32 threads online, scaling 1.93–5.39 GHz; **fans forced on, 100 % minimum and maximum** (profile "Freezy").
- NVIDIA PowerCTRL is available on this unit: default TGP 65 W, maximum 140 W. The GPU drew 73 W on average and peaked at 134.5 W during the run, so the cap never bound.
- TCC itself reports CPU power draw as unavailable (`GetCpuPowerValuesJSON` -> `powerDraw: -1`) on this model, matching the missing k10temp/powercap energy counter.

## Power and thermals during the run

Sampled every 2 s with `nvidia-smi` plus the hwmon CPU sensors while the suite ran
(190 samples under load, GPU > 40 W); raw samples in `telemetry.csv`.

| Quantity | Average | Min | Max |
| --- | ---: | ---: | ---: |
| GPU power (W) | 72.9 | 41.1 | 134.5 |
| GPU SM clock (MHz) | 2259 | 1507 | 2797 |
| GPU temperature (C) | 67.6 | 44.0 | 76.0 |
| GPU utilization (%) | 55.8 | 0.0 | 100.0 |
| VRAM used (MiB) | 11657 | 11523 | 11714 |
| CPU temperature (C) | 87.6 | 69.1 | 93.6 |
| CPU clock (MHz) | 5100 | 3871 | 5391 |

- The GPU cap was set to 140 W; the run drew 73 W on average and peaked at 134.5 W, so the cap was not the binding limit at any point.
- CPU package power is not readable on this platform (`k10temp`/`powercap` expose no energy counter for the Ryzen 9 9955HX here), so CPU is reported as clock and temperature only.
- CPU ran at 5.10 GHz average and up to 93.6 C, i.e. the CPU expert pool was the busiest part of the machine while decode was running.

## Model and configuration

- Model repository and revision: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` at `5348543e0147355ac9cbcb031184a3546350988e` (quant files). MTP block taken from the BF16 checkpoint `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d43b917e7706784d8bb445c9af86a3540` (31 tensors, 5.21 GB).
- Quantization: `IQ1_M` of the Coder family (GSQ-RCO). GGUF filenames: `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` (29,608,446,496 B) and `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` (28,800,138,432 B).
- Vision encoder: none (text-only model; `--vision` not used).
- Custom packs/profiles: preprocessed expert pack `coder-iq1_m` (`tools/iq_pack.py`, native experts sha256 `9ace1b92e1fe12595de469409e69dd52c8d5418c7981c405d9a51aa9d574dd3b`, 25,146,163,200 B); expert cache profile `expert-profile-coder.bin`, pre-loaded with `--expert-profile`.
- Context: 32768 tokens; parallel slots: 1; KV storage `int8`; KV streaming not used (`--kv-resident` off, whole KV in VRAM).
- Cache / low-RAM: `--expert-cache auto` (settles on 1620 resident experts, 3.12 GiB of VRAM), `--prefill auto`, `--resident-experts` (low-RAM mode).
- MTP: `--mtp <mtp dir>` with `--spec 4 --spec-min-p 0.5` (speculative decoding active; draft acceptance 0.73-0.78 in the short-chat runs).
- Reasoning/thinking: off for the suite. Sampling: greedy (temperature 0). Calibration (`--calibrate`): not run. Experimental speed projection: not enabled.
- Settings are the install's own configuration, unchanged - no tuning was applied for this report.

```text
strata --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 32768 --kv int8 --resident-experts
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 20.5 (16.4-23.4) | 2.31 (1.90-2.65) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 387.1 (382.0-392.8) | not measured | 10.44 (10.28-10.58) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 525.3 (520.9-529.4) | not measured | 30.63 (30.39-30.88) |
| agent session, follow-up turns | 14552 (14467-14580) at the last turn | see below | up to 64 | 3 | see below | 20.1 (11.8-22.6) | 10.16 (9.36-11.27) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.5-68.8); earlier-turn tokens read again per session: 4 (4-74) (earlier turns that the cache did not hold; the 13729 (13727-13730) new tokens of the files are read once); first turn (nothing cached) first token: 7.78 s (7.74-8.22).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 75.4 | 8.221 | 13.7 |
| 1 | 3755 | 632 | 3123 | 284.4 | 11.214 | 14.2 |
| 2 | 5288 | 3819 | 1469 | 157.2 | 9.514 | 16.9 |
| 3 | 8700 | 5320 | 3380 | 306.4 | 11.269 | 17.0 |
| 4 | 11626 | 8763 | 2863 | 275.9 | 10.878 | 21.4 |
| 5 | 14552 | 11653 | 2899 | 275.2 | 10.689 | 20.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite - no correctness, needle-recall or quality checks were run.
The conversation uses the model's own replies, so token counts differ a little between runs.

The engine states the limitation itself at startup:

```text
WARNING: the resident RAM mode does not fit (FileExpertSource: resident complement 20.30 GiB
exceeds available RAM (2.97 GiB) minus the 4 GiB safety headroom ... the experts the GPU does not
hold are read from the model folder through the OS file cache (--mmap-experts), which is slower
when the RAM cannot keep them
```

- Repeated runs on this machine measured 14-20 tok/s decode for the same configuration (cold
  page cache at the low end, warm at the high end); the settle point in this report is the warm one.
- Power and thermals were sampled during this run and are reported above; CPU package power is
  `not measured` (unreadable on this platform, see above).
- Total time of the suite: 430 s.
- Not tested: other quantizations, larger contexts, concurrency (`--clients`), KV streaming, and
  the same model on a machine whose RAM holds the resident complement.
