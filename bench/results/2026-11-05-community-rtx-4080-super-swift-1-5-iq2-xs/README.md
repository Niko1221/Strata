# Community benchmark on NVIDIA GeForce RTX 4080 SUPER

Measured on 2026-11-05 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4080 SUPER, 16 GiB VRAM
- CPU: Intel(R) Core(TM) i9-14900KF (24 cores, 32 threads); installed RAM: 31.7 GiB; storage: SAMSUNG MZVL21T0HCLR-00BL7 SSD/NVMe
- OS: Windows 11 (10.0.26300); GPU driver: 621.01; CUDA (driver supports up to): 13.5
- Strata: Windows Desktop GUI (`strata-desktop.exe` SHA-256 `9F17A2BF...`, release `0.2.9`); engine `0.1.56` (`source: f-0291-rc1`, `commit: 06698c674673`, `cuda: 13.0`, `archs: [75, 86, 89, 120]`, see `BUILD.json`)
- GPU detail (after the run, near idle): power_limit_w=320.00, power_draw_w=126.94, pcie=gen4 x16, pstate=P2, temp_c=53.29
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: swift-1.5-iq2_xs (`UkisAI/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, unpruned 512 experts/layer)
- Context: 131072 tokens; parallel slots: 1
- Settings: tuned configuration for 16 GiB VRAM + 31.7 GiB RAM (Windows `SeLockMemoryPrivilege` 2 MiB Large Pages enabled, fixed 32 GiB pagefile; see `strata-swift-iq2_xs.json`); vision off (`--vision` omitted); vision off (`--vision` omitted); thinking off for the suite; temperature 0

```text
component:strata-engine --pack swift-iq2_xs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --spec 4 --mtp rt --max-context 131072 --kv int8 --spec-min-p 0.55 --resident-budget-gib 19 --prefill 2048 --kv-resident 32768 --adapt-async 1 --pcie-frac 0 --pool-workers 12 --vram-reserve-mib 896
```

Environment of the config: `STRATA_UNBUFFERED_LOAD=1`, `STRATA_RESIDENT_HEADROOM_GIB=3`, `STRATA_PIN_RESERVE_GIB=2`, `STRATA_PROMPT_DIRECT=1`, `STRATA_PREFILL_DIRECT_RAM=2`, `STRATA_PREFILL_STREAM_MIN=4096`, `STRATA_MMAP_PREFETCH=2`, `STRATA_COLD_OVERLAP=0`, `STRATA_MMQ_DIRECT=1`, `STRATA_MMQ_GROUP_SORT=1`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a freshly restarted Strata server (0 prior requests); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every fresh-prompt test starts with a unique tag so the prefix cache is bypassed (`0` reused tokens), while the adaptive VRAM expert cache (`--adapt-async 1`) remains warm across the suite.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K / 32K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 88 (85-93) | 0 | 100 (100-100) | 9 | not measured | 61.3 (56.6-86.6) | 0.99 (0.89-1.07) |
| 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 1128.1 (1122.5-1139.8) | not measured | 3.62 (3.57-3.63) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1352.8 (1352.1-1355.1) | not measured | 11.91 (11.89-11.93) |
| 32K prompt, fresh | 32050 (32050-32050) | 0 (0-0) | 24 cap | 3 | 1401.5 (1374.7-1402.6) | not measured | 22.92 (22.90-23.38) |
| agent session, follow-up turns | 9097 (9054-9106) at the last turn | see below | up to 64 | 3 | see below | 99.0 (77.9-112.0) | 2.32 (1.92-2.49) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 71.0 % (71.0-71.1); earlier-turn tokens read again per session: 73 (5-74) (earlier turns that the cache did not hold; the 8207 (8206-8208) new tokens of the files are read once); first turn (nothing cached) first token: 2.24 s (2.20-2.26).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 647 | 0 | 647 | 300.7 | 2.202 | 62.2 |
| 1 | 2301 | 661 | 1640 | 838.9 | 1.982 | 77.9 |
| 2 | 4007 | 2364 | 1643 | 776.4 | 2.177 | 85.5 |
| 3 | 5713 | 4000 | 1713 | 702.6 | 2.474 | 99.2 |
| 4 | 7399 | 5757 | 1642 | 684.5 | 2.427 | 112.0 |
| 5 | 9097 | 7455 | 1642 | 676.0 | 2.449 | 102.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt`.

## Provenance

- Engine `BUILD.json`: `version: 0.1.56`, `source: f-0291-rc1`, `commit: 06698c674673`, `cuda: 13.0`, `archs: [75, 86, 89, 120]`
- `strata-desktop.exe`: SHA-256 `9F17A2BF6187A7165EA9CF458946E4ABA73F25DD24E74BFD5A7025404B8CB7DF`
- `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`: SHA-256 `CA3B302D3D1EB6EDDCE3EA4ACC79854E6CC5FDC0583E374A6A8B6BB67327743B`
- `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf`: SHA-256 `E4996A6DE18FD49C58D9A0040FCFD73787DB6407C51973F29884BEE7B14C9DBB`
- `mtp-q2_0.gguf`: SHA-256 `A22206A4D74EAFBF31063A84610292535FB3B49AF9A339963817FD04AED81E4D`

## Same-machine A/B progression and 887-request session (131K context)

Snapshots from live agentic coding on the same machine prior to the synthetic suite above. Each pass started from a fresh engine restart; `Peak VRAM` is peak allocated VRAM during inference, and `Decode tok/s` separates single-turn peak from multi-turn session averages so short tests and long sessions are not conflated:

| Configuration | Cached experts | Peak VRAM | Prompt tok/s | Decode tok/s (peak / session avg) | Context depth & hardware behavior |
| --- | ---: | ---: | ---: | ---: | --- |
| Pass 1: setup baseline | 5,973 | 15.3 / 16.0 GiB | 291.0 avg | 40.5 avg (5 reqs) | 15K-25K context; default settings before Large Pages and prefill tuning |
| Pass 5: `STRATA_UNBUFFERED_LOAD=0`, no Large Pages | 5,943 | 15.3 / 16.0 GiB | 147.0 avg | 38.0 avg (10 reqs) | 15K-25K context; OS Standby list filled 31.7 GiB RAM, `3.8 MB/s` pagefile writes (#649) |
| Pass 11: Large Pages + `STRATA_UNBUFFERED_LOAD=1` | 6,252 | 15.7 / 16.0 GiB | 413.0 avg | 83.9 peak (10 reqs) | <25K context; 19 GiB locked in 2 MiB pages, SSD reads cut from 1,600 to 811.9 MB/s |
| Pass 13: `--vision` off + `--spec-min-p 0.6` | 6,254 | 15.7 / 16.0 GiB | 569.0 avg | 89.4 peak (10 reqs) | <25K context; SSD reads 590.7 MB/s, 0.0 MB/s pagefile write |
| Pass 16: `--vram-reserve-mib 512` (WDDM spill) | 6,511 | 15.9 / 16.0 GiB | 407.0 avg | 73.9 peak -> 11.1-15.9 | Held 73.9 tok/s at 25K, then crossed 15.85 GiB WDDM cliff at 53K and spilled to shared RAM |
| Pass 18: `--prefill 4096` + `--vram-reserve-mib 768` | 6,442 | 15.8 / 16.0 GiB | 1,349.0 peak | 96.3 peak / 41.8 avg (79 reqs) | 95.1K cold prompt in 76 s (`1,349 tok/s`), but `1.78 GiB` slot borrow drifted to 41.8 tok/s over 79 reqs |
| Tuned lock (`--prefill 2048 --vram-reserve-mib 896`, vision off) | ~6,200 | 15.55 / 16.0 GiB | 769.0 peak / 611.0 avg | 93.2 peak / 55.2 avg (887 reqs) | Zero WDDM spill across 887 requests (24K-100.7K context, 53.4M reused tokens; checkpoints below) |

### Multi-turn coding checkpoints on the tuned lock configuration (continuous session, vision off)

Cumulative averages from the Strata API monitor at four points during the same continuous coding run (turn durations are engine-measured request times):

| Cumulative session checkpoint | Completed requests | Cumulative reused tokens | Session avg prompt tok/s | Session avg decode tok/s | Single-turn speed & context depth recorded at checkpoint |
| --- | ---: | ---: | ---: | ---: | --- |
| 10-request check | 10 | 295K | 702.0 | 77.3 | 93.2 tok/s decode at 24,570 prompt tokens; 89.2 tok/s (5.0 s turn) at 50,986 tokens |
| 100K+ context climb | 34 | 1.9M | 605.0 | 46.3 | 52.3 tok/s (13.4 s turn) at 97,894 tokens; 44.8 tok/s (1,451 output tokens in 35.5 s) at 100,715 tokens |
| 503-request check | 503 | 30.6M | 589.0 | 58.1 | 63.6-75.9 tok/s decode (2.9-5.6 s turns) across 40,046-57,505 tokens; +6,075 tokens prefilled at 695 tok/s (56.9K context) |
| 887-request marathon | 887 | 53.4M | 611.0 | 55.2 | +77,502 tokens prefilled at 769 tok/s at 85,516 context (52.8 tok/s decode); 60.1-67.9 tok/s at 48K-69.3K context (#878-#887) |

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 199 s.