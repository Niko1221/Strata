# Community benchmark: RX 7900 XTX (gfx1100), Qwen3.8-Flash-Next IQ2_XS, 30 GB RAM

Measured on 2026-10-21 by [@blastpepsi1-eng](https://github.com/blastpepsi1-eng) with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. The speed suite is followed by a needle recall check (below).

**What was tested:** how one RX 7900 XTX with 30 GB of RAM runs the original Qwen3.8-Flash-Next IQ2_XS on Strata 0.1.45 with a tuned, day-to-day configuration, through the standard `bench.sh --long` suite. **Headline:** decode 87.3 tok/s in short chats (117-123 on the code question, 83-89 on prose) and 99.7 in the agent session; fresh prompts 802 / 1,280 / 1,462 tok/s at 4K / 16K / 32K; needle recall 9/9 up to 56K; no failures. **Main limitation:** the configuration is not the default (gfx12 switch sets and route tail skip on gfx1100, the latter changes answers) and the engine carries local patches, so these numbers describe this tuned setup, not a stock install; the 30 GB of RAM also makes decode vary between engine starts.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX (gfx1100, RDNA3), 24 GiB VRAM; PCIe 4.0 x16; power cap 291 W (board maximum 350 W)
- CPU: AMD Ryzen 9 7950X 16-Core Processor (16 cores, 32 threads); installed RAM: 30.4 GiB DDR5 (32 GB nominal); storage: Kingston NV2 2 TB NVMe (packs and GGUFs on it); 30 GiB zram swap
- OS: Linux 7.2.9-2-cachyos; amdgpu driver from that kernel, linux-firmware 20260916; ROCm 7.2.4 (Arch/CachyOS packages)
- Strata: project 0.1.45, commit 477ac13c **plus uncommitted local changes** (listed under "Local changes"); engine 0.1.45, local HIP source build for gfx1100
- Background workloads: a normal desktop session (KDE Plasma, Firefox, VS Code, Discord) stayed open during the run

## Model and configuration

- Model and quantization: the original Qwen3.8-Flash-Next (not Coder) as ISTA DASLab's GSQ-RCO IQ2_XS, [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (the revision setup pins), installed by `setup.sh`; experts IQ2_S gate/up + Q2_0 down, dense projections IQ4_XS / IQ3_S
  - `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`, 39,225,954,592 bytes, SHA-256 `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7`
  - `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf`, 28,800,138,432 bytes, SHA-256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`
  - vision: `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, 907,543,008 bytes, SHA-256 `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0` (CPU encoder)
  - hashes are the repository's published LFS SHA-256 at that revision; the local files match its byte sizes and were not re-hashed. Pack `iq2_xs` and `data/expert-profile.bin` as setup made them; MTP draft layer from setup
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own day-to-day configuration (hip backend), **not the defaults** (see "Non-default switches"); thinking off for the suite; temperature 0
- Memory: the server runs inside a systemd user scope, `MemoryHigh=23G MemoryMax=25G MemorySwapMax=2G` (with an earlier 20G cap, 56K prompts stalled)
- Memory in this run (engine log): 18.67 GiB of experts resident in RAM (pageable), 11,836 experts in the VRAM cache (about 15.9 GiB) with 3,072 MiB of VRAM kept free for the desktop; 23.1 of 24.0 GiB of VRAM in use on the card (engine plus desktop, `rocm-smi`, read while the same configuration served right after the run); the scope's memory use stayed between about 18 and 21 GiB in a 15-minute stress run the same day, swap 1.1-1.7 GiB

```text
strata --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 4096 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 65536 --kv int8 --resident-experts --vision --vram-reserve-mib 3072 --pcie-frac 0 --pool-tasks 96 --adapt-swaps 0
```

Environment of the config: `STRATA_HIP_WMMA=1`, `STRATA_SELECT_WMMA=1`, `STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100202.txt`, `STRATA_RESIDENT_PIN=0`, `STRATA_HC_Q8=1`, `STRATA_TSUM=1`, `STRATA_ROUTE_TAIL_SKIP=7`, `STRATA_VERIFY_COHERENT=1`, `STRATA_DOORBELL_STORE=1`, `STRATA_FUSE_WAIT=1`, `STRATA_FUSE_HITS=1`, `STRATA_QFUSE_HC=1`, `STRATA_GR_NORM_FAST=1`, `STRATA_MTP_HIP_PRIO=1`, `STRATA_GDN_HEAD=1`, `STRATA_GDN_PP=2`, `STRATA_GDN_CONVL2=1`, `STRATA_GDN_NOY=1`, `STRATA_CVEC_FUSE=1`, `STRATA_TOPK_REG_MIN_BLOCKS=4096`


### Non-default switches

The configuration above turns on several switches that are not defaults on gfx1100. Each was measured on this machine first (decode tok/s from alternating A/B runs of the same 6 greedy prompts; prompt tok/s from fresh 16K/32K prompts):

| Switches | Upstream default | Effect here | Output |
| --- | --- | --- | --- |
| `STRATA_ROUTE_TAIL_SKIP=7` | on for CUDA and gfx12, off on gfx1100 | decode +15% (84.1 -> 96.9 tok/s median) | **changes answers**: teacher-forced KL 0.036 vs skip off, top-1 agreement 93.95% (`tools/quant_quality.py`, 5 x 1,000 tokens); perplexity of the forced text 4.43 vs 4.53. Larger than the CUDA figures in DETAILS.md (KL 0.003-0.006), presumably because about half of the experts are not in VRAM here |
| `STRATA_FUSE_WAIT=1 FUSE_HITS=1 QFUSE_HC=1 GR_NORM_FAST=1 MTP_HIP_PRIO=1` | gfx12 defaults | decode +3% mean / +4.5% median | unchanged (same drafts accepted) |
| `STRATA_GDN_HEAD=1 GDN_PP=2 GDN_CONVL2=1 GDN_NOY=1 CVEC_FUSE=1 TOPK_REG_MIN_BLOCKS=4096` | gfx12 / gfx1151 prompt defaults | prompt +1.9% at 16K and 32K | exact per upstream |
| `STRATA_HC_Q8=1` | gfx12 default | decode +2-4 tok/s (measured 2026-10-19) | slight: KL 0.0018 |
| `STRATA_VERIFY_COHERENT=1 STRATA_DOORBELL_STORE=1` | #649 opt-ins | kept: with either one removed, 15-minute stress runs stalled ("no progress for 60 s ... waiting for the GPU") on 32K-56K prompts; those runs were under memory pressure, so the cause is not fully isolated | unchanged |

`--pcie-frac 0 --adapt-swaps 0` are deliberate: on this card the adaptive expert tier and a PCIe share above 0 led to GPU hangs (verify TIMEOUT) under memory pressure.

### Local changes

Uncommitted changes on top of 477ac13c in the engine that was measured:

- `src/core/arch_defaults.cpp` (+ test): a gfx1100 table with `STRATA_TSUM=1` (opt-out `STRATA_GFX1100_DEFAULTS=0`)
- `src/spec/controller.cpp`, `src/program/generate.cpp`: the speculation controller wired into the decode loops behind `STRATA_SPEC_CONTROLLER=1` (off in this run)
- `src/prefill/gemm.cu`: hipBLASLt solutions only for exactly calibrated zero-beta rows with 256-byte-aligned operands, hipBLASEx otherwise
- `serve/` (document imports in the web app) and `tools/strata_bench.py` (a one-line fix for starting the server); no effect on the measured paths

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests. Cache state: a freshly started engine; the VRAM expert cache was pre-filled at start from the shipped routing profile (`data/expert-profile.bin`, 11,836 experts) and then adapted by use only through the suite's own requests (`--adapt-swaps 0`: no expert swaps); the experts not in VRAM were resident in RAM (no file reads during decode). Only the agent session reuses earlier tokens, by design.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 87.3 (78.9-122.9) | 0.28 (0.25-0.30) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 802.0 (688.1-806.2) | not measured | 5.03 (5.00-5.86) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1279.7 (1255.5-1289.6) | not measured | 12.57 (12.48-12.82) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1462.5 (1436.1-1464.4) | not measured | 21.95 (21.92-22.35) |
| agent session, follow-up turns | 14558 (14556-14579) at the last turn | see below | up to 64 | 3 | see below | 99.7 (70.7-120.8) | 4.01 (2.49-9.51) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

Failures: none. All 36 measured requests (9 chat, 9 fresh prompts, 18 agent turns) and the 9 needle requests completed; nothing was skipped, cancelled or excluded (`results.json`: `failures: []`, `skipped: {}`).

The short-chat median hides a split by question (from `results.json`): the code question (a Python merge function) decodes at 117-123 tok/s with 77% of the MTP drafts accepted; the prose and list questions at 83-89 tok/s with about 50-54% accepted. Six of the nine requests are prose or list, so the median is 87.3.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 4 (3-74) (earlier turns that the cache did not hold; the 13718 (13717-13718) new tokens of the files are read once); first turn (nothing cached) first token: 1.58 s (1.56-2.02).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 309.8 | 2.017 | 69.6 |
| 1 | 3741 | 620 | 3121 | 340.9 | 9.505 | 113.7 |
| 2 | 5274 | 3804 | 1470 | 375.1 | 3.945 | 95.1 |
| 3 | 8707 | 5267 | 3440 | 739.8 | 4.679 | 94.2 |
| 4 | 11619 | 8757 | 2862 | 695.1 | 4.143 | 103.1 |
| 5 | 14558 | 11659 | 2899 | 722.3 | 4.038 | 92.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt`; the engine's own timing line for every request of the suite, plus its memory and cache lines: `engine-timing-lines.txt`. The commands of this run (speed suite and needle check inside the memory scope): `run_report.sh`.

## Correctness and limitations

Needle recall (`tools/needle_bench.py --lengths 8k,32k,56k --depths 10,50,90`, a fresh server after the speed suite): **9 of 9 found** (8K: 6,920 prompt tokens; 32K: 32,791; 56K: 57,281). Per-needle data: `needles.json`.

Compared with the same machine on 2026-10-18 (engine 0.1.45, the configuration of that day): chat decode 59.8 -> 87.3 tok/s, agent decode 75.5 -> 99.7, 4K prompt 338.7 -> 802.0, 16K 1103.4 -> 1279.7, 32K 1323.6 -> 1462.5. The gains come from the configuration changes since (resident experts, the switches above), not from a newer engine.

The short-chat decode numbers spread widely (78.9-122.9) and move by up to about 10% between engine starts on this machine; the 30 GB of RAM is the limit: when the scope's 2 GiB of swap fills, part of the pageable resident experts is paged out and decode drops for a few requests. 64 GB would remove that.

The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `rocm-smi` and the OS; the tool read the GPU only as "AMD Radeon", so the GPU name, ROCm version, PCIe link, power cap, storage, background load, the needle results and the sections on switches and local changes were added by hand after the run (`results.json` has the same machine facts). Total time of the suite: 218 s.
