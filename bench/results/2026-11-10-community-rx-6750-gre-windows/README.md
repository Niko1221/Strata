# Community benchmark: AMD Radeon RX 6750 GRE 12 GB (gfx1031), Ryzen 5 5600, Windows 11

Measured on 2026-11-10 (starting 09:44 UTC / 17:44 UTC+8) on a Windows desktop, following the method of
[2026-10-05-community-rx-6700-xt](../2026-10-05-community-rx-6700-xt/README.md) with the same `benchmark.py`
(SHA-256 `694df81b3d064ca609c73dc8606baf6cbb8a4837ea800933ee6f2eb80dc52d9a`, unmodified). This report tests the
original Flash-Next **IQ3_S** (the reference report used IQ2_XS) with a 131,072-token context on one RDNA2 card,
using the ready-made Windows `rdna2` engine package, which carries native gfx1031 code — no
`HSA_OVERRIDE_GFX_VERSION` was set.

The median decode throughput was **22.5 tok/s at 4,096 prompt tokens, 24.3 tok/s at 32,768 and 24.3 tok/s at
128,000**. Prompts were read at 559, 604 and 647 tok/s. The speed requests are synthetic code-explanation
requests with greedy decoding and a 256-token output cap. They do not establish general answer quality or
performance on other workloads.

## Hardware and software

- **GPU:** AMD Radeon RX 6750 GRE, 12 GB, gfx1031 (Navi 22, wave32), run natively as gfx1031 by the Windows
  `rdna2` package. Driver 32.0.21045.16010 (2026-10-16). PCIe link as Windows reports it, read while idle:
  current link speed 4 (16.0 GT/s) x16, maximum the same. Power and temperature were not recorded (no sysfs
  on Windows; the telemetry samples GPU engine utilization, dedicated VRAM and system RAM only).
- **CPU and RAM:** AMD Ryzen 5 5600 (6 cores, 12 threads, AVX2, no AVX-512); the engine used 5 expert-pool
  workers plus its host thread (one worker per physical core but the host's). 64 GB DDR4-2667 in three DIMMs
  (32 + 16 GiB on channel A, 16 GiB on channel B — asymmetric), speed and layout from SMBIOS.
- **Storage:** both model shards, including the 28.8 GB n-gram (PLE) shard, are on one NVMe SSD (HYV1TB,
  932 GB, mounted as F:). The engine's PLE direct-read probe measured 64,543 rows/s at start ("fine").
  Loading the 46.84 GiB expert arena into RAM took about 63 s at 0.75 GiB/s (excluded from every timing below).
- **OS and runtime:** Windows 11 Pro, build 26300, WDDM. Engine: ready-made package
  `0.1.57-windows-x86_64-hip-rdna2` (prebuilt, commit `0bdcef4682c5`), ROCm 10.2.0a20260930 (TheRock),
  archs gfx1030 + gfx1031. The server and benchmark ran from a source checkout at `9f7d977d` with its `.venv`
  (the engine itself is the prebuilt binary, not compiled from that checkout).
- **Other services:** an ordinary desktop session; no coding agent was connected during the run.

## Model and configuration

**Model:** the original Qwen3.8-Flash-Next from
[`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
IQ3_S, downloaded by setup on 2026-10-08. The native pack and the MTP draft layer (`mtp-q2_0.gguf`) were
prepared by setup. No vision encoder in this run, no custom pack, the bundled expert profile
(`data/expert-profile.bin`, 24,576 ranked pairs). SHA-256 of the shards as used
([sha256sums.txt](sha256sums.txt)):

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`

The second shard's hash is identical to the IQ2_XS report's: the 28.8 GB n-gram table is quant-independent.

**Configuration** ([strata-iq3_s-bench.json](strata-iq3_s-bench.json)):

- context 131,072, `--kv int8 --kv-resident 32768` (KV streaming: 32,768 of 131,072 cells per QSA layer in
  VRAM, the K/V in 1.55 GiB of pinned RAM);
- expert cache `auto`: **1,906 expert slots (3.67 GiB)** in VRAM after the Windows budget and reserves;
  989 MiB of VRAM free with everything loaded. Windows budgets 11,475 of the card's 12,272 MiB for the
  process. The slots were pre-filled from the bundled profile (PROFILE policy, ranked by routing frequency,
  no eviction) — the adaptive cache was **not** warmed by prior agent traffic, unlike the reference report;
- `--prefill auto`; MTP `--spec 4`, **`--spec-min-p 0.7`** and **`--pcie-frac 0.2`** (the reference report's
  calibrated values, applied by hand here; no calibration was run on this machine);
- `--vram-reserve-mib 1280` (kept from this machine's daily Windows configuration; the reference had no
  explicit reserve);
- environment: `STRATA_HIP_PROMPT_F16=1`, `STRATA_SELECT_SIMT=1`, `STRATA_PREFILL_CPU_SHARE=auto` (this
  machine's daily settings; engine 0.1.57 also defaults the simt2 QSA selector on gfx103x via
  `STRATA_AMD_COMMUNITY_TUNING`);
- `fit_max_tokens` true; reasoning off (`reasoning_effort: none`), temperature 0, 256 generated tokens per
  speed run.

Launch:

```bat
.venv\Scripts\python.exe serve\server.py --engine strata --config strata-iq3_s-bench.json --port 8080
```

## Method

The same [benchmark.py](benchmark.py) as the reference report, unmodified. It builds deterministic synthetic
Python filler, puts a different nonce at the start of each request, and counts the rendered chat prompt with
Strata's tokenizer. One short warm-up request is excluded. Three runs at each of 4,096 / 32,768 / 128,000
prompt tokens ran serially in increasing-length order on the same loaded engine; every request reads its
whole prompt (zero reused tokens). TTFT is streaming over loopback; prompt throughput is freshly read tokens /
`prompt_ms`; decode throughput is `engine_generated / decode_ms`; loading time is excluded.

Windows-specific notes:

- The benchmark ran with `PYTHONUTF8=1`. The script's `read_text()` calls use the OS default encoding, which
  is GBK on this Chinese-locale Windows and mangles the UTF-8 vocabulary; UTF-8 mode reproduces the Linux
  behavior without changing the script.
- [monitor_windows.ps1](monitor_windows.ps1) replaces the reference's `monitor_amd.py` (amdgpu sysfs): it
  samples system RAM, dedicated VRAM and GPU engine utilization (3D + Compute) once per second into
  [telemetry.jsonl](telemetry.jsonl). Power and temperature are not available through Windows counters.
- The needle test was **not** run; this report is the speed matrix only.

## Results

Each cell is the median **[minimum-maximum]** of three runs. Every request generated 256 tokens and stopped at
the output limit (`finish: length`). No speed request failed or was cancelled; all nine read their whole prompt
(zero reused tokens). The speed matrix ran 09:48:39-10:03:22 UTC on the engine session started 09:44:56 UTC.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 559.4 [546.4-563.5] | 22.5 [20.3-24.8] | 7.53 [7.34-7.58] | 18.92 [17.61-19.94] |
| 32,768 | 0 | 603.9 [603.6-620.2] | 24.3 [23.3-25.3] | 54.37 [52.94-54.40] | 64.46 [63.43-65.28] |
| 128,000 | 0 | 646.9 [646.4-647.6] | 24.3 [22.6-25.3] | 198.03 [197.84-198.19] | 208.49 [208.25-209.08] |

Decode speed varied between runs of the same prompt (20.3-24.8, 23.3-25.3 and 22.6-25.3 tok/s). The draft
layer's acceptance was 75-82% (132-147 of 173-194 offered guesses; the counts for every run are in
[results.json](results/results.json)). The resident-expert cache hit rate was 44.7-52.6% per request (47.6%
for the whole session, 62,763 of 131,826 lookups); 6.6% of all routed experts were read over PCIe. KV
streaming: 96.08% of block reads hit VRAM, 291.8 MiB read from RAM over the session. The end-of-session lines
are at the bottom of [engine.log](engine.log).

**Memory** (whole-machine values from [telemetry.jsonl](telemetry.jsonl), 273 one-second samples): RAM in use
58.4-60.2 GB, median 59.6, of 63.9 GiB visible. Dedicated VRAM 10,997-11,087 MB. GPU engine utilization
(3D + Compute summed, so it can exceed 100%): median 105%, maximum 118%. The first ~40 samples predate a
monitor restart and only counted the 3D engine, so utilization in that interval is under-reported; there is a
few-second gap at the restart. Power and temperature were not recorded. No out-of-memory event occurred.

**Needle recall:** not run; this report is the speed matrix only.

## Differences from the reference report

| | Reference (RX 6700 XT) | This run (RX 6750 GRE) |
| --- | --- | --- |
| Quantization | IQ2_XS | **IQ3_S** |
| OS / runtime | Ubuntu 24.04, Docker, ROCm 7.2.4, source build | Windows 11, ready-made package, ROCm 10.2.0a20260930 |
| Engine | 0.1.39 (commit 6f32ec0) | 0.1.57 (commit 0bdcef4682c5) |
| GPU code path | gfx1031 overridden to gfx1030 | native gfx1031 |
| Expert cache | 4,580 slots (6.29 GiB), warm adaptive | 1,906 slots (3.67 GiB), bundled profile, cold |
| CPU workers | 7 (Ryzen 7 5700G, 8 cores) | 5 (Ryzen 5 5600, 6 cores) |
| PLE shard storage | separate NVMe bind-mount | same NVMe as the experts |
| Prompt env | defaults of 0.1.39 | FP16 prompt GEMMs + SIMT selector on |

The two cards share the same Navi 22 silicon (40 CUs, 192-bit, 384 GB/s); the 6750 GRE 12GB is the higher-clock
variant. The quantization difference alone reads roughly 50% more bytes per token (IQ3_S ~3.4 bpw vs IQ2_XS
~2.2 bpw), so decode numbers are not directly comparable across the two reports.