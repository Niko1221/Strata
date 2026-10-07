# Community benchmark: AMD Radeon AI PRO R9700, Ryzen AI 9 HX 370

Measured on 2026-10-07 by [alexhegit](https://github.com/alexhegit). This tests
Strata 0.1.40 with the original Flash-Next IQ2_XS, one GPU, and a 131,072-token
context limit. The desktop is on this GPU, so setup reserved 3,072 MiB of VRAM.

Median decode throughput was **94.1 tok/s at 4,096 prompt tokens and 86.1 tok/s
at 32,768**. These are synthetic code-explanation requests with greedy decoding
and a 256-token output cap. They do not establish general answer quality or
performance on other workloads. A 128,000-token prompt and the recall check were
not run.

## Hardware and software

- AMD Radeon AI PRO R9700 (gfx1201), PCI `0000:c8:00.0`, device `0x7551`.
  The server reported 32,624 MiB of VRAM. The display is connected to this card.
  The negotiated link was 32.0 GT/s, 16 lanes. The engine's startup transfer
  probe reported 7.2 GB/s host-to-device and set `pcie_frac` to 0.20.
  GPU clocks were not fixed. One status sample before the sweep showed 63 C
  and 90 W; that sample is not a load measurement.
- The integrated Radeon 890M (gfx1150) was present and was not used.
- AMD Ryzen AI 9 HX 370, 12 cores / 24 threads, AVX2 and AVX-512. The engine
  used 11 expert-pool workers on logical processors 1–11 and the host thread
  on processor 0.
- Setup's check printed 47 GB of RAM. The server later reported 46.7 GiB.
  Model files were on the NVMe that holds `/home`. Experts loaded at 3.38 GiB/s.
- Ubuntu 24.04.3, kernel 6.17.0-1025-oem, amdgpu DKMS 6.16.13.
- Source commit `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`. Engine 0.1.40,
  compiled here for gfx1201 (`-DSTRATA_ENABLE_HIP=ON`,
  `-DCMAKE_HIP_ARCHITECTURES=gfx1201`, `-DSTRATA_PREFILL_MMQ=ON`).
- The compiler and runtime were ROCm 7.14.0 from rocm-cli's TheRock wheels
  (HIP 7.14.60850, hipBLASLt 1.4.1), selected with `ROCM_PATH`. Strata ships
  no gfx1201 hipBLASLt table for 1.4.1, so prompt GEMMs used plain hipBLAS.
  `/opt/rocm` stayed at 7.2.0 and was not used to compile this engine.
  The kernel driver was not replaced.
- Python 3.12.3 from `/usr/bin` (the virtualenv). The server listened on
  `127.0.0.1:8080` with no API key. Other desktop programs kept running.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, the revision setup pins,
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`:

- `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`
- `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf`

Setup downloaded those files through `HF_ENDPOINT=https://hf-mirror.com`.
The GGUF bytes were not hashed again against the revision's LFS hashes for
this report. MTP is the draft the installer placed under the data directory's
`mtp/rt`. Its upstream revision was not recorded separately.

The unmodified installer prepared the native IQ2_XS pack. Images were off.
The bundled expert profile was used without calibration.

- Context 131,072; INT8 KV.
- Expert cache `auto`: 16,711 slots, 22.41 GiB of VRAM, profile-prefilled,
  **no eviction**. Startup left 25.55 GiB free and reserved 3,072 MiB, plus
  143 MiB for the draft head. After load, 2,982 MiB of VRAM was free.
- Prefill `auto` selected chunks of 8,192 tokens and borrowed 3,107 cache
  slots (4.18 GiB) during the prompt.
- MTP `--spec 4 --spec-min-p 0.5`.
- Experimental speed projection off. Recorded timings have `projection: null`.
- Reasoning disabled, temperature 0, maximum 256 generated tokens per speed run.

The measured configuration is [strata-iq2_xs.json](strata-iq2_xs.json).
Startup lines, before any request, are in [engine-startup.txt](engine-startup.txt).

## Method

The script is the one shipped with the
[RTX 5090 report](../2026-09-30-community-rtx-5090/benchmark.py). It builds a
synthetic Python prompt, counts tokens with Strata's tokenizer, and checks the
count against the server. This run asked only for 4,096 and 32,768 tokens:

```bash
.venv/bin/python bench/results/2026-09-30-community-rtx-5090/benchmark.py \
  --root . \
  --pack /home/alex/HF-MODEL/packs/iq2_xs \
  --url http://127.0.0.1:8080 \
  --out /tmp/strata-iq2xs-bench \
  --targets 4096,32768 \
  --runs 3
```

One short warm-up request was excluded. Three runs at each length ran serially,
shorter length first, on the same loaded engine. All six requests read the whole
prompt: **zero reused tokens**. The expert cache was filled at startup and kept
between requests. Loading time is excluded.

Streaming TTFT is the time from just before the HTTP request until the first
nonempty text delta. Total time ends when the stream finishes. Both include
loopback HTTP. Prompt tok/s is freshly read tokens divided by `prompt_ms`.
Decode tok/s is `engine_generated / decode_ms`.

[initial-status.json](initial-status.json) is the status the script saved before
the sweep. Its `last_timings` field is an earlier 7,253-token request from the
same session, not one of the six runs below.

## Results

Each cell is the median **[minimum–maximum]** of three runs. Every request
generated 256 tokens and stopped at the output limit.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 1,270.3 [1,234.9–1,280.3] | 94.1 [90.6–94.7] | 3.256 [3.231–3.349] | 6.034 [5.933–6.063] |
| 32,768 | 0 | 1,574.1 [1,571.3–1,577.9] | 86.1 [85.3–94.2] | 20.875 [20.837–20.927] | 23.824 [23.575–23.881] |

Per-run records, including draft counts and the answer text:
[results.json](results.json). Aggregates: [summary.json](summary.json).
Decode expert-cache hit rate on these six requests was 96.6–99.2%.

At 4,096 tokens the engine accepted 157, 157, and 156 drafts (of 219, 218, and
227). At 32,768 it accepted 144, 145, and 157 (of 192, 210, and 212).

## Not measured

128,000-token prompts and `tools/needle_bench.py` were not run. When the sweep
started, the server reported 38.6 GiB of 46.7 GiB RAM in use.

This is one machine, one quantization, and a small synthetic workload. Long
output, sampled decoding, thinking, coding-task correctness, vision, tool use,
concurrency, and a sustained thermal run were not evaluated. No second GPU was
measured on this PC, so these numbers do not replace the RTX 5070 row in
[DETAILS.md](../../../docs/DETAILS.md), the RTX 5090 report linked above, or the
R9700 Coder IQ1_M table in [AMD_HIP.md](../../../docs/AMD_HIP.md). Those used
other engines, CPUs, and protocols.
