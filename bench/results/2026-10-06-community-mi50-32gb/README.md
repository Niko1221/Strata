# Community benchmark: AMD MI50 32 GB (gfx906), Strata 0.1.40.1, IQ2_XS: 128K and 252K needles, a 16 GB-limit run, temperatures

Measured on **2026-10-06** by [mathcuei](https://github.com/mathcuei), with the runs driven by Claude Code, on a Linux
workstation with one MI50 32 GB. This tests Strata **0.1.40.1** (`82f46a8`) with the original Flash-Next **IQ2_XS**
at a **131,072-token** context and at **262,144** (`--max-context 262144`, no rope scaling), with KV streaming
(`--kv-resident 32768`, the installer's default for this card).

Decode was **37-55 tok/s** on short prompts, **44.4 / 41.0 tok/s at 34K / 68K** of context, and prompt processing held
**409-428 tok/s from 8K to 119K tokens**. With the full 32 GB, the three-needle check passed at **126K (3 of 3)** and at
**252K (3 of 3)**, with decode **42.5 tok/s at 126K** and **36.3 tok/s at 252K** (second request, prefix reused). The same
check with the card limited to 16 GB passed at 126K (31.2 tok/s) but **failed at 252K: the answer was `!!!!...` (0 of 3)**; that
one was not reproduced, including on 0.1.40.3 and 0.1.41 (see [retest-2026-10-08](retest-2026-10-08/README.md)). One run per case on synthetic text: these numbers do not establish answer quality or
performance on other workloads.

It complements the existing [MI50 run](../2026-10-04-community-mi50/) (0.1.38, a different host, `--spec 3`): this one is a
later engine version, adds 252K, the 16 GB-limit run and temperature readings, and uses an AVX-512 host.

## Hardware and software

- AMD Instinct MI50 32 GB (Vega 20, gfx906), 34,342,961,152 bytes of VRAM; PCIe probe by the engine: 25.1 GB/s host to device.
  No power cap was set by us and the clocks were not changed.
- Intel Core i9-11900H (the OS reports it as "Genuine Intel(R) CPU 0000 @ 2.60GHz", up to 4.8 GHz), 8 cores / 16 threads, AVX-512
  (`avx512f/bw/vl/vnni` and more); 62 GB of RAM; Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic. The engine used 7 expert-pool
  workers on logical processors 1-7 and the host thread on 0.
- ROCm **7.2.2** (AMD clang 22.0.0git, `roc-7.2.2`). `lld` needed a compatible `libxml2` on `LD_LIBRARY_PATH` on this OS.
- Strata `v0.1.40.1`, commit `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`, built with `-DSTRATA_HIP_GFX906=ON` as in
  [AMD_HIP.md](../../../docs/AMD_HIP.md) (see [build-gfx906.sh](build-gfx906.sh); the engine reports version 0.1.40).
  Three small local fixes were needed for the build (see Notes); the diff is [local-patches.diff](local-patches.diff).
- Nothing else used the GPU. No system setting (power, driver, kernel) was changed.

## Model and configuration

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, `IQ2_XS` (the GGUF already on the machine; the pack and the MTP layer were built by
the setup flow). Vision off, temperature 0, no reasoning. The complete configurations are in
[strata-iq2_xs.json](strata-iq2_xs.json) (32 GB, 131,072), [strata-iq2_xs-ctx262144.json](strata-iq2_xs-ctx262144.json),
[strata-iq2_xs-16gb.json](strata-iq2_xs-16gb.json) and [strata-iq2_xs-16gb-ctx262144.json](strata-iq2_xs-16gb-ctx262144.json).
Common settings: KV int8, `--kv-resident 32768` (32,768 of 131,072 cells per QSA layer in VRAM, the K/V in 1.55 GiB of pinned
RAM at 131,072), expert cache `auto`, prompt chunk `auto` (8,192), MTP `--spec 4 --spec-min-p 0.5`.

- **32 GB:** **19,473 of 24,576 experts (~79%, 26.17 GiB)** in VRAM, pre-filled from the profile, no eviction; decode cache hit
  rate **95.8-98.5%** (95.8-95.9% in the short decode right after the long prefill, 98.3-98.5% in the longer decode that follows); about 39 GB of RAM in use.
  Loading the 33.02 GiB of experts took ~1 min (3.68 GiB/s).
- **"16 GB":** `--vram-reserve-mib 16384` (= 32,752 - 16,368, what a 16 GB MI50 exposes): **8,099 slots (~33%, 10.85 GiB)**,
  hit rate **73-91%**. This is a **simulation by Strata's reserve, not a real 16 GB card.**

## Results

Server-side timings (the engine's own `strata serve:` lines, in [engine-32gb.log](engine-32gb.log) and
[engine-16gb.log](engine-16gb.log)); client-side outputs in the `bench-*.out` and `ctx-suite*.out` files (their labels are in
Portuguese: `COMPLETO 32GB` = full card, `LIMITADO 16GB` = the reserve run).

### Speed and short needles (single runs)

| Case | 32 GB | 16 GB (simulated) |
|---|---:|---:|
| Decode, short prompt (4 requests) | 37.2 / 54.2 / 55.1 / 40.6 tok/s (MTP drafts accepted 47-76%) | 30-34 tok/s |
| Decode at 34K context (350 tokens) | 44.4 tok/s | 35.7 tok/s |
| Decode at 68K context (350 tokens) | 41.0 tok/s | 38.9 tok/s |
| Prefill, 26-55-token prompts | 63-79 tok/s | 43-44 tok/s |
| Prefill, 8K to 119K tokens | **409-428 tok/s** (client-side; 410-430 server-side) | **394-419 tok/s** |
| Needle in the middle, 2K to 119K tokens | 6 of 6 | 6 of 6 |

### Long context: three needles at 10 / 50 / 90 % depth (6-digit codes), prompt near each limit

| Mode | Context | Prompt tokens | Prefill | Needles | Decode with the full context |
|---|---|---:|---:|---|---:|
| 32 GB | 131,072 | 125,991 | 409 tok/s (308 s) | **3 of 3** | **42.5 tok/s** (hit 98.3%) |
| 32 GB | 262,144 | 252,304 | 370 tok/s (681 s) | **3 of 3** | **36.3 tok/s** (hit 98.5%) |
| 16 GB | 131,072 | 125,991 | 403 tok/s (313 s) | **3 of 3** | **31.2 tok/s** (hit 91.2%) |
| 16 GB | 262,144 | 252,304 | 382 tok/s (660 s) | **0 of 3: `!!!!...`** | 256 tokens generated at 26.8 tok/s, no draft accepted |

The second request re-sent the same text with another question and reused the first one's prefix (114,688 tokens at 126K,
245,760 at 252K), so the "decode" column is decode, not prefill. The needle harness is [bench-ctx.py](bench-ctx.py)
(`bench-quick.py`, `bench-long.py` and `bench-decode.py` made the short and 2K-119K runs).

### Temperatures (junction, from `rocm-smi`, every 10 s; memory temperature was **not** recorded)

[ctx-temps.csv](ctx-temps.csv) and [ctx-temps2.csv](ctx-temps2.csv) hold the readings of the long-context runs. Prefill of 8K-119K
reached **99-101 C junction at 155-200 W** (edge 78 C) in the 32 GB mode and 89-97 C in the 16 GB mode; the 126K runs stayed at
89-97 C and the 252K runs reached 97-104 C (the 32 GB 252K run peaked at 103 C and passed). The card returned to 40-53 C at idle.
Our guard (stop the run at 104 C twice in a row) fired once, during a repeat of the 16 GB 252K run
([ctx-suite2.out](ctx-suite2.out): the request ended with HTTP 503). The card's sysfs lists critical limits of 105 C
(junction/edge) and 94 C (HBM).

## Notes for other MI50 users

- On ROCm 7.2.2, v0.1.40.1 **did not build** with `-DSTRATA_HIP_GFX906=ON` until three small fixes: a bare `return;` in a `bool`
  function in `fused_gr.cu`, and two `#if` guards in `mtp.cpp` and `vmm.cpp`. **We looked at `main` (d5ea71337): all three are fixed
  there. We have not rebuilt or re-run `main`.**
- `setup.py` has no `gfx906` in `AMD_ARCHS` (checked on `main`), so the installer does not recognise the card: we wrote
  `engine/BUILD.json` by hand with the source hash ([BUILD.json](BUILD.json)) so that setup accepted the engine.
- The full 32 GB card held 19,473 of 24,576 experts at 131,072 and at 262,144; the 252K prefill ran at 370 tok/s, about 10% below
  the 126K one.

## Limits

One quant (IQ2_XS), one run per case, synthetic repetitive text (simple needles), no `--parallel`, no soak, no power limit, no
memory-temperature log. The **252K / 16 GB failure is not diagnosed**: the output was `!!!!...`, the server logged no error and
`dmesg` showed no GPU reset; the repeat was cut by our temperature guard, so we do not know if it reproduces. The same pattern
at long context is reported for other cards (issues #606, #879, #871); we found no earlier gfx906 report. We have a
**hypothesis, not verified**, that heat in the HBM was involved; there is no direct evidence. The 16 GB rows come from a
reserve, not a real 16 GB card. The numbers come from the engine version and host listed above.

## Follow-up

The 16 GB-limit 252K case was repeated on 0.1.40.3 and 0.1.41 on 2026-10-08 (4 runs, 3 of 3 needles each, no `!!!!`): see [retest-2026-10-08/README.md](retest-2026-10-08/README.md).
