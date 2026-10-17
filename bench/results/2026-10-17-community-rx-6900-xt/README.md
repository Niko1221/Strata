# Community benchmark on AMD Radeon RX 6900 XT: six models, 262K context

Measured on 2026-10-17 by ByteJuggler. All six local models run on one tuned configuration, measured with the standard
`BENCH.bat` suite (`tools/strata_bench.py`). The report also shows how much two flags slowed things down on this
machine: `--resident-experts` and `--mmap-experts`, both left over from an earlier install with less RAM. Speed only:
no correctness check was run with this suite.

## Hardware and software

- GPU: AMD Radeon RX 6900 XT, 16 GiB, gfx1030 (RDNA2; no hipBLASLt table, so the plain hipBLAS prompt path runs).
  PCIe 3.0 x16; the engine's probe measured 13.4 GB/s host to device.
- CPU: Intel Core i7-5930K (Haswell-E, 6 cores, 12 threads, AVX2, no AVX-512).
- RAM: 128 GB DDR4 (127.9 GiB) at 2400 MT/s, 15-15-15-36 2T, quad channel with 32 GB per channel. That is six modules:
  2× 32 GB (not a matched pair), 2× 16 GB and 2× 16 GB. The board is an ASUS X99-DELUXE.
- Page file: 32 GB.
- OS: Windows 11 Pro 23H2 (10.0.22631). GPU driver: AMD Software 26.8.1 (32.0.21045.5002).
- Engine: 0.1.45, the official prebuilt Windows HIP zip (ROCm 10.2.0a20260930).
- Strata: v0.1.45 (`477ac13c`) with the one-line BENCH fix #2919, which `tools/strata_bench.py` needs to start a
  server itself.
- Background load: desktop apps closed; no power limits set.

## Configuration

The same settings for every model, in addition to what setup wrote:

```text
--max-context 262144 --kv int8 --kv-resident 20480 --prefill 12288 --vram-reserve-mib 1024 --pool-tasks 36
env STRATA_HIP_PROMPT_F16=1
```

- `--prefill 12288` is above the Windows HIP `auto` cap of 6144; on this card it is stable and faster on long
  prompts (see #2871).
- The full launch line of each model is in its folder's `README.md`. The Flash-Next models use the experimental
  speed projection; the Coder and Swift models don't.
- MTP: `--spec 4 --spec-min-p 0.5`.
- Expert cache: `auto`.
- Thinking is off for the suite, and temperature is 0.

## Results

Medians over 3 runs per test (9 requests for chat). Ranges and per-request data are in each folder.

| Model | Chat decode tok/s | 4K prompt tok/s | 16K prompt tok/s | Agent session decode tok/s | Chat first token |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Flash-Next Q2_0](q2_0/README.md) | **50.2** | **838** | 1,014 | **62.7** | 0.64 s |
| [Swift 1.5 IQ2_XS](swift-iq2_xs/README.md) | 47.4 | 782 | 968 | 55.9 | 0.71 s |
| [Swift 1.5 IQ3_XXS](swift-iq3_xxs/README.md) | 39.2 | 683 | 946 | 43.3 | 0.88 s |
| [Flash-Next IQ3_XXS](iq3_xxs/README.md) | 39.0 | 681 | 940 | 46.4 | 0.88 s |
| [Coder IQ1_M](coder-iq1_m/README.md) | 38.2 | 923 | **1,126** | 46.4 | 0.85 s |
| [Flash-Next IQ3_S](iq3_s/README.md) | 33.9 | 599 | 892 | 39.3 | 0.94 s |

### Two low-RAM flags on a machine with enough RAM

Setup adds these low-RAM modes on machines with less RAM. Both configs here were set up when this machine had
64 GB or less. On 128 GB, removing them made both models much faster:

| Run | Chat decode | 4K prompt | 16K prompt | Agent decode |
| --- | ---: | ---: | ---: | ---: |
| [Coder IQ1_M with `--resident-experts`, `STRATA_RESIDENT_PIN=0`](flag-removal/coder-iq1_m-resident-experts/README.md) | 29.4 | 786 | 962 | 38.0 |
| Coder IQ1_M without | 38.2 (+30%) | 923 (+17%) | 1,126 (+17%) | 46.4 (+22%) |
| [Swift IQ2_XS with `--mmap-experts`](flag-removal/swift-iq2_xs-mmap-experts/README.md) | 35.4 | 485 | 682 | 49.4 |
| Swift IQ2_XS without | 47.4 (+34%) | 782 (+61%) | 968 (+42%) | 55.9 (+13%) |

An install set up with less RAM keeps these flags until setup runs again, so a machine that has since gained RAM may
be running slower than it needs to.

## Method

- `python tools/strata_bench.py --config strata-<model>.json`, one model at a time.
- Each run started its own server from the install's configuration. Model loading is not timed.
- The flag-removal pairs were measured the same day, on the same machine, with only that flag changed.

## Limitations

- Each figure comes from one benchmark session per model. Single decode runs within a session vary widely (see the
  ranges in each folder), so differences of a few percent between models are not meaningful.
- Prompts are at most 16K tokens; longer prompts are not part of this report.
- No quality or recall check was run with this suite.
