# RTX A3000 12 GB laptop (sm_86) + i7-12850HX: host tuning and calibration

A 12 GB Ampere laptop part with a fast link but a large CPU-side expert arena. The two host settings that matter
(hugepages and `RLIMIT_MEMLOCK`), the PCIe probe's spread on this link, and the calibration result. Engine 0.1.34,
Qwen3.8-Flash-Next Swift 1.5 IQ3_XXS (native pack, 48 layers, top-10 routed experts).

## Machine

| | |
| --- | --- |
| GPU | NVIDIA RTX A3000 12 GB Laptop (GA104, sm_86, 32 SMs, 3 MB L2, 115 W default / 130 W max) |
| PCIe | x16 Gen4 (`current_link_width=16`, `16.0 GT/s`) |
| CPU | Intel Core i7-12850HX, 8 P-cores + 8 E-cores (16 cores / 24 threads), AVX2 + AVX-VNNI, **no AVX-512** |
| RAM | 128 GB |
| OS / driver | CachyOS (Arch-based, rolling); NVIDIA 615.71.09, open kernel module |
| Model | Qwen3.8-Flash-Next Swift 1.5 IQ3_XXS, native pack: **39.97 GiB expert arena**, 6.19 GiB pinned KV at `--kv-resident 32768` |

## The arena needs the WHOLE hugepage pool, or none of it

`src/core/pinned.cu` maps the expert arena with `MAP_HUGETLB|MAP_HUGE_2MB`, which is all-or-nothing: a pool even one
page smaller than the mapping makes the **entire** arena fall back to 4 KiB pages. On this host a `vm.nr_hugepages`
of 17408 (34 GiB) never covered the 39.97 GiB arena, so every run printed

```
expert arena: cudaHostRegister PORTABLE ok; MAP_HUGETLB unavailable (no hugetlb pool configured?); using 4 KB pages
```

even though a pool existed. Raising the pool to 22528 (44 GiB) and making `RLIMIT_MEMLOCK` unlimited (the arena is
charged to it; the systemd user manager otherwise caps units at 8 MiB) gave

```
expert arena: cudaHostRegister PORTABLE ok; hugetlb 2 MB pages
```

This is what the `pinned: honor STRATA_NO_LARGEPAGES on Linux; name the hugetlb pool shortfall` branch addresses: the
old message cannot tell "no pool" from "pool too small", and `STRATA_NO_LARGEPAGES` was Windows-only so a same-run
A/B of the large-page path silently did nothing on Linux.

## The PCIe probe read the same x16 link anywhere from 5.8 to 18.5 GB/s

`generate.cpp`'s `probe_pcie_h2d_gbps()` took one 256 MiB x4 pinned burst after a single warmup. Across starts on
this machine:

```
PCIe probe: 18.5 GB/s host->device -> pcie_frac 0.39 (default 0.55)
PCIe probe:  6.9 GB/s host->device -> pcie_frac 0.15 (default 0.55)
PCIe probe:  5.8 GB/s host->device -> pcie_frac 0.12 (default 0.55)
```

The low samples pin `pcie_frac` ~3x below the calibrated value, which the sweep below shows costs decode
(`0.0 -> 27.0`, `0.2 -> 30.1`, `0.35 -> 32.7` tok/s). The `PCIe probe: median of several primed bursts` branch
primes the link and takes the median of five.

## Calibration (`tools/calibrate.py`)

`tok_s` values are the tool's decode rate; a setting is kept only when it beats the default by more than 3% in the
interleaved confirmation. 638 s total.

| Setting | Default | Sweep | Kept |
| --- | --- | --- | --- |
| `--pcie-frac` | 0.55 | 0.0: 27.0, 0.2: 30.1, **0.35: 32.7**, 0.55: 32.5, 0.75: 30.8 | **0.35** |
| `--spec-min-p` | 0.5 | 0.3: 30.9, 0.5: 31.5, **0.7: 34.0** | **0.70** |
| `--pool-workers` | 15 | 15: 33.2/28.1, 10: 29.7/29.3, **8: 32.6/31.9** | **8** |

Confirmation (three interleaved runs each): `0.55/0.5` 31.6, 32.5, 32.2 (median 32.2); `0.35/0.7` 34.1, 33.8, 36.6
(median 34.1) - the calibrated pair is ~6% ahead. Raw output: [`calibrate.json`](calibrate.json).

The `--spec-min-p 0.7` and `--pool-workers 8` results are consistent with the guides: a slower CPU (no AVX-512,
i-quant CPU rows are codebook-lookup bound) pays more per extra window row, and on 8 P + 8 E the default
`--pool-affinity all` runs 15 workers where the E-cores are the tail of every phase. The default `pcie_frac 0.55`
assumes a ~26-28 GB/s link; on this link the best measured arm is 0.35.

## Decode, with the tuning above

Fixed-prose prompt (the `bench/e2e.sh` prompt), greedy, `reasoning_effort=none`, `--expert-cache auto` (2185 slots /
3.58 GiB once the arena is on hugepages and the dGPU is otherwise idle), `--max-context 524288 --kv int8
--kv-resident 32768 --rope-scaling yarn --rope-scale 2`:

| Run | decode tok/s |
| --- | ---: |
| 42-token answer | 45.1 |
| 42-token answer | 46.4 |

For reference, the same host earlier ran the same fixed prompt at ~28-33 tok/s. That is **not** a controlled A/B:
between the two, the engine moved 0.1.31 -> 0.1.34, the arena moved 4 KiB -> 2 MiB pages, and the expert cache moved
1005 -> 2185 slots. It is reported as an observation only.

## Reproduce

- Hugepages: reserve at boot in `/etc/sysctl.d` with `vm.nr_hugepages = 22528` (44 GiB) and `ulimit -l` unlimited.
  Do not apply the reservation live while the arena is pinned; boot-time is reliable.
- Calibration: `python tools/calibrate.py strata-<model>.json` with the server stopped (it spawns its own engine
  sessions), then put the kept values in the config's `args`.
