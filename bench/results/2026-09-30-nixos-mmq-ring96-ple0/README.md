# The GSQ-RCO Coder (IQ1_M) on AMD via HIP — RX 7900 XT, NixOS

First HIP run of this model on an AMD card, on NixOS. Engine 0.1.27 self-built from `a790805` against ROCm
`7.10.0a20251120` (TheRock `gfx110X-dgpu` wheels) for `gfx1100` wave32. Same `data/expert-profile-coder.bin`, the same
`--prefill`-chunk / `--spec 4` / `--mtp` shape and 256-token greedy generation as `2026-09-28-coder`, so the two tables
are comparable; the hardware and the stack are not.

| | ours (4.2K) | ours (8.8K) | `2026-09-28-coder` 4K (CUDA) |
| --- | ---: | ---: | ---: |
| Prompt, tokens/s | 614 | 569 | 1,152 |
| Output, tokens/s | 29.2 | 33.3 | 50.6 |
| Experts in VRAM | 4,886 | 4,886 | 2,615 |

Median of the trials in [`trials.json`](trials.json); 4,210-token range 608-619 prefill and 29.1-29.3 output,
8,830-token range 544-594 and 27.5-39.1. Draft acceptance was 0.90 on a single sample (37 of 41), against the
reference's 0.707. `698 MiB of VRAM free` with everything loaded. Follow-up requests averaged ~181 fresh tokens and
prefilled at 130 tokens/s.

Roughly half the reference's throughput, on a card and CPU stack that is meaningfully weaker: Zen 3 at 8 cores instead
of Zen 4 at 6, a 256-bit bus at a 2,052 MHz ceiling instead of a 384-bit NVIDIA part, and HIP instead of CUDA. The one
number that goes the other way is the expert cache — 16 GiB of VRAM plus the mmap loader keeps 4,886 experts resident
against the 12 GiB card's 2,615, so a much larger share of each token's expert reads hits the GPU.

## The configuration that produced it

The stock `setup.py` output plus the deltas the AMD performance notes ask for:

```
--prefill 8192  --pool-workers 7  --adapt-every 0  --pcie-frac 0
--vram-reserve-mib 1024  --mmap-experts  --resident-cpu-experts
STRATA_PREFILL_MMQ=1  STRATA_PREFILL_RING=96  STRATA_IO_THREADS=32
STRATA_PLE_BATCH=0
```

`STRATA_PLE_BATCH=0` is not a tuning knob — it is the only way the engine ran at all (see below). Two of the others were
already right by default on this machine and are listed only for reproducibility: the engine picks `--pool-workers 7`
for 8 physical cores on its own, and `--prefill auto` already resolves to 8,192.

## What the install needed on NixOS

Three things the installer does not know about, in the order they bite:

1. **Python's CA bundle.** `/etc/ssl/certs` on NixOS holds `ca-bundle.crt` and nothing hashed, so OpenSSL's `cafile`
   resolves to `None` and the first HTTPS fetch dies with `CERTIFICATE_VERIFY_FAILED`. Exporting
   `SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt` fixes every download.
2. **ROCm's HIP clang has no host C++ standard library**, and looks for one under `/usr/include/c++`, which NixOS does
   not have. `CMakeTestHIPCompiler` fails with `"Could not find standard C++ header 'cmath'"` and the build stops at
   `enable_language(HIP)`. Pointing the compiler at the Nix GCC 15.3.0 toolchain fixes it — `--gcc-install-dir` for
   libstdc++, and `-idirafter` for the Nix glibc headers, because the `#include_next` chain inside `<cmath>` skips an
   `-isystem` path added ahead of the GCC directories. Linking needs `-B`/`-L` for the glibc `crt*.o` and `-lstdc++`.
   A wrapper over the ROCm `clang++` applies all of it; the engine then compiles and runs.
3. **Issue #224.** The batched PLE block faults and the request hangs with the GPU at 100% and the memory bus idle —
   the same failure an RTX 5070 Ti reports on `sm_120`. Here it hits during the startup verify, so the very first
   request after a fresh start never returns. `STRATA_PLE_BATCH=0` (the per-token arm) fixes it, as reported there.

## Two things the hardware answered for us

- **The tuning table matched.** The installed ROCm's hipBLASLt reports version `100200`, so the shipped
  `gfx1100-hipblaslt-100200.txt` loads directly: `hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)`. No
  retune was needed on this wheel set.
- **The pinned arena succeeded.** `cudaHostRegister PORTABLE ok` on the 23.42 GiB arena at an 8 MiB `ulimit -l`, so the
  copy-into-RAM default would have worked too. `MAP_HUGETLB` is unavailable (no pool configured), so the mmap loader
  runs in 4 KiB pages.

## The answers are good

The 4,210- and 8,830-token code prompts produced coherent descriptions of the source they were given, and `17 * 23`
answered `391`.
