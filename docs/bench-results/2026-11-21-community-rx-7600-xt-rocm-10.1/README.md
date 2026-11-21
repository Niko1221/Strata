# Community benchmark on RX 7600 XT 16 GB (gfx1102): ROCm 7.2.4 vs ROCm 10.1 on CachyOS

Measured on 2026-11-20 and 2026-11-21 by DamirLukina88. The same engine binary and model were run with two HIP runtimes:
ROCm 7.2.4, and ROCm 10.1 with Arch's `libamdhip64`. The CachyOS build of ROCm 10.1's `hip-runtime-amd` crashes on
this machine (see [ROCm 10.1 on CachyOS](#rocm-101-on-cachyos-the-crash-and-the-workaround)). Main limitations: one run
per runtime, and the two days also differ in kernel version and GPU voltage offset.

## Hardware and software

- GPU: AMD Radeon RX 7600 XT 16 GB (Navi 33, gfx1102), XFX Speedster SWFT 210, PCIe 4.0 x8 (the card's full width)
- CPU: AMD Ryzen 5 3600 (Zen 2, 6 cores / 12 threads, AVX2, no AVX-512)
- RAM: 32 GB DDR4-3600 CL18, dual channel; board ASUS ROG Strix B550-F Gaming, BIOS 3646, Resizable BAR on
- Storage: model on a Kingston NV2 1 TB (PCIe 4.0 x4); 0 NVMe expert reads at this context
- OS: CachyOS (x86-64-v3 repositories)
  - 2026-11-20: kernel 7.3.4-1-cachyos, ROCm 7.2.4 (a private copy of the 7.2.4 packages, loaded through `lib_dirs`)
  - 2026-11-21: kernel 7.3.5-2-cachyos, mesa 26.3.1-5, ROCm 10.1 (CachyOS packages) with `libamdhip64.so.7.16.0`
    taken from Arch's `extra/hip-runtime-amd` 10.1-2
- Strata: a fork at commit `71a5118d` (upstream 0.2.12-0.2.17 server fixes ported); engine `0.1.44.1+0.2.12-fixes`,
  HIP source build for gfx1102 (built 2026-11-07, the same binary on both days)
- Power: LACT, COMPUTE power profile, 150 W power cap (card maximum 198 W), 2500 MHz maximum core clock.
  Voltage offset −50 mV on 2026-11-20, stock (0 mV) on 2026-11-21. Fan curve on the hot spot.
- Background: a desktop session; nothing else was using the GPU

## Model and configuration

- Model: `qwen3.8-flash-next-coder-iq1_m` (Coder IQ1_M)
- Context 65,536 tokens, 1 parallel slot, `--vram-reserve-mib 3072`
- Thinking off for the suite, temperature 0

## Method

`python tools/strata_bench.py --clients 4 --long` against the running server (the standard suite from
[COMMUNITY_BENCHMARKS.md](../../COMMUNITY_BENCHMARKS.md#run-the-benchmark)), one full suite per runtime. Details of every
test are in each run's own README. Medians of the runs are shown; ranges are in the run READMEs.

## Results

| Test | ROCm 7.2.4 (2026-11-20) | ROCm 10.1 + Arch libamdhip64 (2026-11-21) | Change |
| --- | ---: | ---: | ---: |
| Short chat decode, tok/s (n=9) | 29.8 (22.7-40.2) | 29.2 (25.4-43.6) | −2.0% |
| Short chat first token, s | 1.49 | 1.44 | −3.4% |
| 4K prompt, tok/s (n=3) | 339.0 | 344.3 | +1.6% |
| 16K prompt, tok/s (n=3) | 365.4 | 371.5 | +1.7% |
| 32K prompt, tok/s (n=3) | 363.3 | 368.3 | +1.4% |
| Agent session decode, tok/s (n=15) | 32.9 | 33.7 | +2.4% |
| Agent session cache hit | 67.48% | 67.5% | same |
| 4 clients at once, total tok/s | 22.4 | 23.3 | +4.1% |
| Suite wall time, s | 768 | 731 | −4.8% |

GPU readings from a 2-second hwmon log, over the samples at 80% busy or more:

| | ROCm 7.2.4 | ROCm 10.1 + Arch HIP |
| --- | ---: | ---: |
| Power, average / maximum | 117 / 151 W | 128 / 160 W |
| Hot spot, maximum | 72 °C | 74 °C |
| Core clock, average | 2449 MHz | 2425 MHz |
| amdgpu / AER / MCE kernel messages | 0 | 0 |

Per-run reports and data: [rocm-7.2.4/](rocm-7.2.4/README.md), [rocm-10.1-arch-hip/](rocm-10.1-arch-hip/README.md).

**Reading:** every difference is between −2% and +4%. The 10.1 run also had stock voltage (it drew more power) and a
newer kernel, so the fair conclusion is that ROCm 10.1 with Arch's `libamdhip64` is **at least as fast as 7.2.4** on
this card, not that it is faster.

## Correctness and limitations

- One full suite per runtime. No repeated A/B, and no answer-quality checks.
- Kernel and voltage offset differ between the two runs (see above).
- **One stall on 10.1:** the first full suite on ROCm 10.1 (before the run above) stopped on its first 32K prompt
  while reading layer 45 of the first prompt chunk. The engine's watchdog ended it after about 10 minutes. During the
  stall the GPU was idle (2-4% busy, 23 W) and there were no kernel messages: the host was waiting for a completion
  that never came. A 32K-only retest then passed 3/3 (median 367.1 tok/s), and so did the full suite above. That is
  1 stall in 7 32K prompts on 10.1 and 0 seen on 7.2.4; the cause is not known. It looks like the host-memory
  reclaim stall in [AMD_HIP.md](../../AMD_HIP.md#linux-verify-timeouts-while-the-kernel-reclaims-host-memory)
  (32 GB RAM with about 3 GiB free under this model), but that was not checked.

## ROCm 10.1 on CachyOS: the crash and the workaround

After CachyOS moved ROCm from 7.2.4 to 10.1 (`hip-runtime-amd` 10.1-2.1 from `cachyos-extra-v3`), the engine crashed
at every start with SIGSEGV inside `libamdhip64.so.7`, at its first `hipStreamCreateWithFlags`:

```
Signal: 11 (SEGV) si_code: SI_KERNEL
#0 libamdhip64.so.7 + 0x74d5b8
#1 libamdhip64.so.7 + 0x7291af
#2 libamdhip64.so.7 + 0x70e7dc
#3 libamdhip64.so.7 + 0x545a34
#4 libamdhip64.so.7 + 0x54eb54
#5 hipStreamCreateWithFlags (libamdhip64.so.7 + 0x628874)
```

It is not a Strata problem: a 4 KiB `hipMalloc` followed by `hipStreamCreateWithFlags` is enough
([repro/narrow.hip](repro/narrow.hip), `./narrow mS`). Rebuilding the engine against 10.1, a newer kernel, and
`HSA_ENABLE_SDMA=0`, `GPU_MAX_HW_QUEUES=1` or `HSA_OVERRIDE_GFX_VERSION` changed nothing.

**Cause:** an upstream HIP runtime bug, ROCm/rocm-systems#14003 / #11256. `roc::VirtualGPU` is `alignas(64)`, but
`amd::ReferenceCountedObject` has a class `operator new` without a `std::align_val_t` overload, so the object gets
16-byte aligned memory. CachyOS builds the package for x86-64-v3, so the constructor uses aligned AVX stores
(`vmovdqa %ymm1,0xc0(%rdi)` with `rdi` 16 mod 64) and faults. Arch's generic x86-64 build does not hit it.
Preloading a shim that makes the global `operator new` return 64-byte aligned memory
([repro/align64_new.cpp](repro/align64_new.cpp)) makes the unmodified CachyOS library pass every case that crashed.
The fix upstream is ROCm/rocm-systems PR #16099; CachyOS tracks it as CachyOS/CachyOS-PKGBUILDS#2143.

**Workarounds** (pick one):

1. Use Arch's build system-wide: `sudo pacman -S extra/hip-runtime-amd`, and add `IgnorePkg = hip-runtime-amd` to
   `/etc/pacman.conf` until a fixed CachyOS build ships.
2. Without root: unpack Arch's `hip-runtime-amd` 10.1-2 package into a folder and put its `opt/rocm/lib` first in the
   model config's `lib_dirs`, ahead of `/opt/rocm/lib`. That is the setup measured above.
3. Stay on ROCm 7.2.4 (for example a copy unpacked from the pacman cache, pointed to by `lib_dirs`).

Check a new `hip-runtime-amd` with `hipcc -O2 --offload-arch=gfx1102 -o narrow repro/narrow.hip && ./narrow mS`
(use your own `--offload-arch`) before trusting it again: `OK` means the bug is gone.
