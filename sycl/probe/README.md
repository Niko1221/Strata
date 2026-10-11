# The decision probes (`sycl/probe`)

Four things the SYCL engine's design rests on differ between drivers: whether a running kernel sees the host's writes
(the doorbell), how large one device allocation may be (the VRAM arena), which host memory copies and reads fastest
(the expert arena and the pinned mirror), and whether recorded command graphs replay without leaking. These four
programs measure them on the card before a model is loaded, each in a minute or so, and each ends with one line:

    VERDICT: <ok|fail|partial> <one sentence>

The sentence names the setting the result changes. Run them on a new card, a new driver or a new oneAPI before the
first model run, and keep their output with the engine log when you report a result.

Every probe starts the same way: it lists the Level Zero GPUs in SYCL's order (index, name, architecture, global
memory, largest allocation, local memory, sub-group sizes, driver), takes the one `ONEAPI_DEVICE_SELECTOR` names or,
without it, the first whose architecture is `intel_gpu_bmg_g31` (the Arc Pro B70 / B65 die), else the first GPU, and
says which it took and why.

This directory also holds the Linux port's own single-file probes (`alias_scan.cpp`, `arena_rw.cpp`, `fp64_cost.cpp`,
`gemm_*.cpp`, `launch_cost.cpp`), built by hand with `icpx -fsycl`; they are not part of the `probes` target.

## Build

The four probes are part of the engine's SYCL build, behind the CMake option `STRATA_SYCL_PROBES` (on by default):

    sycl\tools\build_windows.bat                         (Windows: builds them with the engine into engine\)
    cmake --build build-sycl-win --target probes          (or by hand, from a oneAPI command prompt)
    cmake --build build-sycl --target probes              (Linux, icpx)

The executables are `doorbell_probe`, `alloc_probe`, `bw_probe` and `graph_probe`. On Windows they load the same
oneAPI runtime DLLs as `strata.exe`, so run them from `engine\` (beside the DLLs the build put there) or from a oneAPI
command prompt. They are built with the engine's flags (`-fsycl-default-sub-group-size=32`, the precise FP model, the
AOT target when `STRATA_SYCL_AOT` is set), so what they measure is what the engine's kernels get.

## Run

Run them with the environment the server gives the engine (docs/INTEL_WINDOWS.md, "The engine's environment"), so
that the measurement is the engine's situation:

    set ONEAPI_DEVICE_SELECTOR=level_zero:0
    set SYCL_UR_USE_LEVEL_ZERO_V2=1
    set SYCL_CACHE_PERSISTENT=0
    set ZES_ENABLE_SYSMAN=1

`level_zero:0` is the Arc when it is the only Level Zero GPU; `strata-device --list-devices` or any probe's first
lines show its index when the processor's GPU is listed too.

### `doorbell_probe [trips]`

Does a running kernel see the host's writes, and the host the kernel's? Five readings of the host's flag (a volatile
load; a volatile load plus a system-scope acquire fence; a system-scope atomic load; an uncached L1+L3 read hint; and
whatever this build's `sys_load` is), 500 round trips each by default, then the build's reading replayed 100 times
inside a `command_graph`. Per reading: how often the ring and the flag were seen, median and p99 latencies, and the
most polls a successful wait took against the engine's spin bound (`strata::spin_max`).

The verdict changes `sycl/include/strata/sycl_doorbell.hpp`'s build option (`-DSTRATA_DOORBELL_FENCED_VOLATILE`,
`-DSTRATA_DOORBELL_ATOMIC_LOAD`; the default is the uncached read hint on every platform) and whether
`STRATA_VERIFY_NO_HOST=1` may be tried at 0. Measured on the Windows B70 (2026-10-10, driver 32.0.101.8976, oneAPI
2026.0.0, the Level Zero V2 adapter, 500 round trips): the ring 500 of 500 for every complete reading (15 us median
from the launch); variant 4, the uncached read hint, 500 of 500 (4.3 us median from the flag to the kernel's end,
16.8 us p99, at most 18 polls); variant 2, the fenced volatile load, 500 of 500 (4.9 us, 14.6 p99, at most 40 polls)
and 100 of 100 inside the replayed graph; variant 1, the plain volatile load, 3 of 6; variant 3, the atomic load
alone, 4 of 9; both abandoned after three misses in a row. `VERDICT: ok variant(s) 2, 4, 5 saw every host write`.

### `alloc_probe [--device-only | --host-only] [--host-max GiB] [--no-fill]`

How large may one device allocation be? One `malloc_device` of 1, 2, 4, ..., 30 GiB and of `max_mem_alloc_size`,
each filled by a kernel in 1 GiB pieces and read back, then freed; then host USM of 1 to 31 GiB. The fill speed
says whether an allocation landed in VRAM (hundreds of GB/s) or spilled into host RAM (tens).

The verdict changes `STRATA_ARENA_CHUNK_MIB` in the config's `"env"`: unset when the whole reported maximum is one
allocation (the engine then makes one allocation first and chunks only after a refusal), else the chunk size to set.
Measured on the Windows B70 PC (128 GiB of RAM, 2026-10-04 and 2026-10-05): every size up to 31.16 GiB passed at
352-467 GB/s, `malloc_host` up to 31 GiB; on Linux a 16 GiB box refuses a single allocation above 19.3 GiB
(compute-runtime#998), which is what the fallback is for.

### `bw_probe [total MiB] [piece MiB]`

Host-to-device and device-to-host copy speed from each kind of host memory (pageable, `malloc_host`, `VirtualAlloc`
registered with `prepare_for_device_copy`, the same unregistered), and how fast a kernel reads `malloc_host` memory in
place (first touch and hot). Measured on the Windows B70 (2026-10-04/05): pageable 12-13 GB/s, `malloc_host` 27.5-27.7,
registered `VirtualAlloc` 27.6-27.8, unregistered 12.6-12.9; a kernel reads `malloc_host` at 15.5 GB/s on first touch
and 23 hot, VRAM at 415-423. The verdict names how the pinned expert arena is backed (`sycl/src/core/pinned.dp.cpp`
registers upstream's `VirtualAlloc` memory with `prepare_for_device_copy`) and that the mirror stays `malloc_host`.

### `graph_probe`

Do recorded command graphs replay on this card, how many nodes may one hold, how many may be alive, and does
finalizing and destroying them leak host memory? Measured on the Windows B70 (2026-10-04): 68 nodes replay in 50.5 us
(eager: 212 us), one graph of 8,192 nodes is fine, 64 graphs alive, 21 bytes per node per finalize-destroy cycle (the
v1 adapter on an A770 leaked about 1.5 KiB per node, intel/llvm#23073). The verdict says whether the engine's graphs
stay on and the V2 adapter stays.

## Recording the results

Paste each probe's complete output into the report you post (`sycl/tools/report_windows.py` collects the engine
log's key lines beside them), with the card, the Windows graphics driver version, the oneAPI version and the date.
A changed verdict after a driver update is a changed setting.
