# Intel Arc on Windows (experimental)

The SYCL port of Strata's engine ([INTEL_ARC.md](INTEL_ARC.md), [INTEL.md](INTEL.md)) builds and runs natively on
Windows 11: no WSL2, no Docker. It was written for and measured on one PC with an **Arc Pro B70 (32 GB)** that is
dedicated to the model (it drives no display; an NVIDIA card drives the desktop). Everything on this page was measured
there unless it says otherwise, and the things it says are open are open. The Linux build is unchanged.

**What ran (2026-10-04 to 2026-10-09).** The engine built with oneAPI 2026.0.0 (127 targets, 0 errors), the device
probe and the four decision probes ran, the Coder IQ1_M installed and answered its first chat at 60 tok/s sampled
(12,284 of its 12,288 experts in VRAM, 2026-10-05), and Swift 1.5 IQ3_XXS, whose experts do not fit the card, served
for several days with the rest of them in a pinned RAM mirror and KV streaming: 53-64 tok/s decode on prose and code,
650-800 tok/s on 12K-40K-token prompts, 98-99% of the KV block reads from VRAM (the engine log, 2026-10-07 to 09).

**What ran on this tree (0.1.41, 2026-10-10).** The build (146 targets, 0 errors, oneAPI 2026.0.0, driver 32.0.101.8976),
the probes, and Swift 1.5 IQ3_XXS at `--max-context 262144` with INT8 KV and 32,768 resident cells, used from qwen-code
for an evening: prompts read at 630-890 tok/s (26,031 fresh tokens in 41 s, 59,015 in 68 s), decode 50-76 tok/s,
97-99.3% of the KV block reads from VRAM, 2,187 MiB of VRAM free with everything loaded (the expert cache 14,678 slots
in 23.81 GiB, the other 9,898 experts mirrored in 16.15 GiB of pinned RAM).

## What you need

- **Windows 10 / 11** and a discrete Intel Arc. Tested: the Arc Pro B70 (BMG-G31). The build is ahead-of-time for that
  die (`bmg-g31`); other Battlemage cards need `STRATA_SYCL_AOT=bmg-g21` (B580, B570, Pro B60, Pro B50) or a JIT build
  (`STRATA_SYCL_AOT=none`), and nobody has run those on Windows yet.
- **The Intel Arc graphics driver.** The B70 PC runs 32.0.101.8976 (Level Zero driver string `1.15.39183+4`), and
  everything on this page ran on it with oneAPI 2026.0.0. With **oneAPI 2026.1.1** the driver must be 32.0.101.9034
  or newer: the Level Zero loader ships inside the driver, and 2026.1.1's Level Zero v2 adapter crashes on its first
  submit with a loader older than 1.32 (intel/llvm#22958); `STRATA_BUNDLE_ZE_LOADER=1` at build time bundles oneAPI's
  loader instead. The build script warns when the driver is older than 9034.
- **Intel oneAPI Base Toolkit 2026.0.0 or newer**: the DPC++/C++ compiler and oneMKL (components `cpp-dpcpp-common`,
  `mkl.devel`, `tbb.devel`). 2026.0.0 built and ran the engine; 2026.1.1 is what llama.cpp's Windows SYCL release
  uses. The C++ compiler is `icx` (the MSVC-compatible driver), the C compiler `cl`.
- **Visual Studio 2022 or 2026 Build Tools** with the C++ workload (`icx` uses MSVC's headers and linker), **CMake
  3.25+**, **Ninja** (both in the Build Tools' "C++ CMake tools" component), **Python 3.10+**.
- **RAM**: 16 GB is enough for the Coder at 32K (`--stream-experts` keeps no host copy of the experts); a model that
  does not fit the card mirrors the rest of its experts in pinned RAM (Swift IQ3_XXS: about 20 GB). **Disk**: about
  150 GB free, the model on an SSD.

## Build

    sycl\tools\build_windows.bat

From a plain command prompt. The script finds Visual Studio's `vcvars64.bat` and oneAPI's `setvars.bat`, configures
`sycl\CMakeLists.txt` into `build-sycl-win\` with Ninja (`cl` for C, `icx` for C++, `-DSTRATA_SYCL_AOT=bmg-g31`,
`-DSTRATA_PORTABLE=ON`, `-DSTRATA_SYCL_PROBES=ON`; the pinned llama.cpp comes from `setup.get_llama_cpp()`, no git),
builds `strata`, `strata-device` and the probes, and runs `sycl\tools\package_windows.py`, which puts into `engine\`:

- `strata.exe`, `strata-device.exe`, `doorbell_probe.exe`, `alloc_probe.exe`, `bw_probe.exe`, `graph_probe.exe`;
- the oneAPI runtime DLLs beside them, the set llama.cpp copies beside its Windows SYCL binaries (`sycl9.dll`,
  `ur_loader.dll`, the three `ur_adapter_*.dll`, the oneMKL BLAS set with the sequential threading layer, TBB, umf,
  tcm, hwloc, SVML, libmmd, OpenMP: 19 DLLs) plus whatever else their import tables name from the oneAPI folders
  (`dumpbin /dependents`); `ze_loader.dll` only with `STRATA_BUNDLE_ZE_LOADER=1`, since the driver ships one;
- `BUILD.json`: `"backend": "sycl"`, `"source": "local-sycl"`, the version, the AOT device, the compiler version, and
  `"src"`, setup's source hash over the engine sources plus `sycl/`, which setup compares at every start.

Without oneAPI or the Build Tools the script stops and lists what is missing. Settings: `STRATA_SYCL_AOT`, `BUILD_DIR`,
`ONEAPI_ROOT`, `JOBS`, `STRATA_GGML_DIR` (the script's header). What the build does differently from Linux
(`sycl/CMakeLists.txt`): MSVC-style flags where `icpx` takes GNU ones (`/fp:precise`, `/EHsc`, `/bigobj`, `/utf-8`,
`/arch:AVX512` plus `-mavx512vnni -mavx512vbmi` for the AVX-512 kernels, `/Qmkl:sequential` and `/Qmkl-sycl-impl:blas
/MD` at link), the dynamic CRT (the oneAPI DLLs use it), `NOMINMAX` and `WIN32_LEAN_AND_MEAN` for every target, the
kernel parity tests off by default (`-DSTRATA_SYCL_PARITY=ON` builds them). The

SYCL flags are the same as on Linux: `-fsycl-default-sub-group-size=32`, `-fsycl-device-code-split=per_kernel`,
`-cl-fp32-correctly-rounded-divide-sqrt` for the device compiler.

## Run the probes first

    engine\strata-device.exe --list-devices
    engine\strata-device.exe --selftest
    engine\doorbell_probe.exe
    engine\alloc_probe.exe
    engine\bw_probe.exe
    engine\graph_probe.exe

`--list-devices` prints every Level Zero device in SYCL's order, `device N: <name>` then `arch <architecture>, <x.y>
GiB`, and `cannot run: <reason>` for one the engine refuses (no 32-wide sub-groups, no fp16, no USM, or another die
than the code was compiled for). That is what setup parses to pick the card. On the B70 PC: `device 0: Intel(R)
Arc(TM) Pro B70 Graphics` / `arch intel_gpu_bmg_g31, 31.2 GiB`, the only device (the NVIDIA card is not a Level Zero
device). `--selftest` plans 20,480 tokens against the free VRAM and exercises a 64 MiB arena.

The four probes measure, on the driver at hand, what the engine's design rests on; each ends with a `VERDICT` line
that names the setting it changes ([sycl/probe/README.md](../sycl/probe/README.md)). On the B70 PC: the doorbell
(2026-10-10, this probe, 500 round trips per reading) sees the GPU-to-host ring 500 of 500 at 15 us median from the
launch, and the host's flag through the uncached read hint (500 of 500, 4.3 us median from the flag to the kernel's
end, at most 18 polls) and through a volatile load with a system-scope acquire fence (500 of 500, 4.9 us, at most 40
polls; 100 of 100 inside a replayed graph), but not through a plain volatile load (3 of 6) or the system-scope atomic
load alone (4 of 9); one device allocation of the whole card (31.16 GiB) at 350-470 GB/s (2026-10-04 and 05);
host-to-device copies 27.6-27.8 GB/s from `malloc_host` and from `VirtualAlloc` memory registered with
`prepare_for_device_copy`, 12-13 from pageable or unregistered memory; graphs of 8,192 nodes replay on the V2 adapter
with 21 bytes leaked per node per finalize-destroy cycle. Run them again after a driver or oneAPI update.

## Install and start

    START-HERE.bat --backend sycl [--family coder --model IQ1_M --context 32768 ...]
    START-HERE.bat --backend sycl --check              what fits this PC, before the engine is built

`START-HERE.bat` creates the `.venv` and runs `setup.py`, which hands `--backend sycl` to `sycl\setup_intel.py`:
upstream's installer with the Intel steps swapped in from outside (no shared file is edited). On Windows it:

- lists the Intel display adapters (PCI `VEN_8086`; `DEV_E223` is the B70, `E222` the B65, the other Battlemage and
  Alchemist ids are in its table) and, once the engine is built, numbers the cards by `strata-device --list-devices`,
  which is SYCL's order and the one the config's `"gpu"` means; the processor's graphics is listed and refused, the
  NVIDIA card is never a candidate and `nvidia-smi` is never asked;
- takes `engine\strata.exe` when its `BUILD.json` says `"backend": "sycl"` and its source hash matches the tree (a
  `git pull` or an edit under `sycl/` makes the engine stale: run the build again; a plain start of an installed model
  warns and starts it anyway); it never downloads an engine and `--build` is refused (the build script builds);
- writes `strata-<model>.json` with `"backend": "sycl"`, native paths, `"lib_dirs": ["engine"]`, the card's SYCL
  index as `"gpu"`, and the engine arguments the Linux port writes (`--stream-experts`, `--vram-reserve-mib 1024` up
  to 32K and `2048` above, `--prefill 4096` on a 24 GB+ card or above 32K, `--kv-resident 32768` from 64K when the RAM
  holds the KV, the MTP draft layer); an Alchemist card gets the Linux port's `i915` config (no `--stream-experts`,
  `--ple-io ram`, a 300 MiB reserve, `STRATA_VERIFY_NO_HOST=0` and the FP64 emulation keys in `"env"`);
- writes `run-<model>.bat`, which starts `sycl\serve\server_intel.py`: `serve\server.py` with the Intel Monitor
  readings (Level Zero Sysman through `ctypes`), the engine's environment below, and the `--stream-experts` start
  narration instead of the CUDA engine's "locking RAM" warning.

Later starts: `run-<model>.bat`, or `START-HERE.bat --backend sycl` again. The engine's stderr goes to
`strata-<model>.log` next to the checkout.

### The engine's environment

`sycl\serve\server_intel.py` sets these for a `"sycl"` config on Windows; a key in the config's `"env"` wins:

| variable | value | why |
|---|---|---|
| `ONEAPI_DEVICE_SELECTOR` | `level_zero:<gpu>` | one device in the engine's one SYCL context: the Arc, never the integrated GPU (a context with two Intel devices mirrors every VRAM allocation into host RAM on Linux, compute-runtime#980) |
| `SYCL_UR_USE_LEVEL_ZERO_V2` | `1` | the v2 adapter: graph record and replay, no event leak per graph finalize (intel/llvm#23073), what ran the B70 |
| `SYCL_CACHE_PERSISTENT` | `0` | the persistent JIT cache segfaulted on Xe2 during a first compile ([INTEL.md](INTEL.md)) |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` | `1` | what the native Windows B70 users of llama.cpp set; moot for device memory on BMG |
| `ZES_ENABLE_SYSMAN` | `1` | the free-memory query (`ext_intel_free_memory`) and the Monitor's readings |
| `STRATA_VERIFY_DEVICE_PLAN` | `1` | the GPU plans each layer itself (the Linux port's setting) |
| `STRATA_VERIFY_NO_HOST` | `1` | the host waits for the whole window; required whenever part of the experts is not in VRAM ([INTEL.md](INTEL.md)), and the fenced flag read answers in 99 us on this driver, too slow for a host answer inside a window |
| `STRATA_STAGER_THREADS` | `min(12, physical cores)` | the Linux port used 12 on a 16-thread CPU |
| `STRATA_SYCL_HEARTBEAT_S` | `10` | the VRAM heartbeat below (0 = off) |
| `PATH` | `engine\` first | the oneAPI runtime DLLs beside `strata.exe` |

## What the Windows build adds to the engine

All of it lives in `sycl/` under `#if defined(_WIN32)` or in the Windows scripts; the Linux port's code paths are as
they were.

- **The card choice and the capability gate** (`sycl/src/core/device.dp.cpp`). The engine numbers the Level Zero
  devices as SYCL does, checks 32-wide sub-groups, fp16, device and host USM, and in an ahead-of-time build that the
  card is the die the code was compiled for (`intel_gpu_bmg_g31`), and stops with the device's name and the reason
  otherwise. `--list-devices` prints the same.
- **A chunked VRAM arena** (`sycl/include/strata/core/device.hpp`, the SYCL build's copy of the header). `DeviceArena`
  makes one allocation for the whole plan first, as upstream; when the device refuses it, chunks of 60% of
  `max_mem_alloc_size` (at most 4 GiB on Windows) behind the same bump allocator, no sub-allocation straddling two
  chunks. `STRATA_ARENA_CHUNK_MIB=N` forces N MiB chunks (0 = one allocation, no fallback), `STRATA_MAX_ALLOC_MIB`
  stands in for the reported maximum. The poison fill runs in 1 GiB pieces. On the B70 PC the one allocation
  succeeds, so the fallback is for the PCs that compute-runtime#998 describes (a single allocation refused above a
  threshold that tracks host RAM plus swap). The arena logs what it did: `strata: VRAM arena N chunk(s) of M MiB
  (max_mem_alloc_size K MiB, STRATA_ARENA_CHUNK_MIB=auto)`.
- **The free-VRAM figure** (`sycl/include/strata/sycl_mem.hpp`). The runtime's figure counts only this process's
  allocations; Windows gives each process a video memory budget (DXGI `QueryVideoMemoryInfo`) that counts the desktop
  and the other programs, and WDDM moves what is allocated past it out to system RAM. Every sizing step reads the
  runtime's figure lowered by what the budget withholds, the adapter found by the SYCL device's LUID, as upstream's
  Windows HIP build does for `cudaMemGetInfo` (#380, #377). `STRATA_WDDM_BUDGET=0` turns it off. On a dedicated card
  the subtraction is about zero; it is logged once when it is not.
- **The VRAM heartbeat** (`sycl/include/strata/sycl_heartbeat.hpp`). WDDM evicts a GPU's idle memory to system RAM,
  and on a card that drives no display nothing else keeps it warm: the B70's memory went cold between requests. Every
  long-lived device allocation registers itself (the arena's chunks, the weight arena, the session, the expert cache's
  slots, the verifier's and the drafter's arenas, the native head and dense matrices, the prompt path's buffers of 16
  MiB and more), and in serve mode a thread touches them every `STRATA_SYCL_HEARTBEAT_S` seconds: one small kernel per
  range on its own queue, reading a word every 64 KiB. The same submission keeps the Level Zero queue alive across
  idle minutes (intel/llvm#23209: `DEVICE_LOST` after sustained work and then idle). Once, after the first touch, the
  log says `strata: VRAM heartbeat every 10 s, touching M MiB` (the Swift runs: 30,113 MiB) and, when that is under
  half of what the card holds, `covers only M of P MiB`. Its effect against a timed eviction has not been measured
  (the idle test below).
- **The doorbell** (`sycl/include/strata/sycl_doorbell.hpp`). `sys_load` is the same on Windows as on Linux, the
  uncached L1+L3 read hint, since the probe measured it complete on the Windows driver (above). The build options
  `-DSTRATA_DOORBELL_FENCED_VOLATILE` (the other complete reading) and `-DSTRATA_DOORBELL_ATOMIC_LOAD` choose
  another. The spin bound is upstream's per-device `spin_max` (2,000,000 reads on Windows, where the driver is not
  `xe`; one fenced poll costs about 0.86 us there).
- **The host memory** (`sycl/src/core/pinned.dp.cpp`, `sycl/src/core/gguf_expert_source.cpp`). The expert arena is
  upstream's `VirtualAlloc` memory (large pages when the privilege allows) registered with `prepare_for_device_copy` in
  pieces no longer than `max_mem_alloc_size`, which doubled its copy speed in the probe; the mirror stays
  `malloc_host`. `--stream-experts` reads the GGUF through overlapped `ReadFile` with per-thread events, since its
  reader threads share one handle per shard and a synchronous handle would serialize them.
- **Small Windows pieces.** The PCIe probe times its bursts on the host clock (Level Zero event profiling is not
  trusted to bracket a copy on this driver); the mirror's RAM cap reads `GlobalMemoryStatusEx` where Linux reads
  `MemAvailable`; the engine logs which `sycl9.dll` and `ze_loader.dll` it loaded, as the HIP build logs its
  `amdhip64` DLL; the two parity programs that called `setenv` use `_putenv_s`.

## The tools

- `sycl\tools\report_windows.py`: a one-shot report for a bug report or a result (the machine and driver version,
  `BUILD.json`, the config, the engine log's key lines, Sysman's readings).
- `sycl\tools\benchy_windows.bat` (`benchy_windows.py`): the Linux port's benchy v1 on Windows (the same prompts,
  greedy, 256 tokens, `--stats`, a fresh engine per run) into `sycl\benchy-results\v1-<date>-windows\matrix.md`. Its
  rows are the ones that compare with the Linux numbers in [INTEL_PERFORMANCE.md](INTEL_PERFORMANCE.md). Not run yet.
- `tools\test_setup_sycl_windows.py`: the installer's and the server wrapper's tests, without a GPU, oneAPI or
  downloads (mocked display adapters and `--list-devices` output).

## Known limits and open questions

- **One card per model on Windows.** `--gpus` / `--layer-split` is refused: a layer split puts every card into one
  SYCL context, and the Windows behaviour of a two-device context is unknown.
- **Images** are not wired on Intel, on Windows as on Linux (setup passes `--vision none`).
- **The heartbeat's effect** has not been timed: the idle test is to start a model, watch Task Manager's dedicated GPU
  memory for 15-20 idle minutes, send a prompt, and compare with `STRATA_SYCL_HEARTBEAT_S=0` in the config's `"env"`.
- **The decode attention kernel is the 0.1.38 migration.** `sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp` is kept
  from 0.1.38 (its header comment has the measurements): the 0.1.41 copy adds an opt-in lane-cell scoring variant as a
  template parameter, and compiled that way by icx 2026.0.0 for this card the INT8 kernel's batched output is inf and
  NaN while the fp16 one is exact. Since 0.1.41 runs SYCL prompts through that kernel, every reply was token 0 until
  the swap. When re-migrating, leave the lane-cell variant out of the SYCL copy.
- **The XMX prompt attention's precision.** Opt-in (`STRATA_PROMPT_ATTN_XMX=1`) and off by default. On this build
  `qsa_prompt_attn_parity` measures it at 2e-3 absolute (5e-4 of the output scale) where [INTEL.md](INTEL.md) reports
  1e-6 of scale on Linux. Not understood; not on the default path.
- **A greedy benchmark** (`benchy_windows.bat`) has not been run, so the Windows numbers above are sampled chats, not
  comparable with the Linux port's greedy rows (Coder IQ1_M 78.5 tok/s on a 20-token prompt, 76.3 after 2,185 tokens).
- **`prepare_for_device_copy` on the real arena**: the probe registered 256 MiB; a `--stream-experts` config builds no
  host expert arena, so the registration of a whole arena has run only in the Swift IQ3_XXS configuration, whose log
  line was not inspected. The registration note is on the engine's `pinned` line.
- **Other cards.** Only the Arc Pro B70 has run this. A B580 / B570 / Pro B60 / Pro B50 needs `STRATA_SYCL_AOT=bmg-g21`
  (or a JIT build) and the probes first; an Alchemist card gets the Linux port's `i915` config but has not been tried.

## Where this comes from

The Windows work was done in [Scythe](https://github.com/K8-and-a-half-1000/Scythe), a fork of Strata 0.1.38 plus
PR #423's overlay, between 2026-10-04 and 2026-10-09, and ported onto this tree's `sycl/` by a three-way merge over
the PR's overlay. Two of its probes (the doorbell and the allocation probe) derived from XeStrata's LGPL files and
were rewritten here from their descriptions, so everything in this PR is under Strata's MIT license.

## Reporting a problem

Open an issue with: the card, the Windows graphics driver version (Settings > System > Display > Advanced display, or
`report_windows.py`'s first lines), the oneAPI version, `engine\strata-device.exe --list-devices`, the four probes'
`VERDICT` lines, the model and flags (the config's `"args"`), and the engine log (`strata-<model>.log`).
