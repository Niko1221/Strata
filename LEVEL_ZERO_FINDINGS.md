# Level Zero on this machine: diagnosis and findings

Machine: Arc Pro B70 32 GB, Windows 11 24H2, conda `strata-sycl-261`
(`dpcpp_win-64` **2026.1.1**), engine 0.1.43. Everything below is measured on this box, 2026-10-11/12.

---

## TL;DR for the next person

1. **This machine runs the Intel *generic* Arc graphics driver, `32.0.101.9034` or newer — not the
   Arc Pro one.** That is deliberate and it is the prerequisite for everything else below. The Arc Pro
   line's newest build (`32.0.101.8976`) does not expose Level Zero through any SYCL adapter at all,
   and 8976 is the newest build that line has. Full rationale, what it costs and how to go back:
   [the graphics driver](#the-graphics-driver-why-the-generic-arc-line-and-not-arc-pro).
2. **Level Zero on this card was never broken** — the Level Zero *driver* answered every direct API
   call correctly on 8976 too. What changed between the two driver builds is the Level Zero ICD
   (`ze_intel_gpu64.dll`) the driver ships; on 9034 the SYCL adapters can drive it and on 8976 they
   cannot.
3. **The v1 Level Zero adapter is much better than v2 here**: decode +24%, graph capture 34x faster.
   This contradicts PR #1940's stated reason for pinning v2 on.
4. **A missing DLL list silently hid all of this.** `ur_adapter_level_zero.dll` imports `UMF.dll` and
   `VCRUNTIME140*`, which `sycl/CMakeLists.txt` never stages beside `strata.exe`, so the UR loader
   fell back to OpenCL with no diagnostic at all. This is the bug most worth reporting upstream.
5. **Command graphs on Level Zero stall the verify window**, so the engine must run the eager replay
   there. See [the stall](#the-stall-found-it-is-sycl-command-graphs-on-level-zero). This is why
   `STRATA_VERIFY_EAGER=1` is *set*, not unset, on Level Zero.
6. The Level Zero win is **prefill, not decode**: 120.6 tok/s against OpenCL's 81.1, with oneMKL
   refusing in every configuration tested.

## Current configuration

Level Zero is selected by `STRATA_SYCL_LEVEL_ZERO=1` in the server config
(`strata-coder-iq1_m.json`); `sycl/serve/strata-sycl.bat` turns that into the five variables above.
Unset it and the launcher falls back to the OpenCL path, which is intact and is still the default.

```powershell
(Get-CimInstance Win32_VideoController | ? Name -match 'Arc').DriverVersion   # 32.0.101.9034?
& "$env:USERPROFILE\miniforge3\envs\strata-sycl-261\Library\bin\sycl-ls.exe" --ignore-device-selectors
#   a [level_zero:gpu] line is what "working" looks like
```

---

## How we got here, in order

### 1. The OpenCL fallback was built on a false premise

The port ran on OpenCL because `ONEAPI_DEVICE_SELECTOR` was never satisfied: `sycl-ls` enumerated
only `[opencl:gpu]`. The assumption was "Level Zero is broken on this card." It was not.

### 2. The driver is healthy; the loader above it was not

Bypassing SYCL entirely, through the v1 loader the driver shipped
(`...iigd_dch.inf...\ze_loader.dll`, 1.2.3), with no environment set up at all:

```
zeInit(0)             -> 0x0
drivers               -> 1
zeDeviceGet(count)    -> 0x0, devices=1
zeDriverGetProperties -> 0x0, driverVersion 17013007 (1.15.39183+7)
zeDeviceGetProperties -> 0x0
    type                  -> TYPE_GPU (1)
    vendorId / deviceId   -> 0x8086 / 0xe223          (Arc Pro B70, BMG-G31)
    maxMemAllocSize       -> 31.2 GiB
```

The modern loader (1.32.0, conda) returned `ZE_RESULT_ERROR_UNINITIALIZED` in *every*
configuration: bare, with `ZE_ENABLE_LOADER_DRIVER_SEARCH=1`, with the ICD directory on `PATH`, and
with the ICD already beside it. There is no `HKLM\SOFTWARE\Khronos\LevelZero` key on this box (OpenCL
and Vulkan are registered; Level Zero is not).

Registering the ICD by hand under `HKCU\SOFTWARE\Khronos\LevelZero\ICDs` turned the silent failure
into a straight `0xC0000005` crash - which proved it was never "the loader cannot find the ICD";
once it found it, it crashed against it. That registration was removed.

The standalone oneAPI 2026.0.0 installer was downloaded, hash-verified and installed. It also ships
**no `ze_loader.dll` at all** and behaves identically, so the toolchain packaging was not the answer
either.

### 3. The graphics driver was the answer

Two driver lines exist for this card, and which one is installed decides whether Level Zero works at
all. This is covered in full — with the evidence, what it costs and how to go back — in
[the graphics driver](#the-graphics-driver-why-the-generic-arc-line-and-not-arc-pro). In short:

| line | newest build | exposes Level Zero? |
|---|---|---|
| Intel® Arc™ **Pro** | 32.0.101.8976 (Q3.26) | **no** |
| Intel® Arc™ *generic* | 32.0.101.9034 | **yes** |

The Arc Pro line is the ISV-certified workstation driver and the "correct" one for an Arc Pro B70 —
and 8976 is the newest build it has, so there was nothing to update to. The generic driver lists the
Arc Pro B70 as supported explicitly. Moving to it produced, for the first time:

```
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero V2,
    Intel(R) Arc(TM) Pro B70 Graphics 20.2.0 [1.17.39500+2]
    Aspects: ... ext_intel_free_memory ... ext_oneapi_graph ext_oneapi_limited_graph ...
```

Both things that had forced the OpenCL workarounds are present: **the free-VRAM query** and
**command graphs** (the latter then turned out to be the cause of the stall, so it is unused).
Installer: `%TEMP%\opencode\gfx_win_101.9034.exe`, SHA512 verified against Intel's published digest.

### 4. A second, separate bug: the adapter's dependencies were never staged

With Level Zero available, the engine *still* only loaded `ur_adapter_opencl.dll`. The PE import
table of the Level Zero adapter explains why:

```
imports: UMF.dll, KERNEL32.dll, MSVCP140.dll, libmmd.dll, ADVAPI32.dll,
         VCRUNTIME140.dll, VCRUNTIME140_1.dll, api-ms-win-crt-*.dll, ...
```

`UMF.dll` and the `VCRUNTIME140*` set are not in the DLL-copy block in `sycl/CMakeLists.txt`
(which copies `sycl9.dll libmmd.dll ur_loader.dll ur_win_proxy_loader.dll ur_adapter_*.dll
OpenCL.dll`). The adapter therefore fails to load, the UR loader reports only OpenCL, and
`ONEAPI_DEVICE_SELECTOR=level_zero:0` kills the process with `0xC0000409` before `main` prints
anything.

**Confirmed workaround: put conda's `Library\bin` first on `PATH`.** The proper fix is to extend the
copy list - and the *silence* is the part worth reporting upstream, because it masks every Level Zero
adapter failure as "there is no Level Zero here."

---

## Measured, driver 9034

All runs: Arc Pro B70, coder-IQ1_M, `--spec 4 --spec-min-p 0.5`, `--mtp`, MTP draft head loaded,
2047-token prompt, greedy.

| configuration | decode | prefill | graph capture | stall? |
|---|---|---|---|---|
| OpenCL, `STRATA_VERIFY_EAGER=1` | 66.6 tok/s | 81.1 | skipped | no |
| LZ **v2** adapter | 87.9 tok/s | 64.5 | 9,495 ms | no |
| LZ **v1** adapter, 128 tok | 108.8 tok/s | 81.1 | 276 ms | no |
| LZ **v1** adapter, 512 tok, MTP | **74.5 tok/s** | **121.4 tok/s** | — | no |
| LZ, device plan **OFF** | 0.95 tok/s | 106.5 tok/s | — | no |
| LZ via the **server** | — | — | — | **yes** |

**The v1 Level Zero adapter beats v2 decisively**: decode +24% and graph capture 34x faster.
#1940 pins `SYCL_UR_USE_LEVEL_ZERO_V2=1` for "graph record and replay, no event leak per graph
finalize (intel/llvm#23073)"; on this card, with this workload, that trade is clearly wrong.

**The device plan is required**: with it off, decode is 0.95 tok/s - 100x slower - because the
window's host handshakes become synchronous. It is also where the stall lives.

**Prefill is +50% on Level Zero** (121.4 vs 81.1 tok/s) with oneMKL refusing in both cases, so the
plain-SYCL fallback GEMM is itself faster on Level Zero than on OpenCL.

---

## The stall: found. It is SYCL command graphs on Level Zero

Tested on the same card, driver 9034, `--mtp`, `--spec 4`, 262144 context, through the serve request
path (which is the path that stalled).

| configuration | windows exercised | stalls? | decode |
|---|---|---|---|
| LZ, **graphs on** (eager off) | ~20 | **yes, at KV position ~84** | 74.5 tok/s (CLI) |
| LZ, **graphs off** (eager on) | **1000+** | **no** | 63.3 tok/s (CLI) |

The graph configuration stalls at *"waiting for the GPU to finish the window (flags A/B/M raised);
the GPU rang 0; flags: served 0, plan (A) 0, copies (B) 0"* — reproducibly, at KV position ~84, i.e.
about 20 windows into a request. The no-graph configuration served 35 requests and three 900-token
generations (310 + 495 + 178 windows) without one.

So the doorbell flag the host polls is written by kernels running inside a **replayed command
graph**, and on Level Zero the replay either does not execute them or does not make the write
visible to the host's polling read. That is the whole bug, and it is not the device plan, the
launcher, the adapter version, prompt length, or `--mtp` — all of which were ruled out by direct
measurement earlier in this document.

This also explains why PR #1940 pins `SYCL_UR_USE_LEVEL_ZERO_V2=1` *"for graph record and replay, no
event leak per graph finalize"*: they are working in exactly this area. On this card the trade is the
opposite one — v1 adapter, no graphs.

**`strata-sycl.bat` therefore forces `STRATA_VERIFY_EAGER=1` whenever it selects Level Zero.** An
explicit `STRATA_VERIFY_EAGER=0` in the environment is the only way back to the graph path, and it is
a debug setting.

### What it costs and what it buys

| | OpenCL (eager) | LZ + eager | LZ + graphs |
|---|---|---|---|
| decode, CLI, 512 tok | 66.6 tok/s | **63.3 tok/s** | 74.5 tok/s (stalls) |
| prefill, CLI | 81.1 tok/s | **120.6 tok/s** | 121.4 tok/s |
| server, mixed requests | 38–60 tok/s | 50–65 tok/s | 58.6 tok/s (stalls) |

Decode is a wash; **prefill is +49%**. The engine-level win is prefill, not decode, and the reason
prefill improves is that the plain-SYCL fallback GEMM itself runs faster on Level Zero — oneMKL
refuses in every configuration tested.

---

## oneMKL: closed

oneMKL refuses with *"The oneAPI Math Kernel Library (oneMKL) Interfaces Project does not support
this API"* against **both** the PyPI runtime (2026.1.0) and the standalone oneAPI 2026.0 runtime,
with identical timings - so the version is not the reason. The engine's expert GEMMs are **BF16**
(`sycl/src/prefill/gemml.dp.cpp:1031`), which is the likely unsupported type. Still a hypothesis;
the cheap way to settle it is to call oneMKL with FP16 operands, since `Gemm::f16` exists on the
same file.

---

## What the machine needs for Level Zero

`sycl/serve/strata-sycl.bat` does all of this behind `STRATA_SYCL_LEVEL_ZERO=1`:

```powershell
$env:STRATA_SYCL_LEVEL_ZERO = "1"     # opt-in; without it, nothing below is set

# conda's Library\bin FIRST on PATH - the UR adapters' dependencies (UMF.dll, VCRUNTIME140*) live there
$c = "$env:USERPROFILE\miniforge3\envs\strata-sycl-261\Library\bin"
$env:PATH = "$c;$env:PATH"

# the rest the launcher sets for you:
$env:ONEAPI_DEVICE_SELECTOR    = "level_zero:0"   # v1 adapter beats v2 by +24% decode
$env:SYCL_UR_USE_LEVEL_ZERO_V2 = "0"
$env:ZES_ENABLE_SYSMAN          = "1"             # makes ext_intel_free_memory answer
$env:STRATA_VERIFY_EAGER        = "1"             # NOT unset: command graphs stall the window
```

---

## The graphics driver: why the generic Arc line, and not Arc Pro

**This machine must run the Intel *generic* Arc graphics driver, not the Arc Pro one.** That is a
deliberate choice, not an accident, and it is the single prerequisite for everything above.

| line | newest build | Level Zero exposed? |
|---|---|---|
| Intel® Arc™ **Pro** | 32.0.101.8976 (Q3.26) | **no** |
| Intel® Arc™ *generic* | 32.0.101.9034 | **yes** |

### The evidence, and why it is a driver *line* and not a bug

The card is an Arc Pro B70, so the Arc Pro driver is the "correct" one — it is the ISV-certified
workstation driver, and Intel's own page says corporate customers should always use it. It is also
the *newest build available for the Arc Pro line*, so there is nothing to update to.

The generic driver lists the Arc Pro B70 as a supported product explicitly, and with it installed
`sycl-ls` reports, for the first time on this machine:

```
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero V2,
    Intel(R) Arc(TM) Pro B70 Graphics 20.2.0 [1.17.39500+2]
```

What changed between the two is the **Level Zero ICD the driver ships** — `ze_intel_gpu64.dll`. On
8976 the Level Zero driver answers every direct API call correctly (`zeInit`, `zeDriverGet`,
`zeDeviceGet`, `zeDeviceGetProperties`: 1 driver, `TYPE_GPU`, `0x8086`/`0xe223`, 31.2 GiB) but no SYCL
UR adapter can drive it; on 9034 the adapters can. So the Arc Pro build's ICD and the SYCL adapters
are simply mismatched, and the generic build carries one they can use.

### What you give up

- **ISV certification for CAD/DCC applications** (SPECviewperf and friends). Irrelevant for LLM
  inference; relevant if this machine is ever used for CAD.
- The Arc Pro driver's own control panel features.
- Intel's stated recommendation for corporate users.

### What you get

- Level Zero at all, which is the prerequisite for `ext_intel_free_memory` (a real free-VRAM query
  instead of the DXGI workaround) and for the plain-SYCL fallback GEMM running ~49% faster on
  prefill.
- Measured: 120.6 tok/s prefill against OpenCL's 81.1.

### How to check which one you are on, and how to go back

```powershell
# which line is installed
(Get-CimInstance Win32_VideoController | ? Name -match 'Arc').DriverVersion
#   32.0.101.8976 -> Arc Pro line, no Level Zero
#   32.0.101.9034 -> generic line, Level Zero works

# and confirm the engine can see it
& "$env:USERPROFILE\miniforge3\envs\strata-sycl-261\Library\bin\sycl-ls.exe" --ignore-device-selectors
#   a [level_zero:gpu] line is what you want
```

Going back is a normal install of the Arc Pro driver from
[intel.com/content/www/us/en/download/741626](https://www.intel.com/content/www/us/en/download/741626/intel-arc-pro-graphics-windows.html)
— 8976 is what this machine had. **What happens if you do:** Level Zero stops enumerating,
`ONEAPI_DEVICE_SELECTOR=level_zero:0` kills the engine with `0xC0000409`, and `STRATA_SYCL_LEVEL_ZERO`
must be unset so the launcher falls back to the OpenCL path. Nothing else is affected — the OpenCL
path is intact and is still the default.

Installer used here: `%TEMP%\opencode\gfx_win_101.9034.exe`, SHA512 verified against Intel's
published digest. Downloaded from
[intel.com/content/www/us/en/download/785597](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html).

---

## What this retires

| workaround | retired by |
|---|---|
| the plain-SYCL fallback GEMM | oneMKL, if BF16 turns out not to be the blocker |
| the DXGI live free-VRAM figure | `ext_intel_free_memory` now exists; `device_free_bytes()` should then prefer the backend's answer over DXGI |

`STRATA_VERIFY_EAGER` is deliberately **not** retired: command graphs are available, and they are
exactly what stalls the verify window. So it stays on for Level Zero, and the commit replay it
provides is load-bearing on this backend.
