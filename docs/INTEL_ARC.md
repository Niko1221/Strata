# Intel Arc (experimental)

Strata 0.1.39 includes an **experimental Intel Arc engine**: Strata's own engine ported to SYCL (Intel oneAPI).
maxfridbe wrote it in [#423](https://github.com/Niko1221/Strata/pull/423), with fixes from the people testing it.
The code is in `sycl/`, and the port's own notes, measurements and maintenance procedure are in
[docs/INTEL.md](INTEL.md). This page covers what you need, how to build it, and what has been tested.

**Experimental means:** the Intel engine is still young. When the port arrived (0.1.39) the maintainers had no
Intel GPU and every result on Arc hardware came from the community. Since 0.1.40.2 the maintainers also test on an Arc Pro B70
(32 GB, `xe`) and an Arc A750 (8 GB, `i915`); the results from those two cards are marked in the table below. Other cards
are still community reports. The NVIDIA and AMD engines are unchanged: the Intel build is a separate CMake target, off by default.

**Where the engine comes from.** There is **no Intel engine in the release zips** (the ones `START-HERE.bat` and `setup.sh`
use): you build it from source on Linux or [native Windows](#windows). A ready-made Intel engine does exist as a Docker image,
`ghcr.io/niko1221/strata:0.2.0-sycl`, built for the 0.2.0 release (the image needs prepared model files; see
[Docker: Intel Arc](DOCKER.md#intel-arc)). The Mac and Windows release packages have no Intel engine.

## What has been run, and by whom

| | Hardware | Result | Reported in |
|---|---|---|---|
| Port author | Arc Pro B70 32 GB, Ubuntu 24.04 | Coder IQ1_M: 70-78 tok/s decode, ~790 tok/s prompt; IQ2_XS: 51-64 tok/s; 256K context measured | #423, [INTEL.md](INTEL.md) |
| Community | 2x Arc Pro B70, `--layer-split` | Flash-Next IQ3_XXS 66 tok/s decode, 394 tok/s prompt (with the `stage_room` fix that is now in 0.1.39) | #423 |
| Community | Arc Pro B50 16 GB | Coder IQ1_M ~23 tok/s, IQ2_XS ~25-27 tok/s, up to 128K | #423 |
| Community | Arc B580 12 GB, WSL2 | IQ2_XS ~21 tok/s, Coder ~15 tok/s; **device loss also seen** | #423 |
| Community | 2x Arc Pro B60 24 GB, `--layer-split` | Coder IQ1_M: 56.6 tok/s decode, 428 tok/s prompt (2,129 tokens); Flash-Next IQ2_XS: 58.6-61.3 tok/s decode, 436 tok/s prompt; all experts in VRAM; 8K context | [bench/results/2026-10-04-community-2x-arc-pro-b60](../bench/results/2026-10-04-community-2x-arc-pro-b60/README.md) |
| Community | one Arc Pro B60 24 GB | Coder IQ1_M with 4,042 of 12,288 experts mirrored in RAM: 11.9 tok/s decode (needs the ring-wait fix from the same report) | same |
| Strata maintainers (0.1.40.2 and later) | Arc Pro B70 32 GB (xe) and Arc A750 8 GB (i915), test machines | Flash-Next IQ3_S on the B70 with 11.5k of 24.6k experts resident: 10-prompt gate passes, prompt 980-1,000 tok/s on 4K tokens, decode 30 (prose) to 41 (code) tok/s; the xe aliased-pages fix and the `NO_HOST` requirement are in [INTEL.md](INTEL.md) | INTEL.md, "Arc Pro B70 with a model that does not fit" |

The 0.1.39 port re-migrates 0.1.38's port onto the 0.1.39 engine sources (the #606 NaN fix, the #649 verify
trace, the new prompt paths). 0.1.39's `sycl/` did not compile against 0.1.39's own engine sources (#784: the
`ThreadAffinity` type of #626 and the layer range of `NativeDense::load` from #559), and its ring waits never saw a
slow layer as still running (#866, #867). Both are fixed in 0.1.40, and the two B60 reports below were measured with
exactly those two fixes on top of 0.1.39. **No one has run an unpatched 0.1.39 port on an Arc**, and the other rows were
measured on earlier versions.

The two B60 rows ran `6f32ec0` plus two small `sycl/` fixes (the compile fix and the ring-wait fix), AOT `bmg-g21`, on Ubuntu 24.04 with
`xe`, Level Zero V2, NEO 26.09.37435.12 and oneAPI 2026.1.1, without Docker. Host: Ryzen 5 5600, 64 GB RAM, PCIe 3.0 x8 per card.
Full flags, per-request timings and engine logs are in the report linked in the table.

## What was tested here (0.1.39)

- **Compiles:** Ubuntu 24.04 (WSL2), Intel oneAPI DPC++ 2026.1.1 + oneMKL 2026.1. The whole `sycl/` project
  builds with 0 errors: the `strata` engine plus all 157 targets (the kernel parity tests and benches). The build is
  SPIR-V (JIT). The AOT build (`STRATA_SYCL_AOT`) was not built here, because it needs `ocloc`.
- **Kernel parity tests on a CPU** (`ONEAPI_DEVICE_SELECTOR=opencl:cpu`, Intel's OpenCL CPU runtime, AMD Ryzen 5 7600):
  14 of 25 pass. That is the same set that PR #423's own 0.1.38 port passes on that device: the failures are tight
  float tolerances on the CPU's math (rel 3e-6 against a 1e-6 limit), model fixtures that are not present, and two
  tests that time out on a CPU. This is a check that the kernels compile and compute, not a test of an Arc.
- **Setup:** `--backend sycl` warns, then hands over to `sycl/setup_intel.py` on Linux (unit tests:
  `tools/test_setup_sycl.py`).
- **Unchanged:** the CUDA engine's greedy output, checked byte-identical against the gated 0.1.39 build (Q2_0 and Coder).
  The HIP build is not affected (the option is off by default).

**Not covered by those 0.1.39 checks:** native Windows (tested later; see below), Arc on WSL2 for the 0.1.39 port, integrated Arc
GPUs (the B390 / Panther Lake in #515; the shared-memory planning does not exist yet), and images (not wired on Intel: there
is no picture reader for the Arc engine). The Alchemist A-series was tested later, on an A750 (rows above and
[INTEL.md](INTEL.md), "Arc A750 and the other Alchemist cards").

## What you need

- **Linux**, e.g. Ubuntu 24.04, with Intel's GPU driver (the `xe` or `i915` kernel driver plus the compute runtime /
  Level Zero; on Ubuntu, `intel-opencl-icd libze1 libze-intel-gpu1`, or Intel's
  [client GPU guide](https://dgpu-docs.intel.com/driver/client/overview.html)).
- **Intel oneAPI**: the DPC++ compiler (`icpx`, 2025.3 or newer; 2026.1 is what was built here) and **oneMKL**. About 5 GB.
- `cmake` 3.24+, `ninja`, `git` (the build fetches ggml unless you point `STRATA_GGML_DIR` at a llama.cpp checkout),
  Python 3.
- For `./setup.sh --backend sycl` today: **Docker**, and nothing else from the list above except the GPU driver.
  `sycl/setup_intel.py` runs the engine in the `strata-sycl-dev` image built from `sycl/tools/Dockerfile`, and the
  engine is built in that image too ("Build with Docker" below). On Ubuntu: `sudo apt install docker.io git`, then
  `sudo usermod -aG docker,render $USER` and log in again. Note that the Dockerfile starts from a community
  llama.cpp SYCL image (`ghcr.io/snailium/...`), not an Intel or Strata image.
- VRAM: the port keeps the experts on the card (`--stream-experts`). A 32 GB card holds the Coder IQ1_M or IQ2_XS.
  Smaller cards mirror part of the experts in RAM and are slower.

## Build with Docker (what `setup --backend sycl` uses)

Two commands, the same as in [INTEL.md](INTEL.md), "How to build it" (about 10 minutes the first time). From the checkout,
called `Strata` here:

```sh
cd Strata
docker build -t strata-sycl-dev sycl/tools            # once: ~6 minutes, a 13 GB image
H=$(basename "$PWD")
# Arc B-series: bmg-g31 (Pro B70), bmg-g21 (B580, B570, Pro B60). Arc A-series (Alchemist): leave out AOT (JIT) and use build-sycl
docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
    -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"
```

Then `./setup.sh --backend sycl` (below). Do not run `python3 sycl/setup_intel.py` by hand: on Ubuntu it fails with
`externally-managed-environment`; `setup.sh` runs it in its own `.venv`.

## Build without Docker (Linux)

This builds the engine for running it by hand ("How to run it by hand" in [INTEL.md](INTEL.md)); `setup` itself uses the
Docker build above. Install oneAPI from Intel's apt repository (this is what was used here):

```sh
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel ninja-build cmake git
```

Then build the engine. Either command works: the first goes through the top-level CMake option, the second
configures the `sycl/` project directly.

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON
#   or: cmake -S sycl -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl --target strata
```

Options:

- `-DSTRATA_SYCL_AOT=bmg-g31` (Arc Pro B70) or `bmg-g21` (B580 / B570 / Pro B60) compiles the GPU code ahead of
  time. This needs `ocloc` (Intel's `intel-ocloc` package). Without it, the first start JIT-compiles every kernel,
  which takes about 47 s.
- `-DSTRATA_SYCL_PARITY=OFF` skips the kernel tests (on by default). Run them with
  `ctest --test-dir build-sycl` on the card.

`setup_intel.py` looks for `build-sycl-aot/strata` or `build-sycl/strata` in the checkout. With the
top-level option the engine is at `build-sycl/sycl/strata`, so either use `-S sycl` or set `STRATA_SYCL_BIN`.

## Setup and running

```sh
./setup.sh --backend sycl [setup's usual options, e.g. --model IQ2_XS --context 32768]
```

This prints the experimental warning and continues with `sycl/setup_intel.py` (in setup's own Python environment).
That script finds the Arc in sysfs, uses the SYCL engine you built, and writes the config and `run-<model>.sh`.
On a card under 12 GB or an `i915` card (the A-series) it writes the small-card config ([INTEL.md](INTEL.md), "Arc A750
and the other Alchemist cards"). It still downloads and packs the model
the usual way. To run the engine by hand (no Docker), see "How to run it by hand" in [INTEL.md](INTEL.md).

Things that matter on an Arc (details in INTEL.md):

- **Do not ask for more VRAM than the card has.** On the `xe` driver, an allocation past VRAM can push buffers into
  RAM until the machine runs out of memory and stalls. Leave about 1.5 GB free.
- `SYCL_CACHE_PERSISTENT=0`: the persistent JIT cache crashed on Xe2 during the first compile.
- Two cards: `ONEAPI_DEVICE_SELECTOR=level_zero:*` (the image pins `level_zero:0`; `strata-sycl.sh` now passes the
  variable through) and `--layer-split`.
- **`--layer-split auto` on two equal cards (0.1.44, #1440):** when several splits hold all the experts and predict the
  same speed, the search now takes the one that leaves its fullest card the most VRAM to spare (within 0.2% of the best
  predicted time, `STRATA_SPLIT_HEADROOM_TOL`). It used to take the first one tried, which filled the second card to its
  last MiB. A full card under `xe` moves buffers to system memory instead of failing: 2x B70 with IQ3_S, auto picked
  K=24 (~11 tok/s) where `--layer-split 30` ran ~54. The startup line names both splits when the choice moves.
  `STRATA_SPLIT_HEADROOM=0` keeps the old pick.
- **Arc Pro B60:** the PCI id is `8086:e211` (`lspci -nn`, the kernel's `xe` id list files it with the BMG-G21 cards), `sycl-ls` prints
  `Intel(R) Arc(TM) Pro B60 Graphics 20.1.0`, and `ocloc ids bmg-g21` prints 20.1.0, so `-DSTRATA_SYCL_AOT=bmg-g21` is the right target.
  `setup_intel.py` knows both B60 ids, `e211` and `e221` (both seen on B60 cards). On a `--layer-split` the startup line "N experts are neither in VRAM nor mirrored"
  counts the other card's layers; the lines `100% of the experts resident` that follow are the ones to read. Set `STRATA_MIRROR_MIB=0` there,
  or the pinned mirror is allocated and not used.
- **`setvars.sh` and `set -u`:** `source /opt/intel/oneapi/setvars.sh` in a shell with `set -u` stops at `OCL_ICD_FILENAMES: unbound variable`
  (oneAPI 2026.1.1). Source it before `set -u`, or run `set +u` around it.
- **Without Docker:** `sycl/serve/strata-sycl.sh` needs Docker. Natively, `source setvars.sh`, export `SYCL_CACHE_PERSISTENT=0`,
  `ZES_ENABLE_SYSMAN=1` and `STRATA_VERIFY_DEVICE_PLAN=1`, then run `build-sycl-aot/strata` or `sycl/serve/server_intel.py` with an `exe` that does this.
  A normal user is not in the `render` group on Ubuntu, and then `sycl-ls` shows only the CPU device.
- `STRATA_VERIFY_NO_HOST=1` (set by `strata-sycl.sh` unless the config's `"env"` says otherwise) is needed on `xe`
  cards when part of the experts is not in VRAM (the host-to-GPU flag stores are not visible there, NaN logits
  otherwise) and is left off on the `i915` Arc A-series: see [INTEL.md](INTEL.md). The config's `"env"` block and any
  `STRATA_*`, `ONEAPI_*`, `UR_*`, `IGC_*`, `SYCL_*` or `ZES_*` variable of the server's environment reach the engine
  through `strata-sycl.sh` (for example `STRATA_MIRROR_MIB`, `IGC_EnableDPEmulation`).

## Windows

Windows has a native SYCL build and server path; Docker and WSL2 are not needed. Install:

- **Visual Studio 2022 or its Build Tools**, with the **Desktop development with C++** workload, MSVC x64 tools
  and a Windows SDK. The build script finds Visual Studio with `vswhere` and loads `vcvars64.bat`.
- **Intel oneAPI DPC++ and oneMKL**, including their runtime dependencies (TBB and TCM's `libhwloc-15.dll`).
  The tested compiler was 2026.1.1 with oneMKL 2026.1.0; use matching compiler/runtime versions.
- **An Intel GPU driver with Level Zero**, GPU free-memory queries and command graphs. OpenCL enumeration alone
  is not enough: setup and the engine reject it before loading the model.
- **CMake 3.24+, Ninja, Git and Python 3**.

The **Level Zero SDK** supplies `level_zero/ze_api.h` and `ze_loader.lib` for direct Level Zero host allocations.
It is optional: CMake enables that path only when both are found, otherwise it uses SYCL host allocations.
For a nonstandard SDK location, pass `-DSTRATA_ZE_INCLUDE_DIR=<include directory>` and
`-DSTRATA_ZE_LOADER=<full path to ze_loader.lib>` to the build script. A working Level Zero runtime is required
either way.

From the checkout, in Command Prompt:

```bat
sycl\tools\build_windows.bat
START-HERE.bat --backend sycl --setup
```

The engine is `build-sycl-win\strata.exe`. Setup builds it if missing; `--build` rebuilds it. On an Intel-only PC,
`START-HERE.bat` selects Intel automatically; with other usable GPUs, the original backend remains the default.
Model selection, downloads and packing use the usual setup flow, and the native server uses the saved config.
The script loads the standard oneAPI installation; `STRATA_ONEAPI_ROOT` can instead name a directory containing
`bin\icx.exe`, `include` and `lib` (the tested local layout was `%LOCALAPPDATA%\Strata-tools\oneapi\Library`).

Setup defaults to `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` and preserves an explicit selector. If the GPU probe
fails, check the driver and runtime DLL paths, including oneAPI's `tcm\latest\bin` when an adapter cannot load
`libhwloc-15.dll`. Seeing an OpenCL GPU in `sycl-ls` does not confirm Level Zero is working.

**Validated in #2079:** Windows 11 Pro for Workstations Insider build 26220, Core i9-13900KF, 64 GB RAM,
Arc Pro B70 32 GB, driver 32.0.101.9034, Visual Studio 2022 17.14.40 and the oneAPI versions above. The merged
`ac9d0800` build completed all native SYCL targets; setup ran 594 tests with no failures and 2 skips, and 31 GPU
regressions passed. Three existing tests were excluded: `gr_parity` (unsupported QFUSE), `ple_parity` (missing
external fixtures) and `s2_expert_grouped_parity` (bitwise differences). Level Zero capability reporting and
OpenCL rejection were also checked on the card.

Earlier measurements on the 0.1.43-based Windows branch used IQ2_XS, native INT8 KV, 262,144-token capacity,
full-GPU KV, prefill chunks of 4,096, greedy sampling and MTP 4: one warmed run with 131,072 uncached prompt
tokens and 256 generated tokens measured 1,018.1 tok/s prompt and 57.6 tok/s decode. Experts were split between
VRAM and GPU-readable system RAM; this was not an all-weights-in-VRAM run. These are not performance
measurements of the later merge. Other Windows GPUs, native Windows multi-GPU and images were not validated
by this PR.

## Reporting a problem

Open an issue with: the card, the driver version, `sycl-ls` output, the oneAPI version, the model and flags, and the
engine's stderr. Results from real cards are what move this from experimental to supported.
