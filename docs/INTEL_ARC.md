# Intel Arc on Linux

Strata runs on Intel Arc cards on Linux through its own engine ported to SYCL (Intel oneAPI). This page is for people with
an Arc A-series (Alchemist) or B-series (Battlemage) card. It covers the released Terminal route, what you need, how to
build the engine from source, and which cards were tested. The port's workflow and measurements are in [INTEL.md](INTEL.md).

maxfridbe wrote the port in [#423](https://github.com/Niko1221/Strata/pull/423). The code is in `sycl/`. Intel Arc is an
official platform since 0.1.40.2. The maintainers test on an Arc Pro B70 (32 GB, `xe` driver) and an Arc A750 (8 GB, `i915`
driver); results from other cards are community reports. The Mac and Windows builds have no Intel engine upstream; this fork adds a native Windows build for the
B-series (see "Windows" below), with its measurements in the tested-cards table.

## Quick start

Install the [Terminal](INSTALL.md#the-terminal), then:

```sh
strata update --check     # shows the Intel engine package the feed would install
strata doctor --live      # checks the driver, the render group and the card
strata launch --plan      # shows what would happen, changes nothing
strata launch
```

On Linux, `strata update` (Terminal) and Settings > Update (app) download the Intel engine package `linux-x86_64-sycl` for a PC
whose graphics card is made by Intel. One package serves Alchemist and Battlemage: the GPU code is compiled from SPIR-V at the
first start, because the A-series cards have no FP64 units. The package carries oneAPI's runtime libraries. The Level Zero
loader and the compute runtime stay the PC's own (see [What you need](#what-you-need)), and your user must be in the `render`
group (`strata doctor` says so). It was tested on an Arc A750 only. The package is built by `packaging/build_sycl_linux.sh`.

**Model preparation is separate.** `strata launch` reuses compatible prepared local files, but it does not run the Intel
source or container setup for you. When a model needs preparation, follow the [source workflow](INTEL.md#setup). Installing the
engine package alone does not prove the selected model is ready; check a loaded server with a real request.

Docker: the ready-made image is `ghcr.io/niko1221/strata:0.2.3-sycl`, published only for a release that carries a Linux Intel
engine package. Otherwise [build the image yourself](DOCKER.md#build-it-yourself). The image needs prepared model files
([Docker: Intel Arc](DOCKER.md#intel-arc)).

## What you need

- **Linux**, for example Ubuntu 24.04, with Intel's GPU driver: the `xe` or `i915` kernel driver plus the compute runtime and
  Level Zero. On Ubuntu: `intel-opencl-icd libze1 libze-intel-gpu1`, or follow Intel's
  [client GPU guide](https://dgpu-docs.intel.com/driver/client/overview.html).
- **The CPU and the model folder.** The CPU does not need AVX-512. A Ryzen 3000-5000 or an Intel Core 12th-14th generation runs the
  experts that do not fit on the card with AVX2 (measured on an A750 with a Ryzen 5 5600X, [INTEL.md](INTEL.md)). This applies to
  model folders that contain `native_experts.txt` (IQ3_XXS, IQ3_S, NVFP4, Unsloth). An older Q2_0 folder has no such file, and its
  CPU kernel needs AVX-512. With that folder the engine stops with "this CPU cannot run the expert kernel", and `strata doctor`
  warns about it before the start. Building the model folder again with `strata setup` (or `tools/iq_pack.py`) fixes it.
- For a source build without Docker: **Intel oneAPI** with the DPC++ compiler (`icpx`, 2025.3 or newer) and **oneMKL**
  (about 5 GB), `cmake` 3.24+, `ninja`, `git`, Python 3. The build fetches ggml unless you point `STRATA_GGML_DIR` at a
  llama.cpp checkout.
- For `./setup.sh --backend sycl`: **Docker** and the GPU driver, nothing else from the list above. `sycl/setup_intel.py`
  builds and runs the engine in the `strata-sycl-dev` image from `sycl/tools/Dockerfile`, which starts from a community
  llama.cpp SYCL image (`ghcr.io/snailium/...`), not an Intel or Strata image. On Ubuntu:
  `sudo apt install docker.io git`, then `sudo usermod -aG docker,render $USER` and log in again.
- VRAM: the port keeps the experts on the card (`--stream-experts`). A 32 GB card holds the Coder IQ1_M or IQ2_XS. Smaller
  cards mirror part of the experts in RAM and are slower.
  Under the wrapper's `STRATA_VERIFY_NO_HOST=1` there is no SSD expert tier on `xe`: every expert must
  be in VRAM or in the pinned RAM mirror (`STRATA_MIRROR_MIB`), and the engine refuses to start when
  the two do not cover the model. Size the RAM accordingly, or lower `--max-context`.

## Build with Docker (what `setup --backend sycl` uses)

About 10 minutes the first time. From the checkout, called `Strata` here:

```sh
cd Strata
docker build --pull -t strata-sycl-dev sycl/tools     # once: about 6 minutes, a 13 GB image
H=$(basename "$PWD")
# Arc B-series: bmg-g31 (Pro B70), bmg-g21 (B580, B570, Pro B60). Arc A-series (Alchemist): leave out AOT (JIT) and use build-sycl
docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
    -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"
```

`--pull` matters: [INTEL.md](INTEL.md#how-to-build-it) says why. Then run `./setup.sh --backend sycl`. Do not run
`python3 sycl/setup_intel.py` by hand: on Ubuntu it fails with `externally-managed-environment`; `setup.sh` runs it in its own
`.venv`.

## Build without Docker

This builds the engine for running it by hand ([INTEL.md](INTEL.md#how-to-run-it-by-hand)). `setup` itself uses the Docker build.
Install oneAPI from Intel's apt repository:

```sh
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel ninja-build cmake git
```

Build the engine through the top-level CMake option or by configuring the `sycl/` project directly:

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON
#   or: cmake -S sycl -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl --target strata
```

| Option | Effect |
| --- | --- |
| `-DSTRATA_SYCL_AOT=bmg-g31` | Compile ahead of time for Arc Pro B70. Needs `ocloc` (Intel's `intel-ocloc` package). Without it the first start JIT-compiles every kernel (about 47 s). |
| `-DSTRATA_SYCL_AOT=bmg-g21` | The same for B580, B570 and Pro B60 |
| `-DSTRATA_SYCL_PARITY=OFF` | Skip the kernel tests (on by default). Run them with `ctest --test-dir build-sycl` on the card. |

`setup_intel.py` looks for `build-sycl-aot/strata` or `build-sycl/strata` in the checkout. With the top-level option the
engine is at `build-sycl/sycl/strata`, so use `-S sycl` or set `STRATA_SYCL_BIN`.

## Setup and running

```sh
./setup.sh --backend sycl [setup's usual options, e.g. --model IQ2_XS --context 32768]
```

Setup continues with `sycl/setup_intel.py`, which finds the Arc in sysfs, uses the SYCL engine you built, and writes the
config and `run-<model>.sh`. On a card under 12 GB or an `i915` card (the A-series) it writes the small-card config
([INTEL.md](INTEL.md#arc-a750-and-the-other-alchemist-cards)). It downloads and packs the model the usual way. Windows has no
path: `setup --backend sycl` on Windows stops and points here.

Things that matter on an Arc (details in [INTEL.md](INTEL.md)):

- **Do not ask for more VRAM than the card has.** On the `xe` driver an allocation past VRAM can push buffers into RAM until
  the machine runs out of memory and stalls. Leave about 1.5 GB free.
- `SYCL_CACHE_PERSISTENT=0`: the persistent JIT cache crashed on Xe2 during the first compile.
- Two cards: `ONEAPI_DEVICE_SELECTOR=level_zero:*` (the image pins `level_zero:0`; `strata-sycl.sh` passes the variable
  through) and `--layer-split`.
- `--layer-split auto` on two equal cards takes the split that leaves the fullest card the most VRAM to spare, within 0.2% of
  the best predicted time (`STRATA_SPLIT_HEADROOM_TOL`). `STRATA_SPLIT_HEADROOM=0` turns that off. A full card under `xe` moves
  buffers to system memory instead of failing. The startup line names both splits when the choice moves.
- **Two Arc B-series cards decode faster by default.** With exactly two B-series cards (B50, B60, B65, B70, B570, B580) in a layer
  split, the first card starts the next window while the second still checks this one (`--pipeline-windows 2`), and setup writes
  `--spec 2` for the pair. On 2x Arc Pro B65 this decodes 17% to 23% faster than the old serial `--spec 4` setup, 5 pairs out of 5:
  chat 71-74 tok/s against 60-63, JSON 80-83 against 67-68, code 72-78 against 59-65, plain text 67-68 against 56-57. The answers are
  those of a serial `--spec 2` run (100 of 100 agent requests); a serial `--spec 4` run differs in a rare near-tie (3 of 100).
  One card, an A-series card, a mixed pair and three or more cards run as before. `--pipeline-windows 0` turns it off, and a
  `--spec` of your own, put in the config's `user_args`, is kept.
- **Arc Pro B60:** PCI id `8086:e211` (`lspci -nn`), `sycl-ls` prints `Intel(R) Arc(TM) Pro B60 Graphics 20.1.0`, so
  `-DSTRATA_SYCL_AOT=bmg-g21` is the right target. `setup_intel.py` knows both B60 ids, `e211` and `e221`. On a `--layer-split`
  the startup line "N experts are neither in VRAM nor mirrored" counts the other card's layers; read the following
  `100% of the experts resident` lines. Set `STRATA_MIRROR_MIB=0` there, or the pinned mirror is allocated and not used.
- **`setvars.sh` and `set -u`:** `source /opt/intel/oneapi/setvars.sh` in a shell with `set -u` stops at
  `OCL_ICD_FILENAMES: unbound variable` (oneAPI 2026.1.1). Source it before `set -u`.
- **Without Docker:** `sycl/serve/strata-sycl.sh` needs Docker. Natively, `source setvars.sh`, export `SYCL_CACHE_PERSISTENT=0`,
  `ZES_ENABLE_SYSMAN=1` and `STRATA_VERIFY_DEVICE_PLAN=1`, then run `build-sycl-aot/strata` or `sycl/serve/server_intel.py`
  with an `exe` that does this. A normal user is not in the `render` group on Ubuntu, and then `sycl-ls` shows only the CPU.
- `STRATA_VERIFY_NO_HOST=1` (set by `strata-sycl.sh` unless the config's `"env"` says otherwise) is needed on `xe` cards when part
  of the experts is not in VRAM, and is left off on the `i915` A-series ([INTEL.md](INTEL.md)). The config's `"env"` block and
  any `STRATA_*`, `ONEAPI_*`, `UR_*`, `IGC_*`, `SYCL_*` or `ZES_*` variable of the server's environment reach the engine
  through `strata-sycl.sh`.

## Tested cards

> **Measured on:** the hardware in each row, Linux. Versions are the engine version of that report, not the current release.

| Who | Hardware | Result | Reported in |
| --- | --- | --- | --- |
| Port author | Arc Pro B70 32 GB, Ubuntu 24.04, 0.1.31-sycl | Coder IQ1_M: 70-78 tok/s decode, about 790 tok/s prompt; IQ2_XS: 51-64 tok/s; 256K context | #423, [INTEL.md](INTEL.md) |
| Community | 2x Arc Pro B70, `--layer-split` | Flash-Next IQ3_XXS 66 tok/s decode, 394 tok/s prompt | #423 |
| Community | Arc Pro B50 16 GB | Coder IQ1_M about 23 tok/s, IQ2_XS about 25-27 tok/s, up to 128K | #423 |
| Community | Arc B580 12 GB, WSL2 | IQ2_XS about 21 tok/s, Coder about 15 tok/s; device loss also seen | #423 |
| Community | 2x Arc Pro B60 24 GB, `--layer-split`, Ryzen 5 5600, 64 GB, PCIe 3.0 x8 per card | Coder IQ1_M: 56.6 tok/s decode, 428 tok/s prompt (2,129 tokens); Flash-Next IQ2_XS: 58.6-61.3 tok/s decode, 436 tok/s prompt; all experts in VRAM; 8K context | [report](../bench/results/2026-10-04-community-2x-arc-pro-b60/README.md) |
| Community | One Arc Pro B60 24 GB | Coder IQ1_M with 4,042 of 12,288 experts mirrored in RAM: 11.9 tok/s decode | same |
| Strata on Windows (this fork's native port; app and engine numbers measured on 0.1.47, rebuilt and parity-checked on 0.1.51, 0.1.52 and 0.1.54) | Arc Pro B70 32 GB (xe), Core i9-9900, 64 GB, Windows 11 25H2, oneAPI 2026.2, Level Zero V2 | Coder IQ1_M: prompt 1,451 / 1,407 tok/s (4K / 16K) as installed and 1,340 / 1,351 tuned; decode 51 / 62 on the chat questions, 66 / 79 on nine short-answer questions, agent session 58 / 64. IQ3_XXS with its experts streaming: 40.3 tok/s on the mirror path, 64.6 on the arena path (the default now when they do not fit). See "Windows" below | [Coder IQ1_M](../bench/results/2026-10-21-community-arc-pro-b70-coder-iq1_m/), [all experts resident](../bench/results/2026-10-21-community-arc-pro-b70-coder-iq1_m-all-resident/), [IQ3_XXS](../bench/results/2026-10-21-community-arc-pro-b70-iq3_xxs/) |
| Maintainers | Arc Pro B70 32 GB (xe), 0.1.40.2 and later | Flash-Next IQ3_S with 11.5k of 24.6k experts resident: 10-prompt gate passes, prompt 980-1,000 tok/s on 4K tokens, decode 30 (prose) to 41 (code) tok/s | [INTEL.md](INTEL.md#arc-pro-b70-with-a-model-that-does-not-fit-flash-next-iq3_s) |
| Maintainers | Arc A750 8 GB (i915) | Flash-Next IQ3_XXS about 10-19 tok/s decode | [INTEL.md](INTEL.md#arc-a750-and-the-other-alchemist-cards) |
| Maintainers | 2x Arc Pro B65 32 GB (xe), 0.2.8 | Flash-Next IQ3_S, every expert in VRAM, the JIT package build: decode 71-83 tok/s (chat, JSON, code), prompts 1,261 / 1,511 / 1,628 tok/s at 4K / 8K / 16K | the bullet above |

Two cards on `xe` (2x Arc Pro B70, IQ3_S, 0.1.44): `--layer-split auto` picked a split at about 11 tok/s where
`--layer-split 30` ran about 54. The auto choice now prefers the split with the most VRAM headroom (above).

**Not tested by anyone upstream:** Arc on WSL2 with the current engine, integrated Arc GPUs (the B390 /
Panther Lake in #515; shared-memory planning does not exist), and pictures on a Battlemage card with a GPU reader.
Pictures work on the A750 (on by default, read on the CPU). This fork adds a native Windows path for the B-series,
with the measurements in the table above and in the Windows section below. See [INTEL.md](INTEL.md#pictures-on-an-arc).

## Windows

Windows has a native path: no Docker, no WSL, no bash. `START-HERE.bat` finds an Intel Arc through the display
registry and offers it beside NVIDIA / AMD when both are present, then hands the run to `sycl/setup_intel.py` as on
Linux. The engine is `build-sycl-win\strata.exe`, which setup builds with `sycl\tools\build_windows.bat` when it is
missing, and the run config carries the oneAPI runtime folders in `lib_dirs` instead of the container's `sycl_root`.
The build uses MSVC and oneAPI (`icx` + oneMKL); SYCL's GCC-style flags are spelled for icx on Windows in
`sycl/CMakeLists.txt` (`/fp:precise`, `/Qmkl:sequential`), and the few POSIX calls are behind `#ifdef _WIN32`.

What has been run â€” measured on 2026-10-20 on Windows 11 25H2, Core i9-9900, Arc Pro B70 32 GB, driver 32.0.101.8976,
oneAPI 2026.2, MSVC 14.51 (Visual Studio 18):

- `sycl\tools\build_windows.bat`: the whole `sycl` project, 232 targets, 0 errors, 50 executables, among them
  `strata.exe` (40 MB) and the kernel parity tests.
- `positioned_file_test.exe`: PASS â€” 8 threads on one shared handle, an offset past 4 GiB, short reads, end of file,
  an invalid handle, a missing file.
- `python tools\test_setup_*.py`: all pass except `test_setup_golden.py` (19) and `test_setup_unsloth.py` (4), which fail
  the same way on upstream main without the Windows work (0.2.3.1: 46 of 48 files pass).
- the model: Coder IQ1_M on the B70 through Level Zero (11,264 experts resident, 21.46 GiB of VRAM, ready in about
  58 s from a cold start), `tools/strata_bench.py`, 3 runs per test, medians: chat decode 51.3 tok/s (34.1-66.6),
  first token 0.45 s; a 4K prompt 1,451 tok/s, first token 2.81 s; a 16K prompt 1,407 tok/s, first token 11.47 s; an
  agent session 57.7 tok/s decode (26.8-74.4), 69.3% of each turn's prompt read from the conversation cache. The
  Linux B70 row for the same model is 70-78 tok/s decode and ~790 tok/s prompt: prompt reading is clearly ahead on
  Windows, and decode depends heavily on the question mix (see the next line). This PC is a Core i9-9900 with 64 GB
  of RAM and the JIT build (`STRATA_SYCL_AOT` not set); the 32K prompt test skipped itself because the context is
  exactly 32,768 tokens.
- Two follow-ups on that decode figure. The AOT build (`STRATA_SYCL_AOT=bmg-g31`, built beside the JIT one) measured
  the same (chat 50.0 vs 51.3 tok/s, 4K prompt 1,394 vs 1,451, agent decode 61.4 vs 57.7: within run-to-run noise), so
  JIT is not what the decode number is about. And the chat median moves with the questions asked: a fixed nine-question
  set (three prompts, three rounds) gave 66.0 tok/s (56.6-70.5) on the same server, which overlaps the Linux figure.
  `--spec 2` instead of 4 gave 56.3 tok/s on those nine, so the default T=4 is the better setting here.
- Fitting every expert changes the balance. The default `--prefill 4096` and 1 GiB reserve leave 11,264 of the Coder
  IQ1_M's 12,288 experts in VRAM (21.46 GiB of 31.16). With `--prefill 2048 --vram-reserve-mib 512` the cache holds
  12,131 (23.12 GiB), and on the same suite chat decode is 61.6 tok/s (53.2-82.8, against 51.3), the agent session
  63.8 (57.7), first token 0.39 s (0.45); the 4K prompt falls to 1,340 tok/s (from 1,451) and 16K to 1,351 (from
  1,407), because the chunk is half the size. On the nine fixed short questions the same change is 78.6 against 66.0.
  That is a trade for chat-heavy use, not a default: the prompt path wants the bigger chunk. The tuned figures are
  engine 0.1.47; the default ones they compare against are 0.1.46.1, and the engine measured the same on both
  (chat 61.8 -> 61.6).
- Three tuning checks on the same unit, so nobody has to repeat them. The prompt chunk trades directly against decode:
  `--prefill 2048` holds 12,219 experts and gives 76.9 tok/s on the nine questions with 1,284 tok/s on 4K; 4096 holds
  11,524 and gives 71.2 / 1,350; 8192 holds 10,015 and gives 53.3 / 1,423; `auto` holds 12,086 and gives 75.9 / 1,244.
  `--batch 4` is a loss on this port: four concurrent 100-token requests total 29.5 tok/s (7-9 each) against 71 for
  one stream, and the batch path drops the cache to 10,223 slots, so serial is the setting for agents here. The card
  is not clock-limited: under decode Sysman reports the GPU at its 2,800 MHz TDP ceiling (driver request 4,250, no
  throttle reasons) and memory at 19,000 MHz, so power and clocks leave nothing on the table.

**Check Level Zero first: the engine needs it.** The port's config asks for the free-VRAM query and command graphs,
so `sycl/setup_intel.py` refuses a card that SYCL offers only through OpenCL. `sycl-ls` is the check: a
`[level_zero:gpu]` line for the card means the engine will run, an `[opencl:gpu]` line on its own means it will not.
On the PC measured here it works: `sycl-ls` lists the Arc as `[level_zero:gpu] ... Unified Runtime over Level-Zero V2
... 20.2.0 [1.15.39183+4]`, and `strata.exe --list-devices` reports `"driver":"level_zero"` with
`free_memory_supported` and `command_graph_supported` both true, against the driver's Level Zero API 1.15.

If the card appears as OpenCL only, check `HKLM\SOFTWARE\Intel\oneAPI\LevelZero\1\LevelZeroLoaderPath`: it has to be
the **folder** that holds `ze_loader.dll` (`C:\Windows\System32`), not the DLL itself. The 2026 adapter carries a
static loader that appends `ze_loader.dll` to that value (`create_library_path` in `source/inc/ze_util.h` of Level
Zero), so a file path makes it look for `...\ze_loader.dll\ze_loader.dll`, `zeInit` then answers
`ZE_RESULT_ERROR_UNINITIALIZED`, and SYCL reports no Level Zero platform while a direct `zeInit` through the loader
still succeeds. That was the whole failure measured here: the value had been written by hand as a file path.
`UR_L0_DEBUG=1 sycl-ls` prints the adapter's own reason, and `ZEL_LIBRARY_PATH` set to the folder is the per-process
equivalent. Fixing it needs an administrator:

```powershell
Set-ItemProperty 'HKLM:\SOFTWARE\Intel\oneAPI\LevelZero\1' -Name LevelZeroLoaderPath -Value 'C:\Windows\System32'
```

The kernel parity tests that need no model were run on the B70 through Level Zero the same day with `ctest`, which
applies each test's own arguments and environment: 30 of 31 pass (rope, sampler, elementwise, cvec, kv_q8 / q4 /
hybrid / steps / stream, quantize_act, router_top10, s_gemv and its q8k form, s2_gemv and its q8 form, bf16_gemv,
dequant_s2, gr_multi, gdn, verify, qsa, esimd_moe, resident_plan, shared_expert, sycl_router_fast, esimd_moe_mode and
sycl_payload_pairs; `xmx_group` skips itself without the AOT build). One fails: `s2_expert_grouped_parity`'s 20-group
`moe_grouped_s2` and resident cases, where the new grouped kernel differs bitwise from the previous one on Battlemage.
Measured: the worst absolute difference is 5.7e-06 on an output scale of 38 (1.5e-7, one to two float32 ULPs), it is
identical across runs (deterministic, not a race), and with an fp16 output the result is bit-identical; both kernels
also stay within 1e-7 of the host reference, and the "hit" variants are bitwise identical. So it is how the new kernel
associates the sum, not a wrong answer; the test's bitwise contract is left as it is. `gr_parity`'s
`STRATA_QFUSE` block prints `SKIP (the SYCL port does not build STRATA_QFUSE)` instead of failing, because the kernels
return false with "SYCL port: never writes the q8_1 images". Run them through `ctest`: `sycl_payload_pairs`
needs `STRATA_DOORBELL_CHECK=1`, because its resident-publish path copies `x` only with that check, and a bare run then
reports 120 failures that are not there. `qsa_prompt_attn_parity` prints PASS for int8 and fp16 at 32K and exits 2.

**If the engine dies with `0xC0000005` at startup, look at `C:\Windows\System32\ze_loader.dll`.** Windows keeps one
Level Zero loader there and every process uses it. A graphics driver installed from a manual extraction can leave an
old one behind (1.2.3, 2021, in the case measured here) while the driver ships its own beside `ze_intel_gpu64.dll` in
the DriverStore (1.32.0, the same driver that ships `ze_intel_gpu64.dll` 32.0.101.8976). The old loader has no
`zeInitDrivers`, which the 2026 SYCL runtime asks for, so the runtime crashes instead of reporting no device. Compare
the two `FileVersion`s; replacing the System32 copy with the driver's needs an administrator, and the file is owned by
`TrustedInstaller` and usually mapped by Intel Graphics Software, so it has to be renamed aside rather than
overwritten. `sycl-ls` stops crashing with the driver's loader in place.

Re-checked after merging upstream's 291 commits (0.2.2, engine 0.1.47) on 2026-10-21: the build is clean again (234
targets, 51 executables, upstream's new `tq_sycl_parity` among them and passing on the B70), `positioned_file_test`
passes, 30 of 31 model-free parity tests pass, and the tuned figures above come from that engine. The default
configuration was re-measured too and did not move either: chat decode 50.1 tok/s against 51.3, first token 0.45 s both
times, 4K prompt 1,436 against 1,451, 16K 1,387 against 1,407, agent session 62.1 against 57.7, cache 11,258 experts
against 11,264 - every pair inside the run-to-run spread, which is wider than the difference between the two engines
(the same question measured 34 to 68 tok/s within one configuration). Upstream's trellis and Gyro kernels do not apply
to the models measured here. One workflow note: a running server holds
`build-sycl-win\strata.exe`, so the link fails with `LNK1104` until it is stopped.

**0.1.48.1 (0.2.3.1, 2026-10-23).** Upstream's release made Intel Arc a released engine package on Linux and turned
pictures on for every Arc card, B-series included. The merge needed one conflict resolved in this file (upstream's
Linux paragraph kept, the Windows sentence restored) and one decision: on Windows pictures stay off by default,
because no picture request has been run on this port. Rebuilt with `sycl\tools\build_windows.bat`: 234 targets, 0
errors, 51 executables, and `strata.exe --version` answers `strata 0.1.48.1` (upstream added `--version` to the SYCL
engine). `positioned_file_test` passes. Of the 39 model-free parity tests in ctest, 37 pass:
`s2_expert_grouped_parity` fails as it did before (about 1.5e-7, 1-2 float32 ULP, deterministic, fp16 output
bit-identical), and `ple_parity` cannot open the Q2_0 fixture it names by default on this PC; pointed at the Coder
IQ1_M PLE shard it reports 0 failures. A request through `serve/server.py` on this build answered normally (32 tokens,
42.6 tok/s decode on a cold cache). The model was not re-benchmarked on 0.1.48.1.

**0.1.51 (0.2.6, 2026-10-25).** Rebuilt against 0.2.6: 234 targets, 0 errors, 51 executables, `strata.exe --version`
prints `strata 0.1.51`; `positioned_file_test` passes and 37 of 39 model-free parity tests pass (the two known ones:
the grouped-MoE difference, worst 7.91e-08 of sum|term|, and `ple_parity`'s missing Q2_0 fixture). The benchmark suite
against this 256K serving configuration (prefill 2048, 512 MiB reserve, all 12,288 experts resident) measured chat
decode 62.6 tok/s (59.3-94.2, n=9), an agent session of six turns 69.2 (58.3-84.0, n=15) with 69.2% of each turn's
prompt from the cache, and prompts of 4K / 16K / 32K at 1,086 / 1,192 / 1,175 tok/s. Long-context retrieval found the
code at 36.7K, 89.2K, 149.2K, 200.9K and 231.2K tokens (1,132 down to 822 tok/s). The full report is
`bench/results/2026-10-25-community-r-arc-tm-pro-b70-qwen3-8-flash-next-coder-iq1-m`.

**0.1.52 (0.2.7, 2026-10-27).** Rebuilt against 0.2.7: 241 targets, 0 errors, and ctest now lists 52 tests (upstream added thirteen host-only ones). 50 pass. The two failures are the ones this PC has always had: `ple_parity` cannot open the Q2_0 fixture it names, and `s2_expert_grouped_parity` differs from the previous kernels on the grouped path only, worst 1.06e-07 of sum|term|, with every cache-hit path bitwise identical. `positioned_file_test` passes. The merge needed two conflicts resolved: `sycl/CMakeLists.txt` keeps this fork's `positioned_file_test` and gains upstream's new tests, and `setup.py` keeps the Windows SYCL start branch. Upstream's start path probed CUDA before it reached the Intel branch, so `setup.py` now sets `found = []` on that path; `tools/test_setup_sycl.py` is 48 of 48 again.

The same 256K serving configuration as 0.1.51 (prefill 2048, 512 MiB reserve, all 12,288 experts resident), run twice on one server, measured prompt 1,094 tok/s at 4K and 1,198 at 16K (0.1.51: 1,072 and 1,172), decode 75.4 / 69.8 / 65.4 tok/s on the short / 4K / 16K prompts (73.7 / 68.2 / 63.9), and first token 0.58 / 3.78 / 13.72 s (0.59 / 3.86 / 14.03). Sanity checks 12 of 12, as before. That is about +2% on everything: the two runs agree within 0.2% on every median, and the 16K prompt rate is outside 0.1.51's range (1,196.5-1,198.0 and 1,197.3-1,199.7 against 1,170.3-1,173.2), so this is a real gain, small as it is. None of the 0.2.7 performance notes names Intel or SYCL and this port's own changes in the release are checkpoint and park handling, the PLE reader and the sampler, so the gain sits in shared code. Reports: `bench/results/2026-10-27-dev-256k-standard-v1` and `-b`.

**0.1.54 (0.2.8, 2026-10-30).** Rebuilt against 0.2.8: 245 targets, 0 errors, and ctest is 53 tests, 51 passing — the same two failures this PC has always had (`ple_parity`'s missing Q2_0 fixture and `s2_expert_grouped_parity`, worst 1.06e-07 of sum|term|), plus the new `pipeline_auto_test`, which passes. The merge was clean; `tools/test_setup_sycl.py` is 48 of 48.

0.2.8's headline numbers are for two Arc B65 cards (decode 71-74 tok/s against 60-63). On one B70 at 256K the decode is unchanged (75.6 / 70.0 / 65.5 on the short / 4K / 16K prompts, against 75.4 / 69.8 / 65.4 on 0.1.52), but **prompts came out slower**: 1,030 and 1,019 tok/s at 4K against 1,094, and 1,078 and 1,071 at 16K against 1,198 — 5.8% and 10.6% down, the two runs within 0.5% of each other, first token 15.3 s against 13.7. `STRATA_ATTN_BATCH=128` — the prompt attention batch is new in 0.2.8, budgeted and 32 by default — recovers about 2% (1,038 at 4K, 1,094 at 16K) with decode unchanged and all 12,288 experts still resident; 256 is no better (1,067 at 27K against 128's 1,073), so the serving config sets 128. The rest of the difference is not the batch and I did not trace it; this release's SYCL work is the verify window (verify.cpp +109, generate.cpp +196), the attention budget and the decode attention kernel. Sanity checks 12 of 12 in all three runs. Reports: `bench/results/2026-10-30-dev-256k-standard-v1`, `-b` and `-attn128`.

**Two settings that bite on a 24 GB card** (measured by an Arc Pro B60 owner, 2026-10-24). `--vram-reserve-mib` below
about 1024 moves the card's memory out from under the engine: decode fell from 36.3 to 3.76 tok/s at 512, and the
700 MiB default is already near the edge there. `--prefill 8192` helps only with `--prefill-borrow` and below about
32K context: at 8K it lifted a 16,000-token prompt to 852 tok/s (19.1 s to first token); at 64K the same pair gave
51 tok/s prefill and 7.9 tok/s decode, against 687 and 37.0 with `--prefill auto`.

**The expert profile tools default to 512 experts; the Coder has 256.** Pass `--n-expert 256` to
`make_profile_mix.py`: a profile traced on eight code prompts (`collect_profile_traces.py --mode routing`) measured
+12% on that B60 (38.35/36.92 to 42.04/42.04 tok/s) and removed the run-to-run spread.

**A build note.** A CMake shim that translates MSVC flag spellings into icpx's GNU dialect compiles and links, then
fails at the first prompt with `No kernel named ... native_q6_k_planar_adopt` (`native_mmvq.dp.cpp`) and
`iq_embed_rows` (`iq_kernels.dp.cpp`), in JIT and in AOT. The direct `icx` path in `build_windows.bat` emits them: a
Windows build is not one recipe, the flag dialect decides whether some kernels exist.

**Experts that do not fit: the arena, not the mirror (2026-10-24).** When a model's experts do not fit the card, setup
used the `--stream-experts` mirror on every xe card, so the experts the cache lacked were read from pinned host memory
over PCIe on most tokens. An Arc Pro B60 24 GB (Coder IQ1_M, 8K) measured 14.88 tok/s that way with 8% of drafts
accepted, against **36.29 tok/s and 62%** with the experts in ordinary RAM and the CPU computing the cache misses
(`STRATA_VERIFY_NO_HOST=0`, `--spec 2`): +88%. The same A/B on this B70 with IQ3_XXS (about 58% of its experts
resident) gave **40.30 tok/s with the mirror and 64.60 with the arena** (+60%), drafts accepted 86% to 100%, expert
cache hit rate 94-96% (one prompt, one warm-up, three requests, medians). Setup now chooses by fit on Windows
(B-series): `vram_fit.experts_fit_mib()` compares the room left for the experts with the model's `arena_gb`, and a
model that does not fit uses the arena and drafts 2 tokens per round instead of 4 (`--spec 4` cost 18% on that path on
the B60). The B70 with Coder IQ1_M at 256K leaves 26.5 GB for 23.4 GB of experts, so its config is byte-identical to
the previous generator (checked at 256K and 32K). Linux and the A-series keep their tested paths.

Token-for-token, the two paths are not the same, and the arena is not repeatable within one session. With the same
`--spec 2` on both (a native IQ pack refuses less) and greedy decoding: prompt A was byte-identical in all three
repeats on the mirror, identical across the paths in the first two, and the arena's third repeat diverged at character
174; prompts B and C gave the same answers with different wording (prose against a numbered list). That is the
near-tie rounding the engine already documents, amplified by the cache: the arena computes some experts on the CPU,
the mirror computes all of them on the GPU, and which experts the cache holds changes as a session goes on. The speed
is the trade: +60% on the B70 and +88% on the B60.

**The arena needs RAM, and it starts faster.** The engine locks the whole expert set in RAM: 40,927 MiB for IQ3_XXS
here (`expert arena: locked 40927 MiB`). On the same card the cache filled in 4.9 s with the arena against 43.0 s with
the mirror, which also builds an 18.29 GiB pinned copy in 37-40 s. Setup therefore takes the arena only when the PC has
the RAM for it (the model's `ram_gb`, or the arena plus 6 GB, whichever is larger) and otherwise keeps the mirror,
which is slower but needs less memory.

The estimate is close to the engine's own measurement: `experts_fit_mib()` leaves 24.7 GB for the experts on the B70
with IQ3_XXS, and the engine printed `expert cache auto: 25.68 GiB free`; on the B60 report it leaves 17.3 GB against
their 19.00 GiB. `--ple-io` stays at its default (`direct`) here: the engine's probe measured 62,991-66,681 rows/s
("fine"; the warning bar is 15,000), and on Windows the engine does not switch to mapped reads by itself - a slow probe
warns instead. `mmap`/`ram` are for a slow drive, not for the expert path.

What has **not** been run on Windows: pictures, the Arc A-series (Windows uses Level Zero for every Arc, `i915` is a
Linux driver), two cards, and the Docker path. WSL2 with an Arc has been used by one tester (the B580 row above), but
setup cannot detect the card there, because it reads `/sys/class/drm`, which WSL2 does not have.

Pictures are therefore off by default on Windows: setup passes `--vision none` unless `--vision yes` is given. (0.2.3
turns them on by default on Linux, where the A750 has run them; this port has not run one yet.)

## Reporting a problem

Open an issue with the card, the driver version, the `sycl-ls` output, the oneAPI version, the model and flags, and the engine's
stderr. Results from real cards are what move more cards from tested to validated.

## History / older measurements

The first port (0.1.39) compiled with 0 errors on Ubuntu 24.04 (WSL2) with oneAPI DPC++ 2026.1.1 and oneMKL 2026.1: the
`strata` engine plus 157 targets, as SPIR-V (JIT). Kernel parity tests on a CPU (`ONEAPI_DEVICE_SELECTOR=opencl:cpu`, Ryzen 5
7600) passed 14 of 25, the same set as the 0.1.38 port; the failures were tight float tolerances, missing model fixtures and two
CPU timeouts. That was a compile-and-compute check, not a test of an Arc. The CUDA engine's greedy output stayed byte-identical.

The two B60 rows ran commit `6f32ec0` plus two small `sycl/` fixes, AOT `bmg-g21`, Ubuntu 24.04, `xe`, Level Zero V2, NEO
26.09.37435.12 and oneAPI 2026.1.1, without Docker. No one ran an unpatched 0.1.39 port on an Arc; the other rows were measured on
earlier versions.
