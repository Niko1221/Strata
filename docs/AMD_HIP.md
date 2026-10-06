# AMD Radeon: the HIP backend (gfx1100, gfx1101, gfx1200, gfx1201, gfx1030)

Strata runs on AMD Radeon cards through its HIP backend, the same engine as on NVIDIA compiled for AMD. This page
covers the build on Linux (on Windows a ready-made engine, see [Windows](#windows)) for the RX 7900 XT / XTX (RDNA3, gfx1100) and the
RX 9070 / 9070 XT / Radeon AI PRO R9700 (RDNA4, gfx1201; see [RDNA4](#rdna4-gfx1201)). The RX 7800 XT / 7700 XT
(gfx1101) and the RX 9060 XT (gfx1200) were validated by their owners (see [Community-validated
cards](#community-validated-cards)); the RX 6800 / 6900 series (RDNA2, gfx1030) builds and runs too, reported by a community machine and not yet validated by the maintainers (see [RDNA2](#rdna2-gfx1030)). Setup chooses it by itself on a PC with no NVIDIA card Strata can use (`--backend hip` on a PC with both); the
install steps for users are in [INSTALL.md](INSTALL.md#amd-cards). gfx906 (Instinct MI50 / MI60, Radeon VII; wave64) has a separate
opt-in build, see [gfx906](#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source). The Ryzen AI Max "Strix Halo" APU (gfx1151, RDNA3.5, unified memory) is built from source on Linux: see [STRIX_HALO.md](STRIX_HALO.md). Other AMD architectures and mixed
AMD/NVIDIA execution in one run are not supported.

The backend maps the CUDA-shaped runtime and BLAS calls to HIP/hipBLAS, uses
RDNA2/RDNA3/RDNA4's signed integer dot instruction for quantized kernels, and supplies
wave32 shuffle/packed-byte operations. CUDA-only QSA matrix instructions have
an ordered FP32 fallback. Prefill supports both dequantization plus hipBLAS GEMM and opt-in HIP ggml MMQ.
An optional, calibrated hipBLASLt path accelerates dense projections (per-architecture tables in `tools/hip`).
This does not claim bit-identical model answers across backends. See
[performance settings and evidence](AMD_HIP_PERFORMANCE.md).

## Install with setup (recommended)

On Linux with an RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT or Radeon AI PRO R9700 and
the kernel's amdgpu driver (no ROCm install needed):

```sh
./setup.sh --backend hip
```

- **Detection:** setup finds the card through the kernel's KFD topology. Integrated Radeon GPUs are listed as not
  supported. On a PC without an NVIDIA card Strata can use, `--backend hip` is chosen automatically.
- **ROCm:** a system ROCm 7 in `/opt/rocm` (or `$ROCM_PATH`) with hipcc and hipBLAS is used when present. Otherwise
  (or when it is older than 7.0) ROCm is installed into `.venv` from AMD's TheRock wheels (~10 GB, no sudo), pinned
  to the version this backend was tested with, from the card family's index: `gfx110X-dgpu` for gfx1100 / gfx1101,
  `gfx120X-all` for gfx1200 / gfx1201, `gfx103X-all` for gfx1030 (`STRATA_ROCM_VERSION` /
  `STRATA_ROCM_INDEX` override them; the gfx1030 index is not checked to carry the pinned version: a system
  ROCm 7 is the tested path there).
- **Engine:** compiled on your PC for the card's architecture (10-20 minutes, once; again after a `git pull` that
  changes it, or when you pick a card of another architecture). This needs a C++ compiler and git
  (`sudo apt install build-essential git`).
- **hipBLASLt tuning table:** setup uses `tools/hip/<arch>-hipblaslt-<version>.txt` only when it matches both the
  card's architecture and the installed hipBLASLt version (read from `hipblaslt-version.h`; 1.2.0 is `100200`).
  Otherwise it says so and the prompt's dense matrix products use plain hipBLAS (slower prompts, same answers).
  A table's solution ids are valid only for that pair, and the engine refuses any other table.
- **Several cards:** setup takes one card (the one with the most VRAM, or `--gpu N`) unless you name more:
  `./setup.sh --backend hip --gpus 1,0` splits the model's layers across them, the first one the main card (numbers
  as setup lists them; `--gpus all` = every supported card, the most VRAM first). Every chosen card must be one of the
  architectures above; the engine is compiled for each of them (cards of two families, e.g. gfx1100 + gfx1201, need
  a system ROCm 7: AMD's wheels hold one family). A split pays only when no single card holds the model's experts
  (see RDNA4 below).
- **Limits for now:** images only through the CPU encoder (`--vision cpu`, 0.1.32). Setup does not offer the tuning
  (calibration) on AMD yet: its controls are being checked on HIP one at a time (#566). Since 0.1.39 a tuning run by
  hand (`./setup.sh --calibrate`) is saved for the AMD card it ran on and reused when setup runs again. The Monitor
  shows the card's load, VRAM, temperature and power from Linux sysfs (0.1.32).

The rest of setup is the same as on NVIDIA: the model download, the start script, the server.

## Windows

Since 0.1.34 an AMD card on Windows is set up like an NVIDIA one: download Strata, double-click `START-HERE.bat`.
On a PC with no NVIDIA card Strata can use, the AMD card is chosen by itself; with both, setup asks
(`START-HERE.bat --backend hip` picks AMD directly).

- **You need:** Windows 10 or 11 (64-bit), one of the cards above, and a current AMD driver ([AMD Software:
  Adrenalin Edition](https://www.amd.com/en/support/download/drivers.html)). Nothing else: no ROCm or HIP SDK
  install, no compiler, no admin rights.
- **Detection:** setup reads the display adapters Windows lists (their PCI ids; the VRAM size from the display
  driver's registry entry). An integrated Radeon is listed as not supported, except the Ryzen AI Max "Strix Halo" (Radeon 8060S / 8050S / 8040S, gfx1151: [STRIX_HALO.md](STRIX_HALO.md)); a Strix Point (890M / 880M, gfx1150), Krackan (860M / 840M, gfx1152) or Phoenix / Hawk Point (780M / 760M, gfx1103) Radeon is named as what it is and not supported.
- **Engine:** the ready-made `strata-windows-x64-hip.zip` from the release (built by `tools\hip\build_windows.bat`
  for gfx1100, gfx1101, gfx1102, gfx1200, gfx1201, gfx1030 and, from 0.1.40, gfx1151 (Strix Halo, unvalidated on Windows)) goes into `engine\`. It carries the ROCm libraries the
  engine loads (`engine\rocm\bin`: the HIP runtime, hipBLAS / rocBLAS / hipBLASLt with their kernels for these cards,
  amd_comgr and the Microsoft C++ runtime; ROCm 10.2.0a20260930 from AMD's TheRock builds, licenses in
  `engine\rocm\licenses`). The HIP runtime works through the AMD driver's own components, so the driver is the one
  thing it needs from the PC.
- **The HIP runtime next to `strata.exe` (0.1.35, #468 #461):** `amdhip64_7.dll` and `amd_comgr.dll` are also put in
  `engine\` (setup copies them there on every start). Windows looks in the program's folder before System32, where
  some AMD drivers install their own `amdhip64_7.dll`; with that one, the bundled libraries crashed on the first
  prompt (an access violation, or `hipErrorInvalidDeviceFunction`). The engine's log names the runtime it loaded
  (`strata generate: HIP runtime ...`).
- **Before the ~60 GB model download** setup runs `engine\strata-device.exe --list-devices` (with `engine\rocm\bin` on
  the PATH): if the HIP runtime does not see the card, setup stops there and points to the driver. It also gives the
  card's HIP number: with an integrated Radeon that is device 1, not 0 (#325). From then on setup lists the AMD cards
  as HIP numbers them, so `--gpu N` and the config's `"gpu"` are HIP numbers.
- **Differences from Windows-on-NVIDIA and Linux-on-AMD:** the stock installer does not install an image encoder;
  a HIP encoder can be built separately ([Windows image encoder](#windows-image-encoder)). One card per model
  (`--gpus` is Linux-only for now), no calibration.
- Two Windows-only engine details (#247, #325): hipBLAS can return success and still leave `hipErrorInvalidValue`
  set after some BF16/FP16 GEMMs (seen on gfx1201); the engine clears that one stale error after a GEMM that
  succeeded, on Windows only. `hipHostGetDevicePointer` returns the host pointer itself on Windows: kernels read
  mapped memory through it correctly, but a device-to-device copy into it does not land, which is why
  `tests/hip/handoff` times out there (the engine does not use that copy; `tests/hip/mapped_alias` reports it).
- Two more (#380, #377, by BlueKingMuch, measured on an RX 6800 that drives the desktop): `hipMemGetInfo` on Windows
  does not subtract what the desktop and other programs hold on the card, so `--expert-cache auto` filled the card
  past what Windows keeps in VRAM and decode fell from 41 to 30 tok/s. The engine now lowers that free figure by
  what Windows' video memory budget for the process withholds (logged once: `strata: Windows budgets N of this
  card's M MiB ...`; `STRATA_WDDM_BUDGET=0` turns it off). And the PCIe probe times its copies on the host clock
  there, since HIP's events read impossible speeds (3,300-26,000 GB/s), so a slow link now gets a smaller
  `pcie_frac` as on NVIDIA.

**What is validated (0.1.34):** #325's author ran the engine of this port on an RX 9070 XT (Windows 11, ROCm
10.2.0a20260930 in `.venv`, compiled on the PC): Coder IQ1_M at 32K, 29.2 tok/s decode, ~181 tok/s prefill,
correct answers; ctest 42 of 46. The maintainers have no Windows AMD card: the release zip was built on an NVIDIA PC,
and checked there on the Ryzen CPU's integrated Radeon (gfx1036, a test build of the same tree): with only the zip's
libraries on the PATH, `strata-device` lists the card and a hipBLAS BF16 GEMM matches the CPU; the HIP ctest passes
52 of 56, the 4 failures the same as on the RX 9070 XT (`hip_handoff`, and three tests that need a pack fixture).
The ready-made zip itself has not run a model on a discrete card yet - please report.

**Reporting a Windows AMD run** (an issue, or on #325): your card and driver version (AMD Software > System), then

```bat
engine\strata-device.exe --list-devices
engine\strata-device.exe --selftest
```

(from the Strata folder, after `set PATH=%CD%\engine\rocm\bin;%PATH%`), the end of `strata-<model>.log`, and the
speed lines the server window prints for a first answer.

**Building it yourself:** `tools\hip\build_windows.bat` (Visual Studio 2022 Build Tools with the C++ workload, Python,
git; no admin, no AMD GPU) installs ROCm from AMD's TheRock wheels into `.rocm-win`, builds, and packages
`dist\strata-windows-x64-hip.zip`; `START-HERE.bat --backend hip --prebuilt dist\` installs that one.
`tools\hip\build_windows.bat tests` also builds the HIP tests (`ctest` in `build-hip-win`, with
`.rocm-win\Lib\site-packages\_rocm_sdk_devel\bin` on the PATH). `STRATA_HIP_ARCHS` picks other architectures.

### Windows image encoder

`tools/vision` can build llama.cpp's HIP image encoder with `STRATA_VISION_HIP=ON`. CUDA and HIP are mutually
exclusive build options; both off gives a CPU encoder. The main Strata engine need not be rebuilt.

From a command prompt, with Visual Studio C++ Build Tools, Python, CMake and Ninja installed:

```bat
set STRATA_HIP_ARCHS=gfx1030
set LLAMA_DIR=C:\path\to\llama.cpp
tools\vision\build_windows_hip.bat
```

Use the card's real architecture and the `LLAMA_CPP_COMMIT` from `setup.py` for the llama.cpp checkout
(`3cf03257f219afbe7334045ff7c6a06ac68c627d` in 0.1.40). The script installs ROCm into `.rocm-win`, uses ROCm's
Clang with the MSVC headers/linker, and builds `build-vision-hip\bin\strata-vision.exe`. It leaves the running
installation alone. `BUILD_JOBS` defaults to 8; `BUILD_DIR`, `ROCM_VENV`, `STRATA_ROCM_VERSION` and
`STRATA_ROCM_INDEX` may be overridden. Use the ROCm version in the installed `engine\BUILD.json`; the default
is 0.1.40's `10.2.0a20260930`. The pinned ggml Windows backend uses `GPU_TARGETS`, not CMake's HIP-language
architecture flag. HIP virtual-memory allocation is disabled in this build.

Stop the server, back up its config and encoder, and copy the new executable to `engine\strata-vision.exe`.
Keep the matching `amdhip64_7.dll` and `amd_comgr.dll` beside it and `engine\rocm\bin` in the config's `lib_dirs`.
Use the text model's matching mmproj, retain `--vision` in the engine arguments, and set `vision.gpu` to `true`
in the model config. An existing CPU setup can retain its `max_tokens` cap initially; a larger cap reserves more
GPU memory. Record the local encoder separately from the official engine in `engine\BUILD.json`.

Restart using the model's launcher. The encoder starts and warms its largest image before the engine chooses
its expert cache size. Its log must identify `GPU backend ROCm`, the device and card, followed by a successful
warm-up; `--gpu` now fails if a GPU cannot initialize instead of silently using the CPU. Then send a real image
request through the API: `/health` reporting `images: true` alone does not prove GPU execution. Without `--gpu`,
the same executable uses the CPU and masks CUDA/HIP devices before backend initialization.

This is a separate source-build path. The 0.1.40 release archive and setup's vision choices have not been
changed; an engine update can require rebuilding/reinstalling this encoder. Other cards and ROCm versions
need their own runtime check.

[Local RX 6900 XT validation](../bench/results/2026-10-07-windows-hip-vision-rx6900xt/README.md): Windows 11,
ROCm `10.2.0a20260930`, Strata 0.1.40 and Huihui Swift IQ3_XXS. HIP initialization, CPU mode, failure with no
visible GPU, and two distinct image requests through Chat Completions and Responses passed. This used a
300-image-token cap and checked basic shapes and large text, not fine-detail OCR or grounding accuracy.

## Build

Requirements: a working ROCm driver/runtime, HIP development headers and
compiler, hipBLAS and (for tuned dense prefill) hipBLASLt development files, CMake 3.24+, a C++20 host compiler, and Git.
The model still needs sufficient system RAM and fast SSD storage for its PLE
table. VRAM occupancy alone is not a throughput measurement.

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2
```

`CMAKE_HIP_ARCHITECTURES` is `gfx1100`, `gfx1101`, `gfx1200`, `gfx1201`, or a list such as `"gfx1100;gfx1201"`
(one binary for both). gfx1102 (the same wave32, 64 KiB LDS and dot4 instruction) builds with a warning: it passed
ctest (#192) but no model run has been reported; so does gfx1030 (RDNA2: the older `v_dot4_i32_i8`, a community run in #311). At startup the engine and `strata-device` compare each GPU they use
(`gcnArchName` up to the `:` feature suffix) with the architectures the binary was compiled for, and require
wave32. A binary carried to another card stops with the card's name, its architecture and the build's list,
instead of failing later with "invalid device function".

If CMake cannot find the HIP compiler, add
`-DCMAKE_HIP_COMPILER=/path/to/rocm/llvm/bin/clang++`. On the Fedora-family test
host this was `/usr/lib64/rocm/llvm/bin/clang++`. Use the compiler and libraries
from the same ROCm installation. Do not enable both GPU backends in one build.

Native IQ experts are enabled by default. CMake fetches the llama.cpp revision
pinned by this repository. For an offline build, point `STRATA_GGML_DIR` at a
checkout of that exact revision; using an arbitrary newer checkout changes the
dependency being tested.

## Model and serving configuration

Prepare a supported model pack and its MTP runtime using the existing tools.
For OrcaRouter IQ3_XXS, follow [the explicit compatibility conversion](ORCA.md);
this backend does not change quantization, model licenses, or tokenizers.
Add `--experts-bin` to the `tools/iq_pack.py` command: mmap requires the pack's
`experts.bin`, which the default native packing command does not emit. This
consumes additional disk space. Original GSQ-RCO IQ3_XXS uses shard 2 for PLE;
Orca uses shard 1. Keep each model's own pack and tokenizer together.

Use `build-hip/strata` as the executable in the server JSON. Select the AMD
device with `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` if necessary. Start with
a modest context and prefill chunk size before measuring larger workloads.
Keep the PLE file on an SSD. Do not assume a configured context length proves
successful full-window inference.

Large pinned host allocations can fail on ROCm even when ordinary RAM is
available. The existing `--mmap-experts` path avoids allocating the full pinned
expert arena; it still depends on OS file-cache residency and may stall on
storage reads. It does not make SSD access equivalent to RAM.

Starting args for the original GSQ-RCO IQ3_XXS model (replace the paths):

```sh
build-hip/strata --serve \
  --pack packs/iq3xxs \
  --native /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache auto \
  --prefill 512 --spec 4 --spec-min-p 0.5 --mtp mtp/rt \
  --max-context 4096 --kv int8 --pool-workers 15 \
  --adapt-every 0 --pcie-frac 0 --vram-reserve-mib 1024
```

`--serve` is the engine's internal token protocol. For a browser or OpenAI API,
put these args in the `args` array of the [server JSON example](ORCA.md#local-server),
set `exe` to `build-hip/strata`, and use the matching pack's `tokenizer` directory.
Launch with `.venv/bin/python -m serve.server --engine strata --config strata-hip.json --port 8080`.
The worker count above was used on a 16-core CPU; measure it for your CPU.
The 4K context is a smoke-test starting point, not a model limit. The expert cache
sizes itself automatically and leaves 1 GiB of VRAM headroom.

**The card also drives a Linux desktop (#560 #516):** keep more VRAM free than the default 700 MiB, e.g.
`./setup.sh --vram-reserve-mib 3072`. With the cache filling the card, the desktop's next VRAM need makes amdgpu move
GPU memory to system RAM (GTT), the OOM killer then ends KWin/plasmashell or `systemd-oomd` ends apps, or the
compositor fails with "Failed to pin framebuffer with error -12".

The installer supports this backend (see "Install with setup" above). Images run through the CPU encoder for now (`--vision cpu`).
Setup installs one AMD card, or several with `--gpus` (the engine's layer split; see RDNA4 below).

**Opt-in switches for RDNA2 prompts and hangs (0.1.40, off by default):** `STRATA_DENSE_MMQ=1` (#820, HIP) runs the
prompt's dense GGUF projections through the int8 MMQ kernels instead of dequantize + hipBLAS: on the RX 6900 XT a 4K
prompt read 436 -> 757 tok/s (greedy text diverges from the default after a few dozen tokens: the q8_1 activation
rounding). `STRATA_HIP_ADAPT_KERNEL_COPY=1` (#884) copies the adaptive tier's swaps with a kernel instead of the SDMA
engine, a workaround for the gfx1030 hang seen with the MMQ prompt path and adaptive swaps (untested on the reporter's
machine). Windows HIP: the doorbell kernels fence their store (#697), and the shared-expert fork is off on HIP (#816).

## RDNA4 (gfx1201)

The RX 9070 / 9070 XT and the Radeon AI PRO R9700 run the same kernels as gfx1100: wave32, 64 KiB of LDS per
workgroup, the signed dot4 instruction (`v_dot4_i32_iu8` through `__builtin_amdgcn_sudot4`) and a 100 MHz wall
clock. The first report and patch came from doplxyz (#178). Validated on 2026-09-30 on doplxyz's test machine:
an RX 9070 XT 16 GB and a Radeon AI PRO R9700 32 GB (both gfx1201), a Ryzen 9 3900X (16 threads, no AVX-512),
47 GB RAM, Ubuntu 24.04 in a KVM/VFIO guest.

- **Build:** complete HIP build with tests (`-DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_BUILD_TESTS=ON
  -DSTRATA_PREFILL_MMQ=ON`), with the system ROCm 7.14 (hipBLASLt 1.4.1) and with setup's own path: TheRock
  7.10.0a20251120 wheels from `gfx120X-all` (hipBLASLt 1.2.0) and `build_engine_hip`.
- **ctest** (all 45 registered tests, R9700): 42 pass, `hip_prefill_hipblaslt_gemm` skips (no table), and two
  fail for reasons outside the GPU: `ple_parity` needs the Q2_0 model fixture, `expert_multi_test` needs an
  AVX-512 CPU. `hip_handoff` needed the volatile ring store: on gfx1201 a plain store to mapped pinned memory
  stays in the GPU's L2 until the stream is synchronized.
- **Arch check:** a gfx1100-only build stops on the gfx1201 card at startup with the message above (engine and
  `strata-device`).
- **End to end** (Coder IQ1_M, setup's arguments: `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5`,
  MTP, `--kv int8`, `--max-context 32768`; 128 greedy tokens, system ROCm 7.14, no hipBLASLt table). The answers
  are coherent and the same on both cards. The live server smoke (`serve/server.py`: web page, models, props,
  chat, prefix reuse, streaming, the Anthropic endpoint, a sampled code answer) passed 9/9 on each card.

  | card | expert cache | peak VRAM | 4K prompt | 4K decode | 16K prompt | 16K decode |
  |---|---|---|---|---|---|---|
  | Radeon AI PRO R9700 32 GB | 12,288 slots, 23.4 GiB | 29.7 GiB | 982 tok/s | 45.5 tok/s | 1,402 tok/s | 48.3 tok/s |
  | RX 9070 XT 16 GB | 4,931 slots, 9.4 GiB | 15.6 GiB | 782 tok/s | 30.8 tok/s | 1,235 tok/s | 35.4 tok/s |
  | R9700, TheRock 7.10 wheels | 12,288 slots | | 957 tok/s | 44.2 tok/s | 1,284 tok/s | 43.9 tok/s |

  The engine's resident memory was about 26 GB in every run. Since 0.1.31 `__byte_perm` is one `v_perm_b32` and the
  packed byte subtracts/compare work on four lanes at once (#262, ttio2tech): decode +15% on the R9700 (46.0 -> 53.0
  tok/s on a 4K prompt, 52.0 -> 60.5 warm) and +5-7% on the 9070 XT, prompts unchanged, the same tokens.
- **hipBLASLt:** the validation above used hipBLASLt 1.4.1. A table calibrated on the R9700 at the engine's shapes
  with that version (0.98-1.76x per GEMM over hipBLAS) changed the end-to-end prompt speed by 0-3%, within noise,
  so none was shipped for it: there the plain hipBLAS path is already close. For hipBLASLt 1.5.0 (ROCm
  10.2.0a nightly) `tools/hip/gfx1201-hipblaslt-100500.txt` is shipped (see "Tuning table" below); on one R9700 it
  measured +3.9% prompt speed on 4,210-token prompts (1,590 vs 1,531 tok/s), a modest gain.
  With the hipBLASLt 1.2.2 of a system ROCm 7.2.4 the plain path is far off, and `tools/hip/gfx1201-hipblaslt-100202.txt`
  is shipped for it (24 of the gfx1100 table's 26 shapes; setup uses it only with that exact version): on an R9700
  with the full IQ3_XXS, `--kv int8 --kv-resident 32768`, a fresh 32K prompt read at 638 -> 1,177 tok/s and a 7K one
  at 656 -> 1,164 tok/s with it, decode unchanged.  `hip_prefill_hipblaslt_gemm` passes with it.
- **Both cards in one run (layer split, engine 0.1.30):** the config's `"backend": "hip", "gpu": [1, 0]` (R9700
  first) runs through `serve/server.py` (setup writes it with `--gpus 1,0` since 0.1.31). Auto split put layers 0-27 on the R9700 and
  28-47 on the 9070 XT. With every expert on the GPUs the split gives exactly the tokens of the R9700 alone (4K and
  16K prompts); checkpoints on the split (second turn, rewind, a prompt sharing a prefix, a cancelled prompt
  retried) give exactly the tokens of a fresh read. The conversation cache refuses a split at start (exit 2).
  On this pair the split does not pay: the R9700 alone already holds all 12,288 expert pairs.

  | run (the same session, warm) | 4K prompt | 16K prompt | decode |
  |---|---|---|---|
  | R9700 alone | 1,794 tok/s | 1,804 tok/s | 51-52 tok/s |
  | R9700 + 9070 XT, auto split | 1,384 tok/s | 1,906 tok/s | 42-43 tok/s |
  | RX 9070 XT alone | 1,016 tok/s | 1,519 tok/s | 38-40 tok/s |

  A split is worth it when no single card holds the model's experts. Since 0.1.31 a split pins the whole expert
  arena on Linux (the 8 GiB cap is for Windows/WSL2 only, #253): with an expert cache of 1,500 on the R9700 (most
  experts streamed) the split read a 16K prompt at 1,803 tok/s instead of 1,287, with the same tokens in 5 + 5
  starts. `"split_skip_if_fits": true` in the config (0.1.31, opt-in) runs such a pair on the first card alone when it
  holds every profiled expert: with the R9700 first, 4K prompts 1,776 tok/s (split: 1,244) and decode ~60 tok/s
  (split: ~51), the tokens of the R9700 alone (docs/MULTI_GPU.md).
- **Speed switches (engine 0.1.32, measured on the R9700 / 9070 XT with the Coder IQ1_M pack):**
  - the MoE router (`router_top10`) runs a HIP kernel without its serial FP64 sum and block barriers by default: the
    same ids and weights bit for bit (`hip_router_fast` checks 65,536 rows), 39 -> 9-12 us per call, the same greedy
    tokens (5 + 5 starts). Decode: 62.4 -> 70.0 tok/s on the R9700 and +4% on the 9070 XT in one A/B here (ROCm 10.2
    nightly); a user's repeated A/B/A/B with setup's TheRock 7.10 wheels and an i5-12600K measured +1-4%, inside a
    12-20% run-to-run spread (#432) - how much it gains depends on how much of decode the router is on that setup.
    `STRATA_HIP_ROUTER_OLD=1` runs the portable kernel.
  - `STRATA_HIP_WMMA=1` (opt-in, gfx12 only, int8 KV): the prompt path's QSA attention on RDNA4 matrix cores
    (`v_wmma_f32_16x16x16_f16`, FP16 hi + lo halves like the CUDA tensor-core kernel; `hip_prompt_attn_wmma` bounds it
    against the FP32 kernel and FP64). 7.2-7.5x the portable kernel; R9700 prompts 4K 1,784 -> 2,427 tok/s, 16K
    1,797 -> 2,700 (a user with TheRock 7.10: +29-34%, #432). Not bitwise: greedy text differs from token ~50 on, as with the CUDA tensor-core attention. With it
    the prompt path's expert ring is 96 slots (as STRATA_PREFILL_RING=96): with the default 384 the 9070 XT's 4K
    prompts fell to 718 tok/s; with 96 they gain (1,017 -> 1,211; 16K 1,518 -> 2,032). PR #329
    (bsorensen110) contributed an equivalent gfx12 WMMA kernel of the same speed (within 1%); this one also masks KV
    pages that KV streaming has not made resident, as the decode kernel does.
  - The prompt path's QSA top-k picks its kernel by the blocks a query actually has, not the cache's capacity (#337,
    bsorensen110): the same ids, on by default with AMD (NVIDIA keeps its capacity rule: there the 64K prompts read
    1-3% slower with it). `STRATA_SELECT_WMMA=1` (opt-in, gfx12) adds #337's
    matrix-core block scorer; it selects slightly differently (254 of 256 queries the same) and gained +1.5% on 16K
    prompts at a 262K context on the R9700.
- **Known:** rarely (about 1 start in 10) a HIP run's greedy output differs from another start's at some token, on
  one card or two and on engine 0.1.29 as well; not yet explained.
- **Not validated:** images, long contexts beyond 16K, answer-quality benchmarks.

## Community-validated cards

Run by their owners, not on the maintainers' machines; setup accepts them like gfx1100 / gfx1201. setup ships a
hipBLASLt table for gfx1200 (the numbers below); gfx1101 has none (make one with [Tuning table](#tuning-table)
and compare the prompt speed with and without it).

- **gfx1101, RX 7800 XT 16 GB** (jhohertz, #254; engine 0.1.29, Ryzen 9 5950X, 121 GiB RAM, system ROCm with
  hipBLASLt 1.4.1): `./setup.sh --backend hip` detected the card and compiled the engine; `strata-device --selftest`
  passed; ctest 30/32 (`ple_parity` needs the Q2_0 fixture, `platform_memory_test` the memlock limit). Coder IQ1_M,
  64K context, MTP, with a table the owner calibrated: fresh prompts of 4K-9K tokens at 898-953 tok/s, decode
  38-44 tok/s (128 tokens).
- **gfx1200, RX 9060 XT 16 GB** (Efeisot, #256 after #176; engine 0.1.29, Ryzen 9 7950X, 64 GiB RAM, ROCm 7.2 with
  hipBLASLt 1.2.2): ctest 32/32 (without `ple_parity` and `platform_memory_test`). Coder IQ1_M, greedy, MTP
  `--spec 4 --spec-min-p 0.5`, with a table the owner calibrated:

  | prompt | prompt speed | decode |
  |---|---|---|
  | 2,374 tokens | 540 tok/s | 27.1 tok/s (0.69 draft acceptance; 31.0 at 1.00) |
  | 65K (`--kv int8 --kv-resident 65536 --prefill 16384`) | 748-753 tok/s | 26-40 tok/s by acceptance |
  | 130K (the same flags) | 725 tok/s | - |

  Greedy output was the same across runs. For comparison, llama.cpp's HIP build measured 20 tok/s decode and
  450 tok/s prompt on that card.

  Since 0.1.38 setup ships that table (`tools/hip/gfx1200-hipblaslt-100202.txt`). On engine 0.1.31, one binary (ctest
  37/37, without `ple_parity` and `platform_memory_test`), prompts of repeated code blocks, prefill 2560 for
  the short prompt and 16384 with `--kv int8 --kv-resident 65536` otherwise, two runs each - prompt speed
  with the table vs plain hipBLAS (no table):

  | prompt | with the table | plain hipBLAS | gain |
  |---|---|---|---|
  | 2,374 tokens | 597-599 tok/s | 396-398 tok/s | 1.50x |
  | 65,045 tokens | 754-757 tok/s | 380 tok/s | 1.99x |
  | 130,091 tokens (the same flags) | 719 tok/s | 363-370 tok/s | 1.98x |

## RDNA2 (gfx1030)

The RX 6800 / 6800 XT / 6900 XT / 6950 XT run the same kernels: wave32 and 64 KiB of LDS per workgroup. The one
difference is the dot instruction: RDNA2 has no `v_dot4_i32_iu8` (gfx11 and newer), so
`dp4a` uses the plain signed `v_dot4_i32_i8` through `__builtin_amdgcn_sdot4`, which compiles to a single
`v_dot4c_i32_i8` (same signed x signed byte products, modulo 2^32). There is no WMMA; the QSA scorer takes the
same ordered FP32 fallback as gfx1100. CMake lists gfx1030 as unvalidated (the build warns) until a maintainer has
run it; the report below is from a community machine: an RX 6900 XT 16 GB (gfx1030), an i7-13700KF (8 P-cores and
8 E-cores, AVX2, no AVX-512), 63 GB RAM, NixOS, ROCm 7.2.3 from nixpkgs (clang 22, hipBLAS 3.2).

- **Build and tests** (engine 0.1.26): a complete HIP build for gfx1030, made by hand with cmake and ROCm's own
  `clang++` (the nixpkgs ROCm is not an `/opt/rocm` tree, so setup's `build_engine_hip` was not exercised; the
  binary was placed in `engine/` for setup to use). All 28 registered ctest tests pass on the card, including
  `hip_device_selftest`. An engine 0.1.30 build of this branch configures, builds and runs clean on the same card;
  the speeds below are measured on it.
- **End to end** (Swift 1.5 IQ3_XXS, `--context 131072` with `--kv-resident 32768 --adapt-every 1
  --vram-reserve-mib 1024`, 200 greedy tokens): 38-42 tok/s decode with the default 15 CPU pool workers (one per
  physical core except the host thread; 8 workers: 36; 24 = every logical core: 27), consistent even with the
  131,072-token context full.
- **Prefill** (the same 15-worker configuration): 246 tok/s on a 2,000-token prompt, 330-339 tok/s at the auto
  8,192-token chunk (7,997 and 15,967 tokens; time to first token 8.9 and 48.3 s), 133 tok/s on a 522-token
  prompt - short prompts are fixed overhead (19-25 tok/s on 34 tokens). Decode after a 16K prefill holds at
  45.5 tok/s.
- **16 GB card:** the expert cache holds 6,310 slots (10.2 GiB) at 16K context and 5,247 slots at 131,072, where
  setup keeps the KV cache in VRAM because IQ3_XXS needs about 60 GB of RAM plus the cache to stream it. The
  `--pcie-frac 0` and `--adapt-every 0` of the 7900 XTX configuration in
  [AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md) cost 8.6 and 13.6 tok/s here (30 with the defaults): keep the
  defaults on a 16 GB card.
- **hipBLASLt:** ROCm's hipBLASLt ships no gfx1030 kernels, so there is no table and the plain hipBLAS path runs.
- **Prompt GEMMs in FP16 (#835):** rocBLAS on gfx1030 has tuned kernels for FP16-in / FP16-out GEMMs only; the prompt
  path's FP16-in / FP32-out and BF16 products ran generic kernels about 6.6x slower (5.6 and ~5.3 TFLOPS against 37.7 at
  N = 10240, T = 7313, K = 2560). `STRATA_HIP_PROMPT_F16=1` (opt-in, off by default; the engine prints a tip on gfx103x) runs the 16-bit prompt GEMMs FP16 in and out;
  unset is the old path. `STRATA_DBG_NAN=1` also counts the non-finite FP16 outputs (a sum past 65504). The output is not bit-identical to it (an FP16 rounding of each GEMM's output replaces the BF16 rounding of
  the BF16 GEMMs' activations); what the distribution check showed and did not show is in
  [bench/results/2026-10-04-rdna2-fp16-prompt](../bench/results/2026-10-04-rdna2-fp16-prompt/README.md). Measured on a
  second community machine - 2x RX 6900 XT 16 GB (one card for these numbers), Ryzen 5 5600X (6 cores, AVX2), 128 GB,
  Ubuntu 26.04, ROCm 10.0.0, engine 0.1.39 built by hand with `-DSTRATA_PREFILL_MMQ=ON` as setup does, GSQ-RCO IQ3_S:
  a 9.4K / 34.7K / 105.8K-token prompt reads at 744 / 915 / 926 tok/s instead of 439 / 466 / 461; decode is unchanged.
  On that machine the HIP ctest passes 60 of 65 with and without this change: 2 skipped (`hip_prompt_attn_wmma`, no
  matrix cores; `hip_prefill_hipblaslt_gemm`, no hipBLASLt table) and 3 that fail for reasons outside the engine
  (`ple_parity` needs a Q2_0 PLE file that is not on that machine, `expert_multi_test` refuses the CPU without AVX-512,
  `platform_memory_test` cannot `mlock` at the shell's default `ulimit -l`).
- **gfx1031** (RX 6700 XT, #524): setup knows it (the `gfx103X-all` wheels, unvalidated); its reporter runs it daily
  on one card.
- **Not validated:** gfx1032 (the same `dp4a` path, no hardware report), setup's own build path and the
  `gfx103X-all` wheels on gfx1030, images, answer-quality benchmarks. RDNA1 (gfx1012, RX 5500 XT) builds by hand:
  [OLDER_GPUS.md](OLDER_GPUS.md#amd-building-gfx906-and-gfx1012).

## gfx906 (Instinct MI50 / MI60, Radeon VII): wave64, built from source

gfx906 is wave64 and has no WMMA and no packed byte arithmetic, so the wave32 backend above refuses it. A separate opt-in build compiles the CUDA sources as HIP through a small compat layer
(`include/strata/platform/hip_compat/`), with a CUDA warp mapped to a logical half of the 64-lane wavefront
(32-wide shuffles, a ballot of its own half). The hot kernels have wave64 layouts of their own (below). Setup does
not build it yet: build by hand, and run `serve/server.py` with a config as on any other card.

**ROCm.** AMD's current ROCm releases no longer ship gfx906 libraries. The build and the measurements below used
HIP 7.14 from the community image [`mixa3607/rocm-gfx906:7.14-complete`](https://hub.docker.com/r/mixa3607/rocm-gfx906)
(rocBLAS/hipBLAS with gfx906 kernels), on the kernel's amdgpu driver (Ubuntu 24.04, kernel 6.8). hipBLASLt is not
used.

```sh
# inside the image, with this checkout at /src and llama.cpp at the pinned commit in third_party/llama.cpp
apt-get update && apt-get install -y cmake ninja-build git python3 build-essential
cmake -S . -B build-906 -G Ninja -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=/src/third_party/llama.cpp \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja -C build-906 strata
```

`STRATA_HIP_GFX906` turns on the CUDA targets and refuses `STRATA_ENABLE_HIP` (one HIP build at a time). Run the
container with `--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --ulimit memlock=-1`; on some boards
`HSA_OVERRIDE_GFX_VERSION=9.0.6` is needed for the runtime to accept the card.

**What differs from the CUDA build** (all under `STRATA_HIP_GFX906`; each A/B switch restores the CUDA layout):

| | gfx906 | switch |
|---|---|---|
| grouped native experts | mode 7: signs as `dp4a(g ^ m, u) - dp4a(m, u)` (no byte SIMD), grids and activations in LDS, one weight load per group; the IQ formats only (Unsloth's K-quant / Q5_1 / Q8_0 experts take the CUDA layout) | `STRATA_EXP_MODE=2` (previous AMD layout) |
| MMVQ | one wavefront per row, 64-lane butterfly, no LDS | `STRATA_MMVQ_WAVE=0` |
| router top-10 | one wavefront, register argmax | - |
| verify-window routing (256-expert router) | the window's tokens in one multi-column BF16 projection + one top-k, ~3 ms of a ~55 ms window | `STRATA_ROUTE_PER_TOKEN=1` |
| hyper-connection read | norm loads in flight, 8 lanes per row for `up` | `STRATA_GR_FAST=0` |
| `gr_down_multi`, opt-in | split along K (8 rows x 5 slices of 2048), ~3% per window; sums in another order than the single-token read, so off by default | `STRATA_GR_SPLIT=1` |
| IQ4 table lookup | llama.cpp's `v_perm_b32` sequence (HIP's `__byte_perm` is a scratch array) | - |
| QSA prompt attention | the FP32 kernel (no tensor cores / WMMA) | - |

Every switch above was checked bitwise against the layout it replaces: the parity tests on synthetic data,
`native_expert_bench` on real GGUF rows (one mode against another), and the greedy text of a fixed request.

**Measured** on 2x Instinct MI50 16 GB (85 W power limit each, PCIe 3.0 x16 both, peer access between them),
Xeon E5-2666 v3 (10 cores, AVX2), 32 GB DDR4, SATA SSD; Coder IQ1_M (`--native` pack), 131,072-token context,
`--kv int8 --kv-resident 32768`, `--layer-split 27`, `--prefill 4096`, MTP `--spec 4 --spec-min-p 0.5`,
`--pcie-frac 0`, `STRATA_ARENA_MMAP=1`. Measured on the port this build was cut from (builds on the 0.1.30 and 0.1.33 bases, the same
kernels, plus the layer-split weight trim and the mapped arena, which are separate pull requests):

- all 12,288 experts held in VRAM across the two cards;
- a 17,043-token agent prompt (Claude Code's first request, tools included) read in 43.5 s (392 tok/s), decode
  on it 57.7-59.3 tok/s;
- 25 GB of the 32 GB RAM available while serving;
- six repeats of that request with a 6,000-token output cap and a 900-second two-client load test run six times
  (407 requests, 202,940 tokens): no stalls, no errors, no GPU faults.

For comparison on the same machine: llama.cpp (3cf0325, ROCm) on the same Coder IQ1_M measured 26.9 tok/s `tg128`.

**Not done:** setup.py detection and an automatic build, Windows, images (`--vision`), MI60 and Radeon VII (the
same gfx906 ISA; not run), a single-card run, the tensor-split experiment (the halves of every layer on two cards;
it works but is not part of this build).

## Tuning table

A hipBLASLt table holds solution ids that are valid only for one GPU architecture and one hipBLASLt version, so it
is calibrated on the card with the ROCm the engine runs with. The shipped gfx1100 table's rows are the engine's
dense GEMM shapes; to calibrate them for another card or version (a few minutes):

```sh
cmake --build build-hip --target tune_hipblaslt
CASES=$(awk 'NR>2 {printf " --case %s,%s,%s,%s,%s", $1, $5, $2, $3, $4}' tools/hip/gfx1100-hipblaslt-100200.txt)
./build-hip/tune_hipblaslt $CASES --tuning-out table.txt
```

The file's second line names the architecture and version (`STRATA_HIPBLASLT_TUNING_V1 gfx1201 100401`); save it
as `tools/hip/<arch>-hipblaslt-<version>.txt` for setup, or point `STRATA_HIPBLASLT_TUNING` at it. Compare the
prompt speed with and without it before keeping it.

Shipped tables:

- `gfx1100-hipblaslt-100100.txt`, `gfx1100-hipblaslt-100200.txt`: RX 7900 XTX.
- `gfx1100-hipblaslt-100401.txt`: RX 7900 XTX, calibrated with the packaged ROCm 10.0.0
  (`rocm/dev-ubuntu-24.04:10.0.0-full`, hipBLASLt 1.4.1) on a Ryzen 7 9800X3D, over the 26 dense GEMM
  geometries of the shipped gfx1100 table. Without it, that stack reads a prompt through plain hipBLAS: on
  Swift 1.5 IQ3_XXS at 262144 ctx, fresh-prompt prefill measured 682 -> 1,038 tok/s at 1.7K tokens and
  857 -> 1,524 tok/s at 6.5K (medians of 3, decode unchanged), with `hip_prefill_hipblaslt_gemm` reporting
  `fallbacks=0`. setup uses it only when the installed hipBLASLt reports 1.4.1.
- `gfx1100-hipblaslt-100500.txt`: RX 7900 XTX (gfx1100, 24 GB), calibrated with a ROCm 10.2.0a20261003 nightly
  SDK (`libamdhip64.so.7.17.26392`, hipBLASLt 1.5.0, version number 100500). Same 16 dense GEMM geometries at
  T=4096 and T=8192 as the other gfx1100 tables (32 rows), calibrated with `tune_hipblaslt` run against that
  library. On this ROCm the engine refuses the 100100/100200 tables (version mismatch) and prefills on plain
  hipBLAS; with this table a 0.1.38-lineage nightly engine prefills a 131071-token prompt at 1687 tok/s vs 926
  without it (median of 3 clean runs each, greedy; decode unchanged). Per-run data and the calibration command:
  the `2026-10-04-community-rx7900xtx-hipblaslt-100500` folder of PR #745.
- `gfx1201-hipblaslt-100500.txt`: Radeon AI PRO R9700 (gfx1201, 32 GB), calibrated with ROCm 10.2.0a20260914
  (AMD's `gfx120X-all` nightly, hipBLASLt 1.5.0, library build `d3164197`). 16 dense GEMM geometries at T=4096 and
  T=8192, 32 rows. setup uses it only when the installed hipBLASLt reports 1.5.0 (it is found in `/opt/rocm`
  when that is a system ROCm 7 or newer). The version number is the only thing the engine can check, so another
  1.5.0 build could number its solutions differently. `hip_prefill_hipblaslt_gemm` (with `STRATA_HIPBLASLT_TUNING`
  set) is a smoke test: it refuses a table for another architecture or version and checks that one BF16 and one
  F16 row exist and agree with hipBLASEx, which covers 2 of the 32 rows. It does not prove that a solution id is
  valid: the engine falls back to hipBLASEx for an id the library rejects, and the test still passes. Run it with
  `STRATA_HIPBLASLT_VERBOSE=1` and look for `fallbacks=0` in its summary line, and recalibrate with
  `tune_hipblaslt` before using this table with a different 1.5.0 build.

## Original backend validation (PR #94)

The following is historical validation of the original backend, not a fresh
test count for this replacement. Current build/test evidence and performance
limits are recorded in [AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md).

Tested against upstream `c1e903310f211e6630780c3bd2038778c071c68d` (0.1.20),
with pinned llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
The host has an RX 7900 XTX (24 GiB), Ryzen 9 7950X3D, 64 GiB system RAM,
and Fedora-family Linux with ROCm 7.1.52802 / Clang 20. The original model used
for the smoke check was on mechanical RAID0; cold page faults were slow. Keep
latency-sensitive data on SSD and measure warm residency separately.

- Complete HIP Release build, including the executable and parity targets.
- All 25 tests selected by the command below passed on 2026-09-29.
- Intrinsic parity includes signed packed-byte dot products and overflow.
- CPU/GPU mapped-memory handoff covers delayed publication, changing payloads,
  and repeated graph replay in both directions.
- GR coverage includes the maximum eight-token fused batch and graph replay
  after inputs change. Native QSA checks the scalar HIP fallback.
- Q8_K quantization is byte-exact against the existing reference. HIP's
  `__fmul_rn` can become ordinary multiplication; disabling contraction in
  `quantize_act.cu` preserves the separately rounded product before the magic
  rounding bias is added. CUDA's compile flags remain unchanged.
- The mmap regression covers canonical and differing native layer sizes,
  truncated/oversized files, lookup bounds, and reopening after errors.
- CUDA 13 / GCC 15 / sm_89 executable build passed; GR, sampler, quantization,
  and mmap tests passed on the same tree. This is a regression check, not a
  claim of complete CUDA inference validation.
- CPU-only `strata-plan` build passed without either GPU backend.
- Real GSQ-RCO IQ3_XXS server smoke with mmap, the shipped expert profile,
  automatic GPU cache, and MTP passed arithmetic, executable Python addition,
  system-marker recall, and 1,170-token batched-prefill recall. All four requests
  ended normally with correct answers; one engine process served them all.
  The cache held 11,142 experts (18.08 GiB), with 874 MiB of VRAM free after
  startup. These short checks do not establish a throughput or quality benchmark.

Build all targets before running the registered focused checks:

```sh
cmake --build build-hip -j2
ctest --test-dir build-hip --output-on-failure --timeout 60 \
  -E '^(ple_parity|platform_memory_test)$'
```

`ple_parity` requires an external model fixture. `platform_memory_test` requests
256 MiB of locked memory; the test shell's 8 MiB memlock limit prevented that
test from passing. These are explicit exclusions, not skipped tests counted as
passes. Additional unpublished upstream fixture suites are not covered.

The published performance table explicitly identifies the measured development
snapshot; it must not be read as a benchmark of every subsequent rebase. Vision, long-context stress, broad answer-quality
equivalence, other AMD cards, and mixed-vendor inference are not validated here.
