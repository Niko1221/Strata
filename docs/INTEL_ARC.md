# Intel Arc (experimental)

Strata 0.1.39 includes an **experimental Intel Arc engine**: Strata's own engine ported to SYCL (Intel oneAPI).
maxfridbe wrote it in [#423](https://github.com/Niko1221/Strata/pull/423), with fixes from the people testing it.
The code is in `sycl/`, and the port's own notes, measurements and maintenance procedure are in
[docs/INTEL.md](INTEL.md). This page covers what you need, how to build it, and what has been tested.

**Experimental means:** the Strata maintainers have no Intel GPU. We compile the port and run its kernel tests, but
we have not run it on an Arc. Every result on Arc hardware below comes from the community. The NVIDIA and AMD
engines are unchanged: the Intel build is a separate CMake target, off by default.

There is **no ready-made Intel engine** in the release zips. You build it from source on Linux.

## What has been run, and by whom

| | Hardware | Result | Reported in |
|---|---|---|---|
| Port author | Arc Pro B70 32 GB, Ubuntu 24.04 | Coder IQ1_M: 70-78 tok/s decode, ~790 tok/s prompt; IQ2_XS: 51-64 tok/s; 256K context measured | #423, [INTEL.md](INTEL.md) |
| Community | 2x Arc Pro B70, `--layer-split` | Flash-Next IQ3_XXS 66 tok/s decode, 394 tok/s prompt (with the `stage_room` fix that is now in 0.1.39) | #423 |
| Community | Arc Pro B50 16 GB | Coder IQ1_M ~23 tok/s, IQ2_XS ~25-27 tok/s, up to 128K | #423 |
| Community | Arc B580 12 GB, WSL2 | IQ2_XS ~21 tok/s, Coder ~15 tok/s; **device loss also seen** | #423 |
| Community | Arc A770 16 GB, Linux VM | Flash-Next IQ2_XS: 17.8 / 16.6 tok/s decode after 511 / 4096 prompt tokens, with the A770 fixes below | [A770 qualification](#a770-16-gb-qualification-2026-10-04) |
| Strata maintainers | no Arc | compile check and kernel tests on a CPU device only (below) | this release |

The 0.1.39 port re-migrates 0.1.38's port onto the 0.1.39 engine sources (the #606 NaN fix, the #649 verify
trace, the new prompt paths). The maintainer qualification below covered compilation and CPU kernel tests.
The A770 row is a later community measurement of **0.1.39 plus the fixes in this branch**, not the unmodified release. The other
hardware numbers were measured on earlier versions.

## Maintainer qualification (0.1.39 release)

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

**Not tested by anyone yet:** Windows (no native build path; see below), Arc on WSL2 for the 0.1.39 port, Alchemist
cards other than the A770 configuration below, integrated Arc GPUs (the B390 / Panther Lake in #515;
the shared-memory planning does not exist yet), and images (not wired on Intel).

## What you need

- **Linux**, e.g. Ubuntu 24.04, with Intel's GPU driver (the `xe` or `i915` kernel driver plus the compute runtime /
  Level Zero; on Ubuntu, `intel-opencl-icd libze1 libze-intel-gpu1`, or Intel's
  [client GPU guide](https://dgpu-docs.intel.com/driver/client/overview.html)).
- **Intel oneAPI**: the DPC++ compiler (`icpx`, 2025.3 or newer; 2026.1 is what was built here) and **oneMKL**. About 5 GB.
- `cmake` 3.24+, `ninja`, `git` (the build fetches ggml unless you point `STRATA_GGML_DIR` at a llama.cpp checkout),
  Python 3.
- For `setup --backend sycl` today: **Docker**. `sycl/setup_intel.py` runs the engine in the `strata-sycl-dev`
  image built from `sycl/tools/Dockerfile`. Note that the Dockerfile starts from a community llama.cpp SYCL image
  (`ghcr.io/snailium/...`), not an Intel or Strata image.
- VRAM: the port keeps the experts on the card (`--stream-experts`). A 32 GB card holds the Coder IQ1_M or IQ2_XS.
  Smaller cards mirror part of the experts in RAM and are slower.

## Build (Linux)

Install oneAPI from Intel's apt repository (this is what was used here):

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

This prints the experimental warning and continues with `sycl/setup_intel.py`. That script finds the Arc in sysfs,
uses the SYCL engine you built, and writes the config and `run-<model>.sh`. It still downloads and packs the model
the usual way. To run the engine by hand (no Docker), see "How to run it by hand" in [INTEL.md](INTEL.md).

Things that matter on an Arc (details in INTEL.md):

- **Do not ask for more VRAM than the card has.** On the `xe` driver, an allocation past VRAM can push buffers into
  RAM until the machine runs out of memory and stalls. Leave about 1.5 GB free.
- `SYCL_CACHE_PERSISTENT=0`: the persistent JIT cache crashed on Xe2 during the first compile.
- Two cards: `ONEAPI_DEVICE_SELECTOR=level_zero:*` (the image pins `level_zero:0`; `strata-sycl.sh` now passes the
  variable through) and `--layer-split`.
- `STRATA_VERIFY_NO_HOST=1` (set by `strata-sycl.sh`) requires `STRATA_VERIFY_DEVICE_PLAN=1` and complete
  expert coverage: each expert must be in VRAM or in the pinned host mirror used by the device plan. The
  A770 configuration below runs with most experts mirrored. The engine now rejects a missing device plan
  or incomplete streaming mirror instead of proceeding with an unusable no-host configuration.

## A770 16 GB qualification (2026-10-04)

**Working, still experimental, and still slower than the preserved CUDA comparison.** These measurements use
Flash-Next GSQ RCO IQ2_XS on an Arc A770 16 GB in a Linux VM, a Ryzen 5 3600 and about 94 GiB of RAM.
Software: Intel NEO 26.35.39758.10, oneAPI DPC++ 2026.1.1 and oneMKL 2026.1; SPIR-V/JIT, not AOT.
This is the official 0.1.39 engine with the A770 changes in this branch. Unmodified 0.1.39 did not reach decode
on this system.

### What was needed

- Refresh the migrated `ThreadAffinity` and `NativeDense::load` signatures to match the shared API.
- Use UUID-matched Level Zero Sysman free-memory telemetry when `ext_intel_free_memory` is unavailable.
  The unsupported extension otherwise reports zero free bytes and prevents a usable expert cache.
- Use Level Zero relaxed allocations for the large device cache and pinned host mirror, with matching frees.
  Ordinary SYCL allocations failed for the 4.22 GiB cache and approximately 30 GiB mirror on this system.
- Fix the native router's barrier: seven of eight subgroup rows return before synchronization, so the remaining
  row must use a subgroup barrier, not a work-group barrier. The original kernel hung on A770; the new
  `native_router_parity` test checks changing inputs, ties, selected expert IDs and weights.
- Reject no-host verification without a device plan or with incomplete streaming mirror coverage.

The complete mirror holds **21,442 of 21,442 experts missing from VRAM (28.80 GiB)**. The GPU reads those
experts directly over PCIe. This path needs substantial system RAM; the 94 GiB machine above is the tested
configuration, not a measured minimum-RAM requirement.

### Measured throughput

Each arm uses one warmup followed by three short and three long requests, all generating 256 tokens with
zero prompt reuse. The table gives median engine-reported tokens/s, with ranges in parentheses. The saved
synthetic data-validation requests are identical across the A770 and preserved CUDA runs. Requests are greedy
(`temperature=0`, `top_k=1`, `top_p=1`, `min_p=0`, `seed=42`, `reasoning_effort="none"`).

| Engine / FP64 emulation | Prompt tokens | Prompt tok/s | Decode tok/s |
|---|---:|---:|---:|
| A770, enabled | 511 | 116.5 (102.3–116.6) | **17.8 (17.4–17.8)** |
| A770, enabled | 4096 | 337.7 (329.0–337.9) | **16.6 (16.6–17.9)** |
| A770, disabled | 511 | 116.5 (100.8–116.6) | 17.8 (17.4–17.8) |
| A770, disabled | 4096 | 337.6 (324.0–337.6) | 16.6 (16.6–17.8) |
| RTX 5060 Ti CUDA, preserved cache3000 | 511 | 83.8 (80.4–84.5) | 22.5 (21.9–22.5) |
| RTX 5060 Ti CUDA, preserved cache3000 | 4096 | 345.6 (343.8–345.6) | 21.1 (19.7–21.3) |

Shared settings: `--expert-cache 3000` (3134 effective slots / 4.22 GiB), `--max-context 32768`,
`--prefill 4096`, INT8 KV, `--spec 4 --spec-min-p 0.5` with MTP, no prefill borrowing, `--prompt-cache 0`. The expert cache evolves between
requests; OS/model pages were warm. These are nonstreaming measurements, not time-to-first-token results.
The historical CUDA comparison uses different hardware, PCIe connectivity and engine execution policies.
A770 decode remains about 21% slower; the separate CUDA cache6000 result is not a matched comparison.

An earlier custom A770 port measured 2.7/2.7 tok/s on this same corpus at cache3000. The new result is over
six times faster, but the release, kernels, host mirror and CPU policy all changed. It does not isolate the
speed contribution of any single fix, and it is not evidence of a new XMX decode implementation.

**FP64 control:** both arms used the same binary and returned exactly matching text for all seven requests.
Decode medians are identical at the reported precision. This shows no meaningful observed emulation penalty
on this native greedy path. ON then OFF was sequential, not randomized; it does not measure the old custom
port's FP64 contribution or qualify every legacy/sampled path with emulation disabled. Keep emulation enabled
for the general configuration below.

The [saved corpus, per-request measurements and output hashes](benchmarks/2026-10-04-a770.json) include the
engine/source hashes and both arms' settings. Model hashes are inherited from the earlier qualification;
the large model files were reused, not rehashed during this run.

### Settings used on the A770

Build the `sycl/` project as above. For a direct native run, set these variables in the **engine's process**
(or the `env` object of its server JSON config); substitute your prepared model and prompt paths:

```sh
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_CACHE_PERSISTENT=0
export STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1
export STRATA_STAGER_THREADS=12 STRATA_MIRROR_MIB=40000
export NEO_FP64_EMULATION=1
export SYCL_PROGRAM_APPEND_COMPILE_OPTIONS=-ze-intel-greater-than-4GB-buffer-required

build-sycl/strata --pack "$PACK" --native "$SHARD1" --ple-gguf "$SHARD2" \
  --expert-profile data/expert-profile.bin --expert-cache 3000 --stream-experts \
  --prefill 4096 --spec 4 --spec-min-p 0.5 --mtp "$MTP" --max-context 32768 \
  --kv int8 --no-prefill-borrow --prompt-cache 0 \
  --tokens-file "$PROMPT_TOKEN_IDS" --max-new 256 --greedy --stop-eos
```

`PROMPT_TOKEN_IDS` is a file of token IDs for a rendered chat prompt, not raw prompt text. The measurements
above used the API and the saved corpus; this command shows the matching engine configuration. For a server,
put the same model/cache arguments and environment in its JSON config and bind to `127.0.0.1`.

The greater-than-4GB compiler option is required in addition to relaxed allocation: allocating a large buffer
does not by itself make kernel offsets above 4 GiB correct. Check startup for complete mirror coverage before
using no-host verification. The opt-in `STRATA_VERIFY_EAGER_TRACE` diagnostics are off in all performance runs.

### Validation and limits

- **5/5 targeted GPU CTests pass:** `gr_parity`, `quantize_act_parity`, `qsa_parity`, `native_grouped_parity`,
  `native_router_parity`. The new router test passes 20 cases (256/512 experts, 1/6 tokens, changing logits and
  ties), exact stable IDs and an independent CPU double-softmax reference within 2e-6. It runs without GPU FP64.
- Real-model expert checks at layers 0/1/3/47 pass the fixture's 3% reference tolerance; GPU relative errors
  are approximately 1.13–1.29%. A separate 5 GiB device plus 5 GiB host allocation probe verifies kernel reads
  and writes beyond 4 GiB and memory release.
- Two fresh-process `17 * 23` tests with emulation enabled and two with it disabled return identical correct
  token IDs. Both seven-request throughput arms complete and their servers stop successfully.
- **Strict API correctness is 4/6, not a full pass.** Paris and `17 * 23` pass twice. The Python expression
  `sum(i*i for i in range(4))` returns 30 instead of 14 twice. The same failure was recorded in earlier CUDA
  and custom A770 runs. Longer answers are coherent but include unsupported dataset claims; throughput
  completion does not establish factual accuracy or full-model numerical parity.
- The earlier broad A770 suite combined to 20/25 across baseline and failed-case reruns. Missing fixtures,
  a `kv_stream_parity` timeout, `s2_expert_grouped_parity` bitwise differences and a
  `conversation_snapshot_test` segfault remain unresolved. The full suite was not rerun after these fixes.

## Windows

There is no Windows path yet. `setup --backend sycl` on Windows stops and points here. oneAPI exists for Windows,
but `sycl/CMakeLists.txt` uses GCC-style flags (`-mavx512f`, `-fp-model=precise`, `-qmkl`) and the runner is a
bash/Docker script, so a native Windows build would need work. Nobody has tried it. WSL2 with an Arc has been
used by one tester (the B580 row above), but setup cannot detect the card there, because it reads `/sys/class/drm`,
which WSL2 does not have.

## Reporting a problem

Open an issue with: the card, the driver version, `sycl-ls` output, the oneAPI version, the model and flags, and the
engine's stderr. Results from real cards are what move this from experimental to supported.
