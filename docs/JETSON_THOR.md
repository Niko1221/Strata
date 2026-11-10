# Strata on the NVIDIA Jetson AGX Thor (Linux ARM64, sm_110)

The Jetson AGX Thor is an embedded module with a Blackwell-class GPU that NVIDIA numbers as compute capability **11.0** (`sm_110`) — a
different part from the DGX Spark's GB10 (`sm_121`), even though both are ARM64 Linux with one shared memory. Machine code for one does
not run on the other: CUDA keeps each card's own image, and `sm_121` code cannot run on `sm_110`. Since 0.2.10 the `linux-aarch64`
engine package therefore carries **machine code for both** (an `sm_121` cubin, an `sm_110` cubin and `compute_110` PTX), built and checked
on the two machines.

This page was written on one Jetson AGX Thor Developer Kit:

| Item | Value |
|------|-------|
| GPU | NVIDIA Thor, compute capability 11.0 (`sm_110`) |
| Memory | 122 GiB unified (the CPU's RAM is the GPU's memory; `nvidia-smi` gives no separate figure) |
| SoC / OS | L4T R38 rev 2.2, Ubuntu 24.04.4, kernel 6.8.12-tegra, aarch64 |
| Driver / CUDA | 580.00 / CUDA 13.0 (nvcc 13.0.48, sbsa-linux) |

## What works, and what does not

- **Works from source:** the engine build (`spark/build_engine.sh`, engine and `strata-dsv41`), the CUDA kernel parity tests on the
  GPU, and the Terminal reading the machine (`strata doctor`, `setup.py --check`: the Thor is found, counted as unified memory and asked
  for `sm_110`).  One test fix came with it: on any ARM64 host the `quantize_act_parity` host references now compile `-ffp-contract=off`
  (CMake sets it) - gcc on aarch64 fuses the host's `iscale * xb[j]` into the rounding trick that follows, which stops it matching the
  device's barriered `__fmul_rn` on exact .5 ties; the device bytes were right all along.
- **Serving a real model, in the container (smoke, not a benchmark):** inside the official CUDA 13.0 devel image
  (`--runtime nvidia --ulimit memlock=-1`) with the container-built engine package as the component, `strata launch --model iq3_s
  --context 32k --vision off --drafting mtp --tune skip` found the already-downloaded 84 GB Qwen3.8-Flash-Next IQ3_S pack and nothing
  to download, loaded it in about 5 minutes and served: 17*23 answered 391, a 450-token completion came back in 13.3 s (~34 tok/s),
  `nvidia-smi` showed the Thor at 98 % during generation, `strata status` reported the model ready at context 32768 with four parallel
  slots and 105.7 of 131.9 GB used, and the running binary was the container-built `strata --serve` from the package folder. The full
  benchmark shape of docs/COMMUNITY_BENCHMARKS.md is still open (#4560).
- **Measured since (2026-11-05):** both served model families ran the standard protocol suite (`strata bench run`,
  `standard-v1`) on this machine - sanity 12/12 on each. IQ3_S at 32K/4 slots: decode 38.3/34.0/35.6 and prompt
  769/1000 tok/s (short/4K/16K), four at once 51-54 tok/s in total; Qwen3.6-35B-A3B UD-Q4_K_XL at 32K/4 slots:
  decode 50.7/37.4/20.2 and prompt 1375/1221 tok/s, four at once ~100.5 tok/s in total. Qwen3.6 at 128K and
  4 slots is refused by the engine itself with exact numbers on this pool; the Tegra page-cache accounting note
  and the full data are in [the report](../bench/results/2026-11-05-community-jetson-thor-sm110/README.md).
- **Run since, with data in the report:** DFlash2 drafting (correct, sanity 12/12, but slower than the MTP on this part - so `_TESTED`
  keeps `sm_110` out and the Terminal still says the default pairing was never measured); serving a DeepSeek-V4.1 set
  (ISTA-DAS Coder, 191 GB, at 8K and 64K context - speed cells in the report; whether a given set downloads is a link-speed
  question, not an engine one); the locked-clock pair; NVFP4 (correct, slower).
- **Not run on this part yet:** picture reading, Qwen3.6 UD-Q6_K_XL / Q8_0 (the same install path as UD-Q4_K_XL, each a further
  33/38 GB download), the 128K context, the other DeepSeek sets (239-274 GB), more than one module, a long soak.
- **Not built for ARM64 at all:** the desktop app, the HIP, SYCL and CPU-only engines, and the Q2_0 and Gyro packs, whose CPU kernels are
  x86-only.

## Install with the Terminal

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
strata launch
```

`install.sh` maps the machine to `linux-aarch64` and names both ARM machines (DGX Spark, Jetson AGX Thor) in its note. `strata launch`
fetches the signed `linux-aarch64` engine package; since 0.2.10 its feed entry declares `sm_121, sm_110, compute_110` and its minimum
NVIDIA driver is 580.00 — the figure L4T R38 reports (DGX OS's Spark reports 580.65 or newer; it is the same r580 branch). The package
needs glibc 2.39 or newer (Ubuntu 24.04, L4T R38, DGX OS 7).

## Build the engine yourself

You need the CUDA 13.0 toolkit (`/usr/local/cuda`, in L4T's JetPack), cmake, ninja and Python 3.10 or newer.

```sh
python3 -m venv .venv-build
.venv-build/bin/pip install -r requirements.txt -r requirements-build.txt
PATH="$PWD/.venv-build/bin:/usr/local/cuda/bin:$PATH" PYTHON="$PWD/.venv-build/bin/python" JOBS=14 sh spark/build_engine.sh
cp -r out-spark/engine ./engine
```

Without `ARCHS` the build asks the machine (nvidia-smi's compute capability): a Thor answers 11.0 and compiles `sm_110`, a Spark answers
12.1 and compiles `sm_121`. `ARCHS="121;110"` makes one binary with both cards' machine code — what the release package is built with
(`packaging/build_cuda_linux_aarch64.sh`).

Then start a model with `./setup.sh` (or `./setup.sh --setup ...`, see docs/SOURCE_INSTALL.md). Setup finds the Thor, counts its memory
as unified, takes `sm_110` from the engine or compiles it, and gives it the unified-memory expert placement (`--mmap-experts`) for the
same reason the Spark gets it: the RAM copy and the GPU's cache are the same memory. The Spark's per-part speed settings (`--prefill
auto:32768`, `--mtp-q4 all`, `--mtp-hnorm stream`) are **not** applied to the Thor: they were measured on the Spark. Nothing is measured
on the Thor yet; `docs/COMMUNITY_BENCHMARKS.md` explains how to add a run.

## What was checked on this machine (2026-11-04, main @ ffa80d73, engine 0.1.56)

1. `ARCHS="121;110" spark/build_engine.sh` end to end: all 179 targets, `strata 0.1.56` and `strata-dsv41 0.1.56` link and print their
   versions on the Thor. `cuobjdump` on the binary: 70 `sm_121` cubins, 70 `sm_110` cubins, 70 `compute_110` PTX modules.
2. The engine and C++ test suite built with `-DCMAKE_CUDA_ARCHITECTURES=110` and run with the GPU in place:
   **193 of 195 ctest tests pass** on the Thor, the CUDA kernel parity suites among them. The two that do not pass
   here fail the same way from unmodified `main` (they are not Thor regressions): `expert_cache_segmented_test`
   asks the driver to show the freed memory in `cudaMemGetInfo` and Tegra gives the pages back to the shared pool
   without moving that number (the cache's own slot and size checks pass), and `ple_parity` wants a Q2_0 model
   file as a fixture that is not on this machine.
3. The Terminal and Control test suites (`tests/terminal`): the two aarch64 files pass fully (33 tests, the new
   Thor cases among them); the whole folder shows 2426 passed with 65 failures that appear identically on
   unmodified `main` on this machine (environment, not Thor). `tools/test_setup_*.py`: 56 of 61, the 5 the same
   on unmodified `main`.
4. `setup.py --check` on the Thor: `GPU: NVIDIA Thor, unified memory (117 GB usable by the GPU), compute capability 11.0,
   driver 580.00`, and the ARM64 path (feed engine or a local compile). `strata doctor` runs; its `--live` device probe wants a saved
   profile, which needs a model (not installed on this machine).
5. `COMMIT=<hex> packaging/build_cuda_linux_aarch64.sh` end to end on the Thor: both engine packages made, `package_engine.py
   --verify` and `binary_deps.py` pass on each, and `engine_component.resolve` on this machine (`sm_110`, driver 580.00) picks the
   package it just built (`sm_121, sm_110, compute_110`; floor 580.00) and its `strata` runs from the package folder.
6. The container path NVIDIA's Thor guide describes (arm64 CUDA 13.0 containers; here the community Thor image
   `eugr/spark-vllm`, CUDA 13.0 + Ubuntu 24.04, run with `--runtime nvidia`): inside it `strata` and `strata-dsv41`
   print their versions, `cuobjdump` counts the 70 `sm_121` + 70 `sm_110` cubins and 70 `compute_110` PTX modules,
   and a small `sm_121 + sm_110` CUDA program compiled inside the container runs on the Thor (20 SMs, cc 11.0).

## The container build reproduces the host build

The whole chain was built a second time from scratch inside the official `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`
image, whose `nvcc` is the same `V13.0.48 / compiler.36260728` this Thor runs (`--runtime nvidia`; the source is a
`git archive` of the branch, and the container only apt-installs `cmake ninja-build git python3`):

- The fat release build (`ARCHS="121;110"`, `spark/build_engine.sh`) completes in the container and matches the host build
  where it matters: same file size, same kernel name set, and the same multiset of per-kernel REG/STACK/SHARED figures over
  all 140 cubins and 70 PTX modules (`cuobjdump -res-usage`, path-hashes normalized). It is not byte-equal to the host
  binary and cannot be: `nvcc` hashes the compile path into anonymous-namespace symbol names, and `strata-dsv41` stamps
  its own build time by design.
- `packaging/build_cuda_linux_aarch64.sh` end to end in the container: both packages build, `package_engine.py --verify`
  and `binary_deps.py` pass on each inside the container and re-verify from the host, and `engine_component.resolve` on
  this machine picks the container-built package - its `strata` runs on the Thor.
- `ctest` there is 190 of 193 on the Thor GPU. Two failures are the same two environment ones as the host run; the third,
  `platform_memory_test`, is Docker's default 64 KiB memlock refusing its `mlock` - it passes with `--ulimit memlock=-1`,
  which is how NVIDIA's own container instructions run. `iq_parity` is not registered in a container without a
  numpy-bearing python (CMake's probe), which is by design.

## Speed: measured defaults and drafting depth

The engine now carries an sm_110 speed table of five switches (`STRATA_SPEC_COST`, `STRATA_HC_Q8`, `STRATA_PF_FUSED`,
`STRATA_HC_UPMIX`, `STRATA_SIDE_MIXER`) - each measured on a Thor before it went in, `STRATA_SM110_PERF=0` turns the set
off. Suite run against suite run - different sessions, labeled as such in the report: prompt +20% (4K) and +10% (16K),
first answer 27% sooner at 4K, decode +8% at 4K and about -6% at 16K; the paired within-session evidence is the
`ab-iq3s-*.jsonl` arm alternation. For drafting: at 120 W Qwen3.6 wants a deeper MTP window than the shipped default
(`--mtp 7` measured best, +5%; the default 2 even loses to drafting off), and at locked MAXN clocks depths 3-6 sit level
with drafting off while the default 2 stays the worst cell - `--mtp 3` is the advice that survives both clock states.
IQ3_S peaks at `--spec 4` with the cost rule on,
and DFlash2's drafter works (sanity 12/12) but is slower than the model's own MTP layer on this part. One machine behavior
to know: after an engine has run, Tegra's `cudaMemGetInfo` can report only a few GiB free while the OS has tens free,
until the machine restarts - the engines' sizing reads that number, so a restart before a measuring run is part of the
method here. Long context measured too ([longctx](../bench/results/2026-11-05-community-jetson-thor-sm110/longctx/README.md)):
IQ3_S at 128K (2-4 slots) and 256K (1 slot), decode flat 36-38 tok/s through 256K context, and at T=0 long prompts none
of the switches separates from the defaults - leave them on. All the tables, pairs and caveats: [bench/results/2026-11-05-community-jetson-thor-sm110/README.md](../bench/results/2026-11-05-community-jetson-thor-sm110/README.md).

## Limits

One Thor was used. Contexts, memory figures and speeds above are the machine's own, not the Spark's: the Spark's measured settings and
DeepSeek speed lines are deliberately not claimed for the Thor (the DeepSeek fit says so on purpose). If you run a model on a Thor, a
run in the shape of docs/COMMUNITY_BENCHMARKS.md is welcome on the issue.

See also [DGX_SPARK_LINUX.md](DGX_SPARK_LINUX.md) (the other ARM64 machine) and issue #4559.
