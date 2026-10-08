# HIP and SYCL compilation on maestro1, 2026-10-08

Host: Intel Celeron N3450, four cores, x86-64, Ubuntu 22.04.5,
kernel 6.8.0-65-generic, 3.7 GiB visible RAM and 4 GiB swap.
The installed graphics device is Intel HD Graphics 500. AMD and Arc GPU
runtime checks were not run. These are compilation checks, not inference
benchmarks or an ARM HIP/SYCL validation.

Source: `13ea3b0f13ecad9da4a6d0915ea6d508b0b4deb2`, based on upstream
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (0.1.41). The source checkout
was clean; [recorded status](source-status.txt). The engine sources are the
same as those used for the rebased 0.1.41 [Orin validation](../2026-10-08-jetson-orin/README.md);
the later commit adds report graphs. Both builds use the pinned llama.cpp
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, obtained as its source archive.

## Toolchains and builds

Toolchains and scratch files are under `~/workspace/strata-validation` on the
SSD mounted at `/mnt/sda2`. Installation did not change system packages or
GPU drivers. CMake 3.31.10 and Ninja 1.13.2 were installed in a private Python
environment; [versions](build-tools-packages.txt).

| Backend | Toolchain | Configuration | Result |
| --- | --- | --- | --- |
| HIP | AMD TheRock wheels `7.10.0a20251120`, AMD Clang 22.0.0git | Release, gfx1100, native experts, MMQ on, tests on | Full build passed |
| SYCL | Intel DPC++ 2026.1.1 and oneMKL 2026.1.0 | Release, SPIR-V, native experts, parity targets on, no AOT | Full build passed |

The complete default target set was requested, including the engine and
configured test executables. These results do not mean that the GPU tests
were executed. The HIP configuration did not register IQ fixture tests
without NumPy or the expert arena load test without its model fixture;
the IQ parity executable was still built.

Both configurations use `-O3 -DNDEBUG`; [HIP flags](hip-flags.txt),
[SYCL flags](sycl-flags.txt).
The first HIP compiler probe selected an incomplete GCC 12 installation
without C++ headers. Both toolchains were then directed to the working GCC 11
installation with `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11`.
The [failed discovery log](hip-configure-incomplete-gcc12.txt) and
[successful configuration](hip-configure.txt) preserve the distinction.
No engine source change was needed.

HIP was initially built with one job, then resumed with two. SYCL began with
one job, used two while HIP ran, then used three after HIP completed. Compiler
flags were unchanged between resumptions. Interrupted logs reflect these
parallelism changes rather than compiler errors.

HIP evidence: [compiler and host](hip-metadata.txt),
[package versions](rocm-packages.txt), [installation](rocm-install.txt),
[initial build](hip-build-initial.txt), [completed build](hip-build.txt),
[result](hip-result.txt), [binary SHA256](hip-binary-sha256.txt).
SYCL evidence: [compiler and host](sycl-metadata.txt),
[configuration](sycl-configure.txt), [initial build](sycl-build-initial.txt),
[two-job build](sycl-build-two-jobs.txt), [completed build](sycl-build.txt),
[result](sycl-result.txt), [binary SHA256](sycl-binary-sha256.txt).
Intel's [package versions and SHA256 values](intel-packages.json) and
[installation log](intel-install.txt) record the extracted vendor packages.

## Checks that ran

The HIP executable loaded and returned success for `--help`:
[output](hip-help.txt), [shared-library resolution](hip-ldd.txt).
The [selected CPU CTest run](hip-cpu-tests.txt) passed the new shared-memory
budget test. The Q8 oracle contract, Q2 portable reference and expert multi-test
returned their configured skip code because this Celeron lacks AVX-512.
They are not counted as numerical parity passes.

The SYCL executable also loaded and returned success for `--help`:
[output](sycl-help.txt), [shared-library resolution](sycl-ldd.txt).
Neither binary's library resolution listed a missing dependency with its
recorded environment.

The Jetson, older-CPU, AMD and golden setup suites ran 62 cases: 61 passed and
one skipped; [log](setup-portability-tests.txt). An initial invocation including
the choices suite reported four errors and one skip across 102 cases;
[log](setup-tests.txt). The unchanged upstream baseline reported the same four
errors and one skip across its 87 cases; [log](setup-baseline-tests.txt).
The choices fixtures assume an AVX2 desktop CPU and omit the older-CPU ISA
floor in their prepared engine metadata. On this Celeron, setup correctly
takes the older-CPU rebuild path, which their fake build does not implement.

With only the choices suite's CPU information mocked to its intended AVX2
desktop fixture, all 40 cases passed; [log](setup-choices-fixture-tests.txt),
[runner](choices-fixture-check.py). CPU detection and the older-CPU tests
were run separately without this override. This fixture check is not a
native AVX2 execution test or a CUDA/HIP runtime check.

## Reproduction and limits

The scripts assume the checkout is `~/workspace/strata-validation/strata`
and the pinned llama.cpp archive is extracted under `dependencies/`.
The [HIP build script](build-hip.sh), [SYCL build script](build-sycl.sh),
[HIP checks](check-hip.sh) and [SYCL checks](check-sycl.sh) preserve the
commands and environment paths. Run them with `bash` from any directory.

Install the HIP SDK into the private environment with AMD's repository pin:

```sh
cd ~/workspace/strata-validation
bin/uv venv --python python3 toolchains/rocm
UV_CACHE_DIR="$PWD/uv-cache" TMPDIR="$PWD/tmp" bin/uv pip install \
  --python toolchains/rocm/bin/python \
  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/ \
  'rocm[libraries,devel]==7.10.0a20251120'
```

The [Intel installation helper](install-intel.py) reads Intel's `binary-amd64`
and `binary-all` package indexes saved as `downloads/intel-Packages.gz` and
`downloads/intel-Packages-all.gz`. They came from
`https://apt.repos.intel.com/oneapi/dists/all/main/`. It resolves the compiler
and MKL dependencies, verifies each package SHA256 from the index, and
extracts them into `toolchains/intel-root` without system installation.
Compare the resulting versions with the recorded package manifest; a later
vendor index may select newer dependency packages.

The successful compilation checks complement the SM87 CUDA build and model
measurements on the Orin. They do not extend those performance measurements
to AMD or Intel hardware, prove desktop inference output byte identity,
or validate ARM hosts with AMD/Intel GPUs. Windows, GPU AOT, other HIP
architectures and HIP/SYCL model execution remain untested in this report.
