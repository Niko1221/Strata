# Experimental gfx906 build on Windows

This is the separate `STRATA_HIP_GFX906` wave64 backend, built from source. It does
not add MI50 to the ready-made Windows engine or to setup's supported-card list.
The Windows compatibility work uses native HIP, hipBLAS and rocBLAS, without ZLUDA.
MI60 and Radeon VII have not been tested with this Windows recipe.

## Toolchain

The tested stack is Windows 11, one Instinct MI50 16 GiB (reported by the driver as
AMD Radeon Pro VII, `gfx906`, wave64), AMD Software PRO 26.Q1, and the HIP 5.7 SDK
from AMD's 23.Q4 Windows HIP installer. The HIP compiler is the SDK's clang 17;
the ordinary C/C++ compiler is clang 23 using the MSVC 14.44 headers and libraries.
The build emits code object V5 and uses the 5.7 rocBLAS library with gfx906 kernels.
A newer Windows ROCm package without gfx906 rocBLAS kernels is not a replacement.

Use a private copy of the SDK. The tested prefix required these packaging repairs;
this is not a claim that an untouched SDK installation builds:

- Put the SDK's `5.7/amdgcn` directory at `<prefix>/amdgcn`, where its compiler
  expects the device bitcode.
- In `lib/cmake/hip-lang/hip-lang-targets.cmake`, replace the hard-coded AMD build
  machine `_IMPORT_PREFIX` with a path computed from the installed file:
  `get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)`.
- In `include/hip/amd_detail/amd_hip_bf16.h`, add `inline` after `__HOST_DEVICE__`
  to the six free functions `__float2bfloat16`, `__bfloat1622float2`,
  `__double2bfloat16`, `__float22bfloat162_rn`, `__high2float`, and `__low2float`.
  In this SDK they otherwise produce duplicate host definitions at link time.

Keep those repairs in the private prefix; do not overwrite the system SDK or
driver. The source patch itself changes no vendor headers.

## Build

Run in an x64 Visual Studio developer PowerShell with CMake, Ninja and Git on PATH.
Adjust the two paths below to the private HIP 5.7 SDK and the host clang installation.
`STRATA_PORTABLE=ON` uses AVX2, so the host CPU must support it.

```powershell
$sdk = 'C:/HIP57-private'
$hostLlvm = 'C:/host-llvm'
$env:HIP_PLATFORM = 'amd'
$env:HIP_PATH = $sdk
$env:ROCM_PATH = $sdk
$env:HIP_DEVICE_LIB_PATH = "$sdk/amdgcn/bitcode"
$env:PATH = "$sdk/bin;" + $env:PATH

$deviceLibs = @(
  'hip.bc', 'ocml.bc', 'ockl.bc', 'oclc_daz_opt_off.bc',
  'oclc_unsafe_math_off.bc', 'oclc_finite_only_off.bc',
  'oclc_correctly_rounded_sqrt_on.bc', 'oclc_wavefrontsize64_on.bc',
  'oclc_isa_version_906.bc', 'oclc_abi_version_500.bc'
) | ForEach-Object { "--hip-device-lib=$_" }
$hipFlags = "--rocm-path=$sdk --rocm-device-lib-path=$sdk/amdgcn/bitcode " +
  ($deviceLibs -join ' ') + ' -mcode-object-version=5 -D__AMDGCN_WAVEFRONT_SIZE=64 -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'

cmake -S . -B build-906-win -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 `
  -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_HIP=OFF `
  -DSTRATA_PORTABLE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON `
  -DCMAKE_HIP_ARCHITECTURES=gfx906 `
  "-DCMAKE_C_COMPILER=$hostLlvm/bin/clang.exe" `
  "-DCMAKE_CXX_COMPILER=$hostLlvm/bin/clang++.exe" `
  "-DCMAKE_HIP_COMPILER=$sdk/bin/clang++.exe" `
  "-DCMAKE_HIP_COMPILER_ROCM_ROOT=$sdk" `
  "-DCMAKE_HIP_COMPILER_ID_FLAGS=--rocm-path=$sdk --rocm-device-lib-path=$sdk/amdgcn/bitcode -mcode-object-version=5" `
  "-DCMAKE_PREFIX_PATH=$sdk" "-DCMAKE_HIP_FLAGS=$hipFlags"

cmake --build build-906-win --parallel 2 --target strata strata-device `
  hip_gfx906_mapped_alias hip_gfx906_gemm_f16_io_parity `
  iq_multi_parity mmvq_multi_parity sampler_parity
```

CMake fetches the pinned llama.cpp revision for ggml. An existing checkout of that
exact revision can instead be supplied with `-DSTRATA_GGML_DIR=<path>`.
The legacy HIP compiler uses C++17 for `remote_expert_opt.cu` only, to avoid a
clang 17 / recent MSVC STL C++20 parsing failure. Other sources keep their standard.

## Runtime and checks

Stage the SDK's matching runtime DLLs and `rocblas/library` directory together in
a private runtime folder, called `C:/HIP57-runtime` in the commands below. The
executable needs these DLLs on PATH: `amdhip64.dll`, `hipblas.dll`,
`rocblas.dll`, and the COMGR DLLs supplied by that SDK. Set
`ROCBLAS_TENSILE_LIBPATH` to its `rocblas/library` directory. Do not mix this stack
with another engine's DLL directory. Choose the MI50's actual HIP ordinal with
`HIP_VISIBLE_DEVICES`; the compat build rejects a wave32 card before launching
its kernels. No `HSA_OVERRIDE_GFX_VERSION` was used in this Windows test.

```powershell
$runtime = 'C:/HIP57-runtime'
$env:PATH = "$runtime;" + $env:PATH
$env:ROCBLAS_TENSILE_LIBPATH = "$runtime/rocblas/library"
$env:HIP_VISIBLE_DEVICES = '0' # confirm this ordinal on this machine
./build-906-win/strata-device.exe --list-devices
ctest --test-dir build-906-win --output-on-failure -R `
  '^(hip_gfx906_.*|iq_multi_parity|mmvq_multi_parity|sampler_parity)$'
```

`hip_gfx906_gemm_f16_io_parity` checks FP16 and BF16-weight GEMMs against a CPU
reference, output padding, the FP32 beta=1 path, and repeated BF16/FP16 activation
symbol toggles. The activation flag has external device linkage on Windows gfx906
because an anonymous-namespace symbol was missing from the legacy PAL symbol table.

`hip_gfx906_mapped_alias` is a diagnostic: Windows can return the host pointer as
its mapped device pointer. The printed `DOES NOT HOLD` for a CUDA/UVA assumption
is not itself a failed GPU read. It skips with return code 77 when no device exists.

**Checked on 2026-10-09:** a Release build from upstream base `fb58e0d`, with ggml
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, built the engine and all seven targets
in the command above. On the MI50, device enumeration, the device arena selftest,
the mapped kernel read, GEMM / activation-symbol parity, IQ multi parity, MMVQ
multi parity and the sampler selftest all passed. The RX 6800 negative check
returned code 1 with `this engine targets gfx906 wave64` before its arena test.
The five tests selected by the CTest command were confirmed registered; GPU checks
were run directly with the newly built executables and the matched runtime.

For GEMMs with beta=0 the relative L2 error was 0.000206-0.000217; beta=1 was
3.87e-7, within the existing test tolerances. IQ and sampler reported zero failures;
MMVQ compared 148,480 outputs with zero bitwise differences. The engine imported
`amdhip64.dll` and `hipblas.dll`, and its embedded gfx906 code objects used V5.
No end-to-end model inference or speed measurement of this new binary is included.

## Limits

This is a compatibility change, not a new performance kernel or a rocBLAS tuning
table change. It makes no speedup claim. No vision, second GPU, MI60 or Radeon VII
validation is included. A successful synthetic probe is not an end-to-end model
quality test. Linux gfx906, CUDA, wave32 HIP and SYCL still need their own builds
before this change is ready for review.

Windows gfx906 uses the existing physical-RAM fallback for the sliced pinned-memory
cap: HIP 5.7 device properties do not expose the DXGI LUID used by the CUDA path.
The compatibility source changes do not alter the default CUDA or wave32 HIP path.
Removing includes and preprocessing the four changed core/prefill translation
units under the CUDA and wave32 HIP defines produced unchanged conditional tokens
against the base commit. This limited source check is not a binary identity check
or a substitute for building those backends.
