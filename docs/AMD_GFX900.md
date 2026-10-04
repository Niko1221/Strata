# AMD Vega 10 (gfx900): experimental Linux setup

This is an opt-in port for discrete Vega 10 cards, such as the Radeon Instinct MI25
and Vega 56 / 64. It uses the existing wave64 compatibility backend, with software
signed byte dot products because gfx900 has no packed dot instruction. It is
separate from the normal wave32 AMD build and from gfx906.

## Install

Use Linux with the kernel's amdgpu driver, access to `/dev/kfd` and the GPU's
`/dev/dri/renderD*`, a C++ compiler and git. Check RAM and disk as described in
[AI_SETUP.md](AI_SETUP.md). The example below downloads about 68 GB. Keep extra
VRAM free when the GPU also drives the desktop.

```sh
STRATA_EXPERIMENTAL_GFX900=1 ./setup.sh --check --backend hip
STRATA_EXPERIMENTAL_GFX900=1 ./setup.sh --yes --backend hip \
  --family qwen --model IQ2_XS --context 8192 --vram-reserve-mib 3072 \
  --vision no --no-start
./run-iq2_xs.sh
```

Setup installs AMD's TheRock `gfx900` wheels at `7.14.0a20260612` inside `.venv`
and builds into `build-gfx900`. The existing normal AMD pin is unchanged.
`STRATA_ROCM_VERSION` and `STRATA_ROCM_INDEX` override the experimental packages;
a suitable ROCm 7 or newer installation can also be selected with `ROCM_PATH`.
The runtime and BLAS libraries must actually contain gfx900 support: a compiler
accepting `gfx900` alone is not sufficient. No GPU architecture override is needed.
Setup runs the device self-test before starting the model download.

The generated run script starts the local API at `http://127.0.0.1:8080`.
Use the same opt-in environment variable when rerunning setup or updating this
experimental installation. Without it, a new installation continues to reject
gfx900. Windows and builds mixing gfx900 with the wave32 architectures are refused.
Images and multiple-card inference are unvalidated.

## Build and check without a model

With a gfx900-capable ROCm development installation on the compiler and library
paths:

```sh
cmake -S . -B build-gfx900 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP_GFX900=ON -DCMAKE_HIP_ARCHITECTURES=gfx900
cmake --build build-gfx900 --target strata strata-device \
  hip_wave64_intrinsics hip_prefill_gemm -j8
ctest --test-dir build-gfx900 --output-on-failure \
  -R 'hip_wave64|hip_prefill_gemm'
```

`STRATA_HIP_GFX900` selects the existing `STRATA_HIP_GFX906` code internally; leave
the latter option and `STRATA_ENABLE_HIP` off. The engine checks the compiled
architecture and wave size before launching on a GPU.

The intrinsic check exercises both logical 32-lane halves of a physical wave64:
signed dot products with overflow, shuffles and ballots, shared-memory exchange,
and byte permutation. The GEMM check compares BF16/FP16 products with a CPU
reference and checks output padding and accumulation.

## Hardware checks

Checked on 2026-10-04 on a Vega 10 (`gfx900`, PCI `1002:6860`, 16 GB HBM;
ROCm names it AMD Radeon Instinct MI25), Xeon W-2191B (18 cores, AVX2/AVX-512),
125 GiB system RAM, Omarchy/Arch Linux, the existing amdgpu driver and the pinned
TheRock packages above. The engine builds and the device, wave64 intrinsic and
BF16/FP16 GEMM checks pass.

The existing engine CTest suite passes 40 of 42 checks on this PC. `ple_parity`
needs an external Q2_0 model fixture; `platform_memory_test` locks 256 MiB,
above the account's 8 MiB memory-lock limit. Neither check was reported as a pass.
The numerical expert, attention, routing, sampler and KV checks pass. Setup's
gfx900, choices, config, older-GPU, dependency-pin and update suites pass 94 tests.
All 26 ROCm SDK installation checks pass with the compiler and runtime paths set
as in the installer.

The complete installer also downloaded and verified both Qwen3.8-Flash-Next
IQ2_XS shards against the pinned Hugging Face SHA-256 hashes, packed the MTP
head, and started the local server. The tested configuration uses an 8192-token
context, a 3072 MiB VRAM reserve, no vision, MTP with up to four draft tokens,
and 17 AVX2 expert workers. The server reports a 33.02 GiB host arena and 6137
GPU expert-cache slots (8.23 GiB).

End-to-end checks pass for health and model discovery, the browser UI, arithmetic
and Python-code responses, repeated-prompt reuse, streaming through its final
`[DONE]` marker, Anthropic Messages, OpenAI Responses, and monitoring metrics.
These are functional checks on one PC, not a throughput benchmark or validation
of every model, context size or GPU listed above.
