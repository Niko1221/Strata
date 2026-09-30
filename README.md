<h1 align="center">Strata for Radeon RX 9070</h1>

<p align="center"><b>HIP engine for the RX 9070 series (gfx1201), on Windows with ROCm 10</b><br>
Also builds for RX 9060 (gfx1200) and RX 7900 XT/XTX (gfx1100) · fork of <a href="https://github.com/Niko1221/Strata">Niko1221/Strata</a></p>

This branch is the Radeon path. It compiles Strata's wave32 HIP backend and runs
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) on an AMD GPU.
The NVIDIA one-click installer and its published speeds belong to
[upstream](https://github.com/Niko1221/Strata). They are not the results of this tree.

A tested Windows engine for gfx1201 is attached to
[win-gfx1201-2026-09-30](https://github.com/jagsan-cyber/Strata/releases/tag/win-gfx1201-2026-09-30).
`strata.exe` still needs a ROCm 10 runtime on `PATH`. It does not include the model.

## Cards

| Card | Architecture | In this branch |
| --- | --- | --- |
| Radeon RX 9070 / 9070 XT | gfx1201 | built and run on Windows, ROCm 10 |
| Radeon RX 9060 | gfx1200 | same wave32 backend, not measured here |
| Radeon RX 7900 XT / XTX | gfx1100 | same backend; upstream's Linux numbers are in [AMD_HIP_PERFORMANCE.md](docs/AMD_HIP_PERFORMANCE.md) |

One GPU. No image input on this backend. gfx1201 has no hipBLASLt tuning table, so dense prefill uses hipBLAS.

## Measured on an RX 9070

Windows 11, RX 9070 16 GB, Core i7-12700 (AVX2, no AVX-512), 96 GB RAM, ROCm 10.0.0.
Model: Unsloth Qwen3.8-Flash-Next UD-Q3_K_XL. MTP on (`--spec 4`, `--spec-min-p 0.5`).
Context 131072. KV is FP16 and stays on the GPU, so the expert cache is smaller than at 8K.

| | RX 9070, this pack |
| --- | ---: |
| Long reply, 128K window | 19.6 tokens/s (28,377 tokens) |
| Short reply | about 20–29 tokens/s |
| Read a 4,445-token prompt | 238 tokens/s |
| MTP drafts kept | 73% (18,272 of 24,916) |
| Experts on the GPU at 128K | 2,357 of 24,576 (4.97 GiB) |
| Expert-cache hits on that reply | 70% |
| VRAM free after load | 498 MiB |
| Experts in RAM | 52 GiB |

At 8K context the same card held 3,914 experts (8.25 GiB). The drop at 128K is the FP16 KV cache, not a missing kernel.

## What this branch adds

- Windows host build with ROCm 10's clang. CMake will not mix MSVC with Clang HIP. Setup prefers a ROCm tree that contains the card's bitcode (`rocm-sdk`, then `ROCM_PATH`, `HIP_PATH`, then `C:\Program Files\AMD\ROCm`).
- Wave32 targets gfx1100, gfx1200, and gfx1201.
- Unsloth's 3-shard GGUF: gate, up, and down of one layer may sit in different shards. Q8_0 expert rows and the Q8_0 embedding have a GPU dot.
- An F32 `ple_conv1d` is converted to FP16 at startup. The conv kernels read FP16.
- After a successful hipBLAS GEMM on gfx1201, a leftover invalid-argument error is cleared so a long prompt does not abort the process.

Build and pack notes: [docs/AMD_HIP.md](docs/AMD_HIP.md).

## Build

ROCm 10 with the gfx1201 bitcode, CMake, Ninja, and git. On Windows the C and C++ compilers are ROCm's `clang.exe` and `clang++.exe`, not `cl.exe`.

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++.exe"
cmake --build build-hip --target strata
```

`$ROCM` is the ROCm 10 tree (the `rocm-sdk` devel prefix, or `ROCM_PATH`). Put that tree's `bin`, `lib`, and `lib/llvm/bin` on `PATH` before starting the engine.

## Run

Serve the pack you built. The engine used for the table above was started with:

- `--pack` the native pack, `--native` and `--ple-gguf` pointing at the weight shard
- `--expert-cache auto`, `--prefill auto`, `--max-context 131072`
- `--mtp` the packed draft runtime, `--spec 4`, `--spec-min-p 0.5`

KV was left at the default FP16, fully resident. `--kv int8` and `--kv-resident 32768` are the upstream knobs that give the expert cache its VRAM back at 64K and above. They were not on for the numbers in the table.

The browser API is OpenAI-compatible (`/v1/chat/completions`). This machine's server listens on port 8095. Upstream's installer uses 8080.

## Upstream

[Niko1221/Strata](https://github.com/Niko1221/Strata) is the project this fork is taken from: the NVIDIA installer, the model menu, and the RTX 5070 speed tables. Engine behavior that this branch does not change is documented in [docs/DETAILS.md](docs/DETAILS.md).

## License

[MIT](LICENSE). `third_party/ggml` is MIT (llama.cpp / ggml). Model files are not in this repository; each model's own license applies.
