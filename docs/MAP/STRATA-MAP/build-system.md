# Strata map: build system

Source: Strata commit `296b59d2`. Paths are relative to the Strata repo root.
Date: 2026-10-14 America/Chicago.

## 1. Overview

- This area is how Strata becomes a runnable engine and how models become something that engine can load.
- Root `CMakeLists.txt` (project version **0.1.44.1**) plus `cmake/hip_backend.cmake` select mutually exclusive GPU backends (CUDA / wave32 HIP / gfx906-as-CUDA / experimental SYCL early-return / separate Metal project), pull pinned ggml via `STRATA_GGML_DIR` or FetchContent, and emit the `strata` binary and helpers.
- User-facing install is almost entirely `setup.py` (entered via `setup.sh` / `START-HERE.bat` / `update.sh`), which downloads or builds the engine, fetches HF/ModelScope GGUF (or ready-made NVFP4 packs), runs `tools/strata_pack.py` or `tools/iq_pack.py` (+ MTP fetch/pack), writes a `strata-*.json` config, and starts `serve/`.
- Docker/`flake.nix` wrap the CUDA path; `third_party/ggml` only pins the ggml commit header used when llama.cpp is not yet unpacked.

---

## 2. File table

| path | lines | purpose |
|------|------:|---------|
| `CMakeLists.txt` | 1793 | Top-level CMake: backends, targets, tests, ggml/MMQ wiring |
| `cmake/hip_backend.cmake` | 128 | Wave32 HIP discovery, arch validation, CUDA→HIP force-include |
| `setup.sh` | 69 | Linux/macOS: ensure Python/.venv, `exec` `setup.py` |
| `setup.py` | 6859 | Installer: engines, models, packs, config, server start |
| `update.sh` | 63 | `git pull` (history rewrite handling) then `setup.sh --update` |
| `SETUP.bat` | 3 | Windows: `START-HERE.bat --setup` |
| `START-HERE.bat` | 52 | Windows: ensure Python/.venv, run `setup.py` |
| `BENCH.bat` / `bench.sh` | 16 / 10 | Run `tools/strata_bench.py` |
| `Dockerfile` | 119 | NVIDIA CUDA 13.0 image; compile engine at build time |
| `docker-entrypoint.sh` | 106 | Map env → `setup.py --setup` / start |
| `flake.nix` | 49 | NixOS CUDA 13.0 + cmake/ninja shell for `setup.sh` |
| `requirements.txt` | 23 | Pinned pip deps for `.venv` |
| `third_party/ggml/VERSION.txt` | 1 | Pinned ggml commit `3cf03257…` (matches `LLAMA_CPP_COMMIT`) |
| `third_party/ggml/ggml-common.h` | 1911 | Vendored ggml common header |
| `third_party/llama.cpp/` | — | **Not in fresh clone**; `setup.get_llama_cpp()` zips it in |
| `metal/CMakeLists.txt` | 123 | Separate Apple Silicon / Metal project |
| `metal/setup_mac.py` | 706 | Mac installer entry (`setup.py` delegates on Darwin) |
| `sycl/CMakeLists.txt` | 396 | Separate Intel SYCL engine project |
| `sycl/setup_intel.py` | 366 | `--backend sycl` installer |
| `tools/vision/CMakeLists.txt` | 61 | Optional `strata-vision` (llama.cpp mtmd) |
| `tools/vision/strata_vision.cpp` | 239 | Vision encoder binary source |
| `tools/_paths.py` | 28 | Locate `gguf-py` (`STRATA_GGUF_PY` / third_party) |
| `tools/gguf_reader.py` | 142 | Lightweight GGUF header/tensor reader |
| `tools/gguf_writer.py` | 234 | GGUF writer helpers |
| `tools/strata_pack.py` | 452 | Q2_0 canonical pack `build`/`verify`/`info` |
| `tools/pack_layer.py` | 123 | Q2_0 expert blob layout (1,382,400 B/expert) |
| `tools/pack_index.py` | 260 | Flatten pack manifest → `index.txt` for C++ loader |
| `tools/iq_pack.py` | 644 | Native pack for IQ/K-quant/etc. (keeps GGUF quants) |
| `tools/canonical_xcheck.py` | 799 | Spec of record for Q2_0/IQ mappings |
| `tools/strata_tokenizer.py` | 368 | Export tokenizer/ from GGUF into pack |
| `tools/nvfp4_convert.py` | 128 | ModelOpt NVFP4 HF dir → Strata-policy GGUF |
| `tools/nvfp4_verify.py` / `tools/nvfp4_verify_gguf.py` | 121 / 161 | Bit-exact NVFP4 convert checks |
| `tools/embd_bf16_pack.py` | 87 | Checkpoint BF16 `token_embd` → one-tensor GGUF |
| `tools/ple_fp8_pack.py` | 123 | Checkpoint FP8 PLE table → I8 GGUF + metadata |
| `tools/mtp_fetch.py` | 314 | Fetch MTP BF16 tensors from HF/MS |
| `tools/mtp_pack.py` | 193 | MTP BF16 → draft GGUF (q2_0/q4_0/q8_0 experts) |
| `tools/mtp_rt.py` | 111 | MTP GGUF → runtime `rt/` pack for engine `--mtp` |
| `tools/strata_inspect.py` | 182 | Identify GGUF vs setup fingerprints |
| `tools/quant_formats.py` | 154 | Per-role quant summary (local or HF range GET) |
| `tools/quantscope.py` | 394 | Header-only GGUF quant inspection |
| `tools/calibrate.py` | 343 | `--calibrate`: PCIe/spec/pool/adapt tuning |
| `tools/draft_vocab.py` | 146 | Build draft-head token subsets |
| `tools/check_isa_guard.py` | 224 | Disasm guard: no AVX-512 leakage on AVX2 paths |
| `tools/strata_bench.py` | 1028 | Community benchmark harness |
| `tools/strata_mcp.py` | 1771 | MCP server for AI assistants |
| `tools/hip/build_windows.bat` | 87 | Windows HIP release build + TheRock ROCm |
| `tools/hip/package_windows.py` | 173 | Zip Windows HIP engine + ROCm DLL closure |
| `tools/hip/dll_closure.py` | 117 | PE import closure for HIP runtime DLLs |
| `tools/hip/tune_hipblaslt.cpp` | 420 | hipBLASLt solution tuning tool |
| `tools/hip/gfx*-hipblaslt-*.txt` | varies | Per-arch hipBLASLt solution tables |
| `tools/image/*` | various | Optional Qwen-Image generation path |
| `tools/test_*.py` / `*_test.py` | many | Setup/pack/serve regression tests (tests-bench-docs.md) |
| `tools/qwen35/*` | various | Qwen3.6 preview parity helpers |

---

## 3. Where it does X

### Backend selection (CUDA vs HIP vs SYCL vs Metal)

- **File:line:** `CMakeLists.txt:14-22` (SYCL early `return()`), `CMakeLists.txt:58-96` (CUDA/HIP/gfx906 mutual exclusion + `include(cmake/hip_backend.cmake)`), `metal/CMakeLists.txt:1-19` (separate project, macOS only).
- **How:** `STRATA_ENABLE_SYCL=ON` configures only `sycl/` and exits the top-level. Otherwise at most one of CUDA or HIP: `STRATA_HIP_GFX906` forces CUDA-shaped build with HIP aliases (wave64 MI50); `STRATA_ENABLE_HIP` includes `cmake/hip_backend.cmake` for wave32 RDNA. Metal is never wired into the top-level; Mac uses `metal/setup_mac.py`.
- **See also:** engine-core.md (device/runtime), gpu-cpu-kernels.md (`.cu` compiled as CUDA or HIP), intel-sycl-backend.md (`sycl/`), apple-metal-backend.md (`metal/`).

### HIP gfx architecture selection

- **File:line:** `cmake/hip_backend.cmake:4-43` (default `CMAKE_HIP_ARCHITECTURES=gfx1100`; validated `gfx1100`/`gfx1201`; community `gfx1101`/`gfx1200`; unvalidated list incl. `gfx1030`, `gfx1151`, …); `setup.py:3002-3006` (HIP cmake defs with detected archs + `STRATA_PREFILL_MMQ=ON`).
- **How:** CMake fatal-errors unknown arches. Runtime gets `STRATA_HIP_ARCHS="gfx…"`. Setup’s local HIP build always passes `-DSTRATA_PREFILL_MMQ=ON` and the host’s arch list. Windows release script defaults `STRATA_HIP_ARCHS=gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030;gfx1031;gfx1151` (`tools/hip/build_windows.bat:21`).
- **See also:** gpu-cpu-kernels.md (WMMA/`STRATA_WMMA_GFX11`), engine-core.md (`src/core/device.cu` arch check).

### CUDA architecture / older GPUs

- **File:line:** `CMakeLists.txt:148-199` (default `CMAKE_CUDA_ARCHITECTURES=120`; floor sm_75 unless `STRATA_EXPERIMENTAL_SM60`); `Dockerfile:59-84` (fat binary `75;80;86;89;120`).
- **How:** Ready-made / Docker engines are multi-arch CUDA 13. Experimental Pascal/Volta need `-DSTRATA_EXPERIMENTAL_SM60=ON` and CUDA 12.x. RTX 50 + CUDA &lt;13 warns.
- **See also:** gpu-cpu-kernels.md, engine-core.md.

### Feature flags (MMQ, K-quants, native experts, NVFP4)

- **File:line:** `CMakeLists.txt:72-87` (`STRATA_PREFILL_MMQ`, `STRATA_MMQ_KQUANTS` default ON, `STRATA_Q6K_EXPERTS` ON, `STRATA_ORCA_Q4KS_MMQ` OFF); `CMakeLists.txt:107` `STRATA_NATIVE_EXPERTS` ON; HIP MMQ block `CMakeLists.txt:1535-1579` (`STRATA_DENSE_MMQ` / `STRATA_PREFILL_FUSED` defs); NVFP4 parity `CMakeLists.txt:1526-1617`.
- **How:** Prefill MMQ requires HIP + native experts. On CUDA, K-quant MMQ can be a separate DSO `strata_kq` loaded when needed; on HIP, code objects load lazily. Runtime opt-in often via env (e.g. `STRATA_MMQ_KQ=1`). NVFP4 GPU paths are CUDA-oriented in CMake tests; CPU NVFP4 parity always built with native experts.
- **See also:** gpu-cpu-kernels.md, model-spec-ngram-platform.md (quant layouts), prefill.md (prefill).

### User install / start path

- **File:line:** `setup.sh:69` → `setup.py`; `setup.py:5705` (`main` argparse); Docker `docker-entrypoint.sh:76-106`.
- **How:** First run creates `.venv`, pip-installs `requirements.txt`, ensures engine (prebuilt zip or `cmake_build`), downloads model shards or ready packs, step 6 packs + MTP, writes config, starts serve (or `--no-start`). `--backend cuda|hip|sycl`; Darwin delegates to Metal setup.
- **See also:** server.md (`serve/server.py`), model-spec-ngram-platform.md (GGUF/packs), tests-bench-docs.md (many `tools/test_setup_*.py`).

### HF / ModelScope / safetensors → something Strata runs

End-to-end pipeline (concrete tools):

1. **Published GGUF (usual path):** `setup.py` downloads pinned HF revisions (`HF_REVISIONS` @ `setup.py:66`) or ModelScope (`--source modelscope` / `STRATA_SOURCE`) into `Strata-data/models/…`.
2. **Pack:**
   - Q2_0 + AVX-512 host + family qwen: `tools/strata_pack.py build` then `tools/pack_index.py` + `tools/strata_tokenizer.py` (`setup.py:6511-6520`).
   - Else: `tools/iq_pack.py --gguf shard1 --out packs/<tag>` (+ optional `pack_args` e.g. `--compat-bf16` for Unsloth) (`setup.py:6521-6525`).
   - Low-RAM: second `tools/iq_pack.py --experts-bin` if no `experts.bin` (`setup.py:6526-6529`).
   - Ready-made NVFP4 family: downloads prebuilt `pack/` + dense/PLE/embd GGUFs (`NVFP4_REPOS` in `setup.py`); `iq_pack` not used for experts.
3. **MTP draft:** `tools/mtp_fetch.py fetch` → `tools/mtp_pack.py --experts q2_0` → `tools/mtp_rt.py` → `rt/` + `draft_vocab.bin` (`setup.py:6537-6550`).
4. **From ModelOpt / HF safetensors (offline convert, not default setup):**
   - `tools/nvfp4_convert.py` wraps llama.cpp `convert_hf_to_gguf` with Strata tensor policy (BF16 small projs, Q8_0 embd/output, lossless NVFP4 experts; drops PLE + MTP).
   - Optional `tools/embd_bf16_pack.py` / `tools/ple_fp8_pack.py` for bit-exact embd/PLE sidecars; engine flags `--embd-gguf` / `--ple-gguf`.
5. **Engine load:** config points at `--pack`, `--native` GGUF shards, `--mtp`, optional PLE/embd; server.md launches `strata`.

- **See also:** model-spec-ngram-platform.md (formats), gpu-cpu-kernels.md (native experts / NVFP4), server.md, moe-experts.md (export/release tooling), prefill.md.

### ggml / llama.cpp pin

- **File:line:** `setup.py:167-168` `LLAMA_CPP_COMMIT = 3cf03257f219afbe7334045ff7c6a06ac68c627d`; `get_llama_cpp()` `setup.py:1967`; `CMakeLists.txt:1262-1310` `STRATA_GGML_DIR` or fetch; `third_party/ggml/VERSION.txt` same commit.
- **How:** Fresh clone has only `third_party/ggml/VERSION.txt`, `third_party/ggml/ggml-common.h` and `third_party/ggml/LICENSE`. Setup downloads llama.cpp zip into `third_party/llama.cpp` for ggml CMake + gguf-py + mtmd.
- **See also:** gpu-cpu-kernels.md (ggml-cpu IQ), engine-core.md, vision encoder build (`tools/vision`).

### Vision encoder build

- **File:line:** `tools/vision/CMakeLists.txt:16-17` (`STRATA_VISION_CUDA`, `STRATA_PORTABLE` default ON); `Dockerfile:86-88`; HIP path `build_vision_cpu` (`setup.py:3035`).
- **How:** Separate CMake project over pinned llama.cpp with `LLAMA_BUILD_MTMD=ON`. CUDA image builds GPU encoder; HIP uses CPU encoder (`STRATA_VISION_CUDA=OFF`).
- **See also:** server.md (multimodal), model-spec-ngram-platform.md.

### Docker / Nix

- **File:line:** `Dockerfile:42-119`; `docker-entrypoint.sh:19-106`; `flake.nix:18-45`.
- **How:** Image = Ubuntu 24.04 + CUDA 13.0 devel; build-time `STRATA_ENABLE_CUDA=ON`; runtime volume `/data`; env `FAMILY`/`MODEL`/`CONTEXT`/… Entry point runs setup then `setup.py` as PID 1 (`STRATA_EXECV=1`). Flake exposes CUDA 13.0 toolkit + cmake/ninja for NixOS (`setup.sh:18` picks uv automatically on NixOS).
- **See also:** server.md.

---

## 4. CMake options / cache vars / CLI / env

### CMake `option()` / CACHE (top-level + hip_backend + vision/metal/sycl)

| name | default | where | meaning |
|------|---------|-------|---------|
| `STRATA_ENABLE_SYCL` | OFF | `CMakeLists.txt:14` | Experimental Intel Arc; exclusive early return |
| `STRATA_PORTABLE` | OFF | `CMakeLists.txt:34` | AVX2 baseline + static MSVC runtime (prebuilts) |
| `STRATA_ISA_FLOOR` | `""` | `CMakeLists.txt:39` | `avx` / `none` older-CPU ggml floor; exclusive with PORTABLE |
| `STRATA_ENABLE_CUDA` | OFF | `CMakeLists.txt:58` | NVIDIA CUDA targets |
| `STRATA_HIP_GFX906` | OFF | `CMakeLists.txt:64` | gfx906 via HIP as CUDA-shaped build |
| `STRATA_ENABLE_HIP` | OFF | `CMakeLists.txt:71` | Wave32 RDNA HIP |
| `STRATA_PREFILL_MMQ` | OFF | `CMakeLists.txt:72` | HIP GGML MMQ prefill (setup HIP turns ON) |
| `STRATA_MMQ_KQUANTS` | ON | `CMakeLists.txt:79` | Q4_K/Q5_K/Q5_1/Q6_K MMQ instances |
| `STRATA_Q6K_EXPERTS` | ON | `CMakeLists.txt:83` | Q6_K grouped expert kernels |
| `STRATA_ORCA_Q4KS_MMQ` | OFF | `CMakeLists.txt:87` | Q5_0 MMQ for OrcaRouter Q4_K_S |
| `STRATA_BUILD_TESTS` | ON if `tests/`+`bench/micro` exist else OFF | `CMakeLists.txt:102` | Test suite |
| `STRATA_BUILD_CONVERSATION_TESTS` | OFF | `CMakeLists.txt:103` | Conversation cache tests |
| `STRATA_WERROR` | OFF | `CMakeLists.txt:104` | Warnings as errors |
| `STRATA_NATIVE_EXPERTS` | ON | `CMakeLists.txt:107` | ggml-cpu IQ expert support |
| `STRATA_EXPERIMENTAL_SM60` | OFF | `CMakeLists.txt:160` | Pascal/Volta (&lt;sm_75) community CUDA |
| `CMAKE_CUDA_ARCHITECTURES` | `120` if unset | `CMakeLists.txt:161-162` | CUDA arches |
| `CMAKE_HIP_ARCHITECTURES` | `gfx1100` (HIP) / `gfx906` (gfx906 path) | `cmake/hip_backend.cmake:5`, `CMakeLists.txt:130` | HIP arches |
| `STRATA_GFX1012_PORTABLE_DOT` | OFF | `cmake/hip_backend.cmake:90` | Portable signed-byte dot on gfx1012 |
| `STRATA_GGML_DIR` | `""` | `CMakeLists.txt:1263` | External llama.cpp for ggml |
| `STRATA_ORACLE_SOURCE_DIR` / `_BUILD_DIR` | `""` | `CMakeLists.txt:456-457` | Pinned llama.cpp for CUDA oracle tests |
| `STRATA_PLE_FIXTURE_DIR` | `bench/micro` | `CMakeLists.txt:1037` | PLE fixture path |
| `STRATA_SDE` | `""` | `CMakeLists.txt:1742` | Intel SDE for emulated-CPU tests |
| `STRATA_VISION_CUDA` | OFF | `tools/vision/CMakeLists.txt:16` | GPU vision encoder |
| `STRATA_PORTABLE` (vision) | ON | `tools/vision/CMakeLists.txt:17` | Portable AVX2 encoder default |
| `METAL_CPP_DIR` | `""` | `metal/CMakeLists.txt:50` | metal-cpp headers; empty → kernels only |
| `STRATA_SYCL_AOT` | `""` | `sycl/CMakeLists.txt:31` | AOT device (else SPIR-V JIT) |
| `STRATA_SYCL_SPIN_MAX` | `""` | `sycl/CMakeLists.txt:35` | Fixed device spin bound |

Referenced but not declared as top-level `option()` (used in `if()` / comments / runtime): e.g. `STRATA_PARITY_PROMPT_ATTN`, `STRATA_DENSE_MMQ`, `STRATA_PREFILL_NVFP4`, `STRATA_PF_FUSED`, `STRATA_Q2_BITPLANE`, `STRATA_MMQ_KQ`, `STRATA_KQ_DSO` — compile/runtime toggles documented in CMake comments / engine env, not configure-time options with defaults in this file.

### Important `setup.py` CLI (argparse at `setup.py:5709-5824`)

| flag | notes |
|------|-------|
| `--family` | `qwen` / `swift` / `coder` / `unsloth` / `nvfp4` / preview `qwen36` … |
| `--model` | e.g. `Q2_0`, `IQ2_XS`, `IQ3_XXS`, `UD-Q4_K_XL`, `NVFP4`, … |
| `--context`, `--rope-scaling`, `--rope-scale` | context / RoPE |
| `--kv` | `int8` \| `q4_0` \| `k8v4` |
| `--vision` | `yes`/`no`/`cpu`/`gpu`/… |
| `--port`, `--host`, `--api-key` | server bind |
| `--gpu` / `--gpus` / `--layer-split` | device selection |
| `--data-dir`, `--models-dir`, `--gguf-dir` | paths |
| `--setup`, `--no-start`, `--update`, `--yes`, `--build`, `--check` | lifecycle |
| `--source` | `auto`/`huggingface`/`modelscope` |
| `--cuda` | `12`/`13`/`auto` (also env `STRATA_CUDA`) |
| `--backend` | `cuda`/`hip`/`sycl` |
| `--low-ram` | `auto`/`on`/`off`/`resident`/`mmap` |
| `--calibrate`, `--draft-vocab`, `--inspect`, `--prebuilt` | tools |
| `--resident-budget-gib`, `--vram-reserve-mib`, `--parallel`, `--replicas`, `--kv-streaming` | capacity |

### Env vars (build/install surface)

| var | role | seen at |
|-----|------|---------|
| `STRATA_UV` | Prefer `uv` venv | `setup.sh:18` |
| `STRATA_SOURCE` | huggingface vs modelscope | `setup.py:125` |
| `HF_ENDPOINT` | HF mirror | `setup.py:81` |
| `STRATA_CUDA` | Toolkit generation | `setup.py:5785` |
| `STRATA_PREBUILT_URL` | Ready-made engine URL | `setup.py:172` |
| `STRATA_GGUF_PY` | gguf-py path for tools | `tools/_paths.py:16`, `setup.py:6508` |
| `STRATA_DATA` | Docker data root | `docker-entrypoint.sh:19` |
| `STRATA_EXECV` | Server as PID 1 | `Dockerfile:47` |
| `STRATA_CPUINFO` | AVX2 warn test hook | `docker-entrypoint.sh:12` |
| `STRATA_ROCM_VERSION` / `STRATA_ROCM_INDEX` | Windows HIP TheRock pins | `tools/hip/build_windows.bat:22-23` |
| `STRATA_HIP_ARCHS` | Windows HIP fat arch list | `tools/hip/build_windows.bat:21` |
| `ROCM_PATH` / `HIP_PATH` / `HIP_*` | Set during local HIP compile | `setup.py:2993-2994` |
| `CUDA_HOME` / `CUDA_PATH` / `CUDAHOSTCXX` | Nix flake shell | `flake.nix:40-43` |
| Docker: `FAMILY`,`MODEL`,`CONTEXT`,`VISION`,`KV`,`GPU`,`GPUS`,`LAYER_SPLIT`,`LOW_RAM`,`HOST`,`PORT`,`API_KEY`,`GGUF_DIR`,`RESIDENT_BUDGET_GIB`,`KV_STREAMING`,`CONFIG`,`REINSTALL`,`BUILD_VISION`,`CUDA_ARCHITECTURES` | entrypoint / build-args | `Dockerfile:59-60`, `docker-entrypoint.sh:19-35`, `docker-entrypoint.sh:61` |

### Tool CLIs (pack/convert)

| tool | key flags |
|------|-----------|
| `tools/strata_pack.py` | `build --gguf --out [--layers] [--skip-hash] [--force]`; `verify`; `info` |
| `tools/iq_pack.py` | `--gguf --out [--base] [--skip-experts] [--compat-bf16] [--experts-bin]` |
| `tools/nvfp4_convert.py` | `--model <ModelOpt dir> --outfile [--dry-run]` |
| `tools/mtp_pack.py` | `--src --experts q2_0\|q4_0\|q8_0 --out` |
| `tools/mtp_fetch.py` | `fetch` / `verify` `--out` |
| `tools/quant_formats.py` | paths or `--hf REPO REV DIR` |
| `tools/strata_inspect.py` | local path / `hf:` / `ms:` / `--fingerprints` |

---

## 5. Gotchas and constants

1. **Rule-of-backends:** CUDA, HIP, SYCL, Metal are separate worlds; enabling two of CUDA/HIP/SYCL is a `FATAL_ERROR`. Metal/SYCL are sibling CMake projects, not `add_subdirectory` of the main CUDA graph (except SYCL early-return).
2. **`third_party/llama.cpp` absent until setup** — only `third_party/ggml` ships. `tools/nvfp4_convert.py` expects `third_party/llama.cpp` (`tools/nvfp4_convert.py:34`).
3. **HIP default arch is gfx1100** (`cmake/hip_backend.cmake:5`); setup passes the detected archs (`setup.py:3002-3006`), raw cmake keeps the default.
4. **Empty `CMAKE_HIP_FLAGS_<CONFIG>` cache pitfall** (`cmake/hip_backend.cmake:46-56`): a failed first configure can leave Release at `-O0` → absurdly slow decode; cmake clears/restores INIT flags.
5. **Q2_0 pack blob size hardcoded** `1_382_400` bytes (`tools/pack_layer.py:40-49`); H=2560, FF=640, NE=512, Q2_0 QK=64 — architecture contract, not discovered.
6. **PLE table (~28–51 GB) never packed into experts:** `iq_pack.NOT_IN_PACK`; engine reads GGUF via `--ple-gguf`. NVFP4 convert drops PLE shards and points at IQ4_NL or `tools/ple_fp8_pack.py` output.
7. **NVFP4 models are NVIDIA-only in setup metadata** (`nvidia_only: True`, `setup.py:299`, `setup.py:304`).
8. **Docker measures host RAM via `/proc/meminfo`** — a memory-capped container sees the host total, so its low-RAM mode is requested with `LOW_RAM=on` (`docker-entrypoint.sh:51-53`).
9. **Repository history rewrite 2026-10-06** — `update.sh` special-cases clones that cannot fast-forward (`update.sh:17-46`).
10. **Python pack is the spec** (`tools/strata_pack.py` header): C++ `strata-pack` must match Python; verify is mandatory for Q2_0 canonical packs.
11. **Ready-made engine vs source hash:** `engine/BUILD.json` `src` fingerprint; source change triggers rebuild. `MIN_ENGINE = (0, 1, 44, 1)` (`setup.py:196`) matches `project(VERSION 0.1.44.1)`.
12. **ROCm pins differ Linux vs Windows:** `ROCM_VERSION_DEFAULT = "7.10.0a20251120"` at `setup.py:2086` vs Windows TheRock `10.2.0a20260930` at `tools/hip/build_windows.bat:22`.
13. **Size:** `setup.py` is ~6859 lines / 421 KB — behavior lives there more than in shell wrappers.

---

## Cross-area index (quick)

| Area | Touchpoints from build-system.md |
|------|------------------------|
| engine-core.md | `strata` target, device/load mains, `BUILD.json` runtime |
| gpu-cpu-kernels.md | `.cu` via CUDA/HIP language; MMQ/NVFP4/IQ libraries |
| prefill.md | `strata_prefill`, `STRATA_PREFILL_MMQ`, fused MoE |
| moe-experts.md | prebuilt zips; `tools/hip/package_windows.py`, release URL |
| server.md | setup → `serve/`; Docker port 8080 (`docker-entrypoint.sh:25`) |
| intel-sycl-backend.md | `sycl/`, `--backend sycl` |
| apple-metal-backend.md | `metal/`, Darwin `metal/setup_mac.py` |
| model-spec-ngram-platform.md | packs, GGUF, NVFP4 convert, MTP, fingerprints |
| tests-bench-docs.md | `tools/test_setup_*.py`, CMake `STRATA_BUILD_TESTS` |
