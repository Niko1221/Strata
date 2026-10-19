# Strata on Apple Silicon (macOS) - Changes & Configuration Guide

This document records the modifications, fixes, and configuration adjustments applied to Strata for Apple Silicon (specifically validated on Apple M5 Max, 128 GB Unified Memory, macOS 27.0.1 / Sequoia+) up through version `v0.1.45`.

---

## 1. Summary of Changes

During the upgrade to Strata `v0.1.45` and subsequent Apple Silicon optimizations, multiple issues were resolved across compile time, link time, runtime initialization, and web UI chat context stability:

1. **CUDA Runtime Compatibility Shims**: Added missing CUDA API definitions in the Metal compatibility layer to allow shared core host files (`src/core/verify.cpp`, `src/core/peer_experts.cpp`) to compile cleanly with `-DSTRATA_GPU_METAL=1`.
2. **Signature Parity & Prefill Stubs**: Updated function signatures in `metal/src/engine/stubs.cpp` to match `v0.1.45` changes (GEMM BF16 parameters and `gr_*` norm/write functions).
3. **Missing Metal Kernel Stubs**: Added out-of-line stubs for 21 newly introduced kernel and prefill entry points.
4. **Runtime Crash Fix (`fused_gr_persist_init`)**: Resolved engine exit code `86` caused by unhandled HIP-specific initialization calls in `src/core/verify.cpp` on Metal.
5. **Safe Fallback Returns**: Changed boolean capability query stubs (`f16_range_read`, `gemm_f16_range_read`, `qsa_block_scores_rows`, `qsa_decode_attn_split`) to return `false` instead of calling `strata::metal_engine::missing(...)`.
6. **RoPE YaRN Scaling on Metal**: Implemented full RoPE scaling (None, Linear, YaRN) support for the Metal decode window (`cs_`) through `metal/src/engine/rope_table.h`, matching the CUDA reference bit-for-bit.
7. **Include Scope Fix in `setup_host.cpp`**: Resolved a compilation failure where `#include "engine/rope_table.h"` was nested inside `namespace strata::kernels { ... }`, causing standard library templates (`<vector>`, `<functional>`) to be declared under `strata::kernels::std` instead of global `::std`.
8. **Interleaved RoPE Table Optimization**: Added `build_rope_table_interleaved` to directly generate `float2` interleaved `(cos, sin)` pairs into Metal buffer `cs_`, eliminating transient memory allocation and precomputing `inv` frequency factors once per table rather than per entry.
9. **3rd-Round Chat Crash Resolution**: Corrected `strata-iq3_s.json` configuration from bloated SSD streaming parameters to optimal unified-memory resident parameters.

---

## 2. Detailed Code Modifications

### 2.1 CUDA Runtime Compatibility Layer
**File:** [`metal/include/metal_compat/cuda_runtime.h`](file:///Volumes/vl/ai/strata/metal/include/metal_compat/cuda_runtime.h)

- Added device attribute enums:
  - `cudaDevAttrComputeCapabilityMajor = 75`
  - `cudaDevAttrComputeCapabilityMinor = 76`
- Added missing CUDA runtime function declarations and stub implementations:
  - `cudaDeviceDisablePeerAccess(int)`
  - `cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority)`
  - `cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority)`
  - `cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count)`
  - `cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream)`

### 2.2 Host-Side Setup & HIP Persistence Stubs
**File:** [`metal/src/engine/setup_host.cpp`](file:///Volumes/vl/ai/strata/metal/src/engine/setup_host.cpp)

- Added no-op / fallback handlers for `fused_gr` persistent functions called in `Verifier::init()`:
  ```cpp
  void fused_gr_persist_init() {}   // opt-in STRATA_HC_PERSIST=1 (HIP): no persistent kernel on Metal
  bool fused_gr_read_persist(const FusedGrArgs*, int, void*) { return false; }
  ```
- **Include Scope Fix**: Placed `#include "engine/rope_table.h"` at the top file level outside `namespace strata::kernels`. This prevents standard library headers like `<vector>` from being pulled into the `strata::kernels` namespace, avoiding compiler errors where libc++ expects `::std` symbols.

### 2.3 Kernel Stubs & Signature Parity
**File:** [`metal/src/engine/stubs.cpp`](file:///Volumes/vl/ai/strata/metal/src/engine/stubs.cpp)

- **Prefill GEMM Signature**: Updated `Gemm::bf16` to 10 parameters (added `bool x_f16 = false`) and added `bf16_converts_x() const { return false; }`.
- **Image Flag in Norm/Write Operations**: Updated `gr_norm`, `gr_norm_rs`, `gr_write_cvec_norm_rs`, and `gr_write_norm_rs` to accept trailing `bool f16img`.
- **Safe Fallback Returns**:
  - `f16_range_read(...)` -> returns `false`
  - `gemm_f16_range_read(...)` -> returns `false`
  - `qsa_block_scores_rows(bool overload, ...)` -> returns `false`
  - `qsa_decode_attn_split(...)` -> returns `false`
- **New Kernel Stubs**:
  - `wait_handoff`, `hc_q8_to_bf16`, `handoff_publish`, `gdn_step_norm_groups`, `wait_flag_ge_stamped`
  - `qsa_block_scores_rows` (void overload), `native_quantize_q8_1_il`, `wait_copy_i32_from_mapped`
  - `qsa_decode_attn_batch_rows`, `copy_rows_from_mapped_after`, `cpu::iq256_gu_rows_exact_one`, `res_patch`

### 2.4 RoPE YaRN Scaling & Layout on Metal
**Files:** [`metal/src/engine/rope_table.h`](file:///Volumes/vl/ai/strata/metal/src/engine/rope_table.h), [`metal/src/engine/window.cpp`](file:///Volumes/vl/ai/strata/metal/src/engine/window.cpp), [`metal/tests/rope_parity.cpp`](file:///Volumes/vl/ai/strata/metal/tests/rope_parity.cpp)

- **Header-Only Table Builder**: Created `engine/rope_table.h` supporting trained unscaled angles, linear scaling, and YaRN ramp correction with magnitude scaling (`mscale`).
- **Pair Invariant Optimization**: Precomputes `inv = theta^(-2i/n_rot)` once per table rather than per entry, eliminating millions of redundant `std::pow` calls at large contexts (e.g. 524K tokens).
- **Direct Interleaved Generation**: Added `build_rope_table_interleaved` to write directly into `cs_` as `float2 (cos, sin)` pairs without staging planar tables in host memory.
- **Bit-Exact Parity Test**: `metal/tests/rope_parity.cpp` verifies planar unscaled, scaled (None, Linear, YaRN), and interleaved layout against 64-bit reference trig calculations.

---

## 3. Configuration & Chat Stability Fix

### 3.1 3rd-Round Chat Freeze / Out-Of-Memory Diagnosis
On 128 GB Apple Silicon systems, IQ3_S (54.8 GB main shard) fits entirely into unified RAM.

Previously, `strata-iq3_s.json` had accidentally accumulated SSD streaming configurations:
- `"STRATA_METAL_STREAM": "1"`
- `"--expert-cache", "24576"`
- `"--max-context", "131072"`
- `"--spec", "3"`
- `"--prefill", "2048"`

**Root Cause:**
1. Combining SSD streaming with an oversized expert cache (`24576`) consumed excessive memory.
2. Allocating a 131,072-token KV cache on top of this caused the Metal residency set (`MTLResidencySet`) to exceed the OS working set limit (~107 GB) as the multi-turn conversation expanded past turn 2-3, resulting in process termination or deadlock.

### 3.2 Corrected Resident Configuration
**File:** [`strata-iq3_s.json`](file:///Volumes/vl/ai/strata/strata-iq3_s.json)

Settings were restored to the verified optimal profile for 128 GB Apple Silicon:
```json
{
  "exe": "/Volumes/vl/ai/strata/build-metal/strata",
  "args": [
    "--pack", "/Volumes/vl/ai/Strata-data/packs/iq3_s",
    "--native", "/Volumes/vl/ai/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--ple-gguf", "/Volumes/vl/ai/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
    "--expert-profile", "/Volumes/vl/ai/strata/data/expert-profile.bin",
    "--expert-cache", "64",
    "--prefill", "auto",
    "--spec", "2",
    "--mtp", "/Volumes/vl/ai/Strata-data/mtp/rt",
    "--max-context", "65536",
    "--mmap-experts",
    "--vision"
  ],
  "cwd": "/Volumes/vl/ai/strata",
  "tokenizer": "/Volumes/vl/ai/Strata-data/packs/iq3_s/tokenizer",
  "model_name": "qwen3.8-flash-next-iq3_s",
  "port": 8000,
  "backend": "metal",
  "env": {
    "STRATA_METAL_STREAM": "0"
  },
  "vision": {
    "exe": "/Volumes/vl/ai/strata/build-vision/bin/strata-vision",
    "mmproj": "/Volumes/vl/ai/Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
    "model": "/Volumes/vl/ai/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "gpu": true,
    "max_tokens": 1024
  }
}
```

---

## 4. Verification & Testing

### 4.1 Compilation
Build the main engine binary and validation targets:
```bash
.venv/bin/cmake --build build-metal --target strata rope_parity arena_budget_test -j 7
```

### 4.2 Automated Test Suites
Run the test executables and Python verification scripts:

1. **RoPE Parity Test**:
   ```bash
   ./build-metal/rope_parity
   ```
   *Expected: Bit-exact parity on unscaled, linear, YaRN, and interleaved layouts (0 failures).*

2. **Arena Budget Test**:
   ```bash
   ./build-metal/arena_budget_test
   ```
   *Expected: All macOS unified memory and residency set budget tests pass (`all passed`).*

3. **Mac Setup Verification**:
   ```bash
   python tools/test_setup_mac.py
   ```
   *Expected: Ran 21 tests, `OK`.*

4. **RoPE YaRN Derivation & Config Verification**:
   ```bash
   python tools/test_setup_rope.py
   ```
   *Expected: Ran 19 tests, `OK`.*

5. **Setup Update Stamp**:
   ```bash
   ./setup.sh --update --yes
   ```
   *Expected: Step 1-4 pass; `[ok] Metal engine built / current`.*

### 4.3 End-to-End Inference
- Model loaded into resident unified memory (~55 GB mapped).
- Endpoints verified: `/v1/models` and `/v1/chat/completions`.
- Measured performance on Apple M5 Max: ~90 tok/s decode, ~99 tok/s prompt evaluation, 5/5 speculative draft tokens accepted.
