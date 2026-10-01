# Adding NVIDIA V100 (sm_70) Support to Strata

## Overview

The NVIDIA V100 (compute capability 7.0, sm_70) is a Volta-architecture GPU with 16 or 32 GB HBM2, 80 SMs, and
96 KB of shared memory per SM. Strata currently requires compute capability 7.5 (Turing) or newer, and this
requirement is written down in four places (two hard gates, one installer gate, and one generated manifest).

The good news is that the sm_70 path is close to already working: the fallback kernels below sm_80 were written for
the "experimental sm_75 build" and are guarded on `__CUDA_ARCH__ < 800` / `cc_major < 8`, which sm_70 satisfies.
The work is therefore mostly **lowering the guards from 75 to 70** and **building the engine locally** (there is
no ready-made V100 binary, and `engine/BUILD.json` is a generated file that must not be hand-edited).

## Gate 1 — CMakeLists.txt (build-time)

**File:** `CMakeLists.txt`, lines 76-91 (verified)

```cmake
    # Turing port: developed and measured on sm_120 (RTX 50); the tf32 mma in the QSA scorer has a
    # portable fp32-FMA fallback, so RTX 20 (75), 30 (86), RTX 40 (89) and RTX 50 (120) build too.
    # Pre-sm_75 is refused.
    ...
    if(_base LESS 75 AND NOT STRATA_EXPERIMENTAL_SM60)
      message(FATAL_ERROR
        "Strata needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20 / 30 / 40 / 50 series), but "
        "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.  (A Pascal build is "
        "-DSTRATA_EXPERIMENTAL_SM60=ON; upstream does not support it.)")
    endif()
```

**Change:** lower `LESS 75` to `LESS 70` (line 86) and update the two comments (lines 76, 78) and the message
(line 88) to name the V100. Note `_base` is stripped of the `-real`/`-virtual` suffix first, so `70` and
`70-virtual` both pass.

## Gate 2 — src/core/device.cu (run-time)

**File:** `src/core/device.cu`, lines 139-144 (verified; comment at 129-134)

```cpp
    if (d.cc_major * 10 + d.cc_minor < 75) {
        throw CudaError("device " + d.name + " reports compute capability " + std::to_string(d.cc_major) +
                            "." + std::to_string(d.cc_minor) +
                            "; Strata needs compute capability 7.5 or newer (RTX 20 / 30 / 40 / 50 series)",
                        -1);
    }
```

**Change:** lower `< 75` to `< 70` (line 73) and update the comment (line 63) and message (line 76). This is the
gate that matters even for a locally built binary: a binary carrying sm_70 code can still be started on a Pascal
card, and this check is what keeps that from silently taking a wrong path.

## Gate 3 — setup.py (installer)

**File:** `setup.py`, lines 337-341 and 403, 438 (verified)

```python
def gpu_problem(g, together=False):
    """Why Strata cannot use this card, in plain words (None: it can)."""
    if int(g["arch"]) < 75:
        return (f"not supported - older than the RTX 20 series (compute capability {cc(g)}; Strata needs 7.5 or "
                "newer)")
```

```python
        fail("none of your GPUs can run Strata", "it needs an NVIDIA RTX 20 series or newer (compute capability 7.5+)")
```

**Change:** lower `< 75` to `< 70` (line 339) and update the three messages (lines 340, 403 and 438). Without
this, setup reports the V100 as unsupported and never reaches the compile step, even after Gates 1 and 2 are
lowered.

## Gate 4 — engine/BUILD.json (do NOT hand-edit)

**This is a generated file.** Both `build_engine()` (`setup.py:1135`) and the HIP builder (`setup.py:849`) write
it, and the downloaded prebuilt carries its own copy. Its `archs` list is produced by the release tooling for the
prebuilt zip (`tools/make_release.py`, referenced at `setup.py:56` — **not present in this source tree**).

Consequences:

- Editing the local `engine/BUILD.json` by hand is not part of the procedure; the next build overwrites it.
- A prebuilt sm_70 asset can only exist if the release tooling is changed to build one. Until then, V100 users get
  the **compile-locally** path, which already works once Gates 1-3 are lowered: `get_prebuilt()` sees the engine has
  no code for arch 70 and falls through to `build_engine()`, which compiles for `gpu["arch"]`.

The existing `docs/AMD_HIP.md` documents exactly this model for an unsupported GPU (opt-in source build, compiled
on the user's PC, no prebuilt). A V100 should follow the same shape.

## Kernel Analysis

### Which kernels actually have an sm_80 path

Only three translation units in `src/kernels/cuda/` use sm_80-only instructions (`cp.async`, TF32 `mma`):

| File | sm_80-only feature | Fallback selected by |
|------|--------------------|----------------------|
| `qsa_select.cu` | `cvt.rna.tf32.f32` + TF32 `mma.sync` | host `cc_major < 8` → warp kernel |
| `qsa_prompt_attn.cu` | FP16 `mma.sync.m16n8k16` + `cp.async` | host `cc_major < 8` → decode kernel |
| `native_qsa_score.cu` | `ldmatrix` + TF32 `mma.sync` | compile-time `__CUDA_ARCH__ < 800` |

(Note: `ldmatrix` itself is sm_75+, not sm_80+ — the guard in this file is `>= 800` because the *combination* with
TF32 `mma` needs Ampere, not because `ldmatrix` is missing on Turing.)

**`native_qsa_score` is not called by the engine.** A tree-wide search finds no caller in `src/core`,
`src/prefill`, or `src/program`; it is referenced only by `tests/hip/native_qsa_score.cpp`. Its sm_70 behaviour is
therefore not on the critical path, though the guard `__CUDA_ARCH__ < 800` already routes sm_70 to its FP32 FMA
loop (using only `fmaf`/`fmaxf`, both available on sm_70).

### The two live kernels fall back correctly

Both live kernels are dispatched with a "try the tensor-core version, else the old one" pattern:

`src/prefill/prefill.cpp:1377`
```cpp
if (old_sel || !strata::kernels::qsa_block_scores_tc(...))
    strata::kernels::qsa_block_scores(...);   // warp kernel
```

`src/prefill/prefill.cpp:1478`
```cpp
if (old_attn || !strata::kernels::qsa_prompt_attn_batch(...))
    for (...) strata::kernels::qsa_decode_attn_batch(...);   // decode kernel
```

`qsa_block_scores_tc` (`qsa_select.cu:506`) and `qsa_prompt_attn_batch` (`qsa_prompt_attn.cu:677`) both begin with
`if (cc_major[dev] < 8) return false;`. On a V100 (`cc_major == 7`) they return false and the caller runs the
fallback. **No kernel source changes are needed** — only the four guards above.

### Shared memory

`fused_gr.cu` (line 334) already queries `cudaDevAttrMaxSharedMemoryPerBlockOptin` per device and sizes its
token chunk from the answer, with a comment that explicitly anticipates a "Turing: 64 KB" card. A V100 reports
96 KB, so it simply gets a larger chunk. **No change needed.**

Other kernels that opt into large dynamic shared memory (`qsa.cu:675`, `qsa_select.cu`) query the same attribute or
call `cudaFuncSetAttribute` and handle failure, so they degrade rather than fail.

## Architecture Comparison

| | V100 (sm_70) | Turing (sm_75) | Ampere (sm_80) |
|---|---|---|---|
| FP16 `mma.sync` | Yes | Yes | Yes |
| `ldmatrix` | No | **Yes** | Yes |
| `cp.async` | No | No | Yes |
| TF32 `mma.sync` | No | No | Yes |
| Shared memory / SM | 96 KB | 64 KB | 100 KB |
| Max warps / SM | 64 | 32 | 64 |
| Max blocks / SM | 32 | 16 | 32 |
| SMs (full die) | 80 | 68 (2080 Ti) | 82 (3090) |
| Memory | 16/32 GB HBM2 | 11 GB GDDR6 | 24 GB GDDR6X |
| Bandwidth | 900 GB/s | 616 GB/s | 936 GB/s |

The V100 has **more** shared memory and **more** resident warps per SM than the Turing card this codebase was
already made to run on. Its disadvantage is on the instruction side: no `cp.async` and no TF32 path, so the QSA
select and prompt-attention kernels will run their older fallbacks, and the prompt path will be slower than on any
sm_80+ card.

## Summary of Changes

| File | Lines | Change | Kind |
|------|-------|--------|------|
| `CMakeLists.txt` | 76, 78, 86, 88 | `LESS 75` → `LESS 70`, comments/message | hard gate |
| `src/core/device.cu` | 129-130, 139, 142 | `< 75` → `< 70`, comment/message | hard gate |
| `setup.py` | 339, 340, 403, 438 | `< 75` → `< 70`, messages | installer gate |
| prebuilt release tooling | — | add sm_70 to the prebuilt archs | optional (else local build) |
| kernels | — | none | — |

## Testing

1. Configure with `-DCMAKE_CUDA_ARCHITECTURES=70`; the CMake gate must pass.
2. `./setup.sh --yes --vision no`; `gpu_problem()` must accept the card and setup must reach the compile step.
3. Confirm the engine built with `archs: [70]` in `engine/BUILD.json` (generated, not edited).
4. Run `strata-device --selftest` (the `cuda_device_selftest` case) on the V100; the run-time gate must pass.
5. Run the CUDA parity tests (`qsa_parity`, `kv_q8_parity`, `kv_stream_parity`, `sampler_parity`, …). The QSA
   tests are the ones worth watching: they exercise the fallback paths that the V100 will take by default.
6. Benchmark prompt and decode against an sm_75 card to quantify the missing `cp.async`/TF32 cost.

## Risks and Open Questions

- **Performance, not correctness, is the main risk.** The fallbacks are already exercised on Turing, but the V100
  has twice the resident warps per SM; register/shared-memory pressure in the fallback kernels should be checked
  on real hardware rather than assumed.
- **The `cc_major < 8` checks are coarse.** They route *all* sm_7x to the fallback, which is correct today but
  means an eventual sm_75 tensor-core path (possible with FP16 `mma`, which Turing has) will need a finer guard.
- **Numerics differ between paths by design.** `native_qsa_score` says so in its own comment ("rounded
  differently from the tensor-core path"). Same-rig answers may not be bit-identical across cards, which the
  bench notes already document for multi-GPU runs.
- **A prebuilt V100 engine is a release decision**, not a source change, and depends on tooling absent from this
  tree.
