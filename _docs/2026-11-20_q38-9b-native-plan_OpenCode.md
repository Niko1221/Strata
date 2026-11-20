# Q38 dense 9B geometry — implementation plan (2026-11-20, OpenCode)

## Overview

Extend the Strata-owned dense CUDA runner (`strata-q38.exe`) so that a
32-layer `qwen35` dense GGUF (the local 9B checkpoint) runs on the same
native path as the validated 27B checkpoint. The goal is a local, general
runner that dispatches on GGUF metadata and tensor geometry, never on the
model file name, and that uses the existing CUDA 13.5 / SM120 Tensor Core
prefill path. MTP geometry is included from the start.

## Background / requirements

User request: build a Qwen3.8-9B-class native engine by applying the
Qwen3.8-27B native engine; local and general-purpose; faster than
llama.cpp with CUDA 13.5 Tensor Cores; full implementation and
verification this session; include MTP from the start.

Prior evidence: `_docs/2026-11-19_Strata-Qwen-CUDA_総合引き継ぎ_OpenCode.md`
section 5 and `2026-11-19_q38-gguf-coverage_OpenCode.md` define this exact
task and warn that merely relaxing the 64-layer guard is incorrect.

## Assumptions / decisions

The 9B GGUF was inspected read-only with `tools/gguf_reader.py`
(`H:/Strata/models/新しいフォルダー/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf`,
5,629,111,584 bytes, 427 tensors, types F32 177 / Q4_K 217 / Q6_K 33).

| Property | 27B baseline | 9B target |
|---|---|---|
| `general.architecture` | `qwen35` | `qwen35` (identical) |
| `qwen35.block_count` | 64 + nextn | 32 + nextn |
| `qwen35.nextn_predict_layers` | 0 or 1 | 1 (declared) |
| MTP tensors present | 15 (`blk.64.nextn.*`, soya fixture) | **none** (32 layers only) |
| `embedding_length` | 5120 | 4096 |
| `feed_forward_length` | 17408 | 12288 |
| vocabulary | 248320 | 248320 |
| Q / KV heads | 24 / 4 | 16 / 4 |
| head dim | 256 | 256 |
| attention layers | 16 | 8 |
| GDN layers | 48 | 24 |
| GDN qkv channels | 10240 = (2*16 + 2*24)*128 | 8192 = (2*16 + 2*16)*128 |
| GDN gate/out inner | 6144 = 24*256 | 4096 = 16*256 |
| `ssm.time_step_rank` | 48 | 32 |
| `ssm.inner_size` | 6144 | 4096 |
| `ssm.state_size` / group / conv | 128 / 16 / 4 | 128 / 16 / 4 |
| `full_attention_interval` | 4 | 4 |
| RoPE sections / base / eps / EOS | 11,11,10,0 / 1e7 / 1e-6 / 248046 | identical |
| Hadamard transform metadata | optional (Hikari fixture) | absent |
| Matrix types | mixed IQ/PQ2_0/Q4_K | F32 / Q4_K / Q6_K |

Decisions:
1. Add a `Geometry` descriptor derived from GGUF metadata. Keep a named
   27B profile whose values equal today's literals; add a named 9B profile.
   Selection is by the metadata/tensor shape fingerprint, not by `general.name`
   or file path.
2. Replace dimensional literals in `admission.hpp`, `engine.cu`, `kv.*` and
   friends with `Geometry` fields. The 27B execution path must remain
   behaviorally identical (same `--inspect`, same numerical tests).
3. Keep MTP support geometry-driven: layer index `64` or `32`, width derived
   from `Geometry`. When a declared MTP has no tensors (this 9B file), MTP
   stays `mtp_present=false` and never fabricates drafting.
4. `qwen38_dense` stays the app/Control family id; both sizes use it. Add
   explicit geometry reporting to `--inspect` and Control preview so the UI
   cannot confuse the profiles.
5. Performance target: same-file fixed-token F16 KV comparison against the
   installed PrismML/llama.cpp runtime. A "faster than llama.cpp" claim is
   made only if measured; otherwise it is recorded as not achieved.

## Changed files (planned)

- `include/strata/qwen38/admission.hpp` — `Geometry` + profile selection + tensor checks
- `src/qwen38/engine.cu` — dimensional literals, layer counts, KV/graph sizing, INFO/version text
- `src/qwen38/kv.hpp`, `src/qwen38/kv.cu` — head counts and attention-layer count as parameters
- `src/qwen38/ops.cuh`, `src/qwen38/ops.cu` — only where head counts are implicit
- `include/strata/qwen38/universal_layer.hpp` — comments/geometry where relevant
- `strata_control/qwen38_dense.py`, `apps/host/server/q38-dense-contract.mjs`,
  `apps/host/server/native-serving-policy.mjs`, `NativeLocalModel.tsx` — 9B preview/admission
- `src/qwen38/CMakeLists.txt`, tests (`test_cli.py`, `reader_metadata_test.cpp` or new geometry test)
- `_docs` implementation log (this file plus a completion record)

## Acceptance criteria

1. `--inspect` on the pinned 9B GGUF succeeds and prints geometry
   `hidden=4096 ffn=12288 layers=32 vocab=248320 q=16 kv=4` and `nextn=1`,
   with `mtp_tensors=0`; the pinned 27B fixture still prints its original
   geometry and byte counts.
2. CPU-only geometry unit tests cover both profiles, reject mixed/missing
   tensors and reject a 9B shape presented with 27B metadata.
3. Build succeeds with five compiler jobs; the existing 17 focused CUDA/CPU
   tests pass without modification of their tolerances, plus new 9B cases.
4. Real 9B GPU load, fixed-token logits (finite, deterministic across reset),
   generation, STOP, unload and normal process exit succeed under unchanged
   RAM 3 GiB / commit 4 GiB / VRAM reserve 768 MiB floors.
5. Same-file fixed-token F16 KV timing vs PrismML is measured. If the native
   runner is slower, that is reported; no unmeasured speed claim is made.
6. MTP drafting is exercised on a checkpoint that actually carries MTP
   tensors; the 9B file without MTP weights is explicitly recorded as
   MTP-off only.

## Verification plan

- Read-only GGUF inventory (done; see evidence above).
- `git diff --check`, CPU tests, `ctest`, five-job CMake build.
- CPU `--inspect` for both pinned GGUFs before any GPU run.
- Serial GPU runs only; no other user process is closed.
- Implementation log records commands, results, skipped checks and risks.

## Residual risks

- "Faster than llama.cpp" is not yet demonstrated even for the 27B profile;
  the 2026-11-19 handover records native warm decode slower than Prism.
  Geometry work does not by itself close that gap.
- The local 9B GGUF declares MTP but ships no MTP tensors, so end-to-end
  MTP drafting cannot be validated on it.
- Large parts of the CUDA path encode 27B head/tile counts. A missed literal
  may compile yet fail only at real-model inference; the GPU acceptance step
  is mandatory, not optional.
- The working tree has unrelated uncommitted Q38 work; it must not be staged,
  reset or cleaned as part of this task.

## Next actions

1. Implement `Geometry` and profile-driven `inspect()`; add CPU tests.
2. Parameterize `engine.cu` and `kv.*`; keep 27B numbers unchanged.
3. Update Control/Host preview and app family metadata.
4. Build, run tests, then serial GPU and same-file comparison.
5. Record results and residual risks in a completion log.
