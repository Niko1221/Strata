# Q38 dense 9B native engine — implementation record (2026-11-20, OpenCode)

## Overview

Added a second geometry to the Strata-owned dense CUDA runner so the local
32-layer dense `qwen35` 9B checkpoint runs on the same native path as the
validated 27B checkpoint. Geometry is selected from GGUF metadata and tensor
extents, never from a file or publisher name. MTP geometry is included from the
start. The 27B path remains behaviorally identical.

## Background / requirements

User request: build a Qwen3.8-9B-class native engine from the Qwen3.8-27B
native engine; local and general-purpose; CUDA 13.5 Tensor Core; faster than
llama.cpp; full implementation and verification; MTP included from the start.

Plan and evidence: `_docs/2026-11-20_q38-9b-native-plan_OpenCode.md`.

## Assumptions / decisions

The 9B GGUF
(`H:/Strata/models/新しいフォルダー/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf`,
5,629,111,584 bytes, 427 tensors, F32 177 / Q4_K 217 / Q6_K 33, SHA-256 pinned
in prior records) declares `nextn_predict_layers=1` but contains no MTP tensors.
Its `general.architecture` is `qwen35`, identical to the 27B baseline.

Decisions: a `Q::Geometry` descriptor with two explicit profiles; all
dimensional literals in admission, engine, KV and Control replaced by geometry
fields; `qwen38_dense` remains the app family id with a `geometry` field in the
preview result; 27B literals and scratch accounting preserved exactly.

## Changed files

- `include/strata/qwen38/admission.hpp` — `Geometry`, `read_geometry()`, profile selection, derived sizes, geometry-driven tensor/MTP/Hadamard checks
- `include/strata/qwen38/universal_layer.hpp` — unchanged (geometry comments reviewed)
- `src/qwen38/engine.cu` — geometry-driven buffer sizes, graph loops, layer indices, MTP names, INFO/inspect output
- `src/qwen38/kv.hpp`, `src/qwen38/kv.cu`, `src/qwen38/kv_config.cpp` — `KvShape` head geometry parameter
- `src/qwen38/geometry_test.cpp` (new), `src/qwen38/CMakeLists.txt` — `q38_geometry` test
- `strata_control/qwen38_dense.py`, `strata_control/models.py` — 9B preview, geometry label
- `tests/terminal/test_qwen38_dense.py` — 9B preview test and Q4_K fixture bytes
- `src/qwen38/README.md`, `_docs/2026-11-20_q38-9b-native-plan_OpenCode.md`, this log

## Implementation details

`Geometry` carries layers, hidden, FFN, vocab, Q/KV heads, head dim, GDN group,
state, rank, inner, conv kernel, interval, RoPE sections/scalars and EOS.
Derived helpers give attention/GDN layer counts, GDN qkv channels
`(2*group + 2*q_heads)*state`, attention Q/KV/output row widths, and the MTP
layer index. The two supported fingerprints are 64/5120/17408/24/48/6144 and
32/4096/12288/16/32/4096; everything else is rejected before CUDA work.

The engine's prefill scratch layout was kept byte-for-byte for 27B: the
sub-buffer base is `hidden*3` floats per token and the rotating scratch sits
after the largest of GDN/attention/FFN sub-regions. The initial implementation
reserved an extra `hidden` floats per token; this was found by comparing
27B scratch estimates with the released binary and corrected.

MTP names are derived from `blk.<trunk_layers>.`, so the 27B predictor remains
`blk.64` and a 9B predictor would be `blk.32`. With no predictor tensors the
9B file reports `mtp_tensors=0`, `mtp_present=0`, drafting off.

## Commands run

- Read-only GGUF inventory: `tools/gguf_reader.py` via a temporary ASCII-safe script.
- Configure/build/test (5 jobs, clean cache): `vcvars64.bat` → `cmake -S src/qwen38 -B H:/Strata/builds/q38-9b-geometry-20261120 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=.../cuda-13.5.0/toolkit/bin/nvcc.exe -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_Q38_REQUIRE_CUDA135=ON` → `cmake --build ... --parallel 5` → `ctest --test-dir ... --output-on-failure`.
- CPU inspect on both pinned GGUFs with the new `strata-q38.exe`.
- Real 9B GPU runs: `--ctx-size 256 --tokens 100,200,300 --n 4 --kv f16`, then `--tokens 14556,1814 --n 3`, then 140- and 1000-token prompts.
- 27B regression: same fixed prompt on installed `0.6.3` binary and the new binary; MTP run with `--spec-draft-n-max 1`.
- Python: `py_compile` on changed files; `pytest tests/terminal/test_qwen38_dense.py tests/terminal/test_discovery.py -q`.
- `git diff --check`.

## Verification results

- Build: passed, CUDA 13.5.37 / MSVC 19.44, SM120, five compiler jobs.
- CTest: 18/18 passed (original 17 plus new `q38_geometry`).
- CPU inspect 9B: `profile=q38-9b layers=32 tensors=427 hidden=4096 ffn=12288 vocab=248320 q_heads=16 kv_heads=4 nextn=1 mtp_tensors=0 gpu_weight_bytes=5046011904 host_embedding_bytes=572129280`.
- CPU inspect 27B: `profile=q38-27b layers=64 tensors=866 hidden=5120 ffn=17408 vocab=248320 q_heads=24 kv_heads=4 nextn=1 mtp_tensors=15`, byte counts unchanged.
- Real 9B generation: passed (`220,16,23,15`), exit 0, finite logits, no resource or numerical error.
- Real 27B regression: old and new binaries both produced `4183,344,248046`; scratch estimate identical after the fix (267,919,084 bytes).
- MTP path: 27B with `--spec-draft-n-max 1` reported `mtp_present=1 mtp_ready=1` and exited normally.
- Python: 55/55 focused tests passed; `py_compile` clean; `git diff --check` clean.

Performance (same 9B GGUF, F16 KV, PrismML 0.4.1 `llama-server` commit
`1965755`): 140-token prompt — Prism 235 ms, native 496–715 ms; 1000-token
prompt — Prism 393 ms, native 1623 ms. Short generation: Prism 30.1 ms/token,
native ~28.2 ms/step. **The "faster than llama.cpp" target is not achieved for
prompt processing.** No unmeasured speed claim is made.

## Residual risks

- The local 9B GGUF declares MTP but ships no predictor tensors, so real MTP
  drafting on 9B cannot be validated with this file. The geometry is ready;
  validation needs a 9B checkpoint that actually includes `blk.32.nextn.*`.
- Prism's generic `llama-batched-bench` (commit c9c34bc) cannot load this 9B
  file because it expects `blk.32.attn_norm.weight`; the installed 0.6.3 Prism
  server includes the absent-MTP patch and was used instead.
- No GUI/Control end-to-end test was run with the new binary; app packaging,
  component signing, feed registration and installer updates are not part of
  this change.
- General model answer quality, full-window inference and concurrent sessions
  remain unqualified; `releaseQualified=false` is unchanged.
- The working tree contains unrelated uncommitted Q38 work; it was not staged,
  reset or cleaned.

## Recommended next actions

1. Decide whether to keep the 9B geometry as a private opt-in candidate or
   prepare a signed component build with matching Control/Host packaging.
2. If the speed target is mandatory, profile the 9B prefill GEMM (cuBLAS vs
   fused Tensor Core path) and the KV/decode path before any release claim.
3. Obtain a 9B GGUF with real MTP tensors and validate drafting end to end.
4. Run the installed-app GUI acceptance in an isolated profile after a signed
   component is available.
