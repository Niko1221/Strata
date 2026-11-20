# Spin8 / Triality applicability to Q38 (2026-11-20, OpenCode)

Repository research triggered by the user question about using Spin8 Triality
and the vector/plus/minus views. Read-only; no files changed.

## What exists in this repository

1. **KV compression mode `spin8_k8v4`** (`src/qwen38/kv_config.cpp:25,31,62-75`,
   `src/qwen38/kv.cu` Spin8 kernels): Q8_0 keys + Q4_0 values after an 8x8
   block-diagonal rotation. Views `vector` (0), `plus` (1), `minus` (2) are
   selected with `--spin8-view`; the matrix is produced by `spin8_plan_matrix`
   from the pinned `data/spin8/reference.profile` and audited at startup.
   `engine.cu:797` reports the view and matrix SHA-256 in INFO.

2. **Spin8 Triality operator** (`src/qwen38/spin8_triality.cuh`): a normalized
   8-element Spin(8)/Hadamard transform plus an experimental 3-view compressed
   attention estimator with a query-dependent Clifford adapter. The mathematical
   contract is documented in `_docs/2026-10-23_integrated_triality_algorithm_Codex.md`.
   That document explicitly states it "does not establish a better Transformer,
   a quality gain, or a speed gain on either required model" (lines 3-7).

3. **Integration status**: `prefill_gemm.cu:3` includes `spin8_triality.cuh` but
   no triality symbol is used there. The Triality query adapter / 3-view
   estimator is **not wired into the Q38 engine**; only the standalone math and
   `spin8_triality_test.cu` exist. The production Spin8 path in `kv.cu` uses
   `spin8_plan_matrix`, not the Triality estimator.

## Can we use it for speed? Assessment

- **As a KV memory/format switch: yes, technically.** Auto KV selection already
  considers `spin8_k8v4` as its third candidate (`kv_config.cpp:106`). Under
  VRAM pressure it reduces KV bytes from 512 to 272/row. Prior measurements in
  `_docs/2026-11-18_q38-cuda135-handoff_Antigravity.md:23-28,45-62` show Spin8
  plus split-KV greatly improved 27B decode (5.23 -> 23.26 step/s), but the
  2026-11-20 f16 batched-attention sweep found `spin8_k8v4` slower for the
  1024-token prefill fixture because its attention path is still per-token, not
  batched. So the benefit is conditional on the workload.

- **As a math/algorithmic speedup: not established.** Triality's three-view
  estimator adds work (three packed views + residual sketch + query adapter)
  and is intended as a compression/quality experiment, not a speedup. The
  document itself withholds any speed claim.

- **Blocker for 9B**: the legacy Spin8 kernels in `kv.cu` still hard-code the
  27B geometry (`HG=6`, `6144` output floats, `dim3(4,...)`, `head/6`, 48-head
  aggregate). The new f16 batched path is geometry-parameterized, but Spin8 on
  9B would corrupt memory until those constants are derived from `KvShape`
  (finding from the 2026-11-20 code-analysis subagent).

## Mathematical note on "vector spinor triple inference"

The three views are the three 8-dimensional representations of Spin(8)
(vector V, half-spin S+, S-), related by the triality automorphism. With
orthogonal codecs U_r and reconstruction weights alpha_r, the score estimate is

s_j = sum_r alpha_r (U_r x)^T Q_r(U_r k_j) + residual term.

If the quantizers Q_r were the identity and sum_r alpha_r = 1, every term
equals x^T k_j (orthogonality), so the estimate reproduces ordinary attention
exactly. The construction is therefore a **distortion-reduction / compression**
scheme: three low-bit views approximate one higher-precision view. Its cost is
three dot products instead of one. In our FP16 / Tensor Core attention path,
that triples the dominant O(position x 256) work, so triple inference is a
**quality-per-bit tradeoff, not a speedup**. A speedup would only be plausible
if the three 2-bit views were evaluated with bitwise/popcount arithmetic that
is cheaper than one 4-bit or FP16 dot product; Q38 currently uses FP16 tensor
cores, so that condition does not hold. The repository's own contract
(`2026-10-23_integrated_triality_algorithm_Codex.md:3-7`) also makes no speed
claim.

## Hikari checkpoint (downloaded 2026-11-20)

The user supplied the HuggingFace repo
`zapabobouj/Hikari-Bonsai-27B-Nimble-LoRA-GGUF`. The single GGUF
`Hikari-LoRA-fixed-scale-PQ2_0.gguf` (7,206,168,928 bytes) was downloaded to
`H:/Strata/models/hikari-pq20/` and its SHA-256 matched the pinned LFS value
`0ab2a1f237b73e5002ecd8c9b764249b90ee59eca749a6226c44bb43b1036a13`.

`--inspect` on the current engine reports:
`profile=q38-27b layers=64 tensors=851 hidden=5120 ffn=17408 vocab=248320
q_heads=24 kv_heads=4 nextn=0 mtp_tensors=0 folded=401 inverse_embedding=1
grouped_gdn=1`.

This is the checkpoint where the Hadamard-folded path is active (the soya 27B
fixture has `folded=0`). A build with per-layer Hadamard reuse for hidden-width
ungrouped folded projections (`q38-9b-hikari-20261120`) was completed with
CTest 18/18 after re-running without GPU contention. Its A/B against the
baseline on this checkpoint is the remaining step.

## Concrete experiment if pursued

1. Parameterize the Spin8 kernels for `q_heads/kv_heads` from `KvShape`.
2. Add a batched Spin8 prefill attention analogous to `store_f16_batch` /
   `score_batch` / `attend_batch`, including the rotation inverse.
3. A/B against f16 on both 9B and 27B with forced-token logits under the
   tolerance contract (3e-5/4e-5), measuring prefill and decode separately.
4. Only adopt if it wins on the target shape set and passes held-out shapes.

Until steps 1-4 are done, Spin8/Triality should be treated as an opt-in VRAM
tool, not a proven acceleration path.
