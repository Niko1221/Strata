# KV Cache Proposals 3, 4, 5 - Detailed Design Notes

*Generated 2026-10-11 by Claude Code driven over the local Strata endpoint
(`claude-strata.cmd`, model `swift-1.5-iq3_xxs`), prompted to expand section 3 items
3/4/5 of `docs/research-kv-cache-compression.md`. Companion to that doc; the ranked
proposal list and literature map live there.*

## Verification pass (Hermes, against the tree)

Claims checked line-by-line. **Confirmed:** `kvalues_iq4nl` grid values and the 18-byte
`block_iq4_nl` layout; `kvalues_fp4` doubled E2M1 grid; `block_nvfp4` = `QK_NVFP4 64`,
`QK_NVFP4_SUB 16`, 4 UE4M3 scale bytes + 32 code bytes = 36 B per 64 values (4.5 b/elem,
144 B per 256-dim head = byte-identical to q4_0); `engine/BUILD.json` =
`archs [75,86,89,120]`, `source: release`, `ptx: true`, `cuda 13.0`; the fork's
CMakeLists 12X -> 12Xa rewrite with the verbatim comment *"the Blackwell FP4 tensor core
instructions are not forwards compatible and therefore need 12Xa"*; baseline numbers
(q4_0 document-8K: top-1 88.0%, KL 0.297, PPL 8.65, 67.4 tok/s, 25.7 rounds/s,
0.83 GiB, needles hold) from `bench/results/2026-09-27-kv-q4/README.md`.

**Two corrections to the text below:**

1. **The FP4 tensor-core instruction DOES exist in the bundled fork.** The analysis
   claims "no FP4 mma exists in the bundled fork" - it grepped `mmq.cuh` only. The
   instruction is present at `ggml-cuda/mma.cuh:1145`:
   `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`,
   guarded by `BLACKWELL_MMA_AVAILABLE` (`common.cuh:297`, `__CUDA_ARCH__ >= BLACKWELL && < RUBIN`).
   What is missing is the **MMQ dispatch** that uses it for FP4 (mmq.cuh carries the SRAM
   layouts and DP4A tile paths only). So proposal 5's tensor-core lane is a *wiring* job
   on an existing primitive, not from-scratch PTX. Related: `ggml_cuda_fp32_to_ue4m3`
   (`common.cuh:885`) is an existing Blackwell-gated device UE4M3 encoder - reuse it
   instead of writing the encoder in step D.2.
2. **`tools/kv_precision_compare.py` does not exist in the tree.** It is named in the
   baseline README as the protocol the 2026-09-27 numbers were produced with, but the
   script is not present (no file, no git history). Every validation plan below that
   cites it needs that harness written first - it is the real gate for all three
   proposals, so budget it as a prerequisite, not a step.

*The MSE figures in the analysis are analytic integrals over an assumed Gaussian source
with block-amax scale statistics - they establish ordering and mechanism, not
measurements. Parity + PPL + needle runs are the arbiter.*

---

# Proposal 3 — IQ4_NL KV as a drop-in over q4_0

## A. What the format is

`block_iq4_nl` (ggml-common.h:448-452) is **byte-for-byte the same shape** as `block_q4_0`:

```c
typedef struct { ggml_half d; uint8_t qs[QK4_NL/2]; } block_iq4_nl;   // QK4_NL = 32 → 18 bytes
```

One fp16 scale + 16 bytes of 4-bit codes per 32 values. Element `j` in the low nibble of `qs[j]`, `j+16` in the high nibble — identical packing to Strata's `block_q4_0` (kv_q4.hpp:24-28). **4.5 bits/element, 18 B/block, 144 B per head per cell.** GGML type id 20.

The only difference is the codebook. `kvalues_iq4nl` (ggml-common.h:1119-1121):

```
-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
```

Dequant is `y = d * kvalues_iq4nl[code]` (`dq_iq4_nl`, iq_kernels.cu:1657-1667) — a 16-entry table lookup, not `d*(code-8)`.

Two properties of the grid matter and are easy to miss:

1. **It is non-uniform and mass-matched.** Consecutive spacings: 23, 21, 18, 16, 14, 13, 12, **11, 12**, 12, 13, 15, 16, 20, 24 units. Narrowest cells in the middle, widest at the extremes — the inverse of the Gaussian density's tail. q4_0's grid is 16 equally spaced levels spanning `[-8d, +7d]`.
2. **It is asymmetric**: `-127` vs `+113`. The positive extreme is 11% closer to zero than the negative one.

The scale rule is also not q4_0's. `quantize_row_iq4_nl_impl` (ggml-quants.c:4966) is called with `ntry = 7` (line 5092) and does:

- seed `d = max/values[0]` (= amax/127 signed by max);
- pick codes by nearest-grid lookup;
- **refit** `d = Σ w·q·x / Σ w·q²` with `w = x²` (weights are `xb[j]*xb[j]` when `quant_weights == nullptr`, line 4987) — this is the least-squares minimizer of `Σ x²(x − d·q)²`, i.e. a **relative-error-weighted** fit, not an amax/8 fit;
- search 15 candidate scales `id = (itry + values[0])/max` for `itry ∈ [-7, +7]`, keeping the one maximizing `sumqx²/sumq2` (lines 5013-5026).

So IQ4_NL = non-uniform grid **+** a 15-candidate relative-error-optimal scale search. q4_0 = uniform grid + `d = amax/−8`, one shot.

## B. Why it beats q4_0 — the mechanism

Strata already did the hard half. `fwht256_inplace_cuda(b.kcur, g.n_head_kv)` (layer.cpp:1096-1097) rotates every 256-dim K and V row by an orthonormal Hadamard **after** `normalize_rotate` (RMS norm + RoPE). A Hadamard rotation of a vector with a few dominant channels produces coordinates that are approximately i.i.d. zero-mean Gaussian with `σ ≈ ‖x‖/16` — that is the QuaRot premise, and it is what makes 4-bit KV viable at all here.

Given Gaussian coordinates, the question is which 16-level grid minimizes MSE. Uniform is Lloyd-Max-optimal for **uniform** sources. For a Gaussian, the optimal grid is dense near 0 and sparse in the tails — which is exactly `kvalues_iq4nl`'s shape. Quantifying it (analytic, block-amax-normalized, `E[max|Z|] ≈ 2.13σ` for n=32, MSE ≈ Σᵢ pᵢΔᵢ²/12 summed over the Gaussian's probability mass in each cell):

| Quantizer (32-value block, Gaussian input) | step near bulk | step at tail | MSE / σ² |
|---|---|---|---|
| q4_0, uniform `Δ = amax/8` | 0.266σ | 0.266σ (clips: max maps to `7d`, error `d`) | **≈ 0.0081** |
| IQ4_NL, LS-fit scale | 0.14–0.16σ | 0.39σ | **≈ 0.004–0.005** |

The q4_0 number includes its clipping term: the block's largest element lands on level `7d = 0.875·amax`, costing one sample per block a full step of error (`(1/32)·(0.266σ)² = 0.0022σ²`, ~27% of q4_0's total MSE). IQ4_NL's LS refit removes exactly that: it trades a little resolution in the bulk to stop clipping the peak, and its asymmetric `-127/+113` grid is built for that trade.

Second, independent mechanism: the **`w = x²` weighting**. Attention scores are dot products `Σ_d q_d k_d`. Error in a large-`|k_d|` coordinate contributes proportionally more to score error than the same absolute error in a small one. `w = x²` weights the fit by relative error, which is closer to the score-error metric than q4_0's amax (which is set by the single largest element and spends resolution on the tail). KVQuant's Fisher-sensitivity weighting is the same idea with a fancier weight.

Net: same bytes, same rotation, same kernels, ~1.5–2× lower coordinate MSE, and the gain is concentrated in the coordinates that dominate the dot product.

Against Strata's own measured baseline (bench/results/2026-09-27-kv-q4/README.md), q4_0's cost is real and context-dependent: document text at 8K, KL from fp16 **0.297 vs int8's 0.114**, top-1 agreement **88.0% vs 92.3%**, PPL 8.65 vs 7.63. That KL gap is the target — the win condition is moving 0.297 toward ~0.20 at identical bytes.

## C. Mapping onto Strata's code

Because the block is 18 B with the same nibble packing, **every byte-moving site is unchanged**:

| Site | File:line | Change |
|---|---|---|
| `q4_store` | kv_q4.cu:93 | **None.** `block_iq4_nl` and `block_q4_0` are layout-identical; the cast and `blk->d`/`blk->qs[lane]` writes are valid for both. |
| `q4_group` | kv_q4.cu:75-91 | **Replace body.** Keep the deterministic cross-lane amax (the `a > amax \|\| (a == amax && v > mval)` tie-break, PR #21 fix). Replace `d = mval/-8` + `trunc(x/d+8.5)` with grid lookup + LS refit + ±7 search. |
| `kv_append_q4_kernel` | kv_q4.cu:100-118 | Grid `(n_head_kv, head_dim/QK4_0, n_steps*planes)` unchanged; page/row math unchanged. |
| `kv_append_q4_batch_kernel` | kv_q4.cu:121-138 | Prompt path + `stage` pool: unchanged. |
| `kv_gather_q4_kernel` | kv_q4.cu:141-175 | Dequant body: `(kq-8)*kd` → `kvalues_iq4nl[kq]*kd`. |
| `load8_q4` | qsa_decode_attn.cu:59-73 | Same nibble extraction; replace `((bytes[k]&0xF)-8)*d` with a LUT. Use `get_int_from_table_16` (vecdotq.cuh:34) — 4 PRMT/byte-permute ops per 8 codes, no int→float conversion, no `-8` bias. |
| `load_v1` | qsa_decode_attn.cu:347-366 | V-side, same substitution. |
| `runs_of` / `kv_block_bytes` | kv_stream.cu:33-62, 199-207 | **None** — `kKvQ4`'s two runs are already `rows * kv_q4_bytes_per_head()`, and IQ4_NL's row bytes are identical. Streaming, ring restore, staging, unstage all inherit for free. |
| `KvHostPools.k_q4/v_q4`, `QsaAttnPools.k_q4/v_q4` | kv_stream.hpp:39-40, qsa_decode_attn.hpp:30-31 | **None** — `uint8_t*` byte pools. |
| `fwht256_kernel` | kv_q4.cu:25-69 | **None.** |
| `kv_hybrid_v_half` | kv_stream.hpp:59-63 | K8V4's V half becomes IQ4_NL for free. |

Reference implementations to transcribe from: `dq_iq4_nl` (iq_kernels.cu:1657) for dequant, `vec_dot_iq4_nl_q8_1` (iq_kernels.cu:273) and `Split<20>` (:883) for the LUT dot. `iq_row_bytes` case 20 (:2709) already returns `(n/32)*sizeof(block_iq4_nl)`.

**The gap:** `iq_kernels.cu` has IQ4_NL *dequant and dot* paths only. There is **no device quantizer** — the fork's `quantize_iq4_nl` is CPU-only (ggml-quants.c:5077). The append kernel needs a new `__device__ iq4_group()` written from `quantize_row_iq4_nl_impl`.

Plumbing: `--kv` accepts `fp16|int8|q4_0|k8v4` (generate.cpp:2385-2387); add `"iq4_nl"`, a `qsa_set_kv_iq4_nl()`, a `bool kv_iq4_nl` beside `kv_q4` (layer.hpp:218), and a branch at layer.cpp:1098.

## D. Implementation plan

1. **Write `__device__ iq4_group(float x, int lane, uint8_t& byte)`** in kv_q4.cu, replacing `q4_group`. Warp = one 32-value block, lane = element. Keep the existing deterministic amax reduction verbatim. Then: `d = mval/-127`; loop `itry ∈ [-7,7]` computing `id = (itry - 127)/mval`, `l = best_index(l, x*id)`, accumulating `sumqx = Σ x²·q·x` and `sumq2 = Σ x²·q²` in a **fixed-order** warp reduction; pick the `itry` maximizing `sumqx²/sumq2`; final codes with that `d`.
2. **Pin the tie rule.** `best_index_int8` must resolve equidistant grid values identically in every lane, or the search's `sumqx/sumq2` diverges across lanes and the block gets two scales — the exact failure mode the PR #21 tie-break fix removed. Use `floor(v + 0.5f)` with an explicit documented half-away-from-zero rule, and assert all lanes agree on `d` (same `__shfl_xor` reduction as `q4_group`).
3. **Add `__constant__ int8_t kvalues_iq4nl[16]`** and a `get_int_from_table_16`-style PRMT helper; use it in `kv_gather_q4_kernel` and `load8_q4`.
4. **Gate on a flag**, default off, so q4_0 stays the shipped default (the README's stated decision: "int8 stays the default").
5. **Cost control.** The ±7 search is 15 grid passes over 32 values per block. The append kernel is 32 threads/block, `grid = (2, 8, n_steps*2)` — decode-side it is ~16 blocks per token per layer, negligible. The **prompt path** is `grid = (T, 2, 8)` over all T tokens: at 128K that is 2M blocks × 15 passes. Measure it; if it hurts, use the closed-form `d = mval/-127` (no search) for the prompt path and the full search for decode appends — they must then be made to agree, so pick **one** rule and validate it, don't split it.
6. **Do not touch** `fwht256`, the page math, `runs_of`, `kv_stream_resolve`, `kv_ring_restore`, `kv_stage_from_host`, `kv_unstage_to_host`, `mtp.cpp:761`, `verify.cpp:1197`. They are format-agnostic byte movers.

**Hard parts:** (i) lane-uniform scale selection (step 2) — the only genuine determinism hazard; (ii) the prompt-path search cost (step 5); (iii) the search's `Σx⁴` terms overflow fp32 for large-magnitude KV rows — accumulate in double, or normalize the block by `amax` before the search (recommended: it makes the search scale-invariant and removes the overflow entirely).

**CUDA-graph safety:** no new launches, no new buffers, no capacity-dependent grids. The existing kernels are already capturable (positions from `st.step`, grids from capacities). This proposal is graph-neutral.

## E. Expected numbers (ctx 131072, 12 QSA layers, 2 KV heads, head_dim 256)

| | q4_0 (baseline) | IQ4_NL |
|---|---|---|
| Bytes per head per cell | 144 B | **144 B** |
| Bytes per cell per layer (2 heads × K,V) | 576 B | **576 B** |
| Bytes per token (12 layers) | 6,912 B | **6,912 B (0%)** |
| Full 131072-token cache | 864 MB (0.84 GiB) | **864 MB** |
| Resident pool (32768 cells) | 226.5 MB | 226.5 MB |
| Streaming block (page_size 4) | 2,304 B | 2,304 B |
| Decode tok/s (baseline 67.4) | 67.4 | **67–68 (±2%)** |
| Rounds/s (baseline 25.7) | 25.7 | **±2%** |

Memory is exactly neutral — that is the point of the proposal. Speed is neutral: the decode dequant swaps 2 int ops + 1 int→float + 1 mul for 1 PRMT + 1 mul per 4 codes, and the append pays the scale search. The deliverable is quality: target **document-8K KL 0.297 → ≤0.22**, **top-1 88.0% → ≥90.5%**, PPL 8.65 → ≤8.0, with needles staying 5/5.

## F. Risks and the validation test

**Risks.**
- *Grid asymmetry.* `+113` vs `-127` biases the positive peak by ~11%. K carries a projection-bias structure (the doc's proposal 11, "Quantization Hurts Reasoning?" 2504.04823, measures 7.8 pts at 3-bit from pre-bias quantization). IQ4_NL's LS refit absorbs most of this, but it is the specific thing to look for in the K-side KL.
- *Grid provenance.* `kvalues_iq4nl` was fit for **weight** distributions, not post-RoPE rotated KV. It is a good Gaussian codebook, not a KV-specific one. If the rotated KV is heavier-tailed than Gaussian (RoPE pairs, attention sinks), the tail cells (24 units) clip harder than q4_0's.
- *Determinism* of the scale search (step 2 above).
- *No quality guarantee.* llama.cpp's "IQ4_NL beats q4_0" is measured on weights with `quant_weights`; Strata's KV has no such weights.

**Validation, in order:**
1. **`kv_q4_parity` extension** (src/kernels/kv_q4_parity.cpp, registered CMakeLists.txt:997-1000). It already checks FWHT orthonormality, bitwise q4_0 packing vs a host reference, paged append/gather, and `(Hq)·(Hk) == q·k`. Add: host `quantize_row_iq4_nl_impl` (compile ggml-quants.c into the test) vs the device `iq4_group` — **bitwise-equal `d` bits and `qs` bytes** over Gaussian, heavy-tailed (Student-t), all-zero, single-outlier, and denormal-adjacent rows; assert every lane agreed on `d`.
2. **`kv_stream_parity`** (CMakeLists.txt:986-989) — streamed vs resident bitwise with ring restore, the PR #21 lesson.
3. **Dequant bound:** max |x − x′| ≤ half the widest grid cell × |d|, and max |x−x′| ≤ |x| for every element (no sign flips, no clipping past the peak).
4. **Quality A/B** with `tools/kv_precision_compare.py`, teacher-forced, same text, fixed expert set (Q2_0, 3,000 slots, `--adapt-swaps 0`) — only `--kv` differs. Rows: code agent 512, document 1024, chat 1024, document 8K every-8th. Report same-top-1, mean KL, PPL, nats/token vs fp16, for `int8` / `q4_0` / `iq4_nl`. Gate: `iq4_nl` strictly better than `q4_0` on document-8K KL and top-1, within 2 s.e. of neutral on code/chat.
5. **`tools/needle_bench.py`** at 1K/32K/128K (10% and 90% depth) with `--kv iq4_nl --kv-resident 32768` — must be 5/5, matching the q4_0 baseline's `needles-q4.*`.

---

# Proposal 4 — Per-channel K scales (KIVI / KVQuant)

## A. What the format is

The current KV quantization is **per-token, per-32-value-group along `head_dim`**: `kv_append_q4_kernel` (kv_q4.cu:100) quantizes one cell's 32 contiguous dims in one warp, scale `d = amax/8` for that group. `KV_Q8_GROUP = 64` in kv_q8.hpp is the same idea at 64. Every scale is computed from, and applies to, **one token**.

Per-channel quantization inverts the axis. For channel `c ∈ [0,256)` and a **channel-block** of `G` consecutive tokens:

```
scale[c]  = amax_{t in block} |k[t,c]| / 8        (fp16, one per channel per block)
code[t,c] = clamp(trunc(k[t,c]/scale[c] + 8.5), 0, 15)
k'[t,c]   = scale[c] * (code[t,c] - 8)
```

The scale is **shared across all tokens in the block**, so it amortizes:

```
16 bits / channel / 128 tokens  =  16/(128*8)  =  0.125 bits per element
```

Layout for K, per KV head, per 128-token channel-block:
- codes: `128 tokens × 256 channels × 4 bits = 128 B/token` (4.0 b/elem, vs q4_0's 4.0 b/elem — identical code density)
- scales: `256 × 2 B = 512 B per 128 tokens = 4 B/token` (0.125 b/elem, vs q4_0's 16 B/token = 0.5 b/elem)

**K total: 4.125 b/elem vs q4_0's 4.5.** K gets a 4× finer scale amortization *and* 8.3% fewer bytes. V stays rotated q4_0 (per-token is correct for V — KIVI's central finding, and "Quantize What Counts" 2502.15075 proves keys deserve the bits).

Two coherent variants, and the choice is a real design decision:

**(a) KIVI form — K unrotated.** Per-channel scales in the original channel basis; K skips `fwht256`; scores pair unrotated `q` with unrotated `k` and are exact. Strata already has this exact shape: `kKvHybrid` stores K as **unrotated INT8** with the comment "the scores pair unrotated q with unrotated INT8 K" (layer.cpp:1082-1084). Proposal 4(a) is literally "swap K8 for K4-per-channel" inside the existing hybrid.

**(b) OScaR form — K rotated, per-channel in the rotated basis.** Keep `fwht256`. The subtlety that makes this legal: `H` is a **fixed** matrix, so rotated coordinate `c` is a fixed linear functional `h_c·k` of the token's key. Its statistics are stationary across tokens, so a per-channel amax over a 128-token window is a meaningful, stable envelope — and you keep the rotation that kills the outliers. This is the version I'd build: it composes with everything proposal 3 does.

One caveat for (b), and for RoPE generally: `normalize_rotate` applies NEOX RoPE to `n_rot = 64` of the 256 dims (qsa.hpp:100) *before* the FWHT. RoPE rotates channel pairs by a position-dependent angle, so an individual RoPE-paired channel's amplitude oscillates with position. Rotation preserves the pair's norm, and a 128-token window covers many phase cycles for the fast pairs (freq_base 1e-7 → the fastest pair completes ~20 cycles per 128 tokens), so a 128-token amax is a valid envelope. A 4-token window (page_size) is not — that is why the scale block must be ≫ page_size.

## B. Why it beats q4_0 — the mechanism

q4_0's per-32-group scale has two structural costs on keys:

1. **Scale cost is 0.5 b/elem, 4× the per-channel cost.** Per token per head, q4_0 spends 16 B on scales for 256 dims (8 groups × 2 B). Per-channel spends 4 B for the same 256 dims over a 128-token block. That is 11% of K's bytes bought back, and it is bought back *from the least informative part of the encoding*.

2. **The scale is estimated from 32 samples and thrown away every token.** Keys have *persistent* channel outliers — KIVI's observation, and the reason KIVI splits K per-channel / V per-token. A channel that is large for token *t* is large for tokens *t±1000*: the outlier is a property of the projection, not of the token. q4_0 re-derives a fresh amax per token per group, so it (a) pays for the outlier in every token's scale, and (b) lets the 31 non-outlier elements of the group ride on a scale set by the one outlier — the same clipping tax as proposal 3's q4_0 analysis, applied per token. Per-channel quantization measures the outlier **once per 128 tokens** and spends the resulting resolution on the bulk.

3. **The per-channel amax is a 128-sample estimate, not a 32-sample one.** `E[max|Z|]` over 128 samples (≈2.45σ) is a far more stable scale statistic than over 32 (≈2.13σ); the variance of the scale estimate is what makes per-token quantization noisy across tokens, and noise in the scale is noise in the score.

4. **Determinism improves for free.** `max` over tokens is exactly associative in IEEE-754 (no rounding, no tie ambiguity for distinct values). The per-channel scale has **no cross-lane reduction at all** — no `__shfl_xor` tie-break, no "one block two scales" hazard. Proposal 4's K path is bitwise-deterministic by construction, which is strictly better than the PR #21 tie-break machinery it replaces.

The literature anchor: KVQuant 2401.18079 gets <0.1 PPL delta at 3-bit with pre-RoPE per-channel K; KIVI 2402.02750 gets 2.6× peak-memory cut at 2-bit with per-channel K. Both rely on this asymmetry. Strata's own data supports the asymmetry too: K8V4 exists because K's exactness matters for scores.

## C. Mapping onto Strata's code

| Site | File:line | Change |
|---|---|---|
| `KvHostPools` | kv_stream.hpp:32-42 | Add `uint16_t* k_chan_scale` (indexed by channel-block, **not** by page). |
| `QsaAttnPools` | qsa_decode_attn.hpp:23-33 | Add `const uint16_t* k_chan_scale`. |
| `KvFormat` | kv_stream.hpp:48 | Add `kKvKChan = 4` (or extend `kKvHybrid`'s K half). |
| `runs_of` | kv_stream.cu:33-62 | K **codes** become a new run of `rows * head_dim/2` bytes (4-bit codes, 128 B/head/cell). Scales are **not** a run — see D.3. |
| `kv_block_bytes` | kv_stream.cu:199-207 | New branch: `rows*(head_dim/2) + rows*head_dim/2 + rows*kv_q4_bytes_per_head` = 8×128 + 8×128 + 8×144 = 3,136 B/block. |
| `kv_append_q4_kernel` | kv_q4.cu:100-118 | Split: V keeps `q4_group`; K becomes "store provisional code + fold amax", see D.2. |
| `kv_append_q4_batch_kernel` | kv_q4.cu:121-138 | Prompt path: K side writes fp16 into the staging window, not codes. |
| **new** `kv_finalize_k_block_kernel` | kv_q4.cu | Per-channel amax over the 128-token window → fp16 scales + requantized codes. |
| `load8_q4` | qsa_decode_attn.cu:59-73 | New `load8_kchan`: `d0` indexes codes at 4 bits, scale from `k_chan_scale[(cell/G)*HD + c]`. |
| `load8` dispatcher | qsa_decode_attn.cu:75-88 | Add `KV_MODE == 4`. |
| `attn_chunk_kernel` / `attn_chunk_kernel_pre75` | qsa_decode_attn.cu:117, 367 | Template dispatch: `kv_mode == 4` at :626 and :723. |
| `kv_gather_q4_kernel` | kv_q4.cu:141-175 | K side reads the channel scale. |
| `normalize_rotate` + `fwht256` order | layer.cpp:1096-1097 | Variant (b): rotate, then append. Variant (a): skip the K rotation, mirror the K8V4 branch at layer.cpp:1082-1093. |
| `kv_hybrid_k_half` | kv_stream.hpp:54-58 | Add the KChan analog (K folded onto itself, V half separate). |
| MTP / verify append sites | mtp.cpp:761-771, verify.cpp:1192-1221 | Must take the new K path; the drafter ring is the sharp edge (D.6). |
| `kv_stream_resolve` 16-byte guard | kv_stream.cu:218-221 | Only fires if scales become a per-page run. Keeping them out of the runs avoids it (and 512 B is a multiple of 16 anyway). |

## D. Implementation plan

1. **Pick variant (b)** (rotated basis, per-channel scales, `G = 128`). It keeps `fwht256`, composes with proposals 1/3, and needs no change to the query-side rotation at layer.cpp:1135.

2. **Two-phase append, because the scale needs a block.** Strata appends **one cell at a time** (`kv_append_q4_step`, graph-capturable, position from `st.step[kStepPos]`). A per-channel scale cannot be computed from one token. Design:
   - a fixed **fp16 staging window of 128 cells per KV head per layer** — this is *the same buffer* proposal 1's sliding FP16 window needs, so build them as one mechanism;
   - `kv_append_kchan_kernel` writes the rotated K cell into the window slot `pos % 128` (fp16, exact) and, if the cell is the last of the channel-block, marks finalization pending;
   - `kv_finalize_k_block_kernel` computes `scale[c] = max_t |W[t,c]|/8` (fp16), requantizes all 128 tokens' K into the 4-bit code pool, and writes the scales.
   - V is unchanged: `kv_append_q4_step` on `v_q4`.

3. **Make the scale pool always-resident — this is the key simplification.** Scales are tiny:
   `131072/128 = 1024 channel-blocks × 256 ch × 2 B × 2 heads × 12 layers = 12.6 MB`.
   That fits in VRAM permanently. Consequence: **scales never stream, never need a page table, never enter `runs_of`, never touch `resolve_kernel`'s clock sweep.** The decode kernel indexes them directly as `k_chan_scale[(cell/G)*HD + c]` — no `table[cell/page_size]` lookup. Codes keep streaming exactly as today. This sidesteps the genuinely hard alternative (a second block→slot residency map for scale blocks, which would need `resolve_kernel` to sweep two maps and `copy_kernel` a second run set).

4. **Graph safety.** Both new kernels follow the established pattern: grid sized from a **capacity** in `st`, real counts read from `st.step` on device, surplus threads guard (the comment at layer.cpp:1070-1073 is explicit that every entry point from there on is the capturable one). The window is a fixed 128-slot ring at a fixed address → capturable. `kv_ring_restore`/`kv_stage_from_host` are host-side DMA and are explicitly "Not capturable" (kv_stream.hpp:96-109) — the window must not be routed through them.

5. **Determinism.** Per-channel amax over 128 tokens: use a fixed tree reduction with a documented tie rule; `max` is exactly associative so the result is order-independent for finite inputs. `d = amax/8` → `f16_from_f32` (kv_q4.cu:90). Code = `__float2int_rz(x/d + 8.5f)` clamped — identical arithmetic to `q4_group`, so the parity reference is the same host code.

6. **The tail problem — the real hard part.** At any moment the current channel-block is **partially filled**: tokens `[128m, 128m + r)` with `r < 128` have no finalized scale, and the decode attention reads them *now*. Three options, and they are not equivalent:
   - keep the last 128 tokens in fp16 and read them from the window (this is proposal 1's window doing double duty — recommended, cost 128×256×2×2×12 = 1.57 MB);
   - finalize incrementally with a running amax (correct only if codes are requantized every time the amax grows — 128 requantizations per block, expensive);
   - finalize with a provisional scale and requantize at block close (needs the raw window anyway).
   All three need the window. Build the window once, use it for proposals 1 and 4.
7. **MTP drafter ring.** `mtp.cpp:756-771` uses a static ring table `block → block % n_slots` with no resolve, and `kv_ring_restore` refills from the host copy on resume. The drafter reads a window of recent cells — which after this change live in the fp16 window, not the code pool. Route the drafter's reads to the window (it only ever reads the last `window` cells, kv_stream.hpp:19-20) and verify the ring restore path.

**Hard parts, ranked:** the partial-block tail (6), the drafter ring (7), the two-phase append's interaction with rejected drafts (the writer must write host copy always and slot only if resident — kv_stream.hpp:16-17 — and a rejected draft must not leave a half-finalized channel-block; the window's `pos % 128` slot addressing makes this safe, because a rejected token's slot is simply overwritten), and the scale-pool allocation sizing at startup.

## E. Expected numbers

| | q4_0 | K per-channel (G=128) + V q4_0 |
|---|---|---|
| K per head per cell | 144 B | **132 B** (128 codes + 4 scale) |
| V per head per cell | 144 B | 144 B |
| Per cell per layer | 576 B | **552 B** |
| Per token (12 layers) | 6,912 B | **6,624 B (−4.2%)** |
| Full 131072-token cache | 864 MB | **828 MB (0.79 GiB)** |
| Resident 32768 cells | 226.5 MB | **216.4 MB** |
| Streaming block (4 cells) | 2,304 B | **2,208 B (−4.2%)** |
| + scale pool (always resident) | — | +12.6 MB |
| + fp16 window (128 cells) | — | +1.57 MB |
| Net vs 864 MB | — | **−23 MB (−2.7%)** |
| Decode tok/s (baseline 67.4) | 67.4 | **68–71** (streaming is PCIe-bound: 4.2% less traffic per miss; the 25.7 rounds/s baseline gains up to ~4%) |

Quality target: K is where the score error is born, and the doc's own ladder puts KIVI-style per-channel K as the Tier-1 lever. Target document-8K KL 0.297 → **≤0.18**, top-1 88.0% → **≥92%** (i.e. reach int8's measured numbers at ~4.1 b/elem for K). This is the proposal most likely to make q4-class KV match int8.

## F. Risks and the validation test

**Risks.**
- *Scale staleness.* A 128-token scale block is stale for a channel whose magnitude shifts abruptly (a topic change, a code block, an attention-sink region). Mitigation: G=128 is the amortization sweet spot; measure G ∈ {64,128,256}.
- *The tail/partial block* (D.6) is the correctness risk, not a quality risk. Get it wrong and the newest tokens — the ones decode attends to most — are quantized with a scale from a stale block.
- *Variant (a) vs (b) is an empirical question.* (a) keeps scores exact; (b) kills outliers. Run both; the code cost is one flag.
- *KVQuant's pre-RoPE finding* (proposal 11): per-channel scales are most stable **before** RoPE mixes pairs. Strata quantizes post-RoPE (layer.cpp:1096 after `normalize_rotate`). Measuring the amax envelope pre-RoPE and applying it post-RoPE is a cheap follow-up; do not conflate it into step 1.
- *12.6 MB resident* is not free on a 16 GB card already holding experts — it competes with expert slots (the baseline table shows 4,190 expert slots at q4_0).

**Validation:**
1. **New `kv_kchan_parity`** (clone kv_q4_parity.cpp, register beside CMakeLists.txt:997): host reference computes per-channel amax over exactly 128 tokens + `q4_group` arithmetic; device codes and scales **bitwise-equal**; assert lane-independence (no reduction → trivially true, which is the point).
2. **Window-boundary test:** sequences of length 127, 128, 129, 255, 256, 257 crossing block boundaries, and a **rejected draft** mid-block; assert the code pool + window contents are bitwise-identical to a single-pass reference.
3. **`kv_stream_parity`** with the new format: streamed vs resident bitwise, ring restore, staging, unstage.
4. **Attention parity:** `attn_chunk_kernel<KV_MODE=4>` vs `kv_gather` + `qsa_attend_step` on the same selection, scores within fp32 rounding.
5. **Quality sweep** (`tools/kv_precision_compare.py`, same protocol as proposal 3) at ctx 1K/8K/32K, `--kv kchan` vs `--kv q4_0` vs `--kv int8`.
6. **`tools/needle_bench.py` needle@128k** at 10% and 90% depth with `--kv-resident 32768` — ≥0.95, and specifically the deep needles, because per-channel staleness is a long-context failure mode.

---

# Proposal 5 — NVFP4 KV cache on Blackwell sm_120

## A. What the format is

`block_nvfp4` (ggml-common.h:221-227), type id `GGML_TYPE_NVFP4 = 40` (ggml.h:430):

```c
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16                       // sub-block size for per-group scales
typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB];        // UE4M3 scales: 4 bytes, one per 16-element sub-block
    uint8_t qs[QK_NVFP4/2];                  // packed E2M1 values: 32 bytes
} block_nvfp4;                               // = 36 bytes for 64 values = 4.5 bits/element
```

Three levels of scale:

1. **E2M1 values** — 4 bits, grid `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}·s`. The fork stores them **doubled** as int8 in `kvalues_fp4` (ggml-common.h:1126-1129): `[0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12]` — so a nibble indexes a signed int8 grid directly, and the `×0.5` is folded into the scale. This is the same `get_int_from_table_16` LUT trick as MXFP4 (`#define kvalues_mxfp4 kvalues_fp4`).
2. **UE4M3 scale per 16 elements** — unsigned E4M3: 4 exponent bits, 3 mantissa, max 448, min normal 2⁻⁶, subnormals to 2⁻⁹. Relative step 1/8 = **12.5%**, mid-tone 6.25%.
3. **FP32 per-tensor scale** — in the mmq path (`vec_dot_nvfp4_q8_1`, vecdotq.cuh:334). NVIDIA's recipe: `s_tensor = (448·6)/amax_tensor`, `s_block = round_E4M3(amax_block/6 / s_tensor)`, `x = s_tensor · s_block · q`.

**Grid geometry, in units of `s`:** 0, 0.5, 1, 1.5, 2, 3, 4, 6. Spacings: 0.5, 0.5, 0.5, 0.5, 1, 1, 2. Non-uniform, dense near zero, coarse at the peak — the same qualitative shape as IQ4_NL, and near-Lloyd-Max for a Gaussian.

**Alignment with Strata is exact.** `head_dim = 256 = 4 × QK_NVFP4`. Four `block_nvfp4` per head per cell = 4 × 36 = **144 B — identical to q4_0's 8 × 18 = 144 B.** 16 sub-blocks of 16 per head per cell. `QK_NVFP4 = 64` is a multiple of `QK4_0 = 32`, so the existing 32-thread group structure maps cleanly onto half a block.

Contrast MXFP4 (`block_mxfp4`, ggml-common.h:214-219): 32 elements, **one** `uint8_t e` scale in **E8M0** — a pure power-of-two exponent, no mantissa. 17 B per 32 = 4.25 b/elem. The scale snaps to the nearest power of two, so it can be up to 2× larger than needed, inflating MSE by ~1.4–2×. That is the mechanism behind the doc's "MXFP4 measurably worse for K" note, and it is why NVFP4's E4M3 mantissa is the point.

## B. Why it beats q4_0 — the mechanism

Same 4.5 b/elem, same 144 B/head. Three independent mechanisms:

**1. 16-element blocks, not 32.** The scale dial adapts 2× faster. Rotated KV coordinates are Gaussian *within* a block, but their variance is not stationary *across* a head's 256 dims (RoPE-paired dims, the 64/256 rot split) or across tokens (sinks, positional structure). A scale estimated from 16 samples tracks that variation; a 32-sample scale averages across it. Cost: 4 B of scale per 16 elements = 0.5 b/elem — exactly q4_0's scale overhead, paid at 2× the resolution.

**2. E4M3 mantissa scale, not fp16 and not E8M0.** q4_0's fp16 scale has 11 mantissa bits — more precision than needed. MXFP4's E8M0 has zero mantissa bits — far too coarse. E4M3's 3 bits (12.5% relative step) is the sweet spot, and it is *the native tensor-core operand format*, so it costs nothing to carry.

**3. The grid is matched to the rotated distribution.** Strata's `fwht256` makes coordinates Gaussian (the proposal-3 analysis). E2M1's spacings — 0.5, 0.5, 0.5, 0.5, 1, 1, 2 — are dense near 0 and coarse at the peak, i.e. the same mass-matching argument. This is the doc's Tier-2 item 7: rotation turns NVFP4 from "lossy 4-bit" into "near-optimal 4-bit".

Analytic MSE, Gaussian input, block-amax normalized, `MSE ≈ Σ pᵢΔᵢ²/12` over Gaussian mass per cell, `E[max|Z|] ≈ 1.92σ` for n=16 and `2.13σ` for n=32:

| Format | block | scale | MSE / σ² |
|---|---|---|---|
| q4_0 | 32 | fp16, `amax/8` | **≈ 0.0081** |
| **NVFP4** | **16** | **E4M3, `amax/6`** | **≈ 0.0077** |
| MXFP4 | 32 | E8M0 (power-of-2 snap) | ≈ 0.011–0.015 |
| IQ4_NL | 32 | LS-fit, non-uniform grid | ≈ 0.004–0.005 |

So NVFP4 ≈ q4_0 in pure coordinate MSE — **the quality win is not the grid, it is the 2× finer scale dial and the mantissa scale**, which show up on non-stationary real KV, not on synthetic Gaussians. NVIDIA's measured "<1% loss vs FP8 KV, ~5% better than MXFP4" is consistent with this: NVFP4 is at the q4_0 tier, MXFP4 is measurably below it.

**4. The tensor-core claim, stated honestly.** `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X` consumes E2M1 operands and takes the 16-element E4M3 scales as **register operands** (SF id), accumulating in fp32. The dequant multiply never executes. That is the ScaleSearchAttention (2605.12464) "zero dequant" pattern.

The shape constraint is the honest part: the FP4 mma is **m16n8k64** — it needs ≥16 rows. Strata's decode attention (`attn_chunk_kernel`, qsa_decode_attn.cu:117) is a **GEMV**: one query row per head, one thread per cell. `M = 1`, and `qsa_decode_attn_batch` gives `n_q = 1..8` for MTP. All below 16. FP4 tensor cores help:
- the **prompt path** (`qsa_prompt_attn.cu`, M = T tokens, T ≫ 16),
- the **MTP drafter** (`mtp.cpp:756-761`, T tokens),
- **verify** (`verify.cpp:1167-1221`, n tokens per draft step),
- `qsa_decode_attn_batch` at `n_q ≥ 16` (batched serving).

They do **not** help single-token decode. Anyone promising decode speedup from FP4 KV on a GEMV is selling a tensor core that will idle.

## C. Mapping onto Strata's code

| Site | File:line | Change |
|---|---|---|
| `block_nvfp4` | ggml-common.h:221-227 | Use as-is. Add a Strata-side mirror in kv_q4.hpp beside `block_q4_0` (line 24) to keep the kernel headers self-contained. |
| `kvalues_fp4` | ggml-common.h:1126-1129 | The doubled E2M1 int8 grid — the LUT. |
| `get_int_from_table_16` | vecdotq.cuh:34-56 | PRMT/byte-permute LUT, 4 ops per 8 codes, with a HIP `__builtin_amdgcn_perm` path already written. Directly reusable. |
| `dequantize_block_nvfp4` | convert.cu:378-405 | Reference for the dequant arithmetic (sub-block index `tid/8`, `qs[sub*8 + j]`, UE4M3→float). |
| `quantize_mmq_nvfp4` | quantize.cu:128-329 | The fork's NVFP4 quantizer, incl. `nvfp4_native_scale_error` (:13) and a scale **search** (:240-266). It is written for **activations** (`NO_DEVICE_CODE` at :329 says "Blackwell NVFP4 activations only") and is gated on `GGML_TYPE_NVFP4` at :641-650. Transcribe the E2M1 rounding + UE4M3 encoding; drop the search for KV. |
| `vec_dot_nvfp4_q8_1` | vecdotq.cuh:334-350 | The DP4A-on-LUT pattern: `get_int_from_table_16(..., kvalues_mxfp4)` then `ggml_cuda_dp4a` against q8_1 — **integer dot product on FP4 data, no FP4 hardware, no float dequant.** Strata has `strata_dp4a` (dp4a.hpp:29). This is the fast software path for QKᵀ. |
| MMQ SRAM layouts | mmq.cuh:128-159, 465-515, 1007-1045 | `GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4`, `MMQ_DP4A_TXS_Q8_0_16`, `ggml_cuda_mmq_load_tiles_nvfp4` — the tile/scale packing to copy. |
| `q4_store` | kv_q4.cu:93-97 | New `nvfp4_store`: 4 scale bytes + 32 code bytes per 64 values; `grid.y = head_dim/QK_NVFP4 = 4` (was `head_dim/QK4_0 = 8`). |
| `q4_group` | kv_q4.cu:75-91 | New `fp4_group`: 16-value sub-block, deterministic amax (keep the tie-break), `s = amax/6/s_tensor` → UE4M3, E2M1 round-to-nearest-even. |
| `kv_append_q4_kernel` / `_batch_kernel` | kv_q4.cu:100-138 | Grid `y` 8→4; 16 lanes per sub-block instead of 32 per block. |
| `kv_gather_q4_kernel` | kv_q4.cu:141-175 | LUT dequant. |
| `load8_q4` | qsa_decode_attn.cu:59-73 | `load8_fp4`: sub-block `d0/16`, scale byte `d[(d0/16)]`, LUT + UE4M3 mul. |
| `load8` / `load_v1` | qsa_decode_attn.cu:75-88, 347-366 | Add `KV_MODE == 5`. |
| `runs_of` / `kv_block_bytes` | kv_stream.cu:33-62, 199-207 | Add `kKvNvfp4 = 5`; runs are `rows * 4 * 36` = same 144 B/head → **streaming, ring, staging all inherit unchanged.** |
| `KvHostPools.k_q4/v_q4`, `QsaAttnPools.k_q4/v_q4` | kv_stream.hpp:39-40, qsa_decode_attn.hpp:30-31 | Reuse as opaque byte pools. |
| `fwht256_kernel` | kv_q4.cu:25-69 | Unchanged — and it is the thing that makes NVFP4 near-optimal. |
| **new** tensor-core kernel | `qsa_prompt_attn.cu` (prefill), `verify_kernels.cu` | FP4 `mma.sync` QKᵀ/PV. |
| `engine/BUILD.json` | `"archs": [75,86,89,120]` | **Must become `120a`** for the FP4 mma. |

## D. Implementation plan

1. **Ship the storage format first, with zero tensor-core dependency.** `kKvNvfp4` as a byte-identical-size replacement for `kKvQ4`: `fp4_group` in the append kernel, LUT dequant in `load8_fp4` and `kv_gather_q4_kernel`. Same 144 B/head, same streaming, same graphs. This is a pure quality experiment against q4_0 and needs no new hardware path. **Do not start with the tensor core.**
2. **UE4M3 encode/decode as a bit-exact device helper.** `ue4m3_from_f32` / `f32_from_ue4m3` with subnormal handling (min subnormal 2⁻⁹) and saturation at 448. This is where FP4 implementations break; make it a standalone unit test with a table of all 256 UE4M3 bit patterns.
3. **Per-tensor scale.** Needs `amax` over the whole K/V tensor, which Strata never computes (it appends one token at a time). Options: (i) a fixed conservative global scale per (layer, head) calibrated once from a short prompt; (ii) a running device-side amax in the append kernel (one `atomicMax` on uint bits per sub-block — deterministic, cheap, capturable); (iii) fold the per-tensor scale into the fp16 `d` slot and keep only UE4M3 per 16. Recommend (ii) + (iii): running amax in a per-layer device scalar, per-tensor scale recomputed at block close.
4. **Determinism.** E2M1 round-to-nearest-**even** at ties (the hardware's rule — match it, don't invent one); UE4M3 rounding round-to-nearest-even; running amax via `atomicMax` on ordered uint bits is order-independent for finite non-negatives. **Skip `quantize_mmq_nvfp4`'s scale search** for KV: it is a 20+ iteration search over candidate scales designed for prefill activations, it is not lane-uniform, and it makes the append non-deterministic. Closed-form scale only.
5. **Then the tensor-core path, behind a flag, prefill-only.** New `qsa_prompt_attn_fp4.cu`: tiles of K in `block_nvfp4` layout, Q quantized to NVFP4 on the fly (reuse `quantize_mmq_nvfp4`'s shape), `mma.sync.aligned.kind::mxf4nvf4...scale_vec::4X` with SF operands, fp32 accumulators, LSE merge in `attn_merge_kernel` (qsa_decode_attn.cu:522).
6. **The arch landmine, and it is the sharpest one in this document.** `BUILD.json` has `"archs": [75,86,89,120]`. The fork's CMakeLists.txt:75-77 states it plainly: *"the Blackwell FP4 tensor core instructions are not forwards compatible and therefore need 12Xa"*, and rewrites `12[0-9]` → `12[0-9]a` (lines 78-90). Strata's own CUDA build must do the same. Consequences: (a) `sm_120a` cubins **will not JIT onto plain `sm_120`** — the shipped `engine/strata.exe` (`"source": "release"`, `"ptx": true`) cannot gain FP4 mma by rebuilding from source on this box, per the doc's §4 note; (b) the FP4 kernels must be `#if defined(CUDA_ARCH_SPECIFIC)`-gated so a plain-120 build still compiles; (c) CUDA 13.0 only — 13.2.0/13.2.1 miscompile IQ2_S/IQ3_S on sm_120 (doc §4), and those kernels live in the same TU family.
7. **Keep the SIMT path as the fallback and the parity reference.** The tensor-core kernel must be bitwise-comparable to `load8_fp4` + `qsa_attend_step` on the same selection, or the FP4 path is unauditable.

**Hard parts, ranked:** the 120a arch split and its effect on the shipped release binary (6); UE4M3 bit-exactness incl. subnormals (2); the per-tensor scale with per-token appends (3); Q-side FP4 quantization for QKᵀ (quantizing Q to 4 bits is a *precision* decision, not a free one — Q is 24 heads × 256 dims of activations with far less redundancy than KV; consider FP8/BF16 Q with FP4 K, which the mma does not allow (both operands must be FP4 in `kind::mxf4nvf4`) — this is the real blocker for QKᵀ and the reason PV-in-FP4 is the safer first target).

## E. Expected numbers

| | q4_0 | NVFP4 (storage) | NVFP4 (+ FP4 mma prefill) |
|---|---|---|---|
| Per head per cell | 144 B | **144 B** | 144 B |
| Per token (12 layers) | 6,912 B | **6,912 B (0%)** | 6,912 B |
| Full 131072-token cache | 864 MB | **864 MB** | 864 MB |
| Resident 32768 cells | 226.5 MB | 226.5 MB | 226.5 MB |
| Streaming block | 2,304 B | 2,304 B | 2,304 B |
| Decode tok/s (baseline 67.4) | 67.4 | **67–68 (±2%)** | **67–68 (no change — GEMV)** |
| Rounds/s (baseline 25.7) | 25.7 | ±2% | ±2% |
| Prompt-path attention (128K prefill) | baseline | ±0% | **1.3–2× on the QKᵀ/PV kernels** |
| MTP drafter / verify | baseline | ±0% | 1.1–1.5× (T tokens, M≥16) |

Memory is neutral by construction. Decode speed is neutral — the dequant is a LUT+mul vs shift-sub-mul. The only real speed number is on the prompt/draft/verify paths, and only with the tensor core. Quality target: match q4_0's needles 5/5 and beat its document-8K KL 0.297 by the finer-scale-dial margin — realistically 0.24–0.29, i.e. a smaller win than proposals 3 and 4. **NVFP4 is a differentiation and prefill-throughput play, not a quality play.** Say that in the proposal, because the doc's framing ("<1% loss, strictly faster") oversells the decode case.

## F. Risks and the validation test

**Risks.**
- **120a is not forwards compatible** (CMakeLists.txt:75-77). A 120a binary will not JIT on a plain sm_120 part, and the shipped `engine/strata.exe` is a release build with `source: release` that the doc explicitly says not to rebuild from source on this box. The tensor-core path is a **new distribution artifact**, not a code change.
- **Q-side FP4 for QKᵀ** — 4-bit queries are a much harder quantization problem than 4-bit keys (no 128-token amortization, no Hadamard over a stationary distribution). `kind::mxf4nvf4` requires both operands FP4. This is the proposal's biggest technical risk and it is under-addressed in the doc.
- **UE4M3 subnormals/saturation** — KV magnitudes span orders of magnitude across layers; a per-tensor scale that is wrong by 2⁴ sends blocks into subnormal or saturation.
- **No FP4 mma exists in the bundled fork** — I grepped: no `FP4_MMA_AVAILABLE`, no `mxf4nvf4` in `mmq.cuh`. The fork's NVFP4 is a **DP4A-on-LUT** format (`vec_dot_nvfp4_q8_1`, `MMQ_DP4A_TXS_Q8_0_16`), not a tensor-core format. The "dequant happens inside the tensor core" story is not in this tree.
- **FP8 accumulation caution** (doc §1, vLLM 2026-04-22) is N/A on Blackwell, but the two-level accumulation note applies if the FP4 path ever runs >100K contractions on the ROCm/RDNA4 backend, where **FP4 is software-emulated** (doc §1, HotChips 2025: RDNA4 has F16/F8/INT8/INT4 tensor types, no FP4).

**Validation:**
1. **`nvfp4_bits_test`** (new, beside `tools/sm60_dp4a_check.cu`): all 256 UE4M3 patterns round-trip; all 16 E2M1 nibbles dequantize to the exact doubled grid; saturation at 448; subnormals to 2⁻⁹.
2. **`kv_nvfp4_parity`** (clone kv_q4_parity.cpp): device `fp4_group` bitwise-equal to a host reference over Gaussian / heavy-tailed / all-zero / single-outlier / subnormal-boundary rows; FWHT orthonormality and `(Hq)·(Hk) == q·k` unchanged; paged append/gather bitwise.
3. **`kv_stream_parity`** with `kKvNvfp4`: streamed vs resident bitwise, ring restore, staging, unstage — the runs are the same size, so this should pass on first run; if it does not, `runs_of` is wrong.
4. **Software-vs-tensor-core parity** (the gate for step 5): `qsa_prompt_attn_fp4` vs `load8_fp4` + `qsa_attend_step` on identical selections — scores within fp32 rounding, softmax outputs within a stated bound, **run at ctx 131072 with the full 2048-cell top-k selection** so the long-contraction path is exercised.
5. **Quality sweep** (`tools/kv_precision_compare.py`, same protocol) `--kv nvfp4` vs `q4_0` vs `int8`, ctx 1K/8K/32K.
6. **`tools/needle_bench.py` needle@128k** at 10% and 90% depth, `--kv-resident 32768` — ≥0.95, plus 262K to match the baseline's 5/5 set.
7. **Arch matrix check:** build with `120` and with `120a`; confirm the plain-120 build compiles and runs (FP4 mma compiled out) and that `cuobjdump -sass` shows the `mma.sync` only in the 120a cubin.

---

## Cross-cutting notes

- **Ordering.** 3 → 4 → 5 is the right order, and 3 and 4 **compose** (per-channel K with an IQ4_NL grid: 4.125 b/elem for K at non-uniform-grid precision). 5 is a separate lane with a distribution cost (120a) and should not block 3 or 4.
- **All three are byte-compatible with the streaming layer** in the sense that `runs_of`/`kv_block_bytes`/`resolve_kernel`/`copy_kernel` are format-agnostic byte movers — that is the strongest thing in this codebase and it is why proposals 3 and 5 are cheap. Proposal 4 is the only one that adds a buffer class (the always-resident scale pool + the fp16 window).
- **The gate is the measured baseline, not the papers.** bench/results/2026-09-27-kv-q4/README.md gives the numbers to beat: document-8K KL 0.297, top-1 88.0%, PPL 8.65, 67.4 tok/s, 0.83 GiB, needles 5/5. Every proposal above must be reported against those five numbers on the same teacher-forced protocol.
- **Two numbers I derived analytically** (the MSE table entries) are integrals over an assumed Gaussian source with block-amax scale statistics, not measurements. They establish the *ordering* and the *mechanism*; the parity + PPL + needle runs are the arbiter. The code facts (block sizes, grid values, type ids, call sites, arch flags, baseline timings) are all read directly from the tree.
