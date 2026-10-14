# EXL3 support in Strata

> **Status: in progress.** The format is specified and a tested Python reference decoder exists
> (`tools/exl3/`, `tools/test_exl3_codebook.py`). The engine does **not** load EXL3 models yet.
> This page is the contract the C++/HIP backend must satisfy. Back to the [README](../README.md).

[EXL3](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md) is turboderp's quantized
format for ExLlamaV3, a variant of **QTIP**. It is **not** a GGUF variant: the weights live in
`safetensors`, one set of tensors per linear, and each weight is reconstructed procedurally at
inference time from a trellis. The model this work targets is
[turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
(`3.05bpw_h5_ng5`) — the same Qwen3.8-Flash-Next architecture the engine already runs, in EXL3.

## The files in an EXL3 model directory

| File | What it is |
| --- | --- |
| `config.json` | the model config, with a `quantization_config` block: `quant_method: "exl3"`, `version`, `bits` (average bpw), `head_bits`, `codebook`, `out_scales`, `calibration` |
| `model.safetensors.index.json` | maps every tensor name to its shard |
| `model-0000N-of-0000M.safetensors` | the weight shards |
| `quantization_config.json` | extra per-tensor quant metadata (large: ~97 MB for this model) |
| `gram_embedding.safetensors` | this model's n-gram embedding table (~32.6 GB) |
| `mtp_hyper_connection_mixer_patch.safetensors` | the MTP mixer patch |

For the target model: `bits: 3.05`, `head_bits: 5`, `codebook: "mul1"`, `out_scales: "always"`,
`version: 1.4.4`. `bits` is the **average**; the actual bitrate `K` is per tensor (see below).

## One quantized linear

Every quantized `Linear` is stored as up to four tensors under its key `prefix`:

| Tensor | dtype / shape | Meaning |
| --- | --- | --- |
| `prefix.trellis` | int16, `[in/16, out/16, 256*K/16]` | the packed trellis, 16×16 tiles |
| `prefix.suh` | fp16, `in` | per-input-row factor (sign × scale) |
| `prefix.svh` | fp16, `out` | per-output-column factor (sign × scale) |
| `prefix.mcg` / `prefix.mul1` | int32, scalar | present only to mark the codebook (`mcg` → 1, `mul1` → 2; neither → 0) |
| `prefix.bias` | fp16, `out` | optional |

Measured from the target model (`model.language_model.layers.47.mlp.experts.10.down_proj`): `trellis`
I16 `[40, 160, 48]` (in 640, out 2560, **K = 3**), `suh` F16 `[640]`, `svh` F16 `[2560]`, `mul1` I32
scalar. `suh`/`svh` are continuous (e.g. `suh ≈ -0.0199`, `svh ∈ [-1.77, …]`); older checkpoints
store them as packed int16 sign bitfields instead, and a key may carry `su`/`sv` rather than
`suh`/`svh`.

`K = trellis.shape[-1] // 16` is the bitrate: an integer 1..8, or a half-integer 1.5 / 2.5 / 3.5
(mul1 only). 3.05 bpw is achieved with a mix, e.g. 2/2.5/3/3.5 for different tensors. `head_bits`
sets the `lm_head` bitrate separately.

## Decoding a tile

The trellis of one 16×16 tile (256 weights) is `256*K` bits, held as `256*K/16` little-endian
`uint16` words. The "trellis" is a set of **overlapping 16-bit windows**: weight `t` is the
codebook lookup of the window that starts at bit

    start(t) = (t*K + K - 16)  mod (256*K)

read little-endian with wraparound (tail-biting). This is exactly `dq()` in
`exl3_dq.cuh`; there is no Viterbi at decode time (the encoder runs the Viterbi search to choose
the bits; the decoder is just the windowed lookup).

The 256 values come out in the **tensor-core-interleaved order** used by the packed tensor; the
reconstruct kernel (`reconstruct.cu`) un-permutes them into a row-major 16×16 tile while writing.

## The procedural codebook

`decode_3inst<cb>` (`codebook.cuh`) maps a 16-bit window to one fp16 weight. Three codebooks:

- **mul1 (`cb=2`)**, used by this model: `x = w * 0x83DCD12D`; `s = 0x6400 + Σ bytes(x)`;
  `h = fp16_bits(s & 0xFFFF)` (a value in `[1024, 2047]`); result `= hfma(h, 0x1eee, 0xc931)` in
  fp16, i.e. `h/147.7 − 10.39`, giving weights in about `[-3.46, 3.47]`.
- **mcg (`cb=1`)**: `x = w * 0xCBAC1FED`; `x = lop3(x, 0x8FFF8FFF, 0x3B603B60, 0x6a)`;
  result `= fp16(lo16) + fp16(hi16)`.
- **3inst (`cb=0`)**: `x = w*89226354 + 64248484`; same `lop3`; same `+` of two fp16 halves.

`lop3(a,b,c,0x6a)` is the SASS ternary with truth table minterms `{1,3,5,6}`, i.e. bit `c XOR (a AND
b)`. The fp16 `hfma` is one rounding of the exact product+addend.

Constants verified against the source: `fp16(0x1eee)=0.0067672`, `fp16(0xc931)=-10.3828`,
`fp16(0x6400)=1024.0`, `fp16(0x67ff)=2047.0`.

## Reconstruction

The stored weight is not `W`; it is the rotated/quantized form. The decoded tile values are in a
**tensor-core interleave** (`tensor_core_perm`, `quantize.py`): `tc[j]` is row-major element
`perm[j]` of the 16×16 tile, with `perm[t*8+0..7] = {r0,r1,r2,r3}×c0 then {r0,r1,r2,r3}×c1` for
`r0=(t%4)*2, r1=r0+1, r2=r0+8, r3=r0+9, c0=t//4, c1=c0+8`. Un-permuting each tile and assembling
gives `W_hat`. The original-basis weight is then

    W = diag(suh) · H128 · W_hat · H128 · diag(svh)

where `H128` is the natural-order Sylvester Hadamard, `H128[i,j] = (−1)^popcount(i∧j)/√128`, applied
in blocks of 128 along each dimension (`hadamard.cu`, `had_r_128`). The fused `reconstruct_had`
kernel emits `W` directly.

## Decode optimizations

These are the levers, in rough order of impact (all implemented in the reference):

1. **Never materialize `W` for decode.** Because `H` is symmetric, the GEMV is
   `y = H(x ⊙ suh) · W_hat · H ⊙ svh`: the two Hadamard passes run on the 1-row activation, not on
   the weight. Only `W_hat` is produced, on the fly, straight from the trellis. (`folded_forward`;
   the test `folded_equals_materialized` pins the two forms equal.)
2. **Codebook LUT — for the CPU only.** `decode_3inst` is a function of a 16-bit window, so a
   65536-entry fp16 table (128 KB) replaces the per-window multiply/`dp4a` chain with one indexed load.
   On the **GPU this is a pessimisation**: a 2-byte random lookup into 128 KB misses L1 and fetches a
   full 32-byte L2 sector per weight, which measured ~2× slower than decoding in registers. The GPU
   GEMV instead uses `decode_mul1`, the mul1 decode folded to pure ALU: the `h2f(bytesum(w·C)+0x6400)`
   intermediate always has exponent 25, so it is exactly the integer `1024 + (s & 0x3FF)`, and the two
   `h2f` constants fold to plain floats.
3. **Batch/vectorize the window read.** Treat the packed tile as a bit array and do the 256
   overlapping 16-bit reads as vector gathers; in C++/HIP this is funnel shifts over 32/64-bit
   words. (Reference: 791 → 111 µs per tile, and one expert `down_proj` reconstruct 6998 → 217 ms
   after 2+3.)
4. **Memory is the point.** A 3 bpw expert is ~0.59 MiB where the fp16 weight is 3.12 MiB, so
   decode GEMV is memory-bound: streams ~5.3× fewer bytes. Prefill (many tokens) instead
   reconstructs `W` to fp16 and uses an HGEMM (`hgemm`), the compute-bound path.
5. **Specialize per K** (1..8 and the 1.5/2.5/3.5 half rates) so the inner loops unroll.

## Verification

Every released bitrate branch is checked on its own weights, not just 3.05: `tools/exl3/fetch_branch_tensors.py`
pulls one expert and one dense linear out of each branch's shards (a few MB) and `strata-exl3-gpu` compares
the GPU decode-GEMV against the CPU reference on them.  Measured on the Radeon PRO W7800:

| branch | expert K | attn K | GPU vs CPU |
| --- | --- | --- | --- |
| 2.05bpw_h4_ng4 | 2 | 4 | rel 2.9e-4 |
| 3.05bpw_h5_ng5 | 3 | 5 | rel 2.9e-4 |
| 4.05bpw_h6_ng6 | 4 | 6 | rel 2.9e-4 |
| 5.05bpw_h6_ng6 | 5 | 7 | rel 2.9e-4 |
| 6.05bpw_h6_ng6 | 6 | 8 | rel 2.9e-4 |

`tools/exl3/codebook.py` and `tools/exl3/reconstruct.py` are a bit-exact numpy transcription of the
codebook, window extraction, tile permutation and Hadamard. `tools/test_exl3_codebook.py` +
`tools/test_exl3_reconstruct.py` (**15 tests, no model needed**) check the constants, the `lop3`
truth table, mul1 against a scalar reference, the window extraction against an independent
bit-by-bit reader for every bitrate 1..8, the LUT against `decode`, `perm` being a bijection, the
Hadamard's Sylvester form and orthonormality, and — the key one — that the **folded GEMV equals the
materialized `x·W`** for random weights, which pins the permutation/Hadamard/sign algebra.

## The n-gram table

The n-gram (PLE) table is **already supported by the engine — from GGUF**: `strata::kernels::PleTable`
(`include/strata/kernels/ngram.hpp`, `src/kernels/ngram.cpp`) reads `per_layer_token_embd.weight` as
**IQ4_NL** (90 B/row), **Q5_0**, or **FP8 E4M3** (160 B/row), with `PleReader` (unbuffered SSD
streaming + row cache, `src/ngram/ple_reader.cpp`), the host `ngram_rows` hash, and the GPU `build_ple`
block (`src/kernels/ple.hpp`). Its geometry is exactly this table's: 320,001,536 rows × 160, 16 rows
per token flattened head-slowest to 2560. The only gap is the EXL3 **row encoding**.

In EXL3 (`exl3_ngram_trellis`, turboderp's `exl3_lib/ngram_codec.py`) the table is
`ngram_embedding.safetensors`:

| tensor | dtype / shape |
| --- | --- |
| `shard_{0..127}.trellis` | I16 `[2500012, 51]` (the rings; 128×2500012 = 320,001,536 rows) |
| `head_bias` | F16 `[16, 160]` |
| `head_offsets`, `head_vocab_sizes` | I64 `[16]` |
| `layer_multipliers` | I64 `[3]` |

A row is a single **160-wide tail-biting ring**: `1 + 160·K/16` little-endian uint16 words where word 0
is an **fp16 scale** and the rest are the `160·K`-bit ring (stream bits `[i·K,(i+1)·K)` are position
i's low K bits; the state stacks the preceding symbols). It decodes to
`mul1_codebook[state_i] · scale + head_bias[head]` — **no Hadamard** (only token-embedding groups
rotate). The branch suffix `ngX` is this table's bitrate (`ng5` → 51 words, `ng4` → 41, `ng6` → 61).

Reference: `tools/exl3/ngram.py` (numpy) and `src/kernels/cpu/exl3.cpp::exl3_ngram_decode_row` (C++),
both bit-exact against each other (`tools/test_exl3_ngram.py`; the fixture's `ngram ring decode` is
0.000e+00). The codebook matches ExLlamaV3's `mul1_codebook` bit-for-bit, and decoding a **real** row
from the shipped file and comparing it to the base model's row gives **correlation 0.99721** (rel-L2
0.107 — 5-bit).

**Integration — done.** `PleTable::open_exl3` (`src/kernels/ngram.cpp`) opens `ngram_embedding.safetensors`:
the 128 ring shards are contiguous, so it is one flat 320,001,536 × `2·(1+160·K/16)` table read through the
**same** `PleReader` (unbuffered SSD + row cache + keep-alive) — no conversion, no duplicate file. Rows
decode with `exl3_ngram_decode_row` (mul1 codebook + fp16 scale) plus the per-head bias, so the whole
`gather`/`build_ple` machinery is unchanged. `src/kernels/ple_exl3_parity.cpp` checks every row of a
synthetic table against the numpy codec, **bit-exact**, in both Direct and Mmap modes (CTest
`ple_exl3_parity`). Per turboderp's discussion #12 the table is a swappable file, and the existing
IQ4_NL/Q5_0/FP8 path already covers dropping in an F16/FP8 table.

## Engine integration plan

The engine only reads GGUF today (`src/artifact/gguf_reader.cpp`), so EXL3 support is staged:

1. **Loader** — *done.* `include/strata/artifact/safetensors.hpp` (mmap reader + header parser) and
   `include/strata/artifact/exl3_model.hpp` (`Exl3Model`: resolves a linear across shards via
   `model.safetensors.index.json`). `strata-exl3-model` reconstructs real experts; the combined
   checksum matches `tools/exl3/check_model.py` on the real model (e.g. layer 0 down_proj ×4 =
   `6bfa3213d22c3096`).
2. **CPU decode** — *done.* `src/kernels/cpu/exl3.cpp` (headers in `include/strata/kernels/cpu/exl3.hpp`)
   implements the codebook LUT, window decode, tile permutation, Hadamard and both the materialized
   and folded paths. `src/kernels/exl3_parity.cpp` checks it against the numpy specification
   (`tools/exl3/emit_fixture.py`) for all three codebooks: **codebook LUT bit-exact, `W_hat`
   bit-exact, reconstructed weight bit-exact fp16, folded GEMV rel-err 2e-7**. It is a normal CTest
   test (`ctest -R exl3`), CPU-only, no model.
3. **GPU decode** — *done.* `src/kernels/cuda/exl3.cu` has both `exl3_reconstruct_weight` (materialize
   fp16 `W`) and `exl3_gemv` (the fused decode-GEMV: `y = H(x ⊙ suh) · W_hat · H ⊙ svh`, no `W` ever
   materialized). `src/kernels/exl3_gpu_parity.cpp` and `exl3_gemv_parity.cpp` check them against the
   CPU reference for all three codebooks including the real `640×2560` expert shape. `strata-exl3-gpu`
   additionally runs the fused GEMV on **real weights** from the downloaded model — gate/up/down
   (K=3), `in_proj_qkv` (2560→10240, K=5) and `lm_head` (2560→248320, K=5) — against the CPU
   reference, all at `rel ≈ 2e-4` (fp16). `exl3_gemv_f32` is the f32-activation wrapper the engine's
   `gemv_quantized` seam will call (also `rel ≈ 2e-4` on real weights). Tuning the expert GEMV (measured,
   layer 0, 10 experts resident, 200 iterations) went **1.9 ms/layer (~11 tok/s) → 0.40 ms/layer
   (~52 tok/s expert-compute-bound, ×48 layers)**, a **~4.7×** gain. What mattered, in order:
   (a) **batch all experts** into `blockIdx.z` — per-expert chains are serial and same-stream kernels do
   not overlap, so throughput had scaled exactly linearly with expert count while the GPU sat idle;
   (b) **drop the codebook LUT** — a 2-byte random lookup into a 128 KB table fetches a whole L2 sector
   per weight (~0.5 GB per GEMV); the procedural mul1 decode folds to ~8 ALU ops and is far faster
   (see below); (c) **raise occupancy** — 16-thread blocks capped at ~24 blocks/CU and the inner loop
   held 152 VGPRs (1 block/CU); 128/256-thread blocks plus `__launch_bounds__(256,6)` brought it to
   ~40 VGPRs / 75%; (d) **word-wise window read** (funnel shift, no integer `%`); (e) **hoist the
   `pinv` permutation** to registers. The per-call `cudaMalloc` was also removed (process-wide workspace).
4. **MoE experts** — *done.* The expert FFN kernel `exl3_moe_ffn` (`src/kernels/cuda/exl3.cu`) runs
   `gate`/`up` (2560→640), `silu(gate)·up → h`, `down` (640→2560) for `n_experts` and accumulates
   `Σ wₑ·downₑ`, reusing one scratch set across all `3·E` GEMV launches. `Exl3ExpertStore`
   (`include/strata/kernels/exl3_experts.hpp`, `src/kernels/cuda/exl3_experts.cu`) uploads a layer's
   routed experts (`Exl3Model` → device `trellis`/`suh`/`svh`) and `run(x, ids, weights, k, out)` calls
   the kernel. Validated on **real** weights: a full layer MoE (10 of 16 experts, layers 0 and 47)
   matches the CPU reference at `rel ≈ 1e-3` (`strata-exl3-moe`). Unlike the GGUF path, EXL3 stores each
   expert as its own tensors (no grouped blob). The engine's expert pool hook is `Exl3ExpertStream`
   (`src/kernels/cuda/exl3_experts.cu`): per token it maps the routed experts' `trellis`/`suh`/`svh` out
   of the safetensors and uploads them, then calls the same kernel, so no layer is resident ahead of time.
5. **Prefill** — reconstruct-to-`half` plus `hgemm`, matching the engine's existing MMQ choice.
6. **n-gram table** — *done.* `PleTable::open_exl3` reads the shipped `ngram_embedding.safetensors` through
   the existing SSD-streaming `PleReader` (see "The n-gram table" above).
7. **Dense linears** — *done.* `WeightRef::exl3` (`include/strata/core/weights.hpp`) carries a
   device `Exl3Mat`, and `gemv_quantized` (`src/core/layer.cpp`) runs `exl3_gemv_f32` for it;
   `sform_of`/`plane_ptrs` early-out for EXL3 tensors. `src/artifact/exl3_pack.cpp` builds the engine
   `WeightTable` from the model directory (1079 roles on the 3.05bpw model: the EXL3 linears attached as
   `Exl3Mat`, the norms/router/hyper-connection/GDN parameters/embedding staged in the engine's own form),
   and `generate.cpp` runs it with `--exl3 <dir>` (plus `--ple-exl3 <ngram_embedding.safetensors>`).

## The engine's form of the weights

An EXL3 model is a straight quantization of the HF checkpoint, so `src/artifact/exl3_pack.cpp` has to
reproduce `llama.cpp`'s converter (`conversion/qwen4exp.py` and the mixins it inherits from
`conversion/qwen.py`) before the engine's kernels see the weights.  The engine's own GGUF packs are built
by that converter, so its rules are the contract.  Four of them are easy to miss, and all four were
wrong in the first working loader — it ran, and produced fluent garbage:

| what | HF stores | the engine reads | why |
| --- | --- | --- | --- |
| the v heads | grouped by k head, `[G0_v0 G0_v1 G0_v2 G1_v0 ...]` | tiled, `v` head `h` paired with `k` head `h % 16` | `gdn_step_kernel`'s `src = h % h_k`; 16 k heads, 48 v heads |
| `A_log` | `log(A)` | `ssm_a = -exp(A_log)` | the GDN gate is `softplus(alpha + dt) · ssm_a` and decays with `exp(gate)` |
| the zero-centred gammas | `w` | `1 + w` | `hc_norm`, `q`/`k` norm, the indexer layernorms, the PLE norms; `linear_attn.norm.weight` is NOT one |
| the indexer q/k projections | `[in, out]` after reconstruction | `[out][in]`, a row of `n_in` per output | `bf16_gemv`'s `row = w + o * n_in` |

The v-head reorder applies to `in_proj_qkv`'s v rows, `in_proj_z`, `in_proj_a`/`in_proj_b`, `A_log`,
`dt_bias`, the conv's v channels and `out_proj`'s columns.  On an EXL3 linear the reorder is done on the
**trellis**, not on a materialized weight: a v head is 128 = 8 whole 16-row tiles, so it is a permutation
of `j` tiles (and `svh`) or `i` tiles (and `suh`).

Each rule is checked on the real model: `strata-exl3-pack <dir> --dump <out>` writes the staged bytes of
22 roles, and `tools/exl3/check_engine_form.py <dir> <out>` recomputes them from the safetensors with an
independent numpy implementation of the converter's rules — **22/22 roles match** on
`Qwen3.8-Flash-Next-exl3` 3.05bpw, byte-exact for the trellis permutations.

## Running the model

Measured on one **Radeon Pro W7800** (gfx1100, 32 GB), 8-thread CPU pool, the model on a SATA SSD,
`--expert-cache 0` (every routed expert is streamed from disk):

| step | measured |
| --- | --- |
| `strata-exl3-pack --run` | 1079 roles, 2117.4 MiB EXL3 linears, 2679.3 MiB staged, 301 dense projections, 0 faults |
| load (`--exl3`) | 15.4 s |
| prefill | 9.0 tok/s (14 tokens) |
| decode | 11.9 tok/s (40 tokens, capital-of-France prompt, steady state); 3.0-4.4 tok/s on a cold page cache |
| expert FFN | 0.36 ms/layer on the GPU (48 layers); 493 ms/layer on one CPU thread |
| layer 0 / 47 vs CPU | rel 1.1e-3 / 1.0e-3 |

Routed experts are cached in VRAM: `--expert-cache N` keeps N experts per layer (`auto` or unset sizes it from the
free VRAM after the stream's own buffers, 0 turns it off).  Without it the stream reads and uploads every routed
expert on every token: 8 experts x 48 layers x 1.86 MB = 890 MB per token at 3.05bpw, and the router picks most of
the same experts again.  Measured on the same machine, same prompt, third run of each (steady state):

| `--expert-cache` | decode | prefill |
| --- | --- | --- |
| 0 | 5.5 tok/s | 4.9 tok/s |
| auto (315 slots/layer, 26.3 GiB) | 11.9 tok/s | 9.0 tok/s |

**Where a token's time goes, and why the first tokens are not the steady state.**  A conversation's routing
starts by exploring and saturates: measured on the W7800, a 220-token run's per-token cache hit rate went
**82% -> 90% -> 92.5% -> 98.8% -> 99.0%**, and the uploads per token fell from 400+ ms to **20-37 ms**.  In that
state a token measures ~76-100 ms (~10-12 tok/s): the dense work 29 ms (Q8_0), the routed experts' FFN ~17 ms,
the host's wait on that FFN ~19 ms, the hyper-connections ~7 ms, the trickle of first-time experts 20-37 ms,
and the smaller kernels ~10 ms.  With the reads priced at zero the same parts sum to ~46 ms (~21 tok/s), so the
remaining first-time experts are what stands between this machine and 16 tok/s.

**The prefill is the same reads, all of them at once.**  A prompt's tokens route to experts the VRAM cache has
never seen, and those are compulsory: measured, a 15-token prompt took 14-22 s (0.6-1.0 tok/s) with the pool's
uploads at 113-2121 ms per token and its FFN at 1.8-3.6 ms.  The distinct experts of a 15-token prompt are
~85% of the sum of its tokens' routes, so batching the prompt's experts would save that ~15% and nothing more;
what makes the same model's prefill 447-966 tok/s on a 64 GB machine (docs/AMD_HIP_PERFORMANCE.md) is that its
whole expert set is resident in RAM, not the batch size.  This model's experts are 45 GB and the machine has
24.5 GiB of free VRAM plus ~20 GB of usable RAM, so they do not fit and the reads are structural.

**The expert GEMV's geometry, swept on the real layer.**  `strata-exl3-moe` runs the whole expert FFN (gate,
up, down, both scale transforms, the silu) on layer 0 with 10 experts, and the tiles per block was swept on it:
T=1 0.534 ms/layer, T=2 0.385 (what it was), T=3 0.321, **T=4 0.308**, T=5 0.319, T=6 0.327, T=7 0.342,
T=8 0.325.  Sweeping the k-group count up from `(ki+15)/16` only lost (x2 0.330, x4 0.543).  T=4 is what the
kernel uses, -20% on the expert FFN, 3.7 ms/token of a saturated token, on every machine and every GPU.

Two more passes after the cache:

- **the dense GEMV** (`gemv_kernel`) recomputed each weight's window geometry (3 multiplies and two integer
  modulos) inside the k loop, ran 128 threads with one output tile per block, and split k 8 rows at a time: 100 us
  per call, 30 ms of a 130 ms token, 40-115 GB/s.  Hoisting the geometry, 256 threads, two output tiles per block
  and a k split that keeps ~256 blocks in flight took it to 22.6 ms and the decode 8.9 -> 10.2 tok/s.
- **the misses are read with `pread`, spread over four threads.**  A page-cached tensor copied through the mmap
  faults every 4 KiB page (measured: 0.3 GB/s against 10.4 GB/s with `preadv` into a reused buffer), and one
  thread did all of it.  `Exl3Model::locate` gives the file offset; the workers read into pinned staging (two
  buffers each, with an event, because the DMA out of a staging buffer is asynchronous) and the caller's stream
  orders the copies before the FFN.  Upload 24 -> 13 ms per token, decode 9.6 -> 10.8 tok/s.

Two more passes on the same idea (the pieces, not the schedule):

- **the resident half's FFN is enqueued before the staging workers start** and the miss half follows the staging
  copies, with a small kernel scattering both parts back into the router's order.  The copies moved to a stream
  of their own: on the caller's stream they queued behind that FFN, and the double-buffered workers then blocked
  on their own DMA events for its whole length (staging 24 -> 37 ms/token when it was first tried on one stream).
- **`gr_inject_kernel` and `gr_down_kernel` got four partial sums.**  `gr_inject` is one block of four warps
  walking 10,240 elements in 320 dependent steps: 75 us per call, 7.2 ms/token over 96 calls, from 4 warps out of
  the whole machine.  (The 4-accumulator experiment recorded in `gr_down_kernel` came out neutral on the card of
  that round; on the W7800 it helps.)

What the token is made of now (W7800, 40 tokens, one decode token): the expert FFN 17 ms (the miss half ~5), the
dense GEMVs 23 ms, the expert reads 13-27 ms (the cache hit rate is 70-94%, so this tracks the routing), the
hyper-connections ~7 ms, and the rest in the layer's smaller kernels.  The per-layer chain is serial by
construction (router -> host reads -> expert FFN -> combine -> next layer), so the next wins are in the pieces,
not in the schedule.

**Why a GGUF pack of the same model decodes faster.**  Measured, not guessed:

- **A trellis weight costs about two loads and eleven ALU ops to decode**; a ggml block quant's `vec_dot` costs
  about one load per 4-5 weights and 1-2 ALU.  The dense GEMV was load-instruction-bound: 3.5 G dense weights x 2
  loads per token, 23 ms.  **The dense linears are therefore requantized to Q8_0 at load**
  (`stage_exl3_requantized`) and served by the engine's native MMVQ, which is the path a GGUF pack's dense
  projections already take.  Measured on the W7800, a 2560x10240 matrix: the decode-GEMV kernel **106.8 us**
  against the native Q8_0 MMVQ kernel's **31.8 us**, 3.4x.  The first attempt used Q5_K and was rejected on the
  measurement: a 5-bit block quant on top of the trellis decode's own 10-bit codes measures a **4%** round-trip
  error, and the model then answers "surface表面的表面的" - Q8_0's round-trip error is **0.65%** and the model is
  correct.  Q8_0 is 1.7x the bytes of the 5-bit trellis it replaces (3.7 GiB against 2117 MiB of dense linears),
  which the expert cache pays for: +31 MiB per layer, about 16 expert slots, measured 24.57 GiB free -> **289
  slots/layer**.  It costs 5.8 s of load time (21.2 against 15.4, measured), streamed through small row-chunk
  buffers: holding a whole matrix on the host (7 GB of fp16
  for the model) leaves that much anonymous dirty memory behind, the kernel evicts the model's page cache for it,
  and the expert reads then come off the SATA SSD - measured 1132 MiB of disk reads for 8 tokens and
  ~500 ms/token of uploads against 14-27 ms.  `STRATA_EXL3_QUANT_THREADS` sets the quantizer's thread count and
  `STRATA_EXL3_DENSE=exl3` keeps the trellis.

  **The bytes are also a cost on a machine whose reads are the bottleneck.**  The Q8_0 dense weights are 1.5 GiB
  more than the trellis, and the expert cache is sized from what is free after them: 17 fewer slots per layer,
  about 4.6% of the miss rate, which on a cold machine is ~41 MB/token of extra reads, ~130 ms at the SATA
  SSD's 300 MB/s, against the 18 ms the faster GEMV saves.  `STRATA_EXL3_DENSE=exl3` is the arm for that
  machine; the measured A/B is below.

  The bytes are the price and they land in the batched prefill, which reads the weights once per batch: the dense
  work of a 14-token prompt measures **20.5 ms against 31.6 ms** with Q8_0 (683.8 against 442.5 tok/s), the same
  1.5x the byte counts predict.  In the pooled prefill the dense projections are 20-32 ms of a 1.55 s batch
  (9.0 tok/s measured), so ~1% there, while the decode reads those weights every token and gains the 18.4 ms:
  with the experts skipped, the dense work goes **21.1 -> 34.5 tok/s** (47.4 -> 28.9 ms/token, four runs).
- **The v-head reorder is part of the weight, and a requantized copy has to carry it.**  The GDN's v projections
  are staged with their v heads permuted (`Vr::OutV`/`Out`/`In`: `attn_qkv`, `attn_gate`, `ssm_out`) and the
  activations are permuted to match; `stage_exl3` reorders the trellis and the scales, so the requantizer runs the
  same host-side reorder before it reconstructs.  Skipping it leaves those rows in the wrong order and every one
  of the three roles then answers nonsense - measured, and it is why the QSA roles (`Vr::None`) were correct in
  the same build while the GDN roles were not.
- **The experts cannot go on the CPU pool the way a GGUF pack's do.**  The engine overlaps a GGUF pack's CPU
  expert work with the GPU's dense work; the CPU's EXL3 folded GEMV measures **493 ms/layer** against the GPU's
  **0.36 ms/layer** (`strata-exl3-moe`, 10 experts), 1370x.  The trellis decode has to stay on the GPU, and its
  streaming is on the per-layer critical path.

The output tokens are unchanged by the cache: the same expert bytes and the same FFN kernels either way.  The
engine's QSA sparse selection is not run-to-run deterministic on its own (about 1 run in 4 differs in a later
token), which is independent of the EXL3 path and of these changes.

The prompts answer correctly, with and without PLE:

    <|im_start|>user
    What is the capital of France?<|im_end|>
    <|im_start|>assistant
     thinking
    The user is asking a simple factual question: the capital of France. The answer is Paris.
    This is straightforward and requires no complex reasoning.
    </think>

    The capital of France is **Paris**.<|im_end|>

    <|im_start|>user
    Write a hello world program in Python.<|im_end|>
    <|im_start|>assistant
     thinking
    The user wants a simple "Hello, World!" program in Python. This is the most basic programming
    task. I'll provide the standard one-liner.
    </think>

    ```python
    print("Hello, World!")
    ```

    This prints the text `Hello, World!` to the console. Run it with:

    ```bash
    python hello.py
    ```

`--stop-eos` stops at `<|im_end|>`; without it the sampler keeps going past the end of the turn.

## Tokenizer

The EXL3 model ships HuggingFace tokenizer files, but the tokenizer is the **same byte-level BPE as the
GGUF** (247,587 merges, the `qwen35` pretokenizer pattern), so no new tokenizer is needed:
`tools/exl3/tokenizer_to_pack.py` reshapes `tokenizer.json` + `merges.txt` + `chat_template.jinja` +
`config.json` into the pack's `tokenizer/` layout (`vocab.json`, `merges.txt`, `token_type.json`,
`tokenizer.json`, `chat_template.jinja`). Measured on the real model: vocab 248320, merges 247587,
and the engine's `Tokenizer` round-trips a boundary corpus **16/16** (`decode(encode(s)) == s`).

Special tokens (added, ids 248044–248076): `special:true → CONTROL(3)` (matched only with
`parse_special`), `special:false → USER_DEFINED(4)` (always matched), base vocab `→ NORMAL(1)`. No BOS;
EOS is `<|im_end|>` = 248046 (also 248044). A tiktoken-based engine could reproduce the same ids but has
no C ABI and buys only speed that is irrelevant next to the forward pass, so it is not used.
