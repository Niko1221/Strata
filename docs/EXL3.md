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
4. **MoE experts** — *core done.* The expert FFN kernel `exl3_moe_ffn` (`src/kernels/cuda/exl3.cu`) runs
   `gate`/`up` (2560→640), `silu(gate)·up → h`, `down` (640→2560) for `n_experts` and accumulates
   `Σ wₑ·downₑ`, reusing one scratch set across all `3·E` GEMV launches. `Exl3ExpertStore`
   (`include/strata/kernels/exl3_experts.hpp`, `src/kernels/cuda/exl3_experts.cu`) uploads a layer's
   routed experts (`Exl3Model` → device `trellis`/`suh`/`svh`) and `run(x, ids, weights, k, out)` calls
   the kernel. Validated on **real** weights: a full layer MoE (10 of 16 experts, layers 0 and 47)
   matches the CPU reference at `rel ≈ 1e-3` (`strata-exl3-moe`). Unlike the GGUF path, EXL3 stores each
   expert as its own tensors (no grouped blob). What remains is the **session/layer-loop wiring**:
   routing (`ffn_gate_inp` + top-k), the shared expert, and the dense linears (attention/GDN/router) via
   `exl3_gemv`, so a full forward runs end to end.
5. **Prefill** — reconstruct-to-`half` plus `hgemm`, matching the engine's existing MMQ choice.
6. **n-gram table** — *done.* `PleTable::open_exl3` reads the shipped `ngram_embedding.safetensors` through
   the existing SSD-streaming `PleReader` (see "The n-gram table" above).
7. **Dense linears** — *seam in place.* `WeightRef::exl3` (`include/strata/core/weights.hpp`) carries a
   device `Exl3Mat`, and `gemv_quantized` (`src/core/layer.cpp`) runs `exl3_gemv_f32` for it;
   `sform_of`/`plane_ptrs` early-out for EXL3 tensors. Inert until the loader fills it (the `exl3` field
   defaults null, so pack/native paths are unchanged; the full `strata` executable still builds).
   Remaining: the EXL3 model-open path in `generate.cpp` (populate the descriptors + the expert arm).

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
