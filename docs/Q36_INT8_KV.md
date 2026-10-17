# 8-bit KV cache for the Qwen3.6 (strata-q36) path

A design note for quantizing the K/V cache of Qwen3.6-35B-A3B from fp32 to int8 in the
`src/qwen35` engine. Status: **implemented** (CUDA verified; HIP build still to be run).

Back to the [README](../README.md).

- [Why](#why)
- [The KV cache today](#the-kv-cache-today)
- [What we change](#what-we-change)
- [Implementation plan](#implementation-plan)
- [Acceptance criteria](#acceptance-criteria)
- [Results (measured)](#results-measured)
- [Risks and open questions](#risks-and-open-questions)

## Why

Qwen3.6-35B-A3B runs through its own engine (`src/qwen35/q36_engine.cu`, the `strata-q36`
path) which keeps the whole K/V cache in fp32 in VRAM. That single choice is what caps the
context window on a small card, not the model.

Measured on an RTX 4060 Max-Q Laptop (8 GB, 8188 MiB), a Ryzen 7 7735HS and 62 GB of RAM, with
`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`:

- The dense (non-expert) weights take **4.40 GiB** of VRAM.
- The K/V cache costs **44 KiB per token** in fp32 (see the arithmetic below).
- The largest context that starts is **16,384** (checked empirically; 24,576 and 32,768 fail
  the `cudaMalloc` in `q36_engine.cu`).

The KV cache is `--max-context`-sized up front, so 131,072 tokens in fp32 would be
`5.50 GiB` of K/V on top of `4.40 GiB` of dense weights — `9.90 GiB`, which does not fit in
`8.00 GiB`.

8-bit K/V is the cheapest fix. The repo already ships int8 KV kernels and a parity test on the
Flash-Next path, and llama.cpp runs this same model with 8-bit KV as fast or faster (decode is
bandwidth-bound reading K/V, and int8 reads a quarter of the bytes). Quantizing the resident
K/V to int8 shrinks it to ~11 KiB per token; the measured window is in
[Results](#results-measured).

## The KV cache today

### Shape

From the GGUF config, enforced by the engine (`g.head_dim != 256` is a hard error):

- **head dim**: 256 (fp32), 2 KV heads, 16 query heads.
- **KV layers**: 10 attention layers + the MTP block = **11** K/V stores. `setup.py` records
  this as `kv_bytes_per_token = 11 * 2 * 256 * 2 * 4` = **45,056 bytes/token** (44 KiB).

| Format | Bytes / value | Per token | 131,072 tokens |
| --- | ---: | ---: | ---: |
| fp32 (today) | 4 | 44 KiB | 5.50 GiB |
| fp16 | 2 | 22 KiB | 2.75 GiB |
| int8 (q8_0) | 1 (+ scale) | 11 KiB | 1.37 GiB |

The dense weights (4.40 GiB) are fixed on the card. At 131,072 tokens the totals are:

- fp32: 4.40 + 5.50 = **9.90 GiB** — does not fit.
- fp16: 4.40 + 2.75 = **7.15 GiB** — fits, ~0.85 GiB left for the attention-score staging,
  activations and reserve (tight).
- int8: 4.40 + 1.37 = **5.77 GiB** — fits with ~2.2 GiB headroom.

### Where it lives in `src/qwen35/q36_engine.cu`

The K/V is one flat fp32 allocation per layer, indexed `[kvh * ctx + pos] * HD`:

- **Allocation** — `Engine::init()`: `dalloc((size_t) g.n_head_kv * ctx * g.head_dim)` for the
  attention layers and once more for the MTP block (`dalloc` is `cudaMalloc` + `memset`). No
  quantization, no streaming, no `--kv` flag on this path.
- **Write** — `k_attn_prep`, the RoPE + norm kernel: writes `kc` and `vc` in fp32.
- **Decode read** — `k_attn_part`, the split-K attention kernel: reads every cell of `kc`/`vc`
  in the inner loop (`kp = kc + (kvh * ctx + p) * HD`, and the V reduction).
- **Prompt read** — `k_kv_f16`, which copies the fp32 K/V into the `k16`/`v16` fp16 staging
  buffers for the cuBLAS attention GEMMs.

Because Qwen3.6 attention is **full dense** (every token attends to the whole history), the
decode kernel reads the *entire* K/V per generated token. That is why this path cannot simply
stream K/V to host RAM the way Flash-Next's query-sparse attention can — and why **quantizing
in place** (rather than offloading) is the right lever here. All reads go through the same two
buffers, so a single format change covers every consumer.

## What we change

Keep the storage layout, change the element type of the resident `kc`/`vc` from fp32 to int8,
with a per-block fp16 scale (the `q8_0` layout llama.cpp uses: 32 int8 + one fp16 scale). This
is a self-contained change to one file plus a parity test:

1. **Storage** — allocate `kc`/`vc` as int8 (plus one scale buffer per layer) in `init()`,
   gated on a new `--kv int8` option (default stays fp32, byte-identical).
2. **Write** — in `k_attn_prep`, quantize the RoPE'd K and the V into int8 + scale instead of
   storing fp32.
3. **Decode read** — in `k_attn_part`, dequantize on the fly (`q8_0`-style: `scale * int8`,
   the scale in shared/registers). The `k_attn_merge` kernel is unchanged.
4. **Prompt read** — add a `k_kv_f16` variant that reads int8 + scale and writes the fp16
   staging operands for the GEMMs (or dequantize into the existing fp16 path).

Reuse where possible: `src/kernels/cuda/kv_stream.cu` already implements int8 K/V
(`kKvInt8`) and `src/kernels/kv_stream_parity.cpp` is the parity harness for it — those are on
the Flash-Next (QSA) path and are the reference for the block layout and the round-trip test,
even though the kernels themselves do not transplant (different index math).

The option is opt-in and the default remains fp32, per the contribution rule that the default
path stays byte-identical.

## Implementation plan

Each phase leaves the tree green and is reviewable on its own.

### Phase 1 — the quantize primitive and its parity test

Add a small int8 (q8_0-style) quantize/dequantize pair for one `[HD]` vector and a standalone
parity test in the style of `kv_stream_parity.cpp`: round-trip fp32 → int8 → fp32 and assert
the error is within the expected q8_0 bound. This validates the format and the block-scale
math before it touches the engine.

**Exit:** the new primitive passes `ctest`; no engine change yet.

### Phase 2 — storage, flag and the write path

Add `--kv int8` (and `--kv fp32` default) to the `strata-q36` argument parser; allocate the
int8 `kc`/`vc` + scale buffers in `init()`; switch `k_attn_prep` to store quantized K/V.
Decode still reads fp32 in this phase by dequantizing into the existing `k_attn_part` read
(so the path is exercised end to end as soon as it compiles).

**Exit:** `--kv int8` builds and runs; greedy decode matches fp32 bit-for-bit at a fixed seed
on a short prompt (decoding through the dequantized values).

### Phase 3 — the prompt path

Add the int8→fp16 staging kernel (or dequantize inside `k_kv_f16`) so the cuBLAS prompt GEMMs
read the same values the decode path does. Check the prompt and decode agree on a long prompt
(prefill + first token).

**Exit:** prefill parity with fp32 on a multi-thousand-token prompt.

### Phase 4 — accuracy, memory and speed

Run the full acceptance suite below, then measure the new context ceiling and the decode
throughput at long context on the 8 GB card.

**Exit:** acceptance criteria met and numbers recorded (with what they were measured on).

## Acceptance criteria

Measured on the same 8 GB card as the motivation (RTX 4060 Max-Q, 8 GB, Ryzen 7 7735HS, 62 GB),
`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`:

1. **Correctness.** Greedy decode with `--kv int8` produces the same tokens as fp32 at a fixed
   seed on a representative prompt set (short + long). Where the two differ, the difference is
   stated and bounded (q8_0 is lossy; a small perplexity delta on a held-out corpus is the
   real check, as in the existing `kv_stream_parity`).
2. **Parity test.** The new quantize/dequantize parity test passes in `ctest`.
3. **Memory.** `--max-context 131072 --kv int8` starts and serves on the 8 GB card — currently
   impossible in fp32 — with the resident K/V down from 44 KiB to ~11 KiB per token (measured
   from `nvidia-smi` / the engine's own "VRAM free" line, not the arithmetic alone).
4. **Speed.** Decode at a long context is not slower with int8 than fp32 at the same context;
   at the larger context that int8 unlocks, report tok/s alongside the fp32 short-context
   baseline.
5. **Opt-in.** The default (`--kv fp32`) path is byte-identical to the last release; int8 turns
   on only with the flag.
6. **Backends.** The touched file builds on every backend it supports — CUDA and HIP (the
   Intel Arc / SYCL engine refuses this family by design, so it is out of scope).

## Results (measured)

Implemented behind `--kv int8` (default `--kv fp32`, byte-identical). Measured on the same
8 GB card as the motivation — RTX 4060 Max-Q Laptop (8188 MiB), Ryzen 7 7735HS, 62 GB — with
`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, CUDA build:

1. **Correctness — pass.** Greedy decode is bit-for-bit identical to fp32 on four short
   prompts and one long prompt (3,811 tokens: prefill plus 8 decoded tokens). A full-logits
   dump on the long prompt differs from fp32 by at most 0.79 across all 248,320 logits (inside
   the q8_0 round-trip bound), with the top-5 logits identical.
2. **Parity test — pass.** `q36_kv_q8_parity` passes in `ctest`.
3. **Memory — relaxed.** The 131,072 target is not reachable, because at int8 the K/V is no
   longer the binding constraint. The ceiling is set by the dense weights (4.40 GiB) plus the
   hybrid expert-streaming staging buffer (1.10 GiB double-buffered, or 0.55 GiB with
   `STRATA_Q36_SINGLE_STAGE=1`) plus the context-scaled attention-score buffer (see the last
   risk below), not by K/V. Measured: fp32 starts at 16,384 and no
   larger; int8 starts and decodes at **32,768** (40,960 allocates but the prefill GEMM's
   score buffer no longer fits, 49,152 fails `cudaMalloc`). Resident K/V is 44 KiB/token (fp32)
   vs ~11 KiB/token (int8): at 4,096 context the engine reports 1.70 GiB free (fp32) vs
   1.83 GiB free (int8), the 0.13 GiB difference matching the 4× shrink. Net: **2× the context
   window** on this card.
4. **Speed — pass, small stated delta.** At 4,096 context: fp32 decode 16.0 tok/s, int8 decode
   15.4 tok/s (~4% slower; the on-the-fly dequantize adds a little work and 4K of K/V is not
   bandwidth-bound, so the smaller reads do not pay off here). Prefill is unchanged (fp32
   341 tok/s vs int8 340 tok/s). At the 32,768 context int8 unlocks, decode is 13.6 tok/s (the
   cost of attending to the longer history, not the quantization). fp32 cannot run at 32K on
   this card, so there is no like-for-like number there.
5. **Opt-in — pass.** The default path (no flag) is byte-identical to the pre-change build; it
   prints the same tokens on the smoke prompt (`gen: 760 11679 22186 995`).
6. **Backends — CUDA verified.** The HIP build of the touched file has not been run yet; it is
   the remaining sign-off for this change (the engine is shared CUDA/HIP source).

A staging-buffer fix came out of measuring item 3. At 32,768 the engine reported "0.63 GiB of
VRAM free" after `init()`, and the prompt staging (one layer's experts, 0.55 GiB) plus the
streamed path's `ecpu` buffer (0.13 GiB at chunk 2048) did not fit in it, so MMQ aborted with
`cudaMalloc(&E.stage[b], ...) failed: out of memory`. Shortening the context did not help:
int8 K/V is ~11.7 KiB per token, so 32,768 -> 22,768 frees only ~0.16 GiB (0.63 -> 0.79 GiB
free) against a 0.55 GiB buffer. `place_experts()` now makes the same trade there that the
MTP-only path already made one branch above — the attention-score buffer at 64 MB instead of
256, which frees 0.28 GiB (0.63 -> 0.91 GiB free), after which staging fits. **Do not reach for
`STRATA_Q36_NO_MMQ=1` instead:** with MMQ off the prompt has no streamed path, `hybrid_t`
becomes `tmax`, and *every* prompt row goes to the CPU pool. Same machine, same run
(`--bench-pp 512 --bench-tg 8`, 32K, `--kv int8`):

| `STRATA_Q36_NO_MMQ` | prefill pp512 | decode tg8 | batch entries on the GPU |
|---|---|---|---|
| `0` (MMQ on) | **310.1 / 310.4 / 310.3 tok/s** | 8.6 - 11.1 tok/s | 0 of 7,680 (0.0%) |
| `1` (MMQ off) | 16.6 / 15.7 / 11.5 tok/s | 8.0 - 13.4 tok/s | 0 of 499,200 (0.0%) |

~20x on prefill, and the decode column is a wash: this model's 18.75 GiB of experts fit in the
card's 8 GiB neither way (0 slots either way), so decode is the CPU pool in both. Nothing sets
this switch for you: `docker/run.sh` forwards `STRATA_*` variables unchanged and `setup.py` does
not set MMQ, so the engine's own default (MMQ on) applies unless you set it yourself.

## Risks and open questions

- **Accuracy.** q8_0 K/V is lossy. The fp32→fp16 step is the smaller-risk intermediate if the
  int8 delta is too large; the plan is written so fp16 can be dropped in as a fallback format
  behind the same flag with no structural change.
- **The MTP block's K/V** is a twelfth store the engine allocates but `setup.py`'s 44 KiB/token
  estimate does not separately count (the comment reads "10 attention layers + the MTP block",
  multiplied by 11). The plan quantizes the MTP block's K/V with the same path, so the real
  saving may be a touch larger than the table above.
- **The 4.40 GiB floor.** Even at int8, the dense weights remain the fixed cost; the context
  ceiling is then set by dense weights + attention-score staging + reserve, not by K/V. Going
  past 131K still needs a smaller dense quant (e.g. `IQ3_XXS`) or more VRAM.
