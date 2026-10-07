# DFlash: standalone block drafter for Qwen3.8-Flash-Next

Experimental support for the DeepSpec DFlash drafter
[`PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash`](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash)
as a **standalone draft model** next to Strata's own MTP layer:

```text
Qwen3.8-Flash-Next target (any supported quant)
        +  standalone DFlash artifact (GGUF, BF16, 498M params, no embed / no LM head)
        ↓
one parallel draft pass per cycle  →  existing Strata Verifier
```

Nothing here changes target-only decoding or MTP when DFlash is not selected.

## Semantics (pinned from the references)

Derived from, in priority order when they disagree: (1) the PixelML serving adapter and
trainer for this exact checkpoint (`PixelML/deepspec-qwen38-flash-next`), (2) llama.cpp
`src/models/dflash.cpp` + `common/speculative.cpp`, (3) z-lab `dflash/dflash/model.py` and
`deepseek-ai/DeepSpec`.

### Target feature taps

Five taps, target layers `[3, 15, 23, 35, 43]` (`target_layer_ids`). At each tap take the
**HC-contracted native-width (2560) residual** — the `mixed` output of the tapped layer's
own attention-half HyperConnection read — not the raw 4×2560 stream, not the MLP input, not
the final hidden state. The PixelML exporter captures at vLLM aux boundaries
`tap + 1 = [4, 16, 24, 36, 44]`, each `attn_hyper_connection.mix(...)[1]`; Strata's
equivalent is the attn-half HC read (`gr_read` / `fused_gr_read`) output `mixed` at layers
`[4, 16, 24, 36, 44]`, which equals the contracted residual *after* layers `[3, 15, 23, 35,
43]` including those layers' FFN write-back. One feature = `concat` in tap order, `[12800]`.

### The drafter

Dense 5-layer Qwen3-style decoder, 58 BF16 tensors, all shapes checked at load:
`fc` [2560×12800], `hidden_norm` [2560], `norm` [2560], and per layer 0..4:
`input_layernorm`, `post_attention_layernorm` [2560], `self_attn.{q,k,v,o}_proj`
[6144/512/512/2560 × 2560], `self_attn.{q,k}_norm` [256], `mlp.{gate,up}_proj` [7680×2560],
`mlp.down_proj` [2560×7680]. RMSNorm eps 1e-6, SiLU MLP, GQA 24 Q / 2 KV, head_dim 256,
scaling 1/16, NeoX RoPE theta 1e7 (full head), no biases. `embed_tokens` and `lm_head` are
**bound from the target at load** (the checkpoint ships neither).

### Context K/V (the "KV cache injection")

For every committed target position `p` the drafter holds a context cell computed once:

```text
ctx(p)   = rmsnorm(fc(feature(p)), hidden_norm.weight, eps 1e-6)     # one vector, all layers
k_l(p)   = rope_neox(k_norm_l(k_proj_l(ctx(p))), p)                  # k_norm applies, per layer
v_l(p)   = v_proj_l(ctx(p))
```

The same `ctx(p)` feeds every layer (vLLM `_project_context_kv`: one fused GEMM over the
normed context; "unnormalised-by-layer context K/V"). Context cells are written **after**
verification, from the verify window's tap rows for the window rows that were committed, in
place at their positions. A rejected position's cell is simply overwritten when that
position is committed later — the same position-addressed overwrite the Verifier already
uses for target K/V; the drafter needs no rollback of its own.

### The DeepSpec anchor layout (query_zero_predicts_next)

This checkpoint is trained with DeepSpec's anchor-as-first-prediction convention, which
differs from generic DFlash's `1+N` fill-in layout (vLLM raises
`sample_from_anchor=True is not supported for DFlash` for it; the PixelML adapter teaches
the kernel `sample_offset = 0`). The concrete rule, in the `_prepare_dflash_inputs_kernel`
wording: with `P` the position of the last valid **context** row, query `j` (0-based, `K`
queries, `j < K`) sits at rope position `P + 1 + j`, query 0's input token is the last
committed token (the anchor id) and queries `1..K-1` are `mask_token_id`; every query is
sampled and `sample_pos = query_pos + 1` — each query's distribution is for the position
**after** its own. So candidate `j` = argmax(query `j`) lands at position `P + 2 + j`,
the `K` candidates cover `[P+2, P+K+1]`, and the verify window is
`[anchor token] + K candidates` = `K + 1` rows — exactly Strata's existing window contract,
with the anchor re-fed as window row 0.

Written out per Strata cycle (anchor = last committed token `x` at position `P`, the draft
cache holds context cells for `[0, P-1]`):

1. Query forward, one pass: `K` rows at positions `[P .. P+K-1]`, inputs
   `[embed(x), embed(mask) × (K-1)]` (raw target embedding, scale 1). Per layer: q-norm,
   rope; query K/V cells written at the query positions; attention is **non-causal** among
   the query rows and full over the context cells (vLLM `get_draft_attn_causal`,
   `dflash_config.causal = false`). Final `norm`, then the target's `lm_head`:
   candidate `j` = argmax(row `j`).
2. Verify: `Verifier::run(K+1, [x, c0..c(K-1)], pos0=P)`; commit `a+1` rows; the target
   correction (bonus) `b = outv[a]` is committed at `P+a+1`.
3. Context update: write feature cells for positions `[P .. P+a]` from the window's tap
   rows `[0..a]`. Position `P+a+1` (the bonus) has no valid features yet — it was never a
   committed target input — and is deliberately left to the next cycle's query 0.
4. `P ← P+a+1`, `x ← b`.

The first cycle after prefill is the same with the prompt's taps as context cells
`[0, N-2]` and the prompt's last token as query 0's input; the prompt's last position
`N-1` *is* a real committed row, so its context cell exists (from the prefill taps) and
query 0 sits at `N`.

Block size: the checkpoint trains `block_size = 7` = `K_max` (7 queries = anchor + 6
masks; llama.cpp: "anchor-first yields a full block_size draft tokens"). Serving at `K < 7`
takes the identical proposer path (the PixelML plugin allowlist `(4,5,7)` was "a scope
marker, not an architectural limit"). `K ≥ 8` is refused: it exceeds both the trained
block and `kVerifyMaxT - 1`. In Strata terms `--spec T` stays the **window size**, so
`K = T - 1` and `T ≤ 8`.

### Greedy only

v1 supports `temperature = 0` acceptance (the verifier's exact-match against the row pick).
A sampled request (`temperature > 0`) with `--dflash` is served **without** the drafter
(explicit `dflash: sampled request decoded without the drafter` line) — greedy draft
acceptance under sampling semantics is not proven for this checkpoint; coupled sampling is
a later milestone. `--mtp` and `--dflash` are mutually exclusive; the flag combination is
rejected at startup.

## Strata integration

| Piece | Where |
|---|---|
| CLI: `--dflash FILE.gguf`, `--dflash-block K`, `--dflash-window N` | `src/program/generate.cpp` |
| Artifact: GGUF v3 reader (existing `strata::GgufFile`), metadata + tensor validation, BF16 → device | `include/strata/core/dflash.hpp`, `src/core/dflash.cpp` |
| Taps: verify window writes the 5 contracted residuals per row | `src/core/verify.cpp` (`pre` lambda, attn-half HC read) |
| Taps: prefill writes them for the anchor (and prompt rows for context cells) | `src/prefill/prefill.cpp` |
| Drafter runtime: fusion, context KV (own QsaState pools), block forward, argmax | `src/core/dflash.cpp` |
| Decode loop wiring, prefill wiring, VRAM reservation, metrics | `src/program/generate.cpp` |

Memory (16 GB card, measured at startup): drafter weights ≈ 950 MiB BF16, context/block
KV 20 KiB per position per... (f32 K/V pools, `--dflash-window` cells, default 32768 →
≈ 655 MiB), fusion/logits scratch < 32 MiB. `DFlashDrafter::load` runs before the expert
cache is sized and reports the same way `MtpDrafter` does, so the cache auto-sizing
reserves the drafter's footprint.

Reference fixtures: the PixelML/DeepSpec runtime exports (taps, fused features, logits per
block position) under `bench/dflash-fixture/`; the alignment unit test
(`src/core/dflash_align_test.cpp`) pins the query→target-position mapping with synthetic
weights; the draft-forward parity harness compares against the exported reference tensors
(max/mean abs error, cosine per stage, exact token IDs).

Known limitations of the first implementation are listed at the end of this file after the
measurements.
