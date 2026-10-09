# DFlash: standalone block drafter for Qwen3.8-Flash-Next

Experimental support for the DeepSpec DFlash drafter
[`PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash`](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash)
as a **standalone draft model** next to Strata's own MTP layer:

```text
Qwen3.8-Flash-Next target (any supported quant)
        +  standalone DFlash artifact (GGUF, BF16 or quantized matrices, 498M params, no embed / no LM head)
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

## One-click setup and server

Run `./setup.sh --setup` (Windows: `START-HERE.bat --setup`) and select MTP, DFlash or Off
in the numbered menu. Off skips both draft models and leaves suffix/prompt lookup enabled.
The same choice is available as `./setup.sh --setup --yes --drafter none`. DFlash then offers original BF16, Q8_0, Q5_0 and
Q4_0. The choice applies to the drafter's matrices. Norm vectors keep their
original BF16 bits; the target's weights, embedding and output head stay as selected.

For a non-interactive install, keeping the same target model:

```sh
./setup.sh --setup --yes --model IQ3_XXS --drafter dflash --dflash-quant q8
```

Setup downloads the pinned PixelML checkpoint (~1 GB), checks its SHA-256,
exports GGUF and optionally quantizes it with llama.cpp's GGML quantizers.
It reuses prepared files on later runs. `--dflash /path/to/drafter.gguf` also
accepts a local artifact; an additional quantization requires an original BF16
source. Selecting DFlash skips the MTP download, preparation and load. Selecting
MTP again writes an MTP-only configuration.

The selected drafter is passed to the server. Setup defaults DFlash requests to
`temperature: 0`; clients can still request sampling, which uses target-only
decoding and logs that DFlash was skipped. Subsequent starts use the saved choice.
DFlash currently supports one GPU and serial requests. The server rereads the
full prompt for each request because target-only snapshots do not contain the
DFlash pools. Elastic target KV growth is disabled with DFlash. Setup uses
`--dflash-window 0`, preserving the full configured context. On CUDA, with one
GPU and a profiled RAM-backed expert cache, the drafter's FP16 K/V and attention
scratch start at 8192 cells and grow as requests use more positions. The expert
cache supplies physical chunks and gets them back after a shorter request. This
is independent of the target's KV streaming. Other configurations allocate the
full draft capacity up front. `STRATA_DFLASH_KV_GROW=0` retains that allocation
for comparisons. A positive `--dflash-window N` is a capacity limit, not a
rolling window: requests beyond it are refused.

The default pass uses the artifact's trained block (seven rows for this checkpoint)
and verifies only the consecutive predictions whose draft probability is at least
`--spec-min-p 0.5`. Cutting the verified prefix does not change the drafter's
non-causal input width. `--dflash-block K` still selects an actual fixed forward
width for reproducible comparisons; its probability gate is off unless explicitly
requested. With `--spec-min-p 0`, the server retains the measured block-length
policy, including periodic probes and pauses.

Setup's existing calibration (`./setup.sh --calibrate`) also measures actual
DFlash forward lengths, within the loaded drafter's and `--spec` limits. It
sweeps the block before and after the PCIe share and probability floor, then
compares the candidate and defaults three times each in alternating order.
A gain must exceed 3% to save `--dflash-block` in the run configuration. CPU
worker and expert-tier measurements then use that chosen block. Saved results
are specific to the PC, target/context, drafter artifact, vocabulary and draft
capacity; DFlash and off do not reuse MTP's calibration.

Setup projects only the chosen draft vocabulary's rows from the shared target
head (`--dflash-vocab FILE`); no second head matrix is allocated. Its default
subset is the same CJK-inclusive subset used by MTP. `--draft-vocab en`, `fr`,
`cyrillic` or `cjk` selects the corresponding indices. Omitting `--dflash-vocab`
uses the full head. This affects draft proposals and their probabilities; the
target always verifies against its own full head.

Drafter forwards reuse GPU graphs while their attention chunk count and scratch
layout are unchanged. Growing or trimming the scratch, changing probability
output, or rebinding the head invalidates them. `STRATA_DFLASH_GRAPH=0` keeps
eager execution for comparisons. Stage dumps and GPU-event profiling also use
eager execution. Prompt-lookup drafts follow `--suffix-draft`; lookup-chain
composition remains unsupported.

Sampled requests use target-only decoding and do not compute or grow the draft
context during prefill. A greedy request exceeding an explicit draft capacity
is rejected before changing state; sampled requests use the target's capacity.
The server and CLI capture the same layer boundaries, and the prompt feature
stride follows the current buffer layout after each relayout.

This branch's CUDA and Linux HIP engines are built from source for DFlash;
released engines may lack its server and quantization support. Windows HIP needs
a compatible build via `--prebuilt`. Setup checks support before writing the
configuration. Quantized drafter weights are experimental; the target still
verifies every proposal. Validation on IQ3_XXS is recorded in
[the setup test results](../bench/results/2026-10-08-dflash-setup/REPORT.md).

## Strata integration

| Piece | Where |
|---|---|
| CLI: `--dflash FILE.gguf`, `--dflash-block K`, `--dflash-window N` | `src/program/generate.cpp` |
| Artifact: GGUF v3 reader (existing `strata::GgufFile`), metadata + tensor validation, BF16/quantized → device | `include/strata/core/dflash.hpp`, `src/core/dflash.cpp` |
| Taps: verify window writes the 5 contracted residuals per row | `src/core/verify.cpp` (`pre` lambda, attn-half HC read) |
| Taps: prefill writes them for the anchor (and prompt rows for context cells) | `src/prefill/prefill.cpp` |
| Drafter runtime: fusion, context KV (own QsaState pools), block forward, argmax | `src/core/dflash_runtime.cpp` |
| Decode loop wiring, prefill wiring, VRAM reservation, metrics | `src/program/generate.cpp` |

The five draft FP16 K/V pools use 10 KiB per position (5 layers × 2 K/V
arrays × 2 heads × 256 values × 2 bytes). DFlash does not allocate the target's
sparse-indexer history. The attention scratch grows with the mapped capacity.
Weights, mapped pools and scratch are loaded before the expert cache is sized.
The target's embedding and output head are shared. The [performance check](../bench/results/2026-10-09-dflash-performance/REPORT.md)
records the measured startup footprint and generation speed for the setup path.

Feature-capture gate (commit 2): `STRATA_DFLASH_TAPS=<file>` makes a target-only run append one
record per prompt chunk and per verify window with the five boundaries' contracted residuals
(prompt path in BF16 - the fusion's input precision; window path in f32), and
`tools/dflash_taps.py` compares the two kernel paths' capture of the same position.  On the
IQ3_XXS target, 400-token prompts: per-tap cos 0.9966-0.9999, rel mean 1.2e-2 to 8.5e-2 - the
prompt path's batched BF16 GEMMs and the window's per-token kernels (multi-row AVX2 CPU experts)
agree to engine-path tolerance, while any wiring bug (wrong layer, the other HC half, the raw
10240-wide stream, a token offset, reordered taps) measures cos <= 0.7.  The reference-forward
comparison against the PixelML exporter's own tensors needs the vLLM stack and is not runnable
here; the exporter's tap definition was transcribed from its published adapter patch instead.

### Stage parity (STRATA_DF_PARITY, per row, all five layers)

`STRATA_DF_PARITY=<dir> [STRATA_DF_PARITY_CYCLE=c] ./build/strata ... --dflash DFLASH.gguf` dumps
one draft cycle's stages — the fused context, and per draft layer `xn`, `qraw`, `qnormed`, `q`,
`k`, `v`, the pool pages the attention reads (`kpool`/`vpool`, with `pt` and the per-row
attention `steps`), `attn`, `h_attn`, `h_mlp` — plus `emb` and `meta.bin`
`[anchor pos, K, anchor token, mask token, page_size]`.  `tools/dflash_stage_parity.py` recomputes
every stage independently from the same GGUF and judges EACH ROW (cos / max-abs / rel-L2 per row;
an aggregate cosine hides one bad anchor row among six exact mask rows), recompute the attention
cell by cell from the dumped pools (failing hard on an out-of-range cell instead of emitting NaN),
roll the reference through all five layers, and verify the context cells' K/V values AND rope
positions.  Measured on the IQ3_XXS target, RTX 4070 Ti SUPER: at the first propose after an
8-token and a 119-token prompt, every stage of every row passes (attention max|d| ~2-4e-6 against
the oracle; the block at bf16-GEMV rounding), and the artifact's safetensors match the GGUF
bit-for-bit.

Three bugs this fixture caught, all fixed:

1. `parity_dump_u16_as_f32` widened the fp16 pools with the BF16 bit shift (no exponent
   re-bias), so every dumped pool decoded as plausible-looking nonsense.  A readback compared
   against such a dump made a correct append look like it wrote garbage.
2. The query rows' and the KV rows' rope positions shared one pinned staging buffer with the
   copies that read it still in flight; a late DMA read the KV-layout bytes and roped query
   row 0's first heads at the wrong positions - the row-0-only, run-to-run-flaky layer-1 Q
   divergence.  The two regions are now disjoint, and the fusion's per-chunk position/step
   stagings (whose values do change) sync after the copy.
3. The decode loop filled the verify window's draft slots from `--spec-oracle`'s fixture list
   (token 0 past its end) whenever `--mtp` was absent - a `--dflash` run verified
   `[anchor, 0, 0, ...]` and accepted nothing, whatever the drafter produced.  DFlash drafts
   now ride the MTP slot.

Measured acceptance after the fixes (greedy, IQ3_XXS target, K=6, RTX 4070 Ti SUPER): the
119-token fixture above accepts 10 of 78 drafts (0.128, 1.71 tokens per round) over 24 new
tokens; a 69-token natural-language prompt accepts 68 of 413 (0.165, 2.13 tokens per round) over
128 new tokens, against 0 of 30 and 0 of 161 before.  With the expert cache pinned to the same
slot count (4283), the temperature-0 output of a `--dflash` run is token-for-token identical to
target-only at 24 and 128 tokens (the drafter's ~1 GiB changes the expert-cache auto-sizing, and
CPU-computed experts round differently from resident GPU ones - pin `--expert-cache` when
comparing).

Known limitations of the first implementation are listed at the end of this file after the
measurements.


## Draft-only head quantization experiment

`--dflash-head HEAD.gguf` loads a separate `output.weight` for DFlash. The target
verifier keeps its original head. The file is loaded before automatic expert-cache
sizing, so its extra device memory reduces the space available for experts.
Without this flag DFlash uses the shared target head and allocates no extra head.

Export a head with the offline tool (the repository's gguf-py is used only here):

```bash
python tools/dflash_head_quantize.py TARGET_SHARD.gguf --type Q8_0 -o head-Q8_0.gguf
python tools/dflash_head_quantize.py TARGET_SHARD.gguf --type Q4_0 -o head-Q4_0.gguf
# Add --dflash-head head-Q4_0.gguf to the existing DFlash command.
```

These files re-quantize the source tensor. Exporting Q8 from a Q5_K source does
not restore precision lost in Q5_K. The tool records tensor hashes and weight
reconstruction errors in a JSON file beside the export.

On an RTX 4070 Ti SUPER, IQ3_XXS target, K=6, fixed expert budget 3500,
128-token math fixture and 256 generated tokens, three alternating runs gave
median throughput 40.20 tok/s with the shared Q5_K head, 40.21 with Q8_0 and
40.80 with Q4_0. Q8 adds 644.2 MiB of weights; Q4 adds 341.0 MiB. A separate
CUDA-event capture measured 1.213, 1.137 and 0.828 ms per head projection,
respectively. The event capture is excluded from throughput measurements.

Q4 changed the generated code continuation and reduced its acceptance in a
single code run. Q8 and Q4 both changed the chat continuation. These heads are
experiments, remain opt-in, and have not passed a general greedy-output gate.
See `bench/results/2026-10-08-dflash-opt/REPORT.md` for commands, cache capacity,
conditional acceptance, common-prefix evidence and limitations.

For diagnostic evidence, `STRATA_DF_EVENTS=1` records individual GPU section
and CPU enqueue times. Their sum is not decode wall time. `STRATA_DF_CYCLES=PATH`
writes proposal IDs, verifier IDs, accepted prefix, anchor position and correction
per cycle. `STRATA_DF_TARGET_PROBE=POSITION` prints the verifier's top two logits
at a requested absolute input position, before any benchmark follow override.
These diagnostics are disabled by default. Parity captures also contain full
head logits, quantized activations and argmax IDs; check these with
`tools/dflash_head_parity.py --head HEAD.gguf --dir PARITY_DIRECTORY`.

The [2026-10-09 review](../bench/results/2026-10-09-dflash-review/REPORT.md) records
follow-up correctness fixes, repeated performance measurements on the same target,
and the remaining generation bottleneck.
