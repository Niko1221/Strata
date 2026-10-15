# The FP16 sliding window + attention sinks (`kv_window.hpp`)

An FP16 overlay over a quantized KV cache: the `sinks` cells at the head of the sequence and the `window` most
recent cells are kept in FP16 and read from there, while the bulk stays in whatever `--kv` chose. The error of the
4-bit (or INT8) rounding is removed exactly where the model attends most closely, for a few hundred KB.

```
                 sinks            bulk (quantized)             window
  cells:   0 ......... 7 | 8 ......................... n-129 | n-128 ... n-1
  read from:  FP16 overlay |    q4_0 / INT8 pools              | FP16 overlay
```

## Why this shape

* SKVQ (arXiv 2405.06219) makes a sliding FP16 window the cheapest large win at <= 2-bit KV, and the sink-token
  result (arXiv 2508.04257) plus RotateKV's sink protection show the first few cells carry disproportionate
  attention mass - a quantized sink is a permanent error, an exact one costs nothing.
* Keys deserve more bits than values (arXiv 2502.15075); the overlay gives both sides exactness at the recent end,
  which is where the next token's attention actually lands.

At the serving geometry (12 QSA layers, 2 KV heads, head_dim 256), counting the 16 slack slots
(`kKvWindowSlack`): `(8 + 128 + 16) slots * 2 heads * 256 dims * 2 B * 2 (K and V)` = 311,296 B = 304 KiB per layer,
3.56 MiB for the whole model - against 6,912 B/cell of q4_0 bulk over the 12 layers (576 B/cell/layer), which is
1.69 GiB at 262,144 cells.

## Design

* **The bulk pools stay COMPLETE.** Every append still writes `k_q4/v_q4` (or `k_q/k_q` INT8) for every cell, so
  snapshots, KV-streaming residency, the ring modes, prompt checkpoints and the prompt/gather readers all keep
  reading exactly the pools they always read. The overlay is a *read-side* overlay plus one small extra write per
  append - it changes what the decode attention reads, nothing else.
* **Row numbering carries the decision.** A reader resolves a cell to a *row*: below `QsaAttnPools::bulk_rows` it is
  the pool row the page table gives, at or above it an overlay row. `kv_window_row()` encodes that, so the single
  `row` a reader already computes says whether to read FP16 - one compare, no second lookup.
* **Slots.** Sinks own slots `[0, sinks)` for the cells at positions `[0, sinks)`; the window ring owns
  `[sinks, sinks + window + slack)` for cell `p` at slot `sinks + p % (window + slack)`. `slack` (16) is what lets
  the window survive speculative decoding: a draft batch appends up to `slack` cells ahead of what it reads, and
  with the slack those look-ahead writes cannot land on a slot a live window cell owns.
* **Capturable.** Every kernel reads the position and the cache length out of the step buffer on the DEVICE
  (`kStepPos`, `kStepNKv`) and launches with a fixed grid, so the append sites stay inside the captured region.
* **Restore.** The overlay is not part of a conversation snapshot; `conversation_kv_restore` rebuilds it from the
  restored bulk (`kv_window_restore`, dequantizing what the snapshot brought back). A restored window is therefore
  at bulk precision rather than exact for its first `window` tokens - correct, where leaving it would have the new
  conversation reading the previous one's K/V. `qsa_state_zero` clears it for the same reason.

## Enabling

```
--kv-window 128 --kv-sink 8        # CLI (wins)
STRATA_KV_WINDOW=128 STRATA_KV_SINK=8   # environment, when no CLI flag asked
```

Off by default: with neither, every reader takes the path it took before the overlay existed, bit for bit. Only the
quantized formats gain from it (`--kv q4_0`, `--kv int8`); a fully resident FP16 state ignores it and the MTP
drafter's ring state keeps its plain format, the same reason the hybrid setting does - the block movers never see an
arena the rest of the engine does not know about.

## Scope (v1)

| path | reads the overlay |
| --- | --- |
| decode attention, fused (`qsa_decode_attn_step`/`_batch`, the default) | yes |
| speculative verification (its own appends and reads) | yes |
| prompt attention (`qsa_prompt_attn.cu`) | no - the bulk (q4_0) |
| `--no-fast-attn` gather + attention (`kv_gather_q4_step`) | no - the bulk (q4_0) |
| MTP drafter | n/a - its ring state has no overlay |

The two "no" rows are correct but unimproved: they read the bulk, which is complete. Extending them is the same
`row` comparison, in `qsa_prompt_attn.cu`'s gather (where a mode-4 tile would carry the FP16 values as int8 codes
with a per-32 scale, so the tile format does not have to change) and in `kv_gather_q4_step`.

## Measured

`bench/results/2026-10-11-kv-window/` has the numbers and the reproduction commands. In short, from a build with
CUDA 13.0 (the release's toolchain - a local 13.2 build is numerically broken for this model, see that README):

* **Representation**, deterministic: K error against an FP16 cache, relative RMS, is 8.94% (sinks) / 8.58% (window)
  / 8.57% (bulk) for `q4_0`, and **0.000% / 0.000% / 8.57%** with the overlay - and the first two are bitwise equal
  to an FP16 cache, not merely close.
* **Cost**: decode attention over 8192 cells 0.080 ms -> 0.080 ms; one ~1 KB row written per append; 304 KB per
  layer of footprint against 288 KB per 1024 cells of `q4_0` bulk.
* **End to end** (exact-copy of 12 five-digit codes, needle recall, short answers, loop rate) at 4K, 16K, 32K, 64K
  and 131K: `q4_0` with the overlay, `q4_0` without it and FP16 are indistinguishable - 12/12 codes and no loops
  everywhere, prefill and decode within run-to-run noise. So at 4 bits the overlay is a **no-regression precision
  improvement** rather than a measurable win; the win it is built for is sub-4-bit bulk, where the same error grows
  several-fold.

## What is verified

`src/kernels/kv_window_parity.cpp` (`ctest -R kv_window_parity`, GPU, synthetic, no model) checks, through the real
fused decode attention:

1. a selection of sinks and window cells only: the overlay's output is **bitwise equal** to a pure FP16 pool's;
2. that same selection with the overlay off is **not** equal - the rounding is what the overlay removes;
3. cells outside the window and the sinks: overlay on and off agree **bitwise** (the bulk read is untouched);
4. `kv_window_restore` reproduces the bulk's own dequantization **bitwise**;
5. membership and slots: exactly the sinks and the last `window` cells are in range, and pairwise distinct.

Cost, when on: one extra kernel launch per append (a few hundred KB of writes per token across the model) plus a
compare per cell in the decode attention's row resolution.
