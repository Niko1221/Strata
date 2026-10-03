# 2026-09-28: the sm_86 baseline, before any tuning

Machine: RTX A3000 12GB Laptop (GA104, sm_86, 60-80 W), driver 615.71.09 (open module),
i7-12850HX (AVX2 only, no AVX-512), 125 GB RAM, CachyOS, desktop on the iGPU (the dGPU
is free for compute).  Model: Qwen3.8-Flash-Next GSQ-RCO quants (original and Swift 1.5).

## The finding that re-set this program: the engine version dominated everything

Same model, pack, tokenizer, MTP layer and run config; **only the engine binary swapped**
(2026-09-28, `swift-1.5-iq3_xxs`, 64K context, cache auto, spec 4, fresh-prose prompt):

| engine | decode |
|---|---:|
| `main` (`b38c183`, "Engine 0.1.18") | **28.4 tok/s** |
| an earlier pin (`a9047fd`) | 8.4 tok/s |

The ~6 tok/s "quant-independent floor" measurements taken before that date were an
artifact of the older pin - upstream had already improved the engine ~3.4x.  Two
consequences:

* any tuning baseline must state the engine commit (this file does);
* `--prefill auto` (0.1.18) is what lets a 262K context fit in 12 GB on this card.

## Decode speed swings with the text, not the engine

Same engine, config and session (swift IQ3_XXS, 64K, thinking off):

| prompt | decode |
|---|---:|
| fresh prose | 23.3 tok/s |
| echo-heavy (repeat a fixed sentence) | 38.8 tok/s |

The mechanism is speculation: copied text is accepted by prompt lookup / MTP far more
often.  **Chat-derived "Nx faster" claims are meaningless**; use `bench/e2e.sh` (fixed
prose prompt) or the teacher-forced paths.

## The experimental speed projection is not a speedup

`data/experimental-speed-projection/` projects out a refusal direction (layers 4-44).
Upstream measures **0.2-0.4% slower on identical tokens**; apparent tok/s rises only
because the model writes different text (it reports 1/50 vs 50/50 refusals on its test
set), and it shifts predictions on ordinary text (mean KL 0.063, code same-top-1 92.8%).
Keep it out of baselines unless the trade is the experiment.

## Headroom

Upstream's RTX 5070 (12 GB, Windows) result for the same quant class is 64.6 tok/s
(IQ3_XXS, 1K context).  Against this card's 28.4 tok/s that is ~2.3x - the shape of a
2x-bandwidth desktop part at 250 W vs a 60-80 W laptop part, not the 9x the stale pin
suggested.

## Next (the actual sweep)

`engine.nix` in the nix packaging lists ~40 ablation flags (`--no-token-graph`,
`--no-capture`, `--no-spec-split`, `--no-host-worker`, `--no-pool`, `--no-hit-poke`,
`--no-prefill-borrow`, `--no-fast-attn`, `--no-fused-*`, `--graph-only`, `--gpu-stages`,
`--pcie-frac`, `--adapt-every`, `--expert-cache` modes, `--kv q4_0`) that bisect where a
token's time goes without a profiler.  Run each through `bench/e2e.sh` on one config at
a time and record the table here; then bring in `ncu` (needs
`NVreg_RestrictProfilingToAdminUsers=0` or sudo) on whatever the sweep flags as hot.
