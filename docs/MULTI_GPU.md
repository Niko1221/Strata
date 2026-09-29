# Strata on two or three GPUs (layer split)

One model can run across several NVIDIA cards in one PC. The layers are split into contiguous ranges, one per GPU:
the first card runs layers 0 to K-1, the next card runs K onward, and so on; the last card also runs the output head
and the draft (MTP) layer. Each card keeps an expert cache for **its own layers only**, so two cards hold about twice
the experts one card holds - for the Coder model on a 16 GB + 24 GB pair, nearly all of them, which is where the
speed comes from (decode then barely touches the CPU pool).

This is pipeline (layer) parallelism, not tensor parallelism: a token crosses from one card to the next once per
verify window (a few hundred KB through pinned RAM), not twice per layer. No NVLink or peer-to-peer access is
needed; cards on x4 or x1 slots work, and the PCIe share of each card is probed on its own link.

## Using it

At setup time (Windows: `START-HERE.bat`, Linux: `./setup.sh`):

```
--gpus 0,2                 the cards, as nvidia-smi numbers them; the first is the main one
--layer-split auto         (default) or the first layer of each later card, e.g. 18 or 16,32
```

or in an existing config (`strata-*.json`), then restart:

```json
"gpu": [0, 2],
"layer_split": "auto"
```

The engine flags behind it: `--layer-split K1[,K2..]|auto` and `--split-device D1[,D2..]` (the later stages'
devices; default the next visible ones). `--layer-split K --split-device 0` runs both stages on one card sharing
everything - the bit-exact check of the hand-off, not a speed mode.

**auto** tries every placement (all of them for two or three cards; proportional to the free VRAM beyond that) and
keeps the one whose caches would hold the most of the expert profile, hottest pairs weighted most; ties go to the
placement that leaves the fullest card the most room. The startup log prints the choice:

```
strata generate: layer split auto: K=19 - the caches hold 11767 of 12288 profiled pairs (fullest device 100%)
strata serve: layer split: layers 0-18 (CUDA0), 19-47 (CUDA1), one hand-off per window
```

## What each card holds

- **every card**: a copy of the dense weights (~3.4 GB for the Coder), its own session state (the KV cache of the full
  context), its verify window and its prompt-path buffers, and an expert cache for its layers filled from the profile;
- **the last card**: also the output head and the draft layer (~0.8 GB);
- **host RAM**: the expert arena once, shared by all cards (the CPU pool computes whatever no card holds).

Prompts are read in chunks that flow through the cards in turn; while a later card reads chunk c, the first card
already reads chunk c+1. Conversation checkpoints save and restore every card's state; the adaptive expert swaps copy
into the card that owns the layer.

## Limits (for now)

- Not with images (`--vision`), control vectors / the speed projection, KV streaming (`--kv-resident`), the older
  helper-GPU experiment (`--expert-cache-remote`, docs/SECOND_GPU.md) or `--mmap-experts`.
- The prompt path has its own buffers on every card (1.5 GB each at the default 2048-token chunk; `--prefill 1024`
  halves that) instead of borrowing cache slots as one card does.
- Mid-prompt checkpoints (`--prompt-cache-every`) are off; the ones at turn boundaries - what a chat reuses - stay.
- On Windows only 8 GiB of the expert arena is pinned (more, mapped into two GPU contexts, leaves WDDM refusing
  allocations); the PCIe share covers those layers.
- Every card needs compute capability 8.0 (RTX 30 or newer). A Turing card (RTX 20, sm_75) builds only with the
  experimental `-DSTRATA_EXPERIMENTAL_SM75=ON`.

## Measured

The Coder on an RTX 5080 + RTX 3090 (Ryzen 9 9950X3D), 32K context; details in
`bench/results/2026-09-29-layer-split/`:

| | Prompt 16K / 28K tok/s | Decode story / code tok/s |
|---|---|---|
| 5080 alone | 1,726-2,017 / 1,970 | 83-87 / 88-105 |
| 5080 + 3090, best split (K=26) | 2,039 / 2,357 | 84 / 110 |
| 5080 + 3090, auto (K=22) | 2,037 / 2,073 | 80 / 109 |

- **Prompts gain the most** (+18-20%): each card reads its own layers of the chunk while the other reads the next.
- **Decode is on par with the faster card alone**, and ahead on code. Once both caches hold nearly every routed
  expert, the per-layer GPU time decides.
- **Correctness:** one GPU is byte-identical to 0.1.20, and the hand-off itself is bit-exact.

**Which cards and in what order:**
- Put the fastest card first; auto gives it as many layers as its cache allows.
- Leave out a much slower card when two already hold the model. An RTX 2080 Ti as a third card made the 5080 +
  3090 pair slower (68 / 90 tok/s decode): every extra card costs its own round per window.
- More cards pay off when the model's routed experts do not fit the faster ones.
