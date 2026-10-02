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

**Nothing to type.** `START-HERE.bat` (Linux: `./setup.sh`) lists your NVIDIA cards and says for each one whether
Strata can use it:

```
  Your NVIDIA GPUs:
    GPU 0: NVIDIA GeForce RTX 5080, 16 GB VRAM - can be used
    GPU 1: NVIDIA GeForce GTX 1080 Ti, 11 GB VRAM - not supported - older than the RTX 20 series (compute capability 6.1; Strata needs 7.5 or newer)
    GPU 2: NVIDIA GeForce RTX 3090, 24 GB VRAM - can be used
  ...
  1) GPU 0 (NVIDIA GeForce RTX 5080, 16 GB) + GPU 2 (NVIDIA GeForce RTX 3090, 24 GB) together   (recommended)
  2) GPU 2 (NVIDIA GeForce RTX 3090, 24 GB) only
  3) GPU 0 (NVIDIA GeForce RTX 5080, 16 GB) only
Which GPUs? [1]:
```

When two or more cards can share the model, the two best together are recommended (the newest generation first:
it becomes the main card). A model installed on one card asks once, at its next start, whether to use both from
now on; the answer is kept.

**Choosing yourself** (at setup or at any start):

```
--gpus 0,2                 these cards together, as nvidia-smi numbers them; the first is the main one. Remembered.
--gpus all                 every card that can share the model
--gpu 0                    one card (at a start: for that start only)
--layer-split auto         (default) or the first layer of each later card, e.g. 18 or 16,32
```

**Not supported** (setup says so and names the cards that can be used instead):
- a card older than the RTX 20 series (compute capability below 7.5: GTX 10 and older);
- a card with less than 8 GB of VRAM, together with others (each card holds a copy of the dense weights and its
  own prompt buffers);
- Intel GPUs, and a mix of NVIDIA and AMD cards. (AMD cards share a model among themselves: `./setup.sh --backend
  hip --gpus 1,0`, see [AMD_HIP.md](AMD_HIP.md).)

Or edit an existing config (`strata-*.json`), then restart:

```json
"gpu": [0, 2],
"layer_split": "auto"
```

**Skip the split when the first card holds everything** (opt-in, 0.1.31): `"split_skip_if_fits": true` in the config
(engine flag `--split-skip-if-fits`, with `--layer-split auto`) runs on the first card alone when it holds every
profiled expert plus the context's KV, the draft layer and the reserve, and says so in the log; otherwise the split
stays. On an R9700 32 GB + RX 9070 XT the R9700 holds all of the Coder's experts: with the flag 4K prompts read at
1,776 tok/s instead of 1,244 (split) and decode runs at ~60 tok/s instead of ~51 (16K prompts ~5% slower than split).

The engine flags behind it: `--layer-split K1[,K2..]|auto` and `--split-device D1[,D2..]` (the later stages'
devices; default the next visible ones). `--layer-split K --split-device 0` runs both stages on one card sharing
everything - the bit-exact check of the hand-off, not a speed mode.

**auto** tries every placement (all of them for two or three cards; beyond that the layers are shared in proportion
to card speed) and keeps the one whose predicted decode-window time is lowest: each card's per-layer time (less on
a card with more SMs and a higher clock) plus the routed mass its caches would miss, hottest profile pairs
weighted most. It is a decode-cost model, not a prompt-bandwidth one. The startup log prints the choice:

```
strata generate: layer split auto: K=22 - predicted 19.7 ms per decode window; the caches hold 11298 of 12288 profiled pairs (~99.7% of the routed mass)
strata serve: layer split: layers 0-21 (CUDA0), 22-47 (CUDA1), one hand-off per window
```

**PCIe share, per card:** without `--pcie-frac`, every card probes its own host-to-device link. A slower link
gets a smaller share of the missed experts. An explicit startup `--pcie-frac` value applies to every card.
A request's `pcie_frac` value overrides every stage only when it differs from the first card's startup share.
Otherwise, later cards keep their own startup shares.

## What each card holds

- **every card**: the canonical weights and native dense projections of its OWN layers (an explicit split
  loads only those; auto placements keep the full tables), its own session state
  (the KV cache of the full context), its verify window and its prompt-path buffers, and an expert cache for
  its layers filled from the profile;
- **the last card**: also the output head and the draft layer (~0.8 GB);
- **host RAM**: the expert arena once, shared by all cards (the CPU pool computes whatever no card holds).

Prompts are read in chunks that flow through the cards in turn; while a later card reads chunk c, the first card
already reads chunk c+1. Conversation checkpoints save and restore every card's state; the adaptive expert swaps copy
into the card that owns the layer.

## Limits (for now)

- **Works across cards** (bench/results/2026-09-29-layer-split-limits):
  - images (`--vision`): each card keeps its own image-position table;
  - control vectors and the experimental speed projection: each card holds the vector's tables, switched on and
    off per request on all of them;
  - KV streaming (`--kv-resident`): each card streams the KV of its own session;
  - the MTP draft layer's own K/V: when a window is set and is smaller than the context (`--mtp-window`;
    default 32,768 cells; a window of 0, or at least the context, keeps the draft layer's K/V fully in VRAM),
    the draft layer's K/V is a ring of that window over its own pinned host copy, independent of
    `--kv-resident`. The host copy holds the WHOLE context in RAM (it is the basis for refilling the ring on a
    resume; no part of it is reduced). If the host copy cannot be pinned, the draft layer's K/V stays fully in
    VRAM. With the ring the draft attends only the last window's cells - a longer window holds more of the
    context at the cost of more VRAM only; the pinned host copy always holds the whole context, whatever the
    window;
  - mid-prompt checkpoints (`--prompt-cache-every`): each card saves its part of a checkpoint when it has read that
    chunk;
  - the older helper-GPU caches (`--expert-cache-remote`, docs/SECOND_GPU.md): they take the visible GPUs no stage
    runs on, and hold only experts no stage's cache holds. On the test rig, a 2080 Ti helper made decoding slower,
    as it did without a split: its per-layer round trip costs more than the CPU pool needs for those experts.
- `--mmap-experts` needs a canonical pack (`experts.bin`), with or without a split; a native (IQ) pack says so at
  start.

- With an expert profile (the default), every stage's prompt path borrows the tail of its own expert cache for its
  chunk buffers and refills it after the prompt; outside the prompt the whole cache is expert cache again, so the
  buffers cost a stage no permanent VRAM. Without a profile, or with `--no-prefill-borrow`, each stage keeps its
  own chunk-sized buffers for the whole session instead; then an explicit `--expert-cache` is capped to leave room
  for them.
- Every stage leaves room for its verify windows (~96 MiB) before its cache is sized; the draft layer and the head
  (~1 GiB) sit on the last stage only.
- Under WDDM (Windows, and WSL2) only 8 GiB of the expert arena is pinned (more, mapped into two GPU contexts,
  leaves WDDM refusing allocations); the rest streams through the pinned staging ring. A Linux driver has no such
  limit, so there the whole arena is pinned. `STRATA_ARENA_PIN_GIB=N` pins at most N GiB, `0` the whole arena, on
  any OS. Only registered layers can use direct PCIe copies; unregistered ones use the existing CPU and
  host-staging paths.
- Every card needs compute capability 7.0 (Volta or newer). The build includes code for each selected card's
  architecture; Ampere-only prompt kernels automatically use their portable fallback on Volta and Turing.

**Decode runs the stages one after another.**  A verify window finishes on one card before the next card starts
its share: the hand-off is one per window (a few hundred KB through pinned RAM).  Prompts gain from the split -
each card reads its own layers of a chunk while the previous card reads the next chunk - but decode is the sum
of the stages, not the longer one.  With two equal cards each card is busy about half the time during decode
(measured on a pair of V100 16 GB: one card at 100% util while the other idles, alternating per window).

Inside a window, each layer's hyper-connection read and its K/V append run as a small fixed set of launches,
whatever the window's size:

- **Hyper-connection read (the hc-read halves).** The read computes the norm, the low-rank down projection and
  the up projection once per layer half.  On a Volta card (sm_70) a group of up to 8 tokens (a group is
  the window, or its half under `--spec-split`) is read in two kernels: the down launch runs the original
  scalar down body once per token. Each block handles one token and one group of 8 projection rows.
  The block recomputes its token's norm with the unchanged 8-warp mapping. There is no separate norm launch and
  the normalized input never round-trips through global memory; the up kernel follows.  Every token's
  outputs are bit-identical to the scalar kernel's.  Every other architecture keeps the original
  three-launch read.
- **K/V append.** Each verify group uses one append launch for Q8, Q4 or FP16 KV. Hybrid KV uses one
  launch for K8 and one for V4. Each token reads its own position record and writes a separate cell.
  The append also writes the host copy when one exists. The quantization arithmetic is the same as
  the single-token append. Page boundaries and rejected-tail overwrite have parity checks.

The host thread that drives the windows is pinned to its own core and spins while it waits for the GPU
(`SessionLoopScratch`, `Verifier::run`); on that rig the wait is 11.3 ms of every 33.6 ms window, against 0.7 ms
of per-layer host work and ~1.3 ms of card-side flag waits.  The pinned core is by design and is not the decode
limit: the GPU is the critical path, and moving the loop's work onto more cores changed no throughput (see the
measured section).  The open limit is pipelining a window's positions across the hand-off - the next card starts
position t while the previous card still runs position t+1 - which would make decode the longer stage instead of
the sum.
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
- **Two equal cards** (a V100 16 GB pair, Qwen3.8-Flash-Next Q2_0, 4K prompts, `--layer-split 20`): decode at
  55-58 tok/s against 19-22 tok/s on one card alone - the split nearly triples decode here because a second
  cache cuts the missed expert mass the CPU pool computes (hit rate 0.93-0.98 vs 0.73).  The stages still run
  one after the other, so two equal cards do not double decode: the per-window time is the sum of the two
  stages and each card is busy about half the time.  The windows drove the host thread's pinned core to a full
  spin while the GPU ran, and the speed did not change when that work could have spread across cores - the
  card-side flag waits on the host (waitA + waitB + waitCPU) total ~1.3 ms of every 33.6 ms window.
- **The hc read on the pair** (captured-event fixture: seed 99, bf16 weights, native 2560/4/320, window
  sizes 1 to 8): per-call GPU microseconds, original vs fused two-kernel read on each card:

  | T | old GPU0 | new GPU0 | old GPU1 | new GPU1 |
  |---|---------:|---------:|---------:|---------:|
  | 1 | 61.9     | 42.2     | 60.5     | 40.6     |
  | 2 | 71.3     | 50.0     | 69.8     | 48.5     |
  | 3 | 79.3     | 56.3     | 77.6     | 54.2     |
  | 4 | 87.8     | 78.6     | 86.4     | 74.0     |
  | 5 | 97.7     | 85.6     | 95.8     | 83.9     |
  | 6 | 105.7    | 90.2     | 104.0    | 88.1     |
  | 7 | 115.2    | 94.7     | 113.6    | 93.2     |
  | 8 | 124.9    | 116.2    | 123.1    | 113.8    |

  The old and new full output dumps of the eight windows are byte-identical (3,360,768 bytes in all);
  scalar parity, capture replay and no-inject checks all pass, and the six selected kernel ctests
  (kv q8/q4/stream/hybrid parity, gr parity, qsa parity) pass on the final build.  A captured-event
  fixture measures kernel latency, not throughput - it is not proof of a runtime gain.

### V100 runtime comparison

The final comparison uses two V100 PCIe 16 GB cards. GPU0 has a Gen3 x2 link.
GPU1 has a Gen3 x16 link. The model is Qwen3.8-Flash-Next Q2_0. The layer split
is 20/28. Main KV is int8 with a 262,144-token capacity. Vision stays enabled.
Prefill chunk size and expert placement use their automatic settings.

Each prompt size has three fresh requests and 256 output tokens per request.
No request reuses prompt tokens. Seed 1001 controls the prompt generator, not
the model sampler. The server uses greedy sampling. The table shows medians
in tokens per second against the installed runtime before these changes.

| Prompt target | Prefill before | Prefill after | Decode before | Decode after | Decode change |
|---|---:|---:|---:|---:|---:|
| 4K | 441.8 | 442.6 | 61.0 | 64.4 | +5.6% |
| 8K | 619.7 | 621.0 | 58.7 | 68.0 | +15.8% |
| 16K | 818.9 | 820.3 | 56.6 | 66.4 | +17.3% |
| 32K | 974.4 | 977.1 | 58.6 | 66.8 | +14.0% |

The goal of a 20% gain in both paths at all four sizes was not met. Prefill
throughput is approximately unchanged. A 14/34 split with a 12,288-token
prefill chunk raised short-prompt prefill to 650.4/778.4 tokens per second,
but reduced 32K prefill to 839.9. The installed configuration keeps the
original split and automatic chunk size. No rejected tuning flag is enabled.

The final build passed six kernel parity tests. GR memory and synchronization
checks reported zero errors. A live request passed the draft ring boundary
with 53,508 prompt tokens and 256 output tokens. Its direct prefix extension
restored a 49,152-token checkpoint and produced 128 output tokens. Main KV
stayed in VRAM for both requests.

Kernel byte parity does not establish identical generated text for all
placements. A fixed no-PCIe-miss control matched all three completion hashes.
Other controls changed expert-cache capacity and could change CPU/GPU
arithmetic. An original-versus-original restart also changed one completion.

Raw matrices, comparison data and verification details are in
`bench/results/2026-10-01-v100-autonomous/`. To repeat the four-size matrix,
use `bench/run_v100_bench.py --model qwen --targets 4096,8192,16384,32768
--seed 1001 --repeats 3 --max-tokens 256` with the model GGUF and output
directory options for the local installation.

**Which cards and in what order:**
- Put the fastest card first; auto gives it as many layers as its cache allows.
- Leave out a much slower card when two already hold the model. An RTX 2080 Ti as a third card made the 5080 +
  3090 pair slower (68 / 90 tok/s decode): every extra card costs its own round per window.
- More cards pay off when the model's routed experts do not fit the faster ones.
