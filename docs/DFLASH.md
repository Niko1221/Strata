# Experimental standalone DFlash

The standalone DFlash infrastructure supplies feature capture, private draft K/V,
block execution and target verification. It is the prerequisite for the official
[DFlash2 integration](DFLASH2.md). Setup now offers MTP, DFlash2 or Off; the older
plain-DFlash checkpoint and private-head experiments use the engine CLI directly.
MTP remains the default drafter. `--dflash` and `--mtp` are mutually exclusive.

## Artifact and execution

`tools/dflash_gguf.py` converts a compatible BF16 safetensors checkpoint and configuration
into GGUF. Plain DFlash uses the Flash-Next geometry: hidden 2560, five draft layers,
five target taps, 24 Q / 2 KV heads of width 256 and MLP 7680. The loader checks tensor
shapes, payload bounds, vocabulary, mask token, normalization, attention and scaling
metadata before uploading anything. Malformed or unsupported artifacts are refused.

The target embedding and output head are shared by default. Each draft layer owns a
separate whole-resident FP16 K/V pool; it does not inherit the target's quantized or
streamed K/V allocation. Feature taps are taken from the checkpoint's selected target
layers. Plain DFlash consumes the learned HC-contracted read; the official DFlash2
checkpoint selects the raw HC stream mean instead.

```sh
build/strata --pack /path/to/target-pack --native /path/to/target-shard.gguf \
  --dflash /path/to/plain-dflash.gguf --dflash-window 0 \
  --spec 8 --dflash-block 5 --spec-min-p 0.5 \
  --tokens-file /path/to/prompt-ids.txt --max-new 128
```

The reference plain-DFlash artifact emits its row-zero prediction. DFlash2 reserves
row zero for its anchor and excludes it from emitted proposals. The effective block
must fit both the trained artifact and the verifier's window. Greedy proposals are
checked by the target, and only the accepted prefix is committed. Sampled requests
use target-only decoding; a later greedy request can reuse committed draft context.

## Context, cache and memory

`--dflash-window N` is a draft capacity, not a rolling attention window. The CLI
default is 32768 cells; 0 uses the target session's full configured context.
Prompt plus requested output must fit that capacity. Proposal guards also account for
the anchor and the remaining target context. Positions retain their true RoPE indices.

Draft weights, private pools, scratch and head binding are allocated before the
automatic expert cache is sized. CUDA's profiled VMM path can grow draft K/V and
attention scratch from 8192 cells by taking memory from expert slots. Other paths
allocate full draft capacity at startup. The configured VRAM reserve also has to
cover later verifier buffers and graph capture; it is not all free after startup.

In-memory prompt checkpoints and parked conversations include the committed draft
K/V prefixes. Rewind overwrites rejected or later cells before use. Parked draft
snapshots are charged against the conversation-cache RAM budget; whole compatible
images are restored rather than borrowing a donor conversation's prefix.
Disk SAVE/RESTORE is refused because the disk schema does not contain the draft pools.

The [cache regression report](../bench/results/2026-10-19-dflash2-pr/cache/README.md)
records exact output controls, actual expert placement and the earlier invalid
comparisons that exceeded the requested cache budget. GPU and CPU expert arithmetic
can differ, so a fixed budget must be verified through native INFO before comparing IDs.

## Manual quantization experiments

`tools/dflash_quantize.py` exports Q8_0, Q5_0 or Q4_0 backbone matrices from an original
BF16 source. Norms retain BF16. DFlash2 convolution and selector tensors also remain
BF16. Quantization tools reject in-place output and validate complete tensor payloads;
prepared artifacts are cached separately from their source.

Plain DFlash additionally accepts `--dflash-head HEAD.gguf`, a private BF16 or supported
quantized `output.weight`. The verifier retains its original head. That extra allocation
reduces room for target experts and is not used by DFlash2. A shared quantized plain-DFlash
head may use `--dflash-vocab FILE` with unique int32 token IDs. BF16 heads use the full
vocabulary. The engine-only experiments are separate from setup's DFlash2 weight menu.

## Checks and diagnostics

Loader and policy checks use `dflash_load_test` and `dflash_policy_test`. CUDA checks
cover taps, RoPE, owned pools, BF16 projections and indexed heads. The independent
`tools/dflash_stage_parity.py` forward checks every plain-DFlash layer against the
actual artifact. `tools/dflash_cache.py` compares live, parked, checkpoint, rewind and
sampled-to-greedy cases with fresh drafter and target-only controls.

`STRATA_DF_PARITY=DIR` writes intermediate tensors for a selected diagnostic cycle;
`STRATA_DF_EVENTS=1` records drafter section timings. These paths are disabled by default.
They are correctness/measurement tools, not evidence of performance by themselves.

This integration is single-GPU and serial. CUDA execution has local evidence; HIP and
SYCL compilation and hardware execution of shared changes are not established here.
See [DFlash2 measurements and limits](DFLASH2.md) before making performance claims.
