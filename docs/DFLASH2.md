# Experimental DFlash2 for Flash-Next

DFlash2 is an opt-in CUDA drafter for Qwen3.8-Flash-Next. It depends on the standalone
DFlash infrastructure from [PR #1426](https://github.com/Niko1221/Strata/pull/1426).
The existing DFlash2 discussion is [PR #1961](https://github.com/Niko1221/Strata/pull/1961).
MTP remains the setup default. DFlash2 and MTP cannot be loaded together.

The supported checkpoint is
[incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/tree/a22a1744e85c17364dcc54a6a420af7135d4c7f6),
revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6`.
Its `model.safetensors` SHA-256 is
`d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f`.
It contains 81 BF16 tensors and no embedding or output-head tensor. The target supplies
both the embedding and the full output head. A Qwen3.8-27B drafter is not a substitute.

## Setup

Run `./setup.sh --setup` and select DFlash2, or use:

```sh
./setup.sh --setup --yes --drafter dflash2 --dflash2-quant q8
```

Setup downloads the pinned checkpoint, checks its hash, converts it to GGUF and builds
the local CUDA engine with `STRATA_ENABLE_DFLASH2=ON`. It offers BF16, Q8_0 and Q4_0.
Q8_0 is the balanced recommendation from the local comparison; Q4_0 saves more memory.
Only the 36 backbone matrices are quantized. The 45 norm, convolution and selector
tensors remain BF16. A local compatible GGUF can be supplied with `--dflash2 FILE`.
The old plain-DFlash installer and separate-head choices are no longer offered.

DFlash2 currently requires one NVIDIA GPU and serial requests. HIP and SYCL cannot
build the DFlash2 extension. Compilation and hardware execution of the shared DFlash
prerequisite on those backends still require validation. Windows was not tested.

`./setup.sh --calibrate` measures runtime settings for the selected artifact: draft
block length, probability floor and PCIe share. It does not select a weight format or
calibrate a separate head. Change `--dflash2-quant` through setup to compare formats.

## Build and manual use

```sh
cmake -S . -B build-dflash2 -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_ENABLE_DFLASH2=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DSTRATA_BUILD_TESTS=ON
cmake --build build-dflash2 --target strata
python tools/dflash_gguf.py /path/to/model.safetensors \
  --config /path/to/config.json -o /path/to/dflash2.gguf
build-dflash2/strata --pack /path/to/target-pack --native /path/to/target-shard.gguf \
  --dflash /path/to/dflash2.gguf --dflash-window 0 --spec 8 --dflash-block 3 \
  --spec-min-p 0.5 --tokens-file /path/to/prompt-ids.txt --max-new 128
```

Choose the CUDA architecture for the actual card. The build flag defaults to OFF;
a normal build rejects DFlash2 metadata. `--spec` reserves the verifier window;
`--dflash-block K` selects K proposals plus the anchor. The official trained block
has eight rows and allows at most seven proposals. The anchor is never emitted.

Manual DFlash2 runs use a probability floor of 0.5, including when a fixed
`--dflash-block` is given. Setup already uses that floor. The block controls the
draft forward width; the floor limits how many consecutive confident proposals
the target verifies. Use `--spec-min-p 0` explicitly for an ungated experiment.
Verifying uncertain suffix rows can cost time even when the target rejects them.

## Context and memory

Setup passes `--dflash-window 0`: the draft capacity follows `--max-context`.
The engine CLI otherwise defaults to a 32768-cell capacity. A positive value is a
capacity limit, not a rolling context window. Requests exceeding the draft capacity
are refused. Near the end of the available context, the proposal-fit guard shortens
verification to a target-only step rather than overrunning a pool.

DFlash2 owns five FP16 K/V pools using its own head geometry. They remain on the GPU
when the target's KV is streamed to RAM. Weights, pools, scratch and head binding
are allocated before `--expert-cache auto` reads free device memory. The auto cache
therefore uses the selected artifact's actual allocation, not a fixed drafter estimate.

On the profiled CUDA/VMM path, draft pools and attention scratch start at 8192 cells.
They grow as requests need more context by exchanging memory with expert-cache slots.
Without that path the full capacity is allocated at startup. An infeasible allocation
or growth reports an error; a larger configured context is not guaranteed to fit every GPU.
The setup low-RAM estimate includes the selected weight format and full draft capacity.

The [128K VRAM audit](../bench/results/2026-10-19-dflash2-vram/README.md) checks full
BF16/Q8_0/Q4_0 allocations and one actual elastic growth step on RTX 4070 Ti SUPER.
It does not test a full 128K prompt.

## Checkpoint and verification contract

The official geometry is hidden 2560, five layers, 32 Q / 8 KV heads of width 128,
MLP 7680, RoPE theta 10000 and symmetric local attention with window 2048.
Target features are completed layer outputs `[5,15,25,35,45]`, averaged across the
four raw HC streams. The converter marks `dflash.target_hc_mean=true` and
`dflash.target_layer_inputs=false`; runtime capture occurs at the next layer boundary.
Other compatible artifacts retain the prior learned HC contraction unless explicitly marked.

The converter accepts `DFlash2DraftModel` and both published selector-codebook naming
forms. GGUF retains `general.architecture=dflash`, with convolution, selector,
`sample_from_anchor=false`, sliding-window and feature-capture metadata. Unsupported
shapes, scaling, private/pruned heads and DSpark extras are rejected.

Dynamic causal convolutions surround attention and FFN operations. The selector takes
the unary top-K candidates, scores predecessor transitions and follows a greedy path.
Top-K ties prefer the smaller token ID. The GPU selector keeps codebooks in BF16,
uses caller-owned scratch and preserves the reference reduction order. Greedy proposals
are point masses verified by the target; only its accepted prefix is committed.
Sampled requests use target-only decoding and maintain draft context for later greedy reuse.

In-memory prompt and conversation snapshots include the drafter's five committed K/V
prefixes. Rewind overwrites later cells. Disk SAVE/RESTORE remains unsupported.
The [cache regression evidence](../bench/results/2026-10-19-dflash2-pr/cache/README.md)
records fixed-residency comparisons with fresh drafter and target-only controls.

## Validation and measurement limits

Run the relevant loader, policy, taps, RoPE, owned-KV and DFlash2 parity CTests, plus
`tools/test_dflash2_gguf.py`, `tools/test_dflash_quantize.py` and
`tools/test_setup_dflash2.py`. `tools/dflash2_stage_parity.py` compares trained-weight
dumps against an independent NumPy forward, including all five layers and selector IDs.
This checks Strata's FP32/BF16 boundaries, not bit equality with every PyTorch backend.

The [quantization comparison](../bench/results/2026-10-19-dflash2-quant-mtp/README.md)
contains individual runs, commands, failures and output controls. Q8_0 and Q4_0 were
within 1% of MTP overall on four local prompts on RTX 4070 Ti SUPER, within observed
run variation. Code improved, while prose and long-prompt results regressed. These
results do not establish an overall or cross-hardware speedup.

[Review evidence](../bench/results/2026-10-19-dflash2-pr/README.md) distinguishes historical
checks from fresh final-branch validation. No AMD, Intel, Windows, multi-GPU or
concurrent-request correctness claim is made. DFlash2 remains experimental and opt-in.

A [fresh PR performance rerun](../bench/results/2026-10-19-dflash2-pr-performance/README.md)
compares BF16/Q8_0/Q5_0/Q4_0 with bracketing MTP runs at prefill 256 and auto
expert residency. Its settings and output controls are recorded separately from
the earlier prefill-512 comparison.

The [long-case diagnosis](../bench/results/2026-10-20-dflash2-long-prompt/README.md)
checks the probability gate separately. On RTX 4070 Ti SUPER, Q4_0 with block 3
improved from 42.57 to 48.29 decode tok/s at floor 0.5 on the 2425-token fixture;
same-day MTP reached 51.57 tok/s. The target checked fewer rejected proposals.
Both fixtures' output IDs match target-only controls at the same expert placement.
