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

`./setup.sh --calibrate` measures settings for the installed artifact on this PC.
After the decode sweep, DFlash2 also compares fresh short and long requests. It tries
prefill sizes 256, 512 and 1024, checks startup VRAM headroom and measures block length
and probability floor together. It records the actual expert-cache size for every
profile. A larger prefill is not assumed to be faster.

The two best prompt profiles and the baseline enter the joint search. The winner is
restarted against the baseline in three alternating pairs, including extra prompt
cases. It is saved only when every pair improves and the median balanced request
score improves by more than 3%. Prompt and decode times stay separate in the report.
Short and long workloads have equal weight; early EOS is normalized to the same output
budget. This is a bounded search, not a guarantee of the best settings for every task.
Allow roughly 30–90 minutes for the full DFlash2 calibration. A failed candidate is
skipped; cancellation leaves the installed configuration unchanged.

The calibration does not choose a weight format or a separate head. Change
`--dflash2-quant` through setup to compare formats. Its saved result is specific to the
card, CPU, RAM, model, artifact, draft capacity and context; changing the calibrated
block itself does not hide the result from a reinstall.

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
when the target's KV is streamed to RAM. The official checkpoint has a local window
of 2048 cells, so the draft K/V now uses a fixed device ring sized for that window,
the largest verifier block and page alignment. On the CUDA test build, four-cell
pages and the eight-row verifier block use 2060 cells per draft layer. RoPE positions
and the logical context still follow `--max-context`; older K/V pages are outside
DFlash2's attention window.
Attention scratch is bounded by the same window and block. It does not grow as the
conversation gets longer.

Weights, the bounded pools and scratch are allocated before `--expert-cache auto`
reads free device memory. The auto cache can therefore use the VRAM that a full
512K draft pool and scratch previously reserved. DFlash2's local pools no longer
borrow chunks from the expert cache while decoding. The target's KV policy is
unchanged. An infeasible startup allocation still reports an error, and the complete
model plus expert cache must fit the PC's available VRAM.

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
pages in logical order, including the retained tail of its device-only local ring.
Disk SAVE/RESTORE remains unsupported for DFlash2.
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

The [attention follow-up](../bench/results/2026-10-20-dflash2-attention/README.md)
removes repeated softmax and page-address calculations in DFlash2's CUDA attention.
On the same card and long fixture, drafting fell from 4.18 to 3.19 ms/window
and decode rose from 48.45 to 49.39 tok/s, with identical output tokens. Fresh
MTP reached 51.24 tok/s; it remains faster on this long fixture. Both DFlash2
arms used the same 3274 expert slots, selected automatically from available VRAM.

## Optional Q8_0 head projection on SM89

The target's actual output-head format matters independently of the drafter's
weight format. A Q4_0 drafter can share a Q8_0 target head. Startup now prints
the loaded head's real format; the older message always said Q5_K.

`STRATA_MMVQ_IL_Q8_HEAD=1` enables the measured interleaved Q8_0 projection for
248320 output rows and 3–8 input rows on SM89. Other shapes and architectures
keep their existing dispatch. DFlash2 also needs `STRATA_DFLASH_HEAD_IL=1` to
use it for its own output projection. The first switch can accelerate the
shared target verifier too; give an MTP control the same switch when comparing.
The kernel keeps the original integer dots, scaling and FP32 reduction order.
Both switches are opt-in. No head weights or vocabulary rows are removed.

The [long-prompt follow-up](../bench/results/2026-10-20-dflash2-long-tuned/README.md)
records the complete measurements, confidence/block settings, expert placement
and output checks. A setting measured on one prompt is not a general default.

The follow-up's Q8_0 drafter profile uses block 7 and floor 0.7, with both head
switches enabled. On RTX 4070 Ti SUPER / Ryzen 9 5900X it reached 56.77 versus
55.31 decode tok/s for MTP on the 2425-token fixture. Expert placement was fixed
and checked: 3188 slots for DFlash2, 3143 for MTP. These are settings for that
measurement, not new defaults or a recommendation for every prompt.

The same change reuses attention's K/V addresses, specializes official power-of-two
page indexing and pairs adjacent values in long windows. The original ordered
sums and masks remain unchanged. General geometry and unaligned buffers retain
a scalar path. The report includes bitwise comparisons and the whole-model limits.

## Fresh long-prompt latency profile

On RTX 4070 Ti SUPER 16 GiB / Ryzen 9 5900X, the measured 2425-token fixture with
96 output tokens took 11.42 s with batched prefill 256 and 7.76 s with the selected
prefill-512 profile (three repeats, -32.04%). That profile uses Q8_0 DFlash2,
block 6, floor 0.85, 2900 actual expert slots and `--short-read 64`. It decodes at
51.95 versus same-day MTP's 50.35 tok/s (+3.18%); whole-request latency is tied.

These counts are for that machine. Let calibration measure another PC. The
[report and raw runs](../bench/results/2026-10-20-dflash2-fast-prompt/README.md)
include the slower prefill-1024 profile, controls and limits. The engine executable
is unchanged; this is a profile improvement. The earlier controlled benchmark's
`--short-read 4096` deliberately bypasses batched prefill for that prompt. It remains
appropriate for reproducing that decode table, but is not the normal prompt path.
Strata already defaults to `--short-read 64`.
