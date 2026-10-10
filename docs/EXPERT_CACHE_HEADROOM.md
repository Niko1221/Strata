# The expert cache's VRAM headroom (#1549)

`--expert-cache auto` sizes the VRAM-resident expert tier to every byte the card reports free. That is
wrong, and on a card that is nearly full it is wrong by a lot: the prompt path then reads tens of times
slower while doing exactly the same work. This is what is measured, and how to test the knob on a card
it has not been measured on.

## The knob

```
STRATA_EXPERT_CACHE_HEADROOM=<fraction>
```

`"1/6"`, `"0.17"`, or `"0"` for the old sizing. It is the share of free device memory that `auto` keeps
for the session's other buffers instead of experts, and it is read by the CUDA/HIP and the SYCL sizing
alike (the helper is in `include/strata/core/expert_cache.hpp`). An explicit `--expert-cache N` is the
user's own budget and is never cut.

**The default is zero on CUDA and HIP**, because the measurement below is from an Intel card. Those
builds' default path is byte-identical to the last release. The SYCL port defaults to a sixth, where it
was measured.

## What is measured on the Arc Pro B70

One card, Windows 11, driver 32.0.101.8976, 32 GB, the Qwen3.8-Flash-Next IQ3_XXS pack, OpenCL backend,
eager replay. Every number below is `--prefill 4096`, greedy, with the same 3,668 gate/up and 3,668 down
naive GEMM calls at every cache size — only the cache size and the memory left over change.

| expert cache | VRAM of experts | prefill, 64-token prompt | ms per expert GEMM call |
|---|---|---|---|
| 17,687 slots (`auto` before) | 28.82 GiB | 426 s | 111 |
| 15,996 slots | 26.03 GiB | 222 s | 54 |
| 15,382 slots | 24.95 GiB | 472 s | 122 |
| **14,704 slots** | **23.87 GiB** | **9.7 s** | **2.32** |
| 10,698 slots | 17.36 GiB | 10.3 s | 2.39 |

The split is a cliff, not a slope: every size up to 14,704 slots is stable at about 10 s, and every size
from 15,382 up takes minutes. Above the cliff the times are noisy (205-472 s measured); below it they
are not.

Where the 426 s goes, and where it does not: the two expert GEMM phases are 95.8% of the GPU timeline,
and the phase that waits for the host-to-device staging copies is **34 ms of 957,283**. The bytes are
not the problem. With the cache cut, 14,688 slots (23.83 GiB, 7.17 GiB kept free) puts the same calls at
2.48 ms and the prompt at 12.8 s.

Memory during the two passes (3-second windows): the slow one peaks at 31.2 GiB of dedicated memory —
the card full — and holds about 15 GiB of shared memory while it runs. The fast one peaks at 30.08 GiB
dedicated and holds about 3.4 GiB shared. The mechanism is consistent with WDDM paging an
over-subscribed allocation to system memory, which the engine's own comment at
`generate.cpp` ("under WDDM an over-subscribed allocation does not fail, it pages to system memory and
crawls") describes, but it is not proven.

**Not the cause, each measured:** the host mirror (35% of the same pack mirrored over PCIe costs
nothing: 2.05 ms per call against 2.01 with every expert resident); the prompt length (the cost is
fixed per expert call, not per token); subnormals in the dequantized weights or the activations (there
are none in either pack); the `--kv-resident` setting the two configs disagree on (7,589 ms against
7,587 ms). The kernel calls and their shapes are identical in every arm.

**No accuracy cost.** The cut and the uncut sizing give identical greedy output tokens — eight tokens,
`271 71093 271 550 18381 198 12 1510`, from a scrambled 48-token prompt.

## Testing it on another card

This is what the knob is for. Nothing here has been measured on NVIDIA, and the mechanism may be
Windows-specific, so the first question the test answers is whether the cliff exists at all.

### 1. Build

A CUDA 12.8 or newer toolkit for Blackwell. `sm_120` is already the CMakeLists default, so on an RTX 50
card:

```sh
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target strata -j
```

For another architecture add `-DCMAKE_CUDA_ARCHITECTURES=<sm_XX>` (`nvidia-smi
--query-gpu=compute_cap --format=csv`). `STRATA_NATIVE_EXPERTS` is on by default, so native packs work
as they are. If `generate.cpp` does not compile, it is the `const size_t headroom = ...` line in the
sized-slots block, and that is the one thing not compiled by the author of the change (no `nvcc` on the
measuring machine).

### 2. Pick a model that does not fit

The headroom only means anything when `auto` wants to fill the card. A model whose experts fit entirely
in VRAM gives identical numbers in both arms and the test says nothing. On a 16 GB card the Coder IQ1_M
(23.42 GiB of experts) is the natural case: about 60% of its experts sit in the host mirror in both
arms, which is more PCIe traffic than the B70 measurement above had, so the comparison will be noisier.

### 3. Run the A/B

Same binary, two environments, interleaved, **medians of three**, so drift cancels:

```sh
export STRATA_EXPERT_CACHE_HEADROOM=0      # the old sizing: the baseline
export STRATA_EXPERT_CACHE_HEADROOM=1/6    # the treatment
```

Run 0, 1/6, 0, 1/6, 0, 1/6.

### 4. Record, per run

- the `expert cache N slots, X GiB of VRAM` line — confirms the knob took effect
- the `prefill ... ms (N tok/s)` line — the headline
- **decode tok/s as well**: the headroom leaves 1/6 of VRAM unspent, which is fewer resident experts and
  more PCIe reads per token. If the prompt wins and decode loses, that is a tradeoff and the size of it
  needs the numbers. The B70 measurement above is prompt-only and does not answer this.
- memory during the run, watching for the spill

### 5. What each outcome means

| Outcome | Conclusion | Next step |
|---|---|---|
| prompt much faster, decode about the same | the cliff is not Intel-specific | the numbers turn the default on for CUDA |
| no difference | the effect is WDDM-specific, or the card's headroom was never the limit | CUDA's default stays 0 |
| prompt faster, decode slower | 1/6 is too big for this card | sweep `1/8` and `1/4`, report the knee |

A null result is a real result: it would say the default belongs behind an OS check rather than
everywhere.

## Not tested

Any NVIDIA card. Linux. Whether the cliff moves with `--prefill`, the KV reservation, or the pack. Any
model other than IQ3_XXS on the B70. Decode on the B70 with the cache cut. The CUDA and HIP sources
carry the change but were not compiled where it was written.
