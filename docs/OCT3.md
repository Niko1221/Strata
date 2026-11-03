# `--kv oct3`: 3-bit Lloyd-Max K/V with an exact fp16 window

Opt-in. Without `--kv oct3` the engine runs the code it ran before; the format touches no other path.

## The problem it answers

A QSA layer's K/V is streamed from pinned RAM (`--kv-resident`) and read back through a sparse top-k selection. At
3 bits per value the codes are small (448 B per cell: 55% of k8v4's 816, 78% of q4_0's 576), but the selection is not
uniform: 35-80% of the attention mass lands on a handful of recent cells, and their quantization error does not
average out over the sequence. The error shows up as a worse next-token distribution, not as noise.

The fix is the one vLLM's Octave KV reports: keep the cells that carry the mass in fp16 beside the codes.

## The two pieces

**`kv_octave.hpp` / `kv_octave.cu`** - K and V stored as 3-bit Lloyd-Max codes over a 256-value block, after the same
Walsh-Hadamard rotation Q4_0 uses (`kv_q4.hpp`). The levels are the Lloyd-Max ones for a unit Gaussian, scaled by a
per-block fp16 `d`; 14 bytes per block (2 scale + 12 codes), 112 bytes per head per K or V. Appends and gathers are
shaped like `kv_q4.cu`'s, so the streaming map, the host copy, the staging pool and the snapshot code treat an oct3
layer as "two byte runs at 14-byte blocks" (`KvFormat::kKvOct3`).

**`kv_exact_window.hpp` / `kv_exact_window.cu`** - a per-layer ring of `sinks + tail` rows in fp16, with a 4-byte tag
per row holding the position a row currently answers for. The append path writes the same rotated K/V into it; the
decode attention reads a selected cell from the window when its tag matches, from the codes otherwise. A row that a
newer position has reused does not match, so a stale row is a miss, never a wrong value.

The window is **not part of a snapshot**: it is a cache. A restore or a copy into another state invalidates every
tag (`qsa_window_invalidate`), and the rows refill as the sequence appends.

## Knobs

| knob | default | meaning |
|---|---|---|
| `--kv oct3` | off | the format |
| `STRATA_KV_OCT=1` | - | the same format without the `--kv` name |
| `STRATA_KV_WINDOW=0` | on | turns the window off (pure 3-bit, for A/B) |
| `STRATA_KV_WINDOW_SINKS=N` | 4 | leading cells kept in fp16 |
| `STRATA_KV_WINDOW_TAIL=N` | 1024 | newest cells kept in fp16 |
| `STRATA_PA_OCT=1` | off | the prompt attention on the matrix cores, reading the 3-bit codes (below; needs `STRATA_HIP_WMMA` on gfx11) |

The startup log names the window the session actually sized:

```
strata generate: --kv oct3: 3-bit Lloyd-Max K/V (448 B/cell, gather path; an fp16 window of 4 sinks + 1024 tail cells per layer)
strata generate: KV window: the first 4 and the newest 1024 cells of every QSA layer in fp16 (24.1 MiB over 12 layers)
```

## Quality

Reference: the same engine and model at fp16 K/V. `tools/quant_quality.py` (the repo's own yardstick), 5 x 6000-token
segments, 29,995 positions, next-token distribution per position:

| K/V | top-1 agreement | mean KL to fp16 |
|---|---|---|
| oct3, no window | 91.63% | 0.05261 |
| oct3, 4 + 64 | 93.62% | 0.03342 |
| oct3, 4 + 256 | 94.24% | 0.02559 |
| **oct3, 4 + 1024 (default)** | **95.27%** | **0.01851** |
| oct3, window on K only | 92.42% | 0.04207 |

Each doubling of the tail is cheaper in quality than the last, and 1024 cells cost 12 x 2 MiB = 24.1 MiB of VRAM -
3.6% of what the same cells would cost in an fp16 cache.

On a second, shorter run (4,995 positions) the formats are compared directly:

| K/V | top-1 | KL |
|---|---|---|
| k8v4 (INT8 K + rotated Q4_0 V) | 93.31% | 0.03961 |
| oct3, no window | 89.75% | 0.08623 |
| oct3, 4 + 32 | 92.19% | 0.05506 |
| oct3, 4 + 128 | 93.59% | 0.04454 |
| oct3, 4 + 256 | 94.03% | 0.03704 |

So at 256 tail cells oct3 is already below k8v4's KL at half its bytes; the 1024 default buys the rest.

Long context (`tools/needle_bench.py`, this model, single card): 96K and 128K at depth 50 both found in the runs
recorded under `perf/raw/needles-win-*.json`. One 96K run missed (`needles-win-pf32n1.json`, a single sample); k8v4
on the same box finds 32K and 128K at depths 10/50/90 (`needles-k8v4.json`).

## Cost

Measured on one RX 7900 XTX, Qwen3.8-Flash-Next GSQ-RCO IQ3_S. The two formats were run back to back on the same
config except `--kv` (`perf/raw/perf-oct3b.jsonl`, `perf/raw/perf-k8v4b.jsonl`, `--kv-resident 32768`,
`--prefill auto:8192`, 4 reps per size):

| | oct3 | k8v4 |
|---|---|---|
| prefill 32K | 1458 tok/s | 1461 tok/s |
| prefill 96K | 1378 tok/s | 1373 tok/s |
| decode, ms per forward, 32K / 96K | 25.64 / 26.17 | 24.91 / 25.55 |
| draft acceptance (16-prompt harness) | 60.0-62.1% | 63.1% |

Prefill is the same. Decode per forward is 2-3% higher in that pair and lower in others (`docs/PERF.md` has every
run); the throughput is 1-3% lower because a 3-bit K loses a point or two of draft acceptance against an 8-bit one.
The production config (oct3, `--kv-resident 65536`, `--prefill auto:32768`) measures 1880 tok/s prefill at 32K,
1794 at 96K, 25.0 / 25.5 ms per forward.

## The matrix-core prompt attention for 3-bit codes (`STRATA_PA_OCT=1`)

Before this switch, oct3 prefilled at 2280 / 2304 tok/s (32K / 96K) against `--kv int8`'s 3506 / 3559 with the same 23
switches (`docs/PERF.md`). The 1.5x was not tuning: on AMD `qsa_prompt_attn_batch` took only INT8 pools, oct3 keeps its
codes in the q4 pointers (`src/core/layer.cpp`: `st.k_q4 = c.take<uint8_t>(rows * oct_row)`), and our own
`if (pools.oct3) return false;` refused the call outright - so every oct3 prefill ran the decode kernel's chunk path
(`attn_chunk_kernel<4>`: 64-cell chunks, FP32 FMA per element, one 3-bit unpack per value, no tensor cores).

`STRATA_PA_OCT=1` opens that gate with an oct3 arm of the same kernel (`prompt_attn_wmma_kernel<SPLIT, OCT>`):

- **The codebook in registers.** An arm's 16 values come from three 16-bit bit-planes - the block's 12 code bytes, low
  halves for the first 16 values, high halves for the next. Four codes at a time are spread out of one nibble of each
  plane by the `0x00204081` multiply, combined into a 4-byte code word and mapped through the 8-entry int8 codebook
  with a single `v_perm_b32`; the values enter the MMA as the int8 path's fp16 fragment does.
- **Two scales per wave.** oct3 carries one fp16 `d` per 32 values where int8 carries one per 64, so a wave (64 dims)
  spans two: q.k keeps two accumulator sets and adds them scaled after the MMA, and p.v builds one A fragment per
  32-dim half (`vs`/`vsb`). `d` is folded with the codebook's fixed 1/32, so everything after the MMA is the int8
  path's arithmetic and the p.v staging rows keep their 80 bytes.
- **The exact window is not read by the prompt path** - it never was. With `--kv-resident` the prompt attention runs
  on the staged pool (`pools_of`, which carries no window) in both formats; the window serves decode only.

Correctness: `qsa_prompt_attn_parity` now runs oct3 beside int8 / FP16 / Q4_0. Against the FP64 reference the new
kernel's worst error is 1.1e-9 on an output scale of 2.2e-3; against the FP32 mode-4 chunk kernel it replaces,
2.0e-9 (7.9e-7 of scale) - the same answer, another summation order. The decode itself is checked bit for bit
against `load8_oct`'s formula over 4096 random blocks (V path and K path).

4 reps each, production config, one session (`perf/raw/perf-paoct2.jsonl`, `perf/raw/perf-int8ref.jsonl`):

| KV | 32K prefill | 96K prefill |
|---|---|---|
| oct3, no `STRATA_PA_OCT` (the 0.1.54 deployment) | 2298 | 2305 |
| `--kv int8` | 3516 | 3557 |
| **oct3 + `STRATA_PA_OCT=1`** | **3443** | **3513** |

The deployed service (the unit, the production config, `perf/raw/perf-deployed-paoct.jsonl`, 4 reps): **3457 / 3455 /
3444** at 32K and **3537 / 3504 / 3507 / 3505** at 96K - the forum's 3.5K, 51% over the oct3 gather path it replaces.
96K needles on the same service: 3 of 3 at depths 10/50/90.

The 3-bit format now prefills within 1-2% of int8, at 2.4x fewer bytes per cell. Kernel level (synthetic, 2,048
queries over a 32K context): 8.59 ms per chunk against int8's 8.05 (+6.7%), 4.2x the FP32 path it replaces.
`HIP_WMMA` alone never helped oct3 because the format gate rejected the call before this arm existed (`2090 / 1928`
with it, `2243 / 1979` without, `perf/raw/perf-fa54-wmma.jsonl`).

### Accuracy

`STRATA_PA_OCT_CHECK=N` (debug) runs both kernels on the same N queries of a chunk and prints the largest difference;
on this box and the production config, a 4,500-token prompt, the first two QSA layers:

| arm | max \|matrix cores - chunk path\| / output scale |
|---|---|
| oct3, `STRATA_PA_FAST=1` | 3.5e-4 |
| oct3, `STRATA_PA_FAST=0` (q and p as fp16 hi + lo) | 7e-7 - 9e-7 |
| int8, `STRATA_PA_FAST=1` (the deployed int8 arm) | 2.4e-4 - 2.7e-4 |
| int8, `STRATA_PA_FAST=0` | 1.0e-6 - 1.1e-6 |

So the fast arm costs what the int8 arm's fast path already costs (a single fp16 image of q and p), and the split arm
is the FP32-level answer. `tools/needle_bench.py` at 32K, depths 10/50/90 (3 samples each, this box): the chunk path
3/3, `PA_OCT` with the split 3/3, `PA_OCT` with `PA_FAST=1` 3/3, `--kv int8` with `PA_FAST=1` 3/3 - but two runs of
the fast arm on an earlier build of the kernel missed all three, so a 3-sample needle run is not a quality gate for a
3.5e-4 change; a longer-sampled run is the open item.

**The exact window in the prompt path.** A session without `--kv-resident` hands the prompt attention
`qsa_attn_pools(st)`, which carries the window; the staged pools of the production config carry none. The matrix-core
arm reads it when it is there:

- **K** - the window row's fp16 values go straight in as the arm's B fragment (each lane holds one cell's 16 dims), with
  the cell's scale 1 instead of the codebook's `d/32`;
- **V** - the window's cells have their codes masked out (`vs`/`vsb` 0) and their exact values added in FP32 as the
  chunk's weights settle (4 V reads + 8 weight reads + 32 FMAs per window cell per lane), skipped entirely by a
  wave-wide vote when a chunk holds none;
- the window's rows live in registers (`__shfl` over lane = cell), not in shared memory: a shared array is rewritten by
  the next chunk's staging before the slower waves have finished their p.v.

Measured (`qsa_prompt_attn_parity`, which now runs an `oct3+win` arm beside the others: 4 sinks + 1024 tail rows, hits
mixed with misses inside a 16-cell tile): every arm passes, and the window arm's output is the FP32 chunk path's -
which reads the same window - to 1.2e-6-1.6e-6 of scale at 32K, 2.1K and 1.5K contexts. Kernel level, 32K context,
2,048 queries: 9.44 ms per chunk against 37.2 for the chunk path (3.9x); the window's fp32 additions cost ~10% over
the 8.60 ms of the no-window arm. End to end, `--kv oct3` without `--kv-resident` (the configuration where the window
is live): **32K 2,305 -> 3,429 tok/s, 96K 2,298 -> 3,087-3,480** (`perf/raw/perf-nonres-*.jsonl`), 3 of 3 32K needles,
the same window quality the chunk path had.

Two bugs the parity arms caught while this was built: the window row travelled through an argument of `fetch()` whose
evaluation order against `row_of()` is unspecified (the row was read as "no hit"), and the shared `wrow[]` array was
stale-racy (nondeterministic output between two runs of the same arm).

## What it does not do

- **Gather path + the prompt attention above.** There is no fused decode-attention mode for oct3 (`qsa_decode_attn.cu`
  falls back to the chunk kernel with `KV_MODE == 4`), and no row-batched or slot-batched attention (those exit with a
  message if asked for oct3). `--kv oct3` with `STRATA_BATCH_ATTN_ROWS` or a multi-slot server is refused, not
  silently misread. The prompt attention takes oct3 (and its exact window) only with `STRATA_PA_OCT=1`
  (`qsa_prompt_attn_batch` otherwise returns false and the caller keeps the chunk path).
- **The MTP drafter stays on plain INT8** (`mtp.cpp`), for the same reason it does under k8v4.
- **Not a snapshot format.** A slot image never contains the window; restoring invalidates it.
- **Experimental.** The format is not a default and upstream has not accepted it.

## Files

New: `include/strata/kernels/kv_octave.hpp`, `src/kernels/cuda/kv_octave.cu`, `include/strata/kernels/kv_exact_window.hpp`,
`src/kernels/cuda/kv_exact_window.cu`, `src/kernels/kv_octave_parity.cpp`, `src/kernels/kv_exact_window_parity.cpp`.

Touched: `CMakeLists.txt`, `include/strata/core/layer.hpp`, `include/strata/kernels/kv_stream.hpp`,
`include/strata/kernels/qsa_decode_attn.hpp`, `include/strata/prefill/prefill.hpp`, `src/core/layer.cpp`,
`src/core/verify.cpp`, `src/core/conversation_snapshot.cpp`, `src/core/mtp.cpp`, `src/kernels/cuda/kv_stream.cu`,
`src/kernels/cuda/qsa_decode_attn.cu`, `src/kernels/cuda/qsa_prompt_attn.cu`, `src/kernels/kv_stream_parity.cpp`,
`src/prefill/prefill.cpp`, `src/program/generate.cpp`, `tools/quant_quality.py` (`--kv-format`, so the yardstick can
measure a KV format).

The patch is `diffs/oct3-v0.2.8.patch`.
