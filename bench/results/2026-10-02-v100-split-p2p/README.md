# The layer-split hand-off across NVLink (2026-10-02)

**Rig:** 2x Tesla V100-SXM2-16GB on one board, `nvidia-smi topo -m` says **NV2** between them (two bonded NVLinks),
114 GB RAM, CUDA 12.8, Linux. Both cards sm_70, `cudaDeviceCanAccessPeer` 1/1, `cudaDeviceEnablePeerAccess`
succeeds both ways.

**Tool:** `bench/split_p2p_bench.cu`, standalone (no engine libraries):

    nvcc -O3 -arch=sm_70 -o /tmp/split_p2p_bench bench/split_p2p_bench.cu
    CUDA_VISIBLE_DEVICES=0,1 /tmp/split_p2p_bench

Raw output: `data/bench.txt`. Every path is checked to copy a pattern byte for byte before it is timed.

## What the engine does today (the audit)

A `--layer-split` hands the activation from one card to the next once per verify window and once per prompt
chunk. Both hand-offs cross through **pinned host RAM** - no peer-to-peer (docs/MULTI_GPU.md says so):

| Hand-off | Write side | Read side | Sync |
|---|---|---|---|
| verify window (decode) | `copy_from_mapped` kernels, device -> mapped pinned host, `src/core/verify.cpp:816-820` (inside the captured window graph, `Verifier::record_window`) | `copy_from_mapped`, mapped host -> device, `src/core/verify.cpp:422-426` (same graph) | `cudaStreamSynchronize(cs_)` at `src/core/verify.cpp:1125`, then `next_->run` at `:1170`; buffers allocated `src/program/generate.cpp:4153-4165` (`cudaHostAllocMapped \| Portable`) |
| prompt chunk | `cudaMemcpyAsync` D2H + `cudaStreamSynchronize`, `src/prefill/prefill.cpp:1868-1873` | `cudaMemcpyAsync` H2D on the next stage's thread, `src/prefill/prefill.cpp:1060-1063` | two pinned chunk buffers used in turn, `src/prefill/prefill.cpp:528-531`, one chunk of overlap |

Sizes: one verify token hands `hc*n_embd + n_embd + hc` = 12,804 floats (51 KB) - `Verifier::handoff_floats`,
`include/strata/core/verify.hpp:118`; a window is 2-8 tokens (`kVerifyMaxT = 8`), so 100-400 KB. One prompt
chunk hands `chunk * hc*n_embd` floats (`Prefill`'s `D` = 10,240), so 20-104 MB at chunks 512-2560.

## Measured (one hand-off, wall clock incl. the sync the engine needs)

Paths: **d2d-auto** = `cudaMemcpyAsync` DeviceToDevice with peer access OFF (the driver stages through host);
**p2p-memcpy** = `cudaMemcpyPeerAsync`; **p2p-kernel** = a copy kernel on the source card storing into the peer
card's memory; **pin-bounce** = D2H + H2D through one pinned buffer (prefill today); **map-bounce** = kernel ->
mapped pinned -> kernel (verify today).

| shape (one hand-off) | d2d-auto | p2p-memcpy | p2p-kernel | pin-bounce | map-bounce |
|---|---|---|---|---|---|
| verify 2 tokens (100 KB) | 0.047 ms | **0.010 ms** | 0.012 ms | 0.046 ms | 0.033 ms |
| verify 4 tokens (200 KB) | 0.081 ms | **0.011 ms** | 0.013 ms | 0.075 ms | 0.066 ms |
| verify 8 tokens (400 KB) | 0.138 ms | **0.015 ms** | 0.017 ms | 0.137 ms | 0.122 ms |
| 2560x256 float (2.5 MB) | 0.581 ms | **0.061 ms** | 0.066 ms | 0.810 ms | 1.063 ms |
| 2560x4096 float (40 MB) | 6.84 ms | **0.873 ms** | 0.926 ms | 12.68 ms | 18.51 ms |
| prompt chunk 512 (20 MB) | 3.51 ms | **0.441 ms** | 0.469 ms | 6.35 ms | 8.58 ms |
| prompt chunk 2048 (84 MB) | 13.50 ms | **1.739 ms** | 1.842 ms | 25.34 ms | 38.05 ms |
| prompt chunk 2560 (104 MB) | 16.85 ms | **2.170 ms** | 2.299 ms | 31.67 ms | 46.98 ms |

Useful GB/s (one payload, so a bounce's second PCIe crossing counts against it):

| shape | d2d-auto | p2p-memcpy | p2p-kernel | pin-bounce | map-bounce |
|---|---|---|---|---|---|
| verify 2 tokens (100 KB) | 2.2 | 10.3 | 8.7 | 2.3 | 3.1 |
| verify 4 tokens (200 KB) | 2.5 | 18.0 | 15.8 | 2.7 | 3.1 |
| verify 8 tokens (400 KB) | 3.0 | 26.6 | 23.5 | 3.0 | 3.4 |
| 2560x256 float (2.5 MB) | 4.5 | 42.9 | 39.7 | 3.2 | 2.5 |
| 2560x4096 float (40 MB) | 6.1 | 48.0 | 45.3 | 3.3 | 2.3 |
| prompt chunk 512 (20 MB) | 6.0 | 47.5 | 44.8 | 3.3 | 2.5 |
| prompt chunk 2048 (84 MB) | 6.2 | 48.2 | 45.6 | 3.3 | 2.2 |
| prompt chunk 2560 (104 MB) | 6.2 | 48.3 | 45.6 | 3.3 | 2.2 |

## What the numbers say

- The link does **48.3 GB/s** one way (`cudaMemcpyPeerAsync`), the pinned bounce **3.3 GB/s** useful and the
  mapped-kernel bounce **2.2-3.4 GB/s**: the hand-off is **15-20x** faster over P2P on a full chunk and **3-9x**
  faster on a verify window (0.015 vs 0.122 ms at 8 tokens, the decode-critical size).
- `cudaMemcpyPeerAsync` is ~5% faster than the copy kernel on the big shapes (48.3 vs 45.6 GB/s) but the same
  order on the small ones the verify path uses.
- **`cudaMemcpyPeerAsync` is REJECTED inside `cudaStreamBeginCapture`** (capture + instantiate + launch, bench
  prints it): the verify hand-off sits inside the captured window graph, so that path must move its bytes with a
  kernel. The prompt path is not captured and takes `cudaMemcpyPeerAsync`.
- `cudaMemcpyAsync` DeviceToDevice with peer access off is the driver's own host staging: 6.2 GB/s on a big
  chunk, better than the two explicit bounces but 8x under the link.

## A/B

`STRATA_SPLIT_P2P` (engine, after this change): unset = P2P when both cards of the pair are Volta (sm_70) and can
peer - the measured configuration - and the pinned path everywhere else (sm_80+ behavior unchanged), `0` = force
the pinned path, `1` = force P2P wherever peer access works. HIP builds keep the pinned path unconditionally. The
A/B is bit-identical: the hand-off is a copy of the same rows, and a 330-token prompt + 32 greedy tokens on 2x V100
(`--layer-split auto`, Swift IQ2_XS) produced the same 32 token ids with the switch on and off.
