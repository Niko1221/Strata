# SYCL port: speedup TODO

Ordered by how little card time each one needs to test. Measured numbers live in
[docs/INTEL_PERFORMANCE.md](../docs/INTEL_PERFORMANCE.md); each item is checked with benchy v1 (`sycl/benchy.sh`) and an
output-identity comparison against the build before it.

## 1. New hardware readiness (no card time)

The B70 trained at PCIe Gen3 x8 in the old host's slot (it can do Gen5 x16); the new host gives both cards Gen5 x8. IQ2_XS and Swift read their ~6,000
mirrored experts over that link at every decode step, and each start reads 30-43 GB from the SSD. A second card
lets a layer split hold the IQ2_XS without a host mirror.

- [x] Multi-card path checked without a second card (2026-10-04): `setup_intel.py --check` with two simulated
      cards now sizes models across both (it used the first card only) and names the Intel path; the config keeps
      `"gpu"`/`layer_split`, the server adds `--layer-split`, `strata-sycl.sh` opens the selector to every card,
      benchy reads every card. A second card of another die needs a two-target build (`AOT=bmg-g21,bmg-g31`,
      checked to compile).
- [x] On the new box (2026-10-04): benchy v1 with the same engine, B70 at Gen5 x8 on a Ryzen 9 9950X with 61 GB:
      IQ2_XS / Swift decode +15-30% and long prompts ~25% faster (the RAM mirror over a 26.5 GB/s link, 6.1 before),
      the Coder +15-40% prompt, +3% decode. The B65 alone: 70-77% of the B70 (INTEL_PERFORMANCE.md).
- [ ] With the second card: a layer split of the IQ2_XS (all experts resident) through setup and benchy.

## 2. Refill lent cache slots in the background (short-prompt test)

Above 32K context the prompt path borrows VRAM cache slots and refills them from the GGUF before the first decode
round: ~1 s per request (950 ms for 1,050 slots). With served 128K/256K configs that is most of the gap between a
2,185-token prompt at 479 tok/s and the 32K config's 790.

- [x] Done another way (2026-10-04, simpler and safer than refilling during decode): the lendable experts are
      mirrored in pinned RAM at start, so the prompt path DMAs them per chunk (it read them from the GGUF) and the
      refill copies from RAM (`STRATA_LEND_MIRROR=0` turns it off). Outputs identical (Coder 2,185 / 8K / 40K,
      IQ2_XS 2,185 / 8K, cold cache). Prompt reading Coder 456 -> 610 / 802 -> 946 / 988 -> 1,046 tok/s, IQ2_XS
      303 -> 500 / 565 -> 720; refill 720-2,240 -> 200-380 ms; Coder 2,185-token TTFT 4.74 -> 3.87 s. Costs 1.9-2.3
      GiB of RAM and ~2 s at start.
- [ ] Still open: decode before the refill lands (would take the remaining ~0.3 s off TTFT).

## 3. Long prompts (parity tests + 40K-256K runs)

At 40K: expert GEMMs 23%, attention 19%, dequant 16%, QSA select 7%, host grouping 6.5%, gather 6%.

- [x] **3a. Expert grouping on the GPU: not worth it** (measured 2026-10-04). Host timers over a 40K prompt's 528
      groupings: the loops 47 ms, the uploads 24 ms, and 2,336 ms the profiler's own event fold - the "6.5%" was
      mostly the measurement. The 26 s "drain wait" is the host waiting for GPU work it had queued, not idle GPU.
- [x] **3b. QSA block selection: done** (2026-10-04). The scores as oneMKL fp32 GEMM tiles plus a relu-sum
      (`sel_scores_bench`: 7.3-8.4x the per-pair kernel at 40K-256K, relative error 2-4e-7). Coder TTFT 256K
      328.6 -> 258.2 s (780 -> 992 tok/s), 128K 138.9 -> 122.9 s, 40K 38.4 -> 37.4 s; IQ2_XS 40K 762 -> 785 tok/s.
      Not bitwise (as the CUDA build's 3xTF32 path): outputs follow the same text and part at a near-tie after 13-93
      tokens, equally coherent. `STRATA_SELECT_GEMM=0`: the old kernel.
- [x] **3c. Prompt attention: done, 1.5x kernel / ~10% TTFT** (2026-10-04, 15 attempts in `attn_bench`, 32 queries x
      2,048 cells, INT8 KV, 131K context; the old kernel 0.57 ms):
      - kept: scores one work-item per cell with the query heads read from local memory, 128-cell chunks: 0.37 ms.
        Coder TTFT 40K 37.2 -> 33.5 s, 128K 123.0 -> 111.3 s, 256K 258.3 -> 235.1 s. `STRATA_ATTN_PERCELL=0`: old.
      - measured: the K/V gather alone is 0.07 ms (the selections share cells and stay in cache) - the kernel is
        arithmetic-bound; the values pass is ~60% of it now, ~0.14 ms of that its V reads.
      - no gain: SYCL's native sub-group reduction in the engine's order (1.03x), K staged in local memory (0.29x),
        64 / 256-cell chunks, 4 dims per work-item (spills: 0.56x), float4 weight loads, 4-byte V loads, V staged in
        local memory (0.54x), V scales in local memory, a V row pointer table, sub-group block reads (0.95x), two
        work-items per cell (0.91x).
      - XMX: the tree's v2 kernel 1.68 ms (120 KB of local memory: one work-group per core), a lean fp16 XMX values
        pass 1.6-2.1 ms; sub-group 16 alone costs only 0.05 ms, so the joint_matrix path itself loses here.

## Later

- Fewer decode graph nodes (norm+rope, scores+top-k, gate+quantize fused): launch gaps are 15-20% of a round.
- Two draft branches per verify window (decode is latency-bound).
