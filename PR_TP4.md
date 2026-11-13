# SYCL: tensor parallelism (`--tp 2..4`) on Intel Arc

## Summary

This adds tensor parallelism to the SYCL engine. `--tp N` runs every layer on every rank, splits the
dense projections across the ranks, and meets on an all-reduce twice per layer. The model's query and
GDN heads split across the ranks; the KV heads are replicated when there are fewer of them than ranks
(Qwen3.8-Flash-Next has 2 KV heads, so TP=4 replicates them); the MoE experts are replicated per rank.

TP=4 is the target: 4× Arc Pro B60 (96 GB) running the Unsloth UD-IQ4_XS pack.

## Design

- **Shard table** (`sycl/include/strata/core/shard.hpp`): one table per rank, built for 2–4 ranks and
  validated as a tiling of each axis. Output-axis splits need no reduce; contraction splits
  (`ssm_out`, `attn_output`) owe one. Replicated: router, shared expert, hyper-connections, indexer,
  PLE, norms. Rank 0 only: head, embedding, draft layer.
- **TP=4 replicates the QSA layers.** The compiled QSA attention kernel requires
  `n_head == G * n_head_kv` with G in {4, 8, 12, 16}. TP=4's 6 query heads over 2 replicated KV heads
  give G=3, which has no kernel. Replicating the QSA projections keeps the model's G=12 on every rank;
  the GDN layers and the dense/MoE path stay split. The QSA layers are 12 of 48 and the decode is
  expert-bound, so the cost is acceptable.
- **Collective** (`sycl/include/strata/kernels/allreduce.hpp`,
  `sycl/src/kernels/cuda/allreduce.dp.cpp`): host-staged all-reduce with a device-side handshake.
  Two ranks use the original peer-add; three or four ranks use a rank-0 gather/broadcast. Every rank
  rounds through bf16 before the add, so the result is bit-identical across ranks.
- **Loader** (`sycl/src/core/weights.cpp`, `sycl/src/core/native_dense.cpp`): shard-aware canonical
  and native uploads. A sharded row's planes are sized from the split axis; a contraction split keeps
  the same block range inside every row.
- **Per-rank shapes** (`sycl/src/core/layer.cpp`, `session.cpp`, `verify.cpp`): GDN buffers use the
  rank's channels/v-heads; QSA uses the rank's query heads and the replicated KV heads; the session
  and verifier carve the same shapes.
- **Serve wiring** (`sycl/src/program/generate.cpp`): ranks 1..N−1 are built from their own shards,
  sessions, expert-cache replicas and verifiers. Every window runs on every rank in lockstep; the
  expert pool dispatch is serialised; rank 0 samples and commits the answer.

## Verification

- Build: `STRATA_ENABLE_SYCL` with `AOT=bmg-g21`, `icpx` 2026.1, clean (`BUILD EXIT 0`, 0 errors).
- Collective: the standalone SYCL test (`sycl/tools/tp_allreduce_sycl_test.cpp`) passes on two B60s
  with 1, 4 and 8 blocks (`mean 3.000000`, `ALLREDUCE OK`). The host-USM primitives are covered by
  `tp_visibility_test`, `tp_hint_test`, `tp_flag_test`, `tp_bulk_test`, `tp_reverse_bulk_test`,
  `tp_dev_to_host_test` and `tp_barrier_test`.
- Engine: TP=2 and TP=4 start, load the model, replicate the expert cache on every rank, wire the PLE
  on every rank and capture the windows. The verifier's window path implements the layers inline
  (`native_mmvq`), not through `gdn_layer`/`qsa_layer`, so the contraction reduces were initially never
  called; they are now wired at `ssm_out` and `attn_output`. With `STRATA_TP_DIAG_REDUCE=1` the first
  prompt window reaches 16 reduces (`TPDIAG owed_reduce`, `TPDIAG rcall`), then the engine segfaults
  (exit -11). The standalone collective passes host-driven (1/4/8 blocks), so the remaining blocker is
  the captured-graph all-reduce path. The PR is a draft.
- Benchmarks below are the measured engine speed, not a correctness claim.

## Benchmarks

Measured on the Arc box, Unsloth UD-IQ4_XS, 32K context, `--spec 4`, MTP, 66-token prompt, 32
generated tokens:

| Configuration | Decode tok/s | Prompt tok/s (66 tokens) | VRAM free after load |
|---|---:|---:|---:|
| 2× B60 TP=2 | 8.8 | 18.3 | — |
| 4× B60 TP=4 | 7.2 | 16.4 | 9.3 GiB |

The TP speeds are far below the single-card/layer-split numbers in `docs/INTEL_ARC.md` (2× B60 layer
split: 56.6 tok/s decode, 428 tok/s prompt on Coder IQ1_M) because the prompt path runs the verify
windows (the batched path is not wired for a TP group) and the CPU pool computes the misses on every
rank. The correctness blocker is the priority; the speed is not yet a result.

## Limitations

- TP=4 replicates the MoE experts; the CPU pool computes the misses on every rank, so the CPU work is
  N× the single-card misses. Expert parallelism (each rank owns a disjoint expert set and the combine
  is reduced) is the next step.
- The KV heads are replicated for TP=4 (2 KV heads cannot split 4 ways). The KV cache is therefore
  duplicated per rank.
- TP requires `--serve`; the generate-mode loop is not wired.
- Vision and KV streaming are not wired for a TP group in this port.
- The adaptive tier and cache admission are disabled under TP: the replicas must stay identical.

## Dedication

For Billy Preston — "Nothing from Nothing".
