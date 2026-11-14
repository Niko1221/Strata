# SYCL: tensor parallelism (`--tp 2..4`) on Intel Arc

## Summary

This adds tensor parallelism to the SYCL engine. `--tp N` runs every layer on N cards at once. It is
opt-in: without `--tp` nothing changes.

The target is 4× Arc Pro B60 (96 GB) with the Unsloth UD-IQ4_XS pack. On that box, with the mode
recommended below, `--tp 4` reads a 2,072-token prompt at 958–1,036 tok/s against 497–531 on one
card, and decodes at 44.2–44.6 tok/s against 12.0. On a 259-token prompt the 128 generated tokens are
the single card's; on longer prompts the output differs after a few tokens (see Verification).

## The three modes

| Mode | How to select it | Dense layers | Experts |
|---|---|---|---|
| Sharded, replicated experts | `--tp N` | cut across the ranks | every rank holds a copy of rank 0's cache |
| Sharded, expert split | `--tp N`, `STRATA_TP_EXPERT_SPLIT=1` | cut across the ranks | each expert belongs to one rank |
| Replicated dense, expert split | `--tp N`, `STRATA_TP_EXPERT_SPLIT=1`, `STRATA_TP_DENSE=replicated` | whole on every rank | each expert belongs to one rank |

The third mode is the one to use. Every rank holds the whole dense model (about 1.4 GiB of pack and
1.4 GiB native) and computes only its own experts. The ranks meet once per layer, after the experts.
It is the only mode with a batched prompt path, and its output matches the single card.

**Where the experts live.** Rank 0 keeps its usual cache. The next-hottest experts are dealt to the
other ranks' VRAM, up to a cap per rank (`STRATA_TP_RANK_CACHE_MIB`, default: the size of rank 0's
cache). The rest stay in pinned system RAM and their owner reads them over PCIe. With the default cap
on 4× B60 about 4,370 experts sit in each card's VRAM and 7,092 stay in RAM.

## Design

- **Shard table** (`sycl/include/strata/core/shard.hpp`): one table per rank, for 2–4 ranks. Used by
  the two sharded modes. Output-axis splits need no reduce; contraction splits (`ssm_out`,
  `attn_output`) owe one.
- **Expert ownership** (`sycl/src/program/generate.cpp`): one owner per expert. A rank's kernels skip
  an expert it does not own; the shared expert is counted on rank 0 only.
- **Collective for decode** (`sycl/src/kernels/cuda/allreduce.dp.cpp`): an in-kernel all-reduce over
  host-staged memory with a device-side handshake. It moves 8 bytes per PCIe transaction: a 4-rank
  reduce of 15,360 values takes 0.26 ms (2.0 ms before). Its give-up bound is at least 400,000 reads
  (`STRATA_TP_SPIN_MIN`).
- **Prompt path** (`sycl/src/prefill/prefill.cpp`, `sycl/include/strata/program/tp_pair.hpp`): every
  rank runs its own batched `Prefill` over the same chunk. After each layer's expert combine the
  ranks' outputs are copied down, added on the host in rank order, and copied back up
  (`TpBulkReduce`). The int8-or-FP16 choice for the experts is made per layer from all ranks' rows, so
  it does not depend on how the experts are divided.
- **Threads** (`TpCpus` in `tp_pair.hpp`): the engine pins its host thread to one CPU and new threads
  inherit it. Under `--tp` the other ranks' threads and the reduce workers move to the CPUs that
  neither the host nor a pool worker uses. `STRATA_TP_CPUS=0` turns this off.
- **Graph recorder** (`sycl/include/dpct/graph.hpp`): the process-wide maps take a lock. Each rank's
  thread records its own decode windows.

## Verification

All on 4× Arc Pro B60, `icpx` 2026.1, AOT `bmg`, Level Zero, xe driver. Unsloth UD-IQ4_XS, 32K
context, greedy, `--spec 4` with MTP. Replicated dense with the expert split unless said otherwise.

- **Same tokens as the single card.** 259-token prompt, 128 generated tokens, compared by hash with a
  single-card run: `--tp 2` 12 of 12 runs, `--tp 3` 4 of 4, `--tp 4` 8 of 8 (and 20 of 20 on the
  build before the last change).
- **Longer prompts: the tokens part ways after a few, and so do the single card's own two paths.**
  819-token natural prompt (the short prompt plus 560 tokens the single card generated): read back
  through the single card's prompt path, the continuation differs from what the single card generated
  in the first place at token 5. `--tp 2` and `--tp 4` differ from it at token 7 and agree with each
  other over the first 14 tokens; `--tp 3` differs at token 1. 2,072 tokens (the short prompt repeated 8 times): `--tp 3` and
  `--tp 4` give the single card's first 9 tokens, `--tp 2` a different first token. 10,360 tokens (two
  chunks): `--tp 4` gives the single card's first 8 tokens.
- **How far the first token's logits are from the single card's** (KL in nats, then the rms logit
  difference over the single card's top 50 tokens; `STRATA_LOGITS_DUMP`):

  | Prompt | Single card, FP16 prompt experts (`STRATA_PREFILL_MMQ=0`) | `--tp 2` | `--tp 3` | `--tp 4` |
  |---|---:|---:|---:|---:|
  | 259 tokens | 0.021, 0.33 | 0.076, 0.65 | 0.548, 1.41 | 0.058, 0.49 |
  | 819 tokens, natural | 0.010, 0.20 | 0.008, 0.29 | 0.030, 0.34 | 0.013, 0.22 |
  | 2,072 tokens, repeated | 0.504, 1.04 | 0.392, 0.96 | 0.449, 0.99 | 0.292, 0.80 |

  The first column is the single card against itself with its other expert arithmetic: the size of
  difference the engine already has between two of its own paths. A second single-card run is
  bit-identical. The top token is the single card's in every cell but two: `--tp 3` on the 819-token
  prompt (the single card's margin there is 0.38 logits) and `--tp 2` on the repeated prompt.
- **Where the difference comes from.** With rank 0 computing every prompt expert the result is
  bit-identical to rank 0's ordinary prompt path with its state copied to the other ranks, and the KL
  to the single card is 0.0006 (819 tokens) and 0.005 (259 tokens): that remainder is the decode
  window. With rank 1 computing every expert the per-expert outputs at layer 0 are the same to 9
  digits and the layer's output differs from rank 0's by 3.9e-8 of its rms (the order of one
  addition). The input to the experts at layer 1 then differs by 2.1e-4 of its rms, and the KL is
  0.0067, the same size as the split's. So a change in the order of the additions alone moves the
  logits this much; dividing the experts cannot avoid such a change. Inside one run every rank's expert
  input and routing are bit-identical (hashes, layers 0 to 3). Carrying FP32 instead of bf16 through the
  decode reduce (`STRATA_TP_WIRE=f32`) does not bring the logits closer (KL 0.080, 0.166, 0.110 on the
  259-token prompt) and does not change the decode speed (44.2 tok/s).
- **Sharded with the expert split**, `--tp 4`: the tokens differ from the single card's from token 21
  on. They are the same from run to run (4 runs). The sharded modes add in a different order.
- **Two faults found and fixed on the way.** (1) 3 of 10 `--tp 4` runs died in the first decode
  windows once the ranks' threads ran on separate CPUs. The cause given is from reading the code (the
  unlocked maps in the graph recorder); six runs under gdb were clean, so there is no backtrace.
  After the lock: 20 of 20. (2) 2 of 4 `--tp 2` runs ended "rank 0 gave up waiting for its peer". With
  the longer bound: 12 of 12.

## Benchmarks

Prompt tok/s and decode tok/s, same setup as above. A range is the lowest and highest of the runs.

| Configuration | 259-token prompt | 2,072-token prompt | 10,360-token prompt | Decode |
|---|---:|---:|---:|---:|
| Single card | 128.9 | 496.6–530.5 | 903.1 | 12.0 |
| `--tp 2`, replicated dense, split | 225.4–227.7 | 767.9–773.6 | not run | 23.0–23.2 |
| `--tp 3`, replicated dense, split | 301.8–307.2 | 913.1 | not run | 34.0–34.1 |
| `--tp 4`, replicated dense, split | 329.0–352.2 | 958.4–1,036.4 | 1,180.8–1,183.2 | 44.2–44.6 |
| `--tp 4`, sharded, split, `STRATA_TP_RANK_CACHE_MIB=15500` | 136.7–138.5 | not run on this build | not run | 66.7–67.6 |
| `--tp 4`, sharded, replicated experts | 19.8 | not run | not run | 12.9 |

The last row is from an earlier build of this branch. The sharded rows read the prompt through the
decode windows; they have no batched prompt path. With the cap at 15,500 MiB every expert is in VRAM,
which is why that row decodes fastest.

What the thread change did, same build, `--tp 4`, 2,072 tokens: 610.7 tok/s with `STRATA_TP_CPUS=0`,
958.4 and 1,036.4 with the default.

Where the prompt time goes at `--tp 4` (2,072 tokens, 2,136 ms on rank 0): the cross-card sum 620 ms,
the dense layers about 950 ms, the experts about 555 ms. The copy from a card to system RAM measured
5.0–5.1 GB/s on the compute engine, the copy engine and two queues alike; the copy back 13.9 GB/s.

## Limitations

- Replicated dense spends VRAM on a dense model per card and does the dense work on every card. Only
  the expert work and the expert copies are divided. On an 8,192-token chunk the dense layers are
  about half the time, so the gain over one card is smaller there (1.3× on 10,360 tokens).
- With the default cap, decode reads the coldest experts from system RAM. A higher cap decodes faster
  and leaves less VRAM. `STRATA_TP_RANK_CACHE_MIB=15500` with the parallel prompt path left about
  56 MiB free on rank 0 and a run failed; it is not a usable setting in that mode.
- The output is not bit-identical to the single card's. On a short prompt the generated tokens were the
  same in every run; on longer prompts they differ after a few tokens, by about as much as the single
  card's own two expert paths differ (see Verification). Quality has not been measured on a benchmark.
- The sharded modes do not match the single card token for token and have no batched prompt path.
- TP needs `--serve`. Vision and KV streaming are not wired for a TP group. The adaptive tier and cache
  admission are off under TP.
- Not tested: any card other than the Arc Pro B60, a second request in the same session, prompts
  past 10,360 tokens.

## Dedication

For Billy Preston — "Nothing from Nothing".
