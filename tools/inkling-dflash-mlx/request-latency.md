# Profiling long decode stalls on Metal

On an M6 with 24 GB of unified memory, a single request produced its first token in 2.25 s and then needed another 126.65 s to finish. One command buffer waited 37.34 s between host commit and GPU start and executed in 0.084 s. During that wait, macOS swapped in 15.85 GB and swapped out 14.03 GB system-wide. The target process read 7.03 GB from disk and the draft process recorded zero disk reads. This localizes a major component of the stall to the interval before GPU execution under memory pressure. The process counters cover all files.

Four directory extraction prompts were replayed in fixed order five times per process launch. The timer starts at request send after `READY` and stops at native `BDONE`, including the first request. Each request uses one slot, 16K context, FP8 KV, seven proposals, greedy-head drafting, no prompt checkpoint reads and GPU keep-alive disabled. Times exclude HTTP and queueing. P95 is the second slowest of 20. Decode throughput divides total post-first tokens by summed post-first duration.

| Worker configuration | Median (s) | P95 (s) | Max (s) | Decode tokens/s | Exact reference sequences | Peak footprint (GB) |
|---|---:|---:|---:|---:|---:|---:|
| Default | 8.22 | 126.50 | 128.90 | 12.72 | 20/20 | 2.729 |
| MLX wired limit set | 6.27 | 8.22 | 13.63 | 62.75 | 20/20 | 2.819 |
| Same wired limit, fresh launch | 7.33 | 15.23 | 107.34 | 27.28 | 20/20 | 2.808 |

The experiment called `mx.set_wired_limit(1085614672)` after loading the draft weights and packed head, using their combined tensor byte size. The MLX allocator cache limit stayed at 32 MiB. The footprint column sums `ri_phys_footprint` for the target, draft and a resident embedding service at each 20 ms sample. File-backed pages make this metric unsuitable for estimating the entire resident working set.

Request 13 in the fresh launch returned its first token in 2.71 s and spent 104.63 s decoding. Two commit-to-start waits accounted for 40.46 s and 28.90 s. Memory pressure varied across the sequential runs. Keep the default worker configuration. For changes intended to fix this tail, compare repeated launches under similar pressure, count timed-out requests and report the worst complete request alongside P95 and throughput. Measure at the HTTP client separately for deployment decisions.

## Random-access advice on the target mapping

Two fresh launches added `madvise(base, fsize, MADV_RANDOM)` immediately after mapping the target pack and kept the draft wired limit above. A third launch restored the original target binary with the same worker. Each replay used the same 20 requests.

| Target mapping | P95 (s) | Max (s) | Decode tokens/s | Exact sequences |
|---|---:|---:|---:|---:|
| Random-access advice | 14.18 | 19.87 | 52.85 | 20/20 |
| Same advice, fresh launch | 7.81 | 14.43 | 60.63 | 20/20 |
| Advice removed | 8.45 | 12.06 | 61.67 | 20/20 |

With the advice enabled, a decode buffer still waited 9.89 s before 86 ms of GPU execution. The target read 2.27 GB and the host swapped in 4.36 GB over that wait. Removing the hint gave the shortest maximum and highest aggregate decode rate of these three runs. The withdrawal control gives no basis for enabling this hint by default.
