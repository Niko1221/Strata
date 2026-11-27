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

## Coverage and latency of frequently selected experts

Following the trace-guided caching idea in [MoE-Infinity](https://arxiv.org/abs/2401.14361), we counted expert selections across the three passes through the 12 recurrent physical layers on seven greedy tagging prompts. Each fold ranked experts per physical layer using the other six prompts. Across 55,296 selections, keeping the most frequent 16, 32 or 64 of 128 experts per layer covered 58.63%, 81.49% or 96.57% respectively. A simulated per-layer LRU starting empty on each request covered 62.28%, 76.71% or 88.47% at the same capacities. These counts measure selections, independent of expert size.

For a separate directory extraction replay, we ranked experts on all seven tagging prompts, then locked the 32 highest-ranked expert bodies per recurrent layer on the target's file mapping before `READY`. The 384 lock calls requested 1,145,102,336 bytes in total. The draft used the same wired limit as the withdrawal control above. All 20 outputs matched the reference token sequences. Median request latency was 7.04 s, P95 was 40.15 s, and the slowest request took 48.06 s. Aggregate decode rate was 31.14 tokens/s. The combined sampled peak footprint was 3.978 GB. During a 30.75 s wait to start an 83 ms command buffer, the target read 5.20 GB and the system swapped in 17.96 GB. High selection coverage on the calibration workload still left long submission stalls on this replay. Retain the default target mapping.

## Wire the complete native pack on a host with memory headroom

Export `STRATA_INK_MLOCK_PACK=1` in the Metal engine environment. The native loader calls `mlock` on the mapped pack before creating the Metal buffers. Startup reports the byte count on success or fails with the OS error. Unsetting the variable restores demand paging. The lock lasts until the mapping is released. KV buffers, the draft process and other applications have separate memory requirements.

| Target residency | Native P95 (s) | Max (s) | First-token P95 (s) | Decode tokens/s | Peak footprint (GB) | Peak RSS (GB) |
|---|---:|---:|---:|---:|---:|---:|
| First 4 GiB locked | 13.72 | 14.08 | 3.52 | 48.35 | 7.04 | 5.51 |
| Pageable control | 10.72 | 11.29 | 1.93 | 62.96 | 2.88 | 1.23 |
| Full pack locked | 6.78 | 7.00 | 1.82 | 65.98 | 13.26 | 11.83 |
| Full pack locked, fresh launch | 6.57 | 6.67 | 1.76 | 68.54 | 13.25 | 12.02 |

Each process launch replayed the same 20 single-slot requests with FP8 KV and the same wired-limit draft worker. All 80 sequences matched the reference. The target reported 4.36 s and 3.20 s to load the locked model before serving. The longest commit-to-GPU-start intervals were 318 ms and 160 ms with the full pack locked. Measurements include the first request after READY. Memory peaks sum the target, draft and embedding service at each sample. On the same 24 GiB host, a separate dual-slot HTTP run without drafting returned the expected JSON for seven of seven tagging cases, with a maximum complete response time of 1.77 s.

For hosts with less available memory, trace-guided expert caching and flash-friendly weight access address different resource constraints. [MoE-Infinity](https://arxiv.org/abs/2401.14361) uses expert activation traces for replacement and prefetch. [LLM in a flash](https://arxiv.org/abs/2312.11514) reduces reads through activation reuse and groups weights to improve flash access. Full-pack residency uses enough host memory to remove demand paging of the mapped target weights throughout serving.

### Decode time after wiring the pack

For each of the two full-pack runs above, 820 rounds offered 5,740 drafts and the target accepted 5,160. The 5,960 post-first tokens give 7.27 emitted tokens per round at width seven. Keeping round duration fixed, improving acceptance alone could raise that to at most eight, about 10% more throughput. The implied optimistic decode rates are 72.6 and 75.4 tokens/s. Wider verification blocks change the round cost and require their own measurement.

We summed target command-buffer GPU duration within each request's post-first interval. It accounted for 73.71 of 90.32 seconds and 71.34 of 86.95 seconds (81.6% and 82.0%). In the same intervals, worker time from receipt of the request header to candidate selection accounted for 16.1% and 16.0%. These measurements prioritize target verification cost for this workload.

A four-request screening run changed the eight-position Q8 row projection from two output rows per SIMD group to four on aligned shapes. All four reference sequences matched. Across 164 verification graphs, median GPU duration was 90.05 ms compared with 91.94 and 87.61 ms on the matching first four requests of the two control runs. This result supports retaining the existing two-row kernel.

## Compile-time unrolling for two-token down projections

In the verification blocks used above, one selected expert served one token in 771 cases, two in 328, three in 153 and four in 102. A test build specialized the two-token outer loop and the one-token remainder of the down projection into separate compile-time loops. Independent 20-request runs produced all correct token sequences, with the full pack locked in every process:

| Build | Median GPU per verification graph (ms) | Max GPU per graph (ms) | Native P95 (s) | Decode tokens/s |
|---|---:|---:|---:|---:|
| Existing control | 86.93 | 111.89 | 7.73 | 63.54 |
| Specialized loop | 85.11 | 106.02 | 9.57 | 59.70 |
| Specialized loop, fresh launch | 93.83 | 114.92 | 10.55 | 54.11 |

The earlier full-pack controls had medians of 87.96 and 85.71 ms. Combined footprint peaks were 13.26–13.34 GB. The result gives no basis for adopting the specialized loop. Retain the existing implementation.
