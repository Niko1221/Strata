# DFlash2: parallel work investigation, 2026-10-22

RTX 4070 Ti SUPER (SM89, 16 GiB), Ryzen 5900X, 11 expert workers. Same immutable tested engine and official Q8_0 drafter as the length diagnosis. Original shared head and all weights unchanged.

Two prompts: code_long (2427 tokens), cache_long (2425 tokens). 96 generated tokens, one warmup and three measured repeats per case, median throughput. Context/resident 8192, INT8 KV, 2900 expert slots, prefill 512, short-read 64, prompt cache off, spec capacity 8, no spec-split, DFlash block 7 / floor 0.7. Existing DFlash2 kernel optimizations remain disabled. Measurements ran sequentially, without compilation or training. See saved command JSONs for complete settings and engine SHA.

| Verifier streams | Code tok/s | Change | Cache tok/s | Change |
|---|---:|---:|---:|---:|
| Default |40.886|—|50.853|—|
| STRATA_SIDE_MIXER=1 |41.031|+0.35%|51.118|+0.52%|
| STRATA_SIDE_MIXER=2 |40.966|+0.20%|50.955|+0.20%|
| STRATA_DF_BRANCH=1 |40.872|−0.03%|50.939|+0.17%|

Every arm passed both same-process target-only and external common target-only token controls. All measured outputs contain the same 96 token IDs per case across arms. Acceptance unchanged: code 58/87, cache 66/78. These small changes do not establish a useful speedup; no default was changed. MTP was not retested with these flags, so these percentages are against DFlash2's own baseline, not an increased advantage over MTP. No new engine code was introduced in this investigation.

## Dependency findings (static analysis)

- `src/core/dflash_runtime.cpp`: Q/K/V projections following `prepare_projection` read the same BF16/Q8 activation and write distinct q/k/v buffers. Gate/up projections likewise read one activation and write separate buffers. These are candidates for additional CUDA streams, with a fork after preparation and a join before their consumers. No prototype or speedup measurement for this drafter change has been made. Projection scratch must not be overwritten before every reader completes, including when mixing BF16 and quantized weights. Kernel profiling/parity paths also need correct synchronization.
- `src/core/verify.cpp`: SIDE_MIXER already branches GDN gates / QSA indexing; mode 2 adds query work. DF_BRANCH is another existing implementation, disabled for split windows, batched requests and verifier profiling; SIDE_MIXER takes precedence. This experiment used unsplit windows and did not enable the verifier profiler. Decode wall timing remains enabled.
- Shared GPU expert execution already uses a separate stream by default on CUDA. CPU experts already use a worker pool. More threads are not automatically additional bandwidth.
- `src/program/generate.cpp`: the next DFlash proposal needs verified tap rows and the accepted-prefix bonus token. Running the full next draft concurrently with current verification would require a different speculative algorithm. Context append already avoids a synchronization when immediately followed by a proposal on its stream.

The length diagnosis measured drafting at roughly 6–7% of long-request decode time. At unchanged acceptance and verification cost, eliminating all draft time would save only that fraction of time (roughly 7% throughput gain), so projection overlap alone cannot demonstrate the requested +32.23% advantage. This is a bound for those measured cases, not a general hardware claim. The experiments above do not identify a specific bandwidth bottleneck; concurrent kernels competing for bandwidth is a hypothesis.

Raw logs, JSONs, summary.json and the runner are beside this report. The common target-only controls and prompt provenance are in the neighboring `2026-10-22-dflash2-length-diagnosis` report. Only the installed NVFP4 target was measured; other setup model families were not tested.
