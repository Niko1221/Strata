# DFlash2 parallel projections — 2026-10-22

Opt-in implementation: `STRATA_DFLASH2_PARALLEL_PROJECTIONS=1`, set before engine startup. Q runs on the drafter's main stream while K and V each run on an additional stream. Gate runs on the main stream while up runs on the first branch. Three timing-disabled events fork after activation preparation and join before consumers or activation scratch reuse. Projections read the same prepared BF16/Q8 activation and write distinct buffers. No weight/head changes, extra weight copies or activation allocation. Streams/events are destroyed on release, including partial upload cleanup; release also synchronizes side streams after a failed execution. Plain DFlash and unset flags retain their original path. Reference, tensor dump, debug and section-event timing use serial execution.

## Whole decode measurement

RTX 4070 Ti SUPER 16 GiB (SM89), Ryzen 5900X, Linux, CUDA 13.4. Same rebuilt executable per arm. Official Q8_0 drafter, unchanged shared Q8_0 target head, NVFP4 Flash-Next target, BF16 embeddings, FP8 PLE, INT8 KV. Fixed 2900 target expert slots / 11 workers / context and resident 8192 / verifier capacity 8; prefill 512, short-read 64, prompt cache 0, PCIe share 0, adaptive swaps off, no verifier splitting. Block 7, confidence floor 0.7. Shared head interleaving enabled in both. Other new optimization flags unset in this comparison. One warmup then three measured repeats; engines ran sequentially without compilation, training or another model job. Decode excludes prefill/startup.

| Prompt | Tokens in | Serial median tok/s | Parallel median tok/s | Change |
|---|---:|---:|---:|---:|
| Code short |27|48.940|49.178|+0.49%|
| Code padded |2427|40.668|40.776|+0.27%|
| Cache repeated |2425|50.791|50.837|+0.09%|

Ranges overlap in every case. This does not establish an end-to-end speedup or the requested +32.23% long-prompt advantage over MTP. MTP was not remeasured here. All warmed/measured/control outputs match the common genuine target-only controls from the length diagnosis. Draft accepted/offered counts match serial: 70/85, 58/87 and 66/78 respectively. See summary.json and individual arm JSONs for every sample.

## Correctness and actual concurrency

Additional warmup plus one measured request per case, each with same-process and external target-only controls:

- Q8_0 with CUDA Graphs disabled.
- Official BF16 and Q4_0 backbones with CUDA Graphs.
- Q8_0 with parallel projections, fused convolution, register top-k and parallel lattice selector all enabled.

All three prompts pass in every arm. BF16/Q4 were correctness checks, not paired quantization speed comparisons. Q5 and mixed-quantization artifacts were not executed.

A separate 32-output short-code integration check compared serial and parallel execution for blocks 1/3/7 at floor 0, reference-mode entry/exit, sampled target-only fallback, and another greedy request. All seven responses match in token IDs and accepted/offered counters across the two processes. Nsight was enabled only on this separate correctness run; its timings are not performance evidence. Both processes reset request state between cases.

Nsight Systems 2026.3.2, CUDA node tracing, confirms actual captured graph execution, successful graph instantiation, and projection kernel overlap across the extra streams. Eight drafter graphs have two extra streams and 775 branch kernels total; all 775 overlap another projection kernel. Two reference-mode drafter graphs have only the main stream. These graph IDs and overlap examples are in trace-summary.json. The raw trace remains outside the checkout at `../Strata-data/dflash2-projection-streams-20261022/projection-trace.nsys-rep`, identified by its saved SHA256. `edge_cases.py` records profiler flags; `trace_analysis.py` reproduces analysis after `nsys export --type sqlite`.

Passed:

```
cmake --build build-dflash2 --target strata dflash2_parity dflash_load_test dflash_policy_test -j4
build-dflash2/dflash2_parity
STRATA_DFLASH2_REGISTER_TOPK=1 STRATA_DFLASH2_LATTICE_SELECTOR=1 build-dflash2/dflash2_parity
build-dflash2/dflash_policy_test
build-dflash2/dflash_load_test
cmake --build build --target strata -j4   # STRATA_ENABLE_DFLASH2=OFF
.venv/bin/python bench/results/2026-10-22-dflash2-projection-streams/run.py
.venv/bin/python bench/results/2026-10-22-dflash2-projection-streams/analyze.py
.venv/bin/python bench/results/2026-10-22-dflash2-projection-streams/edge_cases.py
.venv/bin/python bench/results/2026-10-22-dflash2-projection-streams/trace_analysis.py
git diff --check
```

No failures observed in these checks. Existing compiler warnings remain. HIP/SYCL/Metal/Windows, other model families, multi-GPU, concurrent requests and fault-injected resource-creation failures were not tested. No backend compatibility or universal performance claim follows. The existing PR remains a draft. Runners use repository-relative model files and the external evidence working directory. Published absolute workspace prefixes are replaced with `<WORKSPACE>`, without altering hashes or numeric results.
