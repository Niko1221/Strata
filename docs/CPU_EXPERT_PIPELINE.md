# Pipeline CPU expert work without a batch-wide phase barrier

## Why this helps

CPU expert verification currently finishes Gate/Up for the whole batch, quantizes
intermediate activations, then starts Down. Experts that are ready early must
wait for the rest of the batch.

This opt-in change quantizes each expert as soon as its last Gate/Up tile finishes
and publishes that expert's Down work. All producer tiles are claimed before
dependent tiles, so dependent waits cannot strand unclaimed producers.
The existing row partitions and arithmetic kernels are retained.

## Use

Set `STRATA_POOL_PIPELINE=1` before constructing the expert pool.
It is off by default and applies to the native multi-token path.

## Validation environment

- Two modified RTX 3080 20 GB cards; CUDA SM86, CUDA 13.1, Linux.
- Intel Xeon E5-2686 v4 (18 cores / 36 threads), about 94 GiB usable RAM.
- PCIe 3.0, no GPU peer-to-peer access; 17 expert-pool workers.
- Qwen3.8-Flash-Next IQ3_S, 131,072-token context, INT8 KV,
  32,768 resident KV cells.
- Medium thinking: 2,048-token reasoning cap, 8,192-token output cap.
- Physical GPU power limits stayed at 250 W and 280 W. No power increase.
- Historical performance tests used an isolated engine based on official
  v0.1.38 plus reviewed architectds/Strata changes (best, 05c0f36).
  This PR contains only our change, ported to official main (99f3dbd).
  The historical timings are NOT a measured speedup of this pure-upstream port.

## Historical benefit

This is a small scheduling improvement, not a large tokens/s claim:

| With disjoint cache and helper 9500 | Pipeline off | Pipeline on |
| --- | ---: | ---: |
| 32K complete structured request | 11.128 s | 11.020 s |
| Natural-language generation | 76.5 tokens/s | 77.2 tokens/s |

The extra gain was about 1%. The natural-language outputs differed in length;
their wall-time difference must not be described as equal-work acceleration.
Both variants passed complete-answer and tool-call checks. Broader cache gains
are not attributed to this PR.

## Validation

The historical real-model test passed 810/810 bitwise comparisons across
three layer formats, 1/4/17 workers, host participation on/off, 1/2/4 tokens,
1/2/7/18/97 jobs, and three repeats. An independent serial row reference
also passed. This is not a ThreadSanitizer or general model-quality result.

The included Linux real-model test can be run as:

```sh
pool_pipeline_parity /path/to/native/pack /path/to/corresponding/model-shard.gguf
```

Without model arguments it exits 77 (CTest skip), not a false pass.
It reads the first three layers from the supplied shard; use the matching
Flash-Next pack and shard.

The independent pure-upstream port also passed the 810/810 real-model bitwise
comparisons on the stated Linux host. All known expert-pool header consumers
were recompiled, and the independent engine and parity test linked successfully.
The older upstream VNNI pool selftest was skipped because this CPU lacks the
required AVX-512 features; it was not counted as passing. No model server was
started or restarted, and no production settings were changed.

## Related work and remaining work

PR #500 parallelizes intermediate quantization as a separate phase. This proposal
instead pipelines per-expert readiness across phases. They touch the same code
and should be coordinated rather than merged blindly.
Keep draft until the independent upstream port completes end-to-end performance
and concurrency validation. No ThreadSanitizer run has been performed.
Other CPU architectures are unvalidated.
