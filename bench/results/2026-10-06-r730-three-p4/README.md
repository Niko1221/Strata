# Three Tesla P4s on a Dell R730

Opt-in three-stage, two-window decode pipeline with Qwen3.8-Flash-Next Q8_0 weights. Two E5-2697 v3 CPUs, 251 GiB RAM, three 8 GB P4s in CUDA-visible x16/x8/x16 order. Layers 0-19 / 20-27 / 28-47; final GPU owns head and MTP. One host expert backing shared by all stages.

## Exact branch validation

Release build with GCC 12, CUDA 12.0, sm_61: passed. `commit_limit_test`: passed. 30 requests, 128 input tokens, 192-output cap, FP16 KV, 8,192-token allocation. Five modes: serial, pipeline, forced rollback, ungated pipeline, serial after pipeline; code, prose and counting; one warmup and one measured request per case/mode. Every output token and all three stages' committed GDN, PLE, previous-token, KV, active index-tail and completed pooled-row fingerprints matched serial. Five measured generated merge functions passed the functional checker. See `validation.json` for exact commit/binary identity and code-check details. Hashing is a correctness diagnostic, disabled for timings.

## Preceding R730 performance measurements

These measurements used the preceding complete R730 tuning builds, which also included shared-arena reuse and a late PLE-residency refresh excluded from this PR. They are not measurements of this narrower branch and do not isolate the benefit against stock upstream.

4,096 input tokens, 512-output cap, 8,192 allocated context, Q8_0 weights, FP16 KV, one request at a time. One warmup and three measured requests per case/version; medians:

| Patched version | Case | Prefill tok/s | Decode tok/s | Request wall s |
|---|---|---:|---:|---:|
| 0.1.39 | code | 147.32 | 26.12 | 32.61 |
| 0.1.39 | prose | 148.89 | 17.53 | 56.50 |
| 0.1.40 | code | 147.15 | 26.88 | 32.49 |
| 0.1.40 | prose | 148.66 | 18.12 | 55.61 |

Code finished at EOS (125 output tokens); prose reached the 512-token cap. Decode +2.9%/+3.4% across patched versions; prefill effectively unchanged. All measured requests had zero major faults, process read bytes and swap traffic. CPU pool 27 workers, 28 physical CPU cores, NUMA interleaved, stage weights trimmed, adaptive expert swaps enabled, `--spec 2`, `--pipeline-windows 2`, early prefix enabled, no prompt/conversation cache. Runtime held approximately 119.53 GiB expert backing plus 50.66 GiB PLE backing. CPU governors and GPU application clocks stayed at defaults.

## Recipe and limits

Use a Pascal-enabled build (`STRATA_EXPERIMENTAL_SM60=ON`, CUDA architecture 61). Server config: `"gpu": [0,1,2]`, `"layer_split": "20,28"`, `--trim-stage-weights`, `--kv fp16`, `--spec 2`, `--pipeline-windows 2`; set `STRATA_PIPELINE_PREFIX_EARLY=1`. Check the actual PCIe links before selecting device order. For short-prompt helper experiments: `STRATA_PREFILL_HELP=1`, `STRATA_PREFILL_HELP_DEVICES=2,0,0`, `STRATA_PREFILL_HELP_FRACS=0.45,0.65,0.45`; the earlier 1,024-input experiment also used `STRATA_PREFILL_STREAM_MIN=1023`. Helpers change floating-point grouping and are not expected to match helper-disabled output bitwise; helpers were disabled in the parity protocol.

No stock-versus-patch performance claim, roofline claim, Q4/IQ3_S validation, sustained concurrency result or general quality claim is made here. Exact-branch validation covers three GPUs; the retained two-GPU path has not been rerun on this branch. The preceding 4K measurements leave helper controls inactive because prefill uses multiple chunks.
