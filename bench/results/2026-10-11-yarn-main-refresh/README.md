# YaRN migration refresh on current main

Source tested: `24dc1d6ef5880049b6e7c9b0f4faeeff93afd9f1`, merged with upstream
`61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`. Engine SHA-256:
`f42da1f36de535eab02ad55bfe70dff698d52ce2dc9870865185a978aa94e2bb`.

RTX PRO 6000 Blackwell 96 GB, CUDA sm120, ISTA IQ3_XXS, FP16 KV, MTP width 4,
greedy decoding, 8192-token prefill chunks. This refresh retains the existing
supported migration path. Its memory admission uses the current host/cgroup
memory sampler. Quantized KV migration remains unsupported.

The CUDA engine built; four native tests and the slot API suite passed. The 4K
probe and the 262,144-token source migrated to a 1,048,832-token allocation
completed. All retrieval tasks recalled their three markers. The latter run
also processed full 524,288- and 1,048,576-token sequences after restoring the
migrated prefix, checking prefix reuse rather than substituting full prefill.
Migration at 256K took 25.0496 seconds; restoring the migrated session took
4.3119 seconds. These are individual observations with warm filesystem caches,
not a repeated performance comparison.

Each quality comparison uses 32 forced continuation positions. A is fresh
YaRN, B is migrated ordinary-RoPE history, C is fresh YaRN replay, and D is
ordinary RoPE with the larger allocation. Exact replay controls are useful
here, but migration changes the stored state and is approximate.

| Prompt length | Comparison | Mean KL | Top-token agreement | Perplexity ratio |
|---|---|---:|---:|---:|
| 262,144 | A / C | 0 | 32/32 | 1 |
| 262,144 | A / B | 0.012821 | 32/32 | 1.004367 |
| 524,288 | A / C | 0 | 32/32 | 1 |
| 524,288 | A / B | 0.006129 | 31/32 | 1.017411 |
| 1,048,576 | A / C | 0 | 32/32 | 1 |
| 1,048,576 | A / B | 0.005081 | 31/32 | 0.987061 |
| 262,144 | ordinary / D | 0 | 32/32 | 1 |
| 262,144 | D / A | 0.017300 | 32/32 | 1.008865 |

These are narrow retrieval and forced-token measurements, with no automatic
quality verdict. A lower loss on this continuation does not establish broader
model quality. JSON files preserve all individual positions, generated text,
prefix hashes, timings and draft statistics.

The refreshed HIP engine built for gfx1012 and passed the four native tests.
The standalone SYCL engine built with oneAPI; its host migration test passed.
These checks establish build/host compatibility, not fresh model runtime
validation on either backend. Earlier HIP/XTX model measurements remain in
the separate [2026-10-09 report](../2026-10-09-mtp-migration/README.md).

Unattended machine time: CUDA 4K probe 128.788 seconds, 256K-to-1M campaign
1186.863 seconds, CUDA rebuild 25.618 seconds. Setup/engineering time was not
instrumented and is excluded. Build-status JSON records each operation.

Reproduce with the branch's `tools/bench_rope_migration.py`, a local model
configuration and an idle GPU:

```sh
python tools/bench_rope_migration.py --config model.json --output new-results \
  --compare --mtp /path/to/mtp --tokens 262144 --source-context 262400 \
  --target-context 1048832 --extend-tokens 524288 1048576
```

Model configurations must use private local paths. Public result argv replaces
those paths with placeholders; weights, production services and caches were not
modified by these probes.
