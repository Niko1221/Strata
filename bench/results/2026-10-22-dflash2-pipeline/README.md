# Rejected CPU verification implementations — 2026-10-22

The goal remains about +32% DFlash2 decode throughput over fair MTP on long prompts. These experiments did **not** achieve it. Their engine changes were removed; the active source remains at `649c8e06`. No default, weight or shared head was changed. After user steering, no further launch-parameter search was run. This report records implementation changes at fixed settings, including poor results.

## Implementations tested

Both prototypes keep the native NVFP4 dot products, FMA order and per-block Q8_0 quantization unchanged. They split gate/up work on complete 32-value quantization blocks. Workers quantize their own blocks while the intermediate is hot. Per-expert atomic dependencies replace the global gate/up barrier and serial intermediate quantization. Down work begins after that expert's blocks are ready. Existing buffers are reused; the new synchronization array is 6 KiB of host memory. No VRAM allocation is added.

- V1 assigns each down job to a fixed expert.
- V2 lets a down job claim an unclaimed slice of any ready expert, avoiding waiting behind an unfinished expert while another is ready.

The experimental switch was `STRATA_CPU_EXPERT_PIPELINE=1`. The source guard restricts it to NVFP4 gate/up and down, Q8_0 intermediate activations, no down Hadamard transform, one NUMA pool and automatic task count. Other formats/layouts retain the old implementation. Each tested source snapshot is in its version directory; neither prototype is active in the engine now.

## Fixed-setting performance comparison

RTX 4070 Ti SUPER 16 GiB / Ryzen 5900X / Linux / CUDA 13.4. Same immutable executable within each four-arm comparison. Official Q8_0 DFlash2, unchanged shared Q8_0 target head, NVFP4 Flash-Next, INT8 KV, BF16 embeddings, FP8 PLE. Context/resident 8192, 2900 actual target expert slots, 11 workers plus host, verifier capacity 8, no split; prefill 512, short-read 64, prompt cache 0, no adaptive swaps or PCIe sharing. DFlash block 7 / floor 0.7; MTP max-T 4 (three proposals) / floor 0.5. Both use shared-head interleaving. These settings were held fixed; only the implementation switch changes before/after.

One warmup then three measured repeats per prompt. Sequential processes, no compilation, training or CPU/GPU tests during throughput measurement. Values exclude prefill/startup. Every arm passes common genuine target-only controls, with matching preparation and capacity. Before/after token IDs and accepted/offered counters match exactly in each case.

| Version | Prompt | DFlash before | DFlash prototype | Change | MTP before | MTP prototype |
|---|---|---:|---:|---:|---:|---:|
|V1|Code short, 27 tokens|48.620|49.193|+1.18%|37.728|38.183|
|V1|Code padded, 2427 tokens|40.770|40.792|+0.06%|36.668|37.051|
|V1|Cache repeated, 2425 tokens|50.934|50.858|−0.15%|50.638|51.067|
|V2|Code short, 27 tokens|49.145|49.319|+0.35%|38.130|38.244|
|V2|Code padded, 2427 tokens|40.882|41.038|+0.38%|37.017|37.024|
|V2|Cache repeated, 2425 tokens|50.958|51.156|+0.39%|51.132|51.178|

Units: native decode tok/s, three-repeat medians. V2 throughput ranges overlap in every before/after task comparison. Its median CPU-expert time per emitted token falls by 0.82% on padded code and 0.58% on repeated cache. These small changes do not establish a useful speedup. With V2 on both sides, DFlash leads MTP by 10.84% on padded code and is 0.04% behind on repeated cache. This does not restore the requested long-prompt advantage. Both V1 and V2 were rejected rather than changing a default or presenting them as a demonstrated optimization.

See each `summary.json` for every individual sample, CPU wall counters and the before/after DFlash/MTP ratios. The four command JSONs record actual startup state and executable SHA256. The common controls and prompt provenance are in the neighboring length-diagnosis report.

## Correctness checks

CUDA build passed. `pool_tasks_test --nvfp4` generates synthetic native weights without downloads. The optional real-expert mode used the first 2,764,816-byte expert from the local NVFP4 pack. Each runs the original 360 bitwise partition checks, including 192 NUMA cases, plus 144 comparisons alternating old/new execution on the same pools (72 calls exercise the new path). Counts of experts include 1, 7 and 97, crossing the 96-expert buffer-batch limit. Token groups include 1/4/8; workers 1/3/11, host working on/off. V2 tests assert that the new-path timing counter advances only when the implementation is selected, so fallback cannot silently satisfy the test. The new-path checks force a non-NUMA pool; NUMA retains the original path.

Commands performed before performance runs:

```
cmake --build build-dflash2 --target strata pool_tasks_test dflash2_parity -j4
build-dflash2/pool_tasks_test --nvfp4
build-dflash2/pool_tasks_test 40 40 ../Strata-data/dflash2-acceptance-20261022/expert0.bin
build-dflash2/dflash2_parity
```

V2 successful test outputs are `synthetic-v2.log` and `real-v2.log`. The original plain-DFlash/DFlash2 parity output also passed. No functional failure was observed; performance was insufficient. HIP/SYCL/Metal/Windows, other GPUs/models, production concurrent requests and fault injection were not tested. No compatibility claim beyond these local checks is made.

## Reproduction and evidence

Apply `v1/tested-source.patch` or `v2/tested-source.patch` to parent `649c8e06172676ea40a1d7f7746baed21065b94c`, build the listed targets, then run `run.py` from the repository root with a fresh `--data-dir`. Provide the existing local model files recorded in the commands. The working directories used were `../Strata-data/dflash2-pipeline-20261022` and `../Strata-data/dflash2-pipeline-v2-20261022`. `analyze.py --data-dir ...` checks outputs/settings and rebuilds the summaries.

Published absolute workspace prefixes are replaced by `<WORKSPACE>`. The snapshots are rejected experiments, not a new release build. The neighboring paired-GU report records a separate kernel experiment. The earlier private K/V audit found that DFlash2 already uses a local physical ring; no additional memory saving was identified there.
