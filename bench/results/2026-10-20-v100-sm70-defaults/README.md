# Optional Volta optimization stack: dual V100 PCIe 32 GB

Community diagnostic, 2026-10-20. This is **not** the standard Strata benchmark protocol.

## Hardware and fixed configuration

- Ryzen 9 9950X, 128 GB RAM; 2 × Tesla V100 PCIe 32 GB, each capped at 150 W.
- V100 PCIe links: primary x16, secondary x4; Linux.
- Measured release: v0.2.0-beta (`3d1fb48090e8f2cefc22d7b5845fb7f5b71f33fa`), official CUDA 12 engine (0.1.45.3).
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S, native GGUF expert pack, MTP4 (`--spec 4 --spec-min-p 0.5`).
- Both GPUs, automatic 24/24 layer split; `--expert-cache auto --prefill auto --max-context 32768 --kv int8`.
- Two parallel slots, only one active request in the speed benchmark.
- `--vram-reserve-later-mib 650` on **all** arms: GPU1 held all 12,288 profiled expert pairs and retained the zero-doorbell graph path. A 700 MiB reserve had left nine pairs out in earlier diagnostics. This PR does not change memory reserves or sampling defaults globally.
- Fixed request sampling: temperature 0, seed 42, presence/frequency penalty 0, repetition penalty 1, top_p 1, top_k 0, min_p 0, reasoning_effort none.

## Trial

Fresh server for each arm: baseline → six-switch stack → baseline. Same binary/model/config throughout; no GPU profiling instrumentation. One 16-token greeting warmup per arm; three repeats of each prompt. Japanese/code output cap 256, summary cap 64. The checked-in `measure.py` contains exact prompts and request settings, including the per-repeat prefix. These are short synthetic requests, not private user messages.

```sh
STRATA_SIDE_MIXER=2
STRATA_GR_UP75=1
STRATA_B6_MMVQ_ROWS=1
STRATA_GU_SWIGLU=1
STRATA_QSA_ROWS=1
STRATA_ATTN_DAILY=1
```

| Arm | Japanese decode tok/s | Code decode tok/s | 1,551-token summary decode tok/s |
|---|---:|---:|---:|
| Baseline before | 74.23 | 108.40 | 80.30 |
| Six-switch stack | 78.63 | 116.77 | 86.30 |
| Baseline after | 74.40 | 108.70 | 80.40 |
| Stack after long-context stress | 78.70 | 116.87 | 86.57 |

Against the pooled baseline means, the stack improves Japanese decode **5.8%** and code decode **7.6%**. All nine output texts match across all four arms (UTF-8 SHA-256 in each JSON). This checks decoded text equality, **not** token IDs/logits or a comprehensive bit-exact gate. Some switches may already be active through architecture-dependent defaults; no drop-one experiment was performed, so do not attribute gains to individual switches.

`engine_stats` is the engine log tuple: prompt tokens, reused tokens, read tokens, prefill ms, prefill tok/s, generated tokens, decode ms, decode tok/s. Speed table averages the last field. TTFT is client time to first nonempty content delta. Raw per-request values and output hashes are attached.

Run the measurement against an otherwise idle authenticated local server, restarting between arms and applying/removing the six environment variables on the **engine** process:

```sh
# STRATA_API_KEY must be supplied in the environment; it is never written to results.
BENCH_OUTPUT_DIR=./results python3 measure.py baseline 8080 /path/to/engine.log
```

## Long-context / concurrency check

With the stack on: two simultaneous requests, each 32,000 uncached prompt tokens plus 512 generated tokens; repeated twice. All four returned both start/end sentinels. Synthetic Japanese cafe records were used, with explicit instructions to place both sentinels first. All four ended at their output cap, 32,512 total tokens. This is a recall/liveness check, not factual-quality evaluation.

- Pair completion: approximately 44.28–44.31 s, essentially unchanged from the previous tuned baseline (~44.3 s). **No long-context total-latency gain claimed.**
- Peak core temperatures: 55/57°C; HBM: 57/60°C.
- Minimum observed free VRAM: 4,397/267 MiB; both GPUs limited to 150 W.
- No CUDA, OOM, illegal-memory-access or allocation errors recorded.
- Post-stress speed benchmark still passed all nine output-text comparisons.

## Scope and rollback

The proposed sm70 table enables the tested combined stack automatically, keeps explicit per-switch environment settings, and supports `STRATA_SM70_PERF=0`. Existing startup logic limits automatic tables to homogeneous visible compute capabilities.

Evidence is limited to dual **32 GB PCIe V100**, this model/quantization, MTP4, 32K context and two slots. Single GPU, 16 GB V100, SXM2, other models, mixed GPUs and prolonged endurance were not tested. GPU1 memory headroom is narrow; this does not validate extra slots/vision/co-resident workloads. Maintainers may want those additional hardware checks before broad rollout. The source change is based on current main; GPU measurements above were on the stated release, not a rebuild of current main.
