# Community measurements: two RTX 4080 SUPER cards, pinned main and dual-GPU fork

Measured 2026-10-08. Related to #1352, not a reproduction or resolution of its original twofold slowdown. This is a configuration survey with automatic layouts, not an isolated kernel comparison.

## Hardware and software

- Two NVIDIA RTX 4080 SUPER devices reporting **32,760 MiB each**, compute capability 8.9. This is not the stock 16 GB-per-card configuration and does not match the reporter's 4070 Ti SUPER / 5060 Ti pair.
- Linux 6.17, about 125 GiB system RAM; 8 configured expert-pool workers plus the host thread. The engine selected AVX2 kernels.
- NVIDIA driver 610.43.03; binaries built with CUDA 13.0.88, GCC 13.3.0, Release, architecture 89.
- Physical device 0: PCI bus `17:00.0`; device 1: `65:00.0`. `nvidia-smi topo -m` reports a NODE path between them. Exact physical IDs and topology are preserved in the manifests; GPU UUIDs are redacted. Link generation/width, clocks, thermals, and CPU/disk contention were not continuously sampled.
- Main: `d5ea7133741e67743c0e886bb426c0ce8d69cf6c` (engine 0.1.40.3).
- Comparison: `Hardin22/Strata-DualGPU` at `03a5ec955ba152a5f499a18a0f2f2ac192309c28` (engine reports 0.1.38). Both tracked source trees were clean. These are pinned revisions, not a claim about the latest branches.

The fork needed `CMAKE_CXX_STANDARD_LIBRARIES=-L/usr/local/cuda/lib64/stubs -lcuda` at build time. No product source or runtime stub path was changed; runtime linking resolved the system `libcuda.so.1`. Binary hashes and build evidence are included.

## Model and fixed inputs

- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, IQ3_S. The two shards total 83,617,662,656 bytes and were size/SHA-256 verified before the run; see `model-provenance.json`.
- Native expert pack, not `--compat-bf16`; MTP tensors from `Qwen/Qwen3.8-Flash-Next` revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, converted to the Q2_0 runtime. Tensor and runtime hashes are in the provenance files.
- Both engines used the same tokenizer, frozen prompt token IDs, draft vocabulary and immutable initial expert profile. Profile saving was disabled.
- Prompts are synthetic repetitions of pinned main's `src/program/generate.cpp` followed by a source-code explanation request. They are not the reporter's prompts. `main-inputs.json` and `fork-inputs.json` preserve the exact IDs and tokenizer/source hashes; only their plan binding differs.

## Method

Four requested settings, all with `--layer-split auto`:

| Label | `--pipeline-windows` | `--adapt-async` |
| --- | ---: | ---: |
| serial | 0 | 0 |
| async | 0 | 1 |
| pipeline | 2 | 0 |
| pipeline-async | 2 | 1 |

Common settings: `--expert-cache auto`, `--prefill 8192`, `--spec 4`, `--spec-min-p 0.5`, `--max-context 65536`, `--kv int8`, `--kv-resident 32768`, `--pool-workers 8`, `--prompt-cache 0`, `--conversation-cache-mib 0`, 1024 MiB reserves on both stages, `--resident-experts`, and `--trim-stage-weights`. No `--vision`, `--remote-expert-opt`, batching, or peer-device helper. `STRATA_PIPELINE_ADAPT_ASYNC=1` was set on both engines; the fork also used `STRATA_PIPELINE_RESERVE=0` to disable its extra automatic pipeline reserves. These do not make all internal allocation policies identical.

Each engine start ran a 1,024-token/128-output warmup, then a 6,144-token and a 32,768-token prompt, each capped at 512 generated tokens with temperature 0. There were three repetitions per requested setting; the middle repetition reversed the setting order. Entire matrices ran sequentially: main 0,1; fork 0,1; main 1,0; fork 1,0. Engine/order execution order was not randomized. There were no simultaneous GPU benchmark jobs.

All 48 starts, 48 warmups and 96 measured requests completed. Every measured request returned 512 tokens with finish reason `length`; prompt reuse was zero. Timing records were checked against all 48 raw engine logs. No failed or unattempted request was counted as a success.

The tables show medians of three measured requests. Per-request rates are `prompt_tokens * 1000 / prompt_ms` and `generated * 1000 / decode_ms` from engine DONE metrics. Min/max, individual rates, wall times, draft acceptance, and output hashes are in `summary.json`. Warmups and model loading are excluded; these are not end-to-end cold-start or HTTP throughput measurements. Only a 1K warmup was used, not a separate warmup at each measured length; OS page-cache state was not reset.

## Observed layouts and active paths

Equal CLI flags did **not** produce equal layouts:

| Physical order | Engine / requested labels | Split after layer count | Resident experts |
| --- | --- | ---: | ---: |
| 0,1 or 1,0 | main, all four | 24 | 24,576 |
| 0,1 | fork, serial / async | 29 | 24,576 |
| 0,1 | fork, pipeline / pipeline-async | 30 | 24,065 |
| 1,0 | fork, serial / async | 22 | 24,367 |
| 1,0 | fork, pipeline / pipeline-async | 30 | 23,633 |

Main held all experts in VRAM and logged no asynchronous adaptive worker for either requested async label. Its source suppresses usage collection when all experts are resident (`src/program/generate.cpp`, `all_experts_resident`), and async worker creation requires a nonempty usage vector. Consequently those main rows **do not measure an active async tier**. The fork logged an asynchronous worker for every requested async row, but a worker's presence alone does not establish useful swaps or an isolated speed benefit.

All requested pipeline rows logged that two verifiers per stage were enabled. The actual split, cache residency, prompt-buffer borrowing and adaptive behavior must be read alongside the timings. Automatic placement is part of what was observed, not a controlled constant. This dataset does not separate scheduler, layout, cache, output-text and pipeline effects.

## Decode tokens/s

| Order | Requested label | Main 6K | Fork 6K | Main 32K | Fork 32K |
| --- | --- | ---: | ---: | ---: | ---: |
| 0,1 | serial | 136.5 | 125.3 | 127.7 | 118.6 |
| 0,1 | async | 136.3 | 125.3 | 128.0 | 118.5 |
| 0,1 | pipeline | 138.9 | 124.9 | 141.7 | 127.3 |
| 0,1 | pipeline-async | 139.2 | 131.2 | 141.8 | 121.7 |
| 1,0 | serial | 141.0 | 128.2 | 133.1 | 123.3 |
| 1,0 | async | 141.4 | 129.2 | 133.2 | 127.3 |
| 1,0 | pipeline | 145.5 | 123.6 | 148.9 | 118.3 |
| 1,0 | pipeline-async | 145.9 | 125.1 | 149.3 | 118.6 |

## Prompt tokens/s

| Order | Requested label | Main 6K | Fork 6K | Main 32K | Fork 32K |
| --- | --- | ---: | ---: | ---: | ---: |
| 0,1 | serial | 2082.5 | 2078.4 | 4060.9 | 4226.2 |
| 0,1 | async | 2100.9 | 2087.1 | 4056.5 | 4226.3 |
| 0,1 | pipeline | 2097.7 | 2082.5 | 4070.3 | 4132.7 |
| 0,1 | pipeline-async | 2096.4 | 2066.5 | 4082.9 | 4125.1 |
| 1,0 | serial | 1798.1 | 2094.6 | 3824.7 | 4259.8 |
| 1,0 | async | 1797.9 | 2095.9 | 3837.5 | 4242.1 |
| 1,0 | pipeline | 1796.4 | 1993.5 | 3827.3 | 3626.2 |
| 1,0 | pipeline-async | 1794.7 | 1983.3 | 3814.6 | 3615.4 |

## Output checks and limits

Each main setting/order/prompt cell produced identical token IDs across its three repetitions. Some fork pipeline cells produced two or three distinct sequences despite temperature 0. Only **12 of 48** main/fork request pairs matched token-for-token when paired by device order, requested setting, prompt length and repetition. Raw output IDs and hashes are retained; no semantic quality or answer-correctness evaluation was performed. Different routing and generated text can change the measured decode workload, so these tables are not identical-output speedup ratios.

This run does not establish or disprove the issue's 40-vs-80 tok/s observation: hardware/VRAM, source revision, context limit, effective layouts, prompts, vision and helper-cache settings differ. It also says nothing about the large-prefill-chunk failure in #1468: this run used 8,192-token prefill chunks, not the reported 19-20K chunks. No Windows or AMD result is claimed.

## Reproduction and artifacts

`run_experiments.py` is the runner used for the measurements; it defaults to dry-run. The `raw/` subdirectories contain every request (including warmups), output token IDs, manifests and engine logs. `summary.json` is a frozen analysis of that capture, not a live query or a new measurement.

Place model/pack/MTP/profile inputs at the paths in the config files, or edit only the config paths to your own locations. Point each `--root` at its pinned clean source checkout and each config `exe` at the corresponding binary. Use the saved input files rather than tokenizing the fork source. For example, from this directory:

```sh
python3 run_experiments.py --root /strata-work/issue-1352 \
  --plan main-plan.json --config main-config.json --inputs main-inputs.json \
  --devices 0,1
```

This prints the planned main matrix without starting a GPU engine. After arranging exclusive GPU access, add `--execute --output NEW_OUTPUT_DIRECTORY` under an external wall-time/process-group limit. Use `fork-plan.json`, `fork-config.json`, `fork-inputs.json` and the fork checkout for the fork, and repeat both engines with `--devices 1,0`. Do not run performance jobs concurrently. The runner rejects pre-existing GPU compute processes. Its disabled-path check alone is not sufficient to establish feature activation: audit the startup logs and effective layout as done above.

Runner-only CPU checks:
The plan under `fixtures/` exercises the runner's #1468 control-construction test only; it was not executed in this GPU benchmark. The copied test changes only its fixture paths.
```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_run_experiments
```

Public paths replace the private workspace prefix with `/strata-work`; UUIDs are redacted. Internal provenance hashes describe the original pre-redaction inputs/logs, so they are not checksums of the redacted copies. `artifact-hashes.json` hashes the published bytes. The main frozen plan's historical `comparison_source.status` predates the completed fork build; the fork build provenance and captured fork manifests are the authoritative completed-run evidence.
