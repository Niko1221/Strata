# Qwen3.6 / RAM 32GB / VRAM 16GB baseline slot

This directory holds the workload contract and the machine binding for the
Qwen3.6 startup plan. It is not a measured speed result.

The compared run, when it is later taken, uses one Qwen3.6-35B-A3B UD-Q4_K_XL,
context 65536, MTP 2, parallel 1, FP32 KV, and the effective prefill the engine
prints. A shorter context, a lower KV precision, or another model is a different arm.

`bindings/host.json` records the PC and the source commit that were read while
this slot was created. `windows_gpu_baseline` stays `NOT_RUN` until a paired
load on that PC writes a result file under `results/`. The historical 28.6 s /
266.0 s / 306.6 s figures live in `_docs/q36-32gb16gb/evidence-summary.json`.
They are one past observation, not this slot's median.

Log parsing rules are checked by `tests/qwen36/test_startup_log_analysis.py`.
The original stderr is not stored here.

## PR00 phase observations

The existing `tools/qwen35/q36_bounded_run.py` accepts an optional `observation`
object in its trusted local spec. Ordinary runs keep the same launch flags and
resource floors. Observation runs enable `STRATA_Q36_PHASE_METRICS=1` only in the
owned child and assign a fresh startup generation. This does not select a model,
change the installed application or qualify this directory's 65536-token arm.

```json
{
  "schema_version": 1,
  "family": "qwen36",
  "source_sha": null,
  "binary_source_sha": null,
  "environment_manifest": null,
  "identity_files": {}
}
```

Put this object at `spec.observation`. Fill source declarations with full commit
IDs when known. Each supplied environment/identity file is an absolute `path`
plus its actual `sha256`. Identity file keys are `tokenizer`, `chat_template`,
`sampler`, `model_metadata` and `build_receipt`. The runner checks their bytes
with the same resource guard as model/pack hashing. Use the passive
`q36_collect_environment.py` collector for the environment file. Its unknown
CUDA/DXGI/PCIe fields remain unknown; it does not create GPU measurements.

Each private run directory gains `run_manifest.json`, `phase_metrics.jsonl` and
`report.md`, including preflight aborts. Existing raw streams, tokens, requested
settings, effective receipts and resource samples remain private. New files
refuse overwrite. Source declarations need the hashed build receipt for actual
binary provenance; a checkout HEAD is not a binary source receipt. An incremental
candidate with reused libraries can have no single binary source commit.
Preflight aborts retain expected spec digests under `declared_identities`;
these must not be presented as successful byte verification or model execution.

Native `Q36_PHASE` schema 1 records load, warmup and single-session prefill/decode
with startup generation and request sequence. Layer -1 is the phase total;
layers 0..63 are attributed counters and layer 64 is unattributed work. Do not
sum total rows with layer rows. Warmup is separate. Cancel cleanup is recorded
only when executed. Batch/interleaved scheduling, non-serve execution, snapshot
and process-exit memory reclamation do not get invented single-request records.

Reads count explicit expert-store file API requests and fully successful bytes,
including verification when it runs inside the load phase. They exclude mmap
faults and are not SSD traffic. Failed reads can have uncounted partial I/O;
their failure counters must remain visible. Read wall time includes reader-lock
wait and host copies. Expert H2D/D2D counters count submitted bytes; CPU compute
counts only pool execution, and route wait measures an existing host wait.
No additional CUDA synchronization is inserted. GPU duration, copy overlap,
DXGI budgets, shared memory, hard faults and native process memory peaks remain
unavailable unless separately observed. Owned-Job peak commit includes the
worker/observer as well as the engine; it is not native private commit.

Request rows use only committed `T` tokens. TTFT is separate from inter-token
interval p50/p95/p99, whose sample counts are retained. Post-first-token rate is
`(N-1)/(last-first)` and is unknown for fewer than two distinct timestamps. One
short request cannot establish request-level TTFT p95 or a long-context profile.
Engine start-to-ready excludes preflight hashing and worker startup.
`ready_timeout_s` starts with the owned worker/engine launch; the unchanged
`total_timeout_s` still bounds hashing, startup and the complete run together.

`STRATA_Q36_HOTSPOT_PROFILE` and `STRATA_Q36_VERIFY` enable diagnostic work by
presence, including values such as `0`. Such runs are marked `diagnostic_sync`
and rejected by `q36_compare_runs.py` as speed evidence. Missing instrumentation
also blocks comparison. The existing native input-consumption/state checks
remain required: a protocol PASS, emitted counter or compiled binary is not a
quality, performance, input-integrity or capacity qualification.
