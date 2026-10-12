# Reproduction and data notes

## Scope

This report measured an already-installed custom Strata stage-RPC build. The
upstream base commit, build flags, binary SHA-256 hashes and embedded compatibility
fingerprint identify it, but the uncommitted custom source is **not** distributed
in this results-only submission. Checking out the base commit will not add these
RPC options. No instructions here claim to reproduce the engine from upstream.

On an existing compatible deployment, the following reproduces the HTTP suite:

```sh
STRATA_REPO=/path/to/Strata \
STRATA_CONFIG=/path/to/strata-rpc.json \
STRATA_URL=http://127.0.0.1:8090 \
BENCH_OUT=/path/to/new-results \
bash reproduce.sh
```

Run on the coordinator to ensure the upstream harness's local hardware discovery
describes the coordinator, not the client computer. `--url` prevents the harness
from starting or stopping the server. It should be idle and receive no unrelated
traffic during the experiment. Use `STRATA_KEY` if the server requires an API key;
never publish the key. Keep a non-authenticated API on loopback.

The original benchmark used the tools in Strata commit
`7695859aa13afb248bdb4d36f9b2e95707db079d`; tool hashes and agent document excerpt
hashes are in [provenance.json](provenance.json). No benchmark tool file was modified
for the measured run. The agent excerpts are public project documents: first 1500
whitespace-separated words, joined by spaces, as `Suite.agent_files()` constructs
them. Unique tags deliberately change between runs; no deterministic execution
path or identical generated-token sequence is claimed.

## Both-node measurement wrapper

[measure_rpc.py](measure_rpc.py) is the wrapper used for this report, with private
host/path constants replaced by environment variables. Its benchmark and sampling
logic is unchanged. It runs on the prefix-worker host and attaches by SSH to the
coordinator; it does not start, stop or reconfigure inference services.

```sh
STRATA_REMOTE=user@coordinator \
STRATA_REPO=/path/to/Strata \
STRATA_API_URL=http://coordinator:8090 \
STRATA_REMOTE_API_URL=http://127.0.0.1:8090 \
STRATA_WORKER_CONFIG=/path/to/worker.json \
python3 measure_rpc.py --out /path/to/new-private-artifacts
```

It assumes the same Strata repository path on both hosts, service names
`strata-rpc-worker-3090.service` and `strata-rpc-coordinator-5090.service`, and the
custom both-node `/metrics` schema. For this report it ran every approximately two
seconds and obtained 92 readings. Use SSH credentials and TLS material provisioned
locally; none is included. The wrapper's artifacts are **private by default** and
must be reviewed/sanitized before sharing; the committed telemetry is a sanitized
export, not the wrapper's unfiltered output. Its SHA-256 before replacing private
constants is recorded in provenance.

## Model preparation

Use the four Unsloth `UD-Q4_K_XL` shards at revision
`38bb39ee97821de2c9009abb7e93950eec396e66`. The model and `--compat-bf16` native
pack workflow are described in [UNSLOTH_Q4.md](../../../docs/UNSLOTH_Q4.md);
the base-Qwen MTP workflow is described there too. Each stage keeps its own local
model files. The prefixes in [launch.json](launch.json) are redacted placeholders,
not executable paths. `--rpc-ca`, `--rpc-cert` and `--rpc-key` refer to local files,
not embedded credentials. Packing, model downloads and cold-start time are not
part of these benchmark results.

## Files and interpretation

- `results.json`: upstream suite schema 1. Every numeric request measurement and
  test result is unchanged from the original archive; only metadata strings were
  additionally redacted. `machine` is coordinator-only, and its `cpu_cores=32`
  field actually counted logical CPUs. Consult `hardware.json` for both nodes.
- `benchmark.log`: sanitized upstream stdout, including engine-reported timings
  exposed through the API. No standalone startup engine log was captured because
  the server was already running.
- `telemetry.jsonl`: 92 records, one JSON object per line, elapsed time from the
  first sample, both nodes' hardware/cache capacity, freshness, live phase and
  worker-process VRAM. Actual PIDs, private addresses and hostnames are omitted.
  Cumulative cache and request counters are changed to deltas since the start of
  sampling/benchmark respectively; their baselines do not contain private prompts.
  NIC byte counters remain interface counters, not RPC-only bytes.
- `telemetry-summary.json`: observed hardware peaks, worker-process VRAM range,
  native counter deltas, no-error status and unchanged service/config checks.
- `hardware.json`: both-node device/driver/RAM observations; the PCIe CSV is an
  **idle snapshot**, not a sustained-load link measurement. Physical CPU topology,
  backing storage throughput and GPU overclock offsets were not measured.
- `launch.json`: sanitized observed native argument lists and server settings,
  not turnkey stock-Strata configs.
- `provenance.json`: original engine/tool/model/artifact identities and explicit
  custom-source availability limits. GGUF/pack/projector hashes came from the
  pinned installation manifest; large files were not rehashed for submission.
  The four small MTP-pack files were hashed independently after the benchmark.

Memory uses MiB/GiB (powers of 1024). NIC speeds use MiB/s. `gpu_mem_used` and RAM
fields in telemetry use bytes; `worker_process_vram_mib` uses MiB. Rates in the
results use tokens/s; `prompt_ms`/`decode_ms` use milliseconds, client latency and
TTFT use seconds. The native aggregate includes 4 warm-up/calibration requests;
per-test summaries cover 36 measured requests only. Warm expert cache and reused
conversation prefixes are separate conditions. No results were omitted as failures.

## Separate repository-only quality follow-up

[quality/README.md](quality/README.md) documents the independently recorded
six-case recall test and three community-contributed TypeScript fixtures run
through fresh Codex sessions. It includes per-attempt results, unchanged
generated solutions, original acceptance-test logs, native request accounting,
both-node sampling and reproduction commands. Its medium-reasoning coding
measurements must not be mixed into this speed suite's reasoning-off summaries.
The original speed data files above remain unchanged.
