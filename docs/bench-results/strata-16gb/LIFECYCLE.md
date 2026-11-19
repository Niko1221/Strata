# Windows run observations

This capture uses `apps/host/server/lifecycle-observation.mjs`. Its versioned
snapshot is separate from the baseline shell in `run-observation.mjs` and
`measurement-schema.json`. The existing [baseline workflow](README.md) and saved
results keep their original contract.

RunObservation is an opt-in diagnostic, version 1. It changes no engine arguments,
model files, codec choices, context sizes or memory consent. Enable capture before
starting an isolated source host with `$env:STRATA_RUN_OBSERVATION='1'`. Normal
model load paths then capture startup observations. Read
`GET /api/engine/observation` on that host: `supervisor` is the latest supervised
run or null, and `native` contains up to twelve native runs. The existing status
snapshots are unchanged. Direct host callers can use `supervisor.start` with
`runObservation:true`, or native `operation('load', {model,runObservation:true})`.
This flag does not enable private memory or change private logging policy.

For rollback, remove `STRATA_RUN_OBSERVATION` and restart the source host. Do not
replace an installed app to collect these diagnostics. Capture remains disabled
unless explicitly requested for that launch. No automatic process termination is
performed by the observation code or benchmark runner.

## Attach to an already running local engine

Node 26.10 was used for the mocked CPU tests. This runner does not start or stop a
server. Supply the existing backend executable and model file as local inputs;
it reads them for identity, without moving, converting or modifying either.
The file SHA256 identifies the supplied file; attach mode cannot prove that the
running process is using that file. `identityBinding` records this limit.
The model metadata signature hashes the first 1 MiB, with the full file size. It
is a bounded header/content signature, not the full model SHA256 or parsed GGUF
metadata, and does not establish the identity of all split model shards.

```powershell
node bench/strata-16gb/run.mjs --enable-observation `
  --url http://127.0.0.1:8080 --backend 'H:\runtime\llama-server.exe' `
  --model 'H:\models\model.gguf' --backend-kind PrismML `
  --context 131072 --k q8_0 --v q8_0 --input-tokens 4096 `
  --output-tokens 256 --out 'H:\evidence\baseline-4k.json'
```

For the planned performance baseline, run two warmups and five measured invocations
per input case, each with a new output filename and `--output-tokens 256`. Identify
warmup filenames explicitly and exclude them from summaries. Repeat with
`--input-tokens 32768` for the 32K input case. The runner defaults to 32 output
tokens for smoke checks only; pass 256 explicitly for the performance baseline.
Both examples request capacity 131072; input length is a separate field. A
65536-capacity baseline must explicitly use `--context 65536` and remains a
different condition. The runner never lowers capacity or Q8/Q8 to make a case fit.
For an authenticated local server use `--api-key-env STRATA_BENCH_API_KEY` after
placing its key in that process environment. Do not pass a key on the command
line. Output creation refuses to overwrite an existing file.

The workload is generated synthetic text tokenized by `/tokenize`, then exactly
the requested token IDs are sent to `/completion` with prompt caching disabled.
`cache_prompt:false` requests no prompt caching; it alone does not prove the backend
reused no tokens. The reported evaluated prompt count must equal the target, or
the run fails. The optional `backendTokensCached` field retains the backend's
numeric `tokens_cached` with that exact source; it may count retained KV tokens
and is not interpreted as a cache-hit count. A backend that does not report
reuse separately cannot prove absence of reuse from this field. Record that limit.
The actual generated token count is recorded, including early EOS; 256 requested
tokens do not imply 256 generated tokens.
Only numeric timings and token counts are retained; response content and raw
logs are discarded. A missing tokenizer/completion endpoint records failure;
the native Python server does not currently offer this exact-token workload.
No fallback changes the workload or estimates a token count. Ctrl+C aborts the
HTTP request and writes a cancelled report when possible; it does not stop the
server. Backend handling of an aborted HTTP request remains backend dependent.

Attach mode accepts only the named numeric context capacity from llama `/props`
or native `/v1/status`. No observation route is required on an older installed
host. If those measurements are missing, effective fields stay null. A reported
capacity below the explicit request fails before sending the workload. Completion
without confirmed capacity remains unconfirmed and is not 131K qualification.
This runner does not establish model quality, statistical reliability, real
multi-user isolation or a safe memory capacity recommendation.

## Read the data

The schema is [run-observation.schema.json](run-observation.schema.json). Comparison
scope is `context_capacity_only`: comparable means the observed context meets the
request. It does not confirm K/V codecs, backend lineage or complete run identity.
Automatic backend selection is recorded as unknown unless a runtime is explicitly
verified; an executable basename does not identify a fork. Executable version is
captured only from a verified runtime manifest or a known numeric startup build
message; other formats remain null. No extra version process is launched.
Each
effective measurement has value, source, timestamp, status and a fixed reason.
Requested settings are captured once from launch arguments or native config;
they are never substituted for measured effective settings. Unsupported native
expert/prefill, policy revisions, K/V codecs and offload data remain null when
the backend does not report them. Packaged Control exposes no startup stream:
its observation uses the safe numeric context field in existing engine metrics;
it does not copy log tails, config contents or command lines. Source native
stream parsing currently recognizes only its numeric loaded-weight allocation.
Allocation sizes describe reported backend buffers, with explicit MiB/GiB and
bytes. They are not theoretical capacity estimates or peak total process use.

Read-only GPU queries use a 5-second timeout, 256-KiB output cap and hidden Windows
processes. NVIDIA dedicated total/free/used memory remains separate from Windows
`GPUAdapterMemory.SharedUsage`, whose adapter LUID is not assumed to match an
NVIDIA index. Empty or unsupported values remain null. WDDM NVIDIA compute-process
queries may omit graphics processes, so occupancy is incomplete. Only PID,
allowlisted executable basename and ownership category survive. Only exact PIDs
owned by the current supervisor are current-Strata; observed others are external
or unknown. Nothing is killed. No usernames, paths, command lines, prompts,
personas, memory contents or API keys are added to the observation.

Startup parsing stops at READY. Only anchored known numeric backend messages are
recognized; unrecognized lines are not retained. Failures retain a bounded outcome
category and numeric exit code. Existing engine logs have their existing separate
privacy behavior. GPU samples are point-in-time observations, not peak VRAM or
proof of absence of shared-memory use between samples.
