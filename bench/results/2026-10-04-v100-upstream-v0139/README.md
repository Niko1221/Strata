# 2026-10-04: upstream v0.1.39 merge on fork main (2 x V100)

Curated raw evidence for
[`benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md`](../../benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md).
Every measurement is attributed to the engine binary that produced it by SHA-256. All
merge-tree arms record the fork main `git_commit` (`e448361`, `v0.1.36-58-ge448361`),
because they were built from an uncommitted working tree; the engine SHA-256 is therefore
the authoritative build identifier.

## Prefill arms (geni protocol, `bench/run_v100_prefill.py`, harness `ae03df48…`)

| File prefix | Arm | Engine SHA-256 | Median tps (2,048 / 8,192 / 32,768) |
| --- | --- | --- | --- |
| `result-fork-main` | Baseline (fork main, engine 0.1.36) | `b0cd9d4ad5ea…` | 701.41 / 1,402.82 / 2,262.4 |
| `result-upstream-v0139` | Initial candidate (pre-fix merge) | `2bffacd15ba4…` | 721.505 / 1,465.21 / 2,287.045 |
| `result-upstream-v0139-prefill-help` | Helper arm (initial binary + `STRATA_PREFILL_HELP=1`) | `2bffacd15ba4…` | 723.225 / 1,459.19 / 2,282.455 |
| `result-upstream-v0139-final` | Intermediate candidate (first post-merge fixes) | `5e42229c2d56…` | 717.155 / 1,462.03 / 2,287.495 |
| `result-upstream-v0139-release` | Release binary (batch-slot fallback; **supplementary**, pre-fix) | `9a82265a8df5…` | 720.78 / 1,462.065 / 2,289.96 |
| `result-upstream-v0139-completion` | **Final binary (primary measurement)** | `e9ad337a5977…` | 718.545 / 1,457.97 / 2,283.73 |

Each arm has a matching `summary-*.json`, `gpu-*.csv` (1 Hz telemetry), `protocol-*.jsonl`
(token-by-token serve-protocol records), and `engine-*.log` (sanitized). `comparison.csv`
holds the paired medians for baseline versus the final binary.

The release arm (`9a82265a…`) is the build immediately before the conversation-disk
restore-precedence fix. Its prefill numbers are real measurements of that build, not of
the final binary; the final arm's numbers supersede them.

## Serving / API evidence

- `http-baseline.json` — fork main engine (0.1.36), 524,288 context, client concurrency
  1 / 2 / 4.
- `strata-v0139-http-parallel-32k.json` — preceding build (predates `5e42229c…`), two
  serving slots at 32,768 context.
- `strata-v0139-api-extra-results.json` — Responses API, tool loop, JSON schema, and
  vision checks on the **final binary** `e9ad337a…` (regenerated for the final binary;
  supersedes the initial run against the preceding build).
- `strata-v0139-http-solo32-completion.json` — **final binary** `e9ad337a…`, one serving
  slot, 32,768 context, client concurrency 1 / 2 / 4, three repeats each.
- `strata-v0139-http-parallel32-completion.json` — **final binary** `e9ad337a…`, two
  serving slots, 32,768 context, same sweep (plus API warm-up checks: chat,
  `/v1/responses`, Anthropic-style text, invalid-tool rejection). The concurrency
  criterion is stated explicitly in the benchmark report (stream-interval overlap +
  marker independence + queue telemetry).
- `strata-v0139-concurrency-proof.json` — slimmed proof on the **final binary**
  `e9ad337a…`: four 256-token marked responses (CEDAR / MAPLE / BIRCH / WILLOW), emission
  intervals overlapping 6.1078 s (requests 0-1) and 6.1937 s (requests 2-3, chained by
  0.208 s), and monitor `live` snapshots with `.live.running == 2` (261 of 263) and
  `.live.waiting > 0` (138 of 263, third request queued). Per-snapshot `requests` arrays
  were omitted to slim the file; the file notes what was removed.

## Conversation-cache verification (`5e42229c…`)

- `l3-reuse-final/`, `l3-restart-final/`, `l3-corrupt-final/` — `results.json` (prompts,
  timings, output ids, state hashes, log counters) plus sanitized engine logs. Reuse and
  restart arm byte-exact; the corrupt arm verifies output parity only (the state was
  rebuilt from a cold read after the record was rejected).

## Fallback serving smoke (`5e42229c…`)

- `fallback-final/` — service status with two slots requested and one effective
  (`strata-v0139-fallback-final-status.json`), the matching solo-launch status
  (`strata-v0139-solo512-final-status.json`), the HTTP smoke rows
  (`strata-v0139-http-fallback-final.json`), and the API checks on that server
  (`strata-v0139-api-extra-fallback-final.json`).

## Regression evidence (conversation-disk restore precedence)

- `cache-precedence/before.log` — release binary `9a82265a…`, **failing**: conversation A
  restored the stale 25-token record although the live slot held 152 tokens (`prompt 172
  tokens = 25 reused + 147 read`), suite exit 2.
- `cache-precedence/completion.log` — final binary `e9ad337a…`, **PASS**: A2 reused
  152/152 (`152 reused + 20 read`), next solo reused 299/299 (`299 reused + 17 read`), all
  nine parity cases byte-identical, 126.10 s wall.
- `cache-precedence/parity.json` — the run configuration (3 slots, 32,768 context, 1 GiB
  disk store).

## 512K batch-fallback evidence (`9a82265a…`)

- `balanced512/before.log` — prior build, two slots carved at 524,288 context / split 24,
  OOM on the first request (`verify: instantiate: out of memory`).
- `balanced512/engine.log` — the unsafe session followed by the safe `9a82265a…` session
  (one slot effective: 11,894 resident experts, 5,933 primary, 496 MiB VRAM free; smoke
  reply `cobalt`).
- `balanced512/safe-status.json` — service status snapshot of the safe session (records
  `binary_sha256 9a82265a8df5…`).
- `balanced512/run.json` — the run configuration (split 24, `parallel 2` requested).

## Sanitization

Private paths are redacted: `$ENGINE` (local engine build), `$MODEL_STORE` (model store,
formerly `/mnt/strata-models`), `$WORKTREE` (benchmark worktree), `$HOME` (user home),
`$OUT` (the scratch directory the measurements ran from). No file contains host names,
user names, private IPs, or credentials.
