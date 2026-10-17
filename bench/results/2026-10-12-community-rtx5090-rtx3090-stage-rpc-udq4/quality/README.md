# Repository-only recall and coding checks

Measured starting **2026-10-12T21:03:22.072640+00:00**, completed **2026-10-12T21:23:01.491887+00:00**,
on the same running dual-node Unsloth UD-Q4_K_XL stage-RPC deployment as the
[speed report](../README.md). No engine restart, configuration change, cold-cache
reset, model switch, or custom gistfile1/fable task. The base build and unpublished
custom-source reproducibility limitation remain unchanged.

## Method and scope

- Recall: unchanged upstream `tools/needle_bench.py`, six cases at 32k/128k,
  depths 10/50/90, seed 7, greedy sampling, thinking disabled, 40-token allowance.
  The target size is character-estimated; report actual server token counts below.
- Coding: the repository's [RX 9060 XT community fixtures](../../2026-10-10-community-rx-9060-xt/fixtures),
  not an official general-purpose coding leaderboard. Three fresh Codex sessions
  per fixture (nine attempts), medium reasoning, server sampling defaults retained
  (temperature 1, top-p .95, top-k 20). The baseline source, visible tests and
  instructions were copied byte-for-byte; hashes are in [provenance.json](provenance.json).
- codex-cli 0.162.1, Node v22.23.2.
  Native `/v1/responses`, workspace-write, approvals never, ephemeral sessions,
  user config ignored; the existing runtime-discovery launcher was used. No custom
  llama.cpp proxy and no OpenCode client were used in this run.
- Fixtures permit source changes only, no installation/delegation/network tools,
  and at most two repair rounds. We independently reran the original acceptance
  suites on frozen solution copies, in read-only, network-disabled Bubblewrap
  mounts with a cleared environment and 60-second timeout. No evaluator repairs.
  Generated sources retain their exact bytes, including CRLF inherited from the
  Windows-origin fixtures; Git whitespace validation uses `cr-at-eol` accordingly.
- Each attempt has a 15-minute limit; collection has a four-hour limit. Failures
  are retained, not retried away. A successful agent exit is insufficient: the
  independent test count must match the fixture and tests/instructions must be unchanged.
- The expert cache was already warm and retained between attempts. Prefix reuse
  is measured rather than assumed zero. Sampling and GPU numerics do not promise
  identical output sequences; this is not a same-seed hardware A/B comparison.

## Recall results

**6/6 found**;
complete six-case grid: **True**;
request errors: **0**. Expected words, answers and per-case
data: [needles.json](needles.json); [console output](needles.log).

| Target | Needle depth | Actual prompt tokens | Answer | Found | Wall seconds |
| --- | ---: | ---: | --- | --- | ---: |
| 32k | 10% | 32804 | meadow-copper-504 | True | 22.3 |
| 32k | 50% | 32804 | falcon-quartz-940 | True | 22.4 |
| 32k | 90% | 32805 | willow-cobalt-696 | True | 22.5 |
| 128k | 10% | 124963 | falcon-saffron-138 | True | 84.0 |
| 128k | 50% | 124961 | quartz-tundra-528 | True | 83.9 |
| 128k | 90% | 124961 | quartz-glacier-192 | True | 83.7 |

This measures code-word recall at these positions only, not coding quality or
reliable behavior at the full 196608-token configured context.

## Coding results

**9/9 attempts passed**;
independent assertions: **243/243**. The three fixtures contain
**81 distinct acceptance cases**, repeated three times, not 243 distinct problems.
Tests are visible to the agent, not hidden tests or a broad coding capability score.

| Fixture | Attempt | Task passed | Independent tests | Agent wall seconds | Tool calls | Command errors |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| merge-windows | 1 | True | 25/25 | 82.6 | 7 | 0 |
| merge-windows | 2 | True | 25/25 | 84.5 | 6 | 0 |
| merge-windows | 3 | True | 25/25 | 76.4 | 6 | 0 |
| refactor-orders | 1 | True | 24/24 | 99.9 | 9 | 0 |
| refactor-orders | 2 | True | 24/24 | 85.3 | 7 | 0 |
| refactor-orders | 3 | True | 24/24 | 85.4 | 8 | 0 |
| reservation-workflow | 1 | True | 32/32 | 117.0 | 8 | 1 |
| reservation-workflow | 2 | True | 32/32 | 123.3 | 7 | 1 |
| reservation-workflow | 3 | True | 32/32 | 103.2 | 6 | 0 |

See [coding-results.json](coding-results.json) for completion state, timeouts,
changed files, scope violations, source hashes, Codex usage and per-attempt native
counter deltas. Each `coding/<fixture>-<attempt>/` directory contains the unchanged
generated solution, independent acceptance log, final message, sanitized stderr,
and completed tool events. Command errors include intentional failing tests during
repair; they are not automatically counted as backend failures. In this run the
two nonzero command exits were a `git status` call in a non-Git fixture and a
`grep -c` call finding zero failing tests. Neither was an acceptance-test failure
or a backend error. Total completed tool calls: **64**.

## Native accounting and memory

- Captured **77** completed native request records; full token/timing/draft
  measurements: [native-requests.json](native-requests.json).
- Whole collection weighted fresh-prompt throughput **1427.8 tok/s**,
  decode **81.2 tok/s**, MTP acceptance
  **72.02%**. These include recall
  and medium-reasoning Codex requests, not the original speed suite.
- Per-attempt weighted rates: [per-attempt-native-rates.json](per-attempt-native-rates.json).
  Rates use engine durations, not total agent wall time. `engine_generated` includes
  reasoning and tool serialization; accepted draft counters are not output tokens.
- Both-node metrics-only telemetry: **515** samples, **0**
  collector errors. Observed worker-process VRAM peak **18050 MiB (17.63 GiB)**.
  Whole-card measurements include the desktop; they are separate from worker-process memory.
- The initial controller's sampler incorrectly required optional `/api/requests`,
  disabled on this server (HTTP 404). A separate metrics-only collector was started
  without any engine change or test replay. The first hardware sample is
  **138.5 seconds** after collection began:
  early memory peaks are unmeasured. Completed request timings were recovered from
  the retained native history; these cover all test requests.
- Native monitor TTFT and separate reasoning-token counts are unavailable when the
  request-body monitor is disabled. Codex usage is retained where its events expose it;
  it is not substituted for engine token counts.
- Before/after configuration hashes, binary hashes, base revision, tracked-source
  diff, service PID and invocation identities remained unchanged:
  **True**. Engine remains running afterward.

Counters, peaks, integrity checks and collector incident: [summary.json](summary.json).
Sanitized readings: [telemetry.jsonl](telemetry.jsonl).

## Reproduction

These scripts require the already-installed compatible custom stage-RPC deployment
and its existing service/manager conventions; upstream base commit alone does not
provide network RPC. Both hosts must use the same repository path and native port
8090. Source and fixtures must match the recorded hashes.

```sh
python3 /path/to/Strata/tools/needle_bench.py \
  --url http://127.0.0.1:8090 --lengths 32k,128k --depths 10,50,90 --out needles.json

# Run controller on the worker host; remote endpoint is the coordinator.
chmod +x codex_launcher.sh
python3 run_quality.py --source /path/to/Strata \
  --launcher /path/to/quality/codex_launcher.sh \
  --remote user@coordinator --url http://coordinator:8090 --out /path/to/private-artifacts
# Optional independent collector, as used to recover sampling in the measured run:
python3 sample_quality.py --url http://coordinator:8090 --out /path/to/private-artifacts
```

The standalone needle command is an alternative to the controller's recall phase,
not an instruction to repeat measured cases. The controller's published defaults
and the launcher's fallback URL are portable replacements; measured hashes and
published hash are distinguished. The published controller also handles the
optional monitor's HTTP 404 by immediately collecting native metrics without it.
This is a collection-only fix, not a change to the measured inference requests;
the unmodified measured controller is retained in the private archive. The
companion metrics-only collector was used during the actual measurement.
`python3 -m unittest test_collector.py` checks this collection-only fallback
offline, without sending any inference requests.
Review private artifacts before publishing. No credentials, private addresses,
raw agent request bodies, personal history, models or engine patches are included.
