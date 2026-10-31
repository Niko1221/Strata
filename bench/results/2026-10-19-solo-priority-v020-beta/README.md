# Single-engine priority on 0.2.0-beta

Revalidation for issue #2821 and PR #2822. This is a Windows CUDA experiment,
not an approved standard benchmark or a claim about other backends.

## Revisions and environment

- Upstream Python source: `e88ee39d1d6be6e3364183af3b39b62a7ed6f211`.
  The `serve`, `src`, `include` and tokenizer trees match tag `v0.2.0-beta`
  (`b73dbce3edb8dac72a5642f46d1273197008fc4d`).
- Patched code tested: `4e510b36845019042e48e715f82b829485c9c69e`.
- Official beta release asset: `strata-engine-0.1.45.3-windows-x86_64-cuda-cuda13.tar.gz`.
  SHA256: `1b3845ecc3a46644ddc11e5296ab3d35026dbf650041c53657c43874682d1559`,
  checked against the release checksums. Its BUILD.json reports version 0.1.45.3,
  commit `f6d65cb584c0`; `strata.exe --version` prints 0.1.45.
  No native source was modified or rebuilt.
- Windows 11 build 26100, Python 3.13.1, Ryzen 9 5900X, RTX 3090 24 GiB,
  128 GiB RAM. The other GPU service was stopped for this experiment.
- Original, non-abliterated Qwen3.8 Flash Next, ISTA-DASLab GSQ-RCO IQ3_S;
  not SC117. Existing model files were reused read-only.
- Context 262144, KV int8, KV resident 32768; spec 4, spec-min-p 0.5,
  MTP rt, expert-cache auto, prefill auto, PLE direct, adapt-async 1,
  VRAM reserve 1536 MiB, vision enabled. Conversation cache 32768 MiB,
  8 slots, minimum free RAM 16384 MiB. The same copied expert profile was
  supplied to each arm. The engine's adaptive PCIe policy was left enabled.

## Method

Separate source directories, virtual environment, port 18082 and log files were
used. Production source and configurations were not replaced. Three sequential
arms use the same native binary, model, settings and workload: unmodified beta,
patched with the feature off, patched with the feature on.

Each arm first warms the same 127788-token synthetic archive and verifies its
20-field answer. Each of three timed pairs starts a long 3072-output-token
request, then submits a priority-1 short JSON request after 7 seconds. Long
outputs must retain the exact audit, and the short JSON must match exactly.
The incomplete code string is deliberately output-capped and is not claimed to
be executable or a programming benchmark. TTFT counts the first model content,
reasoning or tool argument, not an SSE header. Pair throughput is total output
tokens divided by wall time until both requests finish; it includes queue and
restoration overhead. The initial cold archive read is excluded from this warm
pair metric. Raw individual runs are retained.

## Warm pair results

| Arm (3 rounds each) | Short TTFT median | Pair wall median | Aggregate output median |
| --- | ---: | ---: | ---: |
| Unmodified beta | 20.784 s | 28.048 s | 110.28 tok/s |
| Patch, option off | 20.523 s | 27.792 s | 111.29 tok/s |
| Patch, option on | 0.610 s | 28.671 s | 107.88 tok/s |

All three unmodified-beta pairs reproduced waiting until the long response
completed. With the patch enabled, short TTFT was 0.918 / 0.594 / 0.610 s.
Relative to the patch-off control, median short TTFT fell by 97.0%; median
aggregate throughput fell by 3.1%. Three repetitions are a small performance
sample; this is not a statistical claim of a fixed overhead. The default-off
control stayed close to upstream throughput. All nine long audits and short JSON
answers passed. Long generated code bytes differ across arms; token-identical
native outputs are not claimed. No native engine code, tokenizer, model or cache
format was changed.

## Validation commands

```sh
python -m unittest discover -s serve -p 'test_*.py'
python tools/make_solo_soak_fixtures.py fixtures
python tools/bench_solo_priority.py --base http://127.0.0.1:18082 \
  --fixture fixtures/fixture-128000-803.json --rounds 3 --out arm.json
python tools/check_solo_priority_lifecycle.py --base http://127.0.0.1:18082 \
  --out lifecycle
python tools/soak_solo_priority.py --base http://127.0.0.1:18082 \
  --fixtures fixtures/fixture-64000-117.json fixtures/fixture-96000-291.json \
  fixtures/fixture-128000-803.json --seconds 3600 --out soak
```

`STRATA_API_KEY` is supplied privately. Use a different output path per run.
The server must already be running with the intended arm's configuration.

## Lifecycle checks before the soak

`check_solo_priority_lifecycle.py` completed 12 requests: 9 successful responses
and 3 intentional client cancellations. It exercised cancellation while queued,
while decoding and while parked, followed by a fresh request after each case.
The surviving tool call retained its exact ordered 450-item argument list.
A Responses tool stream was interrupted and resumed with one call identity,
450 correct items and the correct subsequent tool receipt. No validation
failures or unexpected transport errors were recorded in this phase.
See `lifecycle-before-summary.json`, `lifecycle-before-requests.jsonl` and
`lifecycle-before-responses-check.json`.

## Unit suite

The first port preflight failed 9 of 20 focused checks because a merge missed the
solo lock acquisition branch. This was fixed before the tested revision; the
same 20 checks then passed. The next focused run passed 15 checks including beta
queue deadlines and queued cancellation. The final admitted-state regression is
included in the complete Windows run below. `port-preflight-failed.txt`,
`port-preflight-fixed.txt` and `port-beta-lifecycle-tests.txt` retain these runs.

Windows at the tested code revision: **897 tests, OK, 20 skipped**, 474.820 s.
Mac (macOS 27.0.1, Python 3.13.0) at preceding `482a0c86`: 896 tests, 3 failures, 1 error, 15 skipped.
The same four Mac failures were reproduced with unmodified upstream source:
three `test_lifeline.Lifeline` child-process watcher tests and
`test_server.VisionArgs.test_min_tokens_is_passed_only_when_set` (mock process
missing `pid`). They remain failed, not silently excluded or fixed in this PR.
The later code change only corrects the admitted terminal state of opt-in solo
requests and adds its regression test; the complete Windows suite includes it.

## One-hour multi-conversation soak

Three independent 64K/96K/128K archives evolved through tool calls and receipts,
with priority-1 short requests every 17 seconds after the preceding one completed
and intentional cancellation probes every 240 seconds. Requests already in flight
were drained and final audits completed after the 3600-second load period;
elapsed time was 3707.709 s, excluding initial archive warming.

- **424 requests:** 410 completed, 14 deliberately cancelled, zero unexpected
  transport errors and zero validation failures.
- **100 tool round trips**, with exact ordered arguments and the newest receipt
  checked on each turn. Sessions completed 34 / 33 / 33 turns.
- **30 exact fact audits**, including 3 warm-up audits, 24 periodic audits and
  3 final audits. Missing facts remained null; revision selection, joins,
  arithmetic, booleans and source IDs matched the expected fixture values.
- **180 short requests:** TTFT p50 0.923 s, p95 6.545 s, maximum 46.663 s.
  The harness uses the upper middle observation for p50 and nearest-rank p95.
- **290 preemptions** during the measured soak and drain; 290 continuation
  tokens read and **zero paused-prefix misses**. Counters are deltas from the
  first telemetry snapshot to the final snapshot, not cumulative across the
  earlier benchmark/lifecycle phases. At the end no requests were parked/waiting.
- 422105 completed output tokens. Largest measured input was 237392 tokens.
  The harness performed five explicit history rollovers when a conversation
  exceeded 230K total tokens. It restores that session's original archive and
  latest receipt; this is **not** model-generated compaction.

See `soak-summary.json`, `soak-counters.json` and `soak-requests.jsonl`.
The soak establishes these synthetic invariants under the tested load; it is not
an independent no-feature throughput comparison or a general reasoning benchmark.

## Lifecycle checks and native fault after the soak

Queued/active/parked cancellations and the 450-item Responses round trip passed
again after the soak. A machine-specific extension of the same lifecycle harness
then started a low-priority long Chat stream and a priority-5 Responses tool
stream, waited until `/metrics` showed the former parked, and terminated only
the native `strata.exe` belonging to the isolated server. Its executable path
and server/venv-child ancestry were checked before `Stop-Process -Id <verified
native PID> -Force`. The actual selected PID is in `injected-engine-exit.json`.
No production process was selected by this fault injector.

Both streams explicitly failed within **1.355 s** of starting the injection
command. The next request automatically loaded the engine and returned the exact
recovery JSON in **24.690 s**, without restarting the HTTP server manually.
At the end the scheduler had no parked/waiting requests. The phase had 15 requests:
10 completions, 3 deliberate client cancellations and **2 expected injected-engine
failures**. The generic harness lists these two under `unexpected_transport_errors`
in its raw summary because they are not client cancellations; the fault-case
assertions specifically require those two errors and then successful recovery.
No other failure is being hidden by that classification.

Raw evidence: `lifecycle-after-summary.json`, `lifecycle-after-requests.jsonl`,
`lifecycle-after-failure-delivery.json`, `lifecycle-after-engine-recovery.json`
and `lifecycle-after-responses-check.json`. Host filesystem prefixes are redacted
in the public logs; the recovery extract omits unrelated hardware telemetry.
The measurements and synthetic response data are preserved.

## Observed long-tail limitation

During the soak, `quick-75` waited 46.663 s for its first token. An overlapping
session request had 132420 prompt tokens, of which only 16384 were reused.
The native log records 116036 newly read tokens in 47965 ms, followed by one
generated token. The scheduler then yielded. This is the documented conservative
boundary for a restored partial prefix: finish its prefill and reach decoding
before parking it. The scheduler's zero `resume_prefix_missed` counter covers
paused continuations; it does **not** mean that every new request or harness
history rollover hits a full cached prefix. See `slow-native-excerpt.txt` and the
individual soak requests. Short-request latency is not bounded to one quantum.

## Limits

Single GPU, one model/quantization and deterministic synthetic facts/tool data.
The generated code and synthetic tools are never executed. This does not
establish open-ended development quality or overnight reliability. No native
HIP, SYCL or Metal execution/build was performed; this Python feature changes
no native sources. Vision-bearing requests remain non-preemptible. Cold prefill,
cache eviction/restoration and queue aging can still delay short requests.
