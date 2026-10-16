# Original Qwen: one-hour multi-conversation priority soak

Measured on 2026-10-16 UTC. This extends the
[earlier original-model comparison](../2026-10-16-solo-priority-qwen-original/README.md)
with a corrected scheduler, sustained multi-client load and lifecycle checks.
The original model is the primary contribution evidence; SC117 results are supplementary.

## Hardware, source and settings

- Windows 11, RTX 3090 24 GiB, Ryzen 9 5900X, 128 GiB RAM. No other intentional
  GPU workload during the test. Driver, installed model filenames, sizes and
  provenance details are in the earlier original-model report.
- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, non-abliterated IQ3_S.
  These are quantized default-family weights, not official unquantized weights.
  The original download revision was not independently verified.
- Tested Python runtime: `b121239dd2247eb5eb346e251950c0bf190b4bfb`, based on
  upstream `296b59d2f73c8f04a805df29c2fa97655adace78` (v0.1.44.1).
  The post-soak failure-path fix is `69a6b86599ccc0ecd02eeeb35e49db400f37ff96`;
  final lifecycle checks and the final matched comparison use that revision.
  The full hour was not repeated after this exception-identity fix; its
  successful decoding/preemption policy and statistics formula did not change.
- Unmodified official Windows engine, SHA256
  `bd18453f65b4d03b15de6100b1a9226426266560779b51b9265de176ae8f34d4`.
- Single engine, no batch slots; 262,144-token window, int8 KV, 32 GiB conversation
  RAM cache, eight slots, 16 GiB free-RAM floor. Model/pack/MTP/expert-profile
  inputs and settings match the earlier report. No model tools were executed.

```text
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
--mtp <existing vocabulary> --kv int8 --kv-resident 32768 --vision
--vram-reserve-mib 1536 --max-context 262144 --ple-io direct --adapt-async 1
--conversation-cache-mib 32768 --conversation-cache-slots 8
--conversation-cache-min-free-mib 16384
```

The startup calibrated spec to 6, mtp_max 4 and lookup 3; 11 pool workers,
14,400 MiB / 7,417 expert slots and 47,962 MiB arena. Speed projection was off.
Greedy requests use temperature 0, seed 117 and reasoning effort `none`.
Vision is configured but this workload contains no images.

## Workload and measurement

Three independent synthetic archives begin near 64K, 96K and 128K tokens. Each
worker repeatedly requests a forced `record_patch_manifest` call with 160, 200
or 240 ordered paths, then supplies a synthetic tool result and checks its exact
receipt, session, revision and count. Every four turns it audits 20 distributed
facts, including joins, corrections, arithmetic, source IDs and unknown values.
Each worker also performs a final audit. Returned tools and generated code are
never executed.

A fourth client sends a small exact-JSON request at priority 1 about every 17
seconds plus completion time. Another client deliberately disconnects a large
tool request about every 240 seconds. Development requests use priority 0.
The three archives are warmed before the requested 3,600-second load period;
workers finish outstanding turns and final audits afterwards. The elapsed time
including drain is **3,641.518 seconds**. The load began at 21:51:41 UTC.

When a conversation exceeds 230K total tokens, the harness explicitly rolls it
back to its initial archive plus the latest receipt. Four such rollovers occurred.
This is synthetic history management, **not model-generated or Codex compaction**.
It exercises cold/partial cache hits without exceeding the configured window.

Client times use the PC loopback API and include queueing, pauses and restoration.
Forced tool streams can emit parser-created argument prefixes before model output;
their first stream delta is **not** reported as model TTFT. Short-text TTFT and
whole tool-call latency are kept separate. The JSONL files retain both the raw
first-delta measurement and this distinction.

## Sustained-load result

| Measurement | Result |
|---|---:|
| Request results, including three warm audits | 409 |
| Normal completions | 395 |
| Deliberate cancellations | 14 |
| Unexpected transport / validation failures | 0 / 0 |
| Completed manifest + result round trips | 94 |
| 20-field audits, including warm/final | 28, all exact |
| Short requests | 179 |
| Short TTFT median / p95 / maximum | 0.669 / 7.054 / 46.503 s |
| Largest input reported by API usage | 237,392 tokens |
| Scheduler pauses during timed load and drain | 301 |
| Tokens read on continuation | 301 |
| Previously processed prefix tokens not restored | 0 |
| Final waiting / paused / in-flight requests | 0 / 0 / 0 |
| Observed peak system RAM / GPU memory | 76.1 GiB / 23,144 MiB |
| Observed peak GPU temperature | 80 °C |

The 301 continuation tokens are boundary work; they do not mean that new turns
needed no prompt processing. Cancellation returned to the client in 2.995–3.010
seconds (median 2.999). Those disconnects are expected failures and are excluded
from successful completion counts. The separate lifecycle tests below establish
cancellation at specific queue/engine states.

An independent pass over the saved pilot and soak outputs used canonical JSON
comparison, preserving value types (e.g. `false` differs from `0`). All **415**
successful responses passed: 20 pilot responses and 395 soak responses. The pilot
also exercised four deliberate cancellations. The public harness now performs
that strict comparison directly; the running private harness used Python value
equality, so the stronger saved-output check is reported separately.

## Cancellation, native Responses and engine recovery

Before and after the hour, the harness disconnected requests while queued,
actively generating and parked. In each case the other request completed correctly,
a subsequent health request succeeded, and waiting/paused/in-flight counts returned
to zero. A native Responses tool stream was preempted and still produced exactly
one call containing all 450 ordered items; its tool-result receipt was then recalled.

The post-soak native-exit injection exposed a second defect: when launched as
`python serve/server.py`, the scheduler imported another `serve.server.EngineDied`
class. The HTTP handler used the `__main__` class and did not catch the parked
request's exception. Chat closed without a proper terminal error, while Responses
failed explicitly and the next request restarted the engine. The initial harness
only required a client-visible failure, so inspecting the server traceback was
necessary to find this defect. That first lifecycle pass was not accepted for release.

Commit `69a6b86` passes the launching server's exception class and statistics helper
into the scheduler. A regression loads a second server module to reproduce the class
identity split; it fails on the old implementation and passes on the correction.
No successful-request scheduling policy was changed.

On the corrected revision, all three cancellation positions and the Responses tool
round trip were repeated. Terminating only the isolated native child then caused the
parked Chat request and active Responses request to return **structured server errors
in 1.228 seconds**. Neither emitted a successful completion. The next fresh request
triggered the normal auto-load path and returned the expected JSON in **23.621 seconds**.
This is recovery for a new request, not transparent replay of the interrupted ones.

## Same-runtime flag-off/on comparison

Final runtime `69a6b86`; three pairs per arm, flag off then on, on fresh server
processes. Each arm warms the same ~128K archive. A long response audits all 20
facts then generates code, deliberately capped at 3,072 output tokens. A short
21-token JSON request arrives seven seconds later. The code is intentionally
truncated; these are not compile or game-development acceptance tests.

| Median of three pairs | Flag off | Flag on |
|---|---:|---:|
| Short text TTFT | 20.128 s | 0.561 s |
| Both requests completed | 27.362 s | 28.428 s |
| Completed output tokens / pair wall time | 113.041 tok/s | 108.803 tok/s |

Aggregate throughput changed **-3.75%** and pair completion time **+3.89%**.
The enabled-arm short waits were **0.968 / 0.552 / 0.561 seconds**.
All six long audits were exact 20/20 and all six short replies matched. This is
end-to-end aggregate throughput, not native decode speed. The three-pair sample
was not randomized/ABBA; thermal and run variance can affect a few percent.
The earlier `b121239` comparison is kept separately in the JSON (-0.32% throughput),
not pooled into the final result.

After the final enabled pair was saved, the Windows controller tried to decode the
UTF-8 result as GBK and failed in result loading. The model requests had all completed
and the service was restored. All six original saved pairs and all 15 final lifecycle
records were then independently revalidated using UTF-8 and strict JSON value types;
all passed. The controller's failure remains recorded in the public summary with this
cause. The local reader was corrected; no successful model measurements were discarded
or replaced to obtain a better result.

## Failure found before the final run

The earlier scheduler allowed STOP during a prefill that had restored only part
of its prompt. The native engine consumes the restored snapshot and may be unable
to park an unfinished suffix. A failed preflight lost **229,376 processed prefix
tokens** and a short request waited **84.18 seconds**. That result was not accepted
as a successful soak.

The fix requires a cache-hit request or continuation to reach its first generated
token before yielding. A real-pipe regression with a restored prefix and unfinished
suffix fails on the old implementation and passes after the fix. A wholly cold
first prefill may still yield once and incur reread work.

The conservative guard has a visible cost: at one history rollover the engine
restored only **16,384** tokens and read **116,036** in **49.552 seconds**. The short
request arriving during that protected read waited **46.503 seconds**. The final
run's prefix-miss counter stayed zero. This feature improves interruptibility of
cached decoding; it does not promise immediate service during every partial/cold
read. The maximum is retained rather than excluded from the latency distribution.

Two earlier harness/prompt issues are also retained: a warm audit with tools offered
needed an explicit instruction to answer directly, and a cancellation target answered
with a short Python loop before it could be cancelled. The latter run was stopped;
the final workload uses a forced large manifest and a pilot that actually exercises
cancellation. Neither earlier run is counted as the final one-hour result.

## Automated tests and scope

`python -m unittest discover -s serve -p 'test_*.py'` on Python 3.11.7:
**798 tests, 790 passed, eight skipped, 296.962 seconds**. Thirteen focused scheduler
tests exercise the actual pipe reader and streaming parser, including trailing
tokens after STOP, split UTF-8, tools, prefill, fairness, disconnects, engine death
and the script/module exception-identity split. The earlier revision had 797 tests
(789 passed, eight skipped, 296.016 seconds). Validation initially used the wrong local interpreter: one attempt lacked
`regex`, and a second used Python 3.13 with a Python 3.11 extension wheel. Both
failed on dependency imports. The final run pins Python 3.11.7 and verifies the
`regex` import before testing; the dependency lives in a separate test directory.

The option defaults off. Normalized AST checks preserve the previous dispatch and
validation plus the extracted original native protocol body, excluding only the
opt-in pause initialization/callback. The whole `server.py` file is not byte-identical
because of that extraction and guarded integration. No CUDA/HIP/SYCL file or native
binary changed; no native backend rebuild is claimed.

The earlier model/schema counterexample remains relevant: an explicit boolean
`false` came back as string `"False"` in both flag-off/on tool runs. Those older
cases are **not** schema-correctness passes. Current exact-oracle success is not
a claim that all tools work, that generated code compiles, or that the model is
hallucination-free. This is one hour on one Windows/CUDA machine, not overnight
or cross-backend reliability. GENI/images and checkpoint opt-outs are not preempted.

## Reproduce

Use a dedicated idle server with the model/settings above and `solo_priority: true`.
Export `STRATA_API_KEY` privately if authentication is enabled. These scripts do not
change server configurations or execute model-returned tools.

```sh
python tools/make_solo_soak_fixtures.py /tmp/solo-fixtures
python tools/check_solo_priority_lifecycle.py --base http://127.0.0.1:8080 --out /tmp/solo-before
python tools/soak_solo_priority.py --base http://127.0.0.1:8080 --seconds 3600 \
  --fixtures /tmp/solo-fixtures/fixture-64000-117.json /tmp/solo-fixtures/fixture-96000-291.json /tmp/solo-fixtures/fixture-128000-803.json \
  --out /tmp/solo-hour
python tools/check_solo_priority_lifecycle.py --base http://127.0.0.1:8080 --out /tmp/solo-after
```

The native-exit injection uses a separate Windows controller identifying only the
disposable server's child engine. The public lifecycle CLI does not kill processes.
To repeat that case on an isolated instance: start a long Chat request, interrupt
with a higher-priority long Responses tool request, wait for `paused >= 1`, terminate
that exact native child, and check that both clients fail explicitly. Send a fresh
request to exercise the server's normal auto-load path. This does not replay or
complete the interrupted requests.

Build the exact final-comparison input from the generated 128K archive:

```sh
python - <<'PYGEN'
import json
from pathlib import Path
root = Path('/tmp/solo-fixtures')
archive = json.loads((root / 'fixture-128000-803.json').read_text())
data = {key: archive[key] for key in ('messages', 'expected', 'task')}
data.update(archive=True, max_tokens=3072)
data['messages'][0]['content'] = 'Post-soak same-build comparison A. ' + data['messages'][0]['content']
(root / 'comparison.json').write_text(json.dumps(data))
PYGEN
# Restart the same build with solo_priority=false, then:
python tools/bench_solo_priority.py --base http://127.0.0.1:8080 --fixture /tmp/solo-fixtures/comparison.json --rounds 3 --out /tmp/solo-off.json
# Restart with solo_priority=true; change no other setting, then:
python tools/bench_solo_priority.py --base http://127.0.0.1:8080 --fixture /tmp/solo-fixtures/comparison.json --rounds 3 --out /tmp/solo-on.json
```

[results.json](results.json) contains the phase summaries, all six final control
pairs, the earlier control separately, correction history and restoration results.
The `*-requests.jsonl` files contain one whitelisted record per request: status,
whole-request latency, first streamed delta, text TTFT where applicable, usage and
output/tool hashes. Full model outputs remain in the private raw evidence; they are
checked by the public harness against its generated oracles. No private documents,
credentials, model weights or full model prompts are published.

[native-timings.txt](native-timings.txt) preserves selected initial timing examples
(including the slow partial-cache hit) and all final lifecycle/comparison native
timing lines. These examples are labelled and are not a statistical sample.
[test-output.txt](test-output.txt) records the final test totals and regression results.

Both controllers restored the previous daily SC117 service and verified that its
configuration files had not changed. The final restore uses the corrected scheduler
source. The isolated test service and temporary scheduled tasks were then removed.
These operational checks do not add SC117 requests to the original-model benchmark.
