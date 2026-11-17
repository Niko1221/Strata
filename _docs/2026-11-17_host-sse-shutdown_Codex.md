# Host SSE completion during ordinary shutdown

## Scope and requirements

Base: `77887a6dde1f9564c4ae82541d8a6485a724a3b7`, primary checkout on `main`.
An ordinary shell-requested shutdown previously forced every active HTTP socket
closed while `/api/events` still had an unfinished chunked response. The candidate
GUI diagnostic associated this request with `ERR_INCOMPLETE_CHUNKED_ENCODING`.
This change addresses the production HTTP shutdown path only. Strict candidate
console-error acceptance, resource floors and release qualification stay unchanged.

REQ-SSE-1: stop accepting connections, finish existing SSE response framing, then
perform forced connection cleanup. REQ-SSE-2: make hub shutdown idempotent, clean
its heartbeats and listeners, and complete within one second for a stalled reader.
REQ-SSE-3: terminate reconnects with an empty 204 response after hub closure.

The applicable controls are the canonical pre-implementation prompt, home and
Codex AGENTS, project AGENTS, `milspec-codex-standard`, and the Common Engineering,
Implementation Start Gate, Skill Selection, Security, WebApp Development,
Application Development, Internet Connectivity and TypeScript SOPs. This is a
traceable bounded repair, with no claim of formal MILSPEC certification.

## Implementation and decisions

`apps/host/server/util.mjs`: EventHub owns a cleanup callback for each response.
Disconnects, response errors and failed writes clear that response's heartbeat and
hub listeners. `close()` revokes new subscriptions immediately, ends each remaining
response, waits for finish/close/error or a shared one-second deadline, and removes
its drain listeners. Repeated calls return the same promise. A 204 response closes
late subscriptions without starting a new heartbeat.

`apps/host/server/index.mjs`: shutdown calls `server.close()` before awaiting hub
completion, then retains `server.closeAllConnections()` for remaining connections.
The existing whole-process shutdown deadlines and subsystem shutdowns remain.
One second is an upper bound for this drain within the existing ten-second
source-host and forty-five-second Control shutdown deadlines.

`apps/host/tests/event-hub-shutdown.test.mjs`: an actual isolated child Host receives
an authenticated shell shutdown while its real loopback SSE response is consumed.
It must yield `end`, `complete=true`, `aborted=false`, no response error, and normal
Host exit zero. Another real loopback server covers a disconnected subscriber,
ordinary event delivery, repeated close, late send and empty 204 reconnects. Native
ServerResponse listeners are captured before hub registration so the test checks
that no hub-owned listener remains. A deliberately stalled response with a mock
clock verifies the bounded deadline, heartbeat cleanup and listener cleanup.

All test traffic uses loopback HTTP and synthetic fixture credentials. The child
Host uses a disposable home, settings, model directory and workspace, disables
acquisition, and requests no model load. No dependency, native engine, GPU,
acceptance-controller or QA-helper source was changed.

## Commands and evidence

The executable used for the guard was
`C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe`.
The initial bare `python` command resolved to an unavailable Windows alias and did
not start the guard or test. Subsequent commands used the explicit executable.

Before production edits, the guarded command was
`_docs/upstream-sync-20261116/check.py --seconds 90 sse-shutdown-red-r2 node --test --test-concurrency=1 --test-name-pattern='normal shell shutdown' apps/host/tests/event-hub-shutdown.test.mjs`.
It failed with `ended=false`, `complete=false`, `aborted=true`, and `ECONNRESET`.
The assertion directly observes the interrupted HTTP message.

After the fix, the initial two-case run `sse-shutdown-green-r1` passed. The expanded
focused run `sse-shutdown-focused-r2` retained a test defect: it incorrectly required
zero total response finish listeners, including Node's pre-existing listener.
Production code was unchanged when the test was corrected to compare listener
identity with its pre-registration baseline. This failed evidence is preserved.

The final command was
`_docs/upstream-sync-20261116/check.py --seconds 120 sse-shutdown-focused-r3 node _docs/upstream-sync-20261116/host-focused.mjs event-hub-shutdown event-stream-ownership host-lifecycle`.
It uses the existing private filesystem test guard and serial test concurrency.
Node 24.22.3 reported 10 passed, zero failed and one existing Windows-only skip
for sending POSIX SIGTERM/SIGHUP. The other lifecycle cases and all three new
shutdown cases passed. `git diff --check` passed.

All receipts below are under
`H:/Strata/evidence/upstream-sync-20261116/receipts/`.

| Evidence | SHA-256 |
| --- | --- |
| `sse-shutdown-red-r2.command.txt` | `974192d393ef36cea02c12aa1a34341405e429898f78186c790f8a9a3c52417a` |
| `sse-shutdown-red-r2.json` | `718c1c6f8953ad1ffaa4196ea41227c83d4a763c9ab3e1b6988484d1b2e6c070` |
| `sse-shutdown-focused-r2.command.txt` | `bf05f80adfee838abb80fd4ce921270cd27b29bd806ad9785e323c325ad0cc42` |
| `sse-shutdown-focused-r2.json` | `6a609078df6824de92a07af1ba28a8ddc9258fbe66004fb1b04a81673b049b5e` |
| `sse-shutdown-focused-r3.command.txt` | `27d19fa643ad5b6b7544dcc597f89e4676d93b75f9da26ff6d8f29464ee60fa0` |
| `sse-shutdown-focused-r3.json` | `30e57329a041cc33b0d556863cd85faff44efa5730fea6a676ba67db04db9b9a` |

The final owned Job finished with exit/native exit zero, termination confirmed and
zero active Job processes. No resource floor was breached. Its minimum sampled
available RAM was 6,963,163,136 bytes and commit headroom was 26,499,289,088 bytes.
No low-RAM exception or process diagnostic cap was used. Compiler-job settings
remain five and tests run as one group.

## Self-review and residual risk

Self-review-event: type=qa_defect scope=host/event-hub-shutdown category=qa-coverage evidence=sse-shutdown-focused-r2.command.txt

`agent-self-review-evolution` review found a test ownership mistake in r2 and an
existing release gate correctly rejecting a production shutdown defect. The
smallest prevention is the added real HTTP regression and explicit ownership
assertions. Governance and acceptance criteria need no modification. Ignoring the
console error, removing Node-owned listeners and weakening the QA error gate were
rejected. The failed receipts remain available for review.

The one-second limit permits forced termination of a genuinely stalled reader;
it cannot guarantee remote consumption of every byte. Ordinary loopback readers
now receive a complete HTTP response in the exercised Host path. A rebuilt
candidate's real WebView2 close-and-final-console checks remain NOT_RUN in this
implementation unit and belong to the parent release work. No build, installation,
publication, GPU, general model quality or full-window inference qualification is
claimed. Rollback is a scoped revert of this repair; retain all failed and passing
receipts. Next action is independent source review followed by the parent's
candidate rebuild and unchanged strict GUI acceptance.
