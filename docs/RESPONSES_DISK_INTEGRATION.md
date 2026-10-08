# Responses history and native disk-cache integration

This experimental branch combines `work/responses-core` with the original histories of
PRs #1269 (streaming session restore), #1271 (conversation spill), and #1480 (disk-only
conversation caching). The base is upstream Strata 0.1.40.3. Existing default behavior
is unchanged; Responses persistence, generated summaries, and disk-only caching are opt-in.

## Responsibilities

The Responses store durably records API input/output items and immutable response IDs.
`previous_response_id` reconstructs the selected parent history, including branches.
The native engine matches the rendered token prefix and reuses its inference state.
It does not need to understand response IDs. Missing, evicted or invalid native cache
files must result in ordinary prefill from the retained API history. A missing API
history record is a separate condition and remains a 404.

The native disk-only cache owns automatic conversation switching. Applications using it
should not also restore their own conversation snapshot before each generation. Explicit
slot save/restore remains available for deliberately managed checkpoints or reusable
base prefixes. All generation and explicit session actions use the existing serial queue.

A successful Responses reply guarantees durable API history when storage is enabled.
It does not guarantee that the newest KV checkpoint has been flushed to disk. Disk-only
switch snapshots are an evictable performance cache; clean engine shutdown flushes the
live conversation. API history and KV storage have independent retention and capacity.

Generated reasoning summaries use a separate generation on the same queue. They must
not replace primary reasoning in replay. Automatic switching should park the primary
conversation before the summary prompt runs. Turning summaries off avoids that generation.

## Downstream application contract

Profile selection, profile text and persona behavior belong in downstream applications.
Bind an existing conversation to an immutable profile revision and preserve that rendered
prefix on subsequent turns. Start a new conversation when intentionally changing its
profile. Keep application-owned base-profile checkpoints outside the engine's evictable
spill directory. Their compatible restore/rebuild policy remains application-owned.

The current Responses store checks model aliases, not weight content or a private profile
revision. Changing weights under one alias requires a new store/alias or downstream
version binding. This integration does not add a Codi-specific endpoint or expose private
profile sources. It is a public candidate foundation for downstream work, pending review.

Disk-only mode currently supports one GPU without batching, layer split or peer-device
execution. Cross-backend snapshot portability is not promised. This work tests each backend
with its own snapshots. CUDA and HIP output need not be bitwise identical.

## Merge and downstream plan

This is an integration candidate, not a request to merge overlapping patches twice.
The branch preserves the original commits behind #1269, #1271 and #1480, merged with
`work/responses-core` (`54aa4caa`) on upstream `d5ea7133`. The follow-up cache fix makes
streamed snapshots omit the draft layer when MTP is disabled; both capture and restore
must agree about that layer count. Reference overloads retain MTP callers' behavior.

If these changes are accepted separately upstream, base downstream work on the resulting
main and retain only changes that have not landed. If the combined candidate is accepted,
merge it once. Keep the generic API lifecycle and native snapshot code upstream; implement
profile selection and revision binding in the private application using these contracts.
Do not merge private profiles, modified model metadata or application identity rules here.

Recommended initial downstream settings: persistence on, generated summaries off, one
serialized model worker, and native disk-only switching enabled. Keep reusable base-profile
snapshots separate. A live continuation should continue its current state; it should not
restore the base profile again. Restoring another conversation should reuse its conversation
snapshot when available and otherwise reconstruct its immutable profile plus history.

Before calling this a finished private-server migration, test that application's profile
changes, three-profile switching and continuing chats on the accepted upstream revision.
This public validation does not migrate or start the private server.

## Validation

`tools/responses_disk_live.py --workdir /path/to/disposable-test` runs a real-engine probe.
The directory must contain `source/` (this checkout), `server.json` (local test server,
model, pack and all disk-cache options), and `runtime.json` (Python executable and env).
The endpoint is localhost:18210. The probe starts/stops the model, creates disposable
Responses history and session files, corrupts only its test spill files, and changes the
test disk-cache budget. Never point it at production history or a production spill directory.

Validated engine commit: `e6da7e11` (2026-10-08 UTC). Both Linux runs used one GPU,
8192 context, int8 KV, prefill 1024, mmap experts, a 2048 MiB disk-cache budget, summaries
explicitly toggled, and MTP disabled. The ggml dependency was pinned to
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Archive GGUFs were read without modification.

| Check | RTX 4090 / CUDA 13.3 | RX 7900 XTX / HIP |
|---|---:|---:|
| Model | ISTA Q2_0 | ISTA IQ3_S |
| Native cache test executables | 8 passed | 7 passed |
| API regression suite | 389 run; 4 optional skips | 389 run; 4 optional skips |
| Live assertions | 33 passed | 33 passed |
| A resumed after two other conversations | 1,332 cached tokens | 1,332 cached tokens |
| Next live continuation | 1,363 cached tokens | 1,363 cached tokens |
| Continuation after clean restart | 1,429 cached tokens | 1,429 cached tokens |
| Primary conversation after summary pass | 306 cached tokens; correct 323 answer | 310 cached tokens; correct 323 answer |
| Explicit save, 255,916,472-byte snapshot | 264.7 ms | 1,021.5 ms |
| Explicit streamed restore | 383.6 ms | 1,912.9 ms |
| Complete live sequence, including reloads | 318.7 s | 284.9 s |

The four optional skips were subsequently covered by 25 passing tests on a Windows
Python environment with jsonschema and the OpenAI SDK installed. CUDA additionally ran
the host transfer fault-injection executable, which is not a HIP target. Both backends
ran the actual GPU snapshot test. Native fixtures cover f16, int8 and q4 snapshots;
end-to-end serving here used int8 only.

The live sequence covers three interleaved conversations, live continuation, independent
branches, explicit save/restore, restart, ancestor deletion, changed instructions, summaries
off/on, disconnect cancellation, post-cancellation recovery, corrupt-file fallback and a
1 MiB eviction budget. All marker-recall fields were true, including after corrupted files
forced zero-token cache reuse. Both summary-enabled requests reached an output limit and
correctly returned `incomplete`; their children continued and answered 323. This is not a
model-quality benchmark or a claim that all summaries finish inside the chosen cap.

The first live attempt exposed `invalid K/V extent` when a non-MTP session tried to capture
an uninitialized draft ring. It caused automatic caching and explicit SAVE to fail. The
nullable-draft fix and regression fixtures above were added before these successful runs.
The published probe now also asserts the marker and post-summary answer fields that were
recorded and checked in these receipts, and refuses paths outside its disposable workdir.

[Sanitized results](validation/responses-disk-20261008/) record assertions, token usage and
request/snapshot timings. Raw logs and response objects remain in the private test workspace. These are single functional runs with different
quantizations and storage; the snapshot times are observations, not a CUDA-versus-HIP
performance comparison. No cold-page-cache latency guarantee is made. Windows GPU,
multi-GPU serving, maximum context, live MTP serving and private profile routing were not
validated here. Both temporary test services exited; production/private serving remains off.
