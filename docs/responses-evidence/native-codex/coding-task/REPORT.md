# Native Codex read/edit/verify checkpoint

**Pass, with the pinned local-tool profile below.** Real Windows Codex CLI
0.160.0 used Strata's native Coder IQ1_M model on llm-49 / RTX 4090 to repair a
Python settings parser. The client executed the shell commands and edited the
file. No model output or client tool result was mocked, no request fields were
rewritten, and no grammar was injected into the Codex requests.

The successful turn took **528.784 seconds after server readiness**: eight
Responses requests, nine `exec_command` calls and nine matching tool results.
Codex observed failing tests, repaired the source, ran the tests successfully,
reported the result, and emitted `turn.completed` with process exit 0.

Start with the [plain-text tool loop](TOOL_LOOP.txt), [result](result.json),
[wire audit](audit.json), [independent checks](independent-verifier.json), and
[native fault probes](native-edge-cases.json). [REPORT.txt](REPORT.txt) contains
the same report as plain text. The raw captures, including earlier failures,
are in the four ZIPs listed in [archives.json](archives.json).

## Exact checkpoint

| Component | Pin |
|---|---|
| Native-tested branch | `work/gbnf` |
| Worktree | `C:/Users/dflanag3/Documents/fleet/strata-native-gbnf` |
| Running Python source commit | `36e9035ec4fa789eb5ba1cec5988b8d4ffb889cf` |
| Native implementation commit | `f7cf318ee0d407292afea7782e648d7714069720` |
| Native binary SHA-256 | `656a6c16c2c148ec730c1b4b78f1e904417f2efb5cbce2455473156ce746a1ed` |
| Codex version | `codex-cli 0.160.0` |
| Windows Codex binary SHA-256 | `fdda5fa3cf3fb3d000b876720742857676293e4315e4b045fae6f8bd7e866d1d` |
| Exact coding probe SHA-256 | `0ac1e7fe9dcc52877d47762c8fef655ce9cfb98df51a5a469eec65b3fd921ec3` |
| Responses-only branch/worktree | `work/responses-api`, `C:/Users/dflanag3/Documents/fleet/strata-responses-stateless` |
| Mirrored Responses fixes | `82587463b81c9787f69b72b5f73e008a983d18f9` |

The [manifest](manifest.json) pins all deployed source hashes; the archive's
`source-verification.json` records their equality with the isolated remote
snapshot. The native binary was reused unchanged. No C++, MTP, scheduler or
GPU lifecycle code changed for this checkpoint. The standalone Responses
branch received the two parser fixes and its own Python regression run; the
native coding run used the combined branch, not that separate checkout.

The [environment](environment.json) records Linux Python 3.14.4, an RTX 4090
with driver 595.91.07, and Windows 11 / Python 3.13.15. The native configuration
used the existing Coder IQ1_M assets, context 32768, INT8 KV, MTP maximum four,
suffix lookup off, temperature zero and a 256-token reasoning budget. Exact
local asset paths and server arguments are in the archive's `config.json` and
`native-server.json`; these are installed paths, not download links.

## What the task checked

The [task](TASK.txt) starts with a broken `parse_settings(text)`, a specification,
12 unit tests and two fixture files. Only `settings_parser.py` may change.

- Split at the first `=`, trim keys/values, preserve empty values and inline `#`.
- Reject wrong input types, empty keys, missing `=` and duplicate keys; errors
  identify the original physical line number.
- Handle LF, CRLF, CR, blank lines, comments, case-sensitive keys and ordering.
- Preserve Unicode, combining characters, quotes, backslashes, literal
  `$(...)`, and `<tool_call>{"x":"a=b"}</tool_call>` as data.
- Read paths containing spaces and brackets with PowerShell `-LiteralPath`.
- Recover from a deliberately missing file without creating it.
- Leave tests, specification, fixtures and a BOM/CRLF/trailing-space/NUL sentinel
  byte-identical. Finish the repaired source as UTF-8 without a BOM.

The model also made an unplanned bad `Get-Content -Encoding Text` call. Its
actual nonzero result was returned to the model and it recovered. It initially
wrote a BOM through PowerShell 5.1, detected it and removed it with Python.
These errors and repairs remain visible in the transcript. The model did not
follow every efficiency hint: it surveyed the directory despite being given
the filenames. That is retained too.

The client saw `FAILED (failures=4, errors=7)` before editing, then all 12 tests
passed. An independent rerun also passed all 12, and **45 additional independent
checks passed**. Hashes confirm only the allowed source changed; Python cache
files are excluded from that comparison because verification imports the module.

## Protocol and native recovery

All **3,728 SSE events across eight completed responses** passed the wire audit:
contiguous sequence numbers, stable item IDs/indexes, distinct item IDs and
call IDs, exactly one terminal event, and delta concatenation equal to final
text/arguments/reasoning/summaries. Stream finals equal the monitor snapshots;
captured request bodies equal those actually received by the service.

The model produced eight reasoning items, seven message items and nine function
calls. All nine calls have matching client-owned `function_call_output` items.
The client replayed Strata-issued encrypted reasoning items 28 times across
successive histories. The recorder never decrypted these blobs: the readable
transcript gives their purpose, length and hash; the raw ciphertext is retained.
The server uses its normal authenticated replay path.

The client's eight tool entries, including its `multi_agent_v1` namespace, were
accepted unchanged. Only the flat `exec_command` function was actually called.
This run does not establish native namespaced dispatch, image tools, delegation,
or every declared client function.

The same native server then passed **19 companion HTTP cases**:

- Fourteen boundary rejections: absent authentication, duplicate JSON keys,
  non-finite JSON, background, retained storage, previous-response lookup,
  unknown model, malformed tools, strict functions, Lark custom tools,
  JSON-object output, orphan tool result, foreign replay token and context
  overflow. Responses were JSON errors; valid bodies requested streaming.
- One-token output budget yielded `incomplete / max_output_tokens`.
- Disconnect during actual prefill: 256 of 9,017 prompt tokens had been read;
  the service drained to idle.
- Disconnect an active decode and a second queued request; both were removed
  and the service drained to idle. The queued request emitted `response.created`
  before it could emit `response.in_progress`.
- Ordinary generation after draining returned `READY`.
- A separate request for `purple 999` with `grammar: root ::= "READY"` returned
  exactly `READY`. This manual GBNF probe was not part of the Codex conversation.

The successful archive includes the exact `edge-probe-source.py`. Engine/server
logs and metrics accompany these observations. The test server stopped and the
GPU had no compute processes afterward, as recorded in [cleanup.json](cleanup.json).

## Failures found and repaired

| Attempt | Result | Evidence |
|---|---|---|
| Original source, stock fallback instructions | Failed at an undeclared/disabled function boundary; repair incomplete | [default-profile-failed.zip](default-profile-failed.zip) |
| Original source, explicit local-tool instructions | Failed when a generated reasoning summary quoted literal tool markup | [summary-markup-failed.zip](summary-markup-failed.zip) |
| Summary fix applied | Repair, 12 tests and 45 checks passed, but final answer failed while quoting literal tool markup | [answer-markup-failed.zip](answer-markup-failed.zip) |
| Both parser fixes applied | Complete client turn, independent verification and all 19 native probes passed | [qualified-local-profile.zip](qualified-local-profile.zip) |

The stock fallback instructions told the model to use `apply_patch`, but that
function was absent from the actual tool declarations. The first failure did
not record its rejected function name, so this mismatch is a plausible cause,
not a proven identification of that call. The supported profile uses the
documented `model_instructions_file` setting to tell the model to edit through
the declared shell tool. The [instruction diff](instructions.diff.txt) shows
both changed sentences and whitespace normalization. No tool definitions,
storage settings, reasoning requirements or schema guarantees were downgraded.

The source fixes are small:

1. The answer-only reasoning-summary pass disables tool parsing in the existing
   service parser. Literal tool syntax in a summary remains text. Undeclared
   function errors now name the rejected function separately from duplicate IDs.
2. The main parser requires `<tool_call>` followed by optional whitespace and
   `<function=` before entering a function call. Other marker text remains
   literal. Tests cover fragment boundaries and a literal followed by a real
   call. The grammar output adapter still suppresses an unfinished tool envelope
   at the output limit while that prefix decision is pending.

No HTTP translation layer or alternate inference implementation was added.
The summary fix is `ff227055868a33e83889176a904882afc8d7d32b`; the main parser
fix is the running `36e9035...` checkpoint. Responses-only equivalents are
`f26283816c294c7af8b640d2b39a0b6568127ce5` and `82587463...`.

The early fault-probe harness incorrectly waited for `response.in_progress`
on queued work; its timeout is retained in the second archive. It was corrected
to wait for `response.created`, observe queue depth, close both connections and
verify draining. This was a harness error, not a failed server invariant.
Earlier capture checkers also treated a valid `response.failed` stream as a
structural failure. The separate audits distinguish wire validity from a failed
generation; original result files remain unchanged.

## Reproduce

Use the [Responses enablement instructions](../../../RESPONSES.md), the pinned
Windows Codex binary, the recorded native model/configuration, and environment
variables `STRATA_API_KEY` and `STRATA_RESPONSES_REPLAY_KEY`. Start the existing
server with `--experimental-responses --api-monitor` and enough context. These
flags and a replay key are required; the default remains disabled.

```text
python serve/server.py --engine strata --config YOUR_EXISTING_MODEL_CONFIG.json --host 127.0.0.1 --port 8095 --experimental-responses --api-monitor
python tools/responses_native_coding_probe.py --codex PATH_TO_PINNED_CODEX.exe --base-url http://127.0.0.1:8095/v1 --instructions docs/responses-evidence/native-codex/coding-task/local-model-instructions.txt --out .responses-runtime/coding-evidence
```

For a remote native host, forward its loopback endpoint over an authenticated
SSH connection first. The probe requires a fresh output directory and never
downloads a client or model. It creates a disposable workspace, isolated client
home, byte-copy recording relay and loopback HTTP reject proxy. Its recorded
invocation uses `--sandbox workspace-write`, `--ephemeral`, `--strict-config`,
`--profile strata` and `exec --json`. The [profile template](codex-profile.example.toml)
and [model instructions](local-model-instructions.txt) are included for inspection.
Adjust the template's absolute instruction path and base URL for manual use.

To rerun the native edge probes, extract `edge-probe-source.py` from the
successful archive and run it against the same server with
`--base-url http://127.0.0.1:8095/v1 --out NEW_EDGE_EVIDENCE_DIRECTORY`.
It requires the combined branch's grammar-enabled native build and the recorded
model/context. It deliberately cancels its own requests and does not stop the server.

The focused regression coverage can be reproduced with the worktree's Python
environment after installing its existing requirements:

```text
python -m unittest serve.test_server serve.test_responses serve.test_security serve.test_lifecycle serve.test_grammar_scope
python -m unittest serve.test_structured serve.test_detok serve.test_grammar
```

The [combined regression log](combined-server-tests.txt) records **203 passed**.
On the Responses-only checkout, omit `serve.test_grammar_scope`:
[195 passed](responses-server-tests.txt). The
[additional parser/structured/grammar log](other-server-tests.txt) records
**27 passed, five skipped** because local tokenizer assets were unavailable.
These Python regression tests use synthetic model output where applicable;
they are separate from the real native client run above.

## Limits and next gate

This qualifies one pinned client, model and explicit profile for a small coding
task. It does not claim all Codex versions or default-profile compatibility.
The native coding task did not request GBNF; the final companion request did.
Earlier [grammar/tool evidence](../../../gbnf-evidence/tool-scope/REPORT.md)
separately covers native grammar with reasoning and mocked external results.

All excluded capabilities remain explicit: strict schemas, JSON-object output,
Lark custom tools, hosted tools, retained storage/background, compaction and
WebSockets. Windows-native GPU execution, HIP, multi-GPU and long sessions were
not qualified here. G6 recovery remains deferred. Token usage is preserved as
reported; this checkpoint does not resolve the previously identified cache
accounting issue across reasoning-budget continuation passes. Replaying only
visible raw reasoning requires omitting a nonempty summary when omitting the
encrypted token; full returned reasoning items were used in this task.

The client made 13 background HTTP CONNECT attempts, all denied by the local
reject proxy. No external model was used. This records the test's HTTP behavior,
not an operating-system firewall guarantee for arbitrary programs. Deployment
keys and request authorization headers are excluded from the archives; encrypted
replay bytes remain opaque. Archive contents and hashes are checked byte-for-byte.

The next release gate is maintainer review of these parser fixes, evidence and
the recorded profile limitation. This checkpoint adds local commits on the two
existing branches; no additional branch, remote push, merge or PR publication
was performed for this test.
