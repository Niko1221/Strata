# R0: baseline and initial client profile

Result: pass for preflight; Codex end-to-end compatibility is blocked, not passed.
Branch: `work/responses-api`. Worktree: `<WORKSPACE>/strata-responses-stateless`.
Base: original-author `upstream/main`, `99f3dbd0b21d1401b3769e0c0d963913607f380b`.
The remote tip was checked with `git ls-remote`; the commit was already present locally.
No fetch, push, reset, stash or rebase was used. The existing `feat/responses-api-451`
checkout and PR remain separate. No GBNF branch was created.

## Responsibilities and evidence

- [Source audit](source-audit.json): remotes, existing worktrees, resolved base and hashes.
  Nine of thirteen inspected Git blobs differ from the supplied ZIP baseline.
  The ZIP does not identify a trusted Git commit.
- [Client profile](client-profile.json), [configuration](codex-profile.toml) and
  [captured initial request](request-fixtures/codex-initial-1.json): real Codex CLI
  0.160.0 traffic sent to a loopback recorder, using the actual Strata model name.
  The executable and companion host are copied into ignored, worktree-local runtime
  storage and identified by SHA-256. The Python SDK is pinned to 3.23.0 in the
  [complete test-environment lock](python-lock.txt).
- [Strata baseline results](baseline/test-results.json): seven existing CPU suites,
  195 tests, 190 passed and five tokenizer-pack tests skipped. These use the
  existing mock/scripted engine; they do not test native inference or MTP.
- [Handoff checks](handoff-checks.json): pack structure passes. The initial tooling
  run had two Windows default-encoding failures. The same 51 tests pass under
  Python UTF-8 mode; see [UTF-8 run](handoff-tests-utf8.txt). These test only the
  supplied handoff tooling, not Strata.

Commands from this worktree:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r docs/responses-evidence/R0/python-lock.txt
.venv/Scripts/python tools/responses_capture.py --codex <INSTALLED_CODEX_EXE> --out docs/responses-evidence/R0
.venv/Scripts/python tools/responses_preflight.py --handoff <HANDOFF> --out docs/responses-evidence/R0
.venv/Scripts/python tools/responses_verify.py --out docs/responses-evidence/R0/baseline
# In the handoff directory, with PYTHONUTF8=1 for child processes as well:
python -X utf8 -m unittest discover -s tests -v
```

The preflight script requires HEAD at the selected baseline. Run it before the
phase commits, not as a later implementation test.

## Actual client decisions

The recorder intentionally returns synthetic HTTP 400 `r0_capture_only`. No model
output, tool execution or successful client loop is claimed. All handoff example
events remain synthetic. Credentials and private sessions were not loaded;
captured paths, installation/session IDs and the ephemeral port are sanitized.

The named Codex profile sends `store:false`, `stream:true`, ordinary functions
with explicit `strict:false`, and full content-bearing messages. Those are within
the intended adapter profile. It also sends:

| Request field | Decision |
|---|---|
| `reasoning.summary: "auto"` | Reject; raw model thinking is not a generated summary. |
| `include: ["reasoning.encrypted_content"]` | Reject; no fabricated encrypted state. |
| `tools[4].type: "namespace"` (`multi_agent_v1`) | Reject; initial profile supports flat function tools only. |
| `client_metadata` | Reject as an unsupported extension. |
| `prompt_cache_key` | Reject until its semantics are implemented; not a history lookup. |
| Function `strict:false` | Supported target for R3; no schema guarantee. |
| Strict schemas, Lark, compaction | Not observed here; explicitly excluded regardless. |

Codex warns that it lacks metadata for the Strata model and uses its fallback
metadata. This is recorded, not concealed by impersonating a hosted model or
altering the client's tool descriptions. Later-loop/compaction requests remain
unobserved. The pinned profile cannot pass R4 with the excluded requests above.

The initial capture attempt combined `--ignore-user-config` with `--profile`;
this binary did not apply the profile and attempted the default provider, which
rejected the credential-free request with HTTP 401. No generation occurred.
The corrected recorder uses a fresh client home, an explicit profile, its matching
companion executable, and a rejecting loopback proxy for accidental remote HTTP
destinations. See [first-attempt manifest](capture-attempt-1.json) and
[successful recording output](codex-capture-output.txt).

## Next gate

Implement R1's stateless POST, normalization, capability checks and request-local
assembler. Keep the existing Chat Completions JSON features and native inference
unchanged. R3/R4 must add actual supported SDK exchanges and report the real Codex
blocker; neither an SDK-only result nor a synthetic loop unlocks the GBNF worktree.

Protocol sources: [Responses](https://developers.openai.com/api/reference/resources/responses),
[function calling](https://developers.openai.com/api/docs/guides/function-calling),
[streaming](https://developers.openai.com/api/docs/guides/streaming-responses),
[Codex provider/profile configuration](https://learn.chatgpt.com/docs/config-file/config-advanced).
