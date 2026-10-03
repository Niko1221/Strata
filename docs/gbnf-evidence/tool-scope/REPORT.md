# Grammar with reasoning and mocked client tools

The native loop passed **16/16 HTTP cases** on llm-49 (RTX 4090, Coder IQ1_M).
The model generated a namespaced `facts.lookup` call with `{"text":"purple 999"}`.
The client supplied an arbitrary mock result, then requested a final answer with
`tool_choice: "none"`. The final text was exactly `color=blue;count=1\n` under
`root ::= "color=blue;count=1\n"`. Reasoning, tool arguments and tool results were
not constrained to that answer language. Only the external function result was mocked;
every model token in this native run came from the existing Strata decoder.

The [plain-text transcript](TOOL_LOOP.txt) contains actual request bodies,
output items, tool-result items and final responses. Ciphertext is retained as
an opaque replay token; it was not decrypted for the report. Auth/replay keys
were ephemeral and are not recorded. Captured IDs/ciphertext belong to this run.

## Source and environment

- Branch/worktree: `work/gbnf`, `C:/Users/dflanag3/Documents/fleet/strata-native-gbnf`.
- Passing native source: `f7cf318ee0d407292afea7782e648d7714069720`; [304 file hashes](native-loop/manifest.json).
- Native executable SHA-256: `656a6c16c2c148ec730c1b4b78f1e904417f2efb5cbce2455473156ce746a1ed`.
- Final boundary tests: `42a1db811dc14bdcd2a2d83f5421f6f49d16eafa`. The only implementation addition after the native
  run is a two-line rejection of non-array Chat `tools`; Responses and native code
  are unchanged. The malformed-request HTTP tests pass before any generation/SSE headers.
- Native interval: `2026-10-03T15:03:49Z` to `2026-10-03T15:06:07Z`.
- GPU/driver: `NVIDIA GeForce RTX 4090, 24564 MiB, 595.91.07`. Linux, Python 3.14.4, CUDA 13.3.73,
  pinned XGrammar 0.2.8 with the existing work-budget patch.
- Exact target-only/MTP settings and model paths: [run result](native-loop/run/result.json).
  MTP used `--spec 4 --mtp <existing-directory> --suffix-draft 0`;
  target-only used `--spec 1`. Models/weights are local prerequisites, not bundled data.
- Both engines stopped and the GPU was empty after each mode; [cleanup](native-loop/cleanup.json).
  No other host was used.

## Checks

Each mode covers reasoning off/on, JSON/SSE, and the call/final-answer turns.
The SSE checks require consecutive sequence numbers, exactly one terminal event,
stable item/index references and text/argument deltas that reproduce final items.

| Mode | HTTP cases | Drafts offered | Drafts accepted | Engine stopped |
| --- | --- | --- | --- | --- |
| target | 8/8 | 0 | 0 | yes |
| mtp | 8/8 | 270 | 216 | yes |

| Command/check | Result | Evidence |
| --- | --- | --- |
| `python -m unittest discover -s serve -p test_*.py` | 266 run: 261 passed, 5 skipped (no local pack tokenizer) at `2241806` | [Windows suite](python-tests.txt) |
| `python -m unittest serve.test_grammar serve.test_grammar_scope serve.test_responses` | 68 passed at `42a1db8` | [final HTTP/contract checks](scope-tests.txt) |
| `ctest --test-dir build-gbnf-cpu --output-on-failure -R 'grammar_|serve_input'` | 4/4 passed | [build/CPU log](build-console.txt) |
| `ctest --test-dir build-gbnf-gpu --output-on-failure -R 'grammar_|serve_input' -V` | 10/10 passed, including all three sampler paths | [native/CUDA log](native-tests.txt) |
| ASan + UBSan build, then the same four CPU tests | 4/4 passed, leak detection enabled | [sanitizer log](sanitizers.txt) |
| `grammar_tool_probe.py`, two native configs | 16/16 passed | [exact invocation](native-loop/invocation.json), [console](native-loop/console.txt) |

The synthetic HTTP fixtures additionally cover parallel flat/namespaced calls,
`id` versus `call_id`, client-owned result replay, final answers with tools still
enabled, Chat streaming, reasoning summaries outside the answer grammar, literal
markers/leading newlines, output limits, cancellation and old-capability rejection.
Those model channels are scripted and **are not native enforcement evidence**.

The native matcher tests use real XGrammar with synthetic tokens. The CUDA tests
use real masks and token selection with synthetic logits, including an eight-row
window crossing reasoning, tool and answer phases, plus 5,184 existing exact
speculative decode comparisons across three sampler paths. The native loop above
is the separate real-model check. No handoff tooling result is counted as a Strata test.

## Failures found and retained

Two earlier probes expected `tool_choice: "auto"` to answer immediately after a
result. This model instead generated another valid call, so those probe assertions
failed: [first](earlier-runs/auto-first/result.json),
[explicit follow-up](earlier-runs/auto-followup/result.json). Auto does not promise
the next turn is an answer. The passing native fixture makes the client's
one-call budget explicit with `tool_choice: "none"`; it is not evidence of an
autonomous Codex tool loop. Production does not silently change tool choice.

The next run found a real matcher bug: unlimited separator newlines could starve
the answer after reasoning, reaching the 768-token cap under MTP. The response
correctly reported incomplete, but the final-answer check failed
([receipt](earlier-runs/separator-regression/result.json)). Commit `f7cf318`
bounds protocol separators to two LF bytes and tests the bound and its forked
state. Grammar-permitted answer newlines remain unchanged. The same MTP case
then completed in 71 output tokens, including 60 reasoning tokens. The entire
16-case matrix was rerun after the fix, not just that one request.

## Reproduce and limits

Build/enable GBNF and Responses using the [native guide](../../NATIVE_GBNF.md).
Keep the configured `reasoning_budget_tokens` at zero for constrained reasoning;
injected wrap-up is explicitly unsupported. The retained [manifest](native-loop/manifest.json)
pins this run's exact source and binary. For a new build, generate a new manifest
with `source_commit`, `source_hashes` (relative paths to SHA-256) and `native_sha256`;
do not reuse this binary digest for a different build. Then run:

```text
python tools/grammar_tool_probe.py --source <checkout> --manifest <new-manifest.json> --config <target-config.json> --config <mtp-config.json> --out <new-evidence-directory>
```

The probe binds authenticated loopback HTTP, starts each engine sequentially,
refuses a busy GPU and closes its own resources. Its configuration uses existing
model/tokenizer/MTP artifacts and does not download or execute a client function.

This update qualifies ordinary target-only and MTP tool scopes on Linux/CUDA.
Coupled/suffix scope combinations, Windows native, HIP and multi-GPU are not
newly qualified here. Auto tool use may repeat; strict JSON schemas, Lark tools,
hosted tools and injected thinking-budget wrap-up remain unsupported. G6 stays
deferred. The prior feature-off native build/run and Responses accounting/replay
review follow-ups remain merge-review gates. This receipt is not a universal
Codex compatibility or final merge-readiness claim. Cold/warm timings are not
a performance comparison.
