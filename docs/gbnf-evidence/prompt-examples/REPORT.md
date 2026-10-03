# Explicit prompt examples: Windows and Ubuntu, 2026-10-03

Both real Codex CLI clients repaired the disposable parser project without added
task examples. Each ran 12 passing unit tests and passed 45 independent checks,
including LF/CRLF/CR, Unicode, literal tool tags, malformed input, duplicates,
preserved fixture bytes and the restriction to changing one source file.

Separately, 11 of the 12 captured tool fixtures passed the zero-example baseline.
`get_goal` generated `{"arguments":{}}` instead of `{}`. One explicit native
empty-call demonstration fixed that generated call; the adapter did not strip
or rewrite it. All 30 schema fixtures passed without added examples.

## Exact sources and responsibilities

Branch: `work/gbnf`. Worktree:
`C:\Users\dflanag3\Documents\fleet\strata-native-gbnf`.

- Native tool/schema probes and Windows Codex used server source
  `4db981317a41d973bb4fec3c29fce395b90fc6a6`.
- The fresh Ubuntu check used `0965ddb56bfd41329f4ead65e9d789c8fbfe0657`.
  Its copied source lived at
  `/home/dflanag3/fleet-downloads/strata-native-gbnf-20261003/interactive/chat-20261003T213533Z-d6e3d9/source`.
- The [receipt](receipt.json) verifies that server and native source files are
  identical between those commits. Differences concern fixtures, documentation,
  and the test observer. The native binary SHA-256 is
  `908cb11a334772494b092df15d35267fb8b7a290ad1d431edb90f6a849a7d94c`.
- Codex is pinned to 0.160.0 on both systems, with separate binary hashes in the
  receipt. Ubuntu used the official Linux package in an isolated directory.
  No existing checkout was switched, fetched, reset or rebased.

`serve/responses.py` no longer inserts an automatic empty-call hint.
`tools/responses_prompt_examples.py` supplies explicit synthetic examples to the
test clients through ordinary `instructions` or Codex's `model_instructions_file`.
The native probes record each count and keep the failed baseline before trying
one through three demonstrations. The real Codex harness uses a fresh project
and client home on each attempt. It records actual calls/results and independently
verifies the edited code. The output assembler, engine, queue and inference path
are unchanged in this phase.

Counts mean **added task demonstrations**. The installed Qwen template's ordinary
XML syntax skeleton remains in every variant; its hash is recorded. Do not call
this an instruction-free or universal zero-shot benchmark.

## Commands and measured results

```text
python tools/responses_codex_tools_probe.py --base-url <authenticated-loopback>/v1 --model qwen3.8-flash-next --catalog-dir docs/codex --platform windows --hint-on-failure --out <new-directory>
python tools/responses_schema_inventory_probe.py --base-url <authenticated-loopback>/v1 --model qwen3.8-flash-next --hint-on-failure --out <new-directory>
python tools/responses_native_coding_probe.py --codex <pinned-Windows-binary> --base-url <SSH-loopback>/v1 --hint-on-failure --timeout 600 --out <new-directory>
python3 tools/responses_native_coding_probe.py --codex <pinned-Linux-binary> --codex-sha256 12eb3e81114588aca3b7998f4f19e8997b056aca08e57a7ca7c8a3ec8c652aad --base-url <server-loopback>/v1 --hint-on-failure --timeout 600 --out <new-directory>
```

Exact expanded invocations, profiles, instructions, tasks, requests and responses
are alongside each receipt. API keys are not recorded. Both clients used real
native inference on llm-49 (RTX 4090, Coder IQ1_M, 32768 context, int8 KV,
ordinary MTP `spec=4`, suffix off). Coding used temperature 1.0, top-p 0.95,
top-k 20 and medium reasoning with a 2048-token local thinking budget. The
isolated tool/schema probes used temperature zero. These are qualification runs,
not latency or model-quality comparisons.

| Check | Result and receipt |
|---|---|
| Captured tool fixtures | [11 baseline passes, one pass after one example](native-tools/result.json); mocked external results, no tool actions executed by this probe. |
| JSON schemas | [30 baseline passes](native-schemas/result.json); prompts supply known valid values. |
| Real Windows Codex | [Passed with zero examples](codex-windows/examples-0/result.json), 5 calls/results, 284.163 seconds. [45 independent checks](codex-windows/examples-0/independent-verifier.json). |
| Real Ubuntu Codex | [Fresh run passed with zero examples](codex-ubuntu/examples-0/result.json), 6 calls/results, 312.033 seconds. [45 independent checks](codex-ubuntu/examples-0/independent-verifier.json). |
| PowerShell examples | [Three executed examples](windows-shell-examples.json), including spaces/brackets and exact UTF-8 bytes. |
| Bash examples | [Three executed examples on Ubuntu](ubuntu-shell-examples.json), with the same byte verification. |
| Python regression checks | [239 service/API tests](python-tests.txt) and [5 prompt/harness tests](prompt-tests.txt), including all example schemas, parser round trips, strict variants, count limits and shell-status regression. |

Every real Codex call ID matched its returned result. Typed streams reconstructed
their final responses exactly. The clients observed failing tests before the
edit, then passing tests, and recovered from the deliberately missing file.
The model servers stopped afterward; llm-49's GPU process list was empty.

## A test-observer failure, preserved

The first Ubuntu run completed the coding task, but the old observer called it
failed because it only recognized a failing **whole shell call**. Codex ran a
failing command followed by `echo` of its exit code; the resulting shell call
returned zero even though its output clearly contained the failure.

The original [baseline](ubuntu-original-observer/examples-0/result.json) and
[one-example attempt](ubuntu-original-observer/examples-1/result.json) remain
unchanged. The [explicit reassessment](ubuntu-original-observer/reassessment.json)
distinguishes observed subcommand errors from final shell status. A redundant
two-example attempt was [interrupted](qualification-interruption.json) when this
checker bug was identified; it is not a measured model failure. The fresh Ubuntu
baseline above passed with the corrected observer, so no hint benefit is claimed
for this coding task. Empty/success-only output cannot satisfy the new check.

## Limits and next gate

This qualifies the pinned clients, model and fixtures. It does not prove that
other models, all possible schema values, all interactive permissions or long
conversations work. Shell mistakes and recoveries are retained. Captured tool
fixtures and synthetic strict variants do not mean Strata executes those tools;
Codex owns execution. The 266 exported request variants are labeled synthetic,
and were not all individually generated by the real model. The seven mode
examples and unneeded two/three-example variants are inspection/test inputs.

Encrypted replay remains opaque to this test harness. Examples do not enable
Lark, hosted tools, image input, background execution, storage or compaction.
These are client tests against Ubuntu CUDA inference, not a new Windows native,
HIP, multi-GPU or 4B dense-engine qualification. Native JSON and GBNF limits
remain as documented. The next gate is reviewing and publishing the local
changes, preserving the Responses-first, then GBNF landing order.
