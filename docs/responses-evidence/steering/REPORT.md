# Local Codex steering checkpoint, 2026-10-03

Codex CLI 0.160.0 on Windows completed a real read/edit/test task through Strata's
native Qwen3.8-Flash-Next Coder IQ1_M model on llm-49. After an interrupted long
answer and a permission change to Read Only, it returned exactly `STEER_OK`.
No cloud model, scripted engine output, or manual replay-token decryption was used.

Responses-only fixes: `work/responses-api` at
`c666de0e3add538690d63883cd6b687eb23dddb1`, worktree `strata-responses-stateless`.
Live combined test: `work/gbnf` at `0fc79c445d6a48d41004dbf85ac9490a63337be7`,
worktree `strata-native-gbnf`. The [session receipt](session.json) pins the client,
native binary, catalog and configuration. This live test used the combined branch;
the Responses-only branch has separate protocol tests, not a separate GPU run.

`resolve_input` gathers system/developer instructions into the native template's
leading instruction block in arrival order, retaining roles and tool-result binding.
It preserves completed reasoning when interruption removed the unfinished answer.
The explicit model catalog replaces fallback metadata and selects medium reasoning.
See the [current setup guide](../../CODEX_LOCAL.md).

Real local tool calls read `calc.py` and `test_calc.py`, changed subtraction to
addition and ran `python -m unittest -v test_calc.py`: all three tests passed.
An [independent rerun](independent-tests.txt) and [file receipt](independent-verifier.json)
confirm the result. The test file remained unchanged.

We requested 200 Python-reading tips, pressed Escape during output, selected Read
Only via `/permissions`, and sent: Stop the list. Do not use tools. Reply with
exactly STEER_OK. The [turn log](turn-summary.json) records the result. The
[derived request shape](steering-shape.json) shows standalone reasoning and the late
developer message; encrypted payloads remain opaque. `/quit` exited zero and the
launcher confirmed its server stopped.

`.venv/Scripts/python -m unittest -v serve.test_responses`: **49 passed** on each
branch, including visible/encrypted interrupted replay and late permission updates.
[This branch's protocol output](protocol-tests.txt) is retained.

Earlier attempts reproduced both System message must be at the beginning and
visible reasoning must precede an assistant message or function call. The final
profile uses temperature 1.0, top-p 0.95, top-k 20, 2048 thinking tokens and 32768
context. The fixes address request normalization, not model sampling.

This run exposed a separate native JSON compilation-budget failure in auxiliary
title requests; this steering receipt does not claim those requests succeeded.
The subsequent combined-branch JSON checkpoint records that fix separately.
Long-session compaction and universal Codex compatibility remain unqualified.
The local launcher uses a fresh replay key per invocation; begin a fresh chat.
