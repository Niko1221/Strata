# Parser fixes found by a real native Codex task

This Responses-only branch contains the two parser fixes discovered while
running a real native Codex read/edit/verify task on the stacked `work/gbnf`
branch. The native run was not performed in this separate checkout.

- Branch: `work/responses-api`.
- Worktree: `C:/Users/dflanag3/Documents/fleet/strata-responses-stateless`.
- Source checkpoint: `82587463b81c9787f69b72b5f73e008a983d18f9`.
- Summary-parser fix: `f26283816c294c7af8b640d2b39a0b6568127ce5`.
- Native-tested combined source: `36e9035ec4fa789eb5ba1cec5988b8d4ffb889cf`.

The summary generation pass now treats tool-like markup as literal text through
an explicit option on the existing service parser. The ordinary output parser
requires the native `<tool_call>` / `<function=` preamble before starting a
call, preserving other literal marker text across fragment boundaries. An
undeclared-function error names the function separately from a duplicate call ID.
No native serving, MTP, GPU lifecycle or scheduling code changes are included.

The [regression log](server-tests.txt) records **195 tests passed** in 100.375
seconds. Reproduce that coverage in this checkout's Python environment with:

```text
python -m unittest serve.test_server serve.test_responses serve.test_security serve.test_lifecycle
```

The tests include synthetic output at every fragment boundary for literal tool
markers, a literal marker followed by a real call, literal markup in a generated
summary, and service failures. These Python tests are distinct from native
inference qualification.

The combined branch's native evidence is stored at
`docs/responses-evidence/native-codex/coding-task/REPORT.md` in that branch. Its
Codex 0.160.0 local-tool profile completed eight Responses requests and nine real
shell calls; 12 task tests, 45 independent checks and 19 native companion probes
passed. The unmodified fallback profile failed an earlier attempt. The recorded
profile replaces instructions to call an absent `apply_patch` function with use
of the declared shell function; it does not rewrite tools or weaken API checks.
This is not a universal Codex compatibility claim.

Remaining limits include the known cache-accounting issue across reasoning
budget continuations, the visible-only reasoning replay restriction described
in [the guide](../../RESPONSES.md), and the existing excluded API capabilities.
The next gate is maintainer review of the parser changes and the pinned profile
qualification. This checkpoint creates local commits only; it adds no branch
and performs no remote publication.
