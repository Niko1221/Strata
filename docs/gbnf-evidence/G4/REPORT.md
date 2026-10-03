# G4: state-dependent contracts and local native inspection

Worktree: `C:/Users/dflanag3/Documents/fleet/strata-native-gbnf`.
Branch: `work/gbnf`. Passing G3 parent:
`69f46806bae9af41b6c807f5c2df4aed1affa27e`.
The following `G4-receipt.json` pins the passing implementation/evidence commit.
No remote Git mutations were performed.

## Responsibility and ownership

`Matcher::inspect` observes the existing native matcher using a private copy.
It returns bounded facts: definition fingerprint, committed token/byte counts,
acceptance/termination, legal-token count, mask fingerprint and at most 64 token
previews of 32 bytes each. The parent's progress, failure flag and work budget
remain unchanged. It does not infer probabilities, enumerate parse trees,
advance model state or add a production inspector endpoint.

`tools/grammar_state_contract.py` is an application-side example with one
immutable snapshot and one in-memory task owner. It freezes revision and
permissions, derives bounded `G(S)`, independently verifies a full command and
checks the complete current snapshot under a lock before applying a transition.
No arbitrary task process executes. Seven Python tests cover stale state,
authorization changes, principal mismatch, injection, full-string verification,
two racing candidates and revision exhaustion before effects.

The [Mermaid/plain-text guide](../../GBNF_INSPECTION.md) maps operation definitions
to source and invocation labels to observations. The user explicitly requested
these views instead of supplying the actual Code Visualizer. No ProgramModel,
cross-selection, regeneration or importer is fabricated. That original
integration subgate is not performed; the requested source/trace views and the
native/application gates are complete.

## Commands and evidence

Native environment remains llm-49/r4090, RTX 4090, CUDA 13.3.73, GCC 15.2,
CMake 4.2.3, pinned XGrammar `xgrammar-0.2.8-strata-budget1`. The model is Coder
IQ1_M with the G3 single-GPU target-only configuration, standard end controls,
INT8 KV and context 4096. Official SDK: 3.23.0. Local application tests use
Windows/Python 3.13. No serving, sampler or speculative policy changes are made.

Final native executable SHA-256:
`1659d42772f2577cfa3c27a700d94cdd1bcea7f8d6824ba388a4dc47f354fa83`.
Inspection test SHA-256:
`809c621bb915f734ee8e9db2fef01bc2e4a6c7ec3295e52b6045bc1f38e9409e`.
The [source manifest](native/source-manifest.json) matches all seven uploaded
G4 implementation/test files. One trailing space in the imported CMake build
log is trimmed; diagnostic content is retained. Commands: [release CPU](native/initial-commands.json),
[sanitizer/native build](native/build-commands.json),
[actual state/native probe](native/checkpoint-command.json).

```text
python -m unittest tools.test_grammar_state_contract -v
python tools/grammar_state_contract.py
cmake --build <CPU-build> --target grammar_inspection_test grammar_native_test -j 4
ctest --test-dir <CPU-build> -R '^grammar_(inspection|native)_test$' --output-on-failure
<CPU-build>/grammar_inspection_test <actual-tokenizer-directory>
cmake --build <ASan-UBSan-build> --target grammar_inspection_test grammar_native_test -j 2
ctest --test-dir <ASan-UBSan-build> -R '^grammar_(inspection|native)_test$' --output-on-failure
python tools/grammar_state_probe.py --config <target-config> --inspector <inspection-test> --matcher-tests <native-test> --out <fresh-directory>
```

| Gate | Result |
|---|---|
| G4-01: stale state | Seven [application tests](state-contract-tests.txt) pass. The real model produces `START task-a`; an intervening permission revision makes the client reject it before any task transition. A fresh candidate advances revision 1 to 2 and removes task-a. |
| G4-02: injection | Untrusted names outside the bounded ASCII domain, duplicate/oversized task sets and modified/partial commands fail before effects. The three derived languages pass 18 accept/reject cases against both toy and actual native vocabularies, 36 total. |
| G4-03: inspection/fork/replay | [Native inspection](native/state-checkpoint/inspection.txt) passes with 260-token toy and 248320-token actual vocabularies. Parent/forks remain independent; restore replays the same native implementation. Wrong tokenizer, independent compiled ownership and invalid checkpoint history are rejected. Bounded previews, byte truncation and unchanged parent budget are asserted. |
| G4-04: overlapping alternatives | Both continuations after shared prefix `a`, whole multi-character tokens, accepting prefixes with further legal output and ambiguous recursion pass. Inspection reports the continuation union, not a unique parse or branch mass. Probability categories are explicitly `not_collected` and the mathematical illustration is labelled synthetic. |
| G4-05: visual inspection | User-selected Mermaid and plain text are supplied with source/trace links. Actual Code Visualizer integration is unverified and outside this substituted presentation; no such claim is made. |

Both native tests pass in Release and ASan/UBSan
([release](native/initial-1.txt), [sanitizers](native/build-1.txt)). The original
G1 corpus and resource tests also run in these targets; they are actual Strata
native-library tests, not tests of the handoff tooling.

The [real model result](native/state-checkpoint/result.json) contains three
Responses requests, the native binary identities and actual command outcomes.
Its [plain-text trace](native/state-checkpoint/state-trace.txt),
[stale request](native/state-checkpoint/stale-request.txt),
[fresh request](native/state-checkpoint/fresh-request.txt),
[limited request](native/state-checkpoint/limited-request.txt) and
[derived-language native test](native/state-checkpoint/derived-languages.txt)
are retained. The limited response visibly spells `WAIT` but reports incomplete;
the client refuses it before applying any effect. The tokenizer audit reproduces
the G1 byte-table SHA-256
`6fd80168d9b84473f2ca03963388698cd1f56e85acd6f525b15912e1ac4d7bca`.
Its generated binary scratch table is removed after hashing, not linked as a
downloadable artifact.

The first end-to-end attempt completed the three model requests and inspection
but omitted the native corpus test's required audit-output filename. That test
returned its usage error; it did not run a language check. The corrected command
passes the unchanged cases. The [failed attempt](native/state-native/result.json)
and [usage error](native/state-native/derived-languages.txt) remain as evidence.
The standalone [Python demonstration](synthetic-state-example.txt) is explicitly
synthetic; it is separate from the real model trace.

## Limits and next gate

This bounded example is not a production authorization service, task scheduler,
agent planner or automatic job executor. No model checkpoint is exposed by a
grammar checkpoint. No raw logits, final sampler distribution or branch
probabilities are collected. Backend ambiguity is preserved as viable
continuations; individual derivations may be merged and are not enumerated.
The diagnostic fingerprints do not replace exact checkpoint ownership checks.
Existing G3 hardware/profile restrictions remain unchanged.

Next: G5, explicit constrained MTP and suffix qualification with per-prefix
masks, accurate committed model/grammar state, fixed-logit selection parity and
native numerical diagnostics. G6 recovery remains deferred.
