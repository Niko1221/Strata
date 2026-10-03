# G0: persistent target-only serving

Result: **pass for the recorded Linux/CUDA, single RTX 4090 configuration**.
Branch: `work/gbnf`.
Worktree: `C:/Users/dflanag3/Documents/fleet/strata-native-gbnf`.
Parent: the frozen R4 commit `96092670da0dc3c1cfcce99bb90a3e6ca25ae1d9`.
The companion `G0-receipt.json` names the full passing commit and hashes the
evidence; its commit is separate to avoid a self-referential Git hash.

## Changed responsibilities

`generate.cpp` resolves `--serve --spec 1` before loading a model. It removes
the absent drafter from memory estimates, binding, batched/short prompt
processing, KV restore and draft hooks. It omits suffix allocation/proposals
in that mode. Conflicting explicit proposal settings and unsupported parked
conversation storage are errors. The existing queue, engine pipe, native
session, checkpoint, STOP/drain and restart mechanisms are unchanged.

The existing verifier now permits a one-position allocation, matching its
already-supported one-position execution. Engine `INFO` reports requested and
effective speculation settings and actual MTP allocation. Optional trace lines
record the [cursor relation](CURSOR.md). No grammar dependency, matcher, HTTP
adapter change, new decoder or unrelated hardware patch is part of G0.

## Commands and results

Built from the [recorded native input hashes](native-source-manifest.json),
with the repository's pinned ggml commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. The isolated dependency came from a
commit archive, with [SHA-256 receipt](native/dependency.json); no Git fetch,
reset, stash, rebase or push was performed. The original worktrees and previous
native build were left intact.

Remote root: `/home/dflanag3/fleet-downloads/strata-native-gbnf-20261003` on
authorized `llm-49` (hostname `r4090`). The model is the Coder IQ1_M pack with
INT8 KV, context 4096, prefill 256, two prompt checkpoints and adaptive swaps
disabled. The test shared only existing read-only model/tokenizer data.
CUDA 13.3.73, GCC 15.2.0, CMake 4.2.3, architecture 89, Release build.
The actual [configure command](native/build-commands.json),
[configure log](native/configure.txt) and [build log](native/build.txt) are retained.

```text
cmake --build <remote-root>/build --target strata conversation_cache_test conversation_memory_test -j 4
<remote-root>/venv/bin/python tools/target_only_probe.py --config <target-config> --mode target --out <fresh-dir>
<remote-root>/venv/bin/python tools/target_only_probe.py --config <mtp-config> --mode mtp --out <fresh-dir>
<remote-root>/venv/bin/python tools/target_only_probe.py --config <suffix-config> --mode suffix --out <fresh-dir>
```

Each result records the exact executable SHA-256 and native arguments; the
model output is real, not scripted. The executable was
`ed77a1c3d4bf93b3434383a1b7ed2c303ee0e3d68a802460759793b8a109d029`.

| Acceptance | Result | Evidence |
|---|---|---|
| G0-01: persistent without MTP | Pass; actual allocation 0.0 MiB, repeated requests and semantic `Service.run` | [target results](native/target-1/result.json), [live transcript](native/target-live-1.txt) |
| G0-02: default suffix setting | Pass; requested lookup 3, effective lookup 0, no suffix proposals | [target native trace](native/target-1/engine.txt) |
| G0-03: lifecycle/cache/cursors | Pass; exact live-prefix reuse, checkpoint rewind, seeded penalized replay, one-token cap, decode and prompt cancellation/drain, next-request cleanup, actual process restart; 213 target-only windows checked | [target results](native/target-1/result.json) |
| G0-04: ordinary MTP/suffix | Pass; both execute real requests; suffix windows actually exercised | [MTP results](native/mtp-1/result.json), [suffix results](native/suffix-1/result.json) |

Ten selected native tests passed: MMVQ multi-position, GDN recurrence, GDN,
three sampler implementations, conversation memory/cache, checkpoint retention
and coupled-draft counter/history arithmetic. See
[CTest output](native/target-only-native-tests.txt) and
[test build](native/kernel-build.txt). The first broad CTest invocation also
registered 41 executables that had not been built, so those were **Not Run**;
the [failed invocation](native/cpu-native-tests.txt) is retained. The focused
run explicitly names the built relevant targets. This is not a claim that all
registered native tests passed.

Windows Python regressions:

```text
.venv/Scripts/python.exe -m unittest -v serve.test_responses serve.test_security serve.test_server serve.test_lifecycle serve.test_structured serve.test_monitor serve.test_mcp
.venv/Scripts/python.exe -m unittest -v serve.test_detok
```

232 plus 8 tests: **235 passed, 5 skipped** (optional pack tokenizer unavailable
in the Windows checkout). [Main results](python-regressions.txt),
[detokenizer results](detokenizer-tests.txt), [own venv lock](python-lock.txt).

## Limits and next gate

No native Windows, HIP, multi-GPU or other model qualification is claimed.
Parked conversation snapshots remain unavailable without MTP; ordinary live
prefix/checkpoint reuse is tested. Measurements include tracing and state
hashes and are correctness evidence, not performance comparisons.

The existing MTP loop retains an evaluated tail at some output/EOS caps. The
recorded continuation may therefore use a matching prompt checkpoint instead
of all live KV. G0 preserves this unconstrained behavior. G5 must correct and
qualify the model/matcher/output commitment relation before constrained
speculation is supported. A same-seed text check here is not universal bitwise
parity across verifier shapes or hardware.

Next: G1, evaluate/pin a native XGrammar build, verify the supplied recursive
GBNF dialect corpus, bridge actual tokenizer bytes and special tokens, and
bound compiler/cache/matcher resources. Raw `grammar` is not an enabled API
capability at G0. JSON Schema/JSON-object features remain excluded.
