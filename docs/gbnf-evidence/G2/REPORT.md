# G2: constrained target-only selection

Result: **pass for the recorded Linux/CUDA, single RTX 4090 configuration**.
Branch: `work/gbnf`.
Worktree: `C:/Users/dflanag3/Documents/fleet/strata-native-gbnf`.
Parent: `bd4c3b91f2254f6d92a8fe6a8b990ac543a49e3b`, the G1 receipt commit.
`G2-receipt.json` names the full passing source commit and hashes its evidence.

## Responsibilities and boundary

The existing sampler takes optional device mask buffers. It excludes illegal
IDs, applies penalties once, checks the joint candidate set, then uses the
existing top-k/top-p/min-p/temperature/Philox implementations. Empty rows and
non-finite legal scores produce an error sentinel before selection; no empty
argmax, unconstrained retry or fabricated token is emitted. Raw logits remain
available for diagnostics. Constrained greedy uses the qualified one-block
selector; sm_90 cluster execution is not qualified here.

The verifier uploads request masks and selects after graph replay whenever a
constraint is active, including greedy requests. A captured preliminary pick
cannot bypass the mask. Buffers are allocated lazily and masks are cleared at
request entry/exit. Compilation is shared; every request gets a fresh matcher.
The native decoder advances it through the selected output, commits the model
window, then emits. EOS remains a separate legal control transition. Exhausting
the output budget reports `length`, even at an accepting prefix; STOP reports
`cancel` after the existing engine drain.

The bounded native frame was brought forward from G3 so this selector could be
tested through the actual persistent engine. This is an input extension to the
existing pipe/queue, not another service. [Framing details](NATIVE_PROTOCOL.md)
cover byte counts, EOF, control isolation and capability negotiation. G3 still
owns the shared Python argument, HTTP validation, raw content events and API
qualification. No `grammar` HTTP feature is enabled at this checkpoint.

## Commands and results

Authorized host: `llm-49` (hostname `r4090`), RTX 4090, CUDA 13.3.73, GCC 15.2,
CMake 4.2.3, architecture 89, Release. The Coder IQ1_M pack uses INT8 KV,
context 4096, prefill 256, no MTP weights/suffix in constrained mode, and adaptive
swaps disabled. Model/tokenizer files are the existing read-only installation.
Builds, configs and logs are isolated under
`/home/dflanag3/fleet-downloads/strata-native-gbnf-20261003`.

The exact final native executable SHA-256 is
`67b81927af8e1d4b5170e54bbe8a5eb06f4c5355e02400d22c96d69582159cbd`.
[Build commands](native/build-commands.json),
[kernel/sanitizer commands](native/final-check-commands.json),
[checkpoint commands](native/checkpoint-commands.json) and
[source hashes](native/source-manifest.json) are retained. The first build found
an undefined CUDA infinity macro; the fix uses the engine's existing bit-pattern
convention. Both the [failed build](native/build.txt) and
[passing final build](native/build-checkpoint.txt) are retained.
Trailing whitespace in the imported initial compiler/configure logs is trimmed;
diagnostic text and results are retained.

```text
cmake --build <build-gbnf-gpu> --target strata grammar_sampler_test serve_input_test sampler_parity grammar_native_test coupled_draft_test -j 4
ctest --test-dir <build-gbnf-gpu> -R '^(grammar_sampler.*|sampler_parity.*|serve_input_test|grammar_native_test)$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 <build-gbnf-gpu>/grammar_sampler_test
python tools/grammar_native_probe.py --config <private-target-config> --out <fresh-directory>
python tools/target_only_probe.py --config <private-MTP-config> --mode mtp --out <fresh-directory>
```

| Acceptance | Result and evidence |
|---|---|
| G2-01/02: legality before selection/pruning | Illegal maxima and top-k traps pass against the masked reference distribution. |
| G2-03: all selector paths | 46 cases, each with three rows of 248320 logits, run across the split, one-block and old implementations; captured/uncaptured, greedy/sampled, penalties and changed-mask replay. [Eight selected native tests pass](native/kernel-tests-final.txt). |
| G2-04: no eligible candidate | Empty masks, hard negative infinity, NaN/positive infinity and penalty overflow fail explicitly; no output ID 0 fallback. CUDA [memory checking reports zero errors](native/cuda-memcheck.txt). |
| G2-05: EOS and acceptance | Real forced Unicode and recursive outputs complete only through legal end controls. G1 accepting-prefix tests and the installed native masks preserve optional continuations. |
| G2-06: budget/cancel/commit | Every earlier token cap of the 18-token Unicode example returns a legal byte prefix and `length`. Four decode cancellation prefixes and prefill cancellation drain, then the next request is clean. An actual process restart begins with a fresh matcher. |

The [real native probe](native/native-checkpoint/result.json) records 43 cases
and checks 260 model cursor windows, 256 of them constrained. For each target
window, consumed length is prompt length plus produced count minus one; the
sampling position is consumed length minus one. Matcher token count equals
produced count, including the selected end control. A restart begins at zero
matcher progress. [Trace](native/native-checkpoint/engine.txt).

The [ordinary MTP regression](native/mtp-regression/result.json) passes all 11
recorded cases, including seeded penalties, continuation/checkpoint reuse,
cancellation/drain and the existing semantic service.
[Coupled-draft counter/history tests](native/coupled-draft-cpu.txt) also pass.
These are unconstrained MTP checks; constrained speculation remains unavailable.
The new native framing tests also pass ASan/UBSan
([build](native/input-asan-build.txt), [result](native/input-asan-tests.txt)).

## Limits and next gate

No native Windows, HIP, multi-GPU, custom end-ID or sm_90 cluster qualification
is claimed. GBNF is advertised only for the single-GPU target-only text profile
with the existing two end controls. Numerical/grammar resource failures stop
generation explicitly; there is no recovery implementation. The existing
ordinary speculative tail behavior remains a G5 responsibility.

These native probes bypass HTTP and therefore do not certify API streaming or
Codex model/tool behavior. Timing logs include tracing, cache effects and grammar
work and are not the controlled G3 overhead benchmark. Next: G3, one shared
constraint argument and native preflight, both HTTP adapters, raw byte-preserving
content events, unsupported-combination errors, SDK/SSE/native qualification and
a separate target-only release receipt. JSON frontends remain excluded.
