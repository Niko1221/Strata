# G1: native compiler, vocabulary and matcher

Result: **pass for the recorded Linux CPU build and Coder tokenizer**.
Branch: `work/gbnf`.
Worktree: `C:/Users/dflanag3/Documents/fleet/strata-native-gbnf`.
Parent: `a2e65448d7906494d97a4e55866bca395ef7308a`, the G0 receipt commit.
The companion `G1-receipt.json` names the full passing source commit and hashes
the evidence. No Responses worktree, remote ref or existing native binary was
changed.

## Responsibilities

`grammar_vocabulary.cpp` derives emitted bytes and end/control classification
from existing tokenizer artifacts. `grammar.cpp` owns immutable bounded
compilation and private sequence state. `grammar_budget.hpp` supplies the
cooperative guards used by the pinned backend. `prepare_xgrammar.py` prepares
build dependencies; `cmake/xgrammar.cmake` builds the native library only when
`STRATA_ENABLE_GBNF=ON`. There is no model-selection or HTTP change in G1.

The [backend decision](grammar-backend-decision.md) records dialect, dependency
pins, licenses, bounds and the initial upstream CMake failure. The supplied
nine grammar files and 41 string cases are retained in `data/grammars`.

## Commands and results

Built in isolated directories on authorized `llm-49` (hostname `r4090`), with
GCC 15.2.0 and CMake 4.2.3. These tests are native CPU executions; no GPU/model
generation was needed. [Exact commands](native/commands-final.json),
[final checks](native/final-check-commands.json),
[source hashes](native/source-manifest.json) and
[dependency pin](native/PIN.json) are retained.

```text
python tools/prepare_xgrammar.py --out <new-dependency-directory>
cmake -S . -B build-gbnf-cpu -G Ninja -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_ENABLE_GBNF=ON -DSTRATA_XGRAMMAR_DIR=<prepared-directory> -DSTRATA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-gbnf-cpu --target grammar_native_test -j 4
python tools/grammar_vocab_audit.py --exe <build>/grammar_native_test --cases data/grammars/cases.json --tokenizer <existing-pack>/tokenizer --out <new-evidence-directory>
```

| Acceptance | Recorded result |
|---|---|
| G1-01: corpus | All 41 accept/reject cases pass twice: 261-token test vocabulary and actual 248320-token vocabulary. |
| G1-02/03: recursion, epsilon, overlap, ambiguity | Recursive balanced language, recursive ambiguous derivations, epsilon and accepting-prefix continuations pass; no unique-derivation claim or inspector is fabricated. |
| G1-04/05: bytes and controls | Split Unicode, a multi-character boundary token, escaped bytes, every excluded control ID and both model end IDs pass. All 248044 normal IDs exactly match the existing tokenizer; 276 control/unused IDs are excluded. |
| G1-06: rejection and bounds | Malformed, unproductive, unrealizable, oversized and unsupported source fail. Actual compile/matcher work exhaustion and deadline failure fail explicitly; a fresh sequence remains usable. |
| G1-07: independent state | Cache reuse/eviction, independent forks and fresh grammar states pass. Unconstrained generation has no matcher yet and retains the G0 implementation. |
| G1-08: identity | Wrong grammar or tokenizer checkpoints are rejected. Valid replay preserves the committed prefix and leaves its parent independent. |

[Native test output](native/audit-final/grammar-native-tests.txt) and
[full tokenizer receipt](native/audit-final/tokenizer-identity.json) include the
executable and emitted-byte-table SHA-256 values. The byte table itself is
reproducible with the command above and is not presented as an online artifact.
The reported native test duration covers the corpus and byte-table export;
it is not an inference-throughput benchmark.

The same corpus and native cases pass AddressSanitizer and
UndefinedBehaviorSanitizer: [commands](native/asan-commands.json),
[final results](native/asan-tests-final.txt).
[Feature-off configuration](native/feature-off-result.txt) succeeds without
the dependency and contains no XGrammar target. No Python API implementation
changed since the G0 regression run; the two new build/audit helpers were
syntax-checked separately.

## Limits and next gate

G1 does not enable a `grammar` HTTP field or enforce model output. Its tests
exercise the actual native backend, not a toy replacement or a final regex
validator. Native Windows/HIP/multi-GPU behavior remains unqualified. Resource
deadlines are cooperative, and memory estimates are not allocator-wide quotas.
Only the recorded `gpt2/qwen35` vocabulary profile is supported.

Next: G2, apply legality before all native pruning and selection paths, including
captured greedy execution; fail empty candidate sets, gate EOS and align matcher
progress with committed model output. JSON Schema and JSON-object features
remain excluded. G3 will add the raw GBNF HTTP/IPC boundary.
