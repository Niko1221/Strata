# G3: raw GBNF HTTP integration and target-only release

Worktree: `C:/Users/dflanag3/Documents/fleet/strata-native-gbnf`.
Branch: `work/gbnf`. Parent G2 checkpoint:
`7cec5ca526268acdfeab471f2deb2adf59787c4f`; parent R4 checkpoint:
`96092670da0dc3c1cfcce99bb90a3e6ca25ae1d9`.
The subsequent `G3-receipt.json` pins this passing source/evidence commit.
All work remains local; no fetch, push, reset, stash or rebase was used.

## Responsibilities

`serve/grammar.py` defines one immutable raw-source constraint and the supported
request profile. Both HTTP adapters normalize to that argument. The existing
Service owns preflight admission and generation through its FIFO. The existing
native compiler owns immutable cached languages, and generation owns a fresh
matcher. No HTTP adapter advances grammar state or calls another HTTP endpoint.

The [versioned native preflight](NATIVE_PROTOCOL.md) compiles and checks the
initial mask before successful streaming headers. It verifies the native and
Python emitted-byte vocabularies. Production IPC writes the complete bounded
frame atomically in binary mode. Unknown/old capability, syntax errors, conflicts,
tokenizer drift and unsupported modes fail explicitly. UTF-8 error-message
truncation preserves character boundaries so an ERR cannot break the pipe decoder.

Constrained generation uses the existing engine, queue, sampler, cancellation,
draining and lifecycle. Its semantic content path bypasses marker interpretation;
literal tags remain text. JSON and SSE consume the same service events. Partial
UTF-8 at a token limit is withheld without replacement; usage still counts the
consumed token. No output repair or second matcher is involved.

Responses remains opt-in at startup and stateless. Native grammar remains off
by default at build time. Instructions to enable both, the target-only profile,
SDK `extra_body`, complete `.txt` requests and limitations are in
[NATIVE_GBNF.md](../../NATIVE_GBNF.md). Existing auth and CORS apply.

## Commands, environment and results

Authorized native host: `llm-49` (`r4090`), RTX 4090, CUDA 13.3.73, GCC 15.2,
CMake 4.2.3, Linux, Python 3.14.4, official OpenAI SDK 3.23.0.
Model: existing Coder IQ1_M installation, INT8 KV, context 4096, prefill 256,
single GPU. Constrained mode uses `--spec 1`, no MTP weights and no suffix
proposals. Native build architecture is 89, Release. Local Python regressions
use Windows/Python 3.13. No other GPU host was contacted.

Final executable SHA-256:
`d7c7a16803393823fc7f98846287aa50fdd318f6c5bb8b3205992b24d3c97551`.
The [source manifest](native/source-manifest.json) matches the uploaded raw
source files. Receipts separately hash Git blobs, whose LF representation can
differ from the Windows checkout. Trailing spaces in five imported console-excerpt logs are trimmed; structured
JSON evidence is unchanged. Commands and intermediate attempts are retained:
[build](native/build-commands.json), [UTF-8 framing/sanitizers](native/unicode-build-commands.json),
[final native matrix](native/final-commands.json), [final HTTP probe](native/release-command.json).

```text
python -m unittest discover -s serve -p test_*.py -q
python -m unittest serve.test_grammar -v
cmake --build <build-gbnf-gpu> --target strata serve_input_test -j 4
ctest --test-dir <build-gbnf-gpu> -R '^serve_input_test$' --output-on-failure
cmake --build <build-gbnf-asan> --target serve_input_test -j 2
ctest --test-dir <build-gbnf-asan> -R '^serve_input_test$' --output-on-failure
python tools/grammar_api_probe.py --config <target-config> --out <fresh-directory>
python tools/grammar_native_probe.py --config <target-config> --out <fresh-directory>
python tools/target_only_probe.py --config <MTP-config> --mode mtp --out <fresh-directory>
```

| Gate | Evidence and result |
|---|---|
| G3-01: bounded IPC | Release and ASan/UBSan framing tests pass. Chunked reads, every frame truncation, CHECKG/GEN isolation, quotes/newlines/NUL and UTF-8 error boundaries are covered. [Release](native/unicode-build-1.txt), [sanitizers](native/unicode-build-3.txt). |
| G3-02: old binary | CPU tests reject absent/v1 capabilities before writing. A real G0 native binary is rejected before HTTP/SSE success in the [checkpoint probe](native/http-checkpoint/result.json). |
| G3-03: shared mechanism | The final [official SDK probe](native/api-release/result.json) passes 42 recorded cases with unscripted native model output. Both adapters produce the forced literal, recursive language and epsilon completion. |
| G3-04: explicit conflicts | Both routes reject tools, thinking, stops, JSON requirements, syntax errors and unsupported streaming options before successful headers. Final CPU tests additionally cover tokenizer/template drift, field/range validation, API keys, CORS and request isolation. |
| G3-05: unqualified MTP | The checkpoint HTTP probe rejects grammar on an actual MTP engine advertising `grammar=none`. The final MTP regression confirms that capability and ordinary generation; constrained speculation is not enabled. |
| G3-06: consistent content/lifecycle | JSON/SSE preserve literal markers and Unicode. Sampled/penalized generation, all earlier UTF-8 token limits, concurrent adapters and a real disconnect followed by a clean request pass. The [corrected native probe](native/native-final/result.json) passes 43 cases and checks 257 cursor windows, 253 constrained, including real process restart and prefill/decode cancellation. |
| G3-07: measured overhead | Twelve alternating warm target-only measurements are retained in the [CSV](native/api-release/target-only-benchmarks.csv), with no tracing/speculation in either mode. |

The [full Python suite](python-final.txt) ran 258 tests: **253 passed, five optional
tokenizer tests skipped**. The 16 grammar adapter tests are synthetic model
fixtures and are labelled accordingly ([detailed results](grammar-api-tests.txt)).
The suite emits an unclosed-file ResourceWarning in its existing subprocess
tests; no test failed. [Ordinary MTP](native/mtp-final/result.json) passes 11
cases with the final native binary and correctly forwarded repetition penalty.
The G2 kernel/memory qualification remains the selector evidence; G3 does not
change sampling kernels or speculation.

The earlier checkpoint probe used native SHA
`4e09823fbf7e0abd8316f570ef71bcb2822b5d9869337696f6a5051310fe15ac`
for its real old-engine/MTP HTTP checks, before the UTF-8 diagnostic truncation
fix. The final target API, native grammar and ordinary MTP runs all use the final
SHA above. Both versions, their commands and results remain inspectable.

## Measurement limits

The benchmark forces the unconstrained model's own answer, `4`, with the same
prompt and native output count: two tokens including the end control. Six warm
measurements per mode alternate order, reuse 20 prompt tokens and read seven.
Median native decode time was **27.55 ms plain / 28.30 ms constrained**;
median end-to-end time was **123.57 ms / 124.33 ms**. Median prompt time was
93.25 ms in both modes. These are measurements of a tiny warm literal on this
machine, not general throughput or a promise for recursive/large grammars.
Cold compilation and the initial vocabulary comparison are outside these warm
medians; full first-request timings remain in the SDK result.

## Corrections and remaining limits

Review found that `target_only_probe.py` and `grammar_native_probe.py` had used
native internal penalty names in the Python wrapper's sampling dictionary.
The wrapper ignored those keys. Earlier G0/G2 model runs labelled penalized did
not establish that behavior; the independent G2 kernel penalty tests did.
The probes now use `repetition_penalty`, `frequency_penalty` and
`presence_penalty`, and the wire test asserts their native keys. Final native,
HTTP and MTP receipts above cover the corrected behavior. The first corrected
raw probe stopped at a stale `gbnf-v1` assertion after restart; it now verifies
that the restarted process retains its negotiated version. The failed
[attempt](native/native-penalties/result.json) is retained rather than rewritten.

No native Windows, HIP, multi-GPU, custom-EOS or sm_90 cluster qualification is
claimed. No constrained MTP/suffix, recovery, JSON Schema/JSON-object frontend,
Lark tools, grammar-plus-tools or generated reasoning is advertised. Chat's
existing final-choice usage shape is retained; grammar requests reject
`stream_options` rather than ignoring the requested usage-only chunk semantics.
The separately pinned local Codex R4 tool-loop test still uses scripted model
output; these G3 SDK/native tests do not establish native-model Codex tool skill.

G3 is a useful target-only release checkpoint independent of G5. Next: G4,
bounded state-derived application contracts and private matcher inspection.
The user selected Mermaid/plain text for inspection; no actual Code Visualizer
ProgramModel integration will be claimed. G5 then qualifies constrained MTP and
suffix execution. G6 recovery remains deferred.
