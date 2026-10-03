# G5: constrained MTP and suffix qualification

Result: pass for the recorded Linux/CUDA profile.
Worktree: `C:/Users/dflanag3/Documents/fleet/strata-native-gbnf`.
Branch: `work/gbnf`. Parent G4 implementation:
`568e878dd4c4538c455fd12423b38bf42d66a8be`.
The following `G5-receipt.json` pins the full implementation/evidence commit.
R4 remains the ancestor `96092670da0dc3c1cfcce99bb90a3e6ca25ae1d9`.
No fetch, push, reset, stash or rebase was performed. The original checkout and
frozen Responses worktree remain unchanged.

## Changed responsibilities

`Matcher::prefix_masks` walks a private native matcher through at most seven
proposals. Each mask precedes that proposal; an illegal or end proposal retains
its replacement row and removes descendants. Tentative work is charged to the
request, while tentative progress is discarded. The pending feedback token is
already in the committed grammar prefix and is not accepted again.

`retained_window` is a pure equality/budget/EOS decision shared by persistent
serving and ordinary CLI speculation. Previously the engine could commit model
inputs beyond the output that a length limit or EOS allowed it to emit. Now one
retained count controls target KV/recurrent commitment, emitted output, matcher
progress and the target rows supplied to draft catch-up. The invariant is
`consumed = prompt + produced - 1`; the final output is pending model feedback.
Only the existing target selector determines accepted proposals. Normal mismatch
never changes acceleration mode. There is no recovery framework or second sampler.

The native capability now admits supported single-GPU MTP/coupled/suffix text
configurations. The `gbnf-v2` framing and preflight contract are unchanged. The
shared Python service, FIFO, authentication, cancellation/draining and Responses
assembler remain the HTTP integration. Native diagnostics buffer bounded head
rows only when explicitly enabled; ordinary generation does not copy logits to
the host or write a token log. Mermaid and plain-text views are in the
[inspection guide](../../GBNF_INSPECTION.md).

## Environment and exact evidence

Only llm-49 (hostname r4090), RTX 4090, was used. CUDA 13.3.73, GCC 15.2,
CMake 4.2.3, pinned XGrammar `xgrammar-0.2.8-strata-budget1`, Coder IQ1_M,
INT8 KV, context 4096, prompt cache 2, conversation cache 0, adaptive swaps 0.
The explicit expert-cache request of 6000 yields the same 7815 resident expert
slots / 15233 MiB in every mode; PCIe fraction is 0.55 and the pool has five workers.
MTP uses maximum 4. Suffix lookup 3 retains MTP maximum 4 and permits verifier
windows up to 6 under the existing policy. Coupled mode is exercised by sampled
requests, including penalized Chat benchmark requests. Official Python SDK: 3.23.0.
Windows Python adapter checks use Python 3.13; native model probes use Linux/Python 3.14.4.

Final native executable SHA-256:
`b335d98cf16d08795d2e089e12484c3c7212c7b5b85755676b76aab113ba7bbe`.
[Environment and binary identities](native/environment.json),
[uploaded source-byte manifest](native/source-manifest.json),
[checkpoint commands](native/checkpoint-commands.json),
[continued checkpoint commands](native/checkpoint-rest-commands.json),
[local commands](local-commands.json), [aggregate facts](summary.json).
The manifest hashes uploaded checkout bytes; receipts hash exact normalized Git
blobs. Raw head arrays are local scratch, not claimed public downloads. Their
[size/hash manifest](native/raw-scratch-manifest.json) and the complete
[derived numerical report](native/checkpoint/numerical-audit/result.json) are retained.
There are no document download links to those scratch binaries. Trailing spaces
in CMake logs and truncated console previews are trimmed when importing textual
evidence; complete structured JSON results and diagnostic content are retained.

```text
python -m unittest discover -s serve -p 'test_*.py'
python -m unittest tools.test_grammar_state_contract -v
cmake --build <CPU-build> --target grammar_speculation_test grammar_native_test grammar_inspection_test
ctest --test-dir <CPU-build> -R '^grammar_(speculation|native|inspection)_test$' --output-on-failure
# The same three native targets run under ASan/UBSan.
ctest --test-dir <GPU-build> -R '^(grammar_speculation_(split|one_block|old)|grammar_sampler_(test|one_block|old)|coupled_draft_test)$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 <GPU-build>/grammar_speculation_gpu_test
python tools/grammar_speculation_probe.py --config <mode-config> --mode <mode> --out <fresh-directory>
python tools/grammar_speculation_probe.py --config <mode-config> --mode <mode> --diagnostics --out <fresh-directory>
python tools/grammar_api_probe.py --config <mode-config> --mode <mode> --speculation-benchmark --out <fresh-directory>
python tools/grammar_numerical_audit.py --root <checkpoint-evidence> --out <fresh-directory>
python tools/target_only_probe.py --config <ordinary-config> --mode <mtp-or-suffix> --out <fresh-directory>
```

## Correctness gates

The full [Python suite](python-suite.txt) runs 258 tests: 253 pass, five optional
tests skip. Seven [state-contract tests](state-contract-tests.txt) pass. The
pre-existing test-file-handle ResourceWarning remains visible in the suite log.
Release and ASan/UBSan native tests pass, including the G1 grammar corpus and G4
inspection tests ([initial Release](native/cpu-budget-1.txt),
[sanitizers](native/fixed-parity-2.txt), [final commit-boundary cases](native/checkpoint-1.txt)).

The native prefix/commit test reports 2169 cases, plus stop-EOS flag and resource
accounting assertions. The real CUDA selector passes 1728 fixed-logit decodes
per sampler implementation, 5184 total, with captured/uncaptured, greedy,
sampled, penalized, 2/4/8-row, every rejection position, illegal/end proposal,
UTF-8 and one-token-left coverage. The same selector/matcher and production
commit decision are used; model logits and proposals in those tests are explicitly
synthetic. [Detailed kernel results](native/fixed-parity-details.txt) also record
907257 coupled history/counter checks. [CUDA memcheck](native/checkpoint-5.txt)
reports zero errors. These are tests of actual Strata native code, not handoff tooling.

| Gate | Evidence/result |
|---|---|
| G5-01: unreachable drafts | Native first/middle/last illegal proposals and UTF-8 boundary cases pass. A replacement row remains; no all-masked descendant is sampled. Real model runs also encounter blocked proposals. |
| G5-02: retained state | Every scripted rejection position passes; real cursor traces match generated and matcher lengths. Discarded proposal work remains charged. |
| G5-03: sources | All five configurations below pass. Existing suffix policy supplies actual lookup windows; no proposal injection or forced-lookup hook is used. |
| G5-04: budget/EOS | Caps 1 through 12, Unicode token boundaries, epsilon, end proposals, cancellation/draining and live continuation pass. Ordinary MTP/suffix probes now require exact live-prefix reuse and the cursor invariant too. |
| G5-05: fixed-logit parity | Identical retained token IDs, masks, penalty histories and position counters across reference/speculative synthetic proposals. |
| G5-06: native numerical limits | Each speculative mode matches the reference's 32 committed IDs on the three measured requests. Raw logits differ; exact deviations, legal scores, masks, histories, counters and approximate CDF boundaries are retained below. |
| G5-07: performance | 180 warm measurements, three workloads, JSON and SSE, six repetitions per mode/workload/view. Concurrent pairs and a separate lower-acceptance Unicode stress observation are retained. |
| G5-08: no fallback | Requested/effective INFO, active proposal sources and coupled windows are asserted. Fallback count is zero; no recovery implementation exists. |

| Mode | Native cases | SDK/API cases | Grammar windows | Reachable rows | Actual suffix windows | Active coupled windows |
|---|---:|---:|---:|---|---:|---:|
| target | 37 | 46 | 284 | 1 | 0 | 0 |
| mtp | 37 | 46 | 151 | 1,2,3,4 | 0 | 0 |
| coupled | 37 | 46 | 153 | 1,2,3,4 | 0 | 20 |
| suffix | 37 | 46 | 155 | 1,2,3,4,5 | 10 | 0 |
| suffix-coupled | 37 | 46 | 157 | 1,2,3,4,5 | 10 | 20 |

Each mode's `native-<mode>`, `http-<mode>` and `numerical-<mode>` directory under
`native/checkpoint` contains its full result and native log. The 230 SDK/API cases
include pre-header failures, both adapters' JSON/SSE, sequence/assembly checks,
partial UTF-8, concurrent isolation and disconnect/drain. There are 185 native
cases and 15 diagnostic model requests. The additional ordinary
[MTP](native/checkpoint/ordinary-mtp/result.json) and
[suffix](native/checkpoint/ordinary-suffix/result.json) runs each pass 11 cases,
including exact live reuse, same-seed repeat, cancellation and the semantic service.
All final checkpoint results use the executable hash above.

## Numerical evidence and measured latency

Each accelerated mode compares 32 committed positions over greedy literal,
sampled/penalized literal and sampled-alternative requests. No selected ID differs
in this sample; masks, histories and counters match exactly. Maximum full-vocabulary
logit deviation is 1.674224854 for MTP/coupled, 1.613348246 for suffix combinations.
Maximum legal-token deviation is 0.786769867 and 0.711673737 respectively. The
smallest observed greedy margin is 7.257278442. The closest reconstructed CDF
boundary is 0.024114251 from the position's uniform. CDF/penalty reconstruction
is explicitly offline float64, not an assertion of exact CUDA probabilities.
The raw arrays' hashes and every legal score remain attributable to the real
native rows. This does not establish universal same-seed or bitwise model parity.
No branch mass is inferred.

The following medians are native decode milliseconds from six warmed SSE requests.
All digit outputs use 20 model tokens; repetition outputs use 34, including the
end control. The [CSV](speculation-benchmarks.csv) also records JSON latency,
first-text-delta latency, prompt reuse, accepted/offered drafts and actual lengths.
`sampled-repeats` uses temperature 0.8, top-p 0.85, top-k 20, min-p 0.04, seed 434,
repeat 1.1, frequency 0.1 and presence 0.05 over 64 history tokens. The other two
workloads are greedy. No tracing or logit capture is enabled for these timings.

| Mode | Digits | Repeats | Sampled/penalized repeats |
|---|---:|---:|---:|
| target | 1242.85 | 1895.65 | 1894.60 |
| mtp | 890.55 | 1474.05 | 1465.45 |
| coupled | 880.10 | 1478.80 | 1468.80 |
| suffix | 885.55 | 1505.15 | 1510.05 |
| suffix-coupled | 879.70 | 1470.85 | 1474.60 |

These are narrow literal grammars on one fixed model/profile with warm caches;
no p95, general agent throughput or minimum speedup is claimed. Modes run
sequentially on the same GPU; JSON/SSE order alternates within each repetition.
Each workload also has one concurrent JSON/SSE pair through the existing FIFO;
those individual wall/first-delta latencies are in the SDK result files and are
not attributed from the shared native `last` diagnostic.

Slower cases are retained. On one traced 16-token forced-Unicode request, target
decode takes 1133.6 ms; MTP takes 1243.5 ms. This is a separate single stress observation, not a median benchmark. Legal
verifier-input draft counts exclude proposals removed by the grammar; `SPEC`
traces expose those blocked proposals separately. The measured faster literal
cases do not hide this loss from low proposal acceptance.

## Corrections, limits and next gate

The first CPU assertion exposed uncharged cheap/cached tentative steps. The
implementation now charges them and the unchanged resource assertion passes.
A build command named the suffix unit target in a build where that optional
target was disabled; its tool error is retained and corrected target commands
pass. The first expanded SDK harness shadowed the `concurrent` module with a
local result variable, stopping before its concurrency phase; its
[failed result](native/checkpoint/api-target/result.json) is retained. Renaming
the variable fixes the harness without weakening assertions. All five corrected
HTTP runs pass. Earlier preliminary binaries/runs remain separately identified;
only the `checkpoint` results above establish the final gate.

The qualified native hardware is Linux/CUDA/RTX 4090 with the pinned Coder profile.
Native Windows, HIP, multi-GPU, custom end controls and other precision/cache
combinations need separate qualification. G3 exclusions remain: no grammar with
tools, reasoning, custom stops, JSON Schema/JSON-object interfaces or custom Lark
tools. No new production matcher inspector, application executor or scheduler
is exposed. R4's real local Codex protocol loop used scripted model output;
native-model Codex tool skill and universal client compatibility are not claimed.
The user selected Mermaid/plain text; actual Code Visualizer integration is not claimed.

G5 is the final ordinary gate in this handoff. G6 recovery remains deferred and
requires a separate decision, classified failure sites and fault-injection evidence.
Nothing has been pushed or published. The [enablement guide](../../NATIVE_GBNF.md)
and [Responses guide](../../RESPONSES.md) explain the disabled defaults and the
commands/configuration that enable the features.
