# RTX 5080: current fork and pristine upstream 0.1.39

The fork measured 57.00 decode tokens/s against pristine upstream's 50.01 on
this neutral prose workload, and preserved the continuation prefix in both runs.

Measured on 2026-10-05, RTX 5080 16 GB, Ryzen 7 9700X, 64 GB RAM, Windows,
Huihui Qwen3.8 Flash Next IQ3_S/IQ4_NL. These are neutral prose measurements on
this machine. No model/quantization/context downgrade was made.

## Pristine upstream versus fork

Upstream `6f32ec0` was fetched and built without source edits. The fork is daily
`b3dc0f4`, with live-memory native mode. Fresh processes ran upstream–fork–fork–upstream;
each used 512 warmup outputs, two fresh 1,536-output requests and a 256-output
exact-prefix continuation. Their own Python engine wrappers/tokenizers were used.
Both admitted 2,088 GPU slots and 35,837 MiB RAM; these temporary fixed capacities
isolate throughput and do not replace the daily auto/live configuration.

The measured daily integration has identical native engine/build sources and
tokenizer to contribution head `54cb3c7`. Its native wrapper differs only by
Windows console suppression; generation, parsing and control methods match.
These measurements validate the native path, not the complete PR UI/backend
integration. The documentation commit carrying this report changes no engine code.

| Native configuration | Weighted decode t/s | Fresh prefill t/s |
|---|---:|---:|
| Pristine upstream | 50.01 | 1076.0 |
| Fork, live-memory enabled | 57.00 | 1258.9 |

Observed fork differences: decode +14.0%, prefill +17.0%. Repeat decode spreads
were 0.44% upstream and 0.40% fork. Sampled trajectories/routing
differ, so these bounded observations do not establish an exact causal or universal speedup.

The two pristine-upstream continuation requests each re-read 8,323 input tokens,
reusing 0; both fork continuations reused 8,322 and read the final token.
Continuation TTFT was 8.200, 8.088 s upstream and
0.026, 0.026 s fork. All requested output counts completed. Upstream
failed this exact-prefix reuse assertion; its source was preserved, not patched.
This is native continuation evidence, not complete chat UI or quality acceptance.

Settings: 64K context/int8 KV, prefill auto, spec 4/min-p 0.5, coupled draft off,
temperature 0.6/top-p 0.95/top-k 20/seed 7421, thinking off. Frozen profile SHA256
`da72b00c17afbcec2eac1af5c7ccd663025958e24829c629567437be71970656`. Fixed model buffers and native `--vision` remained;
the vision encoder, archive, MCP, profile saves and physical-pressure allocation
were excluded. The controller did not resize these fixed-capacity timing arms.
MSVC 19.44/CUDA 13/sm120 Release flags and the read-only llama dependency matched.
The unmodified upstream build passed 142 build steps and 5 CPU checks.

## Installed daily configuration

Unmodified auto/live config (RAM cap 42 GiB, headroom 5.5 GiB, reserve 1536 MiB):
three warm API runs were 52.0, 41.7 and 47.3 t/s; weighted decode
**46.66 t/s**. Fresh prefill was 1,369.0/1,402.4/1,203.8 t/s.
The initial 512-output request took 62.61 s including cold model/vision load;
its native decode was 38.3 t/s. It was excluded from warm aggregate throughput.

Natural sustained pressure during this API test shrank admitted RAM
36,995→34,711 MiB and GPU capacity 2,448→2,246 slots while outputs continued.
Native protocol remained live and 64K context was preserved. This auto fixture,
learned profile, loaded vision encoder and transition cost differ from the fixed
native fixture; 46.66 versus 57.00 must not be interpreted as an engine regression.

## Previous deployment and optional tuning

Separate old–new–new–old native comparison, same fixed fixture:
previous `fa16a6a` 0.1.38 **54.99 t/s**,
installed 0.1.39 **54.58 t/s**, observed difference −0.75%.
The predeclared 5% regression gate passed; old–old spread was
0.41%, new–new spread 3.94%. Old repeated fresh
output hashes matched; new repeated fresh hashes differed after identical warmup.
No output/quality parity or exact slowdown is claimed.

Optional `--coupled-draft` measured 52.44 versus
54.58 t/s and did not establish output parity. It was rejected
for deployment. Coupled draft is off by default; omission of
`--no-coupled-draft` does not enable it. Upstream `vram_elastic` and the resident
live-memory allocator are incompatible, so the daily service keeps one live
capacity owner instead of enabling conflicting options.

## Acceptance and preservation

Configured-default gates passed: complete bounded outputs, stable fixed admission,
live protocol/64K context, continuation reuse, daily warm decode ≥40 t/s and ≤5%
observed old/new aggregate regression. This is throughput/functional evidence,
not a code-review capability score, perplexity/quality study or long-duration stress test.
The unchanged installed daily integration already passed the 389-test backend suite
(5 optional tokenizer tests skipped), 31 frontend and 12 native checks plus
real-model prefill/STOP/live shrink/regrowth acceptance in the
deployment task. Those backend/frontend counts belong to the daily integration,
which has separate archive/attachment/UI changes; they are not full-PR test counts.
New pure-upstream CPU checks and measured receipts add to those
checks rather than rerunning unchanged suites.

Daily service was restored to auto/live lazy loading, 300 s idle unload, 64K and
unchanged model/config/binary. The stale BUILD.json version/commit/hash fields
were corrected to the measured 0.1.39 artifact, with previous bytes retained.
SQLite integrity and all saved session payloads/counts matched the deployment backup
(7 sessions, 285 messages, 2 metadata rows). No private chats were used in a benchmark.

Exact source/artifact identities, phase counts, native clocks, reuse and token-ID
hashes are in [measurements.json](measurements.json). Machine-local raw receipts,
logs, harnesses and build provenance remain in the original contributor's private
evidence directory; no model/profile binary or
private conversation is included in this Git record. This documentation update
adds benchmark evidence to the live-memory contribution; it publishes no model weights.
