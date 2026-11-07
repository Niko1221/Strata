# Cache-copy validation: evidence and limits

This record supplements PR #4002. The executable engine code is unchanged from
`cfff8ec3ec1db4f1226d451cd2f8e73b47aa317f`; this addition contains a standalone CPU
ownership probe and sanitized measurements. It is not a passed production-stability
or benchmark-protocol certification.

## Reproduce the ownership probe

On Linux x86-64 with a C++20 compiler, from the repository root:

```sh
g++ -std=c++20 -O1 -Iinclude tests/core/conversation_cache_release_probe.cpp -o /tmp/conversation_cache_release_probe
/tmp/conversation_cache_release_probe
```

The observed output is in [release-probe.log](release-probe.log). All three cases
passed: eviction and clear returned the tracked 5 MiB to the allocator; `take()`
removed cache accounting but retained the allocation until the external owner was
destroyed. Each case constructs one snapshot with 1 MiB each in live state,
checkpoint, later-stage state, and K/V buffers. Five corresponding delete calls
were observed. Runs used one CPU at nice 19, without a GPU or model requests.

The probe replaces global new/delete in its own single-threaded process and tracks
allocations at least 64 KiB using a fixed table. It does not instrument production,
cover smaller/non-new allocations or parallel device resources, or establish that
an allocator immediately returns pages to the OS. Source fingerprints are in
[source-hashes.json](source-hashes.json). The same code was first tested against
the unmodified headers at cfff8ec3.

## Natural traffic, 30 October 2026

The deployed build was revision `71ec4933`, not cfff8ec3. Four RTX 4090s served
Qwen3.8-Flash-Next IQ3_S with parallel copies enabled. The cache budget was 96 GiB.
The recorder used numeric cache counters, process anonymous RSS, smaps_rollup and
cgroup memory. It made no generation requests, restarted nothing and attached no
probes. These observations are diagnostic evidence, not current-head validation.

| Time (UTC) | Cache GiB | Anonymous RSS GiB | Evictions | In flight |
|---|---:|---:|---:|---:|
|03:14:02|58.615|108.518|0|0|
|03:15:02|2.534|107.566|41|1|
|03:15:33|3.953|107.568|41|1|
|03:23:02|46.470|107.669|41|0|

The first observed eviction window reduced cache accounting by 56.080 GiB but
anonymous RSS by only 0.952 GiB. Subsequently the cache refilled by 43.936 GiB
while anonymous RSS increased by 0.103 GiB. This supports reuse of already-resident
memory. **Allocator retention remains an inference**, not an identified root cause:
there are no synchronized allocator free/in-use counters or a census of live
objects, and natural requests/restore can allocate or transfer ownership between
samples. Net cache changes are not exact free-byte measurements.

The second recorder stopped early because its local disk filled: 29 valid samples,
about 25 minutes rather than the planned 30. All sampled production identities and
configuration fingerprints were unchanged. Its sanitized numeric records are in
[natural-rss-samples.json](natural-rss-samples.json); no request text, credentials,
private hostnames, internal addresses or infrastructure paths are published.
An earlier complete 30-minute observation had 31 samples and 22 natural requests,
but no evictions, so it could not answer post-eviction behavior.

## Validation inventory

| Revision/scope | Observed result | Limit |
|---|---|---|
|cfff8ec3 CUDA 13/sm_89|Engine and snapshot-test compiled/linked; six selected CPU CTests passed, zero failures/skips|GPU inaccessible; unchanged build objects reused|
|cfff8ec3 CPU-only|StagePool configure/build/CTest passed|No model/device execution|
|cfff8ec3 Clang 20.1.2 TSan|10/10 host rounds passed, no warnings|Container first round passed, next initialization failed on seccomp personality denial; failure retained|
|cfff8ec3 CI|title, intake check and Metal link passed; sweep skipped|Compile/link is not Metal runtime validation|
|30d39763 controlled integration|A-F host/direct/batchsync/async-conflict and official-main/default-off matrices: 100 synthetic requests|Before shutdown fix/latest-main merge|
|30d39763 OFF / parallel / parallel+reuse|Each arm: 26 functional + 64 performance requests; full paired output/input hashes and counts matched|Sequential arms, one host|
|30d39763 parallel vs OFF|16-client aggregate throughput +7.8%; repeat TTFT 128K 0.926 to 0.349 s, 256K 2.387 to 0.620 s|Cold input/native decode approximately unchanged; no general speedup guarantee|
|30d39763 parallel+reuse|16-client throughput about 6.0% below OFF|Switch remains default-off|
|30d39763 fault injection|20 expected copy failures observed; clean-process recovery passed|Same-process follow-up returned 503; no self-healing claim|
|Historical CUDA fixtures/memcheck|Passed within recorded fixture scopes|Not final-head hardware coverage; harness/readiness failures retained separately|
|Concurrent soak|Parallel stopped at guard after 494 iterations/7904 requests; OFF completed one hour, 533/8528; another candidate hit guard|Two-hour and final-reference parity incomplete; not a passed soak|
|400-request ON/OFF ramps|Both grew; about 227.8 MiB snapshot per park matched initial RSS growth|Post-eviction residency unresolved; not parallel-only leak proof|

The detailed revision-bound measurements and failed-attempt descriptions remain in
the PR discussion. Older v0.2.5 mixed-configuration results are historical and must
not substitute for the controlled 30d39763 results or final-head validation.

## Review boundary

Keep the PR Draft. Current-head GPU/layer-split execution, model parity/performance
and sustained stability have not been rerun under the no-restart/no-load constraint.
HIP/SYCL builds and Metal runtime execution are missing. Backend checks required by
CONTRIBUTING.md have therefore not all been fulfilled. Please help review the
ownership/error paths and identify targeted backend checks needed before promotion.

No malloc_trim, allocator tuning, production debugger or injected diagnostic call
was used. Static review and the CPU probe do not exclude all live objects or prove
the PR unrelated to sustained-memory risk. The available evidence does not justify
adding a speculative memory-reclamation change to this patch.
