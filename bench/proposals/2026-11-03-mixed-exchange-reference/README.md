# Historical mixed-size exchange rotation patch (discussion draft)

Refs https://github.com/Niko1221/Strata/issues/5050.

This directory preserves the actual runtime changes tested in our earlier 0.1.43-based RTX 3090 experiment. Current upstream already supports mixed-size rotation; this is a **reference patch for review and reproducibility**, not a second implementation to apply to main. No production source or defaults are changed by this PR. Please keep this draft until maintainers decide whether these historical artifacts are useful.

## What the experimental patch changes

- `ExchangeStorage` records immutable host/device address pairs and physical capacities, initializes a separate spare bucket per expert block size, and validates capacity and staged-pointer ownership at commit.
- `FileExpertSource` selects a spare by layer/block size. After the existing GPU-completion wait, the evicted expert adopts the spare and the old RAM block becomes the next spare. The host RAM back-copy is removed in the rotation arm.
- `generate.cpp` passes layer information when selecting D2H and bounce buffers. Both experimental arms allocate the same three-bucket budget and CUDA mapping flags; `STRATA_EXCHANGE_BUCKETS` selects copy versus rotation.
- The original CUDA expert calculation kernel and existing synchronization remain in place. The overlay also retains path-count instrumentation used by the experiment.

`historical-bucket-rotation.patch` is an exact four-file diff between the saved experimental preimage and candidate. **Its preimage includes earlier 64-slot and timing overlays; it is not a pristine 0.1.43 checkout.** The SHA256 pairs in `provenance.json` identify the required files. Do not apply this patch to current main or assume that a release tag reproduces its preimage. The integration diff is supplied for inspection; a complete standalone engine rebuild is not supplied here.

`before-exchange_storage.hpp` and `after-exchange_storage.hpp` contain the exact standalone ownership header preimage and candidate. The four `exchange_*test.cpp` files contain the four original host tests. `python3 run_tests.py` verifies the header hashes, applies the header hunk to the preimage, compares it byte-for-byte to the candidate, then builds and runs the tests in a temporary directory. These tests model host/device alias pairs with separate host arrays; they exercise ownership, payload preservation, guards, invalid commits and concurrent residency lookup. They do not validate CUDA/DMA completion or GPU correctness. CUDA callers must finish all readers before recycling a view.

## Measured original experiment

Single RTX 3090 24 GB, 64 GiB guest, original expert kernel, 64 staging slots, observed CPU expert fallback zero. Expert cache 13,537 slots / 18,577 MiB and arena 16,415 MiB matched between arms. One frozen expert profile; temperature 0; MTP enabled; tail skip 0; PCIe fraction 1. Four fresh starts A-B-B-A, eight warmups, sixteen measured 512-token requests and four free-generation repairs (limit 2048). Fine-grained probes disabled for speed measurements.

| Workload | A copy token/s | B rotation token/s | Change | Mean E2E A → B |
|---|---:|---:|---:|---:|
| Code continuation, 512 tokens | 122.7 | 132.0 | +7.52% | 4.509 → 4.209 s |
| Chinese explanation, 512 tokens | 86.8 | 96.9 | +11.61% | 6.229 → 5.619 s |
| Retry-function repair, 1420 tokens | 90.7 | 100.1 | +10.40% | 16.533 → 15.072 s |

Token/s is total output tokens / total engine decode seconds. E2E is client request to complete stream. Four repairs stopped naturally, had identical output SHA256 and passed the same 14 custom checks. These are small, task-specific observations. They are not general coding benchmark scores or expected gains on current main.

## Memory and CPU tradeoffs

For distinct expert sizes D, logical slots N, and b max-size-only bounce slots, let S=sum(D), L=max(D). Copy payload is N*L; bucket payload is (N-b)*S+b*L; additional pinned host payload is (N-b)*(S-L). This excludes allocator, mapping and metadata overhead.

Here D={1,177,600; 1,305,600; 1,510,400} bytes. With 96 slots per size, payload rises from 144,998,400 bytes (138.28125 MiB) to 383,385,600 bytes (365.625 MiB): **+227.34375 MiB pinned host RAM**. For the original speed isolation, both arms reserved the larger budget. GPU cache capacity stayed fixed. Pointer validation and mapping maintenance consume CPU; all PCIe traffic and synchronization still occur. NUMA placement, pinning pressure, extra metadata and total CPU consumption should be measured per deployment.

## Later official-version findings and remaining work

Issue #5050 reports a separate official 0.1.55 experiment with **16 staging slots and nonzero CPU fallback**, not the configuration above. Its latest unprofiled observations were code 133.15→145.30 token/s (+9.13%) and Chinese 96.35→103.10 (+7.01%, sample ranges overlap). CPU-seconds/1k output tokens changed by -8.98% and +1.00%. Those results must not be pooled with this historical 64-slot experiment.

Later free-generation repairs also failed in both arms. In the latest run the unprofiled repairs each passed only 1/2; all instrumented and uninstrumented repairs combined passed A 5/6 and B 3/6. Timing, MTP and output stability remain under investigation; those failures are not attributed to rotation. Host perf identifies busy waits and lookahead as follow-up candidates, but this draft contains no lookahead or spin-policy optimization, and claims no measured benefit from them.

Before adopting any additional runtime change: interleave no-profiler A/B and A/A, hold capacity and initial profile fixed, validate DMA ownership and CPU routing, compare logits/output, and grade complete repairs. Current-main integration, CUDA/HIP/SYCL builds, CUDA sanitizer coverage and cross-machine validation remain open.

Package validation on 2026-11-03 (macOS, host-only): `python3 run_tests.py` passed the hash/patch round trip and all four tests (output in `host-tests.txt`). The first local packaging attempt failed to compile because the runner missed `/include` in its search path; that runner error was corrected before these passing results. No GPU execution was performed for this draft package. Individual historical requests and inputs are in `requests.csv` and `fixtures.json`.

Prepared with AI assistance; raw experimental evidence and failures were retained. This package is a discussion artifact, with no deployment or default-policy change.
