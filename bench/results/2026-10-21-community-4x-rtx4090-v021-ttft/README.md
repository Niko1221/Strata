# Four RTX 4090: v0.2.1.4 TTFT and cache-hit diagnostic

On this machine, upgrading source engine 0.1.45.3 to 0.1.46.3 increased the time to first token of a repeated 128K request from 0.933 s to 1.619 s. Each paired request reused the same prefix and read seven new tokens. Decode became faster, but the extra time before decoding outweighed that gain. Sixteen simultaneous short requests also lost aggregate throughput.

This is a same-day custom serving diagnostic, **not standard-v1**, a full model-quality benchmark, or a proposed engine fix. The root cause is unconfirmed. This PR changes only the report and measurements.

## Machine and settings

- Linux 6.8.0-100, glibc 2.39; 2 x Xeon Platinum 8380, 80 physical cores / 160 threads, 251.3 GiB RAM.
- 4 x RTX 4090, 24,564 MiB per card, driver 595.99.02, 200 W per card. Cards 0/1 and 2/3 share NUMA nodes; cross-pair traffic traverses the socket interconnect. No NVLink.
- Qwen3.8-Flash-Next GSQ-RCO IQ3_S, the same existing native pack and Q2_0 MTP assets in both arms. Weight repository revisions and full weight-file hashes were not collected in this diagnostic; no weights were downloaded or changed.
- GPUs `0,1,2,3`, layer boundaries `12,24,36`, 16 slots, four groups, prefill 2048, context 262144, int8 KV, resident 24576, MTP spec 4 / min-p 0.5, pictures enabled.
- `--prompt-cache 1 --prompt-cache-every 0 --prompt-cache-root 0`; conversation cache 98304 MiB / 1024 entries, minimum free RAM 32768 MiB; GPU reserves 1536 MiB; `--ple-io ram --expert-cache auto --pool-workers 0 --trim-stage-weights`.
- `STRATA_PIPED_CONCURRENT=1`; self-commit and the prior experimental slot-copy/admission/QSA flags absent. Authentication remained enabled.

The sanitized configuration in `results.json` preserves placeholders for local model/profile paths; it is not an executable launch command.

## Source and build provenance

The old source integration was `207fa05843fbec81e3a5fc5fe069ebda8ab0ff30`, with engine 0.1.45.3 and five Python service fixes. New upstream is the peeled `v0.2.1.4` tag, `9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`.

The new [native build tree](https://github.com/oscarlius/Strata/commit/1df49d07b413c415c926cf98ae9270bed73daf8a) retains those Python fixes. The [served tree](https://github.com/oscarlius/Strata/commit/1e0083effde9941606318166d66fc224b3af8626) adds the two-line Web cache-display guard. Its native and Python inputs are identical to the tested build tree. These service corrections do not alter native inference code. The old and new expert-profile and chat-template files have identical SHA-256 values.

Both builds use the same container image, CUDA 13 toolchain and native-expert options, targeting `sm_89`. llama.cpp is pinned to `3cf03257f219afbe7334045ff7c6a06ac68c627d`. New engine and vision builds, dynamic-library checks and CPU vision tests passed. Build work finished before either measured arm. The build resumed with eight workers after an operator interruption of the initial two-worker job; compiler flags were unchanged, and the main engine binary stayed byte-identical across that resume.

The running container's source and binary hashes were checked through its process root. Final engine SHA-256: `30fade64495dcd3b159e530e68301186c04641fd50434c9c0e8b1cfb94868eaa`; vision SHA-256: `afe94e6709c1eb3a54469c4d69b6aa03b54fe17947d08a567b69969e02a2447c`.

## Method

Each arm begins after a process restart and the same excluded 192-token warmup. The new arm was restarted after its functional checks, so those checks did not populate its performance cache. The client runs on the inference host, with environment proxies disabled and redirects rejected. No compilation ran during measurement.

Per arm: one warmup, then three repetitions of a short single request, 16 simultaneous short requests, 128K first/repeat and 256K first/repeat: **64 requests, 19 groups**. Short replies are limited to exactly 192 tokens. Long replies must return six expected lookup values. Input payloads, output hashes, output token counts, cache reuse and newly read tokens must match for a strict pair. The monitor only reads lightweight status/metrics during timed requests; binary hashing happens outside those intervals.

All 128 generated replies passed their checks. All old groups and 18/19 new groups had clean request accounting. Three repetitions give medians and variability, not confidence intervals. The experiment is sequential old/new, not interleaved A/B/A; other host activity was not exhaustively controlled.

## Results

Medians of the three valid pairs; higher TTFT is worse, higher throughput is better.

| Measure | Old 0.1.45.3 | New 0.1.46.3 | Change |
|---|---:|---:|---:|
| Short single TTFT, s | 1.291 | 1.367 | +5.9% |
| 16-client p50 TTFT, s | 3.526 | 4.743 | +34.5% |
| 16-client aggregate output / group wall, token/s | 256.07 | 208.17 | -18.7% |
| 128K first TTFT, s | 18.127 | 18.852 | +4.0% |
| 128K repeat TTFT, s | 0.933 | 1.619 | +73.6% |
| 128K repeat native decode, token/s | 199.8 | 238.3 | +19.3% |

Short single TTFT varied substantially: old 0.307/1.291/1.316 s, new 0.977/2.580/1.367 s. The single-request median alone is weak evidence of a regression.

The 128K repeated requests provide the tighter comparison:

| Repeat | cache_n (both) | prompt_n (both) | Old prompt_ms | New prompt_ms | Old TTFT s | New TTFT s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 130649 | 7 | 876.6 | 1555.2 | 0.940 | 1.619 |
| 1 | 130647 | 7 | 872.8 | 1551.9 | 0.933 | 1.616 |
| 2 | 130657 | 7 | 866.9 | 1566.1 | 0.929 | 1.630 |

This localizes the added wait to the engine's reported prompt phase, which includes work before decoding; it does not identify the responsible function.

### 256K exclusions

The new arm's second 256K first-read group was contaminated by additional requests or accounting uncertainty, and its cache/prompt counts differ from the old arm. Its following repeat might therefore use a different live/parked restoration path even though that repeat's own request count is clean. **Both 256K categories are INCONCLUSIVE for a three-pair claim.**

For transparency, the two uncontaminated repeated-request pairs observed TTFT 1.716 -> 2.864 s and 1.714 -> 2.897 s. They are not substituted for the planned three-pair result. No arm was rerun to obtain a favorable result. `results.json` preserves the runner's original comparison and a separate conservative publication comparison with the carryover exclusion.

## Evidence and reproduction limits

`results.json` contains each request's status, correctness flag, timing, usage, input/output hashes, and group accounting, plus the hardware/configuration and comparisons. `short-workloads.json.gz` contains the 52 public short-request payload references, deduplicated by hash; the original request sequence is retained in the data.

The 12 long-context payload references contain private network/path material and remain local. Their hashes, token counts, timings and six-value retrieval-pass evidence are public, but the exact long inputs cannot be reconstructed from this PR. A newly generated synthetic long prompt would be a different experiment and must not be assigned these hashes. This limits independent byte-for-byte reproduction of the long-context result.

Functional verification was separate: 26 new-version requests covered text/images, 16-way concurrency, long retrieval, a real tool round trip and Responses/Anthropic APIs. After the final restart and performance arm, three additional text/image requests passed. Missing/wrong credentials were rejected; correct credentials worked. Service regression evidence was 563 passes / eight optional-dependency skips. Those suites overlap other checks and are not summed into a unique coverage count.

No HIP, SYCL or Metal hardware execution, standard-v1 submission, complete model-quality assessment, root-cause fix, or claim for other hardware is included. Earlier 0.1.45 measurements are historical context only and are not part of this same-day pair.

Related issue: [#3560](https://github.com/Niko1221/Strata/issues/3560). This evidence does not close or fix that issue.
