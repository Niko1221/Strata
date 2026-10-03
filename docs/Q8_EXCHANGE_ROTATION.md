# Q8 expert exchange buffer experiment

Published branch: `experimental/rtxpro-q8-buffer-ownership`.
Development branch: `perf/q8-exchange-buffer-rotation`.
Target: llm-60, NVIDIA RTX PRO 6000 Blackwell Workstation Edition **96 GB**,
Ryzen 9 7950X, 128 GB installed RAM. Full Unsloth Q8_0, FP16 KV.

**Status: model comparisons completed. Plain decoding and MTP improved in two
paired runs with exact outputs and matching work counters.** N-gram and combined
editing outputs matched and are included as single-pair measurements. Coding
outputs diverged; the trace follow-up below found verification schedules
changing first. All fixture and memory checks passed.

## The change

The resident RAM mode keeps the experts absent from the GPU cache. Promoting
expert A frees its RAM slot; evicting expert B first copies B from VRAM into a
temporary host buffer. The original commit copies B again into A's former RAM
slot. This experiment adopts B's temporary buffer as its resident storage and
recycles A's former RAM slot as the next eviction buffer.

Enable with `STRATA_EXCHANGE_ROTATE=1`. Unset or `0` retains the original copy
path. The RAM and exchange allocations keep their original lifetime owners;
expert ownership moves by atomic slot IDs into an immutable table of host and
CUDA device addresses. Background router lookahead reads those IDs safely. The model
weights, resident expert count, cache replacement policy and quantization stay
the same. The GPU-to-host eviction transfer is still required.

This first implementation accepts **equal-size expert blocks with the entire
allocated resident RAM region pinned and mapped**. It can cover only part of
the GPU cache complement; other experts may remain on the file tier.
Partial pinning of that RAM allocation, pageable memory and
mixed block sizes explicitly fall back to the copy path. The maximum exchange
capacity must be reserved before rotation is initialized; later growth is
rejected because the original exchange allocation may contain live experts.
Existing serve/generate callers reserve their maximum once before inference.

## Correctness contract

- The D2H eviction finishes before the evicted expert is published for readers.
- Existing CPU completion and H2D completion waits precede ownership commit.
- Every resident expert and spare has exclusive ownership of one buffer.
- CUDA aliases move with host pointers. No alias is reconstructed from a stale
  offset into the original RAM allocation.
- Staged overrides, file fallback, `blob`, `copy_blob`, prefetch, pin checks and
  device address lookups resolve the same committed storage.
- No transfer wait, residency publication order, stream lifetime or kernel
  arithmetic changes. Invalid commit metadata fails closed.

The counter `host memcpy bytes avoided` counts payload bytes. Avoiding an N-byte
copy removes an N-byte CPU read and N-byte CPU write at the software level; it
does not itself establish measured DRAM traffic or an end-to-end speedup.

## Validation and measurement

`exchange_storage_test` exercises 12,304 ownership transitions, odd sizes,
unaligned host addresses, actual Q8-sized blocks, distinct device aliases,
repeated reuse, invalid commits, guards and exact bytes. Run under ASan/UBSan.

`file_expert_source_test --rotation-gpu` exercises the real source API and CUDA
transfers with copy, rotation, and pageable fallback: staged reads, promotion,
eviction, host copies, CUDA alias reads, disk fallback, growth rejection,
close/reopen and byte equality. Run under Compute Sanitizer memcheck.

`tools/run_q8_exchange_rotation_fleet.py` joins the private fleet GPU queue:

1. Component tests, source API tests and GPU memory checks.
2. Old binary versus new binary with rotation disabled, 8K input/256 output.
3. Rotation off/on for serial, MTP, n-gram and combined modes, **64K input +
   1,024 output**, 73,728 allocation, FP16 KV, coding then editing requests with fresh prompt state. The adaptive expert cache
   persists between those two requests, identically in the comparison arms.
4. Fixed 16,400 GPU slots for serial/n-gram and 16,192 for MTP/combined; automatic
   CPU/PCIe split, 96 adaptive swaps, completion waits enabled, ESP disabled.
5. Exact token IDs, first divergence, resource telemetry, copy counters and
   unprofiled decode speed. Reverse-order repeat for any exact-output candidate
   with a first-run task improvement of at least 3%.

A token mismatch prevents an exact-output speed claim for that pair. It does
not by itself establish a model-quality regression. The storage fixtures and
model comparison answer different questions. Editing is assessed independently
from coding; its matched results are reported with their single-pair and cache
history limitations. Public serving is unaffected.

## Initial validation, 2026-10-03

Engine source `1a50d913bf910a1f63fbc1a0788a7083e3ca5f8c` built successfully on
llm-60 (CUDA SM120 Release). Binary SHA-256:
`d14ed6b69a1814ce4b5c08932a47d6921a55fa0aa8dea50427ccf0782d1ad997`.

The standalone ownership fixture passed all **12,304 exchanges** under
AddressSanitizer and UndefinedBehaviorSanitizer, including concurrent metadata
readers, exact resident bytes, alias pairs, buffer reuse and malformed commits.
The CPU fixture and compiler overlapped the separate retrieval quality
check, with bounded memory/CPU use; that check's timing is not a controlled speed
comparison. No GPU benchmark overlapped the compilation.

Evidence: `~/fleet-downloads/rtxpro-exchange-cpu-check-20261003` and
`~/fleet-downloads/rtxpro-exchange-build-ahead-20261003`.

The real CUDA source-API fixture subsequently passed all 64 exchanges in each
of three configurations: pinned copy, pinned rotation, and requested rotation
with pageable fallback. Exact bytes, CUDA alias reads, staged transfers, disk
fallback and close/reopen checks passed. Compute Sanitizer memcheck reported
**0 errors**. These source-API fixtures use 1,382,400-byte blocks, so their commit
timing is not a full-model Q8 speed measurement. The CPU fixture separately
includes Q8's 5,222,400-byte expert blocks.

The old `080891d` binary and the new `1a50d91` binary with rotation disabled
produced identical 256-token outputs at 8K input. The model comparison below
enables rotation at 64K input using the model's native RoPE settings, without
YaRN extension. Evidence:
`~/fleet-downloads/rtxpro-q8-exchange-rotation-20261003-r2`.

## Model results, 2026-10-03

Same hardware, full Unsloth Q8_0 and FP16 KV, native RoPE, **65,536 input +
1,024 output tokens**, 73,728 allocated context. Each pair differs only by the
rotation setting. Controls and candidates run in opposite orders in the repeat.
All requests reached the output budget; startup is excluded from request time.

| Mode/task | Copy, first tok/s | Rotation, first | Copy, repeat | Rotation, repeat | Paired decode gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain coding | 70.07 | 75.39 | 70.00 | 75.64 | +7.6%, +8.1% |
| Plain editing | 57.51 | 61.93 | 57.43 | 62.06 | +7.7%, +8.1% |
| MTP coding | 112.62 | 131.87 | 107.49 | 132.53 | +17.1%, +23.3% |
| MTP editing | 88.81 | 107.64 | 86.73 | 107.56 | +21.2%, +24.0% |

These eight task pairs matched every output token. Cache hit counts, lookups,
RAM blob reads, proposed drafts and accepted drafts also match within each pair.
Two pairs are evidence for these workloads, not a confidence interval or a
universal speed claim. The MTP copy control varied between runs; both results
are reported.

Including prompt processing, effective output throughput improved by 3.6-3.9%
for plain coding, 4.2-4.3% for plain editing, 5.8-8.0% for MTP coding and 8.4-9.3%
for MTP editing. The 1,061 resource samples recorded no foreign GPU process and
at least 22.88 GiB available host RAM; maximum sampled GPU allocation was
94,025 MiB. The private unit disabled swap and no allocation error occurred.

The plain request pair avoids 5,354 copies / **27.96 GB** of memcpy payload;
MTP avoids 7,247 copies / **37.85 GB**. These are cumulative across coding and
editing, not per token. Rotation keeps the GPU eviction and promotion transfers.
It removes the subsequent RAM-to-RAM copy. It does not duplicate all experts in
RAM, change weight values, or skip synchronization.

### Suffix paths: results with unresolved coding divergence

| Mode/task | Copy tok/s | Rotation tok/s | First differing output index |
| --- | ---: | ---: | ---: |
| N-gram coding | 70.64 | 76.31 | 415 |
| N-gram editing | 88.42 | 108.14 | None |
| MTP + n-gram coding | 107.91 | 131.06 | 95 |
| MTP + n-gram editing | 88.30 | 106.45 | None |

The original full-mode gate did not select these paths for reverse-order
performance repeats because coding differed. The editing pairs did match and
are accepted as the single-pair results reported above. N-gram coding changes the
docstring word `expiry` to `deadline`; later differences are not counted as
independent events. Window counts and draft choices differ. Timing-driven draft
policy decisions are a hypothesis for the divergence, not a proven explanation.
The preceding coding requests also leave different adaptive cache histories
for editing; a separate editing-only placement follow-up removes that history.

`tools/trace_q8_exchange_policy.py` records window position/width and ordered
expert IDs with the frozen engine, including a copy-versus-copy repeat. Its
instrumented rates are not used as throughput claims. The existing serving
routing dump stores placeholder weights; it cannot establish route-weight,
logit-margin or committed-state equality.

### Suffix trace follow-up

Completed with the same frozen `1a50d913` binary: native 65,536-token input,
512 output tokens, 73,728 allocation, Q8_0 and FP16 KV, coding only. Each mode
ran copy, rotation, then a second copy from fresh engine state. All six reached
512 outputs without file-tier expert reads. These instrumented runs are not
new speed measurements.

| Comparison | First different window (zero-based) | Position; copy/candidate width | First different output token |
| --- | ---: | --- | ---: |
| N-gram: copy vs rotation | 164 | 65,700; 2 / 1 | 412 |
| N-gram: copy vs copy repeat | 165 | 65,702; 2 / 4 | 412 |
| Combined: copy vs rotation | 23 | 65,578; 2 / 4 | 95 |
| Combined: copy vs copy repeat | None | Same schedule | None |

Ordered expert IDs match before the first batch-schedule difference in every
comparison. The n-gram copy-only repeat shows that output divergence is not
unique to buffer ownership. This is consistent with timing-driven draft policy
changing verification shapes; it does not establish the exact numerical or
state cause after the schedules differ. No route-weight, logit-margin or
committed-state trace was captured. The 512-token trace and original 1,024-token
benchmark have different output budgets, so their first differing indices need
not match.

[Machine-readable trace summary](../bench/results/2026-10-03-q8-buffer-rotation/policy-trace.json).

### Fresh editing placement follow-up

With ownership enabled, a separate configuration-only comparison ran **editing
alone from a fresh engine** in each arm. This removes the preceding coding
request's cache history. Q8_0, FP16 KV, native 65,536-token input + 1,024 output,
73,728 allocation, MTP + n-gram, 16,192 GPU expert slots, 96 adaptive swaps and
the same frozen binary. CPU-only miss execution means `--pcie-frac 0`; GPU
resident experts still execute on the GPU.

| Pair | Auto miss placement tok/s | CPU-only misses tok/s | Decode gain | Effective gain |
| --- | ---: | ---: | ---: | ---: |
| Auto then CPU | 89.30 | 92.80 | +3.92% | +2.01% |
| CPU then auto | 89.36 | 92.90 | +3.96% | +1.65% |

All four requests completed their 1,024 outputs, both pairs matched every
output token, rotation activated, and expert file-read counters were zero.
Effective throughput includes prefill. These fresh-engine speeds are not
directly comparable with the earlier coding-then-editing ownership table;
do not multiply the two gains or promote CPU-only misses for every workload.

[Machine-readable placement summary](../bench/results/2026-10-03-q8-buffer-rotation/fresh-edit-placement.json).

## 1M YaRN ownership comparison

The user requested this comparison after accepting the 64K result, with both
positive and negative outcomes to be published. Same hardware and frozen
`1a50d913` binary as above; the only arm setting changed is
`STRATA_EXCHANGE_ROTATE=0` versus `1`. **MTP is on in both arms**, n-gram drafts
are off, Q8_0 weights and FP16 KV are unchanged, and ESP remains disabled.

- YaRN factor 4, original context 262,144; 1,048,576 allocated positions.
- Actual input 1,044,472, output budget 4,096, plus eight reserved positions.
- The same coding prompt, natural EOS allowed, fresh engine in each arm.
- Fixed 10,874 GPU expert slots, a 56 GiB resident RAM expert budget, PLE in RAM.
- Automatic CPU/PCIe miss placement, 96 adaptive swaps, completion waits enabled.
- A short 8K-input/128-output probe first confirmed allocation and rotation at
  the full 1M allocation. It is not a 1M-input throughput result.

| Ownership | Actual output | Prefill seconds | Decode seconds | Output tok/s | Total request seconds | Effective output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Off: original copy | 2,787, natural EOS | 1,041.56 | 67.23 | 41.45 | 1,108.95 | 2.513 |
| On | 2,787, natural EOS | 1,030.47 | 62.14 | **44.85** | 1,092.76 | **2.550** |

**Decode gain: 8.20%. Effective gain including prefill: 1.48%.** Startup was
98.80 seconds for the control and 97.42 seconds for rotation, excluded from the
request columns. This is one off/on pair; no reverse-order performance repeat
or confidence interval. The unreplicated prefill difference is not attributed
to the storage change. The 64K 17-24% result does not generalize to 1M as a
20% claim.

All 2,787 output IDs matched. Each run recorded 1,362,886 cache hits / 1,443,017
lookups, 101,586 RAM blob reads, 18,209 file blob reads, 2,264 proposed drafts
and 1,962 accepted drafts. Both executed 825 decode windows, averaging 3.38
committed tokens per window. The exact output-token digest is
`a230262da355dda856b2a7d1721789608fdaf416191722d11ab5d1ab5d2b4441`
(SHA-256 of the UTF-8 compact JSON token-ID array).

Rotation activated and avoided **12,924 host copies / 67,494,297,600 bytes** of
copy payload. The GPU cache held 52.89 GiB of experts; 11,513 expert blocks
occupied 56 GiB of pinned/mapped RAM, leaving about 10.64 GiB on the file tier.
The PLE table occupies another 50.664 GiB of RAM. This also confirms that the
ownership path can operate with file-tier experts: its allocated RAM region
must be fully pinned/mapped, rather than holding every expert absent from VRAM.

The logged 47,936.4 MB of file reads is **logical expert-file traffic**, not a
measurement of physical NVMe reads. The old generic resident-mode startup
message says "no file reads" even for this partial budget; the actual counters
above establish file-tier use. Different context allocation and expert residency
prevent treating 64K versus 1M as a pure RoPE or KV-kernel comparison.

Both requests completed without an allocation error. The private unit had
`MemorySwapMax=0` and finished with exit 0. Across 1,229 telemetry samples, minimum
available host RAM was 7.83 GiB and maximum sampled GPU allocation was 94,109 MiB;
there were no foreign-GPU samples. These are sampled values, not continuous maxima.

The historical **41.36 output tok/s** 1M result used `e359f448`, before ownership.
The new off control reproduced its complete output, prompt and recorded work
counters exactly, at 41.45 tok/s. The matched comparison in the table uses the
same new binary in both arms and provides the new 8.2% result.

Harness: `tools/run_q8_rotation_1m_fleet.py`, source
`dc37021c7f77050894ab92246ad9e61621a0be65`.
Evidence: `~/fleet-downloads/rtxpro-q8-rotation-1m-20261003`.
[Machine-readable 1M results](../bench/results/2026-10-03-q8-buffer-rotation/rotation-1m.json).

Machine-readable [result summary](../bench/results/2026-10-03-q8-buffer-rotation/summary.json)
includes exact source/binary hashes, per-request prefill and decode timings,
effective throughput, token-stream digests, counters and comparison results.
