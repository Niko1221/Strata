# BF16 conversation checkpoints and session files

`--session-state-codec bf16` stores conversation running-state copies as bfloat16 instead of float32.
For the tested model, this halves the state arrays in retained checkpoints, parked conversations and session files.
The option is off by default. The engine still computes on FP32 state; restoration widens a selected stored state
before computation resumes. Encoding is lossy, so restored values can differ from the original FP32 values.

The measured benefits are smaller session files, lower retained-state storage and faster parking. Continuing
parked resumes cost approximately 2–6 ms more in the tested workloads. At 32K context, BF16 with 8K checkpoint
spacing delivered comparable complete-turn latency to FP32 with the default 16K spacing.

The [public benchmark repository](https://github.com/sumomomomomo/strata-bf16-bench-artifacts) contains the harness,
raw JSON and engine logs. Its [manifest](https://github.com/sumomomomomo/strata-bf16-bench-artifacts/blob/main/MANIFEST.md)
maps results to claims, distinguishes historical runs and records provenance gaps.

## Usage and scope

Add the option to the engine arguments used by the server:

```text
--session-state-codec bf16
```

The default is `fp32`. Unknown codec names are rejected at startup.

The option governs newly captured retained checkpoints and newly written parked images and session files.
Every existing checkpoint is read using its own recorded codec, independently of the configured write codec.
Turning the option off does not recover precision already lost in a BF16 snapshot.

The codec covers GDN recurrence and convolution history, PLE history, and QSA indexer tails and dead keys.
Integer block positions and K/V storage are unchanged. Temporary transfers of working state remain FP32.
No model-computation kernel is changed, and checkpoint spacing and retention defaults remain unchanged.

## State size

These sizes follow from the tested model's geometry: 36 GDN layers, 12 QSA layers, state size 128,
48 value heads, 10,240 convolution channels, convolution width 4 and indexer key dimension 128.
They describe one full-model running state, excluding token lists, image metadata and K/V buffers.
Layer-split states contain only the arrays owned by their stage.

| Array | FP32 bytes | BF16 bytes | Encoding |
| --- | ---: | ---: | --- |
| GDN recurrence and convolution history | 117,669,888 | 58,834,944 | BF16 |
| PLE history | 368,640 | 184,320 | BF16 |
| QSA indexer tails | 18,432 | 9,216 | BF16 |
| QSA indexer dead keys | 6,144 | 3,072 | BF16 |
| Integer block positions | 48 | 48 | Unchanged |
| **Total** | **118,063,152** | **59,031,600** | **Approximately 50% smaller** |

A parked image carries its live state and retained checkpoint chain. A session SAVE selects the deepest retained
checkpoint rather than writing the whole chain. The file-size measurements below include one live state and one
saved checkpoint. State storage is largely fixed per checkpoint, while K/V storage grows with context; consequently,
the percentage saved in a complete file falls as the conversation grows.

## Implementation

### Representation and numerical behavior

[`state_codec.hpp`](../include/strata/core/state_codec.hpp) and
[`state_codec.cpp`](../src/core/state_codec.cpp) define `Fp32` and `Bf16` codecs.
FP32 copies the existing bytes. BF16 encoding rounds to nearest even; decoding widens the 16-bit representation
exactly into FP32. BF16 requires no per-array scale and retains FP32's eight exponent bits.

For values in the normal range that do not overflow on rounding, the relative rounding error is bounded by
2^-8, approximately 0.39%. This is a representation bound, not an output-quality guarantee. In the subnormal range
the absolute rounding error is bounded by half a BF16 subnormal step; sufficiently small values round to signed zero.
Values near FP32's maximum can round to infinity. Signed zeros and infinities are preserved; NaNs remain NaNs,
with payload quieting as needed. Tests cover these cases.

INT8 and FP8 codecs are outside this change. They would require separate range, scaling and quality decisions.

### Capture, parking and restoration

[`conversation_state.cpp`](../src/core/conversation_state.cpp) captures retained checkpoints under their requested
codec immediately, including layer-stage parts. It builds replacement arrays separately and publishes them only
after all copies succeed. A failed capture preserves the checkpoint's existing arrays and does not modify live state.

A coded device-to-host capture uses slices of at most `kStateCodecChunkFloats` FP32 values: 4 MiB of staging memory.
It encodes each slice directly into the stored destination, avoiding a full-size FP32 staging array.

Parking copies retained checkpoints into the destination codec. An already-BF16 checkpoint parked as BF16 takes
a byte copy, without widening and re-encoding. Host-to-host conversion uses `state_transcode_bytes` directly;
copies and conversions retain the existing progress reporting.

When a parked conversation resumes, its retained chain stays in the codecs it already holds. Only the selected
checkpoint is decoded into the session's FP32 buffers. Other checkpoints remain compressed and unmodified.
This applies to continuing resumes as well as rewinds: restoring the parked live state also requires a decode.
Mixed chains are supported, and stage split/merge preserves each part's codec identity.

A selected checkpoint restore uses at most a 4 MiB FP32 staging slice for the large arrays. Indexer tails and dead
keys are widened separately, totaling 24 KiB for this geometry. These are conversion temporaries, not a bound on
the memory of an entire disk RESTORE operation, which also reads and retains conversation data.

Allocation and transfer failures are reported through the existing serve/batch error paths. Capture publication is
atomic; restoration writes live buffers and is not transactional. The serve path treats a failed checkpoint restore
as fatal because live state may already have been partially written.

### File format and compatibility

[`conversation_file.cpp`](../src/core/conversation_file.cpp) writes version 1 when the image is entirely FP32,
preserving the existing format. An image containing coded state uses version 2.

Version 2 adds a `u64` codec value at the start of **each checkpoint record**, including the live-state record.
The file header's reserved fields remain zero. This supports mixed-codec files. Version 1 has no codec field and
is interpreted as FP32 throughout.

Codec-aware engines read both versions and reject unknown codec values. Older engines reject version 2.
The codec is not part of the model/configuration fingerprint: the flag selects how new copies are written, while
the file describes its own representation. Other model and configuration compatibility checks still apply.

Read limits retain the FP32 array ceilings. Checkpoint validation checks exact sizes against the checkpoint's codec
before writing session state. Split/merge and cross-codec serialization have unit coverage.

### Memory accounting

Park admission estimates the destination state at the write codec and includes bounded coding scratch.
SAVE admission accounts for the selected checkpoint copies, live-state storage, serialization buffers and any
additional widening needed when a coded checkpoint is written as FP32.

Cache accounting uses allocated buffer capacity, which can exceed payload size. Neither halving state arrays nor
reducing a serialized payload establishes an equivalent reduction in total process RAM or an equivalent increase
in conversations admitted by a fixed cache budget.

## Benchmark setup and provenance

All engine results below were taken on the same machine:

| Component | Configuration |
| --- | --- |
| Model GPU | Tesla V100-PCIE-32GB; engine runs were single-GPU |
| Other GPU | RTX 2080 Ti, not used by these engine benchmarks |
| CPU | 24 cores / 32 threads; recorded as `Genuine Intel(R) 0000` |
| RAM | 93 GB reported |
| OS / toolkit | Linux Mint 22.1, kernel 6.14.0-33-generic; CUDA 12.8 |
| Session-file storage | Samsung 990 EVO Plus 4TB, ext4 |
| Engine / model | Engine 0.1.43.1; Qwen3.8-Flash-Next IQ3_S |
| Context / K/V | Maximum context 262,144; INT8 K/V; 32,768 resident K/V tokens |
| Conversation cache | 16,384 MiB; 12 slots; 8,192 MiB minimum free RAM |
| Checkpoint defaults | `--prompt-cache 6 --prompt-cache-every 16384` |

Latency comparisons use the same `strata.head` binary for FP32 and BF16, with runs recorded at `e79abfc0` after
capture-time encoding was added. They pin `--adapt-swaps 0 --pcie-frac 0`, reverse arm order across two rounds,
and use two warm-ups plus ten measured repetitions per workload per round. Reported latency figures are medians
of the two round medians. The checkpoint-spacing sweep uses a fresh server per configuration and round.

Memory measurements use the binary recorded at `79955461`, fresh engine processes per arm, and the ordinary
benchmark configurations rather than the pinned latency configuration. The engine arguments differ in the codec.
Disk RESTORE uses nine repetitions per size; the first SAVE after prefill is recorded separately from later saves.

Recorded commits are working-tree HEADs at run time, not automatically proof of the source used to build a binary.
The artifact manifest identifies binary names and known provenance differences. Historical session-file results
come from `d178753d`, before lazy restoration and capture-time encoding; they are labeled accordingly below.

## Measured results

### Checkpoint spacing and turn latency

[`capture_codec.json`](https://github.com/sumomomomomo/strata-bf16-bench-artifacts/blob/main/results/capture_codec.json):
one fresh 31,813-token document followed by short turns. Each timed turn captures a retained checkpoint.
The retained chain contains three checkpoints at 16K spacing and five at 8K.

| Codec / spacing | Fresh prefill | Turn prompt phase | Complete turn |
| --- | ---: | ---: | ---: |
| FP32 / 16K | 21.717 s | 736.5 ms | 1.074 s |
| BF16 / 16K | 21.703 s | 687.5 ms | 1.027 s |
| FP32 / 8K | 21.796 s | 806.0 ms | 1.135 s |
| BF16 / 8K | 21.868 s | 721.0 ms | 1.054 s |

BF16 at 8K spacing delivered comparable complete-turn latency to FP32 at 16K in this workload. Changing spacing
from 16K to 8K added 69.5 ms to the FP32 turn prompt phase and 33.5 ms to BF16's. The corresponding same-arm
round-to-round prompt-phase spreads were 0–3 ms. This is evidence about the whole request, not isolated capture cost.

Complete-turn wall time includes parking, resumption and generation; it is not TTFT. Fresh prefill is measured once
per configuration per round. The turns generate about 24 tokens, so their wall times also depend on generation.

The sweep's `ok` field is not a valid recall score: its turn prompt discourages quoting the document and the
code-word check returns 0/10 in every cell. This table uses timing only; explicit recall is evaluated separately.

### Parking and continuing resumes

[`lazy_resume_head_parity.json`](https://github.com/sumomomomomo/strata-bf16-bench-artifacts/blob/main/results/lazy_resume_head_parity.json):
FP32 and BF16 at default 16K spacing. Context labels round the original document lengths.
The continuing-resume column uses `append_only`; the parking and request columns use `older_ckpt`.

| Original context | Continuing resume, FP32 → BF16 | Parking on rewind, FP32 → BF16 | Complete rewind request, FP32 → BF16 |
| --- | ---: | ---: | ---: |
| 32K | 49.25 → 51.48 ms | 142.13 → 90.78 ms | 0.329 → 0.283 s |
| 63K | 71.15 → 77.05 ms | 163.98 → 104.90 ms | 0.344 → 0.295 s |
| 121K | 121.98 → 127.53 ms | 198.33 → 101.25 ms | 0.432 → 0.333 s |

BF16 adds approximately 2–6 ms to continuing parked resumes while reducing parking time in these requests.
The result is specific to the tested conversations, cache settings and machine; it does not establish a universal
speedup or eliminate the cost of decoding a selected state.

### SAVE / RESTORE RSS

[`memory-fp32.head.json`](https://github.com/sumomomomomo/strata-bf16-bench-artifacts/blob/main/results/memory-fp32.head.json)
and [`memory-bf16.head.json`](https://github.com/sumomomomomo/strata-bf16-bench-artifacts/blob/main/results/memory-bf16.head.json):
observed RSS growth above the operation's baseline, in MiB.

| Context | First SAVE, FP32 | First SAVE, BF16 | Disk RESTORE, FP32 | Disk RESTORE, BF16 |
| --- | ---: | ---: | ---: | ---: |
| 32K | 353.2 | 180.5 | 350.3 | 406.5 |
| 63K | 354.0 | 181.2 | 805.1 | 861.3 |

First SAVE and later saves are different allocation states. Later saves reuse allocations and their median
increments are near zero or negative; combining first and later saves into one median obscures that distinction.
The roughly halved first-SAVE growth is observed behavior, not an allocator-level attribution to the 4 MiB codec scratch.

BF16 disk RESTORE grows RSS by 56.2 MiB more at both sizes. That is consistent with retaining a coded state alongside
FP32 state during restoration: one BF16 state is approximately 56.3 MiB. The experiment does not instrument individual
allocations, so this agreement is not an allocation trace.

`VmRSS` is polled every 20 ms. Pageable data can leave RSS, and sampling can miss short peaks; these figures are
observed increments, not guaranteed high-water bounds. Negative increments do not mean SAVE has no memory requirement.

### Parked allocation and steady RSS

The same memory rerun records four parked images at 32K:

| Metric | FP32 | BF16 |
| --- | ---: | ---: |
| Cache-accounted allocation | 3,727,103,432 B | 3,077,756,360 B |
| Steady RSS before parking cycle | 54,673.3 MiB | 54,325.6 MiB |
| Steady RSS after parking cycle | 57,314.4 MiB | 56,527.0 MiB |
| Within-arm RSS increase | 2,641.1 MiB | 2,201.4 MiB |

Cache-accounted allocation falls by 17.4% in this run. The two arms have different initial RSS, so their final RSS
difference alone is not the memory saved by parking.

The [allocation trace](https://github.com/sumomomomomo/strata-bf16-bench-artifacts/blob/main/logs/strata-trace.log)
shows pre-existing K/V-reuse capacity overhead: 184.7 MiB of payload occupied 184.7 MiB on a fresh park and 554.1 MiB
after reuse. `parked= bytes=` and `image_bytes` therefore represent allocated capacity, not simply payload.
Payload arithmetic must not be used by itself to predict the number of conversations a fixed budget admits.

### Session-file sizes

Historical size measurements from `d178753d`, in
[`save_restore-fp32.json`](https://github.com/sumomomomomo/strata-bf16-bench-artifacts/blob/main/results/save_restore-fp32.json)
and [`save_restore-bf16.json`](https://github.com/sumomomomomo/strata-bf16-bench-artifacts/blob/main/results/save_restore-bf16.json).
Their SAVE/RESTORE timing fields predate the current capture and restoration paths and are not quoted as current performance.

| Prompt tokens | FP32 file | BF16 file | Reduction |
| --- | ---: | ---: | ---: |
| 4,223 | 300,697,908 B | 182,634,820 B | 39.3% |
| 31,797 | 721,838,564 B | 603,775,476 B | 16.4% |
| 63,026 | 1,198,752,588 B | 1,080,689,500 B | 9.8% |
| 121,614 | 2,093,508,528 B | 1,975,445,440 B | 5.6% |

Each pair saves 118,063,088 bytes: two state reductions of 59,031,552 bytes, less the two version-2 codec fields
of eight bytes each. The fixed state contribution explains both the larger percentage at short contexts and why
even a very short conversation can have a large session file.

## Correctness and failure-path validation

The explicit `older_ckpt` and `deep_rewind` code-word workloads in `lazy_resume_head_parity.json` scored
20/20 per workload, context size and codec across the two rounds, at 32K, 63K and approximately 121K.
These runs include restoration from checkpoints captured directly as BF16.

The free-form `append_only` answer check is excluded from recall claims. In its 32K cell FP32 scored 0/20 while
BF16 scored 20/20; this is not sufficient evidence of a BF16 quality improvement.

Earlier disk-restore needle tests at approximately 121K scored 3/3 for each codec, at depths 10%, 50% and 90%:
[`needle-125000-fp32.json`](https://github.com/sumomomomomo/strata-bf16-bench-artifacts/blob/main/results/needle-125000-fp32.json)
and [`needle-125000-bf16.json`](https://github.com/sumomomomomo/strata-bf16-bench-artifacts/blob/main/results/needle-125000-bf16.json).
These are the historical `d178753d` disk path, separately from the later captured-checkpoint tests.

The PR's reported CUDA validation is 125/126 tests passing. `ple_parity` requires a missing model artifact and
also fails before this change. Reported fault-injection coverage includes 43 capture checks and 25 reporting checks,
including sanitizer runs. Relevant coverage includes:

| Test area | Coverage |
| --- | --- |
| State codec | Round-to-nearest-even, representable values, signed zeros, subnormals, Inf/NaN, overflow, direct transcode parity and unknown codecs |
| Session files | Version 1/2, mixed codecs, payload validation, read limits, cross-codec writes and FP32 serialization |
| Checkpoints and snapshots | Capture/restore, retained compressed chains, stage split/merge, memory accounting and invalid snapshots |
| SAVE admission and reclaim | Selected-checkpoint preservation, codec-aware estimates, widening cost and refusal under pressure |
| Capture/report failures | Destination and staging allocation failures, preserved prior state, recovery and reporting while allocation fails |

Default FP32 serialization was checked against the parent build. These checks protect representation and integration;
they do not establish unchanged model output after lossy restoration.

## Limits and remaining validation

- **Quality:** recall is established for the tested workloads. Logit KL, top-1 changes across tokens since resume,
  broader long-continuation quality and cumulative effects of repeated lossy restores remain unmeasured.
  This build's chat API does not supply the logits needed for those comparisons; an engine-level harness is needed.
- **Memory and capture:** retained bytes before the first park, isolated capture latency, capture during prefill,
  equivalent-history total RAM at 8K/16K spacing, and a memory-tight conversation that newly fits remain unmeasured.
  Chains tens of checkpoints deep are also untested; the default retention limit is six.
- **HIP:** shares the changed host engine code, but was not built or run for these results.
- **SYCL:** has separate conversation-state sources and does not implement the BF16 option. This PR adds the shared
  codec source to its build and passes FP32 to its SAVE preflight. The build and handling of BF16 files produced
  elsewhere remain unverified; its checkpoint validator still expects FP32 array sizes.
- **Windows and multi-GPU:** Windows remains untested. Stage split/merge has unit coverage, but all engine
  benchmarks here are single-GPU; real layer-split park/restore remains untested.

Keeping computation FP32 does not erase the perturbation introduced by encoding a stored state. The measurements
support an opt-in storage tradeoff on the tested system; they do not establish quality equivalence or justify
changing the default codec.

To stop writing new BF16 state, use `--session-state-codec fp32`. Codec-aware builds can still restore existing
BF16 files. An older build cannot read version 2, and widening a BF16 snapshot does not restore its original precision.

## Reproducing and interpreting the evidence

Start with the artifact [manifest](https://github.com/sumomomomomo/strata-bf16-bench-artifacts/blob/main/MANIFEST.md)
and [harness README](https://github.com/sumomomomomo/strata-bf16-bench-artifacts/blob/main/harness/README.md).
Reproduction needs a compatible engine build, model/pack/tokenizer files and local FP32/BF16 configurations;
machine-specific configurations are intentionally not included. Results contain recorded engine arguments and
hardware metadata where available. Latency comes from normal engine builds; RSS is sampled externally.

Historical eager-versus-lazy and stored-conversion comparisons remain in the artifact repository, rather than being
mixed into the current tables. Five historical files (`ab_convert*` and `cold_restore*`) lack recorded commit provenance.
The manifest also explains superseded RSS figures and the distinction between first and later SAVE operations.
Use the explicitly identified result files above when quoting this report.
