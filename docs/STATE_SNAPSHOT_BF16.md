# Saved-state codec: bfloat16 running state in parked images and session files

Branch: `feat/state-snapshot-bf16`. One change: a conversation's **running state** can be stored as bfloat16
instead of float32, in parked conversation-cache images and in session files. Off by default; with the flag off the
written bytes are byte-identical to today.

Nothing here touches a kernel. The GDN recurrence keeps reading and writing its fp32 state exactly as now; only the
**copy** of that state that goes into an image or onto disk is coded, and the copy a restore puts back is widened to
fp32 before any kernel sees it.

## Why the state is the expensive part

`plan.hpp` derives it, and the numbers are geometry, not measurement:

```
include/strata/plan/plan.hpp:77-78
//   recurrence    (S, h_v, S) fp32  = 128 * 48 * 128 * 4 = 3,145,728 B per layer
//   conv history  (d_conv-1, C) fp32 =     3 * 10240 * 4 =   122,880 B per layer
```

Derived from `conversation_state_sizes` (`src/core/conversation_state.cpp:107-129`) at this model's geometry
(36 GDN layers, 12 QSA layers, `ssm_state_size` 128, `ssm_v_heads` 48, `ssm_conv_channels` 10240, `ssm_d_conv` 4,
`idx_key_dim` 128, `idx_block` 4, `NG_HIST` 9, `NG_HC_DIM` 10240):

| array | what it is | fp32 bytes | coded bytes | coded? |
| --- | --- | ---: | ---: | --- |
| `gdn` | 36 × (786,432 + 30,720) floats | 117,669,888 | 58,834,944 | yes |
| `ple` | 9 × 10,240 floats | 368,640 | 184,320 | yes |
| `tails` | 12 × 3 × 128 floats | 18,432 | 9,216 | yes |
| `dead` | 12 × 128 floats | 6,144 | 3,072 | yes |
| `block_pos` | 12 × `int32` | 48 | 48 | **no — indices, not values** |
| **one running state** | | **118,063,152** | **59,031,600** | −49.99% |

`ConversationCheckpoint` already holds these as `std::vector<uint8_t>` (`conversation_cache.hpp:26`), so the codec
writes into byte buffers that already exist. It is not a struct change.

Against the session-file example in `docs/DETAILS.md:1115` (63,025 tokens, 1,198,691,396 B written):

| | bytes | per token |
| --- | ---: | ---: |
| that file today | 1,198,691,396 | 19,019 |
| its state part (live + the deepest checkpoint, `session_checkpoints_to_save`) | 236,126,304 | 3,747 |
| its K/V part (includes the indexer's pooled rows) | 962,565,092 | 15,273 |
| **expected with bf16** | **1,080,628,292** | **17,146** (−9.8%) |

A parked image carries the whole checkpoint chain, not one checkpoint (`generate.cpp:7980` passes `checks`, while the
SAVE path passes only the deepest), so the saving scales with the chain: a parked conversation holding four
checkpoints plus the live state stores 590,315,760 B of state today and would store 295,158,000 B — **281.5 MiB back
per parked conversation**, in RAM as well as on disk.

And the reason the spacing is 16,384 in the first place: `checkpoint_save` (`generate.cpp:1508`) copies the whole
fp32 state, so a checkpoint costs 118 MB of RAM. Halving a stored state is what makes a finer spacing affordable.

## Why bfloat16 and not int8 or fp8

| codec | encoder to write | decoder | scale stored | saturation | per-value error |
| --- | --- | --- | --- | --- | --- |
| **bf16** | ~10 lines (shift + round-to-nearest-even) | **exists, exact** | none | none (same 8-bit exponent as fp32) | 0.39% max, 0.17% rms |
| fp8 E4M3 | ~120 lines + a scale policy | exists | per layer/row/head | required, 448 max | 6.25% max |
| int8 | ~60 lines + a scale policy | ~10 lines | per row/head | required | 0.23% of row max, tails arbitrary |

bf16 keeps fp32's exponent range, so there is no scale to choose, no value that can overflow, and no subnormal cliff.
The repo already relies on this exact property:

```
include/strata/kernels/ngram.hpp:134
/// One BF16 row -> 160 floats, EXACTLY: bfloat16 is the top half of a float32, so this is a
/// shift and a copy with no rounding anywhere. ... There is no scale, because BF16 values are the values.
void bf16_dequant_row(const uint8_t* row, float* out160);
```

int8 was rejected for the first cut because its error is relative to the *group max*: at a channel `max/RMS` of 8 it
is ~1.8% of RMS, at 30 it is ~6.8% and the small values are gone. fp8 was rejected because the repo has an fp8
**decoder only** (`src/kernels/ngram.cpp:163`); the PLE table's fp8 bytes ship pre-quantized in the GGUF and
`tools/ple_fp8_pack.py` only copies them with one global scale chosen offline for fixed data. State range is
per-conversation and unknown, so that scale policy would have to be written and characterised from scratch. bf16 is
both the smaller change and ~16x more accurate. fp8 can be added later as a second codec member, behind the same
flag, if the bf16 quality numbers leave headroom and 4,096-token spacing is wanted.

## Scope

**In:** the codec; the file format; `conversation_checkpoint_save` / `_restore` / `_validate`; parked images
(`conversation_snapshot_save`); session files; one engine flag; tests; this document.

**Out, and left alone:**

- the state the engine computes on is never coded, and no kernel is touched. What **is** coded is every stored copy
  of it: the image a parked conversation holds, the file a `SAVE` writes, and - since the lazy-restoration change
  below - the retained checkpoints of a resumed conversation, which keep the codec they were parked in instead of
  being widened at the mount. Denser checkpoints is a separate change with a different risk (it is on every turn).
- no kernel, no CUDA/HIP/SYCL code, no device-side encode. The touched files are host code; build CUDA and HIP
  anyway.
- **the SYCL port does not implement the flag.** `sycl/` keeps hand-ported copies of `conversation_state.cpp` and
  `conversation_snapshot.cpp` that predate this change and contain no codec code, and its `generate.cpp` has no
  `--session-state-codec`, so a SYCL build never writes coded state and passes `StateCodec::Fp32` to the SAVE
  preflight. It does share `src/core/conversation_file.cpp`, which reads the codec byte, so `src/core/state_codec.cpp`
  is in that build's engine list. What a SYCL build does with a file written bf16 elsewhere has not been run here
  (no oneAPI on this box); its own `conversation_checkpoint_validate` still expects fp32 array sizes, which is the
  check that would have to catch it.
- no int8/fp8, no per-channel scales, no rotation/Hadamard preprocessing.
- no disk tier, no chunking, no prefix objects. This PR only makes the state half the size; where those bytes go is
  a later change.
- no change to K/V quantization, `--prompt-cache`, `--prompt-cache-every`, or the defaults of anything.
- no server/Python change: `/slots/0?action=save|restore` reports `n_written`/`n_read` as it does now, just smaller.

## Design

### 1. The codec

New `include/strata/core/state_codec.hpp` + `src/core/state_codec.cpp`:

```cpp
enum class StateCodec : uint8_t { Fp32 = 0, Bf16 = 1 };   // Fp8E4m3 = 2 lands later, same interface

size_t state_codec_size(size_t fp32_bytes, StateCodec codec);       // 0 and unknown codes are refused
bool   state_encode_f32(const float* src, size_t floats, StateCodec codec, uint8_t* dst, std::string& error);
bool   state_decode_f32(const uint8_t* src, size_t floats, StateCodec codec, float* dst, std::string& error);
```

`Fp32` is a `memcpy`. `Bf16` encode is round-to-nearest-even on the top 16 bits; decode is a shift into fp32.
`block_pos` never goes through the codec (it is `int32` positions). Encode and decode must be range-safe on
Inf/NaN and on the one RNE edge that can carry a finite value into infinity — that case is a test, not a comment.

### 2. The codec travels with the checkpoint, and the file says which one it used

`ConversationCheckpoint` gains `uint8_t state_codec = 0`. It is a property of the bytes in that checkpoint, like the
state sizes already are. `conversation_checkpoints_split` (`conversation_cache.hpp:82`) must carry it into
`stage0` / `parts` / `rest`.

Session file header v1 is `magic[8] version:u32 header_size:u32 model:u64 config:u64 payload:u64 reserved:u64[2]`
(`src/core/conversation_file.cpp:804`). Use `reserved[0]` as `state_codec:u64`, and bump `kVersion` (line 48) to 2 on
write while accepting 1 and 2 on read; version 1 means `Fp32`.

Two reasons for that shape:

- an old engine reading a coded file must **refuse**, not misread. Bumping the version makes it say
  `session file: unsupported version 2` (line 1207). Without the bump it would read the half-size arrays and fail
  only later, on a size mismatch, with a confusing message.
- existing user files must still read. Accepting v1 keeps that.

The codec deliberately does **not** go into `SessionConfig` / `session_config_fingerprint`. A file says how its own
bytes are stored; the flag governs what the engine *writes*. Binding it into the identity would make a file unreadable
the moment the user turned the flag off, which is the opposite of what a self-describing file is for. An unknown codec
value in a v2 header is refused as `invalid`, before the payload is parsed.

### 3. Read limits stay the fp32 maxima

`conversation_session_read_limits` runs before the file's codec is known, and
`SessionReadLimits::max_state_bytes` is a ceiling. A coded file is smaller, so the fp32 ceiling is a valid superset —
leave it. The exact per-array size check belongs in `conversation_checkpoint_validate`
(`src/core/conversation_state.cpp:143-153`), which knows the codec:

```cpp
if (c.gdn.size() != state_codec_size(z.gdn, c.state_codec) || ...)   // today: != z.gdn
```

### 4. Where encode and decode happen

**Encode** — `conversation_checkpoint_save` (`conversation_state.cpp:209`) resizes the five vectors to the codec the
engine is configured with and copies the device buffers into them. Under a codec the destination is half-size, so the
state does not cross the device boundary as one fp32 block: `copy_state_out` (`conversation_state.cpp:55-67`) moves it
in slices of at most `kStateCodecChunkFloats` floats (4 MiB), each read into a reused scratch and coded straight into
the destination. **The transient is that 4 MiB scratch, not a staging copy of the whole state** - the only full-size
allocation is the destination vector, which the image or the file has to hold either way. The preflight that admits a
park is `conversation_snapshot_bytes` (`conversation_state.cpp:312-343`), which charges every state at the write codec
and adds `min(gdn, 4 MiB)` for the slice when the codec is not fp32. `session_save_peak_bytes`
(`conversation_file.cpp:1184-1199`) is the SAVE's own bound: the deepest checkpoint twice over, plus the widening on
top when the file will hold it wider than the chain does, plus the 16 MiB write buffer.

An array already under the target codec is copied, never coded twice: `convert_state` (`conversation_copy.hpp:89-108`)
takes the plain beating copy when `from == to`, so a chain that is already coded is not widened and re-coded on its
way into an image written in the same codec.

**Decode** — `conversation_checkpoint_restore` (`conversation_state.cpp:232`) writes into the session's own fp32
buffers (`ss.gdn_state`, `ss.ple_hist`, each owned QSA layer's `idx_tail` / `idx_dead` / `idx_block_pos`), so the
running state needs no destination temporary: `copy_state_in` (`conversation_state.cpp:69-81`) widens it a slice at a
time into the same 4 MiB scratch. The two indexer arrays are the exception - `tails` and `dead` are widened into fp32
vectors beside the coded originals (18 KiB and 6 KiB at the geometry below), because the loop that fans them out to
the owned layers reads fp32 either way. A coded restore therefore asks the heap for 24 KiB and one 4 MiB slice, and
that is the allocation whose failure the serve path reports as a refusal instead of letting it escape
(`strata/core/conversation_report.hpp`).

`SavedConversation::bytes()` / `ConversationCheckpoint::bytes()` count `capacity()`, so parked-image accounting
follows the smaller vectors with no change.

### 5. Cost, measured

See **Measured** below. The arithmetic was right: one running state is 118,063,152 B, bf16 stores it in 59,031,600 B,
and a session file holds the live state plus the deepest checkpoint in the chain (`session_save_checkpoints`,
`conversation_file.cpp:1202` - not every checkpoint the RAM chain keeps), so the saving is a multiple of 59 MB and
does not depend on the conversation's length.

### 6. Stored-to-stored conversion goes in one pass

`convert_state` (the host-to-host conversion a park and a mount do on stored state) first went through an fp32
scratch: decode a slice into it, encode the slice out of it. Neither direction needs that. bf16 to fp32 is a shift,
so the second operation was a full-size `memcpy`; fp32 to bf16 round-to-nearest-even reads one value and writes one
value, so the first operation was a `memcpy` in. `state_transcode_bytes` (`state_codec.hpp`) does the whole
conversion between the two byte buffers: fixed-size `memcpy` loads and stores, `bf16_of` for the rounding and
`bf16_bits_of` for the shift, so the rounding, NaN quieting, signed zeros, subnormal flush and the ceiling behave
exactly as they did. A `from` that is already the `to` copies the bytes that array occupies; a codec with no pair of
paths here is refused rather than treated as one of the two.

The chunking, the beats and the `STRATA_TEST_SLOW_COPY_US` hook stay at the slice boundary in `convert_state`, which
still builds the destination beside the source and swaps it in only once every slice has converted - a mount calls it
with one vector on both sides. `copy_state_in` / `copy_state_out` keep their staging buffers: those exist because a
coded array cannot cross the device boundary in the shape the engine runs on, which is a different problem.

The destination is still a sized `std::vector<uint8_t>`, so its bytes are still initialised before the conversion
writes them. Removing that would need a different buffer storage, not a different conversion.

## Measured

> **The harness is not committed yet.** Every script named below — `bench-state-codec/bench_convert_host.cpp`,
> `bench_ab_convert.py`, `bench_run.py`, `bench_lazy_resume.py`, its README and the raw JSON each phase wrote — is on
> the machine that took these measurements and is not in this pull request. The tables are the record until the
> harness is added; the recipes are in `bench-state-codec/README.md`, which is likewise local for now. Nothing here
> used an instrumented build: every millisecond is what a normal build reports.

**One pass instead of two** (same box, `d178753d` plus this change; the pre-change binary built from the same tree).

Host, allocation inside the timed region, one real conversation's state (29,417,472 fp32 values), both paths in one
binary, the two paths alternating who goes first so allocator warmth cannot favour either — 12 reps, median:

| conversion | before (scratch) | after (one pass) | change |
| --- | ---: | ---: | ---: |
| one whole checkpoint stored bf16 (4 arrays) | 36.4 ms | 28.4 ms | −8.0 ms, −22% |
| one whole checkpoint widened to fp32 | 44.1 ms | 39.9 ms | −4.2 ms, −9.5% |
| `gdn` alone, fp32 to bf16 | 33.8 ms | 28.1 ms | −5.7 ms, −16.7% |
| `gdn` alone, bf16 to fp32 | 44.7 ms | 40.7 ms | −4.0 ms, −9.0% |

Order bias, measured on a build where both paths are the same code and the medians therefore must agree: at most
2.3 ms on the largest pair, under 1.5 ms elsewhere. Every gap above is bigger than that. `bench-state-codec/
bench_convert_host.cpp` prints them.

Engine, both binaries in one session on one pair of conversations, alternating which build goes first, both arms with
`--adapt-swaps 0 --pcie-frac 0` (`docs/DETAILS.md:108`), 10 reps per operation, median of the two rounds:

| operation | before | after | change |
| --- | ---: | ---: | ---: |
| park, 63K/32K pair | 233.7 ms | 202.1 ms | **−31.6 ms, −13.5%** |
| resume, same pair | 328.0 ms | 307.4 ms | **−20.6 ms, −6.3%** |
| park, 121K/63K pair | 233.5 ms | 201.8 ms | **−31.7 ms, −13.6%** |
| resume, same pair | 374.4 ms | 354.7 ms | **−19.7 ms, −5.3%** |
| `SAVE` | 383.8-586.8 ms | 358.2-503.1 ms | positive, too noisy to quote |
| `RESTORE` | 386.2-770.0 ms | 389.1-684.2 ms | **unchanged within noise** |

Park and resume hold the same gap in both order positions at both size pairs. The sizes match the host arithmetic: a
park does one store-conversion per state in the image and a resume one widen per state, and the chain here holds 3-6.
A `RESTORE` is dominated by reading 0.6-2 GB off the drive and the D2H K/V copies; the one or two states it converts
getting 4-8 ms faster does not move it, and one round's −65 ms did not repeat in the matched round, so it is reported
as unchanged. `bench-state-codec/bench_ab_convert.py` and `results/ab_convert*.json`.

**The default path is still byte-identical, and still as fast.** Same alternation with `--session-state-codec fp32`
(`results/ab_convert-fp32.json`): both builds wrote 721,777,476 B files with sha256 `2f92a766f002...` in both rounds,
`cmp` clean. Timings, best of the two rounds per arm: `park` 237.1 vs 233.7 ms, `resume` 70.3-71.1 vs 72.3-72.9 ms,
`SAVE` 381.1-395.7 vs 387.2-505.2 ms, `RESTORE` 533.5-568.7 vs 540.7-568.8 ms. Nothing on that path calls a
conversion at all - `convert_state` takes its copy branch when the codec already matches - so this is confirmation,
not a surprise.

**Sampled memory did not move.** `VmRSS` windows, fresh engine per arm, same phase, median of 3: `SAVE` +240.8 MiB
before against +236.8 MiB after at 32K (+241.2 against +241.2 at 63K), `RESTORE` +351.0 against +350.2 (+804.6 against
+804.6), a park/resume switch +48.0 against +56.2 (0.0 against 2.8). That is what was expected: the scratch the patch
removes is `kStateCodecChunkFloats`, 4 MiB, not a running state.

## Files to change

| file | change | ~lines |
| --- | --- | ---: |
| `include/strata/core/state_codec.hpp`, `src/core/state_codec.cpp` | new: the enum, the size rule, encode, decode | 140 |
| `src/core/state_codec_test.cpp` | new: round-trip bound, special values, RNE edge, odd counts | 220 |
| `include/strata/core/conversation_cache.hpp` | `state_codec` on `ConversationCheckpoint`; carry it through the split | 15 |
| `include/strata/core/conversation_snapshot.hpp`, `src/core/conversation_state.cpp` | codec parameter on checkpoint save/restore/validate (default `Fp32`), sizes through `state_codec_size`, capture-threaded codec in `conversation_snapshot_save` / `_bytes` / `_validate` / `_restore` | 120 |
| `src/core/conversation_file.cpp` | `kVersion` 2, `reserved[0]` as `state_codec`, v1 read path, per-checkpoint codec on write, unknown-codec refusal | 70 |
| `src/program/generate.cpp` | `--session-state-codec`, help text, the codec handed to the capture and the mount, one start-up log line | 60 |
| `CMakeLists.txt` | two targets, the pattern at lines 108-119 | 6 |
| `docs/DETAILS.md` | the measured section, and the format note next to the session-file example | 40 |
| `docs/STATE_SNAPSHOT_BF16.md` | this document | — |

**~670 lines, ~220 of them tests.** No backend file is touched.

Added after the first measurements, from the parked-resume numbers above (section 6):

| file | change | ~lines |
| --- | --- | ---: |
| `include/strata/core/state_codec.hpp`, `src/core/state_codec.cpp` | `state_transcode_bytes`, and `bf16_bits_of` pulled out of `f32_of_bf16` so the shift has one home | 50 |
| `src/core/conversation_copy.hpp` | `convert_state` calls it; the fp32 scratch and the decode-then-encode pair go away | 12 |
| `src/core/state_codec_test.cpp` | byte-for-byte against the two-step path, both directions, ties / zeros / subnormals / infinities / NaNs / random words, odd counts, refusals | 85 |
| `src/core/conversation_snapshot_test.cpp` | widening one vector in place gives the same bytes as widening a buffer of its own, over every slice | 14 |

## Flag

```
--session-state-codec NAME   --serve: store a saved conversation's running state as fp32 or bf16
                             (default fp32; bf16 halves the state, restores widen it back)
```

`o.session_state_codec` beside `o.session_min_free_mib` (`generate.cpp:633`), parsed beside line 2107, help text beside
line 785, config key `session_state_codec` in the same style as `slot_save_path`. An unrecognised name is a start-up
error, not a silent default. Default `fp32` keeps every written byte identical to today.

## Tests

`state_codec_test` (new):

1. round trip of a ramp, a random block, and every bf16-representable value: decode(encode(x)) within 2^-8 = 0.39% relative
   of x, exactly equal for values bf16 already holds.
2. `+Inf`, `-Inf`, `NaN`, `±0`, denormals, and the largest finite fp32 — no crash, and the RNE-carry-to-infinity case
   asserted explicitly.
3. `state_codec_size` for 0 bytes, non-multiples of 4, and an unknown codec: refused.
4. `Fp32` is a byte-for-byte `memcpy` both ways.

`conversation_file_test` (extend):

5. write with `Fp32` → the file is bit-identical to today's writer (the byte-identical-default check).
6. write with `Bf16` → version 2, `reserved[0]` = 1, the state arrays are half size, the payload hash covers them.
7. read a v1 file → accepted, decoded as fp32. **This is the back-compat test; it is the one that protects existing
   user files.**
8. read a v2 file with an unknown codec value → `SessionError::invalid`, nothing allocated.
9. read a v2 file whose array sizes do not match its codec → refused.
10. `SessionReadLimits::max_state_bytes` as fp32 maxima still bounds a coded file (no over-allocation).

`conversation_snapshot_test` / `conversation_validation_test` (extend):

11. save with `Bf16` → `conversation_checkpoint_validate` accepts it and rejects the same image re-labelled `Fp32`.
12. save → restore → the session's fp32 buffers match the originals within 2^-8 = 0.39% per value, and `block_pos`
    bit-exact.
13. a checkpoint already coded is not coded twice (idempotence).
14. `bytes()` reports the coded capacity; `session_save_peak_bytes` charges more than two copies of the deepest
    checkpoint (the live state, its lists and the write buffer on top) and refuses when the `admit` callback does.
15. layer split: `conversation_checkpoints_split` carries `state_codec` into every part, and a stage image validates.

`session_save_reclaim_test` (extend):

16. SAVE with the flag on → the file is ~10% smaller, RESTORE succeeds, and the next token equals the fp32 run's.

End to end, on the box:

17. 63K prompt → `action=save` with the flag off and on; record `n_written`, `save_ms`, `restore_ms`; restore both and
    compare the generated continuation token-for-token against the fp32 file.

Added with lazy restoration (`conversation_snapshot_test`, `conversation_cache_test`):

18. park a chain of three coded checkpoints, mount it the way the resume path does (move the image's checkpoints into
    the active chain, no widen): every snapshot keeps its bytes, its size and its `state_codec`.
19. an append-only continuation captures its own fp32 checkpoint and leaves every older retained snapshot byte-for-byte
    as it was, same codec, same `bytes()`.
20. selecting an older coded checkpoint puts values within 2^-8 of the fp32 originals into the session's buffers
    (`block_pos` bit-exact) and leaves the retained snapshot coded — restoring does not permanently widen it.
21. a chain holding fp32 and coded snapshots together is estimable, parks under either codec, validates as a mixed
    image, restores, and is written to disk both ways: an all-fp32 write of it is still a **version 1** file, a coded
    write is version 2, and a mixed image on disk comes back with each checkpoint's own coding.
22. `session_save_checkpoints` keeps the checkpoint it selects as the chain holds it and `session_save_peak_bytes`
    prices the widening an fp32 file still has to do; `conversation_snapshot_sources` then writes fp32 bytes.
23. layer-stage split and merge carry a checkpoint's own coding and each stage part's separately, and a coded stage
    part restores into its stage's fp32 buffers (`conversation_cache_test` does the same split/merge host-only).
24. a retained snapshot naming an unknown codec, or whose arrays are not its codec's size, is refused before any
    state write — the session still holds what the last capture saw.
25. re-parking a mixed chain under a real `RLIMIT_AS` is refused with an error and not thrown, and leaves the active
    chain exactly as it was. (`conversation_checkpoints_widen` keeps its own pressure test.)

## Measured

One Tesla V100 32 GB (model) plus one RTX 2080 Ti (ComfyUI idle), 93 GB system RAM, Linux, CUDA 12.8, engine built
from this branch. Model Qwen3.8-Flash-Next IQ3_S, `--kv int8`, `--max-context 262144`, `--kv-resident 32768`,
`--conversation-cache-mib 16384 --conversation-cache-slots 12 --conversation-cache-min-free-mib 8192`. The two A/B
configs are the same `args` list with only `--session-state-codec fp32` / `bf16` differing (same pack, same model
files, same `--vision`), so the two arms' session-file identities match — each arm restored the other's file, shown
below.

The conversation is `tools/needle_bench.py`'s haystack (this repository's own docs and source, in a fixed order) with
a code word inserted at the midpoint, asked greedily through `/v1/chat/completions`. Same text in every run.

**63,459 tokens**

| | `fp32` (default) | `bf16` | change |
| --- | ---: | ---: | ---: |
| session file | 1,205,289,012 B | 1,087,225,924 B | −118,063,088 B, −9.8% |
| `save_ms` | 546.4 | 503.2 | −7.9% |
| `restore_ms` | 759.5 | 714.3 | −6.0% |
| next turn on the resumed state | 0.47 s, right word | 0.45 s, right word | — |

**32,650 tokens**: 734,789,228 → 616,726,140 B (−16.1%), `save_ms` 444.1 → 315.5 (−29%), `restore_ms`
468.5 → 400.3 (−15%). Short conversations gain the most, because the state in a file is set by the checkpoint
spacing and not by the length: a 37-token conversation still writes 236,738,756 B, of which two running states are
236,126,304 B, and coded the same file is 118.7 MB.

**What bf16 costs the state, measured on the file's own values** — no model, no GPU, so nothing but the codec is in
these numbers. bf16 encode and decode applied to the 29,417,472 GDN floats of the 63K conversation's live state:

| array | values | worst relative | rms relative | worst absolute | mean absolute | state rms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GDN recurrence | 29,417,472 | 0.3891% | 0.166% | 0.0623 | 4.6e-5 | 0.2598 |
| PLE history | 92,160 | 0.3890% | 0.166% | 0.0156 | 2.5e-4 | 0.3171 |
| QSA indexer tails | 4,608 | 0.3891% | 0.167% | 0.0156 | 1.0e-3 | 0.9855 |
| QSA dead keys | 1,536 | 0.3743% | 0.161% | 0.0076 | 9.3e-4 | 0.9835 |

The worst relative error sits just under the 2^-8 = 0.3906% bound, as it must: bf16 has eight significant bits. No
value's sign flipped, none became zero, none became non-finite. `block_pos` is int32 positions and is never coded.

For scale, the same 63K prompt through two separate engine runs, both fp32, both put on the bf16 grid so only the
run-to-run difference shows: max 1.641, mean 0.00228 on the GDN state. **The engine's own run-to-run difference is
26x the codec's worst error** — the adaptive expert tier and a shared PCIe link move the rounding, as
`docs/DETAILS.md` records for #152 and #410.

**Round trip.** A coded file restored by an engine started with `--session-state-codec fp32`, then saved again, is
bit-identical to the state the coded file held, in all five arrays (`gdn`, `ple`, `tails`, `dead`, `block_pos`).

**The default path is untouched.** With the flag off, a 63,459-token conversation writes 1,205,289,012 B at file
version 1, and that file is byte-identical to the one the previous engine build (this branch's parent, `a0116b27`, same
`args`) wrote for the same conversation: same 232-byte header, same token ids, same bytes in all five state arrays.

**Compatibility.** A version-2 coded file restores on an engine configured `fp32`, and a version-1 file restores on an
engine configured `bf16`; both resumed conversations answered the code word correctly. An engine that does not know
version 2 refuses the file and says so.

**Text, with the rounding pinned.** Left at its defaults the engine is not repeatable on this box — the adaptive
expert tier and a shared PCIe link move the rounding (`docs/DETAILS.md:106`, and `docs/DETAILS.md:108` names
`--adapt-swaps 0 --pcie-frac 0` as the way to get byte-identical repeats) — and the same fp32 file restored twice gave
different continuations. With those two switches pinned off, two separate engine runs of the same 32,650-token
conversation produced states that differ by exactly the codec and nothing else: `gdn` max 0.0620 / mean 4.51e-5,
`ple` 0.0178 / 2.4e-4, `tails` 0.0156 / 1.1e-3, `dead` 0.0076 / 9.3e-4, `block_pos` identical — the same magnitudes as
the codec's own error table above, with no drift left on top. One engine then restored both files, alternating, twice
each, and generated the same four continuations:

| continuation | fp32 state | bf16 state |
| --- | --- | --- |
| the code word | `da454b02b470` | `da454b02b470` |
| what the document is made of | `d2c0d2cde25b` | `d2c0d2cde25b` |
| count to five, name the word | `e00a9fc3f733` | `e00a9fc3f733` |
| a haiku about it | `cded6cce8fb4`, `ede0a43fecf2` | `ede0a43fecf2`, `ede0a43fecf2` |

Three of the four are byte-identical between the coded and the uncoded restore. The fourth — free creative generation,
the one place a small state perturbation can send the greedy path down a different line — varied **on the fp32 file
too**: the same uncoded file gave two different haikus in two passes, and the coded file matched its later answer both
times. So on this box the codec did not add variation that the engine's own fp32 path did not already have.

Without those switches pinned, text identity cannot be used at all: two fp32 runs of the same file differ by 26x more
in the state than bf16 does.

### Parked images, 4K and 128K, cold reads, and RAM

A second pass over both configurations, on the same box and the same build, with the harness in `bench-state-codec/`
— scripts, raw JSON per phase, and a README saying how each number was taken, none of it committed yet (see the note
at the head of **Measured**). Every number below is
the median of 10 measured operations after 2 warm-ups, from the normal build, with nothing instrumented.

`--session-state-codec` is read once at start-up, so each configuration is its own engine process from the same
binary; the two configs differ in the codec, the log path, and `slot_save_path` (which only enables `SAVE`/`RESTORE`
and is on for both). Same prompts, same files, same order.

**Session files, 4K and 128K added to the 32K and 63K above:** (the token counts differ from the first pass by a few
hundred — that table is one hand-built conversation, this one is the harness's own fixed prompts, and the harness's two
arms get identical token counts because they send identical text)

| tokens | file `fp32` | file `bf16` | saved | `save_ms` | `restore_ms` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,223 | 300,697,908 B | 182,634,820 B | −39.3% | 341.0 → 185.2 | 347.0 → 128.9 |
| 31,797 | 721,838,564 B | 603,775,476 B | −16.4% | 538.9 → 336.2 | 737.0 → 420.3 |
| 63,026 | 1,198,752,588 B | 1,080,689,500 B | −9.8% | 547.1 → 457.9 | 915.3 → 701.6 |
| 121,614 | 2,093,508,528 B | 1,975,445,440 B | −5.6% | 916.5 → 906.7 | 1592.0 → 1388.7 |

Every saving is exactly two stored states — the live state plus the one checkpoint a `SAVE` writes — at 59,031,544 B
apiece, which is why the percentage shrinks as the conversation grows. `SAVE` writes version 1 configured `fp32` and
version 2 configured `bf16`; both engines read both.

**Cold page cache, separately.** `posix_fadvise(POSIX_FADV_DONTNEED)` on that one session file (no root, no global
cache drop, no other file's pages touched), then one restore, 3 repetitions: 721,838,564 B read in 443.4 ms configured
`fp32` against 603,775,476 B in 319.7 ms configured `bf16`; 1,198,752,588 B in 785.6 ms against 1,080,689,500 B in
468.4 ms; 2,093,508,528 B in 1423.5 ms against 1,975,445,440 B in 986.1 ms. Fewer bytes off the drive, less to wait for.

**Parked conversation images**, two conversations alternated so every turn parks one and resumes the other
(20/20 switches answered right in every size in both configurations):

| tokens | image `fp32` | image `bf16` | saved | `park_ms` | resume `ms` | first-turn `prompt_ms` | images per 16,384 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 31,798 | 1.414 GiB | 1.194 GiB | −15.5% | 142.2 → 136.5 | 52.3 → 186.1 | 252.9 → 381.2 | 11.31 → 13.40 |
| 63,027 | 2.192 GiB | 1.862 GiB | −15.0% | 211.3 → 200.5 | 86.5 → 300.4 | 357.3 → 561.1 | 7.30 → 8.59 |
| 121,615 | 3.189 GiB | 2.804 GiB | −12.1% | 94.7 → 154.4 | 125.8 → 248.9 | 280.7 → 464.2 | 5.02 → 5.71 |

The saving is the state arrays and nothing else, and the count comes out of the bytes: divided by one state's saving
(59,031,552 B) the three deltas are 4, 6 and 7 coded states, which is the image's live state plus its retained
checkpoint chain of 3, 5 and 6 - the last one being the `--prompt-cache 6` cap. Predicted and measured bytes agree to
8 bytes. This is also where the cost was, on that build (`2274c1c`, eager restoration): a resume widened every state
in the chain back to fp32, so resume went 52 → 186 ms at 32K and the first turn's `prompt_ms` 253 → 381 ms. Under a
fifth of a second, but it grew with the chain. **These two columns are superseded for the resume side by
"Lazy restoration" below**; the parked-image bytes and `park_ms` are unchanged by that change. At long context K/V
(15,271 B/token here) is 79% of an image, so the byte saving flattens toward −16%.

**RAM.** `VmRSS` sampled at 20 ms from outside the process, on a fresh engine, median of 3: `SAVE` +353.2 MiB
configured `fp32` against +240.8 MiB configured `bf16` at 32K (+353.8 against +240.8 at 63K). A bf16 `RESTORE` at 32K
is *bigger*, +415.2 against +350.8 MiB: widening coded state allocates the fp32 arrays the engine runs on beside the
compressed originals. At 63K both read +805 MiB, where K/V dominates the transient. Steady RSS with three
conversations parked 55,026 → 57,342 MiB `fp32`, 54,749 → 56,897 MiB `bf16`; the cache's own accounting 3,842,780,664 B
over 3 images against 3,370,528,248 B. Two limits on these numbers: `VmRSS` misses pages the kernel already wrote out
and most parked state is pageable, so they are lower bounds; and once the cache is at its budget every increment reads
zero, because the allocator already owns the blocks — which is why this runs on a fresh process.

**Encode and decode, host only.** The 117,669,888 B `gdn` array pulled out of a session file this build wrote for the
63,026-token conversation, then `state_encode`/`state_decode` with no GPU and no engine in the way: encode 16.2 ms
(7.3 GB/s of fp32 read), decode 10.0 ms (11.9 GB/s of fp32 written), 6 repetitions, encode 16.1-17.4 ms and decode
9.7-10.6 ms, worst relative error against the file's own values 0.3891% — inside the 2^-8 bound, and every value the
decoder produced was checked back against the input.

## Lazy restoration of the retained chain

**What changed.** Up to and including `2274c1c` the parked-resume path called `conversation_checkpoints_widen()` on
the whole checkpoint chain the moment a parked conversation was resumed: every retained snapshot was allocated an
fp32 array and converted, whether or not the conversation ever went back to it. That made the active chain uniform
fp32 — which is what this document said until it was rewritten — and it made the resume cost grow with the chain.

Now the mount moves the parked image's checkpoints into the active chain **as they are**, buffers and `state_codec`
bytes included. The chain is mixed from that point on: coded snapshots imported from parking beside the fp32 ones the
session captures itself. `conversation_checkpoint_restore()` decodes the **one** snapshot a rewind selects into the
session's own fp32 buffers and leaves the stored snapshot untouched, so the same checkpoint can be restored again,
re-parked, or written to a file in a third codec. `conversation_checkpoints_widen()` is kept — it is still the way to
put a whole chain in fp32 up front, and it is still tested — but the resume path no longer calls it.

Nothing else moved. Newly captured checkpoints are still fp32; the configured `--session-state-codec` still decides
only how the *next* parked image or session file is written, and every existing snapshot is still read by the codec it
names for itself; the file formats and the fingerprints are unchanged (an all-fp32 write of a mixed chain is still a
version 1 file — `any_coded()` looks at what the image actually holds); layer-stage split and merge already carried a
checkpoint's own coding and each stage part's separately, and still do; no kernel, no new flag, no new worker, and the
checkpoint interval is untouched.

**What it costs the reader.** The deferred decode allocates host memory the eager path did not: the widened
`tails`/`dead` arrays (18 KiB + 6 KiB at this geometry) and one fp32 slice of `gdn`/`ple` crossing the device
boundary, bounded by `kStateCodecChunkFloats` — 4 MiB. The serve loop has no handler around the restore, so
`checkpoint_restore()` in `generate.cpp` turns a `bad_alloc` there into the reported `ERR restoring a conversation
checkpoint failed: not enough RAM to widen the checkpoint's running state` the same way it reports any other restore
failure (the session may be half written, so the engine ends and the server starts it again), and the batch slot path
`copy_from_slot()` reports a refusal and lets the caller read the prompt from token 0 as it already did. Neither
handler may build its own message: assembling `"checkpoint restore failed: " + e.what()` wants a heap block on the one
path that has none, so a second `bad_alloc` escaped the catch and ended the engine. The message is a literal or
`what()` read while the exception is alive, and the copy into the caller's string is attempted inside a try
(`strata/core/conversation_report.hpp`); `conversation_report_test` arms the allocator to fail and checks both halves
of that. Re-parking a mixed chain is the unbounded allocation and it was already refused rather than thrown; that is
tested under a real address-space limit.

**Engine lines added.** `retained N checkpoints beside the live state, B bytes … (C coded, X bytes of it)` on the
mount, and `resumed through the checkpoint at T tokens (widened into the session's fp32 buffers; its stored bytes are
kept as they are) in M ms; the other K retained checkpoints were not touched` at the restore. The second is printed
only when the chain actually holds coded snapshots, so the default fp32 path is as quiet as it was.

### Files the two commits touch

| file | change | lines |
| --- | --- | ---: |
| `src/program/generate.cpp` | the mount keeps the parked chain as it is; the deferred restore's failure reporting; `checkpoint_at()` takes a stage part's coding from the part it moves in; two codec-gated log lines | +133 / −56 with the comments |
| `include/strata/core/state_codec.hpp`, `conversation_cache.hpp`, `conversation_snapshot.hpp`, `src/core/conversation_copy.hpp`, `src/core/conversation_state.cpp` | comments and the two engine strings: the active chain can be mixed, and only the working state is fp32 | 53 |
| `src/core/conversation_snapshot_test.cpp` | the mixed-chain tests listed above | 244 |
| `src/core/conversation_cache_test.cpp` | the split and merge carry each part's own coding | 54 |
| `include/strata/core/conversation_report.hpp` | the guarded report: a literal or `what()`, never an assembled message | 27 |
| `src/core/conversation_report_test.cpp` | the same, with the allocator armed to refuse requests of a given size | 230 |
| `CMakeLists.txt` | one host-only test target, the pattern at lines 108-119 | 4 |

No backend file is touched, and the only change outside a header or a test is host C++ in `generate.cpp`.

### Measured: eager (`2274c1c`) against lazy, both `bf16`

Same box and configuration as the tables above (Tesla V100 32 GB model GPU, RTX 2080 Ti idle, 93 GB RAM, Linux, CUDA
12.8 build, IQ3_S, `--kv int8`, 262144 context, `--conversation-cache-mib 16384 --conversation-cache-slots 12
--conversation-cache-min-free-mib 8192`, `--prompt-cache 6 --prompt-cache-every 16384`), with
`--adapt-swaps 0 --pcie-frac 0` pinned (`docs/DETAILS.md:108`). Two engine binaries from the same tree — HEAD
(`engine-cuda12/strata.eager-2274c1c`) and this change (`engine-cuda12/strata.lazy`) — each its own server process,
because both the codec and the executable are read once at start-up. **Two rounds, the arms alternating which goes
first; 2 warm-ups then 10 measured repetitions of each operation in every arm-session.** Every figure is the median of
the 20 measured operations, with the low and high of the two rounds' medians after it. `engine prompt_ms` is the
engine's own number for the prompt phase; `client wall` is the client's own clock around the HTTP request and includes
the queue, the park, the prompt and the decode — they are not the same measurement and are never mixed.
`bench-state-codec/bench_lazy_resume.py` (local, not committed yet), raw JSON `bench-state-codec/results/lazy_resume.json`
on the machine that ran it.

Actual token counts, identical in both arms: prefill **31,801 / 63,029 / 121,441** tokens; the continuing prompt
**31,821 / 63,050 / 121,462**; the edited prompt **19,090 / 19,091**. Retained checkpoints after the continuation:
**3 / 5 / 6**.

**Append-only continuation** — the conversation simply carries on, so the mount is most of the cost. One checkpoint is
still restored — the newest, at the length being resumed, `selected … 31821` — and under `bf16` that one is coded, so
this workload pays the single deferred decode too:

| tokens | resume `ms` | engine `prompt_ms` | client wall `s` | retained checkpoint bytes | parked image |
| --- | ---: | ---: | ---: | ---: | ---: |
| 31,821 | 174.3 (173.8–174.8) → **53.5** (53.0–53.9) | 411.4 → **284.9** | 0.64 → **0.52** | 354.5 MB → **177.4 MB** | 0.41 GiB both |
| 63,050 | 229.6 (229.0–230.2) → **76.8** (76.8–77.0) | 415.5 → **257.9** | 0.61 → **0.45** | 591.2 MB → **296.1 MB** | 0.41 GiB both |
| 121,462 | 266.4 (257.2–275.6) → **126.8** (126.3–127.3) | 420.4 → **276.5** | 0.67 → **0.52** | 710.6 MB → **356.4 MB** | 0.41 GiB both |

The lazy column is not zero because the mount still restores the image's *live* running state, which it always has:
53 ms at 32K, 77 at 63K, 127 at 121K is that restore plus the K/V. What went away is one conversion per retained
checkpoint. The eager byte figures are arithmetic on the engine's own count — `n × 118,063,152 B` plus the token and
image lists — the lazy ones are the `retained=` line the build prints.

**Older-checkpoint resume** — the prompt is an edit that ends at 19,090 tokens, well behind the conversation's live
state, so the mount and one checkpoint restore both happen (the engine's own line names the checkpoint: `resumed
through the checkpoint at 19083 tokens`, `reused: 19083`):

| tokens | resume `ms` | engine `prompt_ms` | client wall `s` | retained checkpoint bytes | parked image |
| --- | ---: | ---: | ---: | ---: | ---: |
| 31,821 | 124.2 (123.7–124.8) → **40.2** (40.2–40.3) | 305.8 → **197.4** | 0.39 → **0.28** | 225.3 MB → **118.2 MB** | 1.19 GiB both |
| 63,029 | 75.0 (74.5–75.6) → **39.4** (39.1–39.6) | 310.6 → **208.2** | 0.39 → **0.29** | 225.3 MB → **118.2 MB** | 1.86 GiB both |
| 121,441 | 71.5 (70.5–72.5) → **37.4** (37.2–37.5) | 318.5 → **201.4** | 0.44 → **0.33** | 225.3 MB → **118.2 MB** | 2.80 GiB both |

**Deep rewind** — the prompt ends at ~18,000 tokens and asks a different question every repetition, so no parked
copy's near-tail checkpoint matches it and the engine mounts through the periodic checkpoint at **16,384** tokens
every time. No parked image is mounted at all (`reused: 16384` in both arms, `resume` reported by neither): this is
the deferred restore on the active chain alone, with the park of the outgoing conversation in the same request.

| tokens | engine `prompt_ms` | `park_ms` | selected checkpoint |
| --- | ---: | ---: | ---: |
| 31,801 | 1372.5 (1371.8–1373.2) → **1353.3** (1350.3–1356.2) | 93.0 → **73.7** | 16,384 both arms |
| 63,029 | 1314.9 (1314.1–1315.8) → **1302.7** (1301.8–1303.7) | 59.7 → **39.5** | 16,384 both arms |
| 121,441 | 1312.2 (1311.5–1312.9) → **1296.0** (1292.9–1299.0) | 59.3 → **39.0** | 16,384 both arms |

**What the one deferred decode costs.** The engine times it: **14.6–17.2 ms** for the selected snapshot, the same at
32K, 63K and 121K and in all three workloads (the state is one running state whatever the conversation's length).
Against the 121–153 ms the mount no longer spends, one 15 ms decode is what every resume pays — a rewind through an
older snapshot and a plain continuation alike, because the newest checkpoint in a parked image is coded too and the
turn that continues still restores it (`selected … 31821` on the append-only rows). What lazy restoration removes is
the widen of every checkpoint the conversation never goes back to, not the decode of the one it does: that is the
whole of the +11 ms the continuing turn is behind `fp32` at 32K in the parity table below. The deep-rewind rows are
*faster* under lazy despite doing that decode, because the park in the same request copies a chain that is already
bf16 instead of converting an fp32 one.

**Quality on the same runs.** With `--adapt-swaps 0 --pcie-frac 0` pinned, the two builds answered **byte-identically
in every size and every workload**: the same single answer string per workload across all 20 measured repetitions of
each arm (`ember` / `velvet` / `tundra` for the two rewinds, and the same one-sentence continuation for the
append-only turn). Needle recall — the code word placed at 10% of each document — was **20/20 in every arm, every
size and every workload**, 120 measured continuations and rewinds. Two notes on that column: the append-only turn asks
for "one short sentence about it", so its check only bites when the sentence happens to name the word (on a separate
run of this same build configured `fp32` the same prompt produced a sentence that described the document instead, and
`ok` read 0/3 there while both explicit "what is the code word?" turns stayed 3/3) — the two rewind columns are the
recall checks, the append-only column is an answer-identity check. The resumed prompts came back with the same byte
counts in both arms.

**Sampled memory.** `VmRSS` polled at 20 ms from outside the process, increment above the steady baseline (median of
the last 5 samples), median of 10: append-only at 32K +529.8 MiB eager against +529.9 lazy, at 63K +261.7 against
+167.6, at 121K +0.0 against +0.0. Two limits, stated as they are in the RAM section above: `VmRSS` misses pages the
kernel has already written out and most parked state is pageable, and once the conversation cache is at its budget the
allocator owns the blocks, so several windows read 0 and some read negative. These are observed lower bounds, not
high-water marks; the retained-bytes columns above are the trustworthy figure for what the chain holds.

**What was not measured.** No HIP build (no `hipcc` on this box) and no SYCL build (no oneAPI): the header changes
here are comments and the only code change outside a header is in `src/program/generate.cpp`, which is plain host C++
shared by the CUDA and HIP builds and not used by the SYCL port, but neither backend was compiled. The layer-split
park/restore path is covered by unit tests and not by these engine runs, which are single-GPU. The KL / top-1-flip
curve across tokens-since-resume is still not available through this build's API, as recorded in the quality gate.

**The default path.** The same harness against this build configured `fp32` (`results/lazy_resume-fp32-default.json`,
one arm, 1 warm-up + 3 measured, 32K): `reused: 16384` on the deep rewind, and **neither new log line printed** — both
are gated on the chain actually holding coded snapshots, so a default park/resume logs what it logged before. That
pass also printed resume figures (50.2 ms append-only, 52.2 ms rewind) which are **3 repetitions on one arm and are
not the fp32 comparison**: the same fp32 rewind measured over 10 repetitions came out at 29.8 ms, so a 3-repetition
number on a cold process reads ~20 ms high on this path. The fp32 numbers this document compares against are the ones
in the table below.

**FP32 against BF16: slightly slower, and only partly matched.** fp32 enters no codec in either build, so neither the
one-pass conversion nor lazy restoration changes what that path does — but an unchanged code path is not a benchmark.
These are the closest figures that exist. The `fp32` column is a pass whose arms were the same binary with only the
config differing (`results/lazy_resume_codec_parity.json`, 2 warm-ups + 10 measured at each size, round 0 of a
2-round plan that was interrupted); the `bf16` column is the eager-against-lazy pass above (`results/lazy_resume.json`,
same harness, same prompts, same token counts, 2 warm-ups + 10 measured × 2 rounds), whose builds differ from the
first by the restore-failure handler alone. Same workload and same engine behaviour — different process runs.

| parked resume, ms | `fp32` | `bf16` | difference |
| --- | --- | --- | ---: |
| 32K, continuing turn | 41.9 (41.5–42.6) | 52.95 (51.5–55.8) | **+11.1** |
| 32K, older checkpoint | 29.8 (29.4–30.2) | 40.15 (38.9–40.6) | **+10.4** |
| 63K, continuing turn | 72.05 (71.1–74.8) | 76.8 | +4.8 |
| 63K, older checkpoint | 30.55 (29.3–35.1) | 39.4 | +8.9 |
| 121K, continuing turn | 124.05 (123.4–131.5) | 126.8 | +2.8 |
| 121K, older checkpoint | 38.95 (32.1–42.2) | 37.4 | −1.6 |

What that supports, and what it does not. Only the two 32K rows were measured against one binary inside one run, and
they say **bf16 costs about 11 ms on a parked resume** — roughly a quarter on the continuing turn, a third on the
rewind. The 63K and 121K rows come from two runs, and the same workload re-measured across two runs of this harness
moved 0.6 ms (32K continuing: 53.5 then 52.95) and 2.7 ms (32K older checkpoint: 37.4 then 40.15), so those four
deltas are the same size as the run-to-run movement and do not separate "slightly slower" from "equal". Byte columns
are deliberately not restated here: the interrupted pass carried another load on the box, which shows in its
`prompt_ms`, `park` and image-size columns and not in `resume_ms`; the byte tables above are matched and stand.

The mechanism is the one lazy restoration leaves behind: every resume out of a parked image restores one coded
snapshot, and decoding it costs the 14.6–17.2 ms measured above — a rewind through an older snapshot and a plain
continuation alike. What lazy restoration takes away is the widen of the other 2–5 snapshots, so the gap against
`fp32` narrows as the conversation grows (11 ms at 32K, 2.8 ms at 121K) because the mount's own live-state restore
dominates more of the total and, on the way back, it copies half the bytes. **What this document claims for `bf16` is half the bytes with
the codec's whole-chain resume penalty gone, parked resume 3–11 ms slower than `fp32` — not millisecond-for-
millisecond equality.** One clean pass with both arms on one binary and the two codecs alternating order is the run
that would settle the last two rows; it has not completed yet.

## Quality gate

This is the part that decides whether the feature ships, and it is a higher bar than K/V quantization faced:
`--prompt-cache` and session restore are byte-exact fp32 copies today, so a coded state introduces error where there
was none. The repo's precedent for accepting a quantization trade with numbers is `docs/DETAILS.md:88`
("perplexity +8-12%; needle tests still pass. 8-bit stays the default").

Run, with the flag off and on, on the same box and the same build:

1. `tools/needle_bench.py` — 32K at depths 10/50/90, after a restore, not just cold. **Done, and through the restore
   path rather than cold:** each depth is read once, saved, restored, then continued, so the answer comes from the
   state that came off disk. 3 of 3 right configured `fp32`, 3 of 3 right configured `bf16`, same words, same box,
   same build:

   ```
   depth  10%: read 32646 tok 22.3s answer='amber'  | bytes=734789228 save=363.6 restore=460.6 | after restore OK 0.77s
   depth  50%: read 32646 tok 21.9s answer='falcon' | bytes=734789228 save=383.6 restore=524.1 | after restore OK 0.41s
   depth  90%: read 32646 tok 22.0s answer='quartz' | bytes=734789228 save=329.1 restore=490.3 | after restore OK 0.39s
   depth  10%: read 32646 tok 22.4s answer='amber'  | bytes=616726140 save=305.1 restore=411.9 | after restore OK 0.74s
   depth  50%: read 32646 tok 22.0s answer='falcon' | bytes=616726140 save=314.6 restore=414.6 | after restore OK 0.43s
   depth  90%: read 32646 tok 22.1s answer='quartz' | bytes=616726140 save=437.3 restore=390.2 | after restore OK 0.39s
   ```

   (`fp32` first, then `bf16`; the wanted word is the one the filler was seeded with.)
2. `tools/ab_engine.py` — the two engines A/B'd on the same prompts. **Done:** 2 rounds, arms differing only in the
   codec, identical answer text in 7 of 8 paired requests, decode medians within 2.8 tok/s, prefill medians
   80.1/74.4/83.2/1422.0 tok/s `fp32` against 81.0/76.8/83.3/1422.7 tok/s `bf16`. The one differing pair is round 0's
   long request; the same request matched in round 1, and this box's own run-to-run drift (max 1.641, mean 0.00228 on
   the same prompt twice with no codec change) is about 26x the codec's error, so one mismatched pair is drift.
3. **error vs tokens-since-resume** — restore a coded state, then measure logit KL and top-1 flips against an fp32
   restore at 1, 10, 100, 1,000 tokens after the resume. This is the decisive measurement: the GDN gate decays old
   state, so a perturbation at the resume point should shrink. If it is flat or grows, bf16 is wrong for this and the
   PR stops here. **Not available through this build's API** — `/v1/chat/completions` ignores `logprobs` and returns
   no logits, so there is nothing to take a KL or a top-1 flip from. It needs a host-side harness that runs both
   restores in one process, which is a change in its own right. Offered in place of it: the value-by-value error
   table above, which bounds the perturbation exactly (every state value moves by at most 2^-8 of itself), and the
   measurement showing the same prompt in two fp32 runs moves that state 26x more.
4. resume at several depths (4K, 32K, 128K) and at several checkpoint ages, since the older the checkpoint the more
   the resumed state matters. **Done at 32K and 121,581 tokens** (needle across a disk restore: 9/9 at 32K over depths
   10/50/90 x 3 seeds, 3/3 at 121,581 over three depths, both configurations, and the same 9 needles 9/9 when the
   resume comes from a parked image instead of a file). Older checkpoint ages **done**: editing a 63,027-token prompt
   back to 20,934 and to 38,498 tokens makes the engine continue from the checkpoint at 16,384 and at 32,768 tokens
   (`reused:` in the engine's own prompt log), and the code word placed before both points is recalled in both
   configurations. A parked chain holding several coded states is covered by the parked-image table above (3, 5 and 6
   checkpoints). 4K was not run as a needle case: a 4K conversation cannot hold a checkpoint older than its live state.

Acceptance: needle recall unchanged, token-for-token identity on the short continuations, and a measured decay curve
reported with the machine it was measured on. **Recall: met. Token-for-token identity: not achievable on this box
even between two fp32 runs of the same file, so the value-by-value bound is offered instead. Decay curve: not
measurable through the API, see 3.**

## Measurements the PR must report

State the machine (CPU, GPU, RAM, OS, engine version, model, flags) next to every number, as `docs/DETAILS.md` does.

- session file bytes, `save_ms`, `restore_ms`, flag off vs on, at 4K / 32K / 63K / 128K. **Done at all four** (the
  4K / 32K / 63K / 128K table above), plus cold-page-cache restores measured separately.
- parked image bytes and `parked=` / `bytes=` log lines, flag off vs on, with the chain length stated. **Done:** the
  parked-image table above, at 31,798 / 63,027 / 121,615 tokens, with chains of 3, 5 and 6 checkpoints plus the live
  state, and the byte deltas agreeing with the arithmetic to 8 bytes.
- encode and decode wall time per running state, and their effect on the request that triggers them. **Both sides
  done:** inside `save_ms` / `restore_ms` as above, and measured on their own host-only — 16.2 ms to code one running
  state, 10.0 ms to widen it. On the request path the cost was visible in the parked resume of the eager build
  (`2274c1c`): 52 → 186 ms at 32K, 86 → 300 ms at 63K, because a resume widened the whole chain. Lazy restoration
  removes that per-chain multiplication; see **Lazy restoration** below.
- decode tok/s and prefill tok/s before and after, to show the codec does not land on the token path. **Done for
  prefill:** the same 63,455-token prompt read in 44.8 s configured `fp32`, 44.6 s configured `bf16`, 44.7 s on the
  previous engine build — the same rate — and `tools/ab_engine.py` medians matching to within 2.8 tok/s on decode over
  2 rounds. No codec code is on the token path.
- peak RAM during SAVE, flag off vs on, to show the capture scratch is admitted and not a surprise. **Done, with the
  limits stated:** `SAVE` grows `VmRSS` by +353.2 MiB configured `fp32` and +240.8 MiB configured `bf16` at 32K on a
  fresh process; a bf16 `RESTORE` is the one operation that grows *more* (+415.2 against +350.8 MiB), because widening
  allocates the fp32 arrays beside the coded originals. The scratch itself is 4 MiB per coding pass, not 118 MB, and
  the preflight admits it.

## What this unlocks (not in this PR)

State is the term that limits how often a conversation can be snapshotted: at 16,384-token spacing the state is
7,206 B/token of the 19,019 B/token this file costs. At bf16, 8,192-token spacing costs the same 7,206 B/token — same
bytes, half the worst case. Using `docs/DETAILS.md:38`'s IQ3_S row (RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5-5200,
Windows, engine 0.1.26 — re-measure on the target box), re-reading 16,383 tokens at the 32K tier's 1,624 tok/s is
10.1 s against 5.0 s for 8,191. The same arithmetic at 4,096 needs fp8, which is why the codec interface is written
for a second member now.

## Risk and rollback

- **One-way door:** once a v2 file exists, an engine older than this change refuses it. It fails closed with a clear
  message; no silent misread. Documented in the flag's help text and in `docs/DETAILS.md`.
- **Quality risk** is the real one and it is front-loaded: the gate above can stop the PR before any user sees a
  coded file.
- **Rollback:** turn the flag off. New writes are v1 again; existing v2 files still restore.
- **RAM:** a coded capture or restore holds one 4 MiB slice plus, on a restore, the widened `tails` and `dead` arrays
  (24 KiB at the geometry below) - not a copy of the 118 MB state. It goes through the existing `admit` preflight, so
  a box that cannot spare it refuses the SAVE instead of degrading, exactly as it does today.

## PR description checklist

Per `AGENTS.md`:

- [ ] one change; say what it leaves alone (the live checkpoint chain, K/V quantization, all defaults)
- [ ] "how I checked" — the test list above, the quality gate, and the measurement table with its machine
- [ ] CUDA **and** HIP builds clean (host-only change, both must be built)
- [ ] default path byte-identical: test 5 proves it
- [ ] new behaviour opt-in, default `fp32`
- [ ] v1 session files still restore: test 7
- [ ] `docs/DETAILS.md` updated with measured numbers, plain words, no claim without a measurement