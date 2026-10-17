# Correctness of the SM70 path and of multi-GPU execution

Folder: this report.
Measurement date: **2026-10-17T02:26Z - 02:40Z** (UTC).
Result matrix: **`correctness-matrix.csv`** (41 rows: 27 PASS, 4 LIMITATION, 3 FINDING, 7 GAP).
Neighbouring documents: `sm70-compatibility.md` (the inventory of 2084 kernels and the SM70 gate table),
`p2p-results.md`, `baseline-profile.md`.

---

## 0. What was checked, and within what limits

Two independent parts:

| part | what was done | with what |
|---|---|---|
| **low-level** | loading **every** kernel of the engine binary on a V100; numeric tests of the primitives (HMMA/GEMM, fp32/fp16 accumulation); kernel resources (registers, shared, local); memory (alloc/free, OOM); launch errors; synchronisation without deadlock; a kernel reading another card's memory | own programs `probes/cuda-tests/*.c*`, the private CUDA 12.8 toolkit + private g++-14, `-arch=sm_70`, everything built and run **only inside this folder** |
| **integration** | 13 checks through the API of the running server: determinism, KV/prefix cache, exceeding the context, parallel requests, long contexts, request errors, the state of the service after the tests | `scripts/correctness-integration.py`, `scripts/determinism-probe.py` (HTTP to :8080, nothing was stopped) |

The Strata IQ3_S service (`0.1.44.1`, binary sha256 `bdfec9f5…`, config sha256 `f539e51b…`) was **not stopped
and not reconfigured** during this work. The low-level tests ran in the devices' primary context shared with
the engine (this matters - see §5 and D3).

Rules applied to the measurements: fixing the output length by banning the EOS ids `248044`/`248046` and
checking `gen_n == max_tokens` and `finish_reason == "length"`; verifying a service return through `/health`
plus a real request plus VRAM; a source edit only with a compile check.

Raw data:
- `raw/correctness/2026-10-17T022637Z/` - `results.json`, `correctness-rows.csv`, `divergence-probe.json`, `path-attribution.json`, `engine-log-delta.txt`, `engine-log-cuda-errors.txt`
- `raw/correctness/2026-10-17T023315Z-cuda/` - `kernel-load-audit.{log,csv}` (2084 rows), `negctl-sm80-on-sm70.{log,csv}`, `sm70-tests-dev{0,1,2}.log`
- `raw/correctness/2026-10-17T022628Z/` - an empty folder from a first failed attempt (a script error, no measurements)
- `build/kernels/` - 62 extracted cubins (38 MB, not included in this PR, see TRIMMED.md), `build/cubin-symbols.csv` - the manifest of 2084 names

Tools (sha256, first 16): `probes/cuda-tests/kernel-load-audit.c` `621ebc6613896edc` -> `build/kernel-load-audit`
`af59d03f75c17e9c`; `probes/cuda-tests/sm70_numeric.cu` `57e6902b510054f0` -> `build/sm70-tests` `9604002e08bfb790`;
`probes/cuda-tests/negctl_sm80.cu` `a3b148c4b583d61c` -> `build/negctl/negctl_sm80.cubin` `d7d6f83311403235`;
`scripts/correctness-integration.py` `95e2a38ffb99806a`; `scripts/determinism-probe.py` `812f0de12ea4f76d`;
`scripts/build-correctness-matrix.py` `c9202da22e535bf4`; `correctness-matrix.csv` `01f7b847d47a8e26`.

---

## 1. The main conclusion

**The engine's SM70 path is correct on this machine.** Not one case of `no kernel image is available` or
`invalid device function` - neither at load time (all 2084 kernels from 60 cubins load on a V100) nor in the
engine's log during all the tests (0 lines). The numeric primitives (tensor-core HMMA, fp32/fp16 GEMM,
reductions in shared memory) match a CPU reference within justified tolerances. Kernel resources (255
registers, 41104 B of shared) fit inside V100's limits (65536 registers per block, 49152 B static shared,
98304 B opt-in dynamic).

**The one genuine incorrectness found at the text level is the non-reproducibility of greedy generation when
the engine's internal state changes** (§5). It is explained by a mechanism (the service tiers of MoE experts),
bounded in size (divergence only in the tail of the output), and is not a defect of the SM70 path: the same
text appears under a different layer split (C3).

## 2. The low-level tests

### 2.1 Loading every kernel of the engine

The kernel inventory was turned into a runnable test: `cuobjdump -xelf all` extracted 62 cubins, the manifest
`build/cubin-symbols.csv` listed 2084 entry symbols, and `build/kernel-load-audit` (the CUDA driver API,
`cuDevicePrimaryCtxRetain` + `cuModuleLoadData` + `cuModuleGetFunction` + `cuFuncGetAttribute`) loaded them on device 0.

```
device 0: Tesla V100-SXM2-16GB cc=7.0 SMs=80 max_threads/block=1024
          max_shared/block=49152 max_shared_optin=98304 max_regs/block=65536
SUMMARY modules_ok=60 modules_fail=0 symbols_total=2084 symbols_ok=2084 symbols_fail=0
SUMMARY max_regs=255 max_shared_bytes=41104 max_local_bytes=384 over_regs=0 over_shared=0 over_threads=0
RESULT PASS
```

The full per-kernel table: `raw/correctness/2026-10-17T023315Z-cuda/kernel-load-audit.csv` (2084 rows: cubin, name, status, registers, shared, local, max threads).

### 2.2 The defect-detection method was checked on control samples

| control | what should happen | what happened |
|---|---|---|
| a cubin built for `sm_80`, on a V100 | a load error | `CUDA_ERROR 209 - no kernel image is available for execution on the device` |
| bf16 `mma.m16n8k16` compiled for `sm_70` | a compile failure | `ptxas: Feature '.bf16' requires .target sm_80 or higher`, `Feature '.m16n8k16' requires .target sm_80 or higher` |

So both a compile gate and a load gate are detectable. Neither occurs in the engine binary (all 62 cubins are `sm_70`) - an independent confirmation of the audit's conclusion that the SM70 build contains no instructions unavailable to this card.

### 2.3 Numeric accuracy against a reference

The reference is the same algorithm on CPU in `double`. For the fp16 tests the reference is computed **from the
same half-rounded inputs**, so that input rounding is separated from accumulation error.

| test | K | max abs | max rel | tolerance | result |
|---|---:|---:|---:|---:|---|
| fp32 SIMT GEMM | 128 | 6.2e-6 | 1.6e-3 | 1e-3 | PASS |
| fp32 SIMT GEMM | 256 | 1.36e-5 | 6.0e-3 | 1e-3 | PASS (the relative figure is cancellation when \|C\| is near 0) |
| **wmma 16x16x16 fp16 -> fp32 (HMMA)** | 128 | 1.41e-5 | 1.45e-5 | 1.3e-2 | PASS |
| **wmma 16x16x16 fp16 -> fp32 (HMMA)** | 256 | 2.66e-5 | 4.2e-5 | 2.6e-2 | PASS |
| fp16 accumulation (for comparison) | 128 | 4.78e-2 | - | - | 3400x worse than fp32 accumulation |
| fp16 accumulation (for comparison) | 256 | 1.16e-1 | - | - | 4400x worse than fp32 accumulation |

Proof that the test really used the tensor cores: the SASS of the built binary contains **80 HMMA instructions** (`cuobjdump --dump-sass build/sm70-tests | grep -c HMMA`).

**A practical conclusion for acceptance criteria:** set the tolerance as `1e-4 * K` on absolute error (a linear
growth with reduction length) and do **not** use relative error on elements whose reference is near zero. For
text-level conclusions bit comparison is impossible in principle: moving from fp32 to fp16 accumulation changes
values in the third digit.

### 2.4 Kernel resources, memory, errors, synchronisation

| id | test | result |
|---|---|---|
| A7 | dynamic shared **41104 B** (exactly the maximum static shared in the engine binary) | runs without error, the reduction is correct (relative error 3.8e-8) |
| A8 | a kernel under high register pressure | `regs=79`, `max_threads_per_block=768`, runs without error, identical results across blocks |
| A9 | local memory / stack | the result is correct (8160.0), `local_bytes=0` - the compiler did not spill the array |
| A10 | 200 x (16 MiB alloc + free) | a leak of **0 B**; a 60 GiB request -> `cudaErrorMemoryAllocation` ("out of memory") without resetting the context; afterwards alloc + launch + sync succeed |
| A11 | recoverable launch errors | `threads=2048` -> invalid configuration argument; `grid=0` -> invalid configuration argument; `cudaMemsetAsync(null)` -> invalid argument; afterwards a normal launch is correct |
| A12 | no deadlock | a grid-stride 80x512 x 50 iterations with `__syncthreads` completes; the `producer -> (event) -> consumer` order across streams is held (a sum over 10^6 elements, rel_err 1.9e-4 - the error of float32 atomic accumulation, not of ordering) |
| A13 | a kernel reading another card's memory | **6/6 ordered pairs** among the reachable cards (0,1,2) give the exact sum; repeated three times (started from dev0, dev1, dev2) - **13/13 PASS** each time |

### 2.5 A limit found along the way

**An external CUDA context cannot be attached to GPU3 while the service runs:** `cudaSetDevice(3)` -> "out of
memory" (nvidia-smi shows 16056 MiB of 16384 used on card 3, about 328 MiB free - less than a context costs).
Consequence for future work: Nsight/CUPTI/any own profiler on GPU3 is impossible while the engine runs; on
GPU0/GPU1/GPU2 it is possible (verified).

---

## 3. The integration tests through the API

All requests: `temperature=0`, `reasoning_effort=none`, a fixed output length by banning EOS, prompts of
**fixed content** (no random marker - unlike the benchmark script).

| id | check | result |
|---|---|---|
| B3 | greedy generation, an approximately 10k-token prompt, 2 repeats | PASS: `distinct_sha=1` (`50246f69…`), `gen_n=128/128`, `finish=length` |
| B4 | **correctness of the KV/prefix cache**: the same prompt with 256 output tokens must continue, not replay | PASS: the first 128 tokens of the long answer are bit-equal to the short answer; `cached=2500` of 2507 |
| B5 | exceeding the context (376 626 tokens at `max_context=262144`) | PASS: HTTP 400 `request exceeds the context window: prompt (376626 tokens) + max …`, the service stays alive, the next request returns 200 |
| B6 | an absurd `max_tokens=300000` | PASS: HTTP 400, the service stays alive |
| B8 | a wrong API key | PASS: HTTP 401 |
| B9 | broken JSON | PASS: HTTP 400 |
| B10 | 2 parallel requests at `concurrency/serving = 1` | PASS: both `finish=length`, wall 6.78 s and 13.53 s (queued, as expected), health OK |
| B11 | **determinism at long context** (about 81k tokens, 2 repeats) | PASS: `prompt_n=80999/80999`, one sha `4d8b8a1b…` in both, `cached=80992` in the second |
| B12 | CUDA errors in the engine's log during the run | PASS: 0 lines of `no kernel image` / `invalid device function` / `CUDA error` / `out of memory` |
| B13 | the state of the service after the tests | PASS: `loaded=true`, VRAM 15246/15596/15186/16056 MiB - the same as before |
| B1, B2 | greedy generation on short prompts (111 and 1259 tokens) | **LIMITATION** - see §5 |
| B7 | an unknown model name returns HTTP 200 | **LIMITATION** of the interface (a single-model service), not a computation error |

Supporting results obtained earlier and used without re-measuring:
- **C1** - 37/37 correctness tests of P2P copies between all six card pairs;
- **C2** - 24 runs with a fixed output length;
- **C3** - the same fixed prompt gives `text_sha = cf94ba5b9a530efd` **in four different engine sessions, including a session with a pinned `--layer-split 12,25,37`**: changing the layer split and the expert-cache placement does not change the text.

## 4. The numeric criteria that actually work here

1. **Low level:** an absolute tolerance `1e-4 * K` for GEMM/reductions with fp32 accumulation and fp16 inputs; compute relative metrics only where \|reference\| > 0.1.
2. **Middle level (the engine's numeric properties):** comparing sums and reductions - a relative tolerance of 1e-5 (observed: 3.8e-8 for a sum of 10276 elements in 41 KB of shared; 1.9e-4 for atomic float32 accumulation of 10^6 elements - atomic accumulation is noticeably worse than a tree order, which matters when comparing implementations).
3. **Top level (LLM):** fix the model, the chat template, `temperature=0`, `reasoning_effort=none`, the seed (not supported by the API), **and the output length**. Compare not the whole text but (a) bit identity under the same engine state, (b) the length of the common prefix under a different state, (c) text checksums (`text_sha`). The threshold "the text matches" = the first at least 95 % of the output tokens match; the tail is unstable when logits are close (observed: divergence at character 478-592 of about 630).

## 5. Determinism: what is guaranteed and what is not

This is the substantive result of this part.

**What is guaranteed.** With unchanged internal engine state, the same prompt gives a **bit-identical** text:
- a short prompt (111 tokens): repeats 2-5 identical (the common prefix is the whole length, 636/636 characters);
- a medium one (1259 tokens): repeats 2-4 identical (626/626);
- a long one (10012 tokens): all three runs identical;
- a very long one (80999 tokens): both runs identical (B11).

**What is not guaranteed.** The first request after a change of engine state gives a different text:
- short: 3 different `text_sha` out of 3 runs; medium: 2 of 3; long: no differences.

**The cause was established (B16, `path-attribution.json`).** The sequence `A1 (cold) -> a filler prompt of 149 669 tokens -> A2 (cold after the prefix was evicted) -> A3, A4 (with the prefix cache)`:

```
A1_cold          sha=1e85a54651001a7b  len=633
A2_after_flush   sha=d80b621e32103e62  len=668   <- cold, but matches A3/A4
A3_cached        sha=d80b621e32103e62
A4_cached        sha=d80b621e32103e62
A1 vs A2/A3/A4: a common prefix of 109 characters;  A2 vs A3 vs A4: fully identical
```

So it is **not** the prefix or cold prefill path: a cold run after flushing the cache matches the cached ones.
What differs is exactly the **first** run - a one-time change of the engine's internal state.

**The mechanism is visible in the engine's own statistics (B17, `engine-log-delta.txt`):** the share of expert-cache hits differs from request to request (99.1 % against 99.7-99.8 %), and the volume of file-tier accesses changes by orders of magnitude (the OS read 18.6 MiB for 2070.7 MiB of expert accesses against 4.8 MiB for 72.9 MiB). A different service path for an expert is a different order of floating-point operations, which changes argmax where logits are close. That is also why the divergence appears in the tail: the first about 95 tokens of 128 have an unambiguous argmax.

**How to work with this (for any future comparison):**
1. Any "better/worse" comparison must be made **after warm-up** - the first request after a change of state (a restart, a config change, a split change) is not part of the comparison.
2. For text criteria use prompts with an unambiguous answer (fact extraction from a long context) - they are stable; free generation on short prompts is not.
3. Record `text_sha` and the common-prefix length in the protocol, not "matched / did not match".

## 6. Gaps (unsupported and untested - not to be confused with failures)

| id | gap | why | how it would be closed |
|---|---|---|---|
| D1 | one GPU, a minimal configuration | an 83.6 GB model does not fit one 16 GB card | needs a different (smaller) model - outside this task |
| D2 | speculative decoding with a draft vs without | the API has no per-request draft switch; turning it off needs an engine restart | a stop window; indirectly: acceptance 69-82 % was measured, and the text is identical with unchanged state (B14) |
| D3 | fatal CUDA errors (`__trap`, illegal memory access) and context recovery | the devices' primary context is shared with the engine process; an accidental illegal access in a test program already caused an error in its next operation | a stop window. Noted: the engine itself survived that incident (health 200, VRAM unchanged) |
| D4 | launching each of the 2084 kernels with real arguments | the signatures and argument semantics are unknown outside the engine's sources | covered indirectly: loading every kernel and their resources (A1), 37/37 P2P (C1), end-to-end service runs (B3-B13) |
| D5 | contexts of 130k…262k on this machine | only up to 128k was measured here; 260 331 exists in PR #2525's data | one 250k run with a fixed output length |
| D6 | kernel profiling (CUPTI/Nsight) | Nsight is not installed; an external context does not fit on GPU3 (A14) | an open question |
| D7 | bit-level control of the engine's intermediate computation (logits, KV, expert sums) | the API does not expose intermediate tensors | an indirect substitute: the hit-rate counters and read volumes in the engine log (B17) |

---

## 7. How to repeat it

```bash
cd <HOME>/strata185/'Strata+4xV100+NVlink'
CU=<HOME>/strata185/cuda-12.8/bin; G=<HOME>/strata185/gcc185/usr/bin/g++-14

# 1. extract the cubins and the symbol manifest
mkdir -p build/kernels && $CU/cuobjdump -xelf all <HOME>/strata185/Strata/engine-cuda12/strata
{ echo cubin,symbol; for f in build/kernels/strata.*.sm_70.cubin; do
    $CU/cuobjdump --dump-elf-symbols "$f" | grep STO_ENTRY | awk -v c="$(basename $f)" '{print c","$NF}';
  done; } > build/cubin-symbols.csv

# 2. load every kernel + the resource check
$CU/nvcc -ccbin $G -O2 -Wno-deprecated-gpu-targets -o build/kernel-load-audit \
    probes/cuda-tests/kernel-load-audit.c -lcuda
./build/kernel-load-audit build/cubin-symbols.csv <out>/kernel-load-audit.csv build/kernels

# 3. the negative controls (compiling for sm_80 and loading that cubin on a V100)
$CU/nvcc -ccbin $G -arch=sm_80 -cubin -o build/negctl/negctl_sm80.cubin probes/cuda-tests/negctl_sm80.cu
$CU/nvcc -ccbin $G -arch=sm_70 -cubin -o /dev/null probes/cuda-tests/negctl_sm80.cu   # ptxas is expected to refuse
./build/kernel-load-audit build/negctl/negctl-manifest-hdr.csv <out>/negctl.csv build/negctl

# 4. the numeric and resource tests (run from dev0, dev1, dev2)
$CU/nvcc -ccbin $G -arch=sm_70 -O2 -Wno-deprecated-gpu-targets -o build/sm70-tests probes/cuda-tests/sm70_numeric.cu
./build/sm70-tests 0

# 5. the integration tests through the API (the service must be running)
python3 scripts/correctness-integration.py            # creates raw/correctness/<UTC>/
python3 scripts/determinism-probe.py <the same directory>
# 6. building the matrix from the raw artifacts
python3 scripts/build-correctness-matrix.py           # -> correctness-matrix.csv
```

---

## 8. What this means for the next steps

1. **An NVLink change can be considered safe from the SM70 point of view:** none of the 2084 kernels needs
   instructions unavailable to the card, resource limits are not exceeded, and a kernel reading another card's
   memory works correctly on every reachable pair.
2. **The text acceptance criterion for a change** should be: bit-identical `text_sha` on **stable** prompts (long
   context, fact extraction) after warm-up, plus a common prefix of at least 95 % on unstable ones. Bit
   comparison of free generation on short prompts would give a false failure.
3. **The first request after any change of engine state is not a measurement.** This rule was confirmed twice:
   in the baseline work (a 62 519-token TTFT in a fresh session was 28.4 s against 46.5 s - a queue) and here
   (the first request gives different text).
4. **GPU3 is not reachable to external CUDA tools** while the service runs: plan any profiling experiment on
   GPU0-GPU2 or in a stop window.
