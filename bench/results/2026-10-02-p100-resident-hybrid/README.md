# Dual P100: resident Q4 experts on CPU and GPU

On the measured workstation, computing the RAM-resident expert misses on the CPU increased decode from
8.72/10.95 to 28.70/35.16 tokens/s on the same two prompts: 3.29x/3.21x. The VRAM cache and dense layers
continued to run on two P100s. All four configurations used the same binary, packs, caches and worker count.

This is a community experiment on **Strata 0.1.35, base `d9ab8435f654c368c586340d490915f6addf56a3`**.
It is not a supported release engine or an optimum established for other hardware. The source contribution
adds an opt-in resident RAM complement for fixed layer-split caches and an sm_60 BF16 prompt fallback.

## Hardware and artifacts

| Component | Measured configuration |
|---|---|
| CPU | Threadripper PRO 3955WX, 16 physical cores / 32 threads, AVX2, no AVX-512 |
| RAM | 128 GiB, eight 16 GiB DDR4-3200 RDIMMs, configured at 3200 MT/s |
| GPUs | Two Tesla P100 PCIe 16 GiB, sm_60, TCC, driver 528.89; display GPU excluded |
| Links | PCIe 3.0 x16 per P100, verified under load |
| OS and toolchain | Windows 10 Pro, CUDA 12.4, MSVC 14.44.35207, CMake 3.31.8, Ninja |
| Storage | Mechanical HDD; cold model loading takes roughly 15-20 minutes |
| GGUF | Qwen3.8-Flash-Next UD-Q4_K_XL, four shards, 103.69 GiB |
| Main expert weights | 71.73 GiB; native Q4_K/Q5_K gate/up, Q5_1/Q8_0 down |
| Placement | 5,491 expert slots in VRAM (3,004 + 2,487), 16.02 GiB total; 55.71 GiB complement in pinned RAM |
| PLE | 26.82 GiB table copied to locked host RAM |
| Working set | About 84.12 GiB process RAM; about 14.70/14.91 GiB total GPU memory used per card |
| MTP | Q2 draft experts, reduced head; `--spec 3 --spec-min-p 0.5` |

The main expert quantized bytes were preserved. The local pack converted small dense tensors to BF16
(195 rounded conversions and 264 exactly representable conversions); the PLE convolution is narrowed to F16
on load. The Q8 PLE key remains native. The MTP draft
was converted from its original Q8 format to Strata's Q2 representation. Consequently this is not a claim
of bit-identical inference with an untouched original GGUF or with another backend's Q8 draft.

## Method and results

[`protocol.json`](protocol.json) freezes the exact French sky and Fibonacci prompts. Each request used
256 output tokens, temperature 0, seed 42, thinking disabled, prompt cache disabled, context 4096, one request
at a time. Each configuration ran twice; the second pass reversed the fraction order. All responses reached
the 256-token limit. These are short-context throughput measurements, not a complete answer-quality study.

`pcie_frac` is the fraction of **cache misses** computed on GPU after fetching the expert weights from RAM.
Cache hits always run on GPU. Fifteen CPU workers plus the host thread were retained for every row.

| PCIe fraction | CPU fraction of misses | Sky decode tok/s | Fibonacci decode tok/s | Average CPU cores, sky / code |
|---|---|---|---|---|
| 1.00 | 0% | 8.719 | 10.952 | 0.99 / 0.99 |
| 0.50 | 50% | 14.653 | 17.618 | 14.91 / 14.69 |
| 0.25 | 75% | 22.462 | 26.731 | 14.39 / 14.12 |
| 0.00 | 100% | **28.701** | **35.161** | 14.05 / 13.64 |

Values are two-run medians. [`runs.json`](runs.json) retains all 16 measurements, token counts, MTP counts,
software expert counters, memory values and output hashes. Process CPU time includes worker spin waits.
Every request's expert-source log reported zero file blob reads. Tiny process I/O counter increments include
the engine's stdin protocol and are not evidence of model-file reads.

A restart with `pcie_frac=0` in the startup configuration, with no request-time tuning override, delivered
28.733/35.603 tok/s. It reported 83,494/72,350 CPU expert jobs and **zero RAM-to-GPU expert-weight fetches**.
Activation and result transfers still occur. [`startup-validation.json`](startup-validation.json) records
these two runs. Output hashes matched the corresponding fraction-zero comparative runs.

Both repetitions at fraction zero produced identical text. CPU/GPU rounding can change text between
fractions; sky outputs also varied within the fraction-0.5 pair. The Fibonacci hashes matched across the
mixed modes. No general quality or determinism guarantee follows from these two prompts.

Benchmark binary SHA256: `9b07c8c1e2367d217e17199f137f6c8d33f5832ab02df594b68652ba5c9488d0`.
The contribution subsequently scopes the 64-slot reserve and extra decode log line to the opt-in RAM-split
mode, registers the prompt test, and adds shard-loader regression cases. The running benchmark binary
has not been replaced; these review cleanups are not presented as a new throughput run.

## Bottlenecks

| Probe | Result and interpretation |
|---|---|
| Nominal RAM bandwidth | Eight DDR4-3200 channels imply 204.8 GB/s of interface bandwidth |
| CPU streaming reads | 80.44 GB/s with 16 physical cores; 78.38 GB/s with 32 SMT threads |
| CPU mixed streaming reads/writes | 90.24 GB/s with 16 cores; 89.38 GB/s with 32 SMT threads |
| Core placement | Eight compact cores: 42.19 GB/s; eight spread cores: 79.40 GB/s |
| Pinned host to GPU | Approximately 12.85-13.07 GB/s per P100; 26.09 GB/s combined |
| Strata mapped-host fetch kernel | 12.68-12.78 GB/s; DMA about 12.86 GB/s, only about 1% raw improvement |
| P100 device copy | About 513 GB/s per card, counting both read and write bytes |

RAM probes used 512 MiB arrays, AVX2/OpenMP, physical-core affinity and three trials. These measure the CPU's
access path rather than every DIMM's aggregate limit. They do not establish disabled channels or a BIOS fault;
fabric clock was not measured. SMT did not increase this workload's memory throughput.

The GPU-only reference fetched roughly 331/267 experts per output token. Using the smallest 3.072 MB native
expert gives at least 1.016/0.821 GB of weight traffic per token. At the measured fetch bandwidth, this accounts
for an estimated 80/64 ms of the 115/91 ms decode time per token. This is a traffic estimate, not an isolated
CUDA timing measurement. The measured 3.2-3.3x gain after removing these fetches supports PCIe weight movement
as the previous bottleneck.

At fraction zero, timed CPU expert calls occupied about 40-43% of decode-window wall time. GPU execution and
CPU calls can overlap, and some timing fields cover only the primary GPU while CPU counters cover both stages;
do not add/subtract those fields to derive an exact GPU share. CPU/RAM throughput, dense GPU work and host/GPU
synchronization now matter. Layer-split stages execute successive layers; the sum of GPU busy percentages
is not a throughput or compute-occupancy metric. Some NVML PCIe samples exceeded the physical link limit;
they were rejected as bandwidth evidence in favor of controlled transfers and software byte counts.

The HDD affects cold loading. With the PLE and expert complement locked in RAM, no expert-file reads were
observed during measured responses. Replacing it with an SSD does not remove the measured decode bottleneck.

## Contribution and constraints

- `STRATA_P100_RAM_SPLIT=1` permits strict pinned residency with a layer split, excludes cache slots on every
  stage from the RAM complement, and copies native projections sequentially before scattering unchanged blobs.
- `STRATA_P100_HYBRID=1` permits the CPU share of misses. Without it, the experiment requires `--pcie-frac 1`
  and refuses accidental CPU fallback. The existing CPU kernels and worker pool perform the calculation.
- Caches must stay fixed: `--adapt-every 0 --no-prefill-borrow`; partial/soft residency and remote caches are
  rejected. Evicting an excluded expert would invalidate the complete-RAM assumption.
- The experimental verifier reserves 64 staging blobs, enough for the measured speculative window; ordinary
  runs retain 16. A larger speculation window is not validated and must not assume GPU-only coverage.
- Windows `--ple-io ram` in this mode creates a private `VirtualAlloc`/`VirtualLock` table and fails explicitly
  if the full table cannot be locked. The mapped source remains open; cold-load transient memory exceeds the
  steady-state working set. Resident headroom and lock limits must be considered on smaller machines.
- The sm_60 BF16 prompt GEMM converts operands to FP16 on GPU, retains reusable conversion buffers and uses
  the existing FP16 cuBLAS path with FP32 accumulation. FP16's smaller dynamic range is a numerical limitation;
  the synthetic test and short responses do not establish parity for arbitrary long prompts or extreme values.
- Additional GGUF shards may repeat only `general.architecture` after shard zero's complete metadata is
  validated; matching split metadata and architecture are still required.
- `STRATA_P100_DIAGNOSTIC_SYNC_FILE` optionally names a marker file that enables synchronous prompt-kernel
  diagnostics while it exists. Leave it unset or absent for performance measurements.

Only Windows and this two-P100 Q4 setup have been measured. No HIP, Volta, single-GPU, long-context or concurrent
request validation is claimed. MTP was on in all four comparative modes; a paired MTP-off series is still needed.

## Reproduction

Use an x64 MSVC developer shell, Ninja and **CUDA 12.x**. The measured toolkit was 12.4; target sm_60 SASS,
not a current CUDA 13 release engine. The existing `STRATA_EXPERIMENTAL_SM60` build flag is reused.

```powershell
cmake -S . -B build-p100 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=60-real -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DSTRATA_PORTABLE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-p100 --target strata prefill_p100_test native_dense_ple_key_test native_expert_parity -j 8
$env:CUDA_VISIBLE_DEVICES = '<first P100 UUID>,<second P100 UUID>'
$env:CUDA_DISABLE_PTX_JIT = '1'
$env:STRATA_P100_RAM_SPLIT = '1'
$env:STRATA_P100_HYBRID = '1'
$env:STRATA_LOOKAHEAD = '0'
$env:STRATA_TRACE = '1'
ctest --test-dir build-p100 -R 'prefill_p100_test|native_dense_ple_key_test' --output-on-failure
```

GGML is pinned by CMake to `3cf03257f219afbe7334045ff7c6a06ac68c627d`. An existing checkout may be supplied
with `-DSTRATA_GGML_DIR=<llama.cpp checkout>`. Model pack preparation follows
[`docs/UNSLOTH_Q4.md`](../../../docs/UNSLOTH_Q4.md); use the same compatibility conversions and MTP pack for
every mode. [`config.example.json`](config.example.json) contains relative placeholder paths to replace with
your own packs and CUDA 12 library directory. For an existing `blk.48` Q8 draft with the Gemma norm offset
already included, the tested tensor mapping is available as `tools/mtp_q8_runtime.py`:

```powershell
python tools/mtp_q8_runtime.py --gguf <Q8 draft GGUF> --out mtp/rt --norms-already-offset
```

The output directory must be new. Copy `data/draft_vocab.bin` into it after conversion. This adapter writes
the runtime expert and dense files; it does not convert the main model. Raw-norm drafts must follow the
standard `mtp_pack.py`/`mtp_rt.py` route instead. The portable adapter keeps the tested arithmetic and mapping;
its CLI guard/path cleanup has not been used to requantize a new draft in this contribution.

Start the usual server from the repository root on loopback with one concurrent request:

```powershell
python -m serve.server --engine strata --config bench/results/2026-10-02-p100-resident-hybrid/config.example.json --host 127.0.0.1 --port 8080
```

```powershell
python bench/results/2026-10-02-p100-resident-hybrid/run.py --url http://127.0.0.1:8080 --model qwen3.8-flash-next --engine-log <engine log> --out <new results directory>
```

The portable harness preserves prompt/settings, reverses every second pass, requires decode expert counters,
and checks that no expert blobs come from files. It does not reproduce the original Windows CPU/NVML monitor;
CPU/memory measurements in `runs.json` came from the archived workstation harness. Use `--fractions 0
--startup-only` to check startup settings without `strata_tune`. Stop other inference work during comparisons.

## Validation of the contribution

The review source built successfully in a separate CUDA 12.4 / sm_60 directory with the shared CUDA runtime.
The four CTest cases `gguf_split_test`, `file_expert_source_test`, `native_dense_ple_key_test` (including the new
partial-metadata regression cases), and `prefill_p100_test` passed. Native expert parity on synthetic
`q4_K/q5_1`, `q4_K/q8_0` and `q5_K/q8_0` blobs reported zero failures. The prompt test also checked mapped host
router/grouping IDs and FP16 dequantization; its maximum relative L2 GEMM error was about 1.6e-4 on these inputs.
The Python files passed syntax checks. The portable throughput harness has not run a new full-model sweep;
the measured data above remains the original fixed-binary experiment. The active server was left running.

The earlier generic `prefill_mmq_kquant_test` did not pass its Q8_0 screen on sm_60. It is not included among
the passing checks; this experiment uses and validates the FP16 prompt route instead. The test's default-stream
upload synchronization correction is retained, but it does not establish MMQ support on a P100.

## Follow-up

1. Run a larger shared prompt corpus, long-context cases and a paired MTP-on/off comparison before generalizing.
2. Tune the CPU worker/core placement and profile dense/verification kernels using timing scopes for both GPUs.
3. Build a frequency profile offline; increasing useful VRAM hits avoids CPU work too. Preserve RAM coverage
   before enabling dynamic cache changes.
4. Rebase/revalidate on 0.1.36. Its new fused prompt kernels target newer GPU architectures; its profile-save
   feature is interesting, but merely saving a profile does not improve these fixed caches during a session.

References: [multi-GPU architecture](../../../docs/MULTI_GPU.md),
[engine details](../../../docs/DETAILS.md),
[0.1.36 release](https://github.com/Niko1221/Strata/releases/tag/v0.1.36).
