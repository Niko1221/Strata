# Community support for the Radeon RX 5500 XT 8GB

Current base: **upstream 0.1.34**. See the
[2026-10-02 rebase checks](benchmarks/2026-10-02-upstream-sync-hardware.md).
The performance tables and fleet/context matrix below were measured on **0.1.33**,
at the exact source commits listed. They remain historical evidence.

This is the hardware split requested in [#323](https://github.com/Niko1221/Strata/pull/323),
based on upstream **0.1.34, `1678de3`**. The tested card is the **consumer AMD
Radeon RX 5500 XT 8GB (`gfx1012`)**. The host's name is `a5500`.
It stays in the community/unvalidated architecture list.
Credit for the engine, expert cache and MTP belongs to
[Niko1221/Strata](https://github.com/Niko1221/Strata).

Validation completed on the two configurations below. The architecture stays
community/unvalidated in the upstream classification; these are local results.

## Scope

- Admit `gfx1012` for manual HIP builds.
- Implement signed-byte dot products with RDNA1 SDWA. This is adapted from
  pinned llama.cpp/ggml code; its MIT license and attribution are retained.
- Supply wave synchronization and host-allocation API names for HIP below 7.
  HIP 7 keeps the existing native operations and spellings.
- For legacy hipBLAS 0.x, use the underlying rocBLAS handle and workspace API.
  Newer hipBLAS follows its existing path.
- Preserve negative zero in Q2_0-to-FP16 dequantization on gfx1012/HIP below 7.
  A focused 1024-value bitwise regression test accompanies the fix.

The shared serving patch and experimental kernel alternatives have independent
branches. This hardware branch includes neither. P4 support uses upstream's
existing `STRATA_EXPERIMENTAL_SM60`; no extra Pascal flag or fallback is added here.

## Tested configurations

[Measured requests and build identities](benchmarks/2026-10-01-community-gfx1012.md).

Engine source: `4a6c47ef837e999f428a0740ecf769daa66fbe15`.

| Card | CPU | System RAM | Toolchain |
|---|---|---|---|
| RX 5500 XT 8GB, gfx1012 | Ryzen 5 3600 | 56 GB installed, about 54.8 GiB usable | HIP 5.7.1, clang 17 |
| RX 7900 XTX 24GB, gfx1100 | Ryzen 7 2700 | 40 GB installed, about 38 GiB usable | HIP 7.15, clang 23 |

The modern HIP build checks that the compatibility changes remain isolated.
Both hosts use GSQ-RCO IQ3_S and Q8 KV, text only. The RX 5500 XT uses five CPU
workers, automatic GPU expert caching and a 768 MiB VRAM reserve. The n-gram
shard remains file-backed. Build type is Release; optional HIP MMQ is disabled.

Validation includes intrinsic checks, BF16/F16 GEMM, signed-zero dequantization,
and native expert parity on real model layers 0, 3, 20 and 47. Model tests include
an independent hardware-branch 8K trial, plus a combined serving/hardware matrix
at 8K, 32K and 64K. Combined results identify their separate source commit.

| Check | RX 5500 XT / HIP 5.7 | RX 7900 XTX / HIP 7 |
|---|---|---|
| Committed-source build | Pass | Pass |
| Signed dot, shuffles, wave/shared-memory synchronization | Pass | Pass |
| BF16/F16 prefill GEMM against CPU reference | Pass | Pass |
| 1024-value Q2_0 signed-zero regression | Pass | Pass |
| Native expert parity, real layers 0 / 3 / 20 / 47 | Pass | Pass |
| Upstream versus hardware-patch token/state identity | Upstream does not admit gfx1012 | Pass, eight requests per arm |

Independent hardware-branch results, **8192 input / 9216 allocated context**, MTP4
at threshold 0.5, with up to 512 output tokens:

| Task | Actual output | Prefill | Generation | Total | Effective |
|---|---:|---:|---:|---:|---:|
| Short code | 125 tokens | 106.17 s | 15.30 tok/s | 114.34 s | 1.09 tok/s |
| Short prose | 218 tokens | 99.47 s | 11.68 tok/s | 118.14 s | 1.85 tok/s |

With the separate serving patch applied, the longer 8K code request generated
2688 tokens at **16.20 tok/s** (271.47 s total, 9.90 effective tok/s). At 64K,
code generation was 14.94 tok/s, but 1270.94 s of prefill made total time
1429.21 s. The matrix shows all input/output lengths and makes that cost visible.
These combined measurements do not isolate a hardware-patch speedup.

The independent hardware branch also passed an **8192-input request with
131072 total context tokens allocated**, using Q8 KV streaming with
`--kv-resident 32768`. It generated **2591 tokens at 15.66 tok/s**, with
139.54 s prefill and 304.98 s total time. Minimum sampled free VRAM was
383.39 MiB. This is an allocation/generation check with an 8K starting prompt.

| Coverage | Independent hardware branch | Hardware + serving combination |
|---|---|---|
| Full 8K input | Passed, MTP on | Passed, MTP off/on |
| Full 32K input | Not repeated alone | Passed, MTP off/on |
| Full 64K input | Not repeated alone | Passed, MTP off/on |
| 8K input with 128K allocated | Passed, MTP on, Q8 KV streaming | Not tested in this combination |
| Full 128K input | Not tested | Not tested |

The largest full input tested on this card is 64K. The allocation result does
not establish the maximum usable full-input length or sustained serving capacity.

The kernel journal check found a workqueue CPU-time warning during the test
window; it showed no GPU reset or memory-fault entry. This observation is
separate from the arithmetic and model checks.

Other RDNA1 cards, Windows HIP, vision and multi-GPU behavior remain untested.
Throughput and functional checks do not establish model intelligence.

## Build and run focused checks

```sh
cmake -S . -B build-gfx1012 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_HIP_COMPILER=/usr/bin/clang++-17 -DCMAKE_HIP_ARCHITECTURES=gfx1012
cmake --build build-gfx1012 \
  --target strata hip_intrinsics hip_prefill_gemm hip_q2_zero native_expert_parity -j 6
ctest --test-dir build-gfx1012 \
  -R '^(hip_intrinsics|hip_prefill_gemm|hip_q2_zero)$' --output-on-failure
./build-gfx1012/native_expert_parity /path/to/IQ3_S-shard-1.gguf 0 3 20 47
```

Adjust the compiler path to the installed toolchain. CMake fetches pinned GGML
when `STRATA_GGML_DIR` is omitted. The modern HIP build used
`-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0` for that host's package layout.

To use this build for HTTP serving, set the existing run config's `exe` to the
absolute path of `build-gfx1012/strata`, then restart with that config:

```sh
python -m serve.server --engine strata --config config.json --host 127.0.0.1 --port 8080
```

Use the Python environment installed for Strata. Keep the matching model pack,
GGUF, tokenizer and MTP paths in the config; see the
[model and serving configuration](AMD_HIP.md#model-and-serving-configuration).
Selecting `gfx1012` at build time enables this hardware path; there is no
additional runtime feature switch. This independent branch still requires an
MTP drafter for serving. The [separate serving contribution](https://github.com/CC-David-CC/Strata-a5500/blob/contrib/non-mtp-serving/docs/NON_MTP_SERVING_REVIEW.md#start-the-server-without-mtp)
adds the option to run without one.

The model-throughput harness belongs to the separate serving contribution:
[tested harness source](https://github.com/CC-David-CC/Strata-a5500/blob/dea58f12e0536eb03a4bd8c2266a38ffbd9d0e28/tools/bench_mtp_modes.py).
Copy that utility into `tools/` to reproduce an MTP-on measurement without
adding the serving engine changes. The JSON records the engine and harness
identities separately. Replace `${HOME}` and local model/build paths as needed.
