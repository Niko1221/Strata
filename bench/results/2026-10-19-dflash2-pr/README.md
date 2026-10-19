# DFlash2 review evidence — 2026-10-19

This is supporting validation for the official-checkpoint engine change, not an
approved benchmark protocol submission. The existing discussion is
[PR #1961](https://github.com/Niko1221/Strata/pull/1961), including its unmerged
[DFlash prerequisite #1426](https://github.com/Niko1221/Strata/pull/1426).
The PR must stay a draft pending missing backend checks and scope agreement.

## Source and environment

Fresh validation uses the prepared `codex/dflash2-main` source against main
`e88ee39d1d6be6e3364183af3b39b62a7ed6f211` (also verified against live origin).
[Source hashes](validation/tested-source-sha256.json) identify the tested source;
committing the prepared files does not change those inputs. Historical checks
below retain their own provenance and executable hashes, rather than being
presented as fresh runs of the submitted revision.

Linux, RTX 4070 Ti SUPER 16 GiB, Ryzen 9 5900X, 125.7 GiB RAM, CUDA 13.4,
GCC 14.2, SM89. Target: local Qwen3.8-Flash-Next NVFP4 pack, BF16 embeddings,
FP8 PLE, shared Q5_K output head and INT8 KV. The pinned official drafter and
model paths are recorded in the [fixture](../2026-10-19-dflash2-quant-mtp/target-config.json).
Models are not included. Logs replace the private checkout with `.` and the
external build/evidence folder with `[VALIDATION_ROOT]`. Set that placeholder
to a local scratch directory when reproducing commands.

## Fresh builds and unit checks

**Passed:**

- CUDA extension ON build; 12/12 selected CTests (loader, policy, BF16 head,
  owned KV, RoPE, taps, DFlash2, BF16 GEMV, MMVQ and GDN parity).
- Fresh CUDA extension OFF build and loader rejection test, and a fresh main
  CUDA control build using the same local ggml dependency.
- CPU-only full build after fixing the loader test's CUDA-header dependency.
  Its full CTest run has 44 passed, one skipped and four failed tests below.
- 203 Python setup/conversion/cache/output-tool tests and 342 server tests.
  The final golden-fixture rerun passes after correcting its portable vision
  executable spelling. Setup selection/choices/golden checks were rerun after
  removing an unrelated automatic engine-update change from the candidate.

Exact Python commands and initial exit codes are in
[python-checks.json](validation/python-checks.json). Per-command logs and the
final reruns are in `validation/`. Build commands (Ninja is in the local venv):

```sh
cmake -S . -B build-dflash2 -G Ninja -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_DFLASH2=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build build-dflash2 --parallel 8
ctest --test-dir build-dflash2 -R 'dflash|bf16_gemv|mmvq_multi|gdn.*parity' --output-on-failure
cmake -S . -B '[VALIDATION_ROOT]/off-build' -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_ENABLE_DFLASH2=OFF -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build '[VALIDATION_ROOT]/off-build' --target strata dflash_load_test --parallel 8
ctest --test-dir '[VALIDATION_ROOT]/off-build' -R '^dflash_load_test$' --output-on-failure
cmake -S . -B '[VALIDATION_ROOT]/cpu-check' -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build '[VALIDATION_ROOT]/cpu-check' --parallel 8
ctest --test-dir '[VALIDATION_ROOT]/cpu-check' --output-on-failure
```

The main controls use an unpacked `git archive` of the stated main commit,
matching CUDA architecture and ggml directory. See configure/build logs.

**Failed:**

- `native_hadamard_test`, `tq_cpu_parity` and `tq_cpu_parity_avx2`: the available
  ggml lacks Trellis types. `isa_guard_check`: `tq_rows_avx512` in
  `tq_avx512.cpp.o` is outside the checker's allowlist. All four failures also
  reproduce on the unmodified main CPU control with that dependency. See
  [candidate](validation/cpu-tests.log) and [main](validation/main-cpu-control.log).
  These are not claimed as passing or fixed by this branch.
- Before the CMake fix, CPU-only `dflash_load_test` failed on missing
  `cuda_runtime.h`; [original failure](validation/cpu-before.log) and successful
  [full rebuild](validation/cpu-build.log) are retained.
- The initial setup golden run failed on 20 vision executable fixture paths;
  only that spelling was updated. Original failure and successful reruns are
  retained, including `test_setup_golden-final.log`.
- The first default-path harness required exactly 32 generated IDs from a
  truncated long prompt, which produced EOS after one. It is retained in
  `default-path-first-attempt/`. The corrected harness uses the complete prompt
  and compares actual IDs, including valid early stops.

**Not run:**

HIP and SYCL compilation/execution: ROCm and `icpx` unavailable; configure
errors are in `validation/hip-configure.log` and `validation/sycl-toolchain.log`.
The DFlash2 extension requires CUDA; shared DFlash changes still need these
backend builds. Metal, Windows, multi-GPU, concurrent requests and a complete
128K prompt were not tested. Disk SAVE/RESTORE with DFlash remains unsupported.

## Default path and cache controls

`default_path.py` starts six engines sequentially: fresh main, candidate with
the extension OFF and candidate with it ON, each in AR and MTP modes.
Three prompts (code, math, full 2425-token notes), greedy and seeded sampling,
32 output tokens, fixed 2000 expert slots validated through native INFO.
CPU prompt share, PCIe share, adaptive swaps and prompt/conversation snapshots
are off in all arms. Executable hashes, full commands, settings and actual IDs
are in `default-path/default-path.json`.

```sh
.venv/bin/python bench/results/2026-10-19-dflash2-pr/default_path.py \
  --baseline '[VALIDATION_ROOT]/main-build/strata' \
  --off '[VALIDATION_ROOT]/off-build/strata' --on build-dflash2/strata \
  --output '[VALIDATION_ROOT]/default-path'
.venv/bin/python tools/dflash_cache.py \
  --config bench/results/2026-10-19-dflash2-quant-mtp/target-config.json \
  --engine build-dflash2/strata --output '[VALIDATION_ROOT]/cache-final'
```

The default-path check passes: 768 candidate token IDs exactly match the main
controls (192 per candidate build/mode; 384 main reference IDs). Final rebuilds
preserve both executable hashes, and the 12 CUDA CTests pass again. See
`validation/default-path-progress.log`, final build/test logs and binary hashes.
This is a controlled regression check, not proof of equality on every
hardware/setting. Cache checks use one-proposal split windows; wider batched
verification has different floating-point arithmetic and is not covered by
that exact-equality claim.

The fresh official Q8 DFlash2 cache run passes all eight cases: 256/256 output
IDs equal both the fresh DFlash2 and target-only controls. Actual expert slots
are 2000 in all three engines. Live continuation reuses 55 tokens, parked
continuation 88, pinned prompt/rewind 16 and sampled-to-greedy continuation 72.
Sampled requests offer zero drafts and subsequent greedy requests resume
drafting. Full records are in `cache-final/`;
[validation summary](validation/summary.json) records the outcomes.

## Historical trained-weight and performance evidence

- [Cache/QFUSE report](cache/README.md): plain DFlash and official Q8 DFlash2,
  eight cache cases each, exact fresh-drafter and target-only controls at fixed
  2000 slots, plus before/after target-state regression for opt-in QFUSE.
- [Stage parity](stage-parity.log): all five trained layers and selector checked
  against independent NumPy forward at Strata's FP32/BF16 boundaries. This is
  not bit equality with every PyTorch backend. `provenance.json` retains its
  earlier base/executable hashes. `feature-alignment.json` contains approximate
  tap comparisons with the supplied checkpoint reference.
- `validation-checks.json` retains an earlier **false** sampled target equality
  check. DFlash2 sampled requests now take the target-only fallback; historical
  multi-proposal sampling is not claimed exact or accelerated.
- [Quantization comparison](../2026-10-19-dflash2-quant-mtp/README.md): BF16,
  Q8_0, Q5_0 and Q4_0 versus MTP with maximum auto expert residency. Full runs,
  variation, quality controls, 256 MiB reserve OOM and interrupted fixed-cache
  attempt are preserved. Q8_0/Q4_0 are within 1% of MTP overall, inside measured
  MTP drift; code improves while prose/long prompt regress. No general speedup
  is established and MTP remains default.
- [VRAM audit](../2026-10-19-dflash2-vram/README.md): full 128K draft allocation
  for BF16/Q8/Q4 and actual 8192→16384-cell growth with an 8448-token prompt,
  nearly full expert cache and target KV streaming. Short generation succeeds;
  a full 128K prompt and unrelated external GPU memory pressure are untested.

Large binary tensor/logit dumps and superseded exploratory runs are archived
outside the checkout; the retained scripts regenerate dumps when required.
The PR contains text evidence and no model weights or generated executables.

## Subsequent performance rerun

The [fresh MTP comparison](../2026-10-19-dflash2-pr-performance/README.md) uses
the committed engine source identified above, all supported DFlash2 weight
formats, prefill 256 and auto expert residency. It retains both MTP reference
runs, all repetitions, output controls and excluded setup attempts. Engine
source inputs remain unchanged; this adds supporting measurement evidence.
