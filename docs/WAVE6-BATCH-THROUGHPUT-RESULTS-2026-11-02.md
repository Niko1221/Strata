# Wave 6 Candidate: gfx1201 Validation and Throughput (2026-11-02)

## Decision

**Do not promote Wave 6.** Focused correctness checks passed, but the controlled end-to-end run did not show a reproducible throughput improvement. The Wave 5 production binary remains the baseline. The Wave 6 candidate was built from the `integ/community-2026-10` working tree at `5e463f9b` plus its then-uncommitted Wave 6 changes; those code changes are not part of PR #4937's head. Candidate binary MD5: `466f6b04e9eb1fc688f89e71e48b5da4`. Wave 5 production binary MD5: `a106bc923fc97bf4ba02ca6efbe07656`.

## Validation

- **gfx1201 focused CTest:** 24/24 passed: `dflash2_slots_test`, `dflash2_tap_test`, `arch_defaults_test`, `qsa_select_bench`, `qsa_select_bench_fp32_tiled`, `native_mmvq_wide_parity`, `mmvq_il_parity`, `quantize_il_parity`, `iq_multi_parity`, `iq_multi_parity_kq_nov2k`, `iq_multi_parity_kq_swiglu`, `iq_multi_parity_kq_nosub16`, `quantize_act_parity`, `bf16_gemv_parity`, `batch_mtp_depth_test`, `batch_mtp_select_test`, `shared_expert_parity`, `spec_verify_parity`, `verify_batch_parity`, `mmvf_rows_parity`, `gr_parity`, `s2_expert_grouped_parity`, `resident_plan_parity`, and `native_multi_parity`.
- **Production restore:** No production config changes were made; the Wave 5 binary hash remained unchanged. The service was active afterward, keyed `/v1/models` returned 200, unkeyed returned 401, and a 256-token-cap arithmetic smoke finished with `stop`.
- **CPU max-8 focused CTest:** `expert_multi_test`, `window_limits_test`, and `pool_tasks_test` passed (3/3).
- **Serving policy tests:** `serve/test_parallel.py` passed (48/48).
- **DFlash2 small benchmark:** bitwise parity checks passed for 4-slot selection/packing, 16/24/32-column GEMM plus convolution quantization, 4-slot batched attention, and the 4-slot × 5-layer fused KV-ring/RoPE path.
- **Microbenchmarks completed:** `dflash2_gemm_bench` and `dflash2_attn_bench` ran; the measurements below are point timings, not an A/B against the Wave 5 kernel implementation.
- **Broader CPU CTest attempt:** not a valid full-suite pass. `build-cpu8` registered 102 tests; the sweep ended with 4 passed, 80 `Not Run` because their executables were not built, and 18 `Failed` at launch with `no such file or directory`. Only the focused three CPU targets above were built and passed; the broad sweep was not used as a release gate.

Selected `gfx1201` microbench timings:

- Small bench: `topk (7 × 106299)` 52.28 µs; `select (7 rows)` 13.66 µs; `select_multi (4 × 3 rows)` 7.94 µs; `pack_head_rows (3 rows)` 3.16 µs; 4-slot, 5-layer fused KV-ring plus RoPE 14.33 µs.
- GEMM bench, variant 9: gate+up 68.2 µs; down 36.4 µs; QKV 30.8 µs; O 22.0 µs; FC 56.6 µs; KV 13.8 µs.
- Attention bench: window 2048, fp32 variant 3 35.5 µs; fp16 variant 0 83.0 µs (bitwise equal to rounded fp32 per the benchmark output). These are different variants, not a same-variant speed comparison.

## End-to-end A/B (preliminary; no promotion)

Workload: `swift-1.5-iq3_s`, `parallel: 4`, DFlash2 batch window 8 / depth 1, temperature 0, reasoning disabled, `max_tokens=256`; concurrent request counts 2 and 4. Each binary had two fresh launches, each with a 4-request × 24-token warm-up, then four measured rounds per concurrency (8 measurements per cell). The launch order was Wave 5 → Wave 6 → Wave 6 → Wave 5 (ABBA), with a 30-second idle cooldown and GPU-free check between launches; this is not the preferred base/treatment/base/treatment order, so treat small differences cautiously. Both arms used the same serving source, production args, and production environment apart from executable/scratch outputs and measurement variables `STRATA_BATCH_DRAFT_TIMING=1` and `STRATA_ROUTE_TAIL_SKIP=0`. The latter was set in both arms; the tested `--batch` route-tail default is gated on `o.batch == 0`, so this override is inert for this workload. `STRATA_BATCH_DRAFT_TIMING=1` is measurement-only. Positive `DFlash2 drafter on` and `DFlash2 batched co-draft head on` banners, per-run `DFlash2 accepted A/D` counters, and timing samples labeled `DFlash2 batched head, one sync per window` were present in all four launches. This isolates engine-binary differences, not the complete server/scheduler change; it does not prove the new MMVQ/SwiGLU kernels were dispatched.

Aggregate client throughput (tokens/s; median and observed range):

- **2 concurrent requests:** Wave 5 median **142.297** (129.796–153.226, n=8); Wave 6 median **131.093** (121.176–147.327, n=8), an observed median change of **−7.9%**.
- **4 concurrent requests:** Wave 5 median **142.927** (142.198–147.213, n=8); Wave 6 median **140.845** (138.028–145.940, n=8), an observed median change of **−1.5%**.

Per-launch client throughput samples (tok/s; round order within each launch):

- c=2: Wave 5 A1 `[132.716, 148.322, 140.059, 144.535]`; A2 `[129.796, 153.226, 137.036, 145.429]`. Wave 6 B1 `[121.176, 127.841, 133.504, 147.327]`; B2 `[128.682, 124.100, 136.254, 140.662]`.
- c=4: Wave 5 A1 `[147.213, 142.845, 142.812, 142.733]`; A2 `[144.915, 143.571, 142.198, 143.009]`. Wave 6 B1 `[140.892, 143.302, 140.880, 140.810]`; B2 `[138.028, 140.727, 145.940, 139.124]`.
- Draft-ms/window samples at c=2: Wave 5 `[3.63, 3.22, 3.08, 3.03, 3.57, 3.39, 3.14, 2.98]`; Wave 6 `[3.63, 3.05, 3.29, 3.13, 3.52, 3.03, 3.04, 3.21]`. At c=4: Wave 5 `[5.99, 5.00, 5.82, 5.07, 5.86, 5.44, 5.92, 5.55]`; Wave 6 `[5.81, 5.60, 5.97, 5.21, 5.72, 5.27, 6.13, 5.27]`.

The first Wave 5 launch at c=4 had one 196-token `stop` response in rounds 1 and 3; the other measured responses reached the 256-token limit. Aggregate throughput used actual completion-token counts and wall time, so those two rounds were not 4×256-token completions.

The observed ranges overlap at both concurrency levels, so the median deltas are **not separable from run-to-run variation**; this run establishes no end-to-end win and no statistically reliable regression. DFlash2 draft-time medians were effectively unchanged: 3.18 vs 3.17 ms/window at concurrency 2, and 5.685 vs 5.66 ms/window at concurrency 4. Engine logs showed VRAM-headroom warnings at 510, 468, 376, and 328 MiB during parts of the run, further limiting confidence in small median differences.

Concurrent reply hashes varied within both same-binary launch floors (5/8 IDs at c=2 and 11/16 at c=4 for Wave 5; 6/8 and 11/16 for Wave 6). No standalone fixed-prompt sequential parity probe ran before throughput, and this was not the `STRATA_PREFILL_CPU_SHARE=0` parity-isolation mode. The hashes are diagnostic only: same-binary variation prevents attributing cross-arm text differences to the candidate. Separate focused kernel/parity tests passed; these concurrent responses are not a bitwise-parity result.

## Recommendation and limits

- Keep the Wave 5 binary in production; no promotion was made.
- Keep the single-card 8-row guidance unchanged. The Wave 6 run does not justify a default change or a 16-row single-card window.
- Before reconsidering promotion, repeat a matched A/B with verified VRAM headroom and a committed candidate source revision; test the server/scheduler change separately on the same binary, and capture dispatch evidence for each new MMVQ/SwiGLU path. Require an end-to-end improvement outside the control spread.
- Not exercised here: layer-split/multi-GPU execution, the 16-row layer-split recommendation, other GPU backends, and a complete old-server/old-binary versus new-server/new-binary integration A/B.

## Reproduction

The historical A/B used a temporary local harness and private scratch configs; neither is included here. The aggregate values above were checked against the local results JSON. The documented commands below reproduce the test gates and workload shape, not the exact historical prompt sequence. The Wave 6 source diff is not included in PR #4937, so a clean checkout of this PR cannot rebuild the tested candidate or reproduce its binary hash; do not attribute these measurements to the PR head.

### Focused tests and microbenchmarks

Run the commands from the repository root in one shell session. Set machine-specific tool paths as variables, capture a read-only production snapshot, and only then stop the service for an authorized exclusive window. Confirm it is inactive, the device has no holder, and VRAM is back near idle before any candidate load:

```bash
REPO="$(git rev-parse --show-toplevel)"
: "${ROCM_ROOT:?set the ROCm installation root}"
: "${STRATA_HIPBLASLT_TUNING:?set the matching hipBLASLt tuning table}"
: "${STRATA_PYTHON:?set the project virtual-environment Python}"
systemctl --user stop strata.service
systemctl --user is-active strata.service  # expected: inactive
fuser /dev/kfd                           # expected: no output
"$ROCM_ROOT/bin/amd-smi" metric --mem-usage  # verify VRAM has returned near idle
```

Keep production stopped throughout all GPU builds/tests/benchmarks. The build/test commands below assume these variables remain set in the same shell session.

```bash
systemd-run --user --collect --scope -p MemoryMax=infinity -p MemorySwapMax=infinity -p TasksMax=infinity \
  --setenv=PATH="/usr/sbin:/usr/bin:/bin:$ROCM_ROOT/bin" --setenv=ROCM_PATH="$ROCM_ROOT" --setenv=HIP_PATH="$ROCM_ROOT" \
  --setenv=LD_LIBRARY_PATH="$ROCM_ROOT/lib" --setenv=STRATA_HIPBLASLT_TUNING="$STRATA_HIPBLASLT_TUNING" \
  --working-directory="$REPO" -- cmake --build build-verify --target strata dflash2_slots_test dflash2_tap_test \
    arch_defaults_test qsa_select_bench native_mmvq_wide_parity mmvq_il_parity quantize_il_parity iq_multi_parity \
    quantize_act_parity bf16_gemv_parity batch_mtp_depth_test batch_mtp_select_test shared_expert_parity \
    spec_verify_parity verify_batch_parity mmvf_rows_parity gr_parity s2_expert_grouped_parity \
    resident_plan_parity native_multi_parity dflash2_small_bench dflash2_gemm_bench dflash2_attn_bench -j12

# Check STRATA_BUILD_TESTS is ON and the selected CTest registrations are present before execution.
ctest --test-dir "$REPO/build-verify" -N -R '^(dflash2_slots_test|dflash2_tap_test|arch_defaults_test|qsa_select_bench|qsa_select_bench_fp32_tiled|native_mmvq_wide_parity|mmvq_il_parity|quantize_il_parity|iq_multi_parity|iq_multi_parity_kq_nov2k|iq_multi_parity_kq_swiglu|iq_multi_parity_kq_nosub16|quantize_act_parity|bf16_gemv_parity|batch_mtp_depth_test|batch_mtp_select_test|shared_expert_parity|spec_verify_parity|verify_batch_parity|mmvf_rows_parity|gr_parity|s2_expert_grouped_parity|resident_plan_parity|native_multi_parity)$'
systemd-run --user --collect --scope -p MemoryMax=infinity -p MemorySwapMax=infinity -p TasksMax=infinity \
  --setenv=PATH="/usr/sbin:/usr/bin:/bin:$ROCM_ROOT/bin" --setenv=ROCM_PATH="$ROCM_ROOT" --setenv=HIP_PATH="$ROCM_ROOT" \
  --setenv=LD_LIBRARY_PATH="$ROCM_ROOT/lib" --setenv=STRATA_HIPBLASLT_TUNING="$STRATA_HIPBLASLT_TUNING" \
  --working-directory="$REPO" -- ctest --test-dir "$REPO/build-verify" -j1 --timeout 900 --output-on-failure \
  -R '^(dflash2_slots_test|dflash2_tap_test|arch_defaults_test|qsa_select_bench|qsa_select_bench_fp32_tiled|native_mmvq_wide_parity|mmvq_il_parity|quantize_il_parity|iq_multi_parity|iq_multi_parity_kq_nov2k|iq_multi_parity_kq_swiglu|iq_multi_parity_kq_nosub16|quantize_act_parity|bf16_gemv_parity|batch_mtp_depth_test|batch_mtp_select_test|shared_expert_parity|spec_verify_parity|verify_batch_parity|mmvf_rows_parity|gr_parity|s2_expert_grouped_parity|resident_plan_parity|native_multi_parity)$'

# Host-only focused tests.
systemd-run --user --collect --scope -p MemoryMax=8G --working-directory="$REPO" -- \
  cmake --build build-cpu8 --target expert_multi_test window_limits_test pool_tasks_test -j8
ctest --test-dir "$REPO/build-cpu8" --output-on-failure -R '^(expert_multi_test|window_limits_test|pool_tasks_test)$'
PYTHONPATH="$REPO" "$STRATA_PYTHON" "$REPO/serve/test_parallel.py"

# GPU microbenchmarks; run each in its own transient scope.
for bench in dflash2_small_bench dflash2_gemm_bench dflash2_attn_bench; do
  args=(); [ "$bench" = dflash2_small_bench ] && args=(200)
  systemd-run --user --collect --scope -p MemoryMax=infinity -p MemorySwapMax=infinity -p TasksMax=infinity \
    --setenv=PATH="/usr/sbin:/usr/bin:/bin:$ROCM_ROOT/bin" --setenv=ROCM_PATH="$ROCM_ROOT" --setenv=HIP_PATH="$ROCM_ROOT" \
    --setenv=LD_LIBRARY_PATH="$ROCM_ROOT/lib" --working-directory="$REPO" -- "$REPO/build-verify/$bench" "${args[@]}"
done
```

### Re-running the burst shape

`tools/ab/batch_burst_bench.py` is a **client only**: the scratch server must already be running with the selected binary. Its config needs `model_name`, `api_key`, and `port`; it does not print the key or response text. Prepare a separate mode-0600 scratch config per arm from the current production JSON, retaining production args/env except the executable/listener/scratch-write paths and the common measurement-only `STRATA_BATCH_DRAFT_TIMING=1`. Verify the DFlash2 banner and accepted-proposal telemetry before timing.

For each concurrency, use two fresh launches per arm in base/treatment/base/treatment order, with an idle/GPU-free check plus 30-second cooldown between launches. Use `--rounds 4 --conc 2` at c=2 and `--rounds 2 --conc 4` at c=4; the utility has eight built-in prompts, so these settings avoid repeating a prompt within one launch. Run this after starting each scratch server:

```bash
PYTHONPATH="$REPO" "$STRATA_PYTHON" "$REPO/tools/ab/batch_burst_bench.py" \
  "$ARM_CONFIG" "$ARM_OUTPUT" --rounds "$ROUNDS" --max-tokens 256 --conc "$CONCURRENCY"
```

The historical result used distinct `[round/slot]` prompt markers and measured c=2 and c=4 in the same launch; the repository utility uses its own fixed prompts, so it reproduces the shape, not identical outputs. Client throughput is actual completion tokens over client wall time, not a guarantee that every response reaches `max_tokens`. Keep production stopped for all GPU loads. Only if an active final state is authorized, run `systemctl --user start strata.service`; then verify keyed `/v1/models` 200, unkeyed 401, a `max_tokens>=256` completion ending with `stop`, and unchanged production config/binary hashes. Otherwise leave the service stopped and verify no candidate scope still owns the GPU.