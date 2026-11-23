# Brief for the Chief Architect — Hy4-Flash `add_f2h` fusion and merge status

Date: 2026-11-23
Audience: Chief Architect / engine maintainer
Author: Stefan Angelovski (local performance work)

## 1. Decision requested

1. Accept the `k_add_f2h` fusion concept as the right direction for the hybrid-MoE elementwise path.
2. Note that the original three-site patch is **already present in upstream `main`** (`6a6db7ff`) — there is nothing left to merge from that exact diff.
3. Approve the one remaining delta: fuse the optional CPU-delta path in the pipelined prefill, which upstream still leaves unfused.
4. Approve the benchmark-provenance and test-stability follow-ups listed in section 5. I do **not** have push access to `Niko1221/Strata`, so delivery is via fork + PR, not a direct merge.

## 2. What the patched change actually did

Baseline: `6e38f588a` (local `strata-release-v0218`, engine 0.1.64 local source build).

After the CPU computes routed-expert deltas (`d_zcpu`), the engine must:
- add them into the float residual `zout_f`;
- convert `zout_f` to half precision `z_h` for the next `lin()` matmul.

Before the patch this was two kernels over `nz = T * 1536` elements:

```
k_add_inplace   zout_f[i] += d_zcpu[i]        // read 2, write 1
k_f2h           z_h[i]     = half(zout_f[i])  // read 1, write 1
```

The patch switched three call sites in `engine/src/hy4flash/run/hy4_graph.hip.h` to the already-existing kernel `k_add_f2h`, which was compiled but unused on this baseline:

```
k_add_f2h       v = zout_f[i] + d_zcpu[i]
                zout_f[i] = v
                z_h[i]    = __float2half(v)   // read 2, write 2, one launch
```

Sites changed:
- MoE prefill gather with CPU pass (`~2121`).
- Decode / non-captured `il > 1` path (`~2395`).
- Captured graph tail path (`~2507`).

Diff size: 3 insertions / 6 deletions. No new kernel, no scheduling or allocator change, no model/layout change. The staged component was `0.1.64-linux-x86_64-hip-local`, binary SHA-256 `e412571aab6e6642f2fcdc5ec9a41b31265af5a766fada00e394dffafd49a83`, with `BUILD.json` source note and the same binary copied into the component cache. No other source file or component was modified.

## 3. Evidence and confidence

A/B run on RX 6900 XT gfx1030, ROCm 7.2.4, k8v4 KV, `--spec 4 --spec-min-p 0.5 --mtp rt`, 32k context, standard benchmark profile, 12 requests, identical `configuration` objects except `track_notes`:

| Case | control decode tok/s | patched decode tok/s | Δ |
| --- | --- | --- | --- |
| short (256) | 18.58 | 18.81 | +1.2% |
| 4k | 15.53 | 18.45 | +18.8% |
| 16k | 14.02 | 16.47 | +17.5% |

Quality was 12/12 on both. First-answer latency also dropped at 4k (18.0 → 9.8 s) and 16k (23.2 → 21.2 s), consistent with less memory traffic when `T` is large. Mechanism: one fewer kernel launch and one fewer full pass over `nz` floats per MoE layer; in the captured path, one fewer graph node.

Confidence limits, to be stated plainly:
- One A/B pair, not repeated samples. The +18.8% decode gain at 4k is larger than a decode-sized-`T` traffic model alone predicts; MTP acceptance or run variance may contribute. The prefill/first-token direction is the stronger evidence.
- `software.strata_commit` in both result files is `2e87c08d` (the CLI release), and `engine_version` is `0.1.64` in both; `build_options` is `null`. The result files therefore do **not** encode which source build ran — only `track_notes` label control vs patched. That is a provenance gap, not proof of a bad run, but it must be fixed before treating these numbers as release-grade evidence.
- Both runs' `config.json` files are byte-identical in `engine_options`, including `--pack swift-iq3_xxs` and the same `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` model. The `IQ3_XXS; custom pack` phrase in the patched `track_notes` is a cosmetic label difference, not a model change. Still, free-text notes should not carry meaningful configuration labels; move them to structured fields.
- Upstream `main` is 152 commits ahead of `6e38f588a` for this file (+937/−272 lines). The old numbers describe that old tree, not current `main`.

Verification already done on the baseline tree:
- `ctest --test-dir build-opt -R 'bench|server' --output-on-failure`: 16/16 passed on both control and patched builds.
- Full `ctest` was attempted but stopped at test 78 (`host_api_nocuda_required`) because it needs the external Strata Host SDK; this is an environment limit, not a patch regression. (A later server-only run also hit an unrelated flaky/timeout in `bench-remote-server-inproc-tui-thirty-sec`, already noted in prior work.)

## 4. Merge status against newest release

`origin/main` is now `6a6db7ff`. Inspecting `engine/src/hy4flash/run/hy4_graph.hip.h` at that commit shows `k_add_f2h` already wired at:
- `~2568` (gather peer-only path),
- `~2673` (gather with CPU pass),
- `~3208` (captured tail).

Those cover the three sites from the local patch. The provenance is upstream commits `a24854fd9` ("simd: fuse force-add+f2h into a single pass") and `cd34ab66b` ("hy4flash: reduce CPU-side token preparation"). **Merging the local patch as-is would be redundant or conflict-prone; there is no new value in it for `main`.**

However, one equivalent site remains unfused in `main`, at `~2628–2630` in the pipelined prefill `need_cpu_pass` path:

```
if (need_cpu_pass) {
    ...
    k_add_inplace(zout_f, d_zcpu, nz);
}
k_f2h(zout_f, z_h, nz);          // unconditional
```

This cannot be replaced blindly: when `need_cpu_pass` is false, `d_zcpu` is stale, so `k_add_f2h` would add garbage. The correct delta is a conditional fusion — call `k_add_f2h` inside the `if`, and keep plain `k_f2h` in an `else`. That is the change prepared for this brief on branch `perf/fuse-cpu-pass-add-f2h` (worktree `/home/stormrage/.hermes/cache/scratch/strata-merge-fused`). It is a local, reviewable change; it has not been pushed or merged.

## 5. What else needs improvement, ranked

1. **Ship the conditional fusion above** (correctness-preserving, same kernel, one fewer launch when the CPU pass runs). Verify with `bench` + targeted CTest on a build from current `main`, then PR.
2. **Fix benchmark provenance.** Add structured `--engine-source-commit` / `--build-options` fields to the bench result (or stop writing a stale CLI `strata_commit` into `software`), so A/B pairs are self-describing. Move labels like `IQ3_XXS; custom pack` out of free-text `track_notes`. This is the largest evidence-quality risk right now.
3. **Fuse the remaining peer-add + f2h chains.** `main` still has `k_add_inplace` followed by `k_f2h` in peer-only paths (`~1177`, `~2734`, `~2946`, `~3003`). Fusing the *last* peer add with the f2h is safe if the loop's final peer is known; this removes another launch per layer. Needs care with multi-peer accumulation and graph-capture equivalence.
4. **Reduce CPU-side prep / H2D serialization.** The current path still does `cpu.wait()` then `hipMemcpyAsync(d_zcpu, ...)`. The CPU pack path and expert-cache work upstream may already help; profile whether the wait or the copy dominates before adding more prefetch complexity.
5. **Test stability / environment gates.** `host_api_nocuda_required` needs the external Host SDK and should be skipped with an explicit reason when absent. The 30-second in-proc TUI bench test has already shown a timeout; add a longer default or retry before treating it as a release blocker. The `memory.max` cgroup guard from the benchmark launcher should be kept — earlier OOM at 16k was real.
6. **Do not chase hipBLASLt / tensor-core paths on gfx1030** without first proving a usable kernel exists. Prior work found no gfx1030 tensor-core GEMM kernels; the wins here are memory-bound elementwise fusions and launch overhead.

## 6. Risks and asks

- Risk of the conditional fusion: low. The two branches are mutually exclusive and reproduce today's behaviour exactly when `need_cpu_pass` is false. Main risk is a typo in the `else` branch, caught by build + tests.
- Risk of over-claiming performance: moderate until A/B is repeated on current `main` with fixed provenance. The mechanism is sound; exact percentages are not yet release-grade.
- Access: no push permission on `Niko1221/Strata`. I will fork to `Stormrage34/Strata`, push the branch, and open a PR. Final merge remains a maintainer action unless access is granted.

Verification on current `main` branch (`6a6db7ff` + this delta):
- Release/HIP build of the `strata` executable succeeded; only pre-existing `-Wunused-result` warnings from ignored `hipError_t` returns on `cudaFree` in `generate.cpp`.
- 11 targeted CTest cases passed (`prefill_plan_test`, `q36_expert_info_test`, `i8_expert_rule_test`, `bench_blob_layout_test`, `qsa_select_bench[_fp32_tiled]`, `tq_pascal_bench`, `file_expert_source_test`, `expert_profile_save_test`, `expert_profile_version_test`, `hip_prefill_expert_paths_bench`).
- A full all-target build fails at `strata-concurrent` and `strata-overlap` with `undefined symbol: strata::kernels::cpu::expert_layout()`. Root cause identified: those two link commands include `libstrata_kernels.a` but omit `libstrata_kernels_cpu.a`, which actually defines the symbol; the main `strata` link includes both. This is a pre-existing CMake link-list bug on upstream `main`, unrelated to this one-line fusion change. Report it separately; do not block this PR on it.

Please confirm: (a) accept this delta for a PR to `main`; (b) accept the provenance fix as a separate follow-up or bundle it here.
