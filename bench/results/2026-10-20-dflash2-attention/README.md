# DFlash2 draft attention latency

Measured on 2026-10-20, RTX 4070 Ti SUPER 16 GiB, Ryzen 9 5900X, Linux, CUDA 13.4 / SM89. Parent source: `c7bebe215a6feb754e08dfbc02a3ac12d38fa55f`. The target and official Q4_0 drafter are the same local artifacts as the preceding long-prompt report.

## Cause and change

The preceding report measured 2.42 ms/window for MTP drafting and 4.15 for DFlash2 on the 2425-token fixture. A fresh baseline reproduces DFlash2 at 4.18 ms/window. A section profile attributes 51.77 ms across 33 proposals to its five attention layers: 1.57 ms/proposal. The shared full-vocabulary output head takes 39.25 ms, or 1.19 ms/proposal. Section profiling disables CUDA Graphs; these diagnostic numbers are not pooled with normal decode timings.

The attention kernel recomputed each cell's exponential and paged value address in all 128 value lanes. It now stages those common values in shared memory once per cell. It retains the original QK dot products, maximum, ordered denominator and value sums, FP16 KV format, symmetric local mask and masked-cell skip. It adds 768 bytes of shared memory per block, no global allocation. Confidence gating, blocks, weights, target verification and expert placement are unchanged.

The full local window exposes the cost: 33 chunks per attention head and query row, across five layers. This is redundant work within a bounded 2048-cell window, not unbounded growth with prompt length. The change applies only to the experimental DFlash2 attention kernel. It does not change an existing default configuration or the MTP/target arithmetic.

## Whole-model confirmation

One warmup per task, then three measured repeats. Each request generates 96 tokens, greedy seed 1234. Prompts have 27 and 2425 input tokens. Context/resident capacity 8192, INT8 target KV, prefill 256, automatic expert cache, 512 MiB reserve; adaptive swaps, prompt snapshots, prompt/PCIe CPU sharing and lookup drafting are off. DFlash2 uses block 3 and request floor 0.5 in both arms. Both before/after processes actually hold **3274 target expert slots**. Runs are sequential, without simultaneous compilation or GPU tests. Decode excludes prompt reading and startup.

| Drafter | Code median tok/s [range] | Long median tok/s [range] | Long draft ms/window |
| --- | ---: | ---: | ---: |
| Q4_0 before | 48.15 [47.93–48.49] | 48.45 [48.42–48.54] | 4.18 |
| Q4_0 patched | 48.29 [47.51–48.76] | 49.39 [49.09–49.56] | 3.19 |
| MTP fresh control | 37.47 [36.54–37.55] | 51.24 [51.24–51.27] | 2.46 |

Long drafting falls **23.7%**, whole decode rises **1.9%**. The code throughput change (+0.3%) is within observed variation. MTP uses its own maximum automatic residency, **3139 slots**, rather than being artificially limited to match DFlash2. Patched DFlash2 leads MTP by **28.9% on code**, but remains **3.6% slower on the long fixture**. On long input, DFlash2 still verifies 34 windows and accepts 62/78 proposals, versus MTP’s 33 windows and 63/75. Median target verification remains 53.95 versus 54.28 ms/window; one additional window and 0.73 ms/window of remaining draft overhead explain why the kernel fix does not overtake MTP.

MTP’s log also reports a 106299-token, 275.7 MiB draft output head; DFlash2 uses the shared full-vocabulary 675430400-byte head and its five-layer drafter and selector. These paths have different work and acceptance behavior. The fix does not substitute MTP’s restricted vocabulary or remove selector candidates.

Each repeat's full token IDs, timings, acceptance counts, native INFO and command/environment are retained. The earlier 4.15/2.42 comparison is historical context; the table uses fresh runs for all arms.

## Validation

Passed: CUDA extension-ON build and the DFlash2 parity, policy and loader CTests. The attention FP64-reference fixture includes 4Q/2KV and official 32Q/8KV geometry, 2/4/8 rows, partial chunks, full local windows, nonidentity page tables and padded graph capacities. Existing nonfinite refusal checks remain in the suite.

All warmed and measured before/after and fresh MTP requests match the two target-only controls exactly: **192/192 fixture token IDs**, no near-tie exemption. Target controls run in the optimized process at the same 3274 actual expert slots, with `temperature=1 top_k=1 top_p=1 min_p=0 seed=1234` and zero offered proposals. This is fixture evidence, not a general identity guarantee across arbitrary hardware or expert placement.

Graph-replayed attention microbenchmark, three trials with alternating before/after order, 20 warmups and 200 timed launches per trial. Both binaries use the current parity harness; the reference links only the parent’s unchanged kernel source. All six shapes have byte-identical dumps in every paired trial: **458752 bytes / 114688 FP32 outputs**. The dump SHA-256 is in `checks.json`; large binary dumps remain local.

| Query rows | Context end | Before median ms [range] | After median ms [range] |
| ---: | ---: | ---: | ---: |
| 2 | 32 | 0.015980 [0.015939–0.016614] | 0.007419 [0.007419–0.007419] |
| 2 | 2053 | 0.170307 [0.167076–0.173967] | 0.062444 [0.058998–0.062843] |
| 4 | 32 | 0.017833 [0.017664–0.017920] | 0.008049 [0.008044–0.008049] |
| 4 | 2053 | 0.332078 [0.328054–0.332769] | 0.112707 [0.109123–0.113956] |
| 8 | 32 | 0.020142 [0.019973–0.022057] | 0.010235 [0.009820–0.011069] |
| 8 | 2053 | 0.653450 [0.652662–0.654377] | 0.228864 [0.217093–0.246989] |

The four-row full-window attention pass is **2.95× faster**. This kernel result is not the whole-model speedup. The initial exploratory microbenchmark is retained separately and is not pooled. The post-change section profile reports attention 25.18/33 = 0.76 ms and head 37.36/33 = 1.13 ms per proposal. These graph-disabled diagnostic runs had separate auto residency and are attribution probes, not the controlled throughput comparison.

The extension-OFF build reports `ninja: no work to do`; its two policy/loader CTests pass. This kernel is excluded from that build.

Failed: no CUDA test or engine request failed. The first launcher invocation used unavailable `python`; it was immediately rerun with `python3` before any engine started.

The parent PR CI runtime job failed its portable child-watcher test (1 failure in 882 tests, 14 skipped). Its observed excerpt is retained in `ci-parent-failure.txt`. The exact failing case passes locally with `.venv/bin/python -m unittest serve.test_lifeline.Lifeline.test_the_portable_watcher_ends_the_child_when_the_server_is_killed -v` (`ci-watcher-local.log`). This change does not touch that process-lifetime code; the CI failure’s cause is not established here. A new-head CI result is not claimed.

Not run: HIP, SYCL, Metal, Windows, multiple GPUs, concurrent requests, other drafter quantizations or a context-length sweep. These remain limits of the broader experimental feature; keep the pull request draft. The unchanged extension-OFF path is checked separately. This small two-task result does not replace the preceding four-task quantization comparison.

## Reproduce

Run from the repository root with the existing local model artifacts; no downloads. Build the parent in a separate checkout and save its executable as `$STRATA_DRAFT_RESULTS/strata-before`. Build the patched CUDA extension with `cmake --build build-dflash2 --target strata dflash2_parity -j 4`.

```sh
export STRATA_DRAFT_RESULTS=/tmp/strata-draft-recheck
mkdir -p "$STRATA_DRAFT_RESULTS"
# Place the parent executable in the directory above first.
python3 bench/results/2026-10-20-dflash2-attention/compare.py before
python3 bench/results/2026-10-20-dflash2-attention/compare.py after
python3 bench/results/2026-10-20-dflash2-attention/compare.py mtp
bash bench/results/2026-10-20-dflash2-attention/microbench.sh
python3 bench/results/2026-10-20-dflash2-attention/analyze.py
ctest --test-dir build-dflash2 -R 'dflash2_parity|dflash_policy|dflash_load' --output-on-failure
```

The helpers reuse the preceding performance report's exact model argv and prompt IDs. Before/after executable and source hashes, build commands, diagnostic invocation and microbenchmark reproduction are recorded in `provenance.json`; `microbench.sh` rebuilds and compares the parent/current kernels. Local raw logs are retained outside the checkout; repository copies replace the absolute checkout prefix with `<repo>`. No weights or executables are committed.
