# Community benchmark on RTX 3060 12 GB with an AVX-only Xeon E5-2630 v2 (no AVX2/FMA): Strata 0.2.0-beta against 0.1.45, PCIe Gen3 x8

Measured on 2026-10-20 by @Norhiro. This is a follow-up to the 2026-10-16 report on the same machine
(`2026-10-16-community-rtx-3060-avx-only-e5-2630v2`, PR #2793, closed in the 0.2.0-beta tracker reset).
That PR asked for the checks to be re-run on 0.2.0-beta; this is that re-run. It checks that 0.2.0-beta still builds and runs with `STRATA_ISA_FLOOR=avx` on a real AVX-only CPU,
and measures what 0.2.0 changes against 0.1.45 on the same hardware.

Main limitations:
- **The GPU is now in a different PCIe slot: Gen3 x8 instead of Gen2 x4.** Most of the speed difference from the 2026-10-16 report comes from that slot, not from 0.2.0. See [Note: PCIe slot change](#note-pcie-slot-change).
- One run per version for speed (the two versions one after the other, not interleaved).
- IQ2_XS only; contexts up to 40,960.
- The network was connected during the tests.

Headline:
- 0.2.0 builds and runs on this AVX-only CPU, with no SIGILL. The CPU's experts run on the AVX1 128-bit kernels (`iq_avx1`).
- Against 0.1.45 on the same slot, 0.2.0's decode is +6-12%: short chat 23.6 → 25.1 tok/s, agent turns 27.5 → 30.9 tok/s, cold. Prompt reading is within 0-5%.
- Needles 9/9 to 32K. HumanEval 0-39 39/40 base and 37/40 plus. Japanese 11/12. Tools 8/8.

## Note: PCIe slot change

| | 2026-10-16 report | this report |
|---|---|---|
| GPU / its root port | `81:00.0` / `80:00.0` | `83:00.0` / `80:02.0` |
| Link (sysfs) | root port max 5 GT/s x4 (Gen2 x4), current 2.5 GT/s x4 | 8 GT/s x8 (Gen3 x8) |
| Engine's PCIe probe, host to device | 1.6 GB/s | 6.0 GB/s |
| NUMA node of the GPU; PCIe errors this boot | 1; 0 | 1; 0 |

The GPU was moved to another slot before this boot. The 2026-10-16 report says long prompts were bound by the Gen2 x4 link. With Gen3 x8, the same engine (0.1.45) reads prompts 2.7-3.6 times faster, and an agent turn's first token comes after about 6 s instead of about 22 s:

| Cold, IQ2_XS, NUMA node 1 | 0.1.45, Gen2 x4 (2026-10-16) | 0.1.45, Gen3 x8 (today) | 0.2.0, Gen3 x8 (today) |
|---|---:|---:|---:|
| Short chat decode, tok/s | 17.1 | 23.6 | 25.1 |
| 4K prompt, tok/s (TTFT s) | 178.3 (22.59) | 632.1 (6.40) | 636.2 (6.36) |
| 16K prompt, tok/s (TTFT s) | 342.1 (46.99) | 950.9 (16.94) | 987.4 (16.32) |
| 32K prompt, tok/s (TTFT s) | 352.4 (91.03) | 972.7 (33.04) | 991.9 (32.40) |
| Agent follow-up turns: TTFT s / decode tok/s | 21.89 / 20.5 | 6.17 / 27.5 | 6.22 / 30.9 |

So the 2026-10-16 numbers describe a Gen2 x4 slot. The numbers below describe Gen3 x8. Only the last two columns compare the two Strata versions.

## Hardware and software

- GPU: NVIDIA GeForce RTX 3060, 12288 MiB; driver 610.57.04; power limit 170 W (default).
- PCIe: `0000:83:00.0`: max 16.0 GT/s x16, current 8.0 GT/s x8, NUMA node 1. Root port `0000:80:02.0`: max 8.0 GT/s x8, current 8.0 GT/s x8. 0 PCIe errors this boot. See `T0-env/environment.txt`.
- CPU: 2 × Intel Xeon E5-2630 v2 (Ivy Bridge-EP, 6 cores / 12 threads each, AVX + F16C, no AVX2/FMA), 2 NUMA nodes. The engine was bound to node 1 (the GPU's node) in every test.
- RAM: 251.8 GiB DDR3.
- Storage: Solidigm SSDPFKNU512GZ NVMe, 8 GT/s x4 behind a PCIe switch card (models and the n-gram table).
- OS: Fedora 44, Linux 7.2.5-200.fc44.x86_64.
- Strata 0.2.0-beta: commit `3d1fb480` (the `v0.2.0-beta` tag), engine 0.1.45.3. Source build: CUDA 13.3 from the pip wheels, g++-14, sm_86, `-DSTRATA_ISA_FLOOR=avx`. ggml from the pinned llama.cpp `3cf03257`, the same as 0.1.45. ggml-cpu built with `-msse4.2;-mavx`. Build script in `scripts/build.sh`; engine SHA-256 in `T0-env/environment.txt`.
- Strata 0.1.45 (comparison): the same engine binary as the 2026-10-16 report (commit `296b59d2`, setup's build).
- Background workloads: the GPU's other services were released for every test (a GPU lease); about 0.7 GB of VRAM was in use before the engine started. Otherwise an idle desktop (load average 0.8).
- Network: connected (the 2026-10-16 tests ran with it disconnected).

## Model and configuration

- ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, revision `ed59f92`, IQ2_XS (`Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{1,2}-of-00002.gguf`). These are the same files as the 2026-10-16 report; 0.2.0 pins the same revision and sizes.
- Pack, MTP draft layer and expert profile: the ones setup 0.1.45 made. The two profiles are byte-identical. 0.2.0's setup.py was not run.
- Context 40,960; KV int8; expert cache auto; prefill auto; MTP spec 4 (min p 0.5); thinking off; temperature 0. No vision encoder (0.2.0 turns pictures on by default, but no mmproj is installed here).
- 0.2.0 turned on its sm_86 table by itself: `ADAPT_NOWAIT ADAPT_LAG CPU_IQ_ALL_EXACT1 SIDE_MIXER ATTN_DAILY GR_UP75 GU_SWIGLU QSA_ROWS B6_MMVQ_ROWS`. `CPU_IQ_ALL_EXACT1` requires ggml-cpu built for AVX2, so it does nothing on this AVX build.
- Both versions used the same configuration file except `exe`, `cwd` and the log path. The 0.2.0 file is shown below; the 0.1.45 one is `T2-speed-0.1.45-same-slot/config.json`.

```text
{
 "exe": "<WORKDIR>/experiments/08_v020/build/strata",
 "args": [
  "--pack", "<WORKDIR>/experiments/03_run/Strata-data/packs/iq2_xs",
  "--native", "<WORKDIR>/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
  "--ple-gguf", "<WORKDIR>/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf",
  "--expert-profile", "<WORKDIR>/experiments/08_v020/Strata/data/expert-profile.bin",
  "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
  "--mtp", "<WORKDIR>/experiments/03_run/Strata-data/mtp/rt",
  "--max-context", "40960", "--kv", "int8"
 ],
 "cwd": "<WORKDIR>/experiments/08_v020/Strata",
 "port": 8080, "gpu": 0
}
```

## Method

- T2 speed: `tools/strata_bench.py --long` (3 runs per test), run twice. Cold: the model files dropped from the page cache with posix_fadvise DONTNEED, no sudo. Warm: started again right after. Loading is not in the timings. The engine was bound to NUMA node 1 (`numa_run.py --node 1`).
  - Order: 0.2.0 at 15:23-15:37, then 0.1.45 at 15:38-15:53.
  - 0.2.0's own `tools/strata_bench.py` crashes with `NameError: name 'port' is not defined` before it starts a server (#3235). The 0.2.0 runs used it with that one line restored: `local-patch/strata_bench.py.diff`. Otherwise it measures exactly like 0.1.45's tool; the only other difference is the server address for a `host` in the config.
- T4 recall: `tools/needle_bench.py --lengths 4k,16k,32k --depths 10,50,90`.
- T5: the checks of the 2026-10-09 RTX 3060 report and of the 2026-10-16 report (HumanEval 0-39 with EvalPlus 0.3.1, 12 Japanese questions, 8 tool calls): same prompts and limits, non-streamed requests.
- T1 unit tests: `tools/test_*.py` and `serve/test_*.py` one module at a time. CTest of a separate build of the same source with `-DSTRATA_BUILD_TESTS=ON`, short form: `-E 'bench|stress'`, 120 s per test.
- Not run this time: the NUMA A/B (T3), the 65-minute soak (T6) and the quant yardstick (T7).
- Memory: VRAM is the GPU lease's 10 s samples (peak) and the engine's start-up lines. Engine RSS was not measured in these runs.

## Results

### Speed (T2), Strata 0.2.0-beta

**cold**

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 25.1 (21.9-29.5) | 1.70 (1.50-2.31) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 636.2 (604.6-636.3) | not measured | 6.36 (6.36-6.69) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 987.4 (967.5-993.9) | not measured | 16.32 (16.20-16.65) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 991.9 (988.3-991.9) | not measured | 32.40 (32.38-32.52) |
| agent session, follow-up turns | 14614 (14584-14643) at the last turn | see `T2-speed-v020/cold/*/README.md` | up to 64 | 3 | see there | 30.9 (22.1-40.2) | 6.22 (5.85-6.95) |

**warm**

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 25.8 (22.0-29.6) | 1.68 (1.50-2.29) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 633.1 (605.0-634.8) | not measured | 6.39 (6.37-6.69) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 983.2 (976.1-989.7) | not measured | 16.39 (16.27-16.52) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1002.3 (1001.2-1002.9) | not measured | 32.06 (32.05-32.09) |
| agent session, follow-up turns | 14586 (14584-14599) at the last turn | see `T2-speed-v020/warm/*/README.md` | up to 64 | 3 | see there | 30.1 (20.1-39.1) | 6.21 (5.85-6.96) |

Agent session conversation-cache hit: 69.5 % (69.4-69.5) cold, 69.4 % warm.

### Speed (T2), Strata 0.1.45 on the same slot (comparison)

**cold**

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 23.6 (20.4-25.3) | 1.72 (1.56-2.32) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 632.1 (599.9-634.8) | not measured | 6.40 (6.38-6.75) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 950.9 (941.9-965.4) | not measured | 16.94 (16.68-17.11) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 972.7 (966.7-974.5) | not measured | 33.04 (32.97-33.24) |
| agent session, follow-up turns | 14552 (14532-14579) at the last turn | see `T2-speed-0.1.45-same-slot/cold/*/README.md` | up to 64 | 3 | see there | 27.5 (15.8-39.6) | 6.17 (5.88-6.98) |

**warm**

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 24.3 (21.4-28.5) | 1.69 (1.52-2.31) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 630.2 (601.4-634.8) | not measured | 6.42 (6.37-6.73) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 959.1 (943.3-965.6) | not measured | 16.80 (16.68-17.08) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 950.2 (918.1-964.5) | not measured | 33.81 (33.31-34.98) |
| agent session, follow-up turns | 14571 (14557-14616) at the last turn | see `T2-speed-0.1.45-same-slot/warm/*/README.md` | up to 64 | 3 | see there | 28.4 (22.5-35.5) | 6.29 (5.87-6.91) |

0.2.0 against 0.1.45, cold medians:
- Decode: short chat +6%, agent turns +12%. Warm: +6% and +6%.
- Prompt: 4K +1%, 16K +4%, 32K +2%.

This is one run per version in sequence, and the ranges overlap for several rows, so differences of a few percent are within what one run shows. The decode gain is close to the +10.2% that the 0.2.0 release notes give for an RTX 3060.

### Memory

| | 0.1.45 | 0.2.0 |
|---|---:|---:|
| Expert cache (auto, context 40,960) | 3825 slots, 5.14 GiB | 3813 slots, 5.12 GiB |
| VRAM free with everything loaded | 476 MiB | 473 MiB |
| VRAM peak during T2 (lease samples, whole GPU) | 11592 MiB | 11587 MiB |

Both versions print the "below 512 MiB free" warning (#2065) at start. Decode did not drop in any run.

## Correctness and limitations

### Long-context recall (T4), 0.2.0

| Length | Depth % | Prompt tokens | Found | Seconds |
| --- | ---: | ---: | --- | ---: |
| 4k | 10 | 3327 | True | 8.0 |
| 4k | 50 | 3327 | True | 6.8 |
| 4k | 90 | 3328 | True | 6.9 |
| 16k | 10 | 15089 | True | 17.5 |
| 16k | 50 | 15087 | True | 16.2 |
| 16k | 90 | 15087 | True | 16.0 |
| 32k | 10 | 32045 | True | 32.8 |
| 32k | 50 | 32045 | True | 32.9 |
| 32k | 90 | 32046 | True | 32.9 |

Found 9 of 9 (2026-10-16 on 0.1.45: 9 of 9, about 23 / 48 / 92 s per case on Gen2 x4). A needle test measures recall on these inputs, not overall quality.

### Code, Japanese and tool checks (T5)

| Run | Code evaluated | Base pass | Base+plus pass | Japanese | Tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.2.0, Gen3 x8 (this report) | 40 | 39/40 | 37/40 | 11/12 | 8/8 |
| 0.1.45, Gen2 x4 (2026-10-16 report) | 40 | 37/40 | 35/40 | 11/12 | 8/8 |
| 2026-10-09 RTX 3060 report (i7-12700, AVX2) | 40 | 36/40 | 35/40 | 11/12 | 8/8 |

- Code answers at the 1,024-token cap: 0. Request errors: 0.
- The missed Japanese question is the same one as on 2026-10-16 (index 8). The answer is right (`["東京","大阪","京都"]`) but wrapped in a ```` ```json ```` fence, which the grader does not accept.
- The +2 HumanEval tasks are not attributed to 0.2.0. The slot change also changes the engine's PCIe share (`pcie_frac` 0.04 → 0.16 at start-up), so which experts run on the CPU differs from the 2026-10-16 run. 0.1.45 was not re-run on T5 in the new slot.

### Unit tests (T1, short form), 0.2.0

- Python: 123 of 130 modules passed.
- CTest: 137 of 144 passed. All AVX1/AVX2 kernel parity tests passed: `router_dot_parity_avx`, `iq_avx1_parity`, `iq_avx2_parity`, `q8k_quant_parity_avx`, `s_gemv_q8k_parity`. No SIGILL anywhere.

| Failure | Kind | Detail |
| --- | --- | --- |
| `tools.test_setup_choices`, `tools.test_setup_engine_hash`, `tools.test_setup_golden`, `serve.test_keepawake` | known on 0.1.45 here | the same as the 2026-10-16 report |
| CTest `ple_parity`, `platform_memory_test` | environment, as on 0.1.45 | no Q2_0 model file here; `ulimit -l` is 8 MB |
| `tools.test_iq_pack` | environment | llama.cpp's gguf-py not found (setup.py was not run in this tree); passes 30/30 with `STRATA_GGUF_PY` pointing at the pinned llama.cpp |
| `tools.test_nvfp4_local_pack` | environment | pytest is not in this venv; with pytest: 9 passed, 2 skipped |
| CTest `tq_cpu_parity`, `tq_cpu_parity_avx2`, `native_hadamard_test` | build choice | "this ggml has no trellis types" (built with upstream llama.cpp as by default); also listed in #3358 |
| `serve.test_lifeline` (portable watcher) | upstream bug, reproduced 5/5 | the watcher's start token includes the `ps` state (R then S), so the watcher does not end the child; the Linux `pdeathsig` path passes. Issue #3407 |
| CTest `isa_guard_check` | upstream test list | `tq_rows_avx512` (tq_avx512.cpp) is not listed. The function is reached only behind `tq_isa()` / `cpu_avx512_ok()`. Also seen in #3358. Issue #3409 |
| CTest `ggml_tq_overlay` | CMake 3.31 here | `IN_LIST` in a `-P` script without CMP0057; passes with CMake 4.4.3. Issue #3412 |

### Files in this folder

- `T0-env/`: machine facts for this boot (PCIe, NUMA, driver, engine SHA-256) and the file check against the 2026-10-16 SHA-256 list (0 missing).
- `T1-unit/`: per-module Python test logs and `summary.tsv`, CTest log and JUnit XML.
- `T2-speed-v020/`, `T2-speed-0.1.45-same-slot/`: `strata_bench.py` output (README, `results.json`) for cold and warm, the bench logs and the engine logs.
- `T4-needle/`, `T5-quality/` (answers, EvalPlus grades).
- `scripts/`: the scripts used (`build.sh`, `common.sh`, `t0_env.sh` … `t5_quality.sh`, `run_short_suite.sh`, `quality_suite.py`, …). `STRATA_ENGINE=v020` selects 0.2.0.
- `local-patch/strata_bench.py.diff`: the one-line fix used for the 0.2.0 speed runs.

Local paths are replaced with `<WORKDIR>` / `<WORKSPACE>`. The user name, GPU UUID and network addresses are removed.

### Not tested

- 0.2.0's setup.py, app, Terminal (`strata launch`) and update feed. This is a source build of the engine and server only.
- Pictures (no mmproj), other quantizations (UD-IQ4_XS), contexts above 40,960, `--batch` or concurrent clients.
- Both sockets (the 2026-10-16 report found them about 40% faster for decode than node 1 alone, on Gen2 x4). The NUMA A/B, the soak and the quant yardstick were not repeated on 0.2.0.
- 0.1.45's T4/T5 on the new slot.
- Stopping: the server stopped normally when sent SIGINT alone (T4, T5). In a separate start test, SIGINT sent to the whole process group (server and engine) did not end it within 240 s. GPU memory was released after SIGTERM. Not investigated.
