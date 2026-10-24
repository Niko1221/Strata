# Engine A/B on an RTX 5090: what the 0.1.45 → 0.1.45.3 change is worth, and what it is not

Measured on 2026-10-19/20 by **IVoyt** on one RTX 5090, with `tools/strata_bench.py`'s standard suite
(chat / 4K / 16K / agent / 32K, three runs each, medians with ranges). Nine engine builds, one box, one
model, one run configuration (byte-identical `args` on every arm), one set of runtime libraries.

**The result:** the engine change **0.1.45 → 0.1.45.3 is worth +9.5 % on a 16K prompt and +10.7 % on a
32K prompt**, measured with the same toolkit and the same build flags on both sides. The **CUDA toolkit
makes no difference** (13.0.0, 13.1 and 13.2.2 of the *same* source agree within ±0.3 %), and the
**0.1.45.3 → 0.1.46 change adds nothing** for a single RTX 50.

The first version of this report compared the published 0.1.45 release binary against a local 0.1.45.3
build, which confounds the engine change with the toolkit and the build flags. This version replaces
that with a controlled series: every arm is a local build of a known tree, and the variables are
changed one at a time.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: 12th Gen Intel(R) Core(TM) i7-12700KF (12 cores, 20 threads); installed RAM: 95.8 GiB
- Storage: NVMe SSD (model not recorded); the experts were read from the SSD into RAM at 0.52 GiB/s
- OS: Windows 11 (10.0.26300); GPU driver: 617.42 (CUDA UMD 13.4)
- Strata: commit `3fa1b524` (v0.2.0-beta-7) for the 0.1.45.3 arm, `c093ac26` (v0.2.1) for the 0.1.46
  arms, tag `v0.1.45` for the 0.1.45 arms; Python 3.13.12; Visual Studio 2022 Community 14.34
- The toolkits were installed admin-free from NVIDIA's redistributable components
  (`redistrib_13.0.0.json`, `redistrib_13.2.2.json`) and the 13.1 toolkit that was already on the PC
- CUDA toolkits used: **13.0.0 (V13.0.48)**, **13.1 (V13.1.115)**, **13.2.2 (V13.2.86)**
- Background workloads and power limits: not recorded; the box was otherwise idle

## Model and configuration

- Model: Swift 1.5 Qwen3.8-Flash-Next (IQ3_S),
  `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` plus its second shard for the PLE table;
  a prepared native pack (`D:\Strata-data\packs\swift-iq3_s`) and the engine's expert profile
- Vision encoder: off; reasoning off for the suite; temperature 0
- Context: 524,288 tokens; `--kv int8`, `--kv-resident 32768`; `--conversation-cache-mib 8192`;
  `--conversation-cache-park-gain 2048`; `--expert-cache auto`; `--prefill auto`; `--spec 4`;
  MTP draft layer at `D:\Strata-data\mtp\rt`; rope scaling yarn, factor 2
- `--spec-min-p 0.5`, `--pool-workers 6`, `--adapt-every 1`, `--adapt-swaps 80`, `--adapt-decay 0.97`
- Environment: `STRATA_HC_Q8=1`, `STRATA_EXCHANGE_ROTATE=1`, `STRATA_ATTN_DAILY=1`, `STRATA_EXPDEC=1`,
  `STRATA_HC_UPMIX=1`, `STRATA_QSA_SPLIT_ATTN=1` (all six exist in the 0.1.45 tree as well)
- **No calibration was applied to this measurement**: the run configuration is the install's own
- Every arm used the same `lib_dirs` (the `nvidia-cu13` wheels in the venv: cudart 13.0.96,
  cuBLAS 13.0.2.14), so the **runtime libraries were identical on every arm**

```text
python serve/server.py --engine strata --config <config> --host 127.0.0.1 --port <port> --api-key <key>
python tools/strata_bench.py --url http://127.0.0.1:<port> --api-key <key> --runs 3 --long
```

## The arms

| arm | engine | tree | compiler | build |
| --- | --- | --- | --- | --- |
| A | 0.1.45 | v0.1.45 | 13.0 | **published release build** (portable, sm_75/86/89/120 + PTX) |
| B1 | 0.1.45 | v0.1.45 | 13.1 | local, sm_120 |
| B2 | 0.1.45 | v0.1.45 | 13.1 | local, sm_120 |
| C | 0.1.45.3 | v0.2.0-beta-7 | 13.1 | local, sm_120 |
| D | 0.1.46 | v0.2.1 | 13.0.0 | local, sm_120 |
| E1 | 0.1.46 | v0.2.1 | 13.1 | local, sm_120 |
| E2 | 0.1.46 | v0.2.1 | 13.1 | local, sm_120 |
| F1 | 0.1.46 | v0.2.1 | 13.2.2 | local, sm_120 |
| F2 | 0.1.46 | v0.2.1 | 13.2.2 | local, sm_120 |

The runs were done in the order A, B1, B2, C, D, E1, E2, F1, F2 over two days, so that **the same
build is measured more than once and the machine's drift stays visible**: this box moves 8-28 %
between engine starts at an unchanged setting (#1332), and a single old-vs-new pair cannot separate
that from a real difference.

## Results

Medians with the range over the runs (`n` is the number of requests).

| arm | 4K prompt | 16K prompt | 32K prompt | 16K first token | 32K first token | chat decode | agent turn TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A 0.1.45 published | 2967.8 | 5412.4 | 5564.3 | 3.02 s | 5.81 s | 149.00 | 1.31 s |
| B1 0.1.45 local @13.1 | 3084.4 | 5497.6 | 5572.8 | 2.97 s | 5.80 s | 156.70 | 1.34 s |
| B2 0.1.45 local @13.1 | 2991.4 | 5614.1 | 5560.4 | 2.91 s | 5.81 s | 170.10 | 1.29 s |
| C 0.1.45.3 local @13.1 | 3090.6 | 6022.1 | 6170.0 | 2.70 s | 5.25 s | 162.90 | 1.24 s |
| D 0.1.46 local @13.0.0 | 2959.5 | 5996.5 | 6204.8 | 2.73 s | 5.22 s | 168.80 | 1.21 s |
| E1 0.1.46 local @13.1 | 3099.4 | 5918.5 | 6181.2 | 2.75 s | 5.23 s | 162.50 | 1.23 s |
| E2 0.1.46 local @13.1 | 3064.2 | 5993.8 | 6188.3 | 2.73 s | 5.23 s | 161.20 | 1.23 s |
| F1 0.1.46 local @13.2.2 | 3091.5 | 5928.5 | 6196.4 | 2.77 s | 5.22 s | 170.60 | 1.23 s |
| F2 0.1.46 local @13.2.2 | 3018.6 | 5998.1 | 6213.0 | 2.72 s | 5.21 s | 165.30 | 1.21 s |

### The controlled comparisons (the bench tool's own `--compare` output)

**The engine change, same toolkit and same flags — B1 → C (0.1.45 → 0.1.45.3):**

```
16K prompt tok/s                    5497.60      6022.10     +9.5%
32K prompt tok/s                    5572.80      6170.00    +10.7%
16K first token (s)                     2.97         2.70     -9.1%
32K first token (s)                     5.80         5.25     -9.6%
agent turn first token (s)             1.34         1.24     -8.2%
chat decode tok/s                    156.70       162.90     +4.0%
4K prompt tok/s                     3084.40      3090.60     +0.2%
```

**The engine change, same toolkit and same flags — B2 → E2 (0.1.45 → 0.1.46):**

```
16K prompt tok/s                    5614.10      5993.80     +6.8%
32K prompt tok/s                    5560.40      6188.30    +11.3%
16K first token (s)                     2.91         2.73     -6.1%
32K first token (s)                     5.81         5.23    -10.0%
```

**The toolkit alone, same source (E2 → D, 13.1 → 13.0.0; E1 → F1, 13.1 → 13.2.2):**

```
16K prompt tok/s                    5993.80      5996.50     +0.0%
32K prompt tok/s                    6188.30      6204.80     +0.3%
16K prompt tok/s                    5918.50      5928.50     +0.2%
32K prompt tok/s                    6181.20      6196.40     +0.2%
```

### What the series says

1. **The 0.1.45 → 0.1.45.3 engine change is worth +9.5 % at 16K and +10.7 % at 32K** on a 5090, with the
   toolkit, the build flags, the runtime libraries, the model and the configuration all held constant.
   The 0.1.45.3 → 0.1.46 step adds nothing for this card (C → E1/E2: 6022.1/5918.5/5993.8, inside the
   drift band), which matches the 0.2.1 release notes: their new defaults are for AMD, Linux, Intel,
   integrated GPUs and two-card setups.
2. **The CUDA toolkit does not matter here.** Three toolkits (13.0.0, 13.1, 13.2.2) building the *same*
   source agree within ±0.3 % at 16K and 32K. The compiler is not where the prompt speed comes from.
3. **The published 0.1.45 release build is not slower than a local build of the same tree** (A 5412.4 vs
   B1 5497.6 / B2 5614.1 at 16K, inside the drift band). Its lower number is the older engine, not the
   packaging: `STRATA_PORTABLE=ON` (which forces `GGML_NATIVE=OFF`), the multi-arch list with PTX and the
   bundled cuBLAS make no measurable difference either.
4. **The gain is in the prompt path, and it shows up as time to first token**: 16K −9 %, 32K −10 %,
   agent turn −8 %. Decode is +4 % (C vs B1), inside the drift band.

### Files

- `results-*.json` — every request of every arm, with the machine facts (already redacted)
- `bench-report-*.md` — the bench tool's own report per arm
- `engine-*.log` — the engine log of each arm (the start-up lines: the PCIe probe, the expert cache
  size, the pool workers, the prompt path's settings)
- `strata-*.json` — the run configuration of each arm (identical `args`; only `exe`, `port` and `log`
  differ)
- `run-ab.py` — starts a server with a config, waits for the model, runs the suite, stops the server
- `run-ab-old-code-arms.py` — the arm list for the controlled series
- `build-a-tree.py` — builds an engine from any checkout with a chosen toolkit (the 0.1.45 arms were
  built from a `git worktree` of tag `v0.1.45`)

## Correctness and limitations

- **Speed only.** No needle test, no HumanEval, no tool-call check was run, and the answers of the arms
  were not compared token by token. The 0.1.45.3 change alters the default prompt path on sm_120 (the
  fused int8 prompt experts, the fused gate/up SwiGLU launch), which the project's own notes describe
  as slightly different answers at the same quality; this report does not measure that difference.
- **One machine, one model, one configuration.** A 24 GiB card, another quantisation or another tree
  may land elsewhere.
- **Decode is not resolvable at this machine's noise level.** The two arms built from the same source
  (F1 and F2) differ by 1.2 % at 16K and 0.3 % at 32K; the two arms of the 0.1.45 build differ by 2.1 %
  and 0.2 %. Anything smaller than that is not a result here.
- The 4K prompt test is the noisiest (each arm's first run is 8-10 % below its own later runs, the
  first request after a start being the cold one); the 16K and 32K tests are the stable ones.
- Background load and power limits were not recorded; the drive model is not in the report.
- The toolkits were installed from NVIDIA's redistributable components into `D:\cuda-13.0.0` and
  `D:\cuda-13.2.2` rather than with the Windows installer, so the layouts are the redistributable ones
  (`bin/`, `include/`, `lib/x64/`, `nvvm/`) instead of `C:\Program Files\...\CUDA\v13.x`. The compiler
  and its libraries are the same bits; the build found the root through the same
  `-DCUDAToolkit_ROOT` the setup uses.
