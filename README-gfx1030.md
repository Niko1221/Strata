# Strata on Radeon V620 (gfx1030) - tuned branch

## 1. Intro and goal

Strata's release binaries target the cards the maintainers test. The Radeon PRO
V620 (gfx1030) is an RDNA2 workstation card that works, but sits outside the
tuned set: several prefill paths that are fastest on newer cards fall back to
slower generic code here, and the default prefill-MMQ kernels do not compile to
the best instruction form for this ISA.

This branch carries a small set of measured local changes on top of upstream
main, tuned for exactly one box: one V620 on a Threadripper PRO workstation. It
is not a general gfx1030 port. It is the daily-driver build for this machine,
with the reasoning behind every change recorded in `bench/notes/strata-migration.md`.

Goal, in one line: **the fastest reading speed (prompt processing) this card can
give for the Qwen3.8-Flash-Next IQ3_S model, without giving up writing speed
(token generation) or stability.**

The branch follows upstream main. When upstream moves, the local commits are
re-based onto the new base, re-benchmarked against the old build, and promoted
only if they measure faster. Every promotion is recorded in the migration note.

## 2. Changes to code

Everything the branch adds vs upstream main is in 9 files, about 1,100 added
lines. There are two functional changes and one test:

### F-2: hyper-connection skip-widen (prefill)

- Files: `include/strata/prefill/gemm.hpp`, `include/strata/prefill/kernels.hpp`,
  `src/prefill/gemm.cu`, `src/prefill/kernels.cu`, `src/prefill/prefill.cpp`
- What it does: in prefill, the hyper-connection up-projection consumer is fed
  packed fp16 halves straight to the GEMM instead of first widening them to
  fp32. The widen pass is skipped; the math result is unchanged.
- How to turn it on: environment variable `STRATA_HC_MIX_H16=1`.
  Without the variable the code path is identical to upstream.
- Measured effect: +1.5 to +2.5% prompt processing on this card, at every
  context tier (see section 4).

### MMQ VOP3P patches (vendored ggml, prefill path)

- Files: `third_party/strata-ggml/ggml/src/ggml-cuda/common.cuh`,
  `third_party/strata-ggml/ggml/src/ggml-cuda/vecdotq.cuh`
- What it does: rewrites the int8 dot chains in the MMQ quantized-kernel
  inner loops to use VOP3P packed-math instructions instead of unpacked
  V_DOT4 chains, and adds a `dp4a_z` helper for zero-extended accumulation.
  Applies to the q8_0 and IQ2_XS/IQ2_S dot chains.
- How to turn it on: build flag `STRATA_PREFILL_MMQ=ON` (requires
  `STRATA_MMQ=ON`, the new split on current main).
- Measured effect: +0.5 to +1.0% prompt processing on top of F-2 at the
  lower tiers, neutral at the top tier (see section 4).

### Test

- `tests/hip/upmix_gemm_bench.cpp`: a HIP benchmark that checks the F-2 GEMM
  path produces bit-identical results to the widen-first path.

Nothing else is changed. Kernel selection, scheduling, quantization formats and
the serving stack are all upstream.

## 3. Installation

### What you need

- The IQ3_S model file, split into 2 GGUF shards (see section 6 for where this
  machine keeps them). Get it the same way as any Strata model - the standard
  install docs cover it.
- A Python virtual environment for the server (`serve/server.py`). If you have
  already run the normal source setup (`setup.sh`), you have this.
- ROCm installed. This machine runs ROCm 10.0.

### Build

Everything after the clone is the standard Strata source build, with four extra
CMake flags. From the checkout:

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_MMQ=ON \
  -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_MMQ_KQUANTS=OFF \
  -DSTRATA_Q6K_EXPERTS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030
cmake --build build-hip -j 32
```

The only gfx1030-specific flags are `STRATA_PREFILL_MMQ=ON` (the tuned MMQ
kernels) and `STRATA_MMQ_KQUANTS=OFF` / `STRATA_Q6K_EXPERTS=OFF` (the k-quant
variants were measured slower here and are left off). Always do a clean build:
delete `build-hip/` first if it exists, because incremental rebuilds after a
flag change produce false comparisons.

The engine binary lands at `build-hip/strata`.

### Serve

The server is started the standard way, with one flag that matters:
`--engine strata` is mandatory on current main (the default is the mock engine).

```sh
cd ~/git/Strata          # or wherever your venv lives
.venv/bin/python serve/server.py --engine strata --config <config.json> --port 8080
```

This machine wraps that in `~/bin/strata-serve` (foreground start, plus `stop`,
`logs`, `status`). The wrapper is a convenience, not a requirement.

### The config

A config JSON tells the server which engine binary to run, which model files to
load, and which environment variables to set. The working config for this
machine is `~/strata-v620-iq3_s-T9-v5.json`; its full contents are in section 6.
The deviations from a default config, and why each one is there:

| Setting | Value | Why |
|---|---|---|
| `spec` | 5 | multi-token prediction draft, measured +5-7% writing speed |
| `spec-min-p` | 0.70 | calibration result; below this the draft hurts |
| `pcie-frac` | 0.00 | calibration result; no prefill CPU assist for PCIe |
| `kv` | int8 | fits the 32 GB card; measured no quality loss |
| `kv-resident` | 32768 | parking window for daily use |
| `--mtp-draft-vocab` | draft_vocab_en.bin | the T9 draft-vocabulary recipe |
| `--ple-io` | ram | PLE pack served from RAM |
| `STRATA_HC_MIX_H16` | 1 | F-2 (this branch's main change) |
| `STRATA_HIP_PROMPT_F16` | 1 | fp16 prompt buffers |
| `STRATA_SELECT_SIMT` | 1 | SIMT1 kernels measured faster than SIMT2 here |
| `STRATA_SH_STREAM` | 1 | calibration result |
| `STRATA_PREFILL_CPU_SHARE` | 1 | measured +2% reading on this CPU |
| `STRATA_HC_PERSIST` | 1 | persistent hyper-connection kernel |
| `STRATA_ROCBLAS_TUNING` | absolute path | rocBLAS tuning table for gfx1030 |
| `STRATA_WATCHDOG_S` | 300 | restart the engine if it stalls 5 min |

For benchmarking, add `--conversation-cache-mib 0` to the args (parking off -
see section 5) and remove `kv-resident`. For daily use, keep them as shown.

## 4. Current results

Model: **Qwen3.8-Flash-Next, IQ3_S**, on this V620. Med-of-3, same-day,
interleaved boots, parking off, 2026-11-18. PP / TG in tokens per second.

| Build | 16k | 32k | 64k | 128k |
|---|---|---|---|---|
| A: previous daily (0.2.10 + F-2 + MMQ) | 964.3 / 95.4 | 977.8 / 94.4 | 982.0 / 91.2 | 971.1 / 90.9 |
| B: vanilla upstream main (0.2.15-dev) | 976.8 / 96.9 | 989.6 / 93.2 | 994.1 / 91.0 | 983.8 / 91.9 |
| **C: this branch** (0.2.15-dev + F-2 + MMQ) | **995.7 / 96.9** | **1008.7 / 94.1** | **1013.5 / 94.0** | **1002.9 / 90.9** |

Reading the table:

- **This branch vs the previous daily build: +3.2 to +3.3% reading speed at
  every tier.** That is the five weeks of upstream work (+1.2-1.3%) plus the
  branch's own changes (+1.9-2.0%) on top.
- **The branch's changes still matter on current main.** C beats B by
  +1.9 to +2.0% PP at every tier, so neither F-2 nor the MMQ patches has been
  absorbed upstream.
- **1000+ PP at 32k, 64k and 128k is a first for this card.** The previous
  IQ3_S bests were 986.0 / 982.3 / 965.8.
- Writing speed (TG) is statistically unchanged across all three builds - the
  changes are prefill-side, and TG noise on this box is about +/-4%.

For context, the same box on other quant lanes (med-of-3, earlier dates):

- UD-IQ1_M (2026-11-06): 1000.6 / 100.9, 1004.4 / 98.1, 999.7 / 97.6, 982.2 / 92.1
- UD-IQ2_XS (2026-11-06): 979.7 / 96.0, 984.5 / 96.6, 980.6 / 94.3, 964.6 / 93.0

The IQ3_S lane is the daily reference; the other lanes were one-off
evaluations. Full history and the reasoning behind every adopted and rejected
change are in `bench/notes/strata-migration.md`.

## 5. The full tier test and methodology

The benchmark is `bench_strata_tiers.py`, from
github.com/vedmichv/llm-benchmark. It sweeps the four context tiers used for
all comparisons here:
**16k, 32k, 64k, 128k** - that is, the same prompt is sent at four different
context sizes so the reading-speed curve can be seen across the whole range.

What one run does:

- For each tier, send a real-text prompt at that context size
  (`NUM_CTX=135168` tokens of context budget per request), generate
  `NUM_PREDICT=1024` tokens.
- Record two numbers per tier: **PP** (prompt processing, tokens/s - the
  reading speed) and **TG** (token generation, tokens/s - the writing speed).
- Each pass writes a JSON to `results/strata_tier_pass_<date>_<time>.json`.

Methodology rules (the parts that matter for comparing numbers):

1. **Soak pass is discarded.** The first sweep after a fresh boot runs the
   expert-cache cold and reads 2-3% slow. It warms the caches; it is not
   counted.
2. **Med-of-3.** Three counted passes per boot; the median PP per tier is the
   result. TG is treated as noise (same-binary day drift of 92.6 to 87.9 tok/s
   at 16k has been demonstrated, so only large TG changes are meaningful).
3. **Parking off.** Bench configs use `--conversation-cache-mib 0`. The
   conversation cache (parking) is a daily-use feature; it makes the first
   request after a change ~1 s slower, which pollutes small measurements.
4. **Interleaved boots for A/B.** When comparing two builds, alternate which
   build boots first, and reverse the order in the second cycle. This cancels
   boot-order and thermal bias (a later boot runs on a warmer card).
5. **Same day, same everything else.** Comparisons are only valid same-day,
   same config, same model file. The engine's context budget must exceed the
   prompt token count.
6. **Env config.** The script has no CLI arguments; it is configured through
   environment variables: `STRATA_BASE`, `STRATA_API_KEY`, `STRATA_MODEL`,
   `NUM_CTX`, `NUM_PREDICT`, `TIERS`.

One full arm (1 soak + 3 counted sweeps, 4 tiers each) takes about 25 minutes.

## 6. Test hardware / software / OS / config

Recorded here because every number in section 4 was measured on this box.

Hardware:

- GPU: AMD Radeon PRO V620 - gfx1030, 40 compute units, 32 GB GDDR6
- CPU: AMD Threadripper PRO 3975WX, 32 cores / 64 threads
- RAM: 207 GB
- Storage: Azure NVv5 (local NVMe)
- Power limit: about 180 W on the card (stock)

Software:

- OS: Linux (Ubuntu, 64-bit), running as a VM on Proxmox 9.2.11
- ROCm 10.0
- Python venv for the server, upstream `serve/server.py`
- rocBLAS tuning table `gfx1030-rocblas-5.6.0.8d1ae90e.txt` (in
  `tools/hip/`, path set via `STRATA_ROCBLAS_TUNING`, absolute path required)

Model and files (IQ3_S lane):

- Model: Qwen3.8-Flash-Next, GGUF quant IQ3_S, split in 2 shards
  (`/var/lib/ollama/gguf/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf`)
- PLE pack: `/home/drewster/git/Strata-data/packs/iq3_s` (loads in 0.3 s)
- Expert profile: `/home/drewster/git/Strata/data/expert-profile.bin`
- T9 draft vocab: `/home/drewster/git/Strata/data/draft_vocab_en.bin`
- MTP draft: `/home/drewster/git/Strata-data/mtp/rt`

Serving posture: `host 0.0.0.0`, port 8080, API key required, model id
`qwen3.8-flash-next-iq3_s-draftvocab-en`, context 262144 tokens, logs in
`~/engines/logs/daily.log`.
