# Jetson AGX Thor (sm_110): the three documented models, compatibility and speed

Run 2026-11-04/05 on one **NVIDIA Jetson AGX Thor** (L4T R38.2.2, Ubuntu 24.04.4, `6.8.12-tegra`, aarch64,
122.8 GiB unified memory, ~117 GiB usable by the GPU; driver 580.00, CUDA 13.0.48; model files on the NVMe
WD PC SN5000S 1 TB). Engine: **0.1.56 built on this machine from `f-4559-thor-sm110`** - the arm64 fat CUDA
package carrying `sm_121 + sm_110` machine code and `compute_110` PTX ([docs/JETSON_THOR.md](../../../docs/JETSON_THOR.md),
issue #4559). Terminal: the repository's own (`strata.py`, dev checkout at `5819cc0b`).

## Compatibility, model by model

| Model | Pack | Runs on the Thor | Correctness | Speed |
| --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next | ISTA-DAS IQ3_S (84 GB) | **yes** - 4 slots, context 32768 | standard suite: **sanity 12/12** | below |
| Qwen3.6-35B-A3B | Unsloth MTP **UD-Q4_K_XL** (22.9 GB) | **yes** - 4 slots, context 32768 | standard suite: **sanity 12/12** | below |
| Qwen3.6-35B-A3B | UD-Q6_K_XL, Q8_0 | not installed: the same path and file layout as UD-Q4_K_XL, but each pack is a further 33/38 GB download and this machine's link runs ~1.3 MB/s | - | - |
| DeepSeek-V4.1-Flash-1017 | any ISTA-DAS set (191-274 GB) | **engine/feed/fit verified; serving not run** (see below) | - | - |

Details worth recording:

- Both served models loaded from their GGUF/pack and answered chat, tool-call and retrieval requests; the
  engine's K-quant prompt library `libstrata_kq.so` (in the fat package, `sm_110` cubin) loads automatically
  (`K-quant prompt kernels from ... (STRATA_MMQ_KQ=1)`, `prompt experts on MMQ (int8)`).
- **Qwen3.6 at context 128K with 4 slots does not fit this machine**: the engine itself refuses with exact
  numbers (`not enough VRAM for 4 slots at a context of 131072 (5.56 GiB each, 0.78 GiB free)`). 32K with
  4 slots runs. A smaller context or fewer slots is the same trade Strata states for other unified-memory PCs.
- Tegra counts reclaimable page cache against the GPU pool (`cudaMemGetInfo`): after a previous 78 GB model
  run the free figure stayed at ~3.4 GiB and the engine refused; `posix_fadvise(DONTNEED)` over the model
  files as their owner freed ~44 GB and it started (reboot or `drop_caches` need root; the fadvise trick does
  not). Same root cause as the `expert_cache_segmented_test` miss on this machine.
- DeepSeek-V4.1-Flash: its engine package (`strata-dsv41` 0.1.56, `sm_121 + sm_110`) builds, verifies
  (`package_engine.py --verify`, `binary_deps.py`), is selected by `strata update` against a feed carrying it,
  and prints its version inside the official CUDA 13.0 container on the Thor. `strata launch --family
  deepseek41 --model ista-das-coder --plan` on this machine says the set **fits**: "About 66 GB of the 66 GB
  of experts stay in it beside 6.1 GB of dense weights and 12 GB kept for the system; the other 0 GB are read
  from the SSD" - every expert lives in the one 123 GB pool. What was not done is the run itself: the
  smallest set is 191 GB (Coder; others 239-274 GB) and this machine's measured download link (HF, hf-mirror
  and ModelScope alike) is ~1.3 MB/s, i.e. 38-56 hours of downloading. Serving and speed are therefore
  **not run**, not failed.

## Speed: the standard protocol suite (`strata bench run`, protocol `strata-qwen38-thinking-v1-draft.1`)

Both runs used the frozen `standard-v1` corpus (SHA `e75eff8b...`), thinking on, the model card's sampling,
three repeats of short / 4K / 16K prompts, the twelve sanity checks and the optional four-at-once rounds.
Cells are median (min-max) n=3; prompt/decode rates are the engine's own timings, TTFT the client's.
Track: custom. Not cold-cache: the expert cache warms within the run, as the protocol notes.

**Qwen3.8-Flash-Next IQ3_S, context 32768, 4 slots** ([results.json](iq3s-32k/results.json), [report](iq3s-32k/README.md)):

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | ---: | ---: | ---: |
| short | n/a | 38.3 (36.4-40.3) | 0.96 |
| prompt-4k | 769.0 (732.6-834.6) | 34.0 (32.5-34.1) | 5.37 |
| prompt-16k | 1000.2 (910.9-1010.5) | 35.6 (33.9-39.2) | 16.44 |

Sanity **12/12**. Four requests at once: 51.0 / 52.9 / 54.4 tok/s in total per round.

**Qwen3.6-35B-A3B UD-Q4_K_XL, context 32768, 4 slots** ([results.json](qwen36-32k/results.json), [report](qwen36-32k/README.md)):

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | ---: | ---: | ---: |
| short | n/a | 50.7 (50.7-51.0) | 0.14 |
| prompt-4k | 1375.1 (1370.1-1378.4) | 37.4 (37.4-37.4) | 2.96 |
| prompt-16k | 1221.0 (1218.9-1227.0) | 20.2 (20.1-21.7) | 13.41 |

Sanity **12/12**. Four requests at once: 100.5 / 100.5 / 100.8 tok/s in total per round - the Thor serves four
Qwen3.6 requests at once at roughly the speed of one IQ3_S request.

A side note, not fitted away: Qwen3.6's decode at prompt-16k (20.2) sits well below its 4K decode (37.4) on
this machine, where IQ3_S stays flat (~34-36). One machine, custom track, three repeats each - worth watching
on other ARM64 boxes rather than presenting as a settled number.

## Reproducing

Same machine and package: the fat engine comes from `ARCHS="121;110" sh spark/build_engine.sh` (or
`packaging/build_cuda_linux_aarch64.sh`); serve with `strata launch --model <size> [--family qwen36]
--context 32k --vision off --drafting mtp --tune skip --yes --no-tui` and measure with `strata bench run
--url http://127.0.0.1:8080 --suite standard-v1 --optional-concurrency --out <dir>`. The container path
(official `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`, `--runtime nvidia --ulimit memlock=-1`) reproduces
the build and serves the same way; docs/JETSON_THOR.md has the recipe and the function-for-function match
between the container and host binaries.

## Not covered here

Vision (this package ships no picture reader), DFlash2 drafting (untested on sm_110; the Terminal says so),
DeepSeek serving (above), 128K context on this machine (above), more than one module, a long soak, and
`nv_defaults` performance switches (none measured on this part yet - the engine uses its plain defaults on
sm_110).
