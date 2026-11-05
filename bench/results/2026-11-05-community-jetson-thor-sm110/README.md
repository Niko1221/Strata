# Jetson AGX Thor (sm_110): the three documented models, compatibility and speed

Runs 2026-11-04/05 on one **NVIDIA Jetson AGX Thor** (L4T R38.2.2, Ubuntu 24.04.4, `6.8.12-tegra`, aarch64,
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

## The sm_110 speed table, measured switch by switch (added on this branch)

Five switches went into `nv_defaults.cpp` for cc 11.0, each measured on this machine with
`tools/ab_engine.py`, balanced pairs, medians of n=2-3 (`ab-iq3s-stack.jsonl` build-up,
`ab-iq3s-dropone.jsonl` drop-one, `ab-iq3s-decide.jsonl` final decision; IQ3_S 32K, thinking off,
256-token caps - the pairs compare switches, not the suite):

| Switch | Measured here | In the table |
| --- | --- | --- |
| `STRATA_SPEC_COST=1` | at `--spec 4`: every case faster or equal (chats 1.041/1.000/1.014, 6K prompt 1.031); fixed windows lost at depths 2 and 6/8 | yes |
| `STRATA_HC_Q8=1` | drop-one: chat 47.5 to 44.4 tok/s, 6K prompt prefill 850 to 790 | yes |
| `STRATA_PF_FUSED=1` | drop-one: 6K decode 36.3 to 32.5, prefill 850 to 764 | yes |
| `STRATA_HC_UPMIX=1` | 3 switches to 4: chats +1.8/+2.7/+1.3, 6K prefill +56; bit-identical | yes |
| `STRATA_SIDE_MIXER=2` | decode inside the noise, 6K prefill +128 (851 vs 723) | yes (for the prompt) |

`STRATA_SM110_PERF=0` turns the set off; `STRATA_DEFAULTS=026` leaves it out, as for sm_121. The engine says
what it set: `strata: sm_110 (Jetson AGX Thor): 5 speed switches on by default (...)`. End to end against
the as-shipped run of the same suite above ([iq3s-32k-tuned/results.json](iq3s-32k-tuned/results.json),
sanity again **12/12**): prompt **+20.3% / +9.5%** (4K/16K), first answer at 4K **-27%**, decode **+7.6%**
at 4K and **-6.2%** at 16K (ranges barely overlap: a real small loss at 16K thinking-on, bought back with
interest by the prompt), four-at-once rounds 48.6 / 51.6 (pool partly stale at that run).

## Drafting depths on this part (Qwen3.6 and IQ3_S)

Speculation is the engine's built-in MTP draft + full verification; depths swept paired
(`ab-q36-depth.jsonl`, `ab-iq3s-spec.jsonl`, thinking off, 256 caps):

- **Qwen3.6 UD-Q4_K_XL**: the shipped `--mtp 2` is the slowest setting here - 53.0-53.9 tok/s against
  54.1-54.5 with drafting off; deeper drafts win up to **`--mtp 7`**: 55.7/56.4/56.7 (chats) and 34.7
  (6K prompt), about +5% over the default. `STRATA_Q36_DRAFT_VOCAB=8000/20000` did not help
  (52.4/51.9/51.4). `--batch-mtp` not swept at 1 slot.
- **IQ3_S**: `--spec` 2/4/6/8 peaks at 4 both ways, and the cost rule (in the table) beat plain 4 in every
  case - which is why the depth stays 4 and only the rule is shipped on.
- **DFlash2** (its drafter fetched and packed on this machine): runs correctly - standard suite sanity
  **12/12** ([iq3s-32k-dflash2/results.json](iq3s-32k-dflash2/results.json), model hash skipped) - but
  decodes slower than the model's own MTP layer: 35.1/30.9/31.6 against 38.3/34.0/35.6. So `launch` keeps
  MTP on sm_110 and `drafting._TESTED` stays without sm_110, now with numbers rather than an absence.

## The pool number again, with the leak counts

The `cudaMemGetInfo`-after-exit behavior above was chased further on 11-05: right after a big engine run it
reported 1.8-5.1 GiB free of 122.8 while `MemAvailable` was ~76 GB; a restart restores 117.4. 103 orphaned
`launch-demo` runtime hosts were found on the running machine, each holding a small CUDA reservation
(~3.3 GiB in total, released on kill). Engines size slots, expert caches and prompt chunks from that
number, so on this device the effects were: a second `strata launch` refused (`0.78 GiB free`, exact quote
above), and Qwen3.6's free-based chunk/expert sizing cut single-stream decode to 11-17 tok/s with the
whole model in page cache (the healthy 50.7/37.4 above were taken with a fresh pool). `--parallel 1` gets
past the slot precheck (it only fires for 2+ sessions) but not the sizing. Not fixed here; named so the
precheck/sizing can be judged against `MemAvailable` on Tegra by whoever owns that code.


## Reproducing

Same machine and package: the fat engine comes from `ARCHS="121;110" sh spark/build_engine.sh` (or
`packaging/build_cuda_linux_aarch64.sh`); serve with `strata launch --model <size> [--family qwen36]
--context 32k --vision off --drafting mtp --tune skip --yes --no-tui` and measure with `strata bench run
--url http://127.0.0.1:8080 --suite standard-v1 --optional-concurrency --out <dir>`. The container path
(official `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`, `--runtime nvidia --ulimit memlock=-1`) reproduces
the build and serves the same way; docs/JETSON_THOR.md has the recipe and the function-for-function match
between the container and host binaries.

## Not covered here

Vision (this package ships no picture reader), DFlash2 as a default (measured slower than MTP here, above),
DeepSeek serving (above), 128K context on this machine (above), more than one module, a long soak, and

