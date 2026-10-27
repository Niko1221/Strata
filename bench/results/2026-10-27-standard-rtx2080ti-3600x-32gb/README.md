# Standard benchmark: RTX 2080 Ti 11 GB, Ryzen 5 3600X (6 cores), 32 GB DDR4-3200, Windows 11

Measured 2026-10-26/27 by IlyaMolochnikov. Protocol `strata-qwen38-thinking-v1-draft.1` (draft), track `custom`, three
complete runs of `strata bench run --url` against a server started by `strata launch`. Tested: what an 11 GB Turing card
with a 6-core Zen 2 CPU and **32 GB of RAM** gives on the smallest pack (Gyro-S TQ1_0, resident experts) and on the
largest one setup offers first (Unsloth UD-IQ4_XS, 55 GiB of experts: most of them read from the SSD). Main limitation:
32 GB of RAM is below what either Unsloth file wants, so run2/run3 measure the SSD-bound regime, not the model's speed.

| | run1 Gyro-S TQ1_0 | run2 UD-IQ4_XS | run3 UD-IQ4_XS (repeat) |
|---|---:|---:|---:|
| `short` decode tok/s (median of 3) | **29.7** (25.4-30.0) | **2.9** (2.6-3.1) | 3.0 (2.7-3.3) |
| `prompt-4k` prompt / decode tok/s, TTFT | 553 / 32.9, 7.4 s | 108 / 3.1, 38.4 s | 108 / 2.9, 38.5 s |
| `prompt-16k` prompt / decode tok/s, TTFT | 536 / 31.6, 30.6 s | 153 / 3.2, 107.6 s | 152 / 3.2, 108.5 s |
| sanity checks | 12 of 12 | 12 of 12 | 12 of 12 |
| required suite after loading | 6 min 25 s | 45 min 16 s | 45 min 33 s |
| run id | `20261026T220124Z-6af0a23e` | `20261027T052010Z-2cec1143` | `20261027T065910Z-8ab56bab` |

Each run folder holds the runner's own `README.md`, `results.json`, `outputs.jsonl`, `config.json` and `provenance/`
untouched (artifact hashes verified), plus `engine-log-excerpt.txt` (the engine session of that run, from its
`engine started` line), `launch-config.json` (the preset `strata launch` wrote; the API key is `<blanked>`) and
`system.txt` (board, BIOS, DIMMs and their clock, page file, disks, `nvidia-smi`, `strata status`). The Windows user
name in paths is replaced by `<user>`.

## Hardware and software

- GPU: NVIDIA GeForce RTX 2080 Ti 11 GB (Turing, sm_75), driver 621.01, CUDA 13.5, power limit 250 W; PCIe 3.0 x16
  (the engine's probe: `13.1 GB/s host->device`).
- CPU: AMD Ryzen 5 3600X, 6 cores / 12 threads, Zen 2, AVX2, no AVX-512. The engine used 5 expert-pool workers plus the host thread.
- RAM: 4x 8 GB Kingston KHX3200C16D4/8GX, **DDR4-3200 (DOCP on), dual channel**; Windows reports 31.9 GiB. Page file 32-48 GB
  (commit limit 63.9 GiB). Board ASUS TUF GAMING B550M-PLUS (WI-FI), BIOS 0245 (2020).
- Storage of the model files: Netac NVMe SSD 1 TB (the system drive). Not benchmarked separately; the engine read experts
  from it at 1.0-1.2 GB/s during decode in run2/run3 (unbuffered reads).
- OS: Windows 11 (10.0.26200). Strata app 0.2.6 / Control 0.1.51 / engine 0.1.50, release build, commit
  `562a0c2ed7169ff6373f893b162106d55a828297`.
- Background work: not measured (`--background` not passed). The PC was otherwise idle; `strata status` ran during collection.

## Model and configuration

**run1:** `--family gyro --model Gyro-S` (`Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf`, sha256 `c6ea1b08…`), 32K context,
`--kv int8`, `--spec 4 --spec-min-p 0.5`, MTP, `--vision` (CPU), `--vram-reserve-mib 700`, `--resident-budget-gib 14`,
env `STRATA_UNBUFFERED_LOAD=0 STRATA_DECODE_TIMING=1`. Engine log: `resident 14.00 GiB, pinned 14.00 GiB`, the other
~6.7 GiB of experts read through the file cache; expert cache 1,342 slots (3.0 GiB); `the CPU pool takes 274 us per missed expert`.

**run2 / run3:** `--family unsloth --model UD-IQ4_XS` with setup's defaults (`--tune quick`): 32K context, `--kv int8`,
`--spec 4 --spec-min-p 0.5`, MTP, `--vision` (CPU), `--resident-budget-gib 8`, env `STRATA_HC_Q8=1`. Engine log:
`55.4 GiB experts, 3.0 GiB in the GPU caches, 8.0 GiB RAM copy (pinned), file tier 44.4 GiB`, `the file tier reads
unbuffered (… 47.4 GiB of experts read from the files: the file cache cannot keep them)`, `the CPU pool takes 129-130 us
per missed expert (18.8 GB/s of expert bytes)`, decode expert-cache hit rate 44-58 %, `only 379 MiB of VRAM free`
(the engine suggests `--vram-reserve-mib 1408`). run3 was meant to add `STRATA_UNBUFFERED_LOAD=0`; the variable was
written to `launch\<preset>.json`, which `strata launch` rebuilds from the install profile on every start, so it did not
apply and run3 is a plain repeat of run2 (its log says `reads unbuffered` too). Kept as the repeatability sample.

## Method

As the runner's README in each folder: 1 warm-up (excluded), 3 repetitions of `short`, `prompt-4k`, `prompt-16k`,
the twelve sanity checks; thinking on at `reasoning_effort xhigh`, 512-token cap (6 of 9 performance requests in run2/run3
and 5 of 9 in run1 reached the cap, as the protocol allows). The server was already running (`--url`), so the runner
could not record the engine options; they are in `launch-config.json`. The optional concurrency stage was not run.

## Observations (engine log, same machine)

- With Gyro-S all experts outside VRAM sit in RAM; two delivery paths (pinned copy vs file cache) gave the same
  **243-274 µs per missed expert**, and 3 pool workers instead of 5 raised it to 382 µs (a fourth run, not included):
  the CPU half of the window is bound by the 6 cores, not by RAM bandwidth, for TQ1_0 on AVX2.
- With UD-IQ4_XS on 32 GB the window is bound by the SSD: 250-330 experts (0.65-0.85 GB) read per verify window at
  1.0-1.2 GB/s, while the CPU share of the window (400-530 experts x 130 µs) is about 10 % of it. Prompts read every
  file-tier expert per chunk (~48 GB per 4K prompt), hence 108 tok/s at 4K and 15 s TTFT on an 80-token prompt.
- Windows page-locks at most about half of the RAM: a 19 GiB budget on this 32 GB box was refused
  (`page-locking refused (out of memory); trying pageable memory`, a run not included), after which the GPU's PCIe share
  of the misses was 0 and decode fell 17-26 % against a 9.5 GiB pinned copy. 14 GiB pinned fine.

## Failures, exclusions and limitations

- No request failed. One earlier Gyro-S run was cancelled by the operator after warm-up and is not included.
- Quantization of Gyro-S is reported `not measured` by the runner (the GGUF carries no quantization field it reads).
- RAM is below the model's needs for UD-IQ4_XS (setup says ~72 GB): run2/run3 are the "setup keeps as many experts as
  fit and reads the rest from the SSD" regime the docs call much slower. They are valid results for that regime only.
- `background_workloads`, `model_card_revision`, `build_options`, storage speed: `null` with reasons in `results.json`.
- No `STRATA_DECODE_TIMING` lines in run2/run3 (see above), so the window is not split into GPU / CPU / file-read time there.
