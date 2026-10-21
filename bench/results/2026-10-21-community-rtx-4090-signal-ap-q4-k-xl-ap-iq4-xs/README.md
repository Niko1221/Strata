# Community benchmark on NVIDIA GeForce RTX 4090: agentionai Signal 3.8 Flash Next (AP-Q4_K_XL and AP-IQ4_XS)

Measured on 2026-10-21 by clothespin. Three configurations of agentionai's Signal fine-tune on one PC (RTX 4090, Ryzen 9 7950X3D,
93.5 GiB of DDR5-3600 with the firmware at its defaults), each run three or four times with one server start per run, plus a comparison of
`--adapt-*` against the engine's default and one of `--kv int8` against `--kv q4_0`. This is a results-only submission: it holds the data
behind the speeds that [#2041](https://github.com/Niko1221/Strata/pull/2041) quotes in `docs/SIGNAL_AP_Q4_K_XL.md` and `docs/MODELS.md`.
#2041 is the code and documentation change that adds `--family signal`; nothing in this folder depends on it being merged. Main limitation:
one PC with a desktop session open; setup's AP-Q4_K_XL configuration, which keeps every expert in RAM, left that PC short of memory, and its
runs have the only slow requests and the only kernel messages of the session (see [Conditions during the runs](#conditions-during-the-runs)).
Please read the speeds as that PC's, not as general numbers. This is the community guide's quick check (`tools/strata_bench.py`), not the draft
standard protocol: the protocol has a profile only for the original Qwen3.8-Flash-Next and says fine-tunes need their own.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 24564 MiB; 178 MiB in use before every run; power limit 600 W (the default, not changed); PCIe 4.0 x16
  (the engine's own probe measured 26.8 to 26.9 GB/s host to device at each start of A and B, the configurations that
  probe; the card drops to a lower link speed when idle); Resizable BAR on
  (BAR1 32768 MiB).
- CPU: AMD Ryzen 9 7950X3D (16 cores, 32 threads), at stock. Motherboard: ASRock X670E Taichi, firmware 4.43 (2026-06-24), at its defaults
  after a CMOS reset.
- RAM: 93.5 GiB usable (95712 MiB) on four DDR5 modules from two different kits: two 32 GB dual-rank modules (Corsair `CMK64GX5M2B5200C40`) and
  two 16 GB single-rank modules of another make. With the firmware's defaults they run at DDR5-3600, 1.1 V. Read while the PC was idle: the
  memory clock, the memory controller clock and the Infinity Fabric clock all at 1800 MHz, SoC voltage 1.09 V, VDDP 0.80 V (the processor's SMU
  power table, read with the `ryzen_smu` kernel module, laid out as ZenStates-Core reads table version `0x540004`), and the main timings
  CL30-30-30-58 (tRCD read and write 30, tRP 30, tRAS 58, tRC 88 and 87 on the two channels, tRFC not decoded; the memory controller's registers, decoded as ZenTimings
  does). `dmidecode` gives the modules and the configured speed and voltage.
- Storage: the model files are on a bcachefs volume on a Nextorage NE1N2TB NVMe SSD (PCIe 4.0 x4).
- Swap: `/dev/zram0` (zstd, 23.4 GB, priority 100) and an NVMe partition (77.5 GB, priority -1); `vm.swappiness` 180.
- OS: Linux 7.2.11-arch1-1; GPU driver 615.78.08; CUDA toolkit 13.4 (`nvcc` V13.4.92); Python 3.14.7 (Strata's virtual environment). The CPU
  governor was `powersave` (`amd-pstate-epp`) during every run (read into each `pre-run-state.txt`; not changed).
- Strata: every run recorded the commit `04443aeb` of the branch of #2041. That branch now ends at `c6d6fc98`, whose code is the same: only
  `docs/SIGNAL_AP_Q4_K_XL.md` and `docs/MODELS.md` differ. `tools/strata_bench.py` and `tools/ab_engine.py` are byte for byte the same there and
  on `main` (`9de04cd4` and `011c9e60`). Engine 0.1.46.3 built from source on this PC (arch 89, source hash `c87773ed5df25479`, vision encoder
  source `41b0903f5d532e38`, vision on the GPU). Every run's `pre-run-state.txt` records the commit and the engine build.
- Background workloads: a desktop session was open during all runs (the Hyprland compositor, the Steam client with no game, JetBrains Toolbox and
  the coding agent that drove the runs; no web browser); llama-swap was running with no model loaded; another GPU server on this PC was stopped.
  The five busiest processes at the start of each run are in its `pre-run-state.txt`.

## What is in this folder

- `runs/<label>/`: one benchmark run each: `results.json` (every request: tokens, timings, errors, the machine facts), `README.md` (the tool's own
  report), `engine-log-excerpt.txt`, `bench-console.txt`, `pre-run-state.txt` and `post-run-state.txt` (GPU, memory, load average, governor, busiest
  processes and the configuration at the start and the end). Labels: `pass1-` to `pass3-` for the three configurations A, B and C (C2 is a fourth
  run of C), and `kv-1-`/`kv-2-` for the `--kv` comparison.
- `ab/<label>/`: the `--adapt-*` comparison (`ab_engine.jsonl` has every request, `ab-console.txt` the console, `pre-run-state.txt` the start).
- `configs/`: the four run configurations and, for each, the engine's log file. Every start of a configuration appended to its file, so each holds
  several starts and is not split by run; each run's own start-up lines are in its `engine-log-excerpt.txt`.
- `sampler.csv`: the engine's resident memory, GPU memory, clock, power, temperature, utilization and the PC's available memory every 10 s,
  2026-10-21 (PDT) 18:48:05 to 20:02:17, the whole session.
- `hash-check.txt`: the SHA-256 of both model files, taken just before the runs, and the bcachefs checksum-error counters of the PC's two bcachefs
  volumes before and after it (they count from the creation of each filesystem).
- `kernel-log-events.txt`: the kernel messages of the whole session that matter here (see below). `scripts/`: the scripts as they were run (the
  paths in them are replaced by placeholders; `full_suite.sh` runs everything in order), and `make_tables.py` and `events_by_run.py`, which rebuild
  the tables below from the data.
- Home folders, the host name and the user name are replaced by placeholders: `<HOME>`, `<HOST>`, `<USER>`; model and data folders by
  `<MODEL_DATA>` and `<DATA>`; the Strata folder by `<STRATA_ROOT>`.

## Model and configuration

- Model: agentionai/Signal-3.8-Flash-Next-GGUF at revision `eba565be` (the revision `setup.py` pins in #2041). `AP-Q4_K_XL`: 101,142,769,984 bytes, SHA-256
  `427a7a10dd5a98d77c71be62a89d14e7408fc61a84c6efe889393852282cd3c6`. `AP-IQ4_XS`: 90,447,294,816 bytes, SHA-256
  `7fadb6b29035377204aed0d0085d3f58cc65fc7b4c3aa79f8f3f84fdfc304529`. Both files were hashed on 2026-10-21 between 18:48 and 18:50 (PDT), just before the runs (`hash-check.txt`), and match these values.
- Vision encoder: Signal's own `mmproj-F16.gguf`, loaded in A and B (`--vision`); no picture was sent in any test. C has no encoder.
- A: AP-Q4_K_XL through `setup.py --family signal --model AP-Q4_K_XL` of #2041 (2026-10-21, into an empty data folder); the config is the one setup
  wrote. B: AP-IQ4_XS the same way (2026-10-19, engine 0.1.45.3; the config is the one setup wrote then). C: AP-Q4_K_XL packed and configured by hand
  as `docs/SIGNAL_AP_Q4_K_XL.md` describes; its `--pcie-frac`, `--pool-workers` and `--spec-min-p` are what setup's calibration chose on this PC
  on an earlier date, and they were not calibrated again for these runs.
- A and C differ in six settings at once, plus the encoder, so the results do not say which of them matters: A and B have `--max-context 131072`,
  every expert kept in RAM, `--ple-gguf ... --ple-io ram`, `--spec-min-p 0.5`, and `--vision --vram-reserve-mib 700`; C has `--max-context 262144`,
  `--resident-budget-gib 67`, `--pcie-frac 0.35`, `--pool-workers 8`, `--spec-min-p 0.70` and no encoder. All use `--kv int8` (except the
  `kv-*-q4_0` runs), `--kv-resident 32768`, `--spec 4`, `--mtp` (the original model's draft head), `--prefill auto`, `--expert-cache auto`
  and `data/expert-profile.bin` (the base model's expert ranking). Reasoning off for the suite, temperature 0. The experimental speed projection
  was not used.

```text
# A (configs/setup-ap-q4_k_xl.json)
<STRATA_ROOT>/engine/strata --pack <MODEL_DATA>/ap-q4kxl/packs/signal-ap-q4_k_xl --native <MODEL_DATA>/ap-q4kxl-gguf/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --ple-gguf <MODEL_DATA>/ap-q4kxl-gguf/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --embd-gguf <MODEL_DATA>/ap-q4kxl/packs/signal-ap-q4_k_xl/token-embd-bf16.gguf --expert-profile <STRATA_ROOT>/data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <MODEL_DATA>/ap-q4kxl/mtp/rt --max-context 131072 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 700
# B (configs/setup-ap-iq4_xs.json)
<STRATA_ROOT>/engine/strata --pack <MODEL_DATA>/ap-iq4xs/packs/signal-ap-iq4_xs --native <MODEL_DATA>/ap-iq4xs/models/signal-AP-IQ4_XS/Signal-3.8-Flash-Next-AP-IQ4_XS.gguf --ple-gguf <MODEL_DATA>/ap-iq4xs/models/signal-AP-IQ4_XS/Signal-3.8-Flash-Next-AP-IQ4_XS.gguf --embd-gguf <MODEL_DATA>/ap-iq4xs/packs/signal-ap-iq4_xs/token-embd-bf16.gguf --expert-profile <STRATA_ROOT>/data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <MODEL_DATA>/ap-iq4xs/mtp/rt --max-context 131072 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 700
# C (configs/docs-handbuilt.json)
<STRATA_ROOT>/engine/strata --pack <DATA>/packs/signal-ap-q4_k_xl --native <DATA>/models/agentionai-signal/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --embd-gguf <DATA>/models/agentionai-signal/signal-ap-q4_k_xl-embd-bf16.gguf --expert-profile <STRATA_ROOT>/data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp <DATA>/mtp/rt --max-context 262144 --kv int8 --kv-resident 32768 --resident-budget-gib 67 --pcie-frac 0.35 --spec-min-p 0.70 --pool-workers 8
```

## Method

- `scripts/full_suite.sh` ran everything below in one go: first `verify_hash.sh` (both model files hashed, see `hash-check.txt`), then the runs.
  `scripts/sampler.sh` ran beside it.
- `scripts/run_one.sh LABEL CONFIG` runs `python tools/strata_bench.py --config CONFIG --out DIR` (suite 1, 3 runs per test, thinking off,
  temperature 0). The benchmark starts the server itself from the configuration, once per run; loading is not timed. Speeds are the engine's own
  timings; first-token times are measured by the client over HTTP. Short chat: the three questions of `tools/ab_engine.py`, up to 100 new
  tokens, 3 runs. 4K and 16K prompts: generated word lists, 24 new tokens, 3 fresh prompts each. Agent session: a system prompt and four tools, then
  5 further turns that each add about 1500 words of a project file and continue from the model's own earlier answers, in 3 sessions of 6 turns; the
  figures here use turns 2 to 6 of each session (the first turn has nothing to reuse). The tool's own `README.md` in each run folder has its full
  method text.
- Before every run `run_one.sh` wrote `pre-run-state.txt` (GPU, llama-swap's running models, memory, load average, governor, busiest processes) and after
  it `post-run-state.txt`. `scripts/chain.sh` and `scripts/ab_run.sh` refuse to start a run when the GPU has more than 2000 MiB in use or llama-swap
  has a model loaded.
- Order (start times are in the table of the conditions): pass 1: A, C, B; pass 2: B, C, A; then the two `--adapt-*` comparisons, then
  `--kv int8` and `--kv q4_0` alternating twice; then pass 3: B, C, C2 (a fourth run of C), A.
- `--adapt-*`: `tools/ab_engine.py`, config C, 4 rounds with the two arms alternating, `--max-new` 100 and 512; arm `default` against arm
  `adapt` (`--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`, the values the setup calibration chose). The figures are medians over the three
  short chats and the long prompt of each round (16 requests per arm). The server was ready 66 to 84 s after start in these 16 starts
  (`ab-console.txt`). `kv`: config C and the same with `--kv q4_0`, two runs of each, alternating.
- Statistics: the summary table gives the median over all requests of all runs of a configuration, with the slowest and fastest request as the range. The
  per-run tables give the median of the runs' own medians (the figure the documentation of #2041 quotes), each run's median and the range of all requests.
  `scripts/make_tables.py summary|perrun|start|ab` rebuilds all of them from `runs/` and `ab/`.

## Results

All figures are from `results.json`. Failed requests: none in any run; skipped tests: none.

### Summary (all requests of all runs of a configuration)

| Configuration | Test | Requests | Actual prompt tokens | Reused tokens | Generated tokens | Prompt tok/s median (range) | Decode tok/s median (range) | First token s median (range) |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| A  AP-Q4_K_XL, config setup writes | short chat | 27 | 46 (43-51) | 0 (0-0) | 100 (100-100) | not measured | 69.9 (46.4-78.9) | 0.49 (0.36-0.60) |
| A  AP-Q4_K_XL, config setup writes | 4K prompt, fresh | 9 | 4019 (4019-4019) | 0 (0-0) | 24 (24-24) | 1607 (718-1681) | not measured | 2.52 (2.41-5.62) |
| A  AP-Q4_K_XL, config setup writes | 16K prompt, fresh | 9 | 16058 (16058-16058) | 0 (0-0) | 24 (24-24) | 3080 (2197-3133) | not measured | 5.24 (5.15-7.34) |
| A  AP-Q4_K_XL, config setup writes | agent session, turns 2-6 | 45 | 9063 (4090-14722) | 5693 (626-11903) | 63 (17-64) | not measured | 86.3 (48.4-117.5) | 2.56 (2.36-11.14) |
| B  AP-IQ4_XS, config setup writes | short chat | 27 | 46 (43-51) | 0 (0-0) | 100 (100-100) | not measured | 91.0 (70.7-104.6) | 0.33 (0.25-0.42) |
| B  AP-IQ4_XS, config setup writes | 4K prompt, fresh | 9 | 4019 (4019-4019) | 0 (0-0) | 24 (24-24) | 2118 (1586-2135) | not measured | 1.91 (1.90-2.55) |
| B  AP-IQ4_XS, config setup writes | 16K prompt, fresh | 9 | 16058 (16058-16058) | 0 (0-0) | 24 (24-24) | 3802 (3358-3840) | not measured | 4.25 (4.20-4.81) |
| B  AP-IQ4_XS, config setup writes | agent session, turns 2-6 | 45 | 9061 (4087-14714) | 5683 (625-11895) | 64 (31-64) | not measured | 108.1 (74.6-145.7) | 2.20 (1.86-3.46) |
| C  AP-Q4_K_XL, the document's hand-built config | short chat | 36 | 46 (43-51) | 0 (0-0) | 100 (100-100) | not measured | 64.2 (50.6-77.4) | 0.43 (0.30-0.55) |
| C  AP-Q4_K_XL, the document's hand-built config | 4K prompt, fresh | 12 | 4019 (4019-4019) | 0 (0-0) | 24 (24-24) | 1690 (1314-1731) | not measured | 2.39 (2.34-3.08) |
| C  AP-Q4_K_XL, the document's hand-built config | 16K prompt, fresh | 12 | 16058 (16058-16058) | 0 (0-0) | 24 (24-24) | 2756 (2491-2786) | not measured | 5.85 (5.79-6.47) |
| C  AP-Q4_K_XL, the document's hand-built config | agent session, turns 2-6 | 60 | 9056 (4090-14724) | 5686 (626-11904) | 64 (18-64) | not measured | 78.0 (58.8-103.1) | 2.38 (2.22-3.73) |
| C with --kv int8 (the same config as C, two more runs) | short chat | 18 | 46 (43-51) | 0 (0-0) | 100 (100-100) | not measured | 60.5 (52.6-74.7) | 0.42 (0.34-0.50) |
| C with --kv int8 (the same config as C, two more runs) | 4K prompt, fresh | 6 | 4019 (4019-4019) | 0 (0-0) | 24 (24-24) | 1686 (1332-1715) | not measured | 2.40 (2.36-3.04) |
| C with --kv int8 (the same config as C, two more runs) | 16K prompt, fresh | 6 | 16058 (16058-16058) | 0 (0-0) | 24 (24-24) | 2741 (2524-2787) | not measured | 5.89 (5.78-6.38) |
| C with --kv int8 (the same config as C, two more runs) | agent session, turns 2-6 | 30 | 9064 (4090-14727) | 5657 (626-11907) | 64 (43-64) | not measured | 74.9 (56.1-95.8) | 2.40 (2.22-3.65) |
| C with --kv q4_0 | short chat | 18 | 46 (43-51) | 0 (0-0) | 100 (100-100) | not measured | 64.5 (50.6-71.4) | 0.41 (0.30-0.54) |
| C with --kv q4_0 | 4K prompt, fresh | 6 | 4019 (4019-4019) | 0 (0-0) | 24 (24-24) | 1678 (1346-1726) | not measured | 2.41 (2.35-3.00) |
| C with --kv q4_0 | 16K prompt, fresh | 6 | 16058 (16058-16058) | 0 (0-0) | 24 (24-24) | 2693 (2456-2716) | not measured | 5.99 (5.93-6.57) |
| C with --kv q4_0 | agent session, turns 2-6 | 30 | 9065 (4090-14698) | 5696 (626-11879) | 58 (33-64) | not measured | 72.1 (52.7-104.2) | 2.39 (2.22-3.71) |

### Per run

**A  AP-Q4_K_XL, config setup writes** (3 runs, failed requests: 0)

| test | median of the runs | each run's median | slowest to fastest request |
| --- | --- | --- | --- |
| short chat, decode (tokens/s) | 71.3 | 71.3, 71.5, 64.5 | 46.4-78.9 (n=27) |
| short chat, first token (s) | 0.49 | 0.49, 0.48, 0.51 | 0.36-0.60 (n=27) |
| 4K prompt (tokens/s) | 1636 | 1669, 1636, 1300 | 718-1681 (n=9) |
| 4K prompt, first token (s) | 2.48 | 2.43, 2.48, 3.11 | 2.41-5.62 (n=9) |
| 16K prompt (tokens/s) | 3080 | 3108, 3080, 2844 | 2197-3133 (n=9) |
| 16K prompt, first token (s) | 5.24 | 5.19, 5.24, 5.67 | 5.15-7.34 (n=9) |
| agent turns, decode (tokens/s) | 86.1 | 86.1, 89.4, 84.8 | 48.4-117.5 (n=45) |
| agent turns, first token (s) | 2.58 | 2.58, 2.50, 2.62 | 2.36-11.14 (n=45) |
| agent turns, prompt reused from the cache (%) | 69.4 | 69.5, 69.3, 69.4 | |

**B  AP-IQ4_XS, config setup writes** (3 runs, failed requests: 0)

| test | median of the runs | each run's median | slowest to fastest request |
| --- | --- | --- | --- |
| short chat, decode (tokens/s) | 91.0 | 90.4, 91.0, 92.1 | 70.7-104.6 (n=27) |
| short chat, first token (s) | 0.33 | 0.33, 0.33, 0.32 | 0.25-0.42 (n=27) |
| 4K prompt (tokens/s) | 2123 | 2123, 2112, 2134 | 1586-2135 (n=9) |
| 4K prompt, first token (s) | 1.91 | 1.91, 1.92, 1.90 | 1.90-2.55 (n=9) |
| 16K prompt (tokens/s) | 3802 | 3802, 3796, 3827 | 3358-3840 (n=9) |
| 16K prompt, first token (s) | 4.25 | 4.25, 4.25, 4.22 | 4.20-4.81 (n=9) |
| agent turns, decode (tokens/s) | 108.0 | 107.1, 108.3, 108.0 | 74.6-145.7 (n=45) |
| agent turns, first token (s) | 2.00 | 1.99, 2.00, 2.78 | 1.86-3.46 (n=45) |
| agent turns, prompt reused from the cache (%) | 69.3 | 69.5, 69.3, 69.3 | |

**C  AP-Q4_K_XL, the document's hand-built config** (4 runs, failed requests: 0)

| test | median of the runs | each run's median | slowest to fastest request |
| --- | --- | --- | --- |
| short chat, decode (tokens/s) | 64.1 | 62.7, 62.3, 66.1, 65.4 | 50.6-77.4 (n=36) |
| short chat, first token (s) | 0.43 | 0.46, 0.43, 0.42, 0.38 | 0.30-0.55 (n=36) |
| 4K prompt (tokens/s) | 1710 | 1646, 1723, 1696, 1724 | 1314-1731 (n=12) |
| 4K prompt, first token (s) | 2.37 | 2.46, 2.36, 2.38, 2.35 | 2.34-3.08 (n=12) |
| 16K prompt (tokens/s) | 2769 | 2743, 2780, 2772, 2765 | 2491-2786 (n=12) |
| 16K prompt, first token (s) | 5.82 | 5.88, 5.80, 5.81, 5.83 | 5.79-6.47 (n=12) |
| agent turns, decode (tokens/s) | 76.5 | 77.4, 73.3, 75.5, 83.2 | 58.8-103.1 (n=60) |
| agent turns, first token (s) | 2.39 | 2.40, 2.39, 2.38, 2.35 | 2.22-3.73 (n=60) |
| agent turns, prompt reused from the cache (%) | 69.3 | 69.3, 69.4, 69.3, 69.3 | |

**C with --kv int8 (the same config as C, two more runs)** (2 runs, failed requests: 0)

| test | median of the runs | each run's median | slowest to fastest request |
| --- | --- | --- | --- |
| short chat, decode (tokens/s) | 60.5 | 60.5, 60.6 | 52.6-74.7 (n=18) |
| short chat, first token (s) | 0.44 | 0.46, 0.42 | 0.34-0.50 (n=18) |
| 4K prompt (tokens/s) | 1688 | 1683, 1692 | 1332-1715 (n=6) |
| 4K prompt, first token (s) | 2.40 | 2.40, 2.39 | 2.36-3.04 (n=6) |
| 16K prompt (tokens/s) | 2756 | 2740, 2771 | 2524-2787 (n=6) |
| 16K prompt, first token (s) | 5.85 | 5.89, 5.82 | 5.78-6.38 (n=6) |
| agent turns, decode (tokens/s) | 73.8 | 75.0, 72.5 | 56.1-95.8 (n=30) |
| agent turns, first token (s) | 2.39 | 2.41, 2.37 | 2.22-3.65 (n=30) |
| agent turns, prompt reused from the cache (%) | 69.3 | 69.5, 69.2 | |

**C with --kv q4_0** (2 runs, failed requests: 0)

| test | median of the runs | each run's median | slowest to fastest request |
| --- | --- | --- | --- |
| short chat, decode (tokens/s) | 64.3 | 64.6, 64.0 | 50.6-71.4 (n=18) |
| short chat, first token (s) | 0.42 | 0.43, 0.41 | 0.30-0.54 (n=18) |
| 4K prompt (tokens/s) | 1695 | 1668, 1723 | 1346-1726 (n=6) |
| 4K prompt, first token (s) | 2.39 | 2.42, 2.35 | 2.35-3.00 (n=6) |
| 16K prompt (tokens/s) | 2698 | 2693, 2704 | 2456-2716 (n=6) |
| 16K prompt, first token (s) | 5.97 | 5.99, 5.96 | 5.93-6.57 (n=6) |
| agent turns, decode (tokens/s) | 70.4 | 68.6, 72.2 | 52.7-104.2 (n=30) |
| agent turns, first token (s) | 2.38 | 2.41, 2.36 | 2.22-3.71 (n=30) |
| agent turns, prompt reused from the cache (%) | 69.4 | 69.5, 69.3 | |

### `--adapt-*` against the engine's default (config C, 4 rounds)

Decode tokens/s, median over the requests of the arm:

| comparison | first arm | second arm | pairs |
| --- | --- | --- | --- |
| ab-adapt-max100 | default 60.8 (n=16) | adapt 49.8 (n=16) | default faster in 15 of 16 pairs |
| ab-adapt-max512 | default 65.7 (n=16) | adapt 59.7 (n=16) | default faster in 16 of 16 pairs |

In the 512-token comparison the three `--adapt-*` settings made the first question's answer shorter (419 tokens against 512), so the texts were
not the same there.

## Conditions during the runs

Start of each run (from `pre-run-state.txt`; the load average is the 1-minute figure at the moment of the snapshot, after the previous server had been stopped):

| run | started (PDT, 2026-10-21) | 1-minute load average at start | available memory at start (MiB) | GPU memory in use at start | seconds measured |
| --- | --- | ---: | ---: | --- | ---: |
| pass1-A-setup-ap-q4_k_xl | 18:50:15 | 1.68 | 83844 | 178 MiB | 107 |
| pass2-A-setup-ap-q4_k_xl | 19:03:43 | 5.31 | 86249 | 178 MiB | 113 |
| pass3-A-setup-ap-q4_k_xl | 19:58:07 | 4.16 | 85249 | 178 MiB | 127 |
| pass1-B-setup-ap-iq4_xs | 18:56:09 | 6.14 | 85725 | 178 MiB | 85 |
| pass2-B-setup-ap-iq4_xs | 18:58:10 | 8.55 | 85826 | 178 MiB | 85 |
| pass3-B-setup-ap-iq4_xs | 19:50:04 | 5.56 | 84759 | 178 MiB | 91 |
| pass1-C-docs-handbuilt-ap-q4_k_xl | 18:52:52 | 4.80 | 85730 | 178 MiB | 108 |
| pass2-C-docs-handbuilt-ap-q4_k_xl | 19:00:29 | 5.20 | 85874 | 178 MiB | 105 |
| pass3-C-docs-handbuilt-ap-q4_k_xl | 19:52:16 | 5.58 | 84970 | 178 MiB | 103 |
| pass3-C2-docs-handbuilt-ap-q4_k_xl | 19:55:06 | 5.82 | 84991 | 178 MiB | 105 |
| kv-1-int8 | 19:37:15 | 5.38 | 85408 | 178 MiB | 108 |
| kv-2-int8 | 19:43:44 | 7.34 | 84653 | 178 MiB | 107 |
| kv-1-q4_0 | 19:40:30 | 6.47 | 85553 | 178 MiB | 108 |
| kv-2-q4_0 | 19:47:01 | 4.57 | 85037 | 178 MiB | 106 |

Kernel messages in each run's window (`kernel-log-events.txt`; the count of each kind):

| run | start-end (PDT) | allocation failures | allocation stalls | zram swap write errors | bcachefs read errors (recovered) | NVIDIA driver: DMA mapping failed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| runs/pass1-A-setup-ap-q4_k_xl | 18:50:15-18:52:51 | 0 | 0 | 0 | 0 | 0 |
| runs/pass1-C-docs-handbuilt-ap-q4_k_xl | 18:52:52-18:56:09 | 0 | 0 | 0 | 0 | 0 |
| runs/pass1-B-setup-ap-iq4_xs | 18:56:09-18:58:10 | 0 | 0 | 0 | 0 | 0 |
| runs/pass2-B-setup-ap-iq4_xs | 18:58:10-19:00:28 | 0 | 0 | 0 | 0 | 0 |
| runs/pass2-C-docs-handbuilt-ap-q4_k_xl | 19:00:29-19:03:43 | 0 | 0 | 0 | 0 | 0 |
| runs/pass2-A-setup-ap-q4_k_xl | 19:03:43-19:07:38 | 1 | 0 | 0 | 0 | 1 |
| ab/ab-adapt-max100 | 19:07:46-19:21:49 | 0 | 0 | 0 | 0 | 0 |
| ab/ab-adapt-max512 | 19:21:57-19:37:07 | 0 | 0 | 0 | 0 | 0 |
| runs/kv-1-int8 | 19:37:15-19:40:30 | 0 | 0 | 0 | 0 | 0 |
| runs/kv-1-q4_0 | 19:40:30-19:43:44 | 0 | 0 | 0 | 0 | 0 |
| runs/kv-2-int8 | 19:43:44-19:47:01 | 0 | 0 | 0 | 0 | 0 |
| runs/kv-2-q4_0 | 19:47:01-19:50:04 | 0 | 0 | 0 | 0 | 0 |
| runs/pass3-B-setup-ap-iq4_xs | 19:50:04-19:52:16 | 0 | 0 | 0 | 0 | 0 |
| runs/pass3-C-docs-handbuilt-ap-q4_k_xl | 19:52:16-19:55:06 | 0 | 0 | 0 | 0 | 0 |
| runs/pass3-C2-docs-handbuilt-ap-q4_k_xl | 19:55:06-19:57:59 | 0 | 0 | 0 | 0 | 0 |
| runs/pass3-A-setup-ap-q4_k_xl | 19:58:07-20:02:16 | 1 | 0 | 0 | 0 | 1 |

events in the file: 10; inside no run window: 0

- The PC logged no bcachefs read error, no swap write error and no allocation stall during the session, and the checksum-error counters of both
  bcachefs volumes did not change from the start of the hash check to the end of the last run.
- Runs 2 and 3 of A each have one event while the server loaded: a page-allocation failure in the engine (an atomic allocation, `GFP_ATOMIC`),
  followed at the same second by the NVIDIA driver failing to create a DMA mapping (`Failed to create a DMA mapping!`, `osIovaMap ... status =
  0x59`, `Assertion failed: status == NV_OK @ mem_desc.c:4326`). The engine carried on, and both runs finished without a failed request.
- A keeps every expert in RAM. During its three runs the PC's available memory (`sampler.csv`) was 12.8 to 15.7 GiB (the lowest and the median
  per run), against 22.9 to 25.4 GiB during the runs of B, C and the `--kv` comparison.
- Run 2 of A (`pass2-A`): its first 4K prompt ran at 984 tokens/s (the first 4K prompt of run 1: 1431; the other 4K prompts of runs 1 and 2: 1636
  to 1681). Run 3 of A
  (`pass3-A`): a 4K prompt at 718 and a 16K prompt at 2197 tokens/s, and one agent turn that took 11.1 s to its first token (the other 44 turns of
  A: 2.4 to 6.3 s). Whether the allocation failure or the memory pressure caused these is not established; they stay in every table.
- No run of B or C stands out: each run's median for the 4K and 16K prompts is within 5% of the other runs' of the same configuration.

## Correctness and limitations

- No output was checked for correctness here: the suite measures speed. Output quality against the base model, the draft head's acceptance on this
  fine-tune, and anything with pictures were not measured. AP-IQ4_XS received no picture at all.
- One PC, one engine build, thinking off, temperature 0, short prompts (the longest is 16K tokens). The memory speed (DDR5-3600 with four modules),
  the desktop session and the `powersave` governor are part of these numbers. Windows, AMD, other cards, other memory settings and other RAM sizes
  were not tested.
- The runs are not a randomized design: the order of the configurations changed between passes but the runs of one pass followed each other.
- Config C's calibrated settings were not calibrated again for these runs (see Model and configuration).
