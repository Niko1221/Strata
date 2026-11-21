# Community benchmark on AMD Radeon RX 9070 XT (Windows, Gyro-S, 31 GiB RAM), app 0.2.17

Measured on 2026-11-21 by @ric20527. This is the same PC and model as #6161 (2026-11-17, app 0.2.14, engine 0.1.60), four
days and three releases later. It uses the official Windows HIP `rx9070` engine 0.1.63 with Qwen3.8-Flash-Next Gyro-S at
a 131,072-token context, and the hand-tuned configuration this PC uses every day. There are two runs, one after the
other, each from a freshly started engine: the quick suite (`quick/`) and the standard benchmark (`standard/`, status
complete, 12 of 12 sanity checks). Main limitation: one PC, one model, one run of each suite.

## Hardware and software

- GPU: AMD Radeon RX 9070 XT 16 GiB (gfx1201), PCIe Gen5 x16; driver 32.0.32029.2010
- CPU: AMD Ryzen 5 7600X3D (6 cores, 12 threads, AVX-512); RAM: 31.4 GiB DDR5-6000; page file on the system SSD
- Model files on an SK Hynix HFS512GDE9X084N NVMe (512 GB)
- OS: Windows 11 Pro 10.0.26300
- Strata: desktop app 0.2.17, control 0.1.63 (commit `12d491da`), official engine package
  `0.1.63-windows-x86_64-hip-rx9070`, not a source build
- Background: the Claude desktop app open; Chrome and the Strata desktop app closed; no other GPU work. In #6161 Chrome
  was open as well, so this run had about 1-2 GB more free RAM at load.

## Model and configuration

- `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` (sha256 `c6ea1b08ca312f99...`), native pack `gyro-gyro-s`, DFlash2 drafter
  `qwen3.8-flash-next-dflash2.q4_0-selq4.pack` with `draft_vocab_latin.bin`, no MTP layer (`--draft-fallback none`)
- Context 131,072, `--kv int8`, `--kv-resident 32768`, `--resident-experts`, vision on (`strata-vision` on the CPU)
- Expert profile: a copy of the one this PC's daily config has learned (`expert_profile_save`), so the starting set of
  hot experts is not the generic one. The copy was used so the benchmark traffic did not change the daily file.

What the configuration adds to the app's own arguments (the full list is in `standard/config.json`):

| Setting | Value | Since |
| --- | --- | --- |
| `STRATA_HIP_WMMA` | `1` | #6161 |
| `vram_elastic` (config) | `false` | #6161 (see #6164) |
| `--vram-reserve-mib` | 1024 | #6161 |
| `--resident-budget-gib`, `STRATA_RESIDENT_PIN`, `STRATA_PARTIAL_PIN`, `STRATA_PARTIAL_PIN_GIB` | 17, `1`, `1`, `8` | #6161 |
| `STRATA_UNBUFFERED_LOAD` | `0` | #6161 |
| `STRATA_PREFILL_COPY_EVENT_EVERY`, `STRATA_PREFILL_COPY_WAIT_DEDUPE` | `16`, `1` (#4005) | new |
| drafting | DFlash2 instead of MTP | new |
| `--spec`, `--draft-conf` | 6, 0.6 | new (#6325) |
| `--pcie-frac` | 1.0 (auto settles at 0.90 here, #6325) | new |
| `--prefill` | 8192 (auto stays at 6144 on Windows) | new |
| `STRATA_ROUTE_TAIL_SKIP` | `7` (0.1.63's control writes `0`) | new |
| `--prompt-cache-spread` | on | new |
| `--conversation-cache-mib`, `--conversation-cache-min-free-mib` | 0, 0 | new |

`--adapt-async 1` is **not** in this configuration. This PC used it from 2026-11-17 to 2026-11-21 (with
`STRATA_WIN_HIP_RISKY=1` on 0.1.63). It was removed after it reset the display driver here too, as #5672 describes for an
RX 7900 XTX (details below). A single run of each suite with and without it shows no consistent difference
(see the last section).

Placement (engine log): 8,734 expert slots in VRAM (8.54 GiB), 17.00 GiB of experts resident in RAM, 8.00 GiB of it
registered for the GPU. The engine command line in `standard/` shows every flag above in effect. The quick suite's
README prints only the config's `args`; the same `user_args`, `--prompt-cache-spread` included, were in effect there too
(checked in the engine log).

## Method

- `quick/`: `python tools/strata_bench.py --config <file> --long` from the app's control, three runs per test, thinking
  off, temperature 0. Started 01:14 local time.
- `standard/`: `strata bench run --profile <same file> --track custom` (protocol `strata-qwen38-thinking-v1-draft.1`),
  started 01:18, right after, from a freshly started engine. Time after the model loaded: 1 min 55 s.
- Model loading is not timed. Each folder's README describes its method in full.

## Results

Quick suite, median (range):

| Test | #6161 tuned (0.1.60, 2026-11-17) | This run (0.1.63) | Change |
| --- | ---: | ---: | ---: |
| short chat decode tok/s (n=9) | 71.3 (60.8-89.9) | 87.7 (66.0-125.3) | +23 % |
| short chat first token s | 0.35 | 0.33 (0.29-0.37) | -6 % |
| 4K fresh prompt tok/s (n=3) | 1236.2 | 1708.0 (1465.1-1708.7) | +38 % |
| 16K fresh prompt tok/s (n=3) | 1571.3 | 2410.6 (2376.8-2416.4) | +53 % |
| 32K fresh prompt tok/s (n=3) | 1568.9 | 2477.0 (2474.3-2477.3) | +58 % |
| agent session cache hit % | 70.8 | 70.8 (70.6-70.8) | 0 |
| agent turn first token s (n=15) | 1.93 | 1.98 (1.55-2.21) | +2 % |
| agent decode tok/s (n=15) | 60.4 | 83.5 (38.2-132.7) | +38 % |

Standard benchmark (thinking on, the model card's sampling), median (range), n=3:

| Case | #6161 tuned (0.1.60) | This run (0.1.63) |
| --- | ---: | ---: |
| short decode tok/s | 83.7 (76.9-84.4) | 94.6 (83.2-97.3) |
| 4K prompt tok/s | 1276.7 (1175.6-1278.5) | 1866.5 (1846.4-1867.5) |
| 16K prompt tok/s | 1608.3 (1602.5-1612.1) | 2494.0 (2483.0-2514.0) |
| decode on the 16K case tok/s | 81.2 (78.6-87.9) | 95.2 (91.7-98.3) |
| sanity checks | 12 of 12 | **12 of 12** |

Four of nine performance requests reached the 512-token cap while reasoning, as the protocol allows (the same count as
in #6161); no request failed.

The #6161 columns are from a different day, not a same-day control. The change against them is the sum of the new
settings and three engine releases, and this run had a little more free RAM at load. I did not run the app's default
configuration or the #6161 configuration on 0.1.63, so the table does not split the gain between settings and
releases. The per-setting measurements are in #6325 (spec/draft-conf/pcie-frac, paired sessions with confidence
intervals).

## The driver resets with `--adapt-async 1`

Earlier the same night, a first run of these two suites with `--adapt-async 1` (and `STRATA_WIN_HIP_RISKY=1`) finished
cleanly. Two engine starts later, in a prefix-reuse test with the same configuration, the engine died during a prompt
read: the engine log shows
`iq_dequant_gu_f16: unspecified launch failure` on the first chunk of a 2,259-token prompt, right after eight rounds of
the asynchronous adaptive tier. Windows wrote a `VIDEO_ENGINE_TIMEOUT_DETECTED` (0x141) live dump in `amdkmdag.sys`.
Seven more 0x141/0x117 dumps from 2026-11-19, a day of heavy benchmarking with the option on, fail at the same offset
in `amdkmdag.sys`; the last one ended in a 0x7E bugcheck in `dxgkrnl.sys`. That day also had a driver reinstall (DDU),
so not every reset can be pinned on the option. **None of these resets logged System event 4101**; they show only as
dumps in `C:\Windows\LiveKernelReports`. The first run's numbers are not published here. Against the run above they went
both ways by up to ~10 % (short decode 88.3 vs 94.6 in the standard suite, 95.8 vs 87.7 in the quick one; prompts
2-7 % lower), so one run of each does not show a consistent cost or gain from the option.

## Correctness and limitations

- The standard run's twelve sanity checks passed. The quick suite is speed only.
- One PC, one model (Gyro-S TQ1_0), one run of each suite. No driver reset or engine error in either run.
- Speed depends on free RAM at load. With about 7 GiB less available, the RAM budget is clipped and the first long
  prompt reads 6-10 s slower.
- Paths are replaced by `<user>` / `<strata>` placeholders where the tools did so. The Spanish folder name
  (`Modelos Ia`) is the owner's.
