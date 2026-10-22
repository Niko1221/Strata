# Community benchmark on AMD Radeon RX 7900 XTX, Windows 11, Strata 0.2.2

Three Qwen3.8-Flash-Next quantizations on one RX 7900 XTX with the release's Windows `hip-rdna3` engine package
(0.1.47), measured with `tools/strata_bench.py --long` (3 runs per test, medians with ranges) and
`tools/needle_bench.py`: IQ3_XXS and IQ2_XS on 2026-10-21, Swift 1.5 IQ3_S re-run on 2026-10-22 (see the notes). Each
model has its own folder with the tool's README, `results.json` and `needles.json`; the 2026-10-21 runs also have a
`telemetry.csv` of the background load.

The 0.2.1 and 0.2.2 release notes say the Windows `rdna3` package had not been run on an RX 7000 card by the
maintainers: on this PC it installs, starts and serves all three models (fresh start each, no errors in the engine log).

## Hardware and software

- GPU: AMD Radeon RX 7900 XTX 24 GiB (gfx1100), PCIe 4.0 x16, driver 32.0.32015.7029; the CPU's integrated
  Radeon is present and not used (`"gpu": 1` in the config)
- CPU: AMD Ryzen 9 9950X (16 cores, 32 threads); RAM 62.7 GiB; models on a Samsung 990 PRO 4 TB (NVMe)
- OS: Windows 11 (10.0.26300)
- Strata 0.2.2 from the source zip of tag v0.2.2 (`setup.py --update`), engine 0.1.47
  `strata-engine-0.1.47-windows-x86_64-hip-rdna3` (SHA-256 checked); hipBLASLt table `gfx1100-hipblaslt-100500.txt`
  used (the engine log prints `prefill gemm: hipBLASLt tuning enabled (32 rows, gfx1100, version 100500)`)

## Configuration (all three)

`--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <draft layer> --max-context 131072 --kv int8`,
`STRATA_HC_Q8=1`; thinking off and temperature 0 for the suite (the tool's defaults). The full command line is in each
model's README.

## Results

| Model | Short chat decode tok/s | Agent turns decode tok/s | Prompt tok/s 4K / 16K / 32K | Needles 4K / 32K / 100K |
| --- | --- | --- | --- | --- |
| [Swift 1.5 IQ3_S](swift-iq3_s/) (2026-10-22) | 70.9 (58.3-73.7) | 69.3 (24.4-104.9) | 956 / 1,207 / 1,172 | 9 of 9 |
| [IQ3_XXS](iq3_xxs/) | 80.3 (63.9-87.6) | 86.2 (71.6-118.1) | 1,113 / 1,362 / 1,317 | 9 of 9 |
| [IQ2_XS](iq2_xs/) | 94.5 (82.3-120.7) | 107.4 (51.5-124.7) | 1,332 / 1,491 / 1,415 | 9 of 9 |

Short chat: 100 new tokens, 9 requests; agent: follow-up turns of a 5-turn session, 64 new tokens each, 3 sessions.
Prompt tok/s: fresh prompts of 4,019 / 16,058 / 32,049 tokens, 3 each, ranges in the model READMEs (every run within 3 % of its median).

## Notes and limitations

- Swift IQ3_S is about 82 GB of files on a PC with 62.7 GiB of RAM: part of the experts is read from the SSD. Its
  2026-10-21 run (short chat 39.7 tok/s, needles 9 of 9) had Windows Defender and memory compression busy, so it was
  re-run on 2026-10-22; the table has the re-run. Background load was not recorded for the re-run.
- Some engine starts read every prompt longer than a few dozen tokens wrongly until the server is restarted (the
  symptom of #3431): two full Swift IQ3_S runs on 2026-10-22 found 0 of 9 needles and answered as if the prompt were
  something else, while short chat questions looked normal. Those runs are left out. Counting fresh starts with a 4K
  needle check: 1 bad start in 10 with the default settings and 1 in 10 with `STRATA_PLE_BATCH=0` (the variable did
  reach the engine: prompts read about 4 % slower). The reported runs were checked: needles 9 of 9 and normal agent
  answers.
- Swift IQ3_S engine log: `expert arena: locked 2365 MiB via working-set minimum + VirtualLock; cudaHostRegister of
  the whole arena FAILED (invalid argument); 46 slices pinned (44 GiB)` - the sliced fallback worked and the model
  served normally. IQ3_XXS and IQ2_XS registered their arena whole. All three log `large pages refused ...
  (VirtualAlloc error 1314); using 4 KB pages` (the account has no lock-pages privilege).
- Windows Defender (`MsMpEng`) used about 50-65 % of one core through the 2026-10-21 runs; it was not paused.
- Speed suite only, plus the needle test; no task-accuracy suite was run. This is not the draft standard benchmark
  (`strata bench run`).
- Single PC, one reported run of the suite per model.
