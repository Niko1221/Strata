# Windows 11 + Radeon RX 6900 XT (gfx1030): the released `rdna2` package on a real RX 6000 card

`docs/AMD_HIP.md` marks the RX 6000 section as having a package that was never run on a real RX 6000 card.
[#4723](https://github.com/Niko1221/Strata/pull/4723) filled that gap for the **Linux** `rx6000` package on an RX 6900 XT.
This report is the Windows side, which #4723 lists under "Not measured: the Windows `rdna2` package, and any Windows machine".

**This is not a standard protocol run.** `strata bench run` was not executed and there is no `results.json` from it.
What follows is one startup's own log lines and one timing run of a server that was already serving, both on 2026-10-31.
Read it as availability evidence for a package configuration that has no report yet, not as a comparable benchmark.

## Result in one line

The released `windows-x64-hip-rdna2` package of engine 0.1.54 starts and serves Qwen3.8-Flash-Next IQ3_S at
262,144 tokens of context on a single 16 GB RX 6900 XT under Windows 11, at 869.0 tok/s prompt and 68.5 tok/s decode
on the run below.

## Hardware and software

- GPU: AMD Radeon RX 6900 XT 16 GB, gfx1030 (Navi 21); driver `32.0.21045.16010` (2026-10-16); PCIe Gen4 x16.
  `Win32_VideoController.AdapterRAM` reports 4 GB on this card — the 32-bit field overflows, the engine sees the full 16 GB.
- CPU: AMD Ryzen 9 5950X, 16 cores / 32 threads, **no AVX-512** (the engine logs that the expert kernels run on AVX-2).
- RAM: 127.92 GiB DDR4-3200 (4 x 32 GB, XMP on).
- OS: Windows 11 Pro for Workstations, build 26300.
- Engine: **0.1.54**, `prebuilt`, backend `hip`, platform `windows-x64`, archs `gfx1030`/`gfx1031`,
  ROCm `10.2.0a20260930`, commit `8c1aea4bca1b`. No system ROCm is installed; the package carries its own runtime
  (`amdhip64_7.dll` from the engine directory).
- Model: Qwen3.8-Flash-Next **IQ3_S**, two shards, 83.6 GB, with a native experts pack and an MTP draft layer.

## Configuration

Environment (all eleven, as the server ran them):

| Variable | Value |
|---|---|
| `STRATA_HIP_PROMPT_F16` | `1` |
| `STRATA_HC_PERSIST` | `1` |
| `STRATA_ROUTE_TAIL_SKIP` | `7` |
| `STRATA_SELECT_SIMT` | `1` |
| `STRATA_ROUTE_PRIOR` | `0.5` |
| `STRATA_PREFILL_LEND_PCT` | `70` |
| `STRATA_DENSE_MMQ` | `1` |
| `STRATA_HIP_ADAPT_KERNEL_COPY` | `1` |
| `STRATA_IO_THREADS` | `32` |
| `STRATA_KV_PREFETCH` | `1` |
| `STRATA_ROCBLAS_TUNING` | `<local>/tools/hip` (self-tuned table, see below) |

Arguments: `--expert-cache auto --prefill auto:16384 --spec 4 --spec-min-p 0.7 --max-context 262144 --kv k8v4
--kv-resident 32768 --prompt-cache 2 --conversation-cache-mib 8192 --conversation-cache-slots 32 --mtp <dir>
--expert-profile <file> --expert-profile-save <file>`, plus the pack, native shard and `--ple-gguf` paths.

This is a tuned configuration, not the setup defaults. Every switch above is documented in the engine's own log lines
(`engine-start.txt` keeps them verbatim). A reader comparing against their own machine should start from the defaults.

## Timings (2026-10-31 10:39 local, server already warm)

One prefill run and one decode run, temperature 0, no repetition:

| Measure | Value | Prompt | Completion |
|---|---|---|---|
| Prompt (prefill) | **869.0 tok/s** | 9,699 tokens | — |
| Decode | **68.5 tok/s** | — | 400 tokens |
| MTP draft acceptance | **78.6 %** | — | 400 tokens |

The same command has reported 1,026 tok/s prompt on an earlier day at a longer prompt (11,470 tokens). The two are
different prompt lengths on different days and are not a regression statement; they are here because a single number
with no history reads more certain than it is.

## What the startup log says (verbatim in `engine-start.txt`)

- **Windows budgets 15,570 of this card's 16,368 MiB for this process**; free VRAM is counted within that budget.
  This is the first thing a Windows reader should look for: the desktop holds ~800 MiB the engine never sees.
- **PCIe probe: 28.0 GB/s host→device** (best of 28.0/27.9/27.9/27.9) → `pcie_frac 0.55`, the default.
- **Expert cache: 3,369 slots, 6.45 GiB of VRAM**, pre-filled from the profile in 0.8 s (9,179 MB/s).
- **KV streaming: 32,768 of 262,144 cells per QSA layer stay in VRAM**, the rest of the K/V in **2.39 GiB of pinned RAM**.
  99.85 % of KV block reads hit VRAM on a served request.
- **PLE table locked in RAM in 26.5 s** (`--ple-io ram` chosen by the residency policy; the table is 26.8 GiB).
  This is the bulk of the ~54 s from launch to ready.
- **CPU pool: 15 workers**, one per physical core, host thread on logical processor 0.
- **rocBLAS tuning enabled (112 rows, 65 with a solution, gfx1030, rocBLAS 5.8.0.efafc1c1)**, no `refused` line in the log.
- **`STRATA_HIP_PROMPT_F16=1`**: the prompt's 16-bit GEMMs run FP16 in and out; the engine states rocBLAS is tuned only
  for that on gfx103x and that it is not bit-identical to the default.
- **Large pages refused**: `VirtualAlloc error 1314` for 50,295,996,416 B, falling back to 4 KB pages. The account lacks
  `SeLockMemoryPrivilege`; see "Poor results" below.

## Poor results and failure conditions

- **Large pages are not available on this machine.** `GetLargePageMinimum` returns 2,097,152 but `VirtualAlloc` fails with
  1314 for the 46.8 GiB expert arena. The engine falls back to 4 KB pages and works. I did not measure the cost, because
  a card-matched report ([#4723](https://github.com/Niko1221/Strata/pull/4723), Linux) found that `--pool-workers` 2/3/4
  and `--ple-io mmap` all moved decode by less than 1 %, i.e. the CPU side has nothing to win on this card.
- **`--prefill auto:16384` warns on startup**: prompt chunks of 8,192 tokens or more crashed the engine on Windows HIP
  in the report behind #1630. This machine has run it without a crash, but the warning is real and the value is a
  deliberate over-request: `STRATA_PREFILL_LEND_PCT=70` holds the actual chunk at 8,448.
- **Both timings are single runs.** No ABBA pairing, no repetition. They are not evidence for any switch above.

## An observation about `--expert-cache auto` worth a maintainer's answer

Two startups on this machine log an estimate and a final count that do not agree, and the shrinking message does not
report a result:

```
expert cache auto: 9.42 GiB free, 2560 MiB reserved (+218 MiB for the draft head, +256 MiB for what is allocated after the cache) -> 2601 slots
expert cache 3369 slots, 6.45 GiB of VRAM; policy is ...
```

A second startup, with 9.33 GiB free and the same reservations, printed

```
... -> 2669 slots
only 2319 MiB free once the slots are written (reserve 2560 MiB); shrinking the expert cache
expert cache 3302 slots, 6.32 GiB of VRAM; policy is ...
```

In both cases the final slot count is **higher** than the estimate that precedes it (2,601 → 3,369 and 2,669 → 3,302),
and the shrinking line is not followed by a shrunken figure. Either the arrow figure is an intermediate bound that a
later step raises, or the reservation arithmetic and the allocation disagree. **I cannot tell which from the log, and I
have not changed anything to find out.** For a reader on a smaller card this matters: the two numbers differ by ~29 %.

The same 262,144-token context holds **4,647 slots** on the Linux `rx6000` package of the same engine in #4723, against
3,369 here. ~800 MiB of that is the Windows desktop budget above; the rest (`--prompt-cache 2`,
`--conversation-cache-mib 8192`, `--expert-profile-save`) is this machine's own configuration and was not isolated.

## Not measured

- The standard protocol (`strata bench run`), the twelve sanity checks, and any tool-call check.
- Concurrency, long soaks, and the serving suites.
- Images: `strata-vision` was not loaded. #4723 reports that at 16 GB the image reader runs on the CPU on gfx1030.
- Clocks, power, temperature, and per-request VRAM.
- A formal A/B against the Linux package, or against the setup defaults on this same machine.
- Any second card: Windows + HIP multi-GPU is Linux-only in setup (`--gpus` fails), and this machine has one card.

## Files

- `engine-start.txt` — the startup lines quoted above, verbatim, with local paths shortened to `<local>`.
- `config.json` — the same configuration in machine-readable form, sanitized the same way.
- `measurements.json` — the two timing numbers above with their prompt lengths and the command that produced them.
