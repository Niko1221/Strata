# Community benchmark: RTX 3070 Ti 8 GB, Ryzen 9 5900X, 64 GB DDR4-3600 (Windows 11), Strata 0.2.0-beta

Measured on 2026-10-19 by [gbwzzy218](https://github.com/gbwzzy218) on a native-Windows desktop whose only GPU also
drives the display. A re-run of the [2026-10-07 report](../2026-10-07-community-rtx3070ti-8gb/README.md) (engine
0.1.40.3) on `v0.2.0-beta` (tag commit `3d1fb48`, release engine 0.1.45.3), with the same method and the same arms,
plus one new arm with the CPU prefill share turned off. On 0.1.40.3, `--kv-resident 20480` (arm C) read prompts about
40% faster than setup's 32,768 (arm B) at 128K with k8v4. On 0.2.0-beta that gap is gone: C was within -2.8% to +2.7%
of B in three within-session comparisons, because the CPU share that 0.1.41 turned on by default for prompt chunks
under 1,024 tokens now lifts B by 36-38%. Main limitations: one machine, one model, configurations measured as blocks
(one server start each), and a display that was on for session 1 and off for session 2.

## Summary

Session 1 (display on), blocks 1-4, five measured requests per prompt length, medians (ranges in [Results](#results)).
Every request generated 256 tokens and reported zero reused prompt tokens. All arms carry the same hand-added
conversation cache on top of the context and KV settings 0.2.0-beta's setup wrote.

| Arm / block | Context; KV; `--kv-resident` | Prompt tok/s at 4,096 / 16,384 | Decode tok/s at 4,096 / 16,384 |
| --- | --- | --- | --- |
| A / 1 - setup's context and KV defaults here | 32,768; int8; none (KV fully in VRAM) | 542.8 / 586.0 | 49.3 / 48.2 |
| B / 2 - setup, `--context 131072 --kv k8v4` | 131,072; k8v4; 32,768 | 485.9 / 498.8 | 46.8 / 47.8 |
| C / 3 - B with `--kv-resident 20480` | 131,072; k8v4; 20,480 | 498.9 / 493.6 | 49.8 / 49.7 |
| D / 4 - setup, `--context 131072` (int8) | 131,072; int8; 32,768 | 438.6 / 432.5 | 52.1 / 48.5 |

- C against B: +2.7% / -1.0% (blocks 3 and 2), +2.6% / -1.2% (blocks 6 and 5), and in session 2 with the order
  reversed +1.7% / -2.8% (blocks 7 and 8), at 4,096 / 16,384 tokens. On 0.1.40.3 the same comparisons measured
  +38.8% to +41.4%.
- The CPU share is what changed: in session 2, B with `STRATA_PREFILL_CPU_SHARE=0` (block 9) read 424.7 / 442.8 tok/s
  against B's 586.6 / 602.8 (block 8), so the share is worth +38.1% / +36.1% to B here. C was +40.4% / +32.3% above
  the share-off B, close to the C-over-B gap 0.1.40.3 measured.
- Against A, B read prompts 10.5% / 14.9% slower, D 19.2% / 26.2% slower and C 8.1% / 15.8% slower.
- Decode block medians at 4,096 and 16,384 tokens ranged from 46.8 to 52.1 tok/s in session 1, with overlapping
  ranges; these runs do not establish a decode difference between the arms.
- With C, three 128,000-token prompts read at a median 464.6 tok/s [463.7-466.7], decoded at 51.0 tok/s [50.4-51.2],
  with a 275.6 s time to first token. Six needle checks all returned their targets (three near 32.0K tokens, three
  near 124.1K tokens), and the 13-prompt task check passed every scored item.

## What changed for this card between 0.1.40.3 and 0.2.0-beta

The engine's plans are the same as on 0.1.40.3: B still gets 768-token prompt chunks and D 512, below the 1,024 from
which the prompt path streams every non-resident expert ahead, so both still use the 8-slot staging ring. What is new
is in `src/prefill/prefill.cpp` (lines 194-198 of this tree): since 0.1.41, CUDA builds with one GPU and no batch
slots hand part of the experts of a chunk under 1,024 tokens to the idle CPU pool (`STRATA_PREFILL_CPU_SHARE` unset
behaves as `auto` with `STRATA_PREFILL_CPU_SHARE_MAX=1024`; `STRATA_PREFILL_CPU_SHARE=0` gives the 0.1.40.3 output).
Every start except the share-off arm logged "prefill: the CPU share is ON by default for prompt chunks below 1024
tokens".

What the engine planned at each start ([`engine-start.txt`](engine-start.txt)), with the 0.1.40.3 values for the same
arm in brackets:

| Arm (blocks) | VRAM free for the cache | Expert-cache capacity | Prompt chunk | Ring printed | Slots lent to the prompt path | VRAM free after load | KV in pinned RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A (1) | 2.21 GiB [2.24] | 1,023 [1,040] | 1,280 [1,280] | 351 [351] | 873 [873] | 334 MiB [432] | - |
| B (2, 5, 8) | 2.14 GiB [2.16] | 968 [985] | 768 [768] | 8 [8] | 458 [432] | 337-408 MiB [424-479] | 1.20 GiB |
| C (3, 6, 7, 12) | 2.25 GiB [2.27] | 1,051 [1,067] | 1,024 [1,024] | 384 [384] | 900 [900] | 337-408 MiB [429-481] | 1.20 GiB |
| D (4) | 2.05 GiB [2.07] | 902 [918] | 512 [512] | 8 [8] | 398 [372] | 337 MiB [481] | 1.55 GiB |
| B-share-off (9) | 2.14 GiB | 968 | 768 | 8 | 458 | 408 MiB | 1.20 GiB |

Every arm has 16-17 fewer expert-cache slots than on 0.1.40.3, and B and D lend 26 more slots to the prompt path. The
two values of "VRAM free after load" for B and C are session 1 (337-338 MiB, display on) and session 2 (408 MiB,
display off). Repeated arms printed the same plan at every start.

## Hardware and software

Probes, hashes and setup excerpts are in [`system.txt`](system.txt).

- **GPU:** NVIDIA GeForce RTX 3070 Ti, 8,192 MiB, compute capability 8.6, PCIe 4.0 x16, power limit 290 W (stock);
  it also drives the Windows desktop. Clocks not fixed.
- **CPU:** AMD Ryzen 9 5900X, 12 cores / 24 threads (Zen 3), AVX2, no AVX-512. 11 expert-pool workers plus the host
  thread.
- **RAM:** 63.9 GiB reported; 4 x 16 GiB DDR4-3600 (F4-3600C16-16GTZNC), configured at 3600, dual channel.
- **Storage:** Samsung 970 EVO Plus 2 TB NVMe.
- **OS and driver:** Windows 11 Pro, build 26200, system-managed page file, large pages not granted
  (`VirtualAlloc error 1314`, 4 KB pages); NVIDIA 617.42; `.venv` CUDA wheels `nvidia-cublas` 13.0.2.14 and
  `nvidia-cuda-runtime` 13.0.96 (the engine also ships its own cuBLAS 13 DLLs).
- **Strata:** `git clone --branch v0.2.0-beta` -> `3d1fb48090e8` (`CMakeLists.txt` project version 0.1.45.3), installed
  beside the 0.1.40.3 folder; setup found the same data folder and reused the model files, the pack and the draft layer.
- **Engine:** the release asset `strata-engine-0.1.45.3-windows-x86_64-cuda-cuda13.tar.gz` of `v0.2.0-beta` (SHA-256
  `1b3845ec...`, as in the release's `SHA256SUMS.txt`; every file matches its `engine-component.json`), placed in
  `engine\` by hand. The tag's `setup.py` looks for the pre-0.2 asset name (`strata-windows-x64.zip`) under `v0.1.45.3`
  and on the latest release, found neither (HTTP 404) and would have compiled instead; with the release engine in place
  it printed "ready-made engine already installed". `BUILD.json`: version 0.1.45.3, CUDA 13.0, archs 75/86/89/120,
  commit `f6d65cb584c0` (the component notes say "0.2.0-beta tester build"); the start line says "engine 0.1.45";
  `strata.exe` SHA-256 `0a38ce07a1e9872bbce13a2d9889e44fb5b2799db251d142031ce85a68486420`. The server and setup are
  the `3d1fb48` tree; that the binary was built from the same source is assumed, not checked.
- **Background:** before session 1 the ChatGPT desktop app, Discord, Steam, NZXT CAM and SteelSeries GG with its
  capture service were closed. Still running: an idle browser, the Claude desktop app, ASUS LightingService (about
  0.6 of a core in every block, see [`monitor/`](monitor/)), SteelSeries GG helpers that restarted themselves, system
  services. Nobody used the PC during the runs.

## Model and configuration

- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (setup's pin, the
  same as in 0.1.40.3): IQ2_XS shards of 39,225,954,592 and 28,800,138,432 B; the native pack and the MTP draft layer
  prepared on 2026-10-07 by the 0.1.40.3 installer and reused; the default draft vocabulary (with CJK); the shipped
  `data/expert-profile.bin` (SHA-256 `8f59b4aa...`, unchanged).
- No vision encoder, no low-RAM mode, no calibration, experimental speed projection off. 0.2.0-beta's setup now marks
  IQ3_S as the recommended size for this PC (its memory tiers); `--model IQ2_XS` was given explicitly, as before.

The configs were written by 0.2.0-beta's own setup, from the new folder, and copied after each run
([`configs/setup-diff.md`](configs/setup-diff.md), [`build_arms.py`](build_arms.py)):

| Arm | Written by | `--max-context` | `--kv` | `--kv-resident` | By hand |
| --- | --- | ---: | --- | ---: | --- |
| A | `START-HERE.bat --yes --family qwen --model IQ2_XS --vision no --no-start` | 32768 | int8 | - | conversation cache |
| B | `START-HERE.bat --setup --family qwen --model IQ2_XS --context 131072 --kv k8v4 --vision no --yes --no-start` | 131072 | k8v4 | 32768 | conversation cache |
| C | B's config | 131072 | k8v4 | 20480 | conversation cache, `--kv-resident 20480` |
| D | `START-HERE.bat --setup --family qwen --model IQ2_XS --context 131072 --vision no --yes --no-start` | 131072 | int8 | 32768 | conversation cache |
| B-share-off | B's config | 131072 | k8v4 | 32768 | conversation cache, env `STRATA_PREFILL_CPU_SHARE=0` |

The conversation cache is `--conversation-cache-mib 4096 --conversation-cache-slots 4`, as in 0.1.40.3. Against the
0.1.40.3 report's `configs/`, the engine arguments setup wrote are identical, word for word, for A, B and D; the only
differences setup made are the folder paths (`exe`, `cwd`, `lib_dirs`). D was edited by hand in 0.1.40.3 and is
written by setup here, with the same arguments. Per-block configs are in [`configs/`](configs/), the five arms in
[`configs/arms/`](configs/arms/).

## Method

- **Harness check:** before the runs, one 1,023-token request through the harness on arm A read the engine's
  `/metrics` counters as before (prompt_ms, decode_ms, generated, reused, drafts): [`harness-check.json`](harness-check.json).
- **Harness:** the community [`benchmark.py`](../2026-09-30-community-rtx-5090/benchmark.py), the same file as in
  0.1.40.3 apart from line endings: a nonce at the front of each request, the length set with Strata's tokenizer,
  greedy, `reasoning_effort "none"`, a 256-token cap, streaming, one short warm-up per invocation (excluded). TTFT is
  the first non-empty streamed delta, always answer text; TTFT and total are client-side over loopback. Prompt tok/s =
  fresh tokens / `prompt_ms`, decode tok/s = `engine_generated` / `decode_ms`, from the engine's `/metrics`
  (per request in `results.json`; the engine's own timing lines are not attached).
- **Requests:** 98 measured requests. Each generated 256 tokens, ended with `finish_reason "length"` and reported zero
  reused prompt tokens; 18 distinct requests repeat across blocks, as in 0.1.40.3. Requests within a block share the
  engine's cache state; each block is a new engine process; OS file caches were not flushed.
- **Order:** session 1 (17:23-18:29): 1-A, 2-B, 3-C, 4-D, 5-B, 6-C (6-C also 32,768 x 5, 128,000 x 3 and
  `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`). Session 2 (18:30-18:44), order reversed: 7-C, 8-B,
  9-B-share-off. Then the task check on C (block 12). Timestamps: [`orchestrator-session1.txt`](orchestrator-session1.txt),
  [`orchestrator-session2.txt`](orchestrator-session2.txt), [`orchestrator-taskcheck.txt`](orchestrator-taskcheck.txt);
  driver script [`bench_orchestrator.py`](bench_orchestrator.py), analysis [`analyze.py`](analyze.py).
- **E1 / E2 (the 0.1.40.3 isolation arms) were not run:** they were planned only if C was still clearly faster than B,
  and it was not (session 1: +2.6% to +2.7% at 4,096 tokens, -1.0% to -1.2% at 16,384).
- **Monitor (new):** every 2 s during each block, `nvidia-smi` (GPU utilization, SM clock, power, temperature, clock
  event reasons) and the CPU time of every process other than Strata's ([`monitor/`](monitor/)). A request counts as
  disturbed when other processes used more than 2.31 cores during it (the median request's 1.31 + 1) or the GPU
  reported a slowing reason (SW power cap, SW / HW thermal slowdown, HW slowdown or power brake). The flag and the
  values are in `results.json`; 28 of 98 requests are flagged, almost all for the GPU's SW thermal slowdown (0x20) or
  SW thermal + power cap (0x24) at 73-83 °C.
- **Memory and temperature:** snapshots after load and after each block, not peaks.

## Results

Every run is in [`results.json`](results.json); per-block medians and ranges (all runs and clean runs only) in
[`summary.json`](summary.json). Drafts accepted is the aggregate accepted / offered.

| Block | Arm | Prompt tokens | Runs | Prompt tok/s median [range] | Decode tok/s median [range] | TTFT s, median | Total s, median | Drafts accepted |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | A | 4,096 | 5 | 542.8 [529.4-547.0] | 49.3 [46.6-50.0] | 7.6 | 12.8 | 67% |
| 1 | A | 16,384 | 5 | 586.0 [585.6-586.6] | 48.2 [47.4-49.2] | 28.0 | 33.3 | 65% |
| 2 | B | 4,096 | 5 | 485.9 [472.8-490.3] | 46.8 [46.1-50.2] | 8.5 | 13.9 | 65% |
| 2 | B | 16,384 | 5 | 498.8 [496.2-500.6] | 47.8 [45.0-50.8] | 32.9 | 38.2 | 65% |
| 3 | C | 4,096 | 5 | 498.9 [496.9-511.4] | 49.8 [43.7-53.1] | 8.2 | 13.4 | 66% |
| 3 | C | 16,384 | 5 | 493.6 [492.4-494.5] | 49.7 [48.0-51.6] | 33.2 | 38.4 | 65% |
| 4 | D | 4,096 | 5 | 438.6 [432.4-440.9] | 52.1 [45.4-54.1] | 9.4 | 14.3 | 69% |
| 4 | D | 16,384 | 5 | 432.5 [431.8-435.0] | 48.5 [47.4-50.4] | 37.9 | 43.0 | 65% |
| 5 | B | 4,096 | 5 | 487.8 [478.5-492.8] | 50.5 [46.2-52.7] | 8.4 | 13.4 | 70% |
| 5 | B | 16,384 | 5 | 497.4 [495.8-497.8] | 49.6 [45.9-55.3] | 33.0 | 38.1 | 70% |
| 6 | C | 4,096 | 5 | 500.6 [500.2-505.6] | 50.4 [45.9-50.8] | 8.2 | 13.2 | 66% |
| 6 | C | 16,384 | 5 | 491.2 [490.3-493.9] | 48.6 [47.8-50.9] | 33.4 | 38.5 | 64% |
| 6 | C | 32,768 | 5 | 484.2 [482.6-485.8] | 47.3 [45.5-48.4] | 67.7 | 73.1 | 63% |
| 6 | C | 128,000 | 3 | 464.6 [463.7-466.7] | 51.0 [50.4-51.2] | 275.6 | 280.6 | 72% |
| 7 | C | 4,096 | 5 | 596.3 [584.1-599.6] | 68.7 [64.4-74.8] | 6.9 | 10.6 | 66% |
| 7 | C | 16,384 | 5 | 585.9 [583.6-586.8] | 70.2 [69.2-72.1] | 28.0 | 31.6 | 67% |
| 8 | B | 4,096 | 5 | 586.6 [580.6-590.8] | 66.7 [66.0-71.8] | 7.0 | 10.8 | 66% |
| 8 | B | 16,384 | 5 | 602.8 [600.8-603.6] | 70.8 [69.4-74.2] | 27.2 | 30.8 | 70% |
| 9 | B-share-off | 4,096 | 5 | 424.7 [422.5-425.7] | 68.7 [65.8-70.7] | 9.7 | 13.4 | 65% |
| 9 | B-share-off | 16,384 | 5 | 442.8 [442.4-443.6] | 68.1 [65.8-72.8] | 37.0 | 40.8 | 66% |

- **Within a session:** every prompt-rate run lies within 2.7% of its block median. The repeated starts of B (blocks 2
  and 5) differ by under 0.4% in prompt rate, those of C (3 and 6) by under 0.5%. With the disturbed requests left
  out, C against B is +2.7% / -1.0%, +4.5% / -1.1% and +1.7% / -2.8% for the three pairs: the same result.
- **Between the sessions:** session 2 read prompts 18.7-20.8% faster and decoded 38.1-48.1% faster than session 1,
  for the same arms (C 498.9 / 493.6 -> 596.3 / 585.9, B 485.9 / 498.8 -> 586.6 / 602.8). The display turns off after
  60 minutes without input; the last input was at about 17:28, and the system log shows the session change at 18:27:59,
  between the two sessions ([`system.txt`](system.txt)). In session 1 the Claude desktop app (about 0.2 of a core) and
  the desktop compositor show up in the monitor and are gone in session 2, and the engine's free VRAM after load rose
  from 337-338 to 408 MiB. GPU clock (1,965-1,980 MHz busy medians), power, temperature and available RAM were alike in
  both. The display state is the most likely cause, but it was not tested on its own; comparisons in this report are
  made within a session.
- **Generated text:** inputs and output lengths matched across blocks, but none of the 10 repeated requests produced
  identical text between any two blocks (0.1.40.3: 3 and 4 of 10), the share-off arm included.
- **Prompt rate across lengths (C, block 6):** 500.6, 491.2, 484.2 and 464.6 tok/s at 4,096, 16,384, 32,768 and
  128,000 tokens.
- **Memory:** `strata.exe` resident set 35.0-36.7 GiB after load, up to 39.6 GiB after the runs; available RAM 14.2-14.8
  GiB after load, lowest 11.5 GiB (after the 128K runs); VRAM in use 7,346-7,440 MiB of 8,192 after load, up to 7,557
  after the runs. Every start logged "that is little room for the verify windows' buffers under WDDM"; all 98 measured
  requests completed.

## Recall and task check

- **Recall:** `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90` on C (block 6) returned the exact target in
  all six cases, at 32,045-32,046 and 124,125-124,127 prompt tokens, one attempt each ([`needles.json`](needles.json)).
  The haystack is built from the repository's own files, which changed since 0.1.40.3, so the prompts differ from that
  report's; the elapsed times (67-68 s and 219-268 s) include prefix reuse between depths and are not throughput.
- **Task check:** the 13 prompts of [`quality/strata_eval.py`](quality/strata_eval.py), built from the same files as in
  0.1.40.3, once on C (block 12, session 2), scored by [`quality/check_results.py`](quality/check_results.py)
  ([`quality/task-check-C.scored.txt`](quality/task-check-C.scored.txt)): the evaluator passes 15/15 expressions and
  rejects 4/4 malformed inputs, the JSON parses and matches the schema, the poem has four lines of seven characters,
  the tool call is `get_weather` with Beijing (written in English) in Celsius, the word problem contains 24/7, both logic
  answers start with 1, the needle answer is exact and the bug fix proposes `lo = mid + 1`. The factual answers, the
  translation and the summary are printed in the scored file and were judged by reading.

## Compared with 0.1.40.3

Measured on different days (2026-10-07 and 2026-10-19) on the same PC, model files, pack and arguments; the display
state was not recorded on 2026-10-07. Session 1 of each report, 4,096 / 16,384 tokens:

| Arm | 0.1.40.3 prompt tok/s | 0.2.0-beta prompt tok/s | Change | Plan (both) |
| --- | --- | --- | --- | --- |
| A | 549.8 / 645.9 | 542.8 / 586.0 | -1.3% / -9.3% | 1,280-token chunk, 351-slot ring |
| B | 363.7 / 380.1 | 485.9 / 498.8 | +33.6% / +31.2% | 768, 8-slot staging (+ CPU share on 0.2.0-beta) |
| C | 508.0 / 537.4 | 498.9 / 493.6 | -1.8% / -8.2% | 1,024, 384-slot ring |
| D | 304.0 / 302.0 | 438.6 / 432.5 | +44.3% / +43.2% | 512, 8-slot staging (+ CPU share on 0.2.0-beta) |

- The two arms whose chunks fall under 1,024 tokens (B, D) read prompts 31-44% faster; the two that stream every
  expert ahead (A, C) read 4K prompts at about the same rate and 16K prompts 8-9% slower. The cause of the A / C change
  was not isolated.
- C against B: +39.7% / +41.4% on 0.1.40.3, +2.7% / -1.0% here (blocks 3 and 2).
- 128,000 tokens on C: 520.8 -> 464.6 prompt tok/s, 43.6 -> 51.0 decode tok/s, TTFT 246.0 -> 275.6 s.
- Decode at 4,096 / 16,384 tokens: 44.7-47.5 tok/s across arms on 0.1.40.3 session 1, 46.8-52.1 here.
- Plans: the same chunk and ring for every arm; 16-17 fewer expert-cache slots per arm; 87-144 MiB less VRAM free
  after load (session 1).
- Recall 6/6 and the task check passed on both.

## Limitations

- One machine, one model; configurations measured as blocks, not interleaved per request; two sessions with different
  display states.
- The engine binary is the release asset, built from `f6d65cb584c0`, not compiled from the `3d1fb48` tree.
- The A / C slowdown at 16K against 0.1.40.3 and the session-to-session difference are reported, not isolated.
- Not tested: other `--kv-resident` values, q4_0, vision, `"parallel"`, the 0.2.0-beta installer (`install.ps1` / the
  Windows `.exe`).

## Notes for the maintainers

- On 0.2.0-beta a smaller `--kv-resident` no longer pays on this card: the CPU share closes the gap the 0.1.40.3 report
  measured, so the suggestion there to pick a smaller window on cards under 10 GB is withdrawn.
- `git clone --branch v0.2.0-beta` + `START-HERE.bat` cannot find a ready-made engine: `setup.py` asks for
  `strata-windows-x64.zip` under `releases/download/v0.1.45.3/` and on the latest release (both HTTP 404), while
  `v0.2.0-beta` ships `strata-engine-0.1.45.3-windows-x86_64-cuda-cuda13.tar.gz`. With `--yes` it then installs
  Visual Studio 2022 Build Tools and the CUDA Toolkit (about 8-10 GB) to compile.
- `setup.py` still prints `KV cache: 4-bit (Hadamard-rotated)` for `--kv k8v4`.
