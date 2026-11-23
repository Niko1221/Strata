# Community benchmark on one Quadro RTX 6000 24 GB (sm_75), Windows release package

Measured on 2026-11-20 and 2026-11-23 by DonSMectron. Unsloth UD-IQ2_XS on one Turing card with 64 GB of RAM at 3200 MT/s:

- **Repeat pass:** the default start and a tuned start, three runs each at 32K, 128K and 256K, whole context filled.
- **Paired A/B** of `STRATA_SM75_PERF=1` and `STRATA_EXP_MODE=8` with `tools/ab_engine.py` (for #2197).
- **Single-run explorations:** a 32K-256K ladder for both starts, a 16K drafter run, a one-change screen of eleven
  settings at 256K with a 3-run check of its two best rows, a full-context confirmation, and two tool-use suites.
- **RAM speed:** the first set of runs had the RAM at 2133 MT/s (XMP off). Everything was run again at 3200 MT/s. The
  3200 MT/s numbers are the main results; the 2133 MT/s numbers are kept for comparison.

**Main result:** `STRATA_BF16_TC=1 STRATA_SELECT_SIMT=1` (with `STRATA_GRAPH_PRIO=1`) read prompts 34% / 55% / 76%
faster at 32K / 128K / 256K, with no decode change. A full 257,474-token prompt took 157 s instead of 276 s. Decode at
256K was about 50 tok/s with either start. **Main limitations:** decode varies by up to 15% between runs of the same
settings, and 3 of 78 requests on 2026-11-23 decoded at 8-19 tok/s (see [Slow requests](#slow-requests)).

## Hardware and software

- GPU: one NVIDIA Quadro RTX 6000 24 GB (TU102, sm_75), VBIOS 90.02.4a.00.02, 260 W limit (default), WDDM, PCIe 3.0 x16
  (engine probe 12.5-12.6 GB/s host to device). The card also drives the display.
- CPU: Intel Core i5-12600K (6 P-cores + 4 E-cores, 16 threads, AVX2). The engine used 9 CPU workers.
- RAM: 64 GB DDR4 (2x 32 GB Corsair CMK64GX4M2D3600C18), dual channel. **3200 MT/s at 1.2 V with XMP on** for the main
  results. The kit is rated 3600 MT/s; with XMP on, Windows reports 3200. The comparison runs were at 2133 MT/s (XMP off).
  Nothing else was changed between the two.
- Storage: model on a 512 GB NVMe SSD (TS512GMTE662T).
- OS: Windows 11 Pro build 26300; NVIDIA driver 621.14 (CUDA 13.5 driver API).
- Strata: Terminal 0.2.16, signed release engine `0.1.62-windows-x86_64-cuda-cuda13` installed by `strata update`
  (not a source build).
- Background: VS Code, a browser, Task Manager and a few small apps open. No other CUDA work. Free RAM while serving
  was about 1.5 to 11 GB.

## Model and configuration

- Model: Qwen3.8-Flash-Next, Unsloth UD-IQ2_XS (`--family unsloth --model UD-IQ2_XS`, downloaded by `strata launch`;
  `Qwen3.8-Flash-Next-UD-IQ2_XS-0000{1,2,3}-of-00003.gguf`, 76 GB). Native pack `unsloth-ud-iq2_xs` made by setup.
  Repository revision: not recorded.
- Vision: off in all speed runs and the tuned tool runs (the default-start tool runs had `--vision gpu`, see below).
- Context 262,144 unless stated; `--kv int8 --kv-resident 32768` (setup's default); `--expert-cache auto` (9,789 slots,
  15.67 GiB at 256K); `--prefill auto`; DFlash2 drafter (`--draft dflash2 --spec 6 --draft-conf 0.7`, chosen by
  `strata launch`); `--vram-reserve-mib` left to setup.
- Setup writes `STRATA_ROUTE_TAIL_SKIP=0` for this pack; it was on in every run and was not varied.
- No calibration, no `--tune`, no experimental speed projection.

Default start:

```text
strata launch --family unsloth --model UD-IQ2_XS --context 256k --vision off --drafting dflash2 --kv int8 --restart --yes --no-tui
```

Tuned start (one change from the above: three environment switches):

```text
strata launch --family unsloth --model UD-IQ2_XS --context 256k --vision off --drafting dflash2 --kv int8 \
  --set STRATA_GRAPH_PRIO=1 --set STRATA_BF16_TC=1 --set STRATA_SELECT_SIMT=1 --restart --yes --no-tui
```

The engine's start-up lines for the tuned start are in [engine-log-excerpt.txt](engine-log-excerpt.txt).

## Method

All requests go through `/v1/chat/completions`, streamed, temperature 0, seed 1234, `cache_prompt: false`, thinking on
(the model's default). Rates are the engine's own `timings` (`prompt_per_second`, `predicted_per_second`); decode counts
reasoning tokens. TTFT is client-side, to the first reasoning or answer token. Every row is a fresh `strata launch
--restart` (load 124-216 s, not included in any rate), then one unscored 64-token request, then the measured request.
The exception is two rows of the 3-run check, marked there. The script is
[scripts/run_strata_speed.ps1](scripts/run_strata_speed.ps1). The whole 3200 MT/s sequence is
[scripts/run_all_3200.ps1](scripts/run_all_3200.ps1) and [scripts/run_followup_3200.ps1](scripts/run_followup_3200.ps1),
and the settings for each run are in [scripts/configs](scripts/configs). The prompts come from an earlier llama.cpp study
on this PC, so the same numbers exist for other models. They are not BENCHMARK_PROTOCOL.md, and `strata bench run` was
not used.

- **Context ladder:** for each size, one prompt that fills the context minus 768 tokens (a synthetic service log of
  about 30 tokens per line, sized with `/v1/messages/count_tokens`), with a code `VAULT-xxxxxx` in the middle line. The
  question asks for the code. 256 generated tokens. Recall = the code appears anywhere in the reply, reasoning included.
- **Repeat pass:** the ladder's request at 32K, 128K and 256K. Six starts in the order default, tuned, tuned, default,
  default, tuned; each start runs the three sizes, each size a fresh `strata launch --restart`. 18 runs.
- **16K run:** `--context 16k`; a 1,024-token-cap warm-up prompt, then a different 1,024-token-cap measured prompt (about
  155 prompt tokens each).
- **Screen at 256K:** the 16K-run prompts, with `--context 256k`, one change at a time from the default start, plus the
  tuned start as a last row.
- **Confirmation at 256K:** the ladder's 256K request with the screen's changes, whole context filled.
- **SM75 A/B:** `terminal/tools/ab_engine.py` from the 0.2.16 Terminal, changed only to keep each reply's full text for an
  identity check ([data/sm75-ab/ab_engine_full.py](data/sm75-ab/ab_engine_full.py)). Arms A = default, B =
  `STRATA_SM75_PERF=1`, C = B + `STRATA_EXP_MODE=8`. 3 rounds with alternating order, a fresh server per arm and round,
  greedy, thinking off, 256 new tokens, three short chats and a 6,345-token prompt. `--adapt-every 1000000` in every arm,
  so the expert placement could not drift between arms ([data/sm75-ab/ab-config.json](data/sm75-ab/ab-config.json)).
- **Tool suites:** an external harness (not in this repository) that sends native OpenAI `tools` requests. Core = 7 tasks
  (read, create, edit, search-fetch on an offline fixture, an allow-listed PowerShell command, error recovery, a
  multi-step workflow). C++ = 4 tasks checked by MSVC `cl /Zs` (find a function, fix two bugs until `static_assert`s
  pass, add an MFC button handler, compute CSV averages). Fixture seed 20260927, 12-turn and 300 s limits per task,
  temperature 0. A task fails on a wrong result, the turn limit, or a change to a file outside the task.

Memory: peak GPU memory (nvidia-smi, whole card), peak working set of `strata.exe`, and minimum free physical RAM, sampled
once a second during the measured request.

## Results

All results in this section are at 3200 MT/s. The 2133 MT/s comparison is in [RAM speed](#ram-speed-2133-against-3200-mts).

### Repeat pass: default and tuned start, three runs each

Measured 2026-11-20 after turning XMP on. Median (min-max) of three runs, whole context filled, 256 generated tokens.

| Context | Start | Prompt tok/s | Decode tok/s | TTFT s | Recall |
| ---: | --- | --- | --- | --- | --- |
| 32,768 (31,525 prompt tokens) | default | 1,347 (1,341-1,359) | 55.6 (52.5-55.6) | 23.5 (23.4-23.7) | 3/3 |
| | tuned | 1,803 (1,798-1,806) | 54.3 (53.5-56.8) | 17.6 (17.5-18.0) | 3/3 |
| 131,072 (128,355) | default | 1,144 (1,141-1,145) | 52.9 (49.2-56.6) | 112.6 (112.2-112.8) | 3/3 |
| | tuned | 1,776 (1,774-1,778) | 55.0 (53.6-55.7) | 72.5 (72.3-72.6) | 3/3 |
| 262,144 (257,474) | default | 933 (931-934) | 50.4 (47.9-51.6) | 276.1 (276.0-278.5) | 0/3¹ |
| | tuned | 1,641 (1,640-1,643) | 50.0 (43.1-50.8) | 157.1 (156.9-157.2) | 3/3 |

Prompt rates repeat within 1%, and the two starts do not overlap at any size: +34% / +55% / +76%. Decode ranges overlap
at every size, so no decode difference is shown. Peak GPU memory 22,856-22,946 MiB; `strata.exe` peak 43,267 MiB at
32K to 47,071 MiB at 256K; minimum free RAM 4,063-11,521 MiB. Per-run rows: [data/repeat.csv](data/repeat.csv).

¹ All three replies were still reasoning when the 256-token cap ended them, and had not written the code yet (they were
counting services for the prompt's other question). No reply gave a wrong code. The tuned replies happened to mention
the code early in their reasoning. Replies: [data/replies-256k-repeat/](data/replies-256k-repeat/).

### Context ladder (one run per size)

| Context | Start | Prompt tok/s | Decode tok/s | Drafts accepted | TTFT s | `strata.exe` peak MiB | Recall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 32,768 | default | 1,348 | 58.6 | 130/194 | 23.4 | 42,816 | yes |
| | tuned | 1,784 | 58.4 | 136/189 | 17.7 | 43,268 | yes |
| 65,536 | default | 1,286 | 43.9 | 108/172 | 49.7 | 44,315 | yes |
| | tuned | 1,836² | 54.7² | 120/166 | 34.8 | 44,317 | yes |
| 131,072 | default | 1,146² | 53.4² | 127/175 | 112.1 | 45,459 | yes |
| | tuned | 1,677 | 49.1 | 116/184 | 76.6 | 45,460 | yes |
| 196,608 | default | 1,007² | 55.5² | 139/187 | 191.6 | 46,255 | yes |
| | tuned | 1,701 | 50.4 | 124/197 | 113.6 | 46,257 | yes |
| 262,144 | default | 931² | 44.4² | 86/118 | 277.4 | 47,067 | no¹ |
| | tuned | 1,642 | 50.5 | 121/146 | 156.9 | 47,068 | yes |

² Re-run. The first default run at 128K decoded at 8.3 tok/s, and the script then skipped 192K and 256K for that start.
The first tuned run at 64K decoded at 16.6 tok/s. Both are kept in the CSV and described under
[Slow requests](#slow-requests). Peak GPU memory 22,856-22,946 MiB at every size.

Decode did not fall steadily as the context grew: it stayed between about 44 and 59 tok/s from 32K to 256K. Per-run
data, first runs included: [data/ladder.csv](data/ladder.csv). 256K replies: [data/replies-256k/](data/replies-256k/).

### 16K run (one run each)

| Drafter | Prompt tok/s | Decode tok/s | Drafts accepted |
| --- | ---: | ---: | ---: |
| DFlash2 (default start) | 168 | 60.7 | 577/788 |
| MTP (`--drafting mtp`) | 139 | 52.9 | 80/130 |
| DFlash2, tuned start | 214 | 66.7 | 571/754 |

The prompts are 155 tokens, too short for a meaningful prompt rate. [data/short-16k.csv](data/short-16k.csv).

### Screen at 256K, one change each (one run each, 1,024 generated tokens, short prompt)

| Change from the default start | Decode tok/s | Drafts accepted |
| --- | ---: | ---: |
| none | 63.3 | 522/603 |
| `STRATA_SM75_PERF=1` | 62.1 | 511/606 |
| `STRATA_GRAPH_PRIO=1` | 62.3 | 509/595 |
| `STRATA_BF16_TC=1` + `STRATA_SELECT_SIMT=1` | 63.1 | 506/592 |
| `--kv q4_0` | 66.3 | 585/766 |
| `--set spec=4` | 64.7 | 486/554 |
| `--set spec=8` | 62.7 | 535/651 |
| `--set draft-conf=0.5` | 66.6 | 576/772 |
| `--set vram-reserve-mib=1088` | 62.9 | 513/593 |
| `--set kv-resident=16384` | 60.9 | 483/589 |
| `--set kv-resident=65536` | 62.3 | 485/572 |
| tuned start (`GRAPH_PRIO` + `BF16_TC` + `SELECT_SIMT`) | 63.5 | 510/592 |

All rows fall within 60.9-66.6 tok/s, inside the run-to-run spread. The generated text differed between rows, even
where the change should not alter answers (for example the VRAM reserve). The likely cause is that the GPU expert cache
holds a different set of experts after each start, and the GPU and CPU round differently. Prompt rates are left out:
the prompt is 155 tokens. [data/screen-256k.csv](data/screen-256k.csv).

**3-run check of the two best rows**, each on top of the tuned start, in the order T Q D D Q T T Q D:

| Setting | Decode tok/s, runs 1 / 2 / 3 | Median | Drafts made |
| --- | --- | ---: | ---: |
| tuned start | 64.7 / 50.9 / 48.5³ | 50.9 | 591-625 |
| + `--kv q4_0` | 65.7 / 63.7 / 67.8 | 65.7 | 775-795 |
| + `--set draft-conf=0.5` | 61.5 / 66.7³ / 68.0 | 66.7 | 767-824 |

³ Same settings as the run before, so `strata launch --restart` kept the running server: load took 3-4 s, and the
prompt came from the cache. Decode is still measured.

Both variants were steady (61.5-68.0 in all seven of their runs, screen included) and made about 30% more draft tokens.
The tuned start's runs 2 and 3 came one after the other and were both low; the tuned start's other two runs on this
prompt were 63.5 and 64.7. So this check does not cleanly separate the settings. Neither variant is recommended from this
data: `--kv q4_0` reads full-context prompts much slower (next table), and `draft-conf=0.5` was 3% slower than the
default in the 2133 MT/s screen. [data/screen-check-256k.csv](data/screen-check-256k.csv).

### Confirmation at 256K, full context (one run each)

| Setting | Prompt tok/s | Decode tok/s | TTFT s | Recall |
| --- | ---: | ---: | ---: | --- |
| default | 935 | 45.9 | 275.6 | yes |
| + `GRAPH_PRIO` | 930 | 44.3 | 277.8 | yes |
| + `GRAPH_PRIO`, `--kv q4_0` | 751 | 46.2 | 343.5 | no¹ |
| + `GRAPH_PRIO`, `--kv q4_0`, `BF16_TC`, `SELECT_SIMT` | 1,157 | 46.0 | 222.7 | yes |
| **tuned start (int8 KV), ladder run** | **1,642** | **50.5** | **156.9** | yes |
| **tuned start, repeat median** | **1,641** | **50.0** | **157.1** | 3/3 |

`--kv q4_0` lowered the full-context prompt rate by about 20% without and 30% with the BF16 switches.
[data/confirm-256k.csv](data/confirm-256k.csv). Replies: [data/replies-256k/](data/replies-256k/).

### SM75 A/B (#2197): 3 rounds, alternating order

| Request | A decode | B decode | C decode | A prompt | B prompt | C prompt |
| --- | --- | --- | --- | --- | --- | --- |
| chat0 | 52.9 (48.7-54.4) | 50.1 (43.6-54.6) | 54.0 (50.9-55.9) | 83.1 | 76.9 | 81.7 |
| chat1 | 38.3 (19.1-40.3) | 36.1 (32.0-39.3) | 38.8 (37.7-40.1) | 61.0 | 65.6 | 63.9 |
| chat2 | 43.1 (36.6-43.4) | 41.5 (39.1-43.9) | 42.4 (40.0-43.6) | 67.4 | 68.1 | 71.2 |
| long | 45.1 (40.1-46.4) | 41.3 (40.5-43.0) | 44.8 (40.8-45.1) | 1,040 (742-1,046) | 759 (748-762) | 755 (750-765) |

- **Speed:** decode ranges overlap between all arms for every request. B's medians were 4-8% below A's, C's within 2%.
  No gain from either switch.
- **Long prompt:** it ran at either about 750 or about 1,045 tok/s, in any arm (A twice here; C once at 2133 MT/s). This
  two-level behavior is not tied to the switches, so the prompt column does not compare the arms.
- **Identity** (SHA-256 of the full reply, 4 requests × 3 rounds): all three arms produced the same few reply variants.
  chat0 was identical in all 9 runs. For chat1, the default arm itself gave three different replies in three rounds. No
  arm produced a reply variant of its own, apart from one of A's chat1 replies.
- Neither switch prints a start-up line, so whether their kernels ran on this pack is not confirmed.
- A's round-1 chat1 decoded at 19.1 tok/s (see [Slow requests](#slow-requests)).

Rows with full replies: [data/sm75-ab/ab.jsonl](data/sm75-ab/ab.jsonl).

### Tool suites (one run each)

| Start | Core | Time s | C++ / MFC | Time s | Unintended changes |
| --- | --- | ---: | --- | ---: | ---: |
| default, 64K context, `--vision gpu`, DFlash2 `--draft-conf 0.5` (setup's first start) | 7/7 | 67 | 4/4 | 83 | 0 |
| tuned, 256K context | 7/7 | 71 | 4/4 | 99 | 0 |

Per-task rows: [data/tools-core-default-tasks.csv](data/tools-core-default-tasks.csv),
[data/tools-cpp-default-tasks.csv](data/tools-cpp-default-tasks.csv),
[data/tools-core-tuned-tasks.csv](data/tools-core-tuned-tasks.csv),
[data/tools-cpp-tuned-tasks.csv](data/tools-cpp-tuned-tasks.csv).

### Slow requests

On 2026-11-23, 3 of 78 measured requests decoded far below the rest:

| Run | Decode tok/s | Same request otherwise |
| --- | ---: | ---: |
| ladder, default start, 128K | 8.3 | 53.4 (re-run) |
| ladder, tuned start, 64K | 16.6 | 54.7 (re-run) |
| A/B, arm A, round 1, chat1 | 19.1 | 38.3-40.3 |

Their prompt rates were normal, and the expert cache hit rate in the slow ladder run (89.9%) was the same as in normal
runs. The engine warned at every start that only 653 MiB of VRAM was free (below 1,024 MiB) and that the driver can
then page GPU memory to system RAM, so that decode falls to a few tok/s (#2065). That fits, but I could not confirm it
was the cause. The same warning appeared in all the normal runs. Peak GPU memory was 22,834-22,948 MiB in every run,
and at one check during the runs no other program used more than 20 MB of VRAM. A per-process GPU-use log from 08:00 (after the two slow ladder runs) showed
other programs at 9% or less of the GPU. The GPU showed no thermal or power slowdown.

`--set vram-reserve-mib=1088` (the fix the warning suggests) was as fast as the default in both screens. Whether it
prevents these slow requests was not tested: they are too rare to show up in a few runs.

### Memory

At 256K with the tuned start: peak GPU 22,946 MiB of 24,576; `strata.exe` working set 47,068 MiB at its peak; minimum
free physical RAM 1,554-8,687 MiB over all 256K runs. The engine reported about 653-785 MiB of VRAM free after loading and
warned at each request that free VRAM was below 1,024 MiB (#2065). No out-of-memory errors or failed requests occurred
in any run.

## RAM speed: 2133 against 3200 MT/s

The same repeat pass at both speeds, three runs each (median tok/s):

| Context | Start | Prompt 2133 | Prompt 3200 | Decode 2133 | Decode 3200 |
| ---: | --- | ---: | ---: | --- | --- |
| 32K | default | 1,348 | 1,347 | 54.8 (53.0-55.9) | 55.6 (52.5-55.6) |
| | tuned | 1,800 | 1,803 | 53.6 (53.0-55.4) | 54.3 (53.5-56.8) |
| 128K | default | 1,143 | 1,144 | 48.8 (46.2-53.1) | 52.9 (49.2-56.6) |
| | tuned | 1,772 | 1,776 | 53.0 (51.7-53.2) | 55.0 (53.6-55.7) |
| 256K | default | 932 | 933 | 45.5 (43.6-46.2) | 50.4 (47.9-51.6) |
| | tuned | 1,640 | 1,641 | 45.9 (44.8-49.3) | 50.0 (43.1-50.8) |

- **Prompt:** unchanged, within 0.3% at every size.
- **Decode:** the medians were 1-5 tok/s higher at 128K and 256K. Only the default start at 256K separates clearly
  (+11%, ranges do not overlap). A likely but small gain at long context, not shown for every row.
- **Everything else** gave the same conclusions at both speeds:
  - DFlash2 was faster than MTP at 16K (67.6 against 47.5 tok/s at 2133).
  - No screen row was clearly better than the default.
  - `--kv q4_0` lowered the full-context prompt rate.
  - The SM75 switches gave no decode gain.
  - The 2133 A/B's "+2.6% long prompt" for `SM75_PERF` was not seen at 3200, where the long prompt ran at two levels
    in every arm. I no longer think it was an effect of the switch.
- Tool suites at 2133: 7/7 and 3/4 with the default start (one `cpp-tests` turn-limit failure after editing files of the
  next task), 7/7 and 4/4 with the tuned start.

All 2133 MT/s data is in [data/ddr2133/](data/ddr2133/), with the same file names as above. The A/B rows are
[data/ddr2133/sm75-ab.jsonl](data/ddr2133/sm75-ab.jsonl).

## Correctness and limitations

- Checks: the recall code in 29 of 34 full-context requests at 3200 MT/s. All 5 misses are at 256K, and all were
  replies cut off by the 256-token cap while still reasoning; none gave a wrong code. The default start's ladder reply
  at 256K ends with "Need vault code: VA". All 22 tool tasks passed at
  3200 MT/s.
- The 256-token cap with thinking on is too short for a fair recall check at 256K. A longer cap or thinking off would
  fix this; it was kept to match the earlier llama.cpp runs on this PC.
- **Only the repeat pass, the A/B and the 3-run check have three runs.** The ladder, 16K run, screen, confirmation and
  tool suites are one run each. Decode varied by up to 15% between runs of the same settings, more than most of the
  screen's differences.
- `STRATA_BF16_TC=1` changes rounding. Tool-suite results with it were 11/11 at both RAM speeds; no reply-by-reply
  comparison was made.
- The kit's rated 3600 MT/s was not tested.
- Not tested: `STRATA_ROUTE_TAIL_SKIP=7`, `--tune`, `strata bench run`, other quantizations, more than one request at a
  time, images, a source build.
