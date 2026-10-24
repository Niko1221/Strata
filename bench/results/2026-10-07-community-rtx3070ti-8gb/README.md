# Community benchmark: RTX 3070 Ti 8 GB, Ryzen 9 5900X, 64 GB DDR4-3600 (Windows 11)

Measured on 2026-10-07 by [gbwzzy218](https://github.com/gbwzzy218) on a native-Windows desktop whose only GPU also drives the display.
Flash-Next IQ2_XS on Strata's release engine 0.1.40.3. On this 8 GB card, lowering `--kv-resident` from 32,768 to
20,480 in setup's 128K k8v4 configuration increased median prompt throughput by 38.8–41.4% in three
within-session comparisons, including one with C before B; decode did not change by a comparable amount. Follow-up
configurations are consistent with a substantial contribution from the prompt chunk size the automatic planner
can afford, but do not independently quantify the mechanisms. Main limitations: one machine, one model, and
configurations measured as blocks (one server start each), not interleaved request by request.

## Summary

Initial comparison: blocks 1–4, five measured requests per prompt length, medians (ranges in [Results](#results)).
Every request generated 256 tokens and reported zero reused prompt tokens. All arms carry the same hand-added
conversation cache on top of their context and KV settings.

| Arm / block | Context; KV; `--kv-resident` | Prompt tok/s at 4,096 / 16,384 | Decode tok/s at 4,096 / 16,384 |
| --- | --- | --- | --- |
| A / 1 — setup's context and KV defaults here | 32,768; int8; none (KV fully in VRAM) | 549.8 / 645.9 | 47.0 / 45.9 |
| B / 2 — setup, `--context 131072 --kv k8v4` | 131,072; k8v4; 32,768 | 363.7 / 380.1 | 45.9 / 46.9 |
| C / 3 — B with `--kv-resident 20480` | 131,072; k8v4; 20,480 | 508.0 / 537.4 | 47.5 / 47.1 |
| D / 4 — setup's KV-streaming choice for `--context 131072` | 131,072; int8; 32,768 | 304.0 / 302.0 | 47.1 / 44.7 |

- Against A, B read prompts 33.8% / 41.2% slower and D 44.7% / 53.2% slower (4,096 / 16,384 tokens).
- C read prompts 39.7% / 41.4% faster than B and stayed 7.6% / 16.8% below A, while keeping the 128K context. The
  repeated pair (blocks 5-B, 6-C) measured +40.2% / +41.1%; a second session with the order reversed (9-C before
  10-B) measured +38.8% / +40.9%.
- Decode block medians at 4,096 and 16,384 tokens ranged from 44.7 to 47.5 tok/s in the first session, with
  overlapping ranges. These runs do not establish that decode is equivalent across arms; they show no difference
  comparable to the prompt-rate differences.
- With C, three 128,000-token prompts read at a median 520.8 tok/s [520.7–522.7], decoded at 43.6 tok/s
  [43.3–46.3], with a 246.0 s time to first token [245.2–246.1]. Separately, six needle checks all returned their
  targets (three near 32.9K tokens, three near 125.9K tokens).

## What changed between B and C

Lowering `--kv-resident` from 32,768 to 20,480 was the only setting changed by hand, but it changed several
quantities in the engine's automatic plan at once (start lines in [`engine-start.txt`](engine-start.txt)):

| Planned at start | B | C |
| --- | ---: | ---: |
| VRAM free for the expert cache | 2.16 GiB | 2.27 GiB |
| Expert-cache capacity | 985 slots | 1,067 slots |
| Prompt chunk (`--prefill auto`) | 768 tokens | 1,024 tokens |
| Ring printed by the planner | 8 slots | 384 slots |
| Cache slots lent to the prompt path | 432 | 900 |

In the source revision inspected (`src/prefill/prefill.cpp`), `ring_slots()` returns `STAGE` (8) for any chunk
below `stream_all_min()`, which defaults to 1,024 tokens (`STRATA_PREFILL_STREAM_MIN`); only chunks at or above it,
with a ring larger than `STAGE`, stream every non-resident expert ahead of use (`stream_all`, line 1990), smaller chunks stage the routed experts
layer by layer. B's 768-token and D's 512-token chunks are therefore below that threshold, and their 8-slot ring
follows from the chunk size.

A second session added two configurations with C's KV settings and B's 768-token chunk. The table is in comparison
order; the execution order was E1, E2, C, B (see [Method](#method)):

| Block (second session) | Change from the row above | Expert-cache capacity | Chunk | Slots lent | Prompt tok/s 4,096 / 16,384 |
| --- | --- | ---: | ---: | ---: | --- |
| 10-B | — | 985 | 768 (auto) | 432 | 378.9 / 395.3 |
| 7-E1 | `--kv-resident 20480`; `--prefill 768` instead of `auto` | 1,067 | 768 (fixed) | 431 | 380.7 / 397.3 (+0.5% / +0.5%) |
| 8-E2 | add env `STRATA_PREFILL_STREAM_MIN=512` | 1,067 | 768 (fixed) | 821 | 404.5 / 436.7 (+6.3% / +9.9%) |
| 9-C | `--prefill 768` back to `auto`; remove `STRATA_PREFILL_STREAM_MIN=512`, restoring the source default of 1,024 | 1,067 | 1,024 (auto) | 900 | 525.8 / 556.8 (+30.0% / +27.5%) |

The follow-up configurations are consistent with a substantial contribution from chunk size, but do not
independently quantify the underlying mechanisms. At a 768-token chunk, E1's prompt medians were 0.5% above B's.
Lowering `STRATA_PREFILL_STREAM_MIN` to 512 in E2 increased them by 6.3% / 9.9%, while borrowed cache slots
increased from 431 to 821. Returning to C removed the threshold override and selected a 1,024-token automatic chunk
with 900 borrowed slots; medians increased by another 30.0% / 27.5%. These comparisons do not isolate expert-cache
capacity, chunk size and staging behavior: each step also changed the lent slots, E1 and E2 each had one server
start, and their actual ring sizes and staging paths were not logged by the release binary.

## Hardware and software

Probes and file hashes are in [`system.txt`](system.txt).

- **GPU:** NVIDIA GeForce RTX 3070 Ti, 8,192 MiB, compute capability 8.6, PCIe 4.0 x16 (setup's probe and
  `nvidia-smi`), power limit 290 W (stock). The card also drives the Windows desktop. Clocks not fixed.
- **CPU:** AMD Ryzen 9 5900X, 12 cores / 24 threads (Zen 3), AVX2, no AVX-512. The engine used 11 expert-pool
  workers plus its host thread.
- **RAM:** 63.9 GiB reported; 4 x 16 GiB DDR4-3600 (F4-3600C16-16GTZNC), configured at 3600, dual channel
  (~57.6 GB/s theoretical).
- **Storage:** Samsung 970 EVO Plus 2 TB NVMe; repository and model files on it.
- **OS:** Windows 11 Pro, build 26200; system-managed page file. Large pages were not granted
  (`VirtualAlloc error 1314` in every block's engine log; the line is quoted in `system.txt`), so the expert arena
  used 4 KB pages.
- **Driver:** NVIDIA 617.42. CUDA runtime from setup's wheels in `.venv`: `nvidia-cublas` 13.0.2.14,
  `nvidia-cuda-runtime` 13.0.96.
- **Engine:** 0.1.40.3, the ready-made release binary installed by setup (`engine/BUILD.json`: source `release`,
  CUDA 13.0, archs 75/86/89/120; `strata.exe` SHA-256
  `34cde150b21148e6cf192dbf2242e5544d3456c09ab0010837aaacfda11eca5b`). No source build. No `STRATA_*` variable was
  set in the environment; E2's variable was set only through its config's `env`.
- **Repository tree:** commit `d5ea713374` of the fork `gbwzzy218/Strata`, identical to `Niko1221/Strata` `main`
  at that commit (GitHub's compare: 0 ahead, 0 behind; `CMakeLists.txt` version 0.1.40.3), downloaded as a zip; the server, setup and the source lines quoted above come
  from this tree. The release binary was downloaded separately by setup; that it was built from the same source is
  assumed, not checked.
- **Background:** a normal desktop session (a browser, Steam, a chat app, the NVIDIA App overlay) open but idle;
  nobody used the PC during the runs.

## Model and configuration

- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (setup's pin):
  `IQ2_XS/…-IQ2_XS-00001-of-00002.gguf` (39,225,954,592 B) and `…-00002-of-00002.gguf` (28,800,138,432 B).
- Prepared by the unmodified installer (`START-HERE.bat --yes --family qwen --model IQ2_XS`): native pack
  `packs\iq2_xs`; the MTP draft layer fetched by setup from `Qwen/Qwen3.8-Flash-Next` and packed as `mtp-q2_0`;
  the default draft vocabulary (with CJK); the shipped `data/expert-profile.bin`
  (SHA-256 `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`).
- No vision encoder, no low-RAM mode (no `--resident-budget-gib`), no calibration, experimental speed projection
  off.

Engine arguments shared by every arm (paths shortened; each block's full arguments and `env` are in
[`configs/`](configs/)):

```text
strata.exe --pack <data-dir>\packs\iq2_xs
  --native   <data-dir>\models\IQ2_XS\…-IQ2_XS-00001-of-00002.gguf
  --ple-gguf <data-dir>\models\IQ2_XS\…-IQ2_XS-00002-of-00002.gguf
  --expert-profile <repo>\data\expert-profile.bin
  --expert-cache auto
  --spec 4 --spec-min-p 0.5
  --mtp <data-dir>\mtp\rt
  --conversation-cache-mib 4096 --conversation-cache-slots 4
```

| Arm | `--max-context` | `--kv` | `--kv-resident` | `--prefill` | `STRATA_PREFILL_STREAM_MIN` | Where the setting comes from |
| --- | ---: | --- | ---: | --- | --- | --- |
| A | 32768 | int8 | — | auto | unset (source default 1024) | `START-HERE.bat --yes` on this PC (32K below 14 GB of VRAM) |
| B | 131072 | k8v4 | 32768 | auto | unset | written by `START-HERE.bat --setup --family qwen --model IQ2_XS --context 131072 --kv k8v4 --yes --no-start` |
| C | 131072 | k8v4 | 20480 | auto | unset | B with `--kv-resident` edited by hand |
| D | 131072 | int8 | 32768 | auto | unset | what setup's KV-streaming branch writes for `--context 131072`; this file was edited by hand |
| E1 | 131072 | k8v4 | 20480 | 768 | unset | C with the chunk fixed |
| E2 | 131072 | k8v4 | 20480 | 768 | 512 (config `env`) | E1 with streaming allowed at 768 |

The conversation cache is not a setup default; it was added by hand and is identical in every arm.

Plans the engine printed at each start:

| Arm (blocks) | VRAM free for the cache | Expert-cache capacity | Prompt chunk | Ring printed | Slots lent to the prompt path | VRAM free after load | KV in pinned RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A (1) | 2.24 GiB | 1,040 (1.41 GiB) | 1,280 | 351 | 873 | 432 MiB | — |
| B (2, 5, 10) | 2.16 GiB | 985 (1.33 GiB) | 768 | 8 | 432 | 424–479 MiB | 1.20 GiB |
| C (3, 6, 9) | 2.27 GiB | 1,067 (1.45 GiB) | 1,024 | 384 | 900 | 429–481 MiB | 1.20 GiB |
| D (4) | 2.07 GiB | 918 (1.25 GiB) | 512 | 8 | 372 | 481 MiB | 1.55 GiB |
| E1 (7) | 2.27 GiB | 1,067 (1.45 GiB) | 768 (fixed) | not printed | 431 | 481 MiB | 1.20 GiB |
| E2 (8) | 2.27 GiB | 1,067 (1.45 GiB) | 768 (fixed) | not printed | 821 | 481 MiB | 1.20 GiB |

Each repeated arm printed the same plan at every start, with small differences in free VRAM after load. This shows
the plan was repeatable under the desktop load of this session; how the planner responds to a different desktop
VRAM load was not tested.

## Method

- **Harness:** the repository's community [`benchmark.py`](../2026-09-30-community-rtx-5090/benchmark.py),
  unchanged: synthetic Python functions with a nonce at the front of each request, the length set exactly with
  Strata's tokenizer, greedy decoding, `reasoning_effort "none"`, a 256-token cap, streaming. One short warm-up
  request (`Reply with exactly the word READY.`) per harness invocation is excluded; it does not exercise a long
  prompt.
- **Requests:** 108 measured requests (68 in the first session, 40 in the second). Each generated 256 tokens, ended
  with `finish_reason "length"`, and reported zero reused prompt tokens. The nonces depend only on the prompt length
  and run index, so they distinguish requests within one harness invocation, while the same five 4,096-token and
  five 16,384-token requests repeat in every block (18 distinct request hashes overall, including C's longer runs).
  "Fresh" here means the engine reported no reused prompt tokens. Requests within a block share the running
  engine's cache state. Each block starts a new engine process and initializes its process-local caches. OS file
  caches were not deliberately flushed, and a controlled cold-cache state was not established.
- **Blocks and order:** each block is a fresh server start (`serve/server.py --engine strata --config <arm>.json`,
  no browser), loading excluded; within a block, five runs per length (three at 128,000), serial, in increasing
  length. First session (17:44–18:50): 1-A, 2-B, 3-C, 4-D, 5-B, 6-C. Second session (19:14–19:36): 7-E1, 8-E2,
  9-C, 10-B. Timestamps: [`orchestrator-session1.log`](orchestrator-session1.log),
  [`orchestrator-session2.log`](orchestrator-session2.log); driver script:
  [`bench_orchestrator.py`](bench_orchestrator.py) (local paths replaced by placeholders). In `results.json`, block
  6's 128,000-token runs are listed before its shorter ones because of file ordering; the log shows they ran last.
- **Timings:** prompt tok/s = freshly read tokens / `prompt_ms` and decode tok/s = `engine_generated` / `decode_ms`,
  both from the engine's `/metrics`. TTFT and total are client-side over loopback (first non-empty streamed delta,
  always answer text since reasoning is off; request start to end of stream). The engine's own timing lines are
  not attached; their per-request counters (`prompt_ms`, `decode_ms`, generated, reused and draft counts, read from
  `/metrics`) are in `results.json`.
- **Temperature and memory:** endpoint snapshots only (`nvidia-smi` used VRAM and temperature, `strata.exe` resident
  set, Windows' available RAM) after load and after each block; per-request clocks, power and throttling were not
  recorded.

## Results

Every run is in [`results.json`](results.json) (engine counters, client timings, draft counts, generated text,
request hash); per-block medians and ranges in [`summary.json`](summary.json). Drafts accepted is the aggregate
accepted / offered over the block's runs.

| Block | Arm | Prompt tokens | Runs | Prompt tok/s median [range] | Decode tok/s median [range] | TTFT s, median | Total s, median | Drafts accepted |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | A | 4,096 | 5 | 549.8 [542.9–554.9] | 47.0 [43.2–48.4] | 7.5 | 12.8 | 68% |
| 1 | A | 16,384 | 5 | 645.9 [644.3–647.1] | 45.9 [45.7–48.4] | 25.4 | 30.9 | 70% |
| 2 | B | 4,096 | 5 | 363.7 [360.2–365.4] | 45.9 [44.6–47.8] | 11.3 | 16.9 | 68% |
| 2 | B | 16,384 | 5 | 380.1 [379.4–380.7] | 46.9 [46.6–47.7] | 43.2 | 48.6 | 73% |
| 3 | C | 4,096 | 5 | 508.0 [506.5–511.4] | 47.5 [46.3–49.4] | 8.1 | 13.5 | 72% |
| 3 | C | 16,384 | 5 | 537.4 [536.8–541.4] | 47.1 [46.1–47.5] | 30.6 | 36.0 | 70% |
| 4 | D | 4,096 | 5 | 304.0 [301.4–305.7] | 47.1 [44.4–48.9] | 13.5 | 18.9 | 72% |
| 4 | D | 16,384 | 5 | 302.0 [301.7–302.3] | 44.7 [44.2–46.7] | 54.3 | 60.0 | 71% |
| 5 | B | 4,096 | 5 | 363.5 [360.9–363.7] | 46.4 [45.5–47.9] | 11.3 | 16.8 | 71% |
| 5 | B | 16,384 | 5 | 380.0 [379.0–380.5] | 46.3 [42.6–47.0] | 43.2 | 48.6 | 70% |
| 6 | C | 4,096 | 5 | 509.7 [502.1–512.5] | 47.5 [46.3–47.9] | 8.1 | 13.5 | 72% |
| 6 | C | 16,384 | 5 | 536.3 [531.8–539.3] | 47.0 [43.2–47.8] | 30.6 | 36.2 | 71% |
| 6 | C | 32,768 | 5 | 534.3 [529.8–534.7] | 47.2 [45.7–47.9] | 61.4 | 66.8 | 72% |
| 6 | C | 128,000 | 3 | 520.8 [520.7–522.7] | 43.6 [43.3–46.3] | 246.0 | 251.6 | 69% |
| 7 | E1 | 4,096 | 5 | 380.7 [377.9–381.0] | 48.1 [47.4–51.9] | 10.8 | 16.1 | 67% |
| 7 | E1 | 16,384 | 5 | 397.3 [396.3–397.7] | 51.1 [50.4–53.2] | 41.3 | 46.3 | 73% |
| 8 | E2 | 4,096 | 5 | 404.5 [401.8–410.9] | 49.8 [47.6–52.5] | 10.2 | 15.1 | 68% |
| 8 | E2 | 16,384 | 5 | 436.7 [434.8–438.0] | 51.7 [48.7–52.9] | 37.6 | 42.6 | 72% |
| 9 | C | 4,096 | 5 | 525.8 [519.9–531.4] | 51.3 [49.6–51.8] | 7.8 | 12.8 | 72% |
| 9 | C | 16,384 | 5 | 556.8 [554.6–558.4] | 51.0 [50.4–52.1] | 29.5 | 34.5 | 71% |
| 10 | B | 4,096 | 5 | 378.9 [376.0–379.0] | 49.4 [48.2–51.6] | 10.8 | 16.0 | 68% |
| 10 | B | 16,384 | 5 | 395.3 [395.2–395.8] | 51.1 [50.5–51.3] | 41.5 | 46.5 | 73% |

- **Repeatability within a session:** every prompt-rate run lies within 2% of its block median. Between the two
  starts of B in the first session the prompt-rate medians differ by under 0.1%, between the two starts of C by
  under 0.4%; the decode medians by up to 1.3%.
- **Between sessions:** in the second session both B and C read prompts 3.2–4.2% faster than in the first (at
  4,096 tokens B 363.5–363.7 → 378.9, C 508.0–509.7 → 525.8), and decode block medians at 4,096 and 16,384 tokens were
  48.1–51.7 tok/s against 44.7–47.5 in the first. The cause is unknown; the GPU started the second session cooler (39 °C after its first
  load against 47–57 °C in the first session; 74–81 °C after every block in both). Comparisons above are made
  within a session.
- **Order:** the first session ran B before C both times; the second ran C before B, with the same ~40% gap. A
  simple time trend or a B-then-C order effect does not explain it; per-request thermal state was not recorded.
- **Generated text:** inputs and output lengths matched across blocks, but the text did not: 3 of 10 repeated B
  requests and 4 of 10 repeated C requests (first session) produced identical text. Decode figures therefore
  include variation in the generated sequences and in draft acceptance.
- **Prompt rate across lengths:** C's block medians ranged from 508.0 to 537.4 tok/s from 4,096 to 128,000 tokens
  in the first session.
- **Memory:** `strata.exe` resident set 34.9–36.6 GiB after load and up to 39.5 GiB after the runs; Windows'
  available RAM 12.0–14.5 GiB after load, lowest 9.4 GiB (after the 128K runs, conversations parked). VRAM in use
  7,339–7,508 MiB of 8,192 MiB after load, up to 7,603 MiB after the runs. Snapshots, not peaks; paging was not
  measured. Eight of the ten startup records include the warning "that is little room for the verify windows'
  buffers under WDDM"; the E1 and E2 records (explicit chunk) do not. All 108 measured requests completed.

**For context only (not an A/B):** the [RTX 5070 Ti report](../2026-10-06-community-rtx5070ti-5900x/README.md) has
the same CPU, the same 64 GB DDR4-3600 in four modules and PCIe Gen4 x16, with a 16 GB card, engine 0.1.39, a 64K
context, Windows 10 and driver 591.86. Its IQ2_XS medians are 1,992 prompt / 102.0 decode tok/s at 4K and 2,817 /
101.3 at 32K; here 549.8 / 47.0 at 4K (A, block 1) and 534.3 / 47.2 at 32K (C, block 6).

## Recall

`tools/needle_bench.py --lengths 32k,128k --depths 10,50,90` on arm C (block 6) returned the exact target in all
six cases: depths 10%, 50% and 90% at 32,895–32,896 prompt tokens and at 125,868–125,870 tokens, one attempt per
length and depth ([`needles.json`](needles.json)). The output does not record reused-token counters, so its
elapsed times (33–63 s and 204–242 s) are not used as throughput. Three lengths appear in this report: the
configured limit 131,072, the throughput prompt 128,000, and the longest recall prompt 125,870.

## Supplementary checks (single runs, own scripts)

These are functional checks with one greedy run each; their request-level records are in [`quality/`](quality/).

- **Task check:** 13 prompts defined in [`quality/strata_eval.py`](quality/strata_eval.py) (`TASKS`), run once on
  the 32K default before the conversation cache was added (`task-check-32k-default.json`) and once on C
  (`task-check-C.json`). They cover a rate word problem and a trick logic question (each with and without
  reasoning), six factual questions in one prompt, a Python expression evaluator without `eval`, a binary-search
  bug fix, strict JSON output, a four-line seven-character Chinese poem, a Chinese-to-English translation, a Chinese
  summary, a needle in ~21.8K tokens, and an OpenAI tool call. [`quality/check_results.py`](quality/check_results.py)
  scores each record; its outputs are `task-check-32k-default.scored.txt` and `task-check-C.scored.txt`. Scored by
  the script on both records: the generated evaluator passes 15/15 listed expressions and rejects 4/4 malformed
  inputs; the JSON parses and matches the schema; the poem has four lines of seven characters; the tool call is
  `get_weather` with Beijing in Celsius (the city written in English on the 32K record, in Chinese on C); the word
  problem contains 24/7, both logic answers start with 1, the needle answer is exact, and the bug fix proposes
  `lo = mid + 1`. The six factual answers, the translation and the summary are printed in the scored files and
  were judged correct by reading; they are not machine-scored.
- **Conversation cache** ([`quality/parking_and_long_context.py`](quality/parking_and_long_context.py), records
  `parking-and-long-context-C.json`, quoted here, and `parking-and-long-context-B.json` from an earlier run with
  B's settings): two conversations of ~4.7K tokens alternating A1, B1, A2, B2 on C. The
  follow-ups read 24 and 30 fresh tokens with 4,621 and 4,835 reported as reused, and took 0.6 s and 1.4 s against
  10.2 s and 11.1 s for the first turns. One run; it shows the feature working, not a measured speedup.

## Limitations

- One machine, one model, one engine version; configurations measured as blocks, not interleaved per request.
- Only two `--kv-resident` values were tried with k8v4 (32,768 and 20,480); 20,480 was not tried with int8;
  `q4_0`, other explicit chunk sizes, vision and `"parallel"` were not tested.
- The isolation runs (E1, E2) used an explicit chunk and an environment variable; the binary does not log which
  staging path each chunk took.
- The display shares the GPU, so the VRAM left for the expert cache depends on the desktop; a PC without a display
  on the card would plan differently.

## Suggestions for the maintainers

- On this machine, `--kv-resident 20480` was a useful alternative to setup's 32,768 for 128K with k8v4: it left
  the planner room for a 1,024-token chunk, at or above the streaming threshold. Broader testing (other cards,
  int8, other window sizes) is needed before changing a default.
- A start-log line saying that `--prefill auto` chose a chunk below `stream_all_min()` (and therefore 8-slot
  staging) would make this easy to spot; today only the chunk and ring numbers hint at it.
- `setup.py` prints `KV cache: 4-bit (Hadamard-rotated)` for `--kv k8v4` (line 5034 labels every non-int8 choice
  as 4-bit); the config it writes does hold `k8v4`.
