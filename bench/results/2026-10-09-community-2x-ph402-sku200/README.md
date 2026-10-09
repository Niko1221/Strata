# Community benchmark on 2× NVIDIA PH402 SKU 200 (four GP100 dies, sm_60)

Measured on 2026-10-09 by ruibeikaa.

**Setup:** Swift 1.5 IQ3_XXS with Q8_0 dense weights, 1,048,576-token context, layer split `12,24,36` over four
GP100 dies, every expert in VRAM. The engine is a source build for sm_60 that carries open PRs and local patches
(listed below).

**Main result:** `--pipeline-windows 2` (#1656 + #1674) on the four-stage split raises decode by
**+14.6% to +28.8%** on 9 prompts of 87 to 301,147 tokens. Replies were identical bit for bit across all three arms.
Prompt reads were unchanged.

**Prompt path:**
- A local sm_60 MMQ patch (next9) takes 2.5–6.0% less time to read the batched prompts (bracketed starts), with
  replies identical.
- `STRATA_PREFILL_PIPE` 290 instead of 768 takes 13.2% less time to read a 1,158-token prompt and 5.2% less for a
  34,835-token prompt, but 1.7% more for 1,027 tokens appended to that conversation (one run each). It changes the
  chunking, and with it the output bits.
- `spec_min_p` 0.85 stays the best draft floor with pipelining on, for both code prompts.

**Main limitations:**
- One run per arm for the pipelining A/B.
- No stock v0.1.41 arm.
- One model.

## Hardware and software

- **GPU and VRAM:** 2× NVIDIA PH402 SKU 200, all four GP100 dies used.
  - Each die: compute capability 6.0, 48 SMs, 31.91 GiB visible to CUDA, TCC mode.
  - Links: the two dies of a board copy peer-to-peer at 73.0–73.2 GB/s one way (NVLink inside the board). Peer
    copies across the two boards run at 3.3 GB/s.
- **CPU and board:** AMD Ryzen 9 9950X3D (16 cores, 32 threads), ASUS ProArt B850-CREATOR WIFI NEO.
- **RAM:** 256 GiB (4 × 64 GiB DDR5 at 5600 MT/s).
- **Storage:** model, pack and PLE table on a BIWIN X570 PRO 8 TB NVMe SSD.
- **PCIe:** the engine's probe measures 3.3 GB/s host to device on every die. Each die reports a PCIe 3.0 x16 link
  to its board's switch, and both boards sit behind one further switch on one root port (Windows device
  properties). The generation and width of the boards' uplinks were not recorded; those properties do not report
  them.
- **Power and clocks:**
  - Power limit 140.00 W per die (default = max; min 120 W).
  - Application clocks set to 715 MHz memory / 1050 MHz graphics with `nvidia-smi -ac 715,1050`.
- **Other workloads:**
  - Nothing else ran on the four dies during the A/B runs.
  - The draft-floor sweep ran on an engine that was already serving.
  - The PC also holds two RTX 3090 Ti, which ran no Strata engine during these runs. Other use of them was not
    monitored.
- **OS, driver, CUDA:** Windows 11 Pro for Workstations (build 26300), driver 581.80 (TCC), CUDA 12.9.41.
- **Builds:** all from source with MSVC 19.44 and Ninja, using
  `-DCMAKE_CUDA_ARCHITECTURES=60 -DSTRATA_EXPERIMENTAL_SM60=ON`. All arms used the server from v0.1.41
  (`serve/server.py`, fb58e0d).

| Build | Engine | Source | Binary SHA-256 |
| --- | --- | --- | --- |
| next7 | 0.1.40.4 | e9265395: v0.1.40.4 + the same PH402 patch set and PRs as next8 (#1441 in an earlier form), plus upstream's `STRATA_GR_FAST` pair a7334265 + a01a789e (off on sm_60); no #1674, #1656 or ba156160 | `878fd580…f13281` |
| next8 | 0.1.41 | v0.1.41 (fb58e0d) + #1674 (9e67094a) + the same PH402 patch set + #1656 (06abdfa5, cherry-picked as c467317a) + ba156160; built from ba156160 (e90187a3 on top adds docs only) | `4a2c9505…fe11d2` |
| next9 | 0.1.41 | next8's source, linked against ggml-cuda as v0.1.41 pins it (3cf03257) plus one local patch to the sm_60 MMQ path (the dp4a fallback as PRMT sign extension + 16-bit products; the q8_0 dot product for J ≤ 16 unrolled with a VMAD tree). The diff is in [this comment on #1639](https://github.com/Niko1221/Strata/issues/1639#issuecomment-6086238713). | `8a631d56…22803f` |

next8's commit list, oldest first (I can push either tree on request):
- 9e67094a (#1674, @CC-David-CC)
- c8fb4876 (local: tiled QSA block scorer below sm_80)
- 02ad1ee8, 24f7f712, ffb6fcd9, 80131a7d, 3a70155d (#1424)
- 1801d32a, 6a48d8c4, 1c8cc07c (the first three commits of #1441)
- 0f9beb97 (#1660)
- 72d1acac (#1368, @sergqwer)
- f3f04d9b (#1525, @sergqwer)
- 34d090ac, 60113077, e344d920, 98ea126f (local: async commit on the split, MTP head on the Pascal Q8_0 GEMV,
  long-context QSA select)
- c467317a (#1656, @Cass67)
- ba156160 (local: pipelined windows use the long-context select too)

## Model and configuration

- **Model:** [`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  IQ3_XXS, downloaded 2026-10-01. The repository revision was not recorded.
  - `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`, 39,785,790,560 bytes, repository sha256
    `3bddaa667c750f63baca766df9433e1afd35a139f401c5ca5e7ff560ceff3d23`.
  - `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf`, 36,180,282,560 bytes, repository sha256
    `b0b15f782af71eb471909d2f3313b2927c324962f86c084325c247cd411a0160`.
  - The local copies match these by size. The local hashes were not re-checked.
- **Q8_0 dense weights:** the dense projections and the head were converted to Q8_0 on 2026-10-07 with an earlier
  revision of `tools/q8_dense_gguf.py` (#1424); the shards predate ffb6fcd9, which made it convert every shard.
  This gives two shards of 41,278,700,640 and 38,454,463,680 bytes. Shard 1 is the engine's `--native` and
  `--ple-gguf` file, used with `STRATA_Q8_SM60=1`. They are local conversions, not downloads; their sha256 and the
  exact command were not recorded.
- **Pack, MTP head, profile:**
  - Pack `swift-iq3_xxs`: 8 files, 1,548,226,697 bytes, per-file sha256 manifest `110caea5…ff229`.
  - MTP head `mtp/rt`: 4 files, 824,314,948 bytes, manifest `c74a4334…5aad251`.
  - Expert profile: `data/expert-profile.bin` as shipped in v0.1.41 (sha256 `8f59b4aa…975baf`).
  - The pack and MTP preparation commands were not recorded.
  - The manifest is the sha256 of the `sha256  relative/path` lines (two spaces, `\n` after each), sorted by path.
- **Vision:** no vision encoder loaded (text-only run config).
- **Context and KV:** 1,048,576 tokens (`--rope-scaling yarn --rope-scale 4` over a trained 262,144). KV int8 in
  VRAM, no KV streaming.
- **Expert cache:** `auto`, 6,144 slots per die, so all 24,576 experts resident ("100% of the experts resident").
- **Prompt path:** prefill `auto` (8192-token chunks, a 96-slot ring). Chunk size for the split comes from
  `STRATA_PREFILL_PIPE` (b/a): 768 in every arm except the 290 and 450 arms.
- **MTP and drafting:** `--spec 4 --spec-min-p 0.85 --mtp-window 8192`. The suffix drafter is at its default, so
  verify windows reach up to 6 tokens.
- **Sampling:** greedy (`temperature 0`). `reasoning_effort` not set, so the template default applied (thinking on,
  effort xhigh). Whether each capped reply ended in reasoning or in answer text was not recorded.
- **Not used:** `--calibrate`, the experimental speed projection, low-RAM mode.
- **CPU workers:** engine default (15 pool workers + the host thread).

```text
serve/server.py (v0.1.41) --engine strata --config <run config, gpu = the four dies>
engine command line as the engine logged it (paths shortened):
  strata.exe --serve --pack <packs/swift-iq3_xxs> --native <Q8_0-dense shard 1> --ple-gguf <Q8_0-dense shard 1>
    --expert-profile <data/expert-profile.bin> --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.85
    --mtp <mtp/rt> --max-context 1048576 --kv int8 --layer-split 12,24,36 --mtp-window 8192
    [--pipeline-windows 2] --conversation-cache-mib 16384 --conversation-cache-slots 3 --rope-scaling yarn --rope-scale 4
environment, every arm: STRATA_Q8_SM60=1 STRATA_MMQ_RESIDENT_SORT_NE=1 STRATA_DF_BRANCH=1 STRATA_PREFILL_PIPE=768

arms:
  next7, next8-pw-off      no --pipeline-windows;  + STRATA_PREFILL_TIMING=1 STRATA_DECODE_TIMING=1
  next8-pw-on              --pipeline-windows 2;   + STRATA_PREFILL_TIMING=1 STRATA_DECODE_TIMING=1
  next8-768, next9-768     --pipeline-windows 2
  next9-290, next9-450     --pipeline-windows 2;   STRATA_PREFILL_PIPE=290 / 450
  next9-768-prefill-timer  --pipeline-windows 2;   + STRATA_PREFILL_TIMING=1
  draft-floor sweep        next8, --pipeline-windows 2; per-request "strata_tune": {"spec_min_p": v}
```

## Method

- **Prompts.** [`make_prompts.py`](make_prompts.py) rebuilds the long text and checks its sha256:
  - Source: three engine source files of v0.1.40.3 (d5ea713), each after a `===== FILE: <path> =====` line:
    `src/prefill/prefill.cpp`, `src/core/verify.cpp`, and the first 647,925 characters of
    `src/program/generate.cpp`.
  - Size: 1,059,516 characters, 301,005 tokens by `tools/strata_tokenizer.py`.
  - Slices: the requests use its first 3,500 characters, its first 105,000, characters 105,000–108,500, and the
    whole text, each followed by a question.
  - Short prompts: a CSV-parser task, an LRU-cache module task and two arithmetic word problems.
  - Every new prompt starts with a run tag such as `[pc]`; follow-up turns do not. The exact strings are in
    [`bench.py`](bench.py).
- **Requests.** [`bench.py`](bench.py) holds the four request plans (`pw`, `ab`, `pf`, `mp`) and the log parser.
  - Requests are sent one at a time, non-streaming, to `/v1/chat/completions`.
  - Every reply ran to its `max_tokens` (64, 128, 256 or 512).
  - The engine start and stop between arms is not in the script. The launch lines are above.
  - `bench.py` is a cleaned copy of the scripts that produced these numbers (local paths, ports and the engine
    start/stop removed). Run once afterwards with plan `ab` against the same route on next9 at
    `STRATA_PREFILL_PIPE=290`, it reproduced that arm's 5 reply hashes; the 34,835-token read took 46.423 s
    (46.545 s in the arm).
- **Starts and warm-up.** Every A/B arm is a fresh engine start with loading excluded. The draft-floor sweep is the
  exception: it ran on an engine that was already running, after one warm-up request ("Say hello", 8 tokens).
  - Each start serves two warm-up requests that are not counted: "Say hello" and a 1,159-token read, 16 tokens
    each.
  - In the A/B starts the PLE table was not pre-read before the first request.
- **Order.**
  - Pipelining A/B: next7, then next8 with the flag off, then next8 with `--pipeline-windows 2`.
  - Prompt path: next8, next9, next8, next9 with the prefill timer, then next9 at 768, 290, 450 and 768 again.
- **Reuse.** In the `pw` plan, three requests form one conversation:
  - B reads 301,108 tokens fresh.
  - C follows up and reuses 301,103 of them.
  - E is another conversation and parks it.
  - F returns to it from the conversation cache and reuses 301,125.

  In the `ab` plan, the 1,027-token request is appended to the 34,835-token conversation and reuses 34,830. All
  reused and read counts come from the engine's per-request line.
- **Timing.**
  - Prompt and decode tok/s are the engine's per-request numbers (`strata serve: prompt N tokens = R reused + F read
    in … ms, G generated in … ms`). Decode is never generated tokens / request time.
  - ms per window comes from `STRATA_DECODE_TIMING`.
  - Window classes come from the engine's `strata pipeline` lines.
  - `client_latency_s` is the whole non-streaming request at the client.
  - TTFT was not measured.
- **Memory.** Startup snapshot only: the engine's "VRAM free with everything loaded" line. Peak was not measured.
- **Reply check.**
  - Plans `pw`, `ab` and `pf` hash each reply as sha1(content + "|" + reasoning)[:12].
  - Plan `mp` uses sha256(reasoning + NUL + content)[:12].

## Results

### 1. Pipelined decode windows on four stages

Decode tok/s, one run per arm.
- **next7:** v0.1.40.4 + patches.
- **off:** next8 without the flag.
- **on:** next8 with `--pipeline-windows 2`.

The reply hash of every prompt is the same in all three arms (9 of 9 prompts, 27 replies).

| Prompt | Prompt tokens | Reused | Generated | Prompt tok/s off / on | next7 | off | on | On vs off | ms per window off → on | Drafts accepted off → on |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CSV parser + 3 tests | 90 | 0 | 256 | 67.2 / 67.3 | 40.3 | 40.5 | **51.2** | +26.4% | 38.58 → 29.60 | 95/105 → 89/99 |
| word problem (pens) | 112 | 0 | 256 | 74.6 / 67.6 | 42.5 | 42.7 | **51.2** | +19.9% | 40.75 → 33.10 | 111/127 → 107/137 |
| LRUCache module + tests | 87 | 0 | 512 | 65.2 / 63.3 | 47.5 | 47.6 | **61.3** | +28.8% | 44.13 → 32.73 | 268/282 → 257/286 |
| source text, fresh read | 1,158 | 0 | 64 | 230.1 / 230.2 | 45.0 | 45.1 | **53.6** | +18.8% | 47.27 → 36.16 | 34/43 → 31/44 |
| source text, fresh read | 34,835 | 0 | 64 | 681.6 / 681.1 | 43.0 | 43.2 | **49.5** | +14.6% | 42.36 → 35.92 | 29/29 → 28/30 |
| source text, fresh read (B) | 301,108 | 0 | 64 | 812.1 / 811.0 | 42.9 | 43.2 | **53.7** | +24.3% | 47.80 → 37.24 | 33/34 → 32/34 |
| follow-up in B's conversation (C) | 301,130 | 301,103 | 256 | 9.3 / 10.0 | 41.8 | 41.7 | **50.3** | +20.6% | 48.01 → 37.12 | 130/143 → 121/142 |
| another conversation (E) | 114 | 0 | 64 | 28.0 / 29.2 | 46.6 | 46.6 | **55.2** | +18.5% | 47.36 → 34.13 | 36/42 → 31/37 |
| back to B's conversation (F) | 301,147 | 301,125 | 128 | 11.2 / 11.1 | 39.5 | 39.6 | **46.3** | +16.9% | 43.12 → 33.33 | 53/54 → 45/60 |

- **Runs and TTFT.** One run per cell. TTFT not measured.
- **Prompt reads are unchanged by the flag.** The 34,835-token read took 51.111 s off and 51.144 s on (next7
  50.971). The 301,108-token read took 370.772 s and 371.265 s (next7 368.376).
- **Low prompt tok/s on C, E and F.** C and F read only 27 and 22 new tokens. Inside C and E the engine logs a
  conversation-cache park of the 301K conversation (on arm: 2,182.8 and 2,122.4 ms), and inside F a restore
  (1,482.8 ms). Their read time includes that work (an inference from where those lines fall in the log), so
  their prompt tok/s is not a throughput figure.
- **next7 against next8 off:** 0.0–0.3 tok/s apart on every prompt, with the same replies.
- **Startup VRAM free with everything loaded:** next7 8,379 MiB, off 8,450 MiB, on 8,325 MiB. With the flag the
  engine logs "216 MiB of CUDA0 kept out of the expert cache".

Where the decode time goes with the flag on. Counts and class means are from the engine's `strata pipeline` lines
("speculative" = guesses launched, "on the path" = held). Held of launched is held / launched. The fresh share of
decode time is fresh n × mean over the sum of n × mean for both classes (both derived).

| Prompt | Windows | Guesses launched | Held | Held of launched | Fresh windows: n, mean ms | Held windows: n, mean ms | Fresh share of decode time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CSV parser + 3 tests | 169 | 153 | 85 | 55.6% | 84, 37.86 | 85, 21.44 | 63.6% |
| word problem (pens) | 151 | 145 | 72 | 49.7% | 79, 43.23 | 72, 21.98 | 68.3% |
| LRUCache module + tests | 255 | 236 | 146 | 61.9% | 109, 43.83 | 146, 24.44 | 57.2% |
| 1,158-token read | 33 | 31 | 17 | 54.8% | 16, 47.96 | 17, 24.62 | 64.7% |
| 34,835-token read | 36 | 33 | 13 | 39.4% | 23, 42.55 | 13, 24.17 | 75.7% |
| 301,108-token read (B) | 32 | 29 | 14 | 48.3% | 18, 49.53 | 14, 21.41 | 74.8% |
| follow-up at 301K (C) | 137 | 124 | 66 | 53.2% | 71, 48.81 | 66, 24.54 | 68.1% |
| another conversation (E) | 34 | 32 | 19 | 59.4% | 15, 52.20 | 19, 19.84 | 67.5% |
| back to B's conversation (F) | 83 | 75 | 44 | 58.7% | 39, 43.90 | 44, 23.89 | 62.0% |

- **Held vs fresh windows.** A held window (a guess that was kept) took 19.84–24.62 ms on average, a fresh one
  37.86–52.20 ms.
- **Draft acceptance.** With the flag, 70.5–94.1% of drafts were accepted. Without it, 79.1–100%.
- **Gate.** The gate (theta 0.20, not changed) kept 1–18 guesses per request from launching.

### 2. Prompt path: the sm_60 MMQ patch and STRATA_PREFILL_PIPE

All arms run `--pipeline-windows 2`. Cells show prompt tok/s, with read seconds in parentheses. The two next8 runs
are given as both values in start order. The three next9 768 runs are given as the median with the range in
brackets, and the median's read seconds. The run count is per column.

| Request | Prompt tokens | Reused | Generated | next8, 768 (2 runs) | next9, 768 (3 runs) | next9, 290 (1 run) | next9, 450 (1 run) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source text, fresh read | 1,158 | 0 | 64 | 236.9, 234.7 (4.888, 4.933 s) | 250.0 [246.0–250.3] (4.632 s) | 288.2 (4.018 s) | 288.4 (4.015 s) |
| source text, fresh read | 34,835 | 0 | 64 | 692.0, 689.6 (50.337, 50.516 s) | 708.2 [708.1–711.4] (49.186 s) | 748.4 (46.545 s) | 737.5 (47.233 s) |
| 1,027 tokens appended | 35,857 | 34,830 | 64 | 223.6, 223.1 (4.593, 4.603 s) | 236.7 [236.2–237.7] (4.339 s) | 232.4 (4.419 s) | 235.3 (4.365 s) |
| CSV parser + 3 tests | 90 | 0 | 256 | 70.4, 70.5 (1.278, 1.277 s) | 76.9 [76.9–77.3] (1.170 s) | 77.2 (1.166 s) | 76.9 (1.170 s) |
| word problem (pens) | 112 | 0 | 256 | 65.7, 65.5 (1.704, 1.710 s) | 69.0 [68.8–71.1] (1.623 s) | 68.4 (1.638 s) | 61.5 (1.820 s) |

Chunks, from the engine's `strata prefill: N tokens in C-token chunks` line:

| Arm | 1,158-token read | 34,835-token read | 1,027 appended |
| --- | ---: | ---: | ---: |
| 768 | 1,153 tokens in 768-token chunks | 34,830 in 3584 | 1,022 in 512 |
| 290 | 512-token chunks | 1792 | 512 |
| 450 | 512-token chunks | 2560 | 512 |

- **MMQ patch (next8 → next9), bracketed starts only (next8, next9, next8).** Derived against the mean of the two
  next8 starts:

  | Request | Change |
  | --- | ---: |
  | 1,158-token read | −4.1% |
  | 34,835-token read | −2.5% |
  | 1,027 tokens at 34,830 | −6.0% |
  | 90-token read | −8.4% |
  | 112-token read | −4.6% |

  - All five replies were identical in all five starts of next8 and next9 at 768.
  - Decode moved by at most 0.8 tok/s between the builds in these three starts, and by up to 1.1 between the two
    next8 starts.
  - The two later next9 starts at 768 read the 1,158-token prompt in 4.627 and 4.632 s, faster than the bracketed
    next9 start (4.707 s). No next8 start ran beside them.
- **STRATA_PREFILL_PIPE 768 → 290 → 450.** Derived against the mean of the two next9 768 starts that bracket them:

  | Value | 1,158-token read | 34,835-token read | 1,027 appended |
  | --- | ---: | ---: | ---: |
  | 290 | −13.2% | −5.2% | +1.7% |
  | 450 | −13.3% | −3.8% | +0.5% |

  - The two 768 starts gave identical replies on all five requests.
  - 290 and 450 changed the reply of every request whose prompt was chunked differently:
    - The 1,158-token read: both gave `88782948173b`.
    - The 34,835-token read.
    - The 1,027-token follow-up, whose own chunks stay 512 but whose 34,830-token history was chunked differently.
  - The two short prompts kept their replies.
  - Starts with the same chunking gave the same reply: the two 768 starts on all five requests, and 290 and 450 on
    the 1,158-token read (512-token chunks in both). Other chunkings gave other bits than 768.
- **Why 290 (derived).** A `STRATA_PREFILL_TIMING` profile of next9 at 768 gives, per stage and chunk,
  281.5 ms fixed + 0.971 ms per token. b/a is therefore 290 tokens.
  - This is a two-point fit: 512-token chunks at about 30K context, and 3584-token chunks from 0 to 30K. It
    over-predicts short-context chunks (768 tokens at 1K: 1,027 ms predicted, 954.7 measured).
  - The timer itself made the reads 0.7–2.7% slower (4.753 s and 49.545 s against the untimed next9 starts).

### 3. Draft floor (`spec_min_p`) with pipelining on

Decode tok/s in round 1 and round 2, with drafts accepted/offered in parentheses. The sweep ran on next8 at
production settings with one-shot requests (`"strata_checkpoint": false`), in two rounds: per prompt, the values in the
order 0.85, 0.5, 0.7, 0.95, then in the reverse order (`round` 0 and 1 in the data). Every
prompt gave one reply hash across all four values and both rounds.

| Prompt (tokens in / out) | 0.5 | 0.7 | 0.85 | 0.95 |
| --- | ---: | ---: | ---: | ---: |
| CSV parser + 3 tests (90 / 256) | 45.3, 45.4 (112/170) | 47.4, 47.4 (99/119) | **47.8, 47.7** (86/95, 86/99) | 47.5, 47.5 (64/66) |
| LRUCache module + tests (87 / 512) | 47.9, 47.9 (242/358, 243/361) | 50.4, 50.5 (209/261) | **51.2, 51.2** (177/197, 178/196) | 50.1, 50.1 (136/146) |
| word problem (train timetable) (108 / 256) | 66.6, 66.6 (171/220) | **68.9, 69.0** (163/189) | 68.6, 68.6 (150/167) | 67.8, 67.8 (138/146) |

The largest difference between rounds is 0.1 tok/s. 0.85 is fastest on both code prompts, and 0.7 is 0.3–0.4 tok/s
faster on the word problem.

### Files in this folder

- `runs-pipeline-windows.jsonl` (27 rows), `runs-prompt-path.jsonl` (38 rows) and `runs-draft-floor.jsonl`
  (24 rows): one row per measured request.
  - **Identity:** `arm`, `binary`, `binary_sha256`, `engine_start` (start number within the file), `plan`,
    `order`, `request`, `turn` (new or follow-up), `max_tokens`.
  - **Tokens and timings (engine):** `prompt_tokens`, `reused_tokens`, `read_tokens`, `read_ms`, `prompt_tok_s`,
    `generated_tokens`, `decode_ms`, `decode_tok_s`, `drafts_accepted`, `drafts_offered`, `batched_chunks`
    ([tokens, chunk size] from the engine's prefill line).
  - **Timers and pipeline (pipelining file only):** `decode_timing` (from `STRATA_DECODE_TIMING`; ms) and
    `pipeline` (window counts, class means in ms, tokens per window, calibration as held/scored per p_on decile).
  - **Client and replies:** `client_latency_s` (client-side, non-streaming; null for the sweep), `reply_hash` and
    `hash_rule`.
  - **Sweep only:** `round` and `spec_min_p`.
- `engine-pipeline-windows.txt`, `engine-prompt-path.txt`, `engine-draft-floor.txt`: engine log excerpts.
  - They contain each start's command line (paths shortened), the startup lines quoted above, and every
    per-request line.
  - The per-request `strata prefill timing` lines are left out.
  - The sweep's excerpt starts at its own warm-up request.
- `make_prompts.py`, `bench.py`: the prompt builder and the request driver.

## Correctness and limitations

- **Reply checks:**
  - Pipelined equals serial bit for bit on all 9 prompts, up to 301,147 tokens, on CUDA sm_60 with four stages.
  - next8 with the flag off equals next7 on all 9.
  - The MMQ patch left all 5 replies unchanged.
  - `STRATA_PREFILL_PIPE` 290 and 450 changed the reply of each of the 3 requests whose chunking they changed, and
    of neither of the 2 whose chunking they did not.
  - The replies were not graded. Every one stopped at its token cap.
- **Runs:**
  - Pipelining A/B: one run per arm.
  - Prompt path: two and three runs for the 768 arms, one each for 290 and 450.
  - Draft floor: two rounds.
  - This is below the three-run rule. The noise checks are the bracketing arms:
    - next7 vs next8 off: decode 0.0–0.3 tok/s apart.
    - The two next8 768 starts: reads 0.1–0.9% apart, decode within 1.1 tok/s.
    - The two next9 768 starts around 290 and 450: the three batched reads 0.1–0.5% apart, the 112-token read
      1.575 vs 1.623 s.
- **Timers.** The three pipelining arms had `STRATA_PREFILL_TIMING` and `STRATA_DECODE_TIMING` on, so their read
  times include the timer. On next9 the prefill timer cost 0.7–2.7% (derived above). The timers were on in all
  three arms. The decode timer's own cost was not measured.
- **No stock baseline.** There is no stock v0.1.41 arm. Every arm carries the PH402 patches and the open PRs listed
  above.
- **#1674 not isolated.** #1656 was not run without #1674 here, so this does not show whether the #1674 guard is
  needed on this rig. With it, the replies match.
- **Clocks.** In a separate 300K fresh read earlier the same day, the dies averaged 995–1021 MHz (2.7–5.3% under
  1050), with minima of 797–911 MHz and peaks of 144–148 W. The driver flag read "SW thermal slowdown". Our
  reading (an inference) is the 140 W cap. Long reads on this rig therefore run a little below the locked clock.
- **Draft-floor sweep.** It ran on an engine that had served other requests before, with one-shot requests and
  different prompt tags, so its speeds are not comparable one to one with section 1.
- **Not tested:**
  - stock v0.1.41
  - `--pipeline-windows` on two or three stages
  - the Flash-Next model on this rig with and without the flag
  - sampled decoding, other reasoning efforts or reasoning off
  - images and the vision encoder
  - `--batch`
  - a long soak
  - TTFT and peak memory
  - needle recall
  - contexts above 301,147 tokens

## Related comments

- [#1656](https://github.com/Niko1221/Strata/pull/1656#issuecomment-6086159617): the same pipelining A/B, plus a trace of where the pipelined decode time goes (window classes,
  late held windows, rollback cost, gate calibration).
- [#1674](https://github.com/Niko1221/Strata/pull/1674#issuecomment-6086162177): the equality check for the one-token commit guard.
- [#1639](https://github.com/Niko1221/Strata/issues/1639#issuecomment-6086238713): the sm_60 MMQ patch as a diff, its harness numbers per rows per expert, and what did not help.
