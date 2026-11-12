# Community benchmark on 4x RTX 3090 (multi-user, custom track)

Measured on 2026-11-08 by havok2-htwo. Three arms on one machine: stock v0.2.12 as `setup` writes it (S), the same stock
binary and server with a 16-slot production config (C), and v0.2.12 plus an opt-in patch series (PT). This folder holds the
custom multi-user data. The standard protocol runs of the same servers are the six folders
`2026-11-08-standard-4x-rtx-3090-*` next to it. Main limitation: one machine, one model and quant, one day, 2 runs per arm.

## Results

`bench/results` has no 4x RTX 3090 multi-user report yet. TEST_REQUESTS.md lists "Three or four NVIDIA GPUs" as a wanted test (#880). This PR adds results only. It has no engine change, makes no speed claim and proposes nothing for the README headlines.

Three arms were measured on one machine on 2026-11-08, interleaved, each on a freshly loaded server. All arms use the same model files, the same expert profile (same SHA-256) and the same chat template (same SHA-256):

| Arm | What it is |
| --- | --- |
| **S** | Stock v0.2.12 (`146df3c4`) as `setup` writes it for this box: 128K context, `--batch 3`, `--prefill auto`, `--layer-split auto`, int8 KV, MTP, vision off. |
| **C** | The same stock binary and stock server, with our production args/env: 16 slots, 262K context, `--prefill 512`, `--layer-split 12,25,37`, resident experts, conversation and prompt cache, the vision encoder loaded on the CPU, the stock sm_86 default table. Bit-changing defaults are pinned off. |
| **PT** | v0.2.12 plus an opt-in patch series by havok2-htwo (branch head `a099d295`, 56 commits on `146df3c4`). It uses the same args as C plus the series' engine switches, and runs with the series' server. |

Run order: C, PT, S, C-r2, PT-r2, S-r2.

**Standard protocol** (`strata bench run`, required stage plus `--optional-concurrency`, `--track custom`). All six runs are `complete` and score 12/12 on sanity. The table shows medians from run 1. Run 2 is within 2 % on every value shown, except S's c4 burst rounds 2 and 3 (184 / 184 tok/s in run 2).

| Case | Metric | S | C | PT |
| --- | --- | ---: | ---: | ---: |
| short | decode tok/s | 127.4 | 136.1 | 154.3 |
| short | TTFT s | 0.49 | 0.45 | 0.23 |
| prompt-4k | prompt tok/s | 1,921 | 4,998 | 4,999 |
| prompt-4k | decode tok/s | 128.3 | 138.4 | 155.2 |
| prompt-16k | prompt tok/s | 3,219 | 5,774 | 6,009 |
| prompt-16k | decode tok/s | 127.9 | 117.4 | 124.6 |
| prompt-16k | TTFT s | 5.12 | 2.86 | 2.75 |
| short-burst (c4) | aggregate tok/s, 3 rounds | 198 / 202 / 196 | 278 / 283 / 268 | 282 / 296 / 277 |
| short-burst (c4) | TTFT s | 0.75 | 0.40 | 0.49 |

**Multi-user, custom harness.** Greedy (temperature 0), with `sampling.presence_penalty: 0` in every arm's config. Values are the mean of 2 runs per arm, or the range where both runs are shown.

| Measure | S | C | PT |
| --- | ---: | ---: | ---: |
| 16 users, short (~150-token prompts), aggregate tok/s | 220.1 | 539.0 | 527.2 |
| 16 users, long (~1.2K-token prompts), aggregate tok/s | 174.7 | 400.8 | 405.1 |
| 8 users, short, aggregate tok/s | 216.6 | 412.8 | 400.1 |
| Per-user decode p50 at 1 / 2 / 4 users | 142 / 86 / 82 | 131 / 87 / 75 | 145 / 128 / 90 |
| TTFT short at 16 users, p50 / p95 (s) | 14.6-20.5 / 44.5-51.5 | 0.99 / 2.95-3.08 | 0.37-0.38 / 4.23-4.32 |
| TTFT long at 16 users, p50 / p95 (s) | 18.7-18.8 / 65.8-69.4 | 2.54 / 7.76-7.78 | 0.79 / 9.39-9.42 |
| 16 users x 3 turns (8K doc): wall / turn-1 TTFT p50 / follow-up TTFT p50 (s) | 266-281 / 46.6-67.1 / 23.6-48.8 | 97.6-97.7 / 17.4-17.6 / 2.03-2.08 | 90.8-91.3 / 26.4-27.9 / 2.74-2.95 |
| Cold 32K joiner beside 4 streams (`tools/piped_read_join_bench.py`): worst stream gap / joiner TTFT (s) | queue-bound, not compared | 5.80 / 5.80 | 0.10-0.11 / 8.2-8.3 |
| `tools/strata_bench.py --clients 16 --long`, 16 clients, tok/s, median (range) of 3 runs, first server start only | 160.7 (158.1-161.5) | 275.3 (247.8-275.3) | 266.9 (253.3-267.7) |

**How to read this:**
- **Most of the 16-user gain over S comes from configuration, not engine code.** The configuration changes are 16 slots instead of 3, `--prefill 512` and the caches. With the stock engine, C already gets +145 % (short) and +130 % (long) over S.
- **What the patch series itself adds (PT vs C):**
  - per-user decode is +10 % at 1 user, +47 % at 2 users and +19 % at 4 users;
  - single-request decode in the standard protocol is +12-14 % on short and 4K, and +6-7 % on 16K;
  - a cold 32K joiner no longer freezes the running streams (gap 0.1 s instead of 5.8 s);
  - multi-turn wall time is -7 %.
- **Where PT is worse than C** (listed again under Limits):
  - the 16-user short aggregate is -2.2 %, and the 8-user short aggregate is -3.1 %;
  - TTFT p95 at 16 users is worse with short prompts (4.2-4.3 s instead of 3.0-3.1 s) and with long prompts (9.4 s instead of 7.8 s);
  - the 32K joiner's own TTFT is 8.2 s instead of 5.8 s;
  - c4 burst TTFT is 0.49 s instead of 0.40 s;
  - in the multi-turn test, turn-1 TTFT p50 is 26-28 s instead of 17-18 s, and follow-up TTFT p50 is 2.7-3.0 s instead of 2.0-2.1 s;
  - `strata_bench.py` at 16 clients is -3 %, inside the overlap of the two arms' run ranges.
- **At 16K, single-request decode is lower than S in both 16-slot configurations** (C -8 %, PT -3 %).

## Hardware, software and method

- **One fresh server per arm, with the standard runner first.** Each arm's server started fresh. Its first workload was `strata bench run --profile <arm> --no-start --optional-concurrency --track custom`. After that it ran, in order:
  - `tools/strata_bench.py --runs 3 --clients 16 --long` (first start of each arm only);
  - our `bench_concurrency.py`: 1/2/4/8/16 users, short and long prompts, 2 requests x 512 tokens per user;
  - `bench_multiturn.py`: 16 users x 3 turns, 8K document plus 3K per turn, 400 output tokens;
  - upstream's `tools/piped_read_join_bench.py --scenarios cold32k`.
- **Two stock baselines:**
  - **S** is what a user gets from `setup`.
  - **C** keeps the stock engine and server but uses PT's args. It isolates what the patch series adds.
  - Stock v0.2.12 already enables six switches from our earlier series by default when the server runs batch slots: CKPT_POOL, SLOT_CKPT_MOVE, SLOT_COPY_DIRECT, SLOT_COPY_BATCHSYNC, BATCH_SELF_COMMIT and QSA_SLOTS. C sets them explicitly and S gets them from the server default (`--batch 3`), so the PT-C delta shows only the part that is not yet upstream.
- **Env differences between PT and C** (full lists in each run's `results.json`, under `configuration.experimental_options`):
  - attention: `ATTN_DAILY=0` + `QSA_SLOTS_STREAM=1`;
  - `CONV_MMAP`, `CONV_TIER_*`, `EARLY_SAMPLE`, `GATE_EVICT`, `GIVEBACK_CHECKS`, `GR_DOWN_PF=2`, `MROPE_SLOTS`, `MSG_CKPT_TAIL=8192`;
  - `PARK_BORROW`, `PARK_CAPTURE`, `PIPE_OVERLAP(+_KVS)`, `READ_YIELD`, `SERVE_STOP_BOUND`, `SLOT_CHAIN_REUSE`, `SLOT_PARK`, `SOLO_PIPELINE`, `SPEC_LANES=4`.
  - The servers differ as well. Both set `ADMIT_SHORT_FIRST=8192`. PT's server adds `ADMIT_BY_NEW`, `YIELD_SHORT=2048`, `YIELDS_MAX=4` and `PREFIX_DIAG` (diagnostics only).
  - An earlier window on the same machine tried PT with the table's DAILY attention instead of slots-stream. It did not close the 16-user short gap (519 vs 529 tok/s). That window's numbers are not mixed into the tables above.
- **Hardware:**
  - 4x RTX 3090 24 GB, 320 W limit each, layers split over the 4 cards (S: `auto`; C/PT: `12,25,37`);
  - PCIe as reported under load: cards 0-1 gen3 x16, cards 2-3 gen4 x8; no NVLink;
  - Threadripper 3990X, 251.5 GiB RAM, model files on an NVMe RAID 0 (3x Crucial P310).
- **Software:** Ubuntu 24.04.5, kernel 7.0.0-38, driver 580.196.07, CUDA 13.0. Both binaries are source builds with CUDA 13.0, sm_86, `STRATA_MMQ_TAB=ON`, `NATIVE_EXPERTS=ON`, `Q6K_EXPERTS=ON` and `MMQ_KQUANTS=ON`. The engine reports version 0.1.58 in all arms.
- **Model files:**

  | File | SHA-256 |
  | --- | --- |
  | `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | `5ce89370720f8bf9...` |
  | `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` | `577a38a2392b40ca...` |
  | `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` | `d4634e6d84f0ebb0...` |
  | `expert-profile.bin` (same in all arms) | `8f59b4aa8873209d...` |

  The full hashes are in `results.json`.

## Files

- `strata_bench/{S,C,PT}/`: `tools/strata_bench.py --runs 3 --clients 16 --long`, its own README and results (first server start of each arm).
- `data/`: our harness, see `FIELDS.md` for fields and units. `run-<arm>[-r2]-<kind>` labels.
- `scripts/`: the harness scripts (`bench_concurrency.py`, `bench_multiturn.py`, `compare_replies.py`).
- The cold-32K join test is upstream's `tools/piped_read_join_bench.py --scenarios cold32k` (`data/join-*.json`).

## Correctness

**Passed:**
- **`strata bench run`: all six runs are `complete`, with 12/12 on sanity each.**
  - S: Strata `146df3c4`, runs `e32ea56b` and `8be2bf4a`;
  - C: `146df3c4`, runs `eb4f9649` and `be712cdc`;
  - PT: `a099d295`, runs `d8cef8bb` and `45387cb6`.
  - No request failed. No request was flagged for unexpected reuse or for a length outside tolerance. Performance requests capped at 512 tokens stay in the summaries, as the protocol says. The protocol ID and the corpus SHA-256 `e75eff8b...` match.
- **A freshly loaded engine every time:** every runner README says "the server had answered no request before this run" (`server_requests_before_run: 0`).
- **Greedy reply identity:** we compared 62 short greedy replies of 512 tokens each, one per request at 1/2/4/8/16 users with 2 requests each.
  - Each of C, C-r2, PT and PT-r2 matches a PT reference run from an earlier window, 62/62 by output SHA-256. So C and PT match each other.
  - S differs (0/62). It runs another configuration (3 slots, 128K, auto split), so identical replies are not expected.
- **An earlier window of PT against our own previous build:** 62/62 identical HTTP replies, and `batch_test` 16/16 in both solo and batch mode.
- **Run-to-run spread:** run 2 is within 2 % of run 1 on every value in the standard-protocol table except S's c4 burst rounds (see above). On the 16-user short aggregate the spread is 0.6 % for C and 0.1 % for PT.

**Failed:**
- None observed in the benchmark runs.
- Outside this report:
  - Stock C fails our vision check (NO-GO: the long image request in a busy batch differs from the same request alone, 3 of 4). This matches the batch M-RoPE bug from #1242, which is not fixed in v0.2.12. PT passes the check with the per-slot fix. No vision workload is part of these numbers.
  - In an earlier window, our `batch_overlap` test failed on PT (`ERR BGEN: no such free slot`). This is still open.

**Not run:**
- `batch_test` engine identity and the GPU ctest in this window: switched off for time. Here, the only identity evidence for PT vs C is the HTTP greedy reply comparison.
- HIP, SYCL and Metal builds: no hardware.
- Single-GPU runs.
- A soak of an hour or more (#848).
- Multi-user runs with T 1.0 sampling: the standard protocol covers T 1.0 only at concurrency 1 and 4.
- Stock as shipped with a client that sends no sampling, which gets the penalty.
- A third run per arm for the custom harness. COMMUNITY_BENCHMARKS.md says to try at least three; there are 2 per arm.

## Limits

- **Where PT loses against stock with the same config (C):**
  - The 16-user short aggregate is -2.2 % (539.0 vs 527.2), and the 8-user short aggregate is -3.1 % (412.8 vs 400.1). PT's steady-window aggregate at 16 users is higher (599 vs 573 tok/s), but admission puts the late requests behind: short TTFT p95 is 4.2-4.3 s instead of 3.0-3.1 s, and long TTFT p95 is 9.4 s instead of 7.8 s.
  - The cold 32K joiner's own TTFT is 8.2 s instead of 5.8 s. PT keeps the 4 running streams going (gap 0.1 s instead of 5.8 s), and the joiner pays for it.
  - c4 burst TTFT is 0.49 s instead of 0.40 s.
  - Multi-turn TTFT p50 is 26-28 s instead of 17-18 s on turn 1, and 2.7-3.0 s instead of 2.0-2.1 s on follow-ups. Total wall time is lower (91 s vs 98 s).
  - `strata_bench.py` at 16 clients is -3 %, inside the overlap of the run ranges.
- **No end-to-end speed claim at 16 users:** over two interleaved pairs, PT vs C is -2.2 % (short) and +1.1 % (long). The gains are in per-user decode at 1-4 users, in single-request decode, and in stream continuity during a join.
- **The large S-vs-PT ratios at 16 users come from configuration:** 16 slots, prefill 512 and the caches. S queues 16 clients on 3 slots. These ratios do not measure the engine.
- **Memory:** PT leaves less VRAM headroom: the lowest free VRAM was 810 MiB on card 3, against a minimum of 1,372 MiB for C. Peak host memory of the server's systemd scope (`memory.peak`, recorded by our window script, not in `data/`) is 90-101 GiB, against 82-93 GiB for C.
- **One machine, one model and quant, one day.** The protocol is still a draft (`draft.1`). The model-card, tokenizer, source-model and quant-repo revisions are `null`, each with a reason.
- **Prompt reuse:** the conversation and prompt caches are on in C and PT. In the standard runs the runner flagged no unexpected reuse, so it excluded no request.
