# Community benchmark on 4x RTX 3090 (layer split): stock 0.1.44 against 0.1.43 with a patch series, 1 to 16 users

Measured on 2026-10-14 by **havok2-htwo**.

**What was tested:** concurrent serving through each build's own `serve/server.py`, on one machine, on the same
day, with the same model files, engine arguments and requests. The two builds are:

- **stock**: v0.1.44 (`ed4773eb`), built from source.
- **ours**: v0.1.43 plus our patch series, which is not upstream. This is the build we run in production.

The workloads were 1, 8 and 16 users with short prompts, 16 users with 1.2K-token prompts, and 16 agents
running 3 turns each. All tests were text only.

**Main limitation:** stock ran 8 batch slots and ours ran 16. 8 is the most 0.1.44 runs on this layer split with
its default pipelined groups (without `--batch-mtp`). With `--batch-mtp` and serial windows (`--batch-groups 1` or
`STRATA_BATCH_MTP_SERIAL=1`) it accepts more slots; we did not test that. The 16-user rows therefore measure two
things together: twice the slots, and everything else that differs between the builds. The 8-user row is the
closer comparison. Besides the patches and their opt-in switches (listed below), the builds differ in their base
version (0.1.43 against 0.1.44) and in one build option, so no row isolates one change. **Runs: 1 per arm.**

| Workload (aggregate output tok/s) | ours (0.1.43 + patches, 16 slots) | stock 0.1.44 (8 slots) | ours / stock |
| --- | ---: | ---: | ---: |
| 1 user, ~147-token prompt, 512 tokens out | 116.2 | 111.7 | 1.04 |
| 8 users, same | 399.4 | 314.4 | 1.27 |
| 16 users, same | 516.7 | 292.9 | 1.76 |
| 16 users, ~1,213-token prompt, 512 tokens out | 406.5 | 238.1 | 1.71 |
| 16 agents x 3 turns (11K-token start, temperature 0.7) | 151.9 | 92.0 | 1.65 |

At temperature 0, the greedy replies were **byte-identical between the two builds for all 82 requests** of the
sweep. This is checked by the SHA-256 of each reply, request by request.

## Hardware and software

- **GPUs:** 4x NVIDIA GeForce RTX 3090 24 GB (sm_86). Power limit 380 W each. PCIe links as reported by
  `nvidia-smi`: GPU0 and GPU1 Gen3 x16, GPU2 and GPU3 Gen4 x8. The engine's own probe measured 13.0 to 13.5 GB/s
  host to device on every card. No P2P/NVLink is used.
- **CPU:** AMD Ryzen Threadripper 3990X, 64 cores / 128 threads, AVX2, no AVX-512.
- **RAM:** 251 GiB (as `free -g` reports it). Each bench server ran in a systemd scope with `MemoryMax=224G`.
- **Storage:** 3x Crucial P310 1 TB NVMe in md RAID0 (ext4). It holds the models, packs and data.
- **OS:** Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic, NVIDIA driver 580.178.04. Both builds used CUDA 13.0
  (nvcc V13.0.88) and GCC 13.3.0. The HTTP servers ran on Python 3.12.3, with the same venv for both arms.
- **stock build:** `Niko1221/Strata` v0.1.44 (`ed4773eb`), built with the CMake options `setup.py` uses for a
  CUDA engine: `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=86`. The ggml source is
  the pinned llama.cpp `3cf03257`. Engine SHA-256 `4e6bdc675c41474245207827b9381c77908aa72d3f37e809e55f55bab2e46925`.
  The server is v0.1.44's own `serve/server.py`, and the engine reports `engine 0.1.44`.
  v0.1.44.1 (`296b59d2`, committed at 01:51 CEST on the day of the run) was not measured. It changes
  `serve/server.py`, `setup.py`, `src/qwen35/q36_engine.cu` (host-side F16 widening for older CUDA toolkits), the
  CPU NVFP4 files and Metal.
- **ours:** v0.1.43 (`7695859a`) plus 53 commits, ending at `08830ddf`. The full source tree is in our fork,
  pinned to that commit:
  [havok2-htwo/Strata at `08830ddf`](https://github.com/havok2-htwo/Strata/tree/08830ddf94213866c29dcd8ae4b70f0674670bed)
  (branch `bench-source-08830ddf`). The same commits as a patch series:
  - 0001-0052 in
    [havok2-htwo/StrataQ38FN `linux/patches/v0.1.43/` at `4ece0ca`](https://github.com/havok2-htwo/StrataQ38FN/tree/4ece0ca829c17853b84742c31948c12b8c66c478/linux/patches/v0.1.43)
  - the last commit, `08830ddf` (`STRATA_ADMIT_SHORT_FIRST`, `serve/server.py` and a new test file only, see
    below), which is not in that patch folder; it is on the fork branch above.

  All engine switches this run used come from 0001-0039.

  The engine was built with `-DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_MMQ_TAB=ON` and the same ggml source.
  Engine SHA-256 `39f0adea03f7ccfebabae9d1c3ddb001d84d62add14208cfa40ce833f79b8da5`. It reports `engine 0.1.43`.
  The binary contains the `STRATA_MROPE_SLOTS` switch of `d6d896fb` (0050), and the commits from there to
  `08830ddf` change only `serve/`, `tools/` and `docs/`, so its engine sources are those of `08830ddf`. The server
  is that checkout's `serve/server.py`.
- **Background:** this machine is our production server. The production service was stopped for the whole run
  (04:12:10 to 04:25:06 CEST). The bench servers listened on 127.0.0.1 only. The GPUs show 0 to 5 % utilisation in
  the samples taken before the first request of each phase. We did not otherwise audit background processes.

## Model and configuration

- **Model:** Unsloth `unsloth/Qwen3.8-Flash-Next-GGUF`, UD-IQ4_XS, three shards
  `Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf`. Their SHA-256 values were checked at download and match
  the pins in v0.1.44's `setup.py` (revision `38bb39ee`):
  - `5ce89370...e71a4`
  - `577a38a2...aa7`
  - `d4634e6d...8833`
- **Pack and draft:** the native-experts pack (`unsloth-ud-iq4_xs`, `--compat-bf16`) and the MTP draft head
  (`mtp/rt`) were both made by `setup.py` on v0.1.41 (`--family unsloth --model UD-IQ4_XS`). Both arms used them
  unchanged.
  - `index.txt` SHA-256 `e61c8352...364d`
  - `native_experts.txt` `f9a8bdec...68db`
  - `draft_vocab.bin` `b1e1d3a7...cb91`
- **Expert profile:** `data/expert-profile.bin`, identical to the one shipped in v0.1.44 (SHA-256 `8f59b4aa...af`).
- **Vision:** off in both arms (no `--vision`). Our build's vision patches (0042, 0043, 0049, 0050) are compiled in
  but inactive.
- **Context and memory:** context 262,144; `--kv int8`; `--kv-resident 32768` (K/V streamed, 32K cells per QSA
  layer in VRAM); `--expert-cache auto`; `--resident-experts`. Both engines log `100% of the experts resident`, so
  no expert is read from RAM or disk while serving. The settings are `--prefill 512`, `--trim-stage-weights`,
  `--ple-io ram` and `--aux-cpus 65-127`. Both arms enabled the conversation cache: 96 GiB, 320 slots,
  16 GiB minimum free, 1,024 minimum tokens.
- **MTP:** `--spec 4 --spec-min-p 0.5 --mtp <draft>`, without `--batch-mtp`. Reasoning was off in every request
  (`reasoning_effort: none`). Calibration and speed projection were not used.
- **Layer split** `12,25,37` on GPUs 0-3, with batch groups left at the default (auto).

Both arms use the same engine arguments. The engine logged them like this (paths replaced):

```text
<engine> --serve --pack <work>/Strata-data/packs/unsloth-ud-iq4_xs \
  --native <models>/Strata-GGUF/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --expert-profile <work>/Strata/data/expert-profile.bin --expert-cache auto --prefill 512 \
  --spec 4 --spec-min-p 0.5 --mtp <work>/Strata-data/mtp/rt --max-context 262144 --kv int8 \
  --resident-experts --trim-stage-weights --ple-io ram --aux-cpus 65-127 \
  --conversation-cache-mib 98304 --conversation-cache-slots 320 --kv-resident 32768 \
  --conversation-cache-min-free-mib 16384 --conversation-cache-min-tokens 1024 \
  --layer-split 12,25,37 --batch 16        # stock: --batch 8
```

The full server configs are in [`configs/`](configs/).

| | stock 0.1.44 | ours |
| --- | --- | --- |
| `"parallel"` (batch slots) | 8, the most 0.1.44 runs here with its default pipelined groups (with more than 8 and no `--batch-mtp`, `generate.cpp` caps `--batch` at the 8 rows of a window; with `--batch-mtp` in serial windows it accepts more, not tested) | 16 (patches 0001, 0018: more than 8 slots on a layer split, 8 rows per pipeline group) |
| batch groups (auto) | 4 groups of 2 slots | 4 groups of 4 slots |
| environment, upstream switches | `STRATA_SPLIT_MTP_BATCH=1 STRATA_NAN_GUARD=0 STRATA_KV_PREFETCH=1` | the same |
| environment, glibc | `MALLOC_MMAP_THRESHOLD_=16777216 MALLOC_TRIM_THRESHOLD_=67108864` | the same |
| environment, our opt-in switches | none (they do not exist in 0.1.44) | see the list below |

Our opt-in switches, with the patches that add them:

- `STRATA_MMQ_DIRECT=1` (0003-0011): the prompt MMQ reads resident experts in their cache slots. This needs
  `-DSTRATA_MMQ_TAB=ON`.
- `STRATA_MMQ_GROUP_SORT=1` (0029, 0031)
- `STRATA_SLOT_COPY_DIRECT=1` (0002): the device-to-device slot copy.
- `STRATA_SLOT_COPY_BATCHSYNC=1` (0027)
- `STRATA_CKPT_POOL=1 STRATA_CKPT_THREADS=1` (0025)
- `STRATA_SLOT_CKPT_MOVE=1` (0026)
- `STRATA_PIPE_BYIELD_NODRAIN=1` (0023)
- `STRATA_ADMIT_MERGE_FIRST=1` (0024)
- `STRATA_BATCH_SELF_COMMIT=1` (0030)
- `STRATA_QSA_SLOTS=1` (0034)
- `STRATA_HANDOFF_PIPE=8` (0036)
- `STRATA_PARK_ASYNC=1` (0037)
- `STRATA_SLOT_PARK=1` (0038)
- `STRATA_CONV_TIER_DIR=<data>/strata-kvspill-bench STRATA_CONV_TIER_WARM=1 STRATA_CONV_TIER_MIN_TOKENS=8192
  STRATA_CONV_TIER_WRITE_GB_DAY=300` (0039): conversations evicted from RAM spill to disk. The log shows no
  spill in this run.
- `STRATA_ADMIT_SHORT_FIRST=8192` (`08830ddf`, the last commit, not in the patch folder): prompts of up to 8,192
  tokens are admitted before longer ones. Every sweep prompt is shorter than that, so this setting changes nothing
  in the sweep. It can reorder admissions in the multi-turn run.

## Method

The run order was ours first, then stock, with one engine start per arm. Each engine start served the following
phases in order:

1. **short**: one warm-up request (not in the tables), then 1, 8 and 16 users. Each user sends 2 requests back to
   back (closed loop). Each prompt is about 147 tokens: a unique header, a filler document and a "write a long
   explanation" task. `max_tokens` is 512, at temperature 0, with streaming.
2. **long**: 16 users, 2 requests each, prompts of about 1,213 tokens, 512 tokens out, temperature 0, no warm-up.
3. **multi-turn**: 16 agents, 3 turns each, run concurrently. Turn 1 is an 11,131-token prompt, mostly a
   document. Each later turn resends the full history, previous replies included, plus about 4,100 tokens of new
   material. Each turn allows up to 400 tokens out, at temperature 0.7, with a 1 s pause between turns. The wall
   time includes the pauses. The run was capped at 240 s and finished well within that limit (87 s and 146 s).

Each prompt depends only on the request id and the prompt kind, so both arms sent byte-identical sweep requests.
The prompts are synthetic, and [`scripts/`](scripts/) builds them. Every sweep request generated exactly 512
tokens: no request stopped early and none failed (`ok` = `requests` in every row).

The two kinds of figures come from different places:

- **Client figures** come from `scripts/bench_concurrency.py` and `scripts/bench_multiturn.py`, using streamed
  chunks:
  - TTFT runs from sending the request to its first non-empty content chunk.
  - Per-request decode = (tokens - 1) / (end - first token).
  - e2e is the whole request.
  - Aggregate = all completion tokens / wall time of the level.
  - "Steady" counts output only over the window where every user has its first token, up to the moment the first
    user finishes. It is undefined (`–`) when that window is empty, which is what happens at 16 users on 8 slots:
    users who wait for a slot get their first token only after others are done.
- **Engine figures** come from the engine's own log lines, summarized by `scripts/engine_log_stats.py`:
  - prompt read time and rate per request
  - solo decode and MTP drafts
  - admission copy time into a slot
  - rows per second of the pipelined batch, which includes admissions

GPU clocks, power, temperature and utilisation were sampled every 1 s by `nvidia-smi` during each sweep phase.

The model was loaded before timing started, and loading is not part of any figure. Both engines prefilled the
expert cache from the same profile at start. The page cache held the GGUF in both arms: the engine logs
`the GGUF experts are 100% in the page cache` at start. The file-tier counter of the first (warm-up) request still
shows 43,074 MB read from storage in our arm (15,516 major faults), against 65 MB in the stock arm, which started
second. Every later request in both arms shows 0 to 0.1 MB. The warm-up request is not in the tables.

The commands were `scripts/community_bench_144.sh`, which calls `scripts/run_strata_variant.sh` and then
`scripts/bench_multiturn.py`:

```text
run_strata_variant.sh cb144-<arm>-short <config> 1,8,16 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0
ATTACH=1 run_strata_variant.sh cb144-<arm>-long <config> 16 --rounds 2 --max-tok 512 --temperature 0 --warmup 0
bench_multiturn.py --url http://127.0.0.1:8095 --label cb144-<arm>-mt --users 16 --turns 3
compare_replies.py sweep.requests.jsonl cb144-ours-short cb144-stock-short    # and -long
```

## Results

Runs: 1 per arm. In earlier runs of our build, the run-to-run standard deviation at 16 users was about 0.2-0.4 %
(up to 0.8 % for one older config), measured on other days. We have no repeat of the stock arm.

### Short prompts (~147 tokens in, 512 out, greedy)

| Users | Build | Wall s | Agg out tok/s | Steady tok/s | Decode p50 per request tok/s | TTFT p50 s | TTFT p95 s | e2e p50 s | e2e p95 s | Engine batch rows/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ours | 8.8 | **116.2** | 118.9 | 121.5 | 0.20 | 0.20 | 4.4 | 4.4 | (solo path) |
| 1 | stock | 9.2 | 111.7 | 116.2 | 120.8 | 0.35 | 0.35 | 4.6 | 4.6 | (solo path) |
| 8 | ours | 20.5 | **399.4** | 421.1 | 54.5 | 0.65 | 1.66 | 10.2 | 10.9 | 408.6 |
| 8 | stock | 26.1 | 314.4 | 343.3 | 46.0 | 1.68 | 3.60 | 13.0 | 14.2 | 320.3 |
| 16 | ours | 31.7 | **516.7** | 548.9 | 34.4 | 0.89 | 3.10 | 15.8 | 17.9 | 523.1 |
| 16 | stock | 55.9 | 292.9 | – | 45.9 | 8.69 | 29.47 | 18.9 | 39.6 | 315.5 |

### 1.2K-token prompts (~1,213 tokens in, 512 out, greedy), 16 users

| Build | Wall s | Agg out tok/s | Steady tok/s | Decode p50 per request tok/s | TTFT p50 s | TTFT p95 s | e2e p50 s | e2e p95 s | Engine batch rows/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ours | 40.3 | **406.5** | 454.3 | 28.3 | 1.91 | 6.61 | 20.0 | 24.8 | 418.9 |
| stock | 68.8 | 238.1 | – | 38.7 | 6.96 | 39.91 | 18.9 | 52.1 | 242.4 |

At 16 users, stock decodes each request faster (45.9 against 34.4 tok/s, and 38.7 against 28.3). Only 8 requests
share the GPUs at a time there, and the other 8 users wait for a slot, which shows up as TTFT. Rows per
group-step in the engine's batch summary are 1.94-2.00 for stock and 3.98-3.99 for ours.

**8 users:** both builds ran 4 pipelined groups carrying 2.00 rows per group-step:

- ours: 8,175 rows in 4,096 group-steps, 4.88 ms per group-step
- stock: 8,175 rows in 4,094 group-steps, 6.23 ms per group-step

Admissions are included in both. So at 8 users the extra slots of our build are not used. The difference comes
from the other differences between the builds: the patch series and its opt-in switches, the
`-DSTRATA_MMQ_TAB=ON` build option, and the base version (0.1.43 against 0.1.44, engine and server). This run does
not separate them, and the base-version changes could move the 8-user figure either way.

### Multi-turn: 16 agents x 3 turns (temperature 0.7)

| Build | Wall s | Output tokens | Agg out tok/s | Turn 1 TTFT p50 / max s | Turn 2 TTFT p50 / max s | Turn 3 TTFT p50 / max s | Follow-up TTFT p50 s | Decode p50 per turn tok/s |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| ours | 86.9 | 13,202 | **151.9** | 15.25 / 28.80 | 1.82 / 5.38 | 0.98 / 3.61 | **1.52** | 10.8 / 16.2 / 22.2 |
| stock | 146.3 | 13,453 | 92.0 | 28.02 / 82.98 | 14.43 / 51.40 | 4.68 / 76.65 | 9.91 | 12.0 / 14.1 / 16.5 |

Mean prompt and cached tokens per request, ours / stock:

- turn 1: 11,131 / 11,131 prompt tokens, 0 cached in both
- turn 2: 15,589 / 15,587 prompt tokens, 11,440 / 11,131 cached
- turn 3: 20,017 / 20,019 prompt tokens, 15,863 / 15,608 cached

`bench_multiturn.py` reports the per-turn p95 as the element at index int(0.95 n). With 16 requests per turn,
that is the slowest request, so the table says max. At temperature 0.7 the replies differ between the arms, so
later turns resend slightly different histories: this row compares like-sized work, not identical work.

### Engine timings (from the engine logs)

| | ours | stock |
| --- | --- | --- |
| Solo decode, the 2 requests of the 1-user level | 121.2, 121.6 tok/s | 120.7, 120.8 tok/s |
| MTP drafts accepted in those 2 requests | 586 of 948 (61.8 %) | 586 of 948 (61.8 %) |
| Prompt read, ~147 tokens, 1-user level | 183, 183 ms | 337, 337 ms |
| 1.2K prompt read, first of the level (nothing else decoding) | 1,211 tokens in 415 ms (2,917 tok/s) | 1,212 tokens in 708 ms (1,712 tok/s) |
| 1.2K prompt reads, all 32, median (min-max) | 386 ms (376-415), 3,148 tok/s | 831 ms (708-844), 1,458 tok/s |
| 11K first-turn reads (multi-turn), median of 16 | 1,701 ms, 6,498 tok/s | 2,881 ms, 3,836 tok/s |
| Follow-up reads (multi-turn), median of 32 | 4,110 new tokens in 828 ms | 4,396 new tokens in 1,657 ms |
| Admission copy into a slot, short phase, median (min-max) | 1.0 ms (0.9-63.2) | 153.3 ms (142.2-155.2) |
| Same, long phase | 3.5 ms (2.2-9.6) | 155.6 ms (154.7-157.5) |
| Same, multi-turn | 13.2 ms (9.3-71.3) | 197.4 ms (177.7-212.4) |

The admission copy is the engine's `strata batch: slot N takes M tokens (copied in X ms)` line. In our arm, four
switches act on it together: `STRATA_SLOT_COPY_DIRECT`, `STRATA_SLOT_COPY_BATCHSYNC`, `STRATA_SLOT_CKPT_MOVE`
and `STRATA_CKPT_POOL`. The figure is not a measurement of one patch.

### GPUs and memory

Per-GPU means over each sweep phase (1 s samples; [`bench-output.txt`](bench-output.txt), [`gpu/`](gpu/)):

| Phase | Build | GPU0 | GPU1 | GPU2 | GPU3 |
| --- | --- | --- | --- | --- | --- |
| short | ours | 1858 MHz, 290 W, 70 % | 1872 MHz, 286 W, 58 % | 1897 MHz, 283 W, 56 % | 1842 MHz, 290 W, 64 % |
| short | stock | 1856 MHz, 274 W, 59 % | 1870 MHz, 274 W, 58 % | 1891 MHz, 265 W, 48 % | 1825 MHz, 276 W, 63 % |
| long | ours | 1868 MHz, 318 W, 74 % | 1883 MHz, 321 W, 72 % | 1924 MHz, 316 W, 62 % | 1870 MHz, 314 W, 69 % |
| long | stock | 1890 MHz, 291 W, 69 % | 1912 MHz, 295 W, 55 % | 1949 MHz, 289 W, 45 % | 1874 MHz, 292 W, 62 % |

Maximum temperatures were 67-78 °C in both arms. The mean SM clocks of the two arms are within about 2 %, so the
order (ours first) did not leave stock on slower clocks.

VRAM used on the four cards together, from the server's `/v1/status` right after load (uptime 3 s ours, 5 s
stock): ours 91,381 MiB, stock 81,568 MiB, both out of 98,304. The difference is about the 8 extra slot sessions at
0.29 GiB each per card. The expert caches are the same size in both arms (6,144, 6,656, 6,144 and 5,632 slots on
CUDA0-3), and all experts are resident in both.

System RAM in use, from the same status (`machine.ram.used_gib`, a whole-machine figure, out of 251.5 GiB): right
after load, ours 96.2 GiB and stock 70.4 GiB; at the start of the long phase, 98.5 and 71.5 GiB. Our build used
about 26 GiB more. We did not break that difference down. The status files are in [`runs/`](runs/).

## Correctness and limitations

- **Identical output:** `scripts/compare_replies.py sweep.requests.jsonl cb144-ours-short cb144-stock-short`
  reports 50 of 50 identical (1, 8 and 16 users). The same check for `-long` reports 32 of 32. Each level's
  `replies_sha256` digest in `sweep.jsonl` is also equal across the two arms. The multi-turn run was sampled, so
  it has no identity check.
- **Not separated:** this run does not attribute the difference to individual patches, and the 16-user rows
  include 16 against 8 slots. We intend to offer two parts of the series upstream as separate pull requests, each
  with its own same-day A/B on default settings:
  - more than 8 slots on a layer split (0001, 0018)
  - the device-to-device slot copy (0002, with 0027)

  Nothing in this report measures either of them alone.
- **One run per arm**, in one order (ours first). There was no warm-up before the long phase or the multi-turn run.
- **Untested:**
  - images
  - prompts above 20K tokens
  - more than 16 users
  - `--batch-mtp` in either build, including stock with more than 8 slots in serial windows (`--batch-groups 1`
    or `STRATA_BATCH_MTP_SERIAL=1`)
  - other splits
  - stock's default settings for the switches both arms set (`STRATA_NAN_GUARD=0`, `STRATA_SPLIT_MTP_BATCH=1`,
    `STRATA_KV_PREFETCH=1`, `--aux-cpus`, `--prefill 512`)
  - v0.1.44.1 (`296b59d2`), and any build other than these two
- **Issue [#2038](https://github.com/Niko1221/Strata/issues/2038)**, about counting zero-proposal requests in the
  engine log, as done in
  [this comment](https://github.com/Niko1221/Strata/issues/2038#issuecomment-6148276416) (86 % of requests with
  zero proposals on 0.1.43.1 against 11 % on 0.1.41, on a server with `--batch 8 --batch-groups 4`). Observed on
  4 stages with short prompts only:
  - Every request that ran on the solo path drafted with MTP, in both builds. Ours had 3 solo requests in the
    short phase (the warm-up and the 2 of the 1-user level) and 1 in the multi-turn run. Stock had 4 in the short
    phase, including one handed back from a slot when the 16-user level drained, and 1 in the multi-turn run.
  - Solo decode in the 1-user level (~147-token prompts) was 120.7-121.6 tok/s in both builds. At a 20K-token
    conversation (the last multi-turn request) it was 118.2 (ours) and 117.9 tok/s (stock).
  - Every `strata serve: prompt` line of a request admitted into a batch slot (132 such lines per arm) logs
    `1 generated ... drafts accepted 0 of 0`, in both builds. That line covers only the first token, which is read
    on the solo path before the slot takes over. The slot's later tokens appear only in the
    `strata batch (pipelined ...)` summary, and pipelined slots do not draft (`docs/BATCHING.md`). So lines with
    1 generated token and 0 of 0 drafts are batch-slot admissions. No line with more than 1 generated token showed
    0 of 0 in this run. Filtering on generated > 1 before counting zero-proposal requests leaves only requests
    that decoded on the solo path.
  - We did not run 0.1.41 or a 200K prompt here, so this says nothing about the solo-decode regression reported
    in that issue.

## Files

All paths are sanitized: `<work>` is the working folder, `<models>` the model folder and `<data>` the data disk.
Times are in seconds unless a field says `_ms`.

- `sweep.jsonl`: one row per level and arm, from `bench_concurrency.py`. The fields are:
  - `users`, `rounds` (requests per user), `requests`, `ok`, `errors`
  - `wall_s`, `max_tok`, `temperature`, `prompt_kind` (100 = short, 1000 = 1.2K), `prompt_tok_avg`
  - `out_tok_total`, `agg_out_tps` (completion tokens / wall), `agg_out_tps_steady` and `steady_s` (see Method;
    null when undefined), `agg_total_tps` ((prompt + completion) / wall)
  - `ttft_p50/p95/max`, `dec_tps_p50/p5/p95` (per-request decode, tok/s), `e2e_p50/p95`
  - `cached_tok`
  - `replies_sha256`: one digest over all (uid, reply SHA-256) pairs of the level
- `sweep.requests.jsonl`: one line per level and arm. `reqs` holds one array per request with `uid`, `user`,
  `max_tok`, `gen` (completion tokens), `prompt` (prompt tokens), `t0`, `t_first`, `t_end` (seconds from the
  level's start), `sha256` (of the reply text) and `err`.
- `multiturn.jsonl`: one row per arm with `wall_s` and `turn1..3`. Each turn has the mean `prompt` and `cached`
  tokens, `ttft_p50`, `ttft_p95` (index int(0.95 n)) and `dec_p50` in tok/s. The row also has `out_tok`,
  `agg_out_tps`, `prompt_tok` (all prompt tokens sent), `follow_ttft_p50` (turns 2 and 3) and `replies`
  ([uid, turn, tokens, SHA-256]).
- `configs/`: the two server configs as used (`exe`, `args`, `env`, `parallel`, `layer_split`, `gpu`).
- `runs/engine-<arm>.txt`: engine log excerpts.
  - Kept: start-up lines (without the per-window `strata verify: capturing/captured` lines), then per request
    `strata serve: prompt ...` (prompt tokens, reused, read, read ms, generated, decode ms, drafts accepted of
    proposed), `strata batch: slot ... (copied in X ms)`, the `strata batch (pipelined ...)` summaries,
    conversation-cache park/restore and file-tier I/O.
  - Dropped: the `strata KV prefetch` lines.
  - `# ==== phase` markers separate start-up, short, long and multi-turn.
- `runs/run-<arm>-<phase>.txt`: the run line with the engine hash and server commit, the client table and the
  GPU means.
- `runs/status-<arm>-<phase>.json`: the server's `/v1/status` at the start of each phase (uptime 3 s and 76 s for
  ours, 5 s and 107 s for stock).
- `bench-output.txt`: the console output of the whole run.
- `gpu/gpu-<arm>-<phase>.csv`: `nvidia-smi` every 1 s, with timestamp, index, SM clock (MHz), power (W),
  temperature (°C) and utilisation (%).
- `scripts/`: `bench_concurrency.py`, `bench_multiturn.py`, `compare_replies.py`, `community_bench_144.sh`,
  `run_strata_variant.sh`, and `engine_log_stats.py`, which prints the engine-timing figures above from
  `runs/engine-*.txt`. The two shell scripts differ from the copies that ran in four ways, so that they run on
  another machine:
  - the folders are environment variables (`STRATA`, `OURS_DIR`, `STOCK_DIR`) instead of our paths
  - the config names match `configs/`
  - the listen address defaults to 127.0.0.1 (the run set `BIND=127.0.0.1`)
  - stopping and restarting a production service happens only when `PROD_UNIT` names one (the run stopped ours)

  The configs still hold our paths as `<work>`, `<models>` and `<data>`. Edit them for your machine.
