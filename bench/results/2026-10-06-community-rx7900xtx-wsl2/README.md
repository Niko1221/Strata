# Community benchmark on RX 7900 XTX (gfx1100, Windows 11 + WSL2): Strata 0.1.33 vs 0.1.40, Coder IQ1_M

Measured on 2026-10-06 by theGiallo. One consumer RDNA3 card on a PC with a 24-thread CPU and 54 GB of RAM visible to WSL2, serving the two-shard Qwen3.8-Flash-Next GSQ-RCO IQ1_M GGUF with MTP through `serve.server`. The main finding is a **prefill regression on this GPU between 0.1.33 and 0.1.40** (about 6.5-7x slower cold prompts at the defaults) and a one-variable workaround that restores and exceeds the 0.1.33 speed: `STRATA_PREFILL_STREAM_MIN=65536`. Limitation: one machine, one model, one pack; the 10k-token prompt is a neutral documentation text (Strata's own `docs/`), the long needle prompts are `tools/needle_bench.py`'s. A second session on 2026-10-07 (BIOS memory profile enabled, DDR4-3200) repeated the community measurements and added a **prompt-length sweep from 16k to 256k tokens**: at long context the 0.1.33 and 0.1.40 defaults both prefill at only about 85-105 tok/s (a 256k prompt takes 45-47 minutes to the first token), while `STRATA_PREFILL_STREAM_MIN=65536` holds about 500 tok/s up to 255k tokens (8.5 minutes).

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GB (gfx1100). Core and VRAM clocks capped at 2850 MHz by a driver setting (the driver reset under load without the cap). CPU: AMD Ryzen 9 5900X, 12 cores / 24 threads. Motherboard: MSI MAG B550 TOMAHAWK (MS-7C91), AMD B550 chipset, BIOS A.K1 (2025-09-09). Installed RAM: 2 x 32 GB Kingston KHX3200C16D4/32GX (a DDR4-3200 CL16 kit) running at **DDR4-2400** (CPU-Z: DRAM frequency 1187 MHz, CL17-17-17-39, command rate 2T, 1.2 V), not at the kit's rated XMP-3200 (CL16, 1.35 V), which was not enabled in the BIOS for the measurements of the first tables (the second session, from 2026-10-07, ran with it enabled: CPU-Z DRAM frequency 1582 MHz), 64 GB in total, 54 GB visible to the WSL2 VM. Storage: Windows `C:` (NTFS) and the WSL2 virtual disk that holds the models are on one Kingston SFYRD2000G 2 TB NVMe SSD (the distribution's VHDX sits under the user's AppData/Local/wsl folder on `C:`); the model files are read through the WSL ext4 disk and the VM's page cache. PCIe link of the GPU and of the SSD: not measured.
- OS: Windows 11 Pro 10.0.26200 host, Ubuntu 26.04 LTS in WSL2 (kernel 6.18.33.2-microsoft-standard-WSL2). Driver: AMD Radeon Software driver 32.0.31035.1003 (Windows). ROCm 7.14 (pip `_rocm_sdk_devel` toolchain, built with `amdclang++`), hipBLASLt 1.4.1 (100401).
- Strata: commit `aeb35be` (engine 0.1.33, source build) and commit `1735d64` (tag v0.1.40, source build), both `-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DSTRATA_NATIVE_EXPERTS=ON`, the pinned llama.cpp `3cf0325`.
- Background workloads: none running during the measurements; the page cache held the whole model (50 GB buff/cache, see Method). Power limit: none besides the clock cap.

## Model and configuration

- Model: Qwen3.8-Flash-Next (125B MoE), GSQ-RCO IQ1_M GGUF, `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` (29.6 GB) and `-00002-of-00002.gguf` (28.8 GB, the PLE table); Coder pack from `tools/iq_pack.py` (`--native` shard 1, no `experts.bin`), `data/expert-profile-coder.bin`.
- MTP: the 31 `mtp.*` tensors fetched with `tools/mtp_fetch.py` from Qwen/Qwen3.8-Flash-Next (SHA256 verified), packed with `tools/mtp_pack.py --experts q2_0` and `tools/mtp_rt.py`. No vision encoder. No hipBLASLt tuning table (the engine was launched by hand; a table calibrated for hipBLASLt 1.4.1 gave no change on 0.1.33, +-5% on prefill).
- Context 131072 for the runs below (32768 for the environment A/B), KV int8, expert cache auto (8277-8300 experts, 15.8 GiB of VRAM), `--prefill auto` (8192-token chunks), `--spec 4 --spec-min-p 0.5`, `--mmap-experts`, greedy (temperature 0), reasoning content counted as generated tokens.

```text
build-hip/strata via serve.server, args:
--pack coder-IQ1_M --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf
--mmap-experts --expert-profile data/expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp mtp-rt --max-context 131072 --kv int8
(the third configuration adds only the environment variable STRATA_PREFILL_STREAM_MIN=65536)
```

## Method

`scripts/strata_start.sh` starts the server, `scripts/strata_community.py` sends 3 iterations of three streamed chat requests (temperature 0, `max_tokens` 256): a **short** prompt (a coding question, 76 tokens), a **long** prompt (the first 30,000 characters of Strata's `docs/` plus a question, ~9,960 tokens) and a **follow-up** (the same conversation plus a second short question, which reuses the long prefix from the conversation cache). A random nonce opens each cold prompt so no earlier prefix can be reused; the numbers of reused tokens come from the `cached_tokens` field of the usage block (about 9,958 of 10,242 on the follow-ups). One server per configuration run, no restart between iterations, so iteration 1 includes the expert-cache warm-up (visible in the short prompt: 10-18 tok/s then 28-35). The three 0.1.40 default runs and two 0.1.33 runs are separate server starts, interleaved (0.1.40, 0.1.33, 0.1.40) to rule out order effects; earlier runs on the same machine suggest the **OS page cache** matters too: the first prefills after a WSL restart (30 GB expert shard not yet cached) were 70-110 tok/s on 0.1.33, most likely because of cold reads, which was not isolated. The numbers reported here are warm-cache (about 50 GB of file cache in WSL for all reported runs).
Timing boundaries (client side, streaming): **TTFT** = request sent to the first non-empty content or reasoning delta of the stream (keep-alives and empty deltas ignored; which kind came first was not recorded); **prompt tok/s** = (prompt tokens - reused tokens) / TTFT; **decode tok/s** = completion tokens / (end - first token), not total time; total latency is in `runs.csv`. Memory: peak resident size of the engine process sampled every 2 s (`rss.csv`; the peak of about 26.9 GB is the mmap'd experts, not an allocation) and the engine's own VRAM line (`vram_line.txt`: "filling the GPU's expert cache (N experts, 15.8 GiB of VRAM)"; about 0.5 GiB of VRAM stays free after load). No paging or out-of-memory event occurred; no request failed or was cancelled.

## Results

Prompt = freshly processed tokens per second (TTFT-based), decode = generated tokens per second, medians over all iterations of all server runs of the configuration, ranges in brackets. Per-run values: `runs/<config>/runs.csv` and `runs.jsonl`.

| Configuration | Request | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0.1.33 (aeb35be), defaults | short, cold | 76 | 0 | 218 | 6 | not measured | 27.8 (18.1-31.2) | 2.2 (1.9-15.2) |
| 0.1.33 (aeb35be), defaults | long, cold | 9963 | 0 | 256 | 6 | 442.9 (375.8-473.4) | 39.1 (36.2-44.3) | 22.5 (21.0-26.5) |
| 0.1.33 (aeb35be), defaults | follow-up | 10242 | 9958 | 256 | 6 | 94.7 (79.6-96.1) | 51.8 (41.8-57.4) | 3.0 (3.0-3.6) |
| 0.1.40 (1735d64), defaults | short, cold | 76 | 0 | 242 | 9 | not measured | 29.0 (10.2-32.4) | 2.7 (2.2-14.3) |
| 0.1.40 (1735d64), defaults | long, cold | 9963 | 0 | 256 | 9 | 66.7 (61.4-68.5) | 40.5 (27.7-47.3) | 149.4 (145.4-162.3) |
| 0.1.40 (1735d64), defaults | follow-up | 10242 | 9958 | 256 | 9 | 103.5 (53.2-108.8) | 55.3 (46.2-59.7) | 2.7 (2.6-5.3) |
| 0.1.40, STRATA_PREFILL_STREAM_MIN=65536 | short, cold | 76 | 0 | 220 | 3 | not measured | 34.0 (28.4-35.3) | 2.1 (2.1-2.2) |
| 0.1.40, STRATA_PREFILL_STREAM_MIN=65536 | long, cold | 9963 | 0 | 256 | 3 | 455.9 (436.7-456.5) | 45.9 (41.4-47.9) | 21.9 (21.8-22.8) |
| 0.1.40, STRATA_PREFILL_STREAM_MIN=65536 | follow-up | 10242 | 9958 | 256 | 3 | 111.5 (79.3-116.3) | 61.3 (53.2-65.5) | 2.5 (2.4-3.6) |

Needle in a haystack (`tools/needle_bench.py --lengths 32k,128k --depths 50`, same server settings, `-c 131072`): all found in all three configurations.

| Configuration | 32k depth 50% | 128k depth 50% |
| --- | --- | --- |
| 0.1.33, defaults | found, 31,075 tokens in 173 s (180 tok/s) | found, 122,150 tokens in 979 s (125 tok/s) |
| 0.1.40, defaults | found, 32,308 tokens in 287 s (113 tok/s) | found, 125,704 tokens in 993 s (127 tok/s) |
| 0.1.40, `STRATA_PREFILL_STREAM_MIN=65536` | found, 32,308 tokens in 58 s (557 tok/s) | found, 125,704 tokens in 194 s (648 tok/s) |

### Where the 0.1.40 prefill regression comes from

0.1.40 logs `prompt chunk auto: 8192 tokens, a 96-slot ring`, 0.1.33 does not: since 0.1.30 (`src/prefill/prefill.cpp`, `stream_all_min()`, default 1024) every non-resident expert of every layer streams through a ring once the prompt chunk is at least 1024 tokens. Single-iteration A/B on 0.1.40, `-c 32768`, same long prompt (`runs/envab/`):

| Setting | Long prompt TTFT | Prompt tok/s |
| --- | ---: | ---: |
| defaults (96-slot ring) | 147.9-149.7 s | 66.7-67.4 |
| `STRATA_RING_BYTES=0` | 147.8 s | 67.4 |
| `STRATA_PREFILL_RING=384` (384-slot ring) | 116.2 s | 85.8 |
| `STRATA_PREFILL_STREAM_MIN=65536` (8-slot ring, no all-expert streaming below 65536-token chunks) | 22.1 s | 451.6 |

The streaming path was measured faster on an RTX 5070 (see the comment in `prefill.cpp`); on this RX 7900 XTX with 24 GB, a 15.8 GiB expert cache and the 30 GB expert shard cached by the OS (`--mmap-experts`), it is 6.5-7x slower than not streaming. Decode and the follow-up path are unchanged or slightly faster with the setting (decode 45.9 vs 40.5 tok/s median on the long request, 61.3 vs 55.3 on follow-ups; run-to-run noise on decode is 10-15%, so only the prefill difference is clear).

## Second session, 2026-10-07: XMP enabled (DDR4-3200) and a prompt-length sweep to 256k

After the tables above (DDR4-2400) the BIOS memory profile was enabled: CPU-Z then reads a DRAM frequency of **1582 MHz** (DDR4-3200, the kit's rating). Everything else is unchanged (same builds, same pack, same flags, same clock cap). Raw data: `runs-xmp/` (the community script, `-c 131072`) and `runs-sweep/` (the sweep, `-c 262144`); both tables below are produced by `make_xmp_tables.py`.

### The same community measurements at DDR4-3200

3 iterations per configuration (the first table had 6-9), same prompts and method as above. The 0.1.40 defaults run was preceded by a discarded warm-up pass (it gave the same numbers, so it was not needed).

| Configuration | Request | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0.1.33 (aeb35be), defaults | short, cold | 76 | 0 | 234 | 3 | not measured | 31.4 (28.1-33.1) | 2.3 (1.9-2.6) |
| 0.1.33 (aeb35be), defaults | long, cold | 9963 | 0 | 256 | 3 | 507.7 (497.5-523.7) | 40.1 (35.4-43.4) | 19.6 (19.0-20.0) |
| 0.1.33 (aeb35be), defaults | follow-up | 10242 | 9958 | 256 | 3 | 98.8 (96.5-100.5) | 55.5 (46.7-56.3) | 2.9 (2.8-2.9) |
| 0.1.40 (1735d64), defaults | short, cold | 74 | 0 | 246 | 3 | not measured | 37.8 (33.2-39.5) | 2.1 (2.1-2.1) |
| 0.1.40 (1735d64), defaults | long, cold | 9961 | 0 | 256 | 3 | 63.3 (62.8-66.1) | 49.7 (37.5-51.9) | 157.3 (150.7-158.7) |
| 0.1.40 (1735d64), defaults | follow-up | 10240 | 9956 | 256 | 3 | 112.1 (97.8-122.1) | 60.5 (60.4-69.5) | 2.5 (2.3-2.9) |
| 0.1.40, STRATA_PREFILL_STREAM_MIN=65536 | short, cold | 77 | 0 | 240 | 3 | not measured | 37.6 (28.3-38.0) | 2.1 (2.1-2.3) |
| 0.1.40, STRATA_PREFILL_STREAM_MIN=65536 | long, cold | 9964 | 0 | 256 | 3 | 476.5 (442.9-479.9) | 49.5 (44.0-51.3) | 20.9 (20.8-22.5) |
| 0.1.40, STRATA_PREFILL_STREAM_MIN=65536 | follow-up | 10243 | 9959 | 256 | 3 | 98.3 (89.9-100.7) | 68.3 (66.2-70.4) | 2.9 (2.8-3.2) |

Needle in a haystack (`-c 131072`, depth 50%), DDR4-3200:

| Configuration | 32k depth 50% | 128k depth 50% |
| --- | --- | --- |
| 0.1.33 (aeb35be), defaults | found, 31,075 tokens in 184 s (169 tok/s) | found, 122,150 tokens in 1000 s (122 tok/s) |
| 0.1.40 (1735d64), defaults | found, 32,308 tokens in 298 s (108 tok/s) | found, 125,704 tokens in 1040 s (121 tok/s) |
| 0.1.40, STRATA_PREFILL_STREAM_MIN=65536 | found, 32,308 tokens in 56 s (574 tok/s) | found, 125,704 tokens in 190 s (663 tok/s) |

Compared with the DDR4-2400 table: the faster RAM changes little on this workload. The only difference outside the run-to-run range is **0.1.33's cold 10k prompt, 507.7 tok/s (497.5-523.7) against 442.9 (375.8-473.4), about +15%**. 0.1.40 with `STRATA_PREFILL_STREAM_MIN=65536` is +4.5% (476.5 against 455.9, ranges overlap), 0.1.40 at the defaults is -5% (63.3 against 66.7, inside the earlier range 61.4-68.5), and decode and the needle timings move by less than the 10-15% decode noise. The conclusion above (the 0.1.40 regression and the workaround) is the same at both RAM speeds.

### Prompt-length sweep, 16k to 256k tokens (k = 1024)

The 10k-token prompt above says little about long contexts, so each configuration was also run over eight prompt lengths with the server started at `-c 262144` (`--kv int8`, same flags; the expert cache is smaller at this context: 7360 experts, 14.01 GiB, against 8277-8300 and 15.8 GiB at `-c 131072`, so the two sections are not directly comparable). Prompt = Strata's own source tree (`*.md`, `*.py`, `*.cpp` of the checkout under test, in a fixed order) cut to the target length minus 800 tokens, behind a random nonce so nothing is reused, followed by one question; 256 output tokens, temperature 0, streaming; TTFT, prefill = prompt tokens / TTFT, decode = generated tokens / (end - first token). The length label is the target; **the actual prompt tokens are in the second column** (the token count per character grows deeper into the source tree, so the prompts land within about -10% / +10% of the label; 256k reached 254,912-261,017 of the 262,144 context). A request that did not fit would have been retried 8% shorter; none was needed and none failed. Script: `strata_ctx_sweep.py` (+ `strata_ctx_sweep_run.sh`) in `scripts/`; every length of one pass is a separate request on the same running server, in ascending order.

**0.1.33 (aeb35be), defaults**, 2 pass(es)

| Length label | Actual prompt tokens | Prefill tok/s | Decode tok/s | TTFT seconds |
| --- | ---: | ---: | ---: | ---: |
| 16k | 17,302 | 93 / 548 | 40.8 / 49.9 | 187 / 32 |
| 32k | 32,343 | 156 / 153 | 54.7 / 53.5 | 207 / 211 |
| 64k | 58,913 | 103 / 102 | 56.1 / 51.3 | 572 / 578 |
| 96k | 96,820 | 103 / 104 | 43.4 / 45.4 | 939 / 934 |
| 128k | 126,301 | 97 / 98 | 44.1 / 42.5 | 1297 / 1289 |
| 192k | 186,682 | 96 / 96 | 41.0 / 45.2 | 1935 / 1939 |
| 228k | 229,101 | 97 / 96 | 52.8 / 50.9 | 2363 / 2393 |
| 256k | 261,017 | 95 / 96 | 48.8 / 53.1 | 2737 / 2731 |

**0.1.40 (1735d64), defaults**, 1 pass(es)

| Length label | Actual prompt tokens | Prefill tok/s | Decode tok/s | TTFT seconds |
| --- | ---: | ---: | ---: | ---: |
| 16k | 14,699 | 75 | 38.4 | 196 |
| 32k | 33,980 | 74 | 52.7 | 457 |
| 64k | 63,792 | 89 | 56.4 | 720 |
| 96k | 91,319 | 85 | 53.6 | 1079 |
| 128k | 124,034 | 87 | 55.5 | 1420 |
| 192k | 197,503 | 93 | 55.5 | 2114 |
| 228k | 230,912 | 90 | 59.9 | 2578 |
| 256k | 254,912 | 91 | 48.0 | 2806 |

**0.1.40, STRATA_PREFILL_STREAM_MIN=65536**, 2 pass(es)

| Length label | Actual prompt tokens | Prefill tok/s | Decode tok/s | TTFT seconds |
| --- | ---: | ---: | ---: | ---: |
| 16k | 14,696 | 552 / 475 | 41.5 / 52.8 | 27 / 31 |
| 32k | 33,986 | 500 / 495 | 59.7 / 51.4 | 68 / 69 |
| 64k | 63,782 | 530 / 519 | 60.1 / 58.9 | 120 / 123 |
| 96k | 91,319 | 520 / 516 | 62.4 / 57.8 | 176 / 177 |
| 128k | 124,032 | 514 / 515 | 59.1 / 57.0 | 241 / 241 |
| 192k | 197,507 | 507 / 506 | 57.1 / 51.4 | 390 / 390 |
| 228k | 230,913 | 501 / 501 | 45.0 / 49.9 | 461 / 461 |
| 256k | 254,912 | 499 / 499 | 46.0 / 47.5 | 511 / 511 |

Errors/retries: {'strata-0.1.33': 0, 'strata-0.1.40': 0, 'strata-0.1.40-streammin65536': 0}

What the sweep shows:

- **Prefill is flat in the prompt length from about 60k tokens up, and the three configurations sit on three plateaus**: 0.1.33 defaults 96-104 tok/s (59k-261k), 0.1.40 defaults 85-93 tok/s (64k-255k, a single pass), and 0.1.40 with `STRATA_PREFILL_STREAM_MIN=65536` **499-530 tok/s** (two passes that agree within 1-3% at every length from 32k up). A 256k prompt reaches its first token in 2,734 s (45.6 min) on 0.1.33 defaults, 2,806 s on 0.1.40 defaults and **511 s (8.5 min)** with the setting.
- **The 10k result of the first table does not extrapolate for 0.1.33.** 0.1.33 runs at about 500 tok/s up to 10k tokens (first table) and again at 548 tok/s for a 17k prompt in the second pass of the sweep, but at 156 tok/s for 32k and about 100 tok/s from 59k up. The step is between 17k and 32k tokens; its cause was not investigated (a dependence on the prompt chunk size is a guess, not a measurement). The first 16k request of the 0.1.33 server ran at 93 tok/s and the same prompt length in the second pass at 548 tok/s; a first-request effect (the expert cache was still filling) is possible, but 0.1.40 with the setting shows none (552 tok/s on its first request), so that one 0.1.33 point is unexplained and both values are reported.
- **Decode is about 40-60 tok/s at every depth up to 261k tokens and shows no systematic decline** (run-to-run spread 10-15%); the 0.1.33 numbers at 96k-192k (41-45 tok/s) are the lowest, with the setting 45-62 tok/s.
- Not measured in the sweep: the page cache state and the disk read rate. The 0.1.33/0.1.40 plateau of about 100 tok/s is close to the 70-110 tok/s seen earlier on this machine with a cold page cache (see limitations), so whether it is a compute or a file-cache effect is open; the engine's resident set was 27 GB in all three runs.

## Correctness and limitations

- Needle recall: 2 of 2 found per configuration at each of the two RAM speeds (tables above); no failed or skipped cases. A needle test measures recall on those inputs, not overall quality.
- Not checked: output equality between the three configurations (the draft acceptance and the exact text can differ; the per-run texts are not stored), tool calls, image input, reasoning mode, sampled (non-greedy) decoding. The GPU clock cap (2850 MHz) lowers absolute numbers a little relative to an uncapped card.
- Cold page cache (first start after a WSL restart): prefill was 70-110 tok/s on 0.1.33 for 11k-113k prompts (earlier runs, same machine; not in this folder); the reported tables are warm-cache. `STRATA_PREFILL_STREAM_MIN` was not tested with a cold cache.
- One model and one pack (IQ1_M Coder); other quantizations or packs may behave differently.
- Sweep: one request per length and pass (2 passes, 1 for 0.1.40 defaults), so no range for that configuration; the prompts are source code, which tokenizes denser than documentation text, and one text per length (only the nonce changes between passes). The server was started at `-c 262144` for the sweep (smaller expert cache than at `-c 131072`). Output quality at long context was not checked beyond the needle test (which stops at 128k); decode speed at depth is for 256 greedy tokens that may include reasoning text.
