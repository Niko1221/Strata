# Community benchmark on RX 7900 XTX (gfx1100, Windows 11 + WSL2): Strata 0.1.33 vs 0.1.40, Coder IQ1_M

Measured on 2026-10-06 by theGiallo. One consumer RDNA3 card on a PC with a 24-thread CPU and 54 GB of RAM visible to WSL2, serving the two-shard Qwen3.8-Flash-Next GSQ-RCO IQ1_M GGUF with MTP through `serve.server`. The main finding is a **prefill regression on this GPU between 0.1.33 and 0.1.40** (about 6.5-7x slower cold prompts at the defaults) and a one-variable workaround that restores and exceeds the 0.1.33 speed: `STRATA_PREFILL_STREAM_MIN=65536`. Limitation: one machine, one model, one pack; the 10k-token prompt is a neutral documentation text (Strata's own `docs/`), the long needle prompts are `tools/needle_bench.py`'s.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GB (gfx1100). Core and VRAM clocks capped at 2850 MHz by a driver setting (the driver reset under load without the cap). CPU: AMD Ryzen 9 5900X, 12 cores / 24 threads. Motherboard: MSI MAG B550 TOMAHAWK (MS-7C91), AMD B550 chipset, BIOS A.K1 (2025-09-09). Installed RAM: 2 x 32 GB Kingston KHX3200C16D4/32GX (a DDR4-3200 CL16 kit) running at **DDR4-2400** (CPU-Z: DRAM frequency 1187 MHz, CL17-17-17-39, command rate 2T, 1.2 V), not at the kit's rated XMP-3200 (CL16, 1.35 V), which was not enabled in the BIOS for these measurements, 64 GB in total, 54 GB visible to the WSL2 VM. Storage: Windows `C:` (NTFS) and the WSL2 virtual disk that holds the models are on one Kingston SFYRD2000G 2 TB NVMe SSD (the distribution's VHDX sits under the user's AppData/Local/wsl folder on `C:`); the model files are read through the WSL ext4 disk and the VM's page cache. PCIe link of the GPU and of the SSD: not measured.
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

## Correctness and limitations

- Needle recall: 2 of 2 found per configuration (table above); no failed or skipped cases. A needle test measures recall on those inputs, not overall quality.
- Not checked: output equality between the three configurations (the draft acceptance and the exact text can differ; the per-run texts are not stored), tool calls, image input, reasoning mode, sampled (non-greedy) decoding. The GPU clock cap (2850 MHz) lowers absolute numbers a little relative to an uncapped card.
- Cold page cache (first start after a WSL restart): prefill was 70-110 tok/s on 0.1.33 for 11k-113k prompts (earlier runs, same machine; not in this folder); the reported tables are warm-cache. `STRATA_PREFILL_STREAM_MIN` was not tested with a cold cache.
- One model and one pack (IQ1_M Coder); other quantizations or packs may behave differently.
