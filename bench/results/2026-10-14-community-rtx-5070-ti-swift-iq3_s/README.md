# Community benchmark on RTX 5070 Ti 16 GB (Linux), Swift 1.5 IQ3_S at 1,048,576 tokens

Measured on 2026-10-14 (22:51-23:42 CDT) by [qiuzan2001](https://github.com/qiuzan2001). Strata 0.1.44.1 source
build, Swift 1.5 IQ3_S, one GPU, the 1,048,576-token YaRN context the install was set up with. The official suite
(`tools/strata_bench.py --long`) ran 3 times per test, plus a 256-token decode sweep at about 4K, 32K and 128K
prompt tokens (3 runs each), needle recall from 32K to about 1M tokens, and a 3-round A/B of this 1M context against
the trained 262,144 tokens. Main limitation: the main tables come from one engine session on a desktop in use;
only the A/B restarted the engine, and no other model size was run.

## Hardware and software

- **GPU:** NVIDIA GeForce RTX 5070 Ti, 16,303 MiB (`nvidia-smi`), VBIOS 98.03.58.00.E6, power limit 300 W
  (the default and the maximum), clocks not locked. **PCIe:** the card's link is gen 5 x16, the board's slot gen 4,
  so it ran at **gen 4 x16** under load (`lspci`: 32 GT/s capable, 16 GT/s x16; `nvidia-smi` gen 4 x16 in every busy
  sample of windows 1 and 2; window 3 did not record the link). The engine's probe: 27.9 GB/s host to device.
- **CPU:** AMD Ryzen 7 5800X3D, 8 cores / 16 threads, AVX2, **no AVX-512** (the engine: `the expert kernels run on
  AVX-2`); 7 expert-pool workers on CPUs 1-7 and the host thread on CPU 0. Board MSI MAG B550M MORTAR WIFI.
- **RAM:** 80 GiB installed (2x 32 GiB + 2x 8 GiB DDR4, 3,200 MT/s; `dmidecode` prints the sizes as "GB"),
  76.5 GiB usable (`MemTotal`). 8 GiB swap (8,191 MiB), 2.9-3.8 GiB of it in use during the windows (`free -m`);
  which processes' pages were in swap was not checked.
  Transparent huge pages `madvise`, no reserved huge pages (`vm.nr_hugepages=0`).
- **Storage:** SK hynix Gold P31 1 TB NVMe (`SHGP31-1000GM-2`), ext4. The model files were read through the page cache.
- **OS:** Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic. **Driver** 595.99.02 (`nvidia-smi` reports CUDA 13.2,
  the driver's version); **CUDA toolkit 13.1** (`/usr/local/cuda-13.1`, the build's nvcc).
- **Strata:** source build at `296b59d2` (tag line 0.1.44.1), engine 0.1.44.1, `engine/BUILD.json` source `local`,
  archs `[120]`, vision `none`. CMake: Release, `STRATA_ENABLE_CUDA=ON`, `STRATA_NATIVE_EXPERTS=ON`,
  `STRATA_PORTABLE=OFF` (native CPU build), everything else at setup's defaults. No local source change.
- **Background:** a GNOME desktop session (GNOME Shell, Firefox and a terminal held 0.75 GiB of the card's VRAM),
  and several mostly idle terminal programs. No other inference, game or build ran during the measurements, and
  nothing else used the GPU. Nothing was pinned or isolated besides the engine's own pool threads. Raw captures
  (`free -m`, `lspci`, `dmidecode`, `nvidia-smi -q -d POWER`, `sha256sum`): [hardware.txt](hardware.txt).

## Model and configuration

- [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  IQ3_S, downloaded by setup on 2026-10-14. Setup pins this repository at `b22d729e` (2026-09-24), which has no
  IQ3_S files, so setup took the repository's current files (its fallback for a file missing at the pin); on that
  day the head was `99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d` (last changed 2026-10-07). That is the revision by
  date; setup records none:
  - `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`, 44,900,816,736 bytes; the repository's
    SHA-256 `46cb3996cd1a17fe8bc5ae1f9bb16869480de5a03f17f868d6047a20a46edd3b`
  - `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`, 38,842,920,384 bytes; the repository's
    SHA-256 `70ce6fb48a94fe34bd7e9a53a526be68066c64b5a5077cdc82fb31350b66b608`
  - The local files have the repository's sizes and SHA-256 (`sha256sum`, run on 2026-10-14 after the measurements).
- Pack `packs/swift-iq3_s` written by setup's `tools/iq_pack.py` (dense weights 1,466 MiB; experts served natively
  from shard 1). Expert profile: the shipped `data/expert-profile.bin` (24,576 ranked pairs). MTP draft layer
  `mtp/rt` (836 MiB of VRAM). Served model id `swift-1.5-iq3_s`. No vision encoder, no custom pack or profile.
- **Context** 1,048,576 tokens (`--max-context 1048576 --rope-scaling yarn --rope-scale 4`, against the trained
  262,144; the engine: `freq_scale 0.25 ... mscale 1.138629`). **KV** int8, 32,768 of 1,048,576 cells per QSA layer
  in VRAM, the rest in **12.38 GiB of pinned RAM**. **Expert arena** 46.84 GiB in RAM (loaded at 2.99 GiB/s,
  `cudaHostRegister` ok, transparent huge pages). **Expert cache** auto: **3,132 slots, 6.00 GiB of VRAM**, profile
  order, no eviction. **Prefill** auto (8,192-token chunks; the prompt path borrows 2,695 slots, 5.15 GiB).
  No low-RAM mode.
- **Speculation** `--spec 4 --spec-min-p 0.5` with the MTP draft layer (the server reports `spec 6`, `mtp_max 4`,
  `lookup 3`). `STRATA_ROUTE_TAIL_SKIP=7` on (the default), adaptive overlap off (the default),
  `--adapt-async` off (the default). Greedy (temperature 0), thinking off, no calibration, no experimental speed
  projection.
- **VRAM:** the engine logged `325 MiB of VRAM free with everything loaded` and its warning that below 512 MiB the
  driver can page GPU memory to RAM (#2065). No such slowdown was seen in these runs (every decode rate is in
  the tables below); `--vram-reserve-mib` was not added.

The settings file as setup wrote it, paths replaced: [strata-swift-iq3_s.json](strata-swift-iq3_s.json).

```text
.venv/bin/python serve/server.py --engine strata --config strata-swift-iq3_s.json --port 8080   (with an API key)
engine: --pack <data>/packs/swift-iq3_s --native <data>/models/swift-IQ3_S/...-00001-of-00002.gguf
        --ple-gguf <data>/models/swift-IQ3_S/...-00001-of-00002.gguf --expert-profile <strata>/data/expert-profile.bin
        --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <data>/mtp/rt
        --max-context 1048576 --kv int8 --kv-resident 32768 --rope-scaling yarn --rope-scale 4
```

## Method

- Windows 1-3 went to the server that was already running (started about 2 h 45 min before, model loaded,
  expert cache warm from normal use; loading is not timed). Requests were serial. The server was not restarted
  between windows.
- **Window 1:** `STRATA_KEY=... python tools/strata_bench.py --url http://127.0.0.1:8080 --long` (the official
  suite: three short chats x 3 runs with up to 100 new tokens; fresh 4K / 16K / 32K word-list prompts with 24 new
  tokens, 3 each; three 6-turn agent sessions with up to 64 new tokens per turn), then
  `python tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`.
- **Window 2:** [decode-sweep.py](decode-sweep.py) (written for this report; it uses `strata_bench.py`'s streamed
  client and `ab_engine.py`'s word list): fresh prompts of about 4K, 32K and 128K tokens that ask for a long Python
  module (the word list keeps its own closing line asking for a summary, and the request for the module follows
  it), 256 new tokens, 3 runs per size, one discarded warm-up. Every prompt starts with a unique tag; the engine
  log shows `0 reused` on all nine.
- **Window 3:** `needle_bench.py --lengths 256k`, `512k`, `1000k` with `--depths 50`, one request each.
- **Window 4:** the context A/B, described in its own section below; the only window that stopped the running server.
- **Timing boundaries:** prompt and decode tok/s are the engine's own (`prompt_per_second`,
  `predicted_per_second`; the same numbers as the engine log's `read in` / `generated in` lines); decode counts every
  generated token, drafts included. First-token time and total latency are client-side over HTTP on the same PC
  (from sending the request to the first non-empty streamed piece, and to the end of the stream).
- **GPU telemetry:** `nvidia-smi --query-gpu=... -lms 500` during windows 1 and 2, every 2 s in window 3
  ([gpu-telemetry-window1.csv](gpu-telemetry-window1.csv), [2](gpu-telemetry-window2.csv),
  [3](gpu-telemetry-window3.csv)). "Busy" below means samples with utilisation of 50% or more.
- **Memory:** start-up figures from the engine log, and `free -m` before and during the windows; no per-request peak.

## Results

### Official suite (window 1)

Medians and ranges; `Runs` counts requests that returned a token.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat (3 questions x 3) | 46 (43-51) | 0 | 100 each | 9 | not reported (43-51-token prompts) | 61.2 (54.9-67.4) | 0.64 (0.55-0.68) |
| 4K prompt, fresh | 4,019 | 0 | 24 cap | 3 | 1,987.4 (1,671.2-2,005.1) | not reported (24-token cap) | 2.06 (2.03-2.44) |
| 16K prompt, fresh | 16,058 | 0 | 24 cap | 3 | 3,121.6 (2,989.5-3,137.4) | not reported (24-token cap) | 5.18 (5.17-5.42) |
| 32K prompt, fresh | 32,049 | 0 | 24 cap | 3 | 3,053.8 (3,046.1-3,058.7) | not reported (24-token cap) | 10.54 (10.53-10.58) |
| agent session, all turns | 609-14,577 | see below | 11-64 | 16 | see below | 66.1 (55.6-82.3), 13 follow-up turns | 2.16 (2.01-2.37), 13 follow-up turns |

Agent sessions: 68.6% (54.3-68.8) of the follow-up turns' prompt tokens came from the conversation cache; 73
(3-142) tokens of earlier turns were read again per session; the first turn (609 tokens, nothing cached) took
1.42 s (1.40-1.49) to the first token. The first 4K prompt (1,671 tok/s) was the first long prompt after the
short chats; the two after it read at 1,987 and 2,005.

**One failure:** in the first agent session, turn 4 (the fifth of six; the suite counts from 0; 11,627 prompt
tokens, 8,765 reused) generated one token and no text, so the suite counts it as "no token came back" and that
session stopped there. The engine log has the request (`1 generated in 15 ms`) and no error line; why the model
ended the turn at once was not looked into. It is listed in [results.json](results.json) under `failures` and left
out of every median above. The other two sessions ran all six turns.

The four engine log lines before the first chat (34, 260, 906 and 1,028 prompt tokens) are the suite's own
warm-up answer and its tokens-per-word calibration; they are not in the tables.

### Decode at three context lengths (window 2)

256 new tokens on every run (no early stop), fresh prompts, 3 runs each.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| ~4K prompt, 256 out | 4,027 | 0 | 256 | 3 | 1,988.0 (1,964.6-1,996.0) | 72.7 (68.9-75.8) | 2.06 (2.05-2.07) |
| ~32K prompt, 256 out | 31,963 | 0 | 256 | 3 | 3,057.8 (3,035.8-3,069.5) | 70.5 (65.4-73.7) | 10.50 (10.46-10.58) |
| ~128K prompt, 256 out | 127,605 | 0 | 256 | 3 | 2,902.0 (2,899.6-2,914.7) | 66.3 (66.3-72.7) | 44.08 (43.88-44.11) |

Total latency (client, whole request): 5.58 s (5.42-5.74), 14.08 s (14.03-14.40) and 47.91 s (47.38-47.95).
Drafts accepted 131-156 of 206-246 per run. Every run: [decode-sweep.json](decode-sweep.json).

### Needle recall (windows 1 and 3)

`tools/needle_bench.py`; one request per cell, exact code word in the answer. Prompt tok/s from the engine log.

| Length label | Prompt tokens | Depth | Found | Prompt tok/s | Client seconds |
| --- | ---: | --- | --- | ---: | ---: |
| 32k | 32,778-32,779 | 10 / 50 / 90% | yes / yes / yes | 2,957 / 3,125 / 3,132 | 11.6 / 10.7 / 10.7 |
| 128k | 125,044-125,046 | 10 / 50 / 90% | yes / yes / yes | 2,967 / 2,944 / 2,843 (16,384 reused) | 42.6 / 42.8 / 38.6 |
| 256k | 257,323 | 50% | yes | 2,807 | 93 |
| 512k | 505,716 (16,384 reused) | 50% | yes | 2,484 | 199 |
| 1000k | 1,034,541 (16,384 reused) | 50% | yes | 1,967 | 519 |

**9 of 9 found.** The three requests marked "16,384 reused" started with the same 16K tokens of text as the
request before them, so the conversation cache served those; the rest were read fresh. At 1,034,541 tokens
the KV streaming line reads `76.86% of 332724 block reads hit VRAM, 310.1 MiB read from RAM`. Data:
[needles-32k-128k.json](needles-32k-128k.json), [needles-256k-512k-1000k.json](needles-256k-512k-1000k.json).

### GPU clocks, power and memory

| Window | Busy samples | SM clock MHz median (range) | Memory clock MHz | Power W median (range) | Temperature C median (range) | VRAM used MiB |
| --- | ---: | --- | ---: | --- | --- | --- |
| 1 (suite, needles 32K/128K) | 559 of 591 | 2,872 (2,857-2,917) | 13,801 | 245 (64-280) | 63 (40-69) | 15,444-15,587 |
| 2 (decode sweep) | 403 of 406 | 2,872 (2,865-2,910) | 13,801 | 238 (59-267) | 63 (43-68) | 15,444-15,587 |
| 3 (needles 256K-1M) | 402 of 406 | 2,865 (2,842-2,895) | not recorded | 263 (147-275) | 68 (52-71) | 15,444-15,587 |

The clock-event field showed the software power cap (`0x4`) in 189 of 559 busy samples in window 1, 73 of 403
in window 2 and 281 of 402 in window 3 (long prompts only), with the measured draw at most 280 W against the 300 W limit (`nvidia-smi -q -d POWER`: current, default and
maximum limit 300 W); the flag also shows at draws of 127 W and up, and why it is set there was not looked into.
There was no thermal or hardware slowdown flag. RAM: 73,973-74,512 MiB of 78,301 MiB in use during window 3
(`free -m` after each needle), 3.8-4.3 GiB available, 3,537-3,868 MiB of swap in use ([hardware.txt](hardware.txt)). VRAM: the engine
reports 325 MiB free inside its own process while `nvidia-smi` shows 15,444-15,587 of 16,303 MiB in use (716-859 MiB
not in use); the two count differently and the difference was not looked into.

Engine start-up lines and every engine line of the three windows: [engine-log-excerpt.txt](engine-log-excerpt.txt).
The decode expert-cache hit rate per request is in that log (53-72% on the short chats).

### What the 1M context costs on this card (window 4, a separate A/B)

The 1M context is not setup's default. To show its price on a 16 GB card, the same config ran with
`--max-context 262144` and no `--rope-scaling` / `--rope-scale` (the model's trained window, everything else the
same) in an A/B with [context-ab/ab_more.py](context-ab/ab_more.py): a fresh server per arm and round, order
1M, 262K / 262K, 1M / 1M, 262K, one discarded warm-up answer, then eight short prompts and one 6,345-token prompt,
256 new tokens each, greedy, thinking off. The user's own server was stopped for this window (23:34-23:42 CDT).
"Pooled short decode" is the engine's sum of generated tokens over the sum of decode time across the eight
short answers of one round.

| Arm | Cached experts | Pooled short decode tok/s per round | Median (range) | 6,345-token prompt tok/s median (range) | Its decode tok/s median (range) |
| --- | --- | --- | --- | --- | --- |
| 1,048,576, YaRN x4 | 3,214 / 3,138 / 3,135 (6.00-6.15 GiB), K/V 12.38 GiB pinned RAM | 73.38 / 71.77 / 72.21 | 72.21 (71.77-73.38) | 2,580.5 (2,577.9-2,600.9) | 64.0 (62.7-64.7) |
| 262,144 | 3,925 / 3,928 / 3,849 (7.35-7.50 GiB), K/V 3.09 GiB pinned RAM | 78.19 / 75.91 / 77.54 | 77.54 (75.91-78.19) | 2,714.2 (2,686.1-2,719.3) | 64.4 (62.8-65.5) |

262K against 1M in each round: **+6.6%, +5.8%, +7.4%** pooled short decode (3 of 3 faster) and +5.2% on the
6,345-token prompt. The 1M context's K/V needs more of the card (the prompt path borrows 2,695 slots against
2,306), so about 710-790 fewer experts stay in VRAM. The two arms do not write the same text (the RoPE differs):
5 of the 27 answer pairs are identical, so the decode figures include different draft acceptance. Per-request data,
both arms' engine logs and the summary: [context-ab/](context-ab/). During this window three other processes
(git, pip and a terminal program) took 25-90% of one core at times in the first 90 s (10-second samples); nothing
else used the GPU. The arms as `ab_more.py` takes them (the config is
[strata-swift-iq3_s.json](strata-swift-iq3_s.json) with `<strata>` and `<data>` replaced by your folders):

```text
python context-ab/ab_more.py strata-swift-iq3_s.json ctx1m: \
  "ctx262k:drop=--rope-scaling;--rope-scale;--max-context,xargs=--max-context;262144" --rounds 3 --out ab-context
```

The script writes its engine logs as `engine_<arm>_r<round>.log`; they are stored here as `.txt`.

## Correctness and limitations

- **Recall:** 9 of 9 needles found, from 32,778 to 1,034,541 tokens, one request per cell; an outcome on these
  inputs, not a success rate and not a quality test. No coding, tool-call or answer-quality check was run.
- **One session.** The suite, the decode sweep and the needles come from one engine session of the installed
  configuration, already warm. Only the context A/B started fresh engines (six of them); its 1M arm's pooled short
  decode moved 71.8-73.4 tok/s between starts. No repeat on another day. The short-chat range (54.9-67.4 tok/s)
  holds both question-to-question and run-to-run spread: per question the medians are 63.2 / 61.2 / 59.6 tok/s
  with 68-73 / 54-58 / 49-51 drafts accepted of 82-105.
- **Other RTX 5070 Ti results, not like-for-like** (all Windows, original Flash-Next IQ3_S, different CPUs and
  engines):
  - [2026-10-04-community-rtx-5070ti-windows](../2026-10-04-community-rtx-5070ti-windows/README.md) (Ryzen 7
    9800X3D, 0.1.39, 262,144 tokens): decode 74.7-82.8 tok/s after its prompts, prefill 3,012 tok/s at 5,213
    tokens. This box's 262K arm above decodes 77.5 tok/s (pooled, short prompts).
  - [2026-10-06-community-rtx5070ti-5900x](../2026-10-06-community-rtx5070ti-5900x/README.md) (Ryzen 9 5900X,
    0.1.39, 65,536 tokens, three sizes).
  - [#2173](https://github.com/Niko1221/Strata/issues/2173) (Threadripper 7960X with AVX-512, 0.1.43.1, 65,536
    tokens, 3,845 experts in VRAM, the same `strata_bench.py --long` suite): 96.3 tok/s short-chat decode and
    2,865 / 3,448 / 3,458 tok/s prompt at 4K / 16K / 32K. Here: 61.2 tok/s (-36%) and 1,987 / 3,122 / 3,054 tok/s
    (-31% / -9% / -12%). CPU, engine version, fine-tune and context all differ, so this report does not say which
    of them causes the gap.
- **Not measured:** other model sizes (IQ2_XS, Q2_0, IQ3_XXS), vision, concurrency (`--clients`) and `--batch`,
  thinking on, the opt-in engine settings (`--adapt-async`, other `--spec` values; measured separately), peak RAM
  and VRAM per request, a cold page cache, power-limited or locked clocks, Windows on this PC.
- **Units:** VRAM and RAM in MiB and GiB as `nvidia-smi`, `free` and the engine print them; installed RAM in GiB
  (DIMM sizes); file sizes in bytes; download speeds and disk sizes in GB; link rates in GB/s and GT/s as the probe and `lspci` print them.
