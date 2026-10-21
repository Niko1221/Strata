# Community benchmark on Strix Halo + Radeon AI PRO R9700 (OCuLink), Linux, Strata 0.2.2

Measured on 2026-10-21 by jiayeZhu. One Ryzen AI MAX+ 395 (Strix Halo, Radeon 8060S, 124 GB RAM) with a Radeon AI PRO
R9700 32 GB on an OCuLink PCIe 4.0 x4 link, Qwen3.8-Flash-Next IQ3_S, Strata 0.2.2 built from source. This is the
"Strix Halo with a card of its own on Linux" pair that the 0.2.2 release notes list as not tested. It compares the two
ways to run the pair:

- **split**: the 8060S (layers 0-29) and the R9700 (layers 30-47), every expert resident.
- **R9700 alone**: the 8060S hidden (`ROCR_VISIBLE_DEVICES=0`), the R9700 computes, and the whole expert arena sits in
  the Strix Halo's RAM (page-locked). The R9700's expert cache holds about half of the 24,576 experts (11,215 with
  `--batch 4`, 13,323 with `--kv-grow`), the CPU pool computes decode misses, and prompts stream the uncached experts
  over the link.

Main limitation: one machine, one model and quant, headless (no desktop, so nothing here speaks to the screen freezes
in the original report), and ROCm TheRock 7.14.1 rather than a packaged ROCm.

## Summary

| | split | R9700 alone, `--batch 4` | R9700 alone, one request at a time |
| --- | ---: | ---: | ---: |
| 64K / 128K / 200K prompt, tok/s | 1,627 / 1,570 / 1,499 | **2,530 / 2,599 / 2,426** | 2,519 (64K) |
| decode after the prompt, tok/s (64K / 128K / 200K) | 49 / 49 / 46 | **74 / 71 / 73** | 80 (64K) |
| decode, 1 / 2 / 4 requests at once (sum), tok/s | 72 / 66 / 82 | 82-87 / 112 / **136-140** | **103-108** / queued |
| ~570-token fresh prompt, tok/s (median) | **540** | 186 | 429-443 with the CPU share |
| ~1.8K / ~3.6K fresh prompt, tok/s | **822 / 964** | 326 / 614 | 616-624 / 750-775 with the CPU share |

- On this pair the R9700 alone is faster than the split for long prompts (+56% at 64K, 200K first token 141 s -> 87 s)
  and for decode (+50% at depth, +66-71% with four requests). The 8060S stage set the pace of the split.
- Short prompts are the R9700-alone layout's weak side: every chunk streams the half of the routed experts that is not
  in VRAM over a 7.2 GB/s link. The split, with every expert resident, reads a 570-token prompt about 2.9x faster.
- `STRATA_PREFILL_CPU_SHARE=auto` recovers most of that, but only without `--batch` (with `--batch 4` it is inert, no
  share line in the log): 570 tokens 224 -> 429 tok/s, 1.8K 402 -> 616, against a same-session control.
- `--kv-grow` on HIP fails at the default open-file limit (see below); with `ulimit -n` raised it works and puts 14%
  more experts in VRAM (11,697 -> 13,323; +2-3% on a 64K prompt).
- In standard-v1 (three fresh engines, 12/12 sanity checks each) the R9700 alone, one request at a time with the CPU
  share and `--kv-grow`, is fastest in every case but the 4K prompt read (split 1,017, it 965 tok/s), and four
  requests released together finish within 1-3 s of the `--batch 4` engine.
- 0.2.2 reads and decodes at the same speed as 0.2.1 on the split (within 1%).

## Hardware and software

- GPU 0 (HIP numbering): Radeon AI PRO R9700, 32 GB, gfx1201, OCuLink PCIe 4.0 x4 (7.2 GB/s measured by the engine's
  PCIe share probe). GPU 1: Radeon 8060S (gfx1151), the Strix Halo's integrated GPU, BIOS carve-out small, GTT used.
- CPU: Ryzen AI MAX+ 395, 16 cores / 32 threads. RAM: 124 GiB (LPDDR5X). Storage: NVMe.
- OS: Ubuntu, kernel 7.0.0-38-generic, in-tree amdgpu. Headless server, no desktop session.
- ROCm: TheRock 7.14.1 multiarch build (gfx1151 + gfx1201), hipBLASLt 1.4.1.
- Strata 0.2.2, `011c9e600a93bbc9fd451c563501b5831c32bbca`, unpatched source build, engine 0.1.47. One binary for
  gfx1151 + gfx1201 (`CMAKE_BUILD_TYPE=Release`, `STRATA_PREFILL_MMQ=ON`, `-DSTRATA_W_LB=8`, ggml `3cf03257`).
- Background: the machine's other inference containers stopped for every run; nothing else on either GPU. Load average
  from SSH sessions only.

## Model and configuration

- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, IQ3_S, Strata's
  own `iq3_s` pack (setup's `iq_pack`), `data/expert-profile.bin` of 0.2.2. MTP draft head on. Vision on (CPU encoder).
- Common: `--max-context 262144 --kv int8 --spec 4 --spec-min-p 0.5 --expert-cache auto`; env `STRATA_PF_FUSED=1
  STRATA_PF_GEMM=1 STRATA_PA_FAST=1 STRATA_HIP_WMMA=1 STRATA_SELECT_WMMA=1`. The 0.2.2 gfx12 and gfx1151 defaults apply.
- **split**: `gpu [1, 0]`, `layer_split 30`, `--mmap-experts --prefill 4096 --no-prefill-borrow --trim-stage-weights
  --batch 4 --batch-groups 1 --batch-mtp --kv-resident 32768`; env `STRATA_PREFILL_STREAM_MIN=64`, the gfx1151 hipBLASLt
  table `tools/hip/gfx1151-hipblaslt-100401.txt`, `STRATA_HC_UPMIX` off (#2706, still present in 0.2.2).
- **R9700 alone**: `gpu [0]`, `ROCR_VISIBLE_DEVICES=0`, no `--mmap-experts` (the arena is read into RAM and registered,
  46.84 GiB at 17.3 GiB/s), `--prefill 32768`, `--batch 4 --batch-mtp --kv-resident 32768`. Expert cache auto: 8,586
  slots, 11,215 of the 24,576 experts in VRAM (21.29 GiB).
- **R9700 alone, one at a time**: as above without `--batch` and without `--kv-resident`, plus
  `STRATA_PREFILL_CPU_SHARE=auto` and `--kv-grow` (with `ulimit -n` at the hard limit): 13,323 experts in VRAM
  (25.30 GiB; 11,697 without `--kv-grow`).

The exact configs (key removed, home folder replaced) are in `configs/`.

## Method

- A fresh engine for every configuration; `drop_caches` between them. 0.2.1 and 0.2.2 were compared in the same hour,
  interleaved with a repeat of 0.2.2.
- Long prompts: `scripts/llm-bench.py`, a fresh prompt cut from the Python standard library sources (no prefix reuse;
  `per-char.txt` is its tokens-per-character figure), a 16K warm-up, then two runs each at 64,537, 128,440 and 210,963
  tokens, 600 generated tokens, temperature 0, thinking at the server's default. Client-side figures: prompt tok/s =
  prompt tokens / time to the first streamed token (request overhead included), decode = (tokens - 1) / (last token -
  first token). The server's own `timings` are in the JSONL too (e.g. 2,407.3 against the client's 2,398 for the first
  64K run of the 32768 arm).
- Concurrency: `scripts/speed-check.py`, 1, 2 and 4 requests at once, 500 tokens each, the text checked.
- Short prompts: `scripts/shortp.py`, three ~570-token prompts, one ~1.8K and one ~3.6K, each with a random tag so no
  pass reuses a cached prefix; 50 tokens out, temperature 0. The figure is the server's `prompt_per_second`. Medians
  over every fresh pass (the number of samples is in the table).
- kv-grow: `scripts/vmm-probe.cpp`, the HIP virtual-memory calls of `src/core/vmm.cpp` in a standalone program.
- Per-run data: `data/*.jsonl` (llm-bench), `logs/` (the run logs and the engine logs).

## Results

### standard-v1 (draft protocol `strata-qwen38-thinking-v1-draft.1`)

`strata bench run --url http://127.0.0.1:8080 --suite standard-v1 --optional-concurrency --track custom`, one fresh
engine per run, the server configs as above. Medians of n=3 (ranges in each run's README). Folders:
`../2026-10-21-standard-strix-halo-r9700-split-b6c95103/`, `../2026-10-21-standard-r9700-strix-halo-ram-batch4-c7a2be23/`,
`../2026-10-21-standard-r9700-strix-halo-ram-one-74ba69f0/`.

| Case | split | R9700 alone, `--batch 4` | R9700 alone, one at a time |
| --- | ---: | ---: | ---: |
| short: TTFT s / decode tok/s | 0.67 / 48.1 | 1.22 / 70.2 | **0.56 / 82.9** |
| prompt-4k: prompt / decode tok/s | **1,016.7** / 42.6 | 689.7 / 65.4 | 965.4 / **78.1** |
| prompt-16k: prompt / decode tok/s | 1,544.0 / 40.3 | 1,829.0 / 60.9 | **2,166.8 / 74.7** |
| tool-call sanity T1-T4: TTFT s | 0.91-0.98 | 2.87-3.34 | **0.85-1.12** |
| concurrency, 3 rounds of 4: wall s | 32.8 / 29.0 / 29.2 | **25.9 / 24.2 / 24.0** | 26.5 / 27.2 / 25.1 |
| concurrency: aggregate tok/s | 62.2 / 70.5 / 70.1 | **74.0 / 80.7 / 77.0** | 71.9 / 74.7 / 75.2 |
| required stage after load | 3 min 17 s | 2 min 59 s | **2 min 21 s** |
| sanity checks | 12 / 12 | 12 / 12 | 12 / 12 |
| status | complete | complete | complete |

The runner reports the GPU as "AMD Radeon" with no memory, driver or ROCm version on this machine (`rocm-smi` and
`amd-smi` are installed); it is the R9700 in the two R9700 runs and both cards on the split.

### Long prompts (fresh, two runs each)

| Configuration | 64K tok/s | 128K tok/s | 200K tok/s | decode after it, tok/s |
| --- | ---: | ---: | ---: | ---: |
| split, 0.2.1 | 1,622 / 1,611 | 1,549 / 1,554 | 1,492 / 1,491 | 47.5-49.5 / 48.3-48.9 / 45.4-46.0 |
| split, 0.2.2 | 1,631 / 1,623 | 1,572 / 1,567 | 1,505 / 1,493 | 48.0-50.1 / 48.8-49.5 / 45.7-46.4 |
| split, 0.2.2 again | 1,629 / 1,623 | | | 48.2-50.2 |
| R9700 alone, `--mmap-experts`, `--prefill 16384` | 1,865 / 2,012 | | | 64.2-65.8 |
| R9700 alone, arena, `--prefill auto` (16384 on gfx12) | 1,757 / 2,163 | 2,061 / 2,121 | 1,974 / 2,000 | 77.1-78.4 / 66.9-69.0 / 69.9-74.7 |
| R9700 alone, arena, `--prefill 32768` | 2,398 / 2,662 | 2,553 / 2,645 | 2,426 / 2,425 | 73.4-74.3 / 70.0-72.5 / 70.7-75.5 |
| as above, `--adapt-async 1` (prefill auto) | 1,978 / 2,158 | | | 71.3-75.4 |
| as above, one at a time, `--prefill auto` | 2,187 / 2,164 | | | 79.8-80.9 |
| one at a time, 32768, CPU share | 2,530 / 2,508 | | | 76.3-83.3 |
| one at a time, 32768, `--kv-grow` | 2,585 / 2,603 | | | 76.9-85.7 |

### Requests at once (sum of decode tok/s, 500 tokens each, text right in every case)

| Configuration | 1 | 2 | 4 |
| --- | ---: | ---: | ---: |
| split, `--batch 4 --batch-groups 1` | 72.4 | 65.5 | 82.1 |
| R9700 alone, `--batch 4` (two engines) | 86.8 / 82.4 | 111.6 / 111.8 | 135.9 / 140.1 |
| R9700 alone, `--batch 4 --adapt-async 1` | 81.0 | 111.3 | 138.0 |
| R9700 alone, one at a time | 100.3 / 108.5 / 107.5 | | |
| one at a time, CPU share | 107.4 | | |
| one at a time, `--kv-grow` | 103.3 | | |

### Short fresh prompts (server prompt tok/s; median, samples in brackets)

| Configuration | ~570 tokens | ~1.8K | ~3.6K |
| --- | ---: | ---: | ---: |
| split | 540 (3) | 822 (1) | 964 (1) |
| R9700 alone, `--batch 4`, `--prefill 32768` | 186 (6) | 326 (2) | 614 (2) |
| R9700 alone, one at a time (control, two engines) | 224 (9) | 402 (3) | 634 (3) |
| one at a time, `STRATA_PREFILL_CPU_SHARE=auto` (with and without `_MAX=3072`, two engines) | 429 (12) | 616 (4) | 775 (4) |
| one at a time, `--kv-grow` | 207 (6) | 434 (2) | 793 (2) |
| one at a time, `--kv-grow` + CPU share | 443 (6) | 624 (2) | 750 (2) |
| R9700 alone, `--batch 4`, `STRATA_PREFILL_CPU_SHARE=auto` (inert) | 214 (3) | 382 (1) | 606 (1) |

The ~3.6K column is within pass-to-pass noise (the control's second pass read 798). `STRATA_PREFILL_CPU_SHARE_MAX=3072`
made no difference here. The first prompt after an engine start is always the slowest of its pass.

### `--kv-grow` and the open-file limit

`--kv-grow` maps the expert cache in 2 MiB VMM chunks. At the default soft `RLIMIT_NOFILE` of 1024, `hipMemCreate`
returns `hipErrorOutOfMemory` at the 1,017th chunk (2.0 GiB) with 31.8 GiB free, so the engine cannot map any cache
size. At the hard limit (524288) the probe maps 12,000 chunks (23.4 GiB) and the engine starts with 13,323 experts in
VRAM instead of 11,697; as the context grows the K/V takes chunks back (`K/V grown to 65536 cells (0.86 GiB); the expert
cache gave 320 slots for it`). This is the limit #2398 found and raised in the engine; #2398 was closed in the
0.2.0-beta tracker reset without a merge, and its raise is now the first commit of the open #2399. 0.2.2 does not have
it, so on Linux/HIP `--kv-grow` needs `ulimit -n $(ulimit -Hn)` before the server starts (the engine inherits it).
`scripts/vmm-probe.cpp` is the standalone probe (`hipMemCreate` / reserve / map / set access / memset of N chunks).

## The 0.2.2 "Strix Halo with a card of its own" items on this pair

- **A real Strix Halo plus R9700 on Linux:** the unpatched 0.2.2 split ran every prompt here (64K to 200K, short
  prompts, 1 / 2 / 4 at once) with `STRATA_PF_GEMM=1` and the gfx1151 defaults (which include `PF_PAD`): no exit on the
  first prompt, so PR #3484 holds on the real pair. The local `PF_PAD` patch we carried for 0.2.0 / 0.2.1 is no longer
  needed. No amdgpu reset, timeout or fault in the kernel log during the day's runs (8 days of uptime).
- **The unified-memory cache cap:** it fires on the 8060S stage of the split: `this GPU shares system RAM: 97.62 GiB
  free on the GPU, but the OS has 97.40 GiB available; 9.96 GiB stay for the desktop, so the cache is sized from 87.44
  GiB`. Every expert is resident on the split anyway, so it changed nothing measurable here.
- **nice +5:** the server logs `engine priority: nice +5` and the engine runs at nice 5 in both layouts, including the
  R9700-alone layout where the 8060S is hidden from the engine and no desktop runs. Not measured whether it costs
  anything when the CPU pool is busy with decode misses.
- **Whether the freeze comes from the engine's queues:** not tested; this machine is headless. The engine opens 2 GPU
  queues in a fixed order on the R9700 (queue priming on by default); priming on/off was not measured.
- **The iGPU is never the last stage:** with the explicit `layer_split 30` and `gpu [1, 0]` the R9700 is last, as before.

## Correctness and limitations

- speed-check.py checks each answer's text; every request was right. The three standard-v1 runs passed 12 of 12 sanity
  checks each (instruction 4/4, retrieval 4/4, tool calls 4/4).
- One machine, one model (IQ3_S). Other quants, contexts above 262,144, and Windows were not run.
- Short-prompt medians come from 1 to 12 samples per cell; the 3.6K column is noisy.
- `STRATA_PREFILL_CPU_SHARE` changes the last bits of the answer (as documented); not checked here.
- `--kv-grow` was measured only with the open-file limit raised by hand.
- Headless: nothing here confirms or rules out the reported screen freezes.

## Files

- `configs/`: the server configs of every arm (API key removed, `/home/<user>` replaced).
- `data/`: llm-bench JSONL per arm (timings, token counts; no prompt text).
- `logs/`: the run logs and each arm's whole engine log (paths, host name, LAN addresses replaced).
- `scripts/`: `llm-bench.py`, `speed-check.py`, `shortp.py`, `vmm-probe.cpp`.
