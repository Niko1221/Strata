# Community benchmark on 2x AMD Radeon PRO W6800 (64 GB, gfx1030, Linux HIP)

Measured on 2026-11-15 by @Net-Sentinel with `tools/strata_bench.py` (suite 1, `--runs 1`, plus production 56k-context agent turn measurements from the same engine session). Speeds are the engine's own timings (`prompt_per_second`, `predicted_per_second`); first-token times are measured by the client over HTTP.

This is the first **Dual Radeon PRO W6800 (64 GB VRAM, `gfx1030`) Linux HIP** report for **DeepSeek-V4.1-Flash 2.00 bpw** at the full **262,144-token (`256k`) context window** (complementing the single-W6800 Windows report in #5144, and providing the benchmark package for #6038).

Main limitation: because multi-chunk layer-split prefill (`>15k` tokens) currently deadlocks inside `run_dev_chunks -> host_wait_at_least` on Linux HIP `gfx1030` (see #6038 / #4211), this configuration runs with `--prefill-chunk 0` (16-token non-blocking windows). That makes fresh uncached prompt prefill ~127 tok/s instead of chunked prefill speed, while `ds4-slots` cached turns reuse prefixes at `0.0–0.1 s` slot restore.

## Hardware and software

- **GPU and VRAM:** 2x AMD Radeon PRO W6800 32 GB GDDR6 (`64 GB` total VRAM, `gfx1030`, selected via `HIP_VISIBLE_DEVICES=1,2`; a 3rd card, RX 6800 16 GB at ordinal 0, is isolated for Whisper STT and Plex VAAPI)
- **CPU:** AMD Ryzen 9 5950X 16-Core Processor (16 cores, 32 threads)
- **Installed RAM:** 128 GB (125.7 GiB usable, 4x 32 GB DDR4-3600 MT/s matched Corsair CMK64GX4M2D3600C18)
- **Storage:** Model pack and 110.6 GB Engram on PCIe 4.0 x4 NVMe (`nvme0n1` FIKWOT FN950 2 TB, ext4 `/srv/llms`)
- **OS & Kernel:** Ubuntu 24.04 LTS, Linux `6.8.0-149-generic` (`lru_gen.enabled=0 amd_pstate=active`), `THP=always` (`shmem_enabled=within_size`), `vm.swappiness=1`
- **Engine:** `0.1.59-linux-x86_64-hip-rx6000/strata-dsv41` (release binary from `v0.2.13 d2f09647`, portable `gfx1030`/`gfx1031`, bundled ROCm 7 / hipBLASLt / Tensile libraries)
- **Background workloads:** Idle home server (ZFS pool online, Whisper/Piper idle; W6800s idle at `8 W + 9 W` before benchmark)

## Model and configuration

- **Model and quantization:** `deepseek-v4.1-flash` (DeepSeek-V4.1-Flash `1017`, ISTA-DASLab `2.00 bpw` GSQ-RCO pack `deepseek-v4.1-flash-ist-das-2.00bpw` + `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` 110.6 GB)
- **Context:** `262,144` tokens (`256k`); parallel slots: 1 (FIFO, `ds4-slots` enabled: 4 in RAM + disk folder)
- **Layer & Expert placement across Dual W6800 (64 GB VRAM) + 128 GB DDR4:**
  - `--tp 2` falls back to 2-GPU layer split (`tensor parallel 2 -> 1`):
    - **GPU 0 (`layers 0..20`):** weights `2.96 GB`, experts `27.43 GB` (`3,321 / 8,064` in VRAM), KV/caches `0.24 GB`, `900 MiB` free
    - **GPU 1 (`layers 21..39 + head`):** weights `2.91 GB`, experts `27.67 GB` (`3,045 / 7,296` in VRAM), KV/caches `0.10 GB`, `878 MiB` free
    - **Total VRAM experts:** `6,366 / 15,360` (`55.10 GB` across both W6800s)
    - **Total VRAM KV cache at 262,144 context:** `0.34 GB` combined across the 4 global KV layers (`2, 8, 14, 20`) + 48 sliding-window (`512`) layers
    - **CPU-lane experts in DDR4 RAM:** `8,994 / 8,994` copied into RAM (`77.3 GB`, 1 NUMA part) in `22.1 s` (`100%` RAM resident, zero SSD demand-read spill)
- **Settings:** thinking off for the suite (`reasoning_effort: "none"`, `reasoning_budget_tokens: 0`); temperature 0; `DS4_COPY_PIN` unset (default off on RDNA2 without 2MB `hugetlb`, avoiding the 303 s `kfd_ioctl` page-pinning stall)

```text
strata-dsv41 --serve --pack /srv/llms/strata/data/packs/deepseek-v4.1-flash-ist-das-2.00bpw --max-context 262144 --eos 1 --vision on --dspark auto --dspark-conf 0.65 --tp 2 --reserve-mib 900 --cpu-threads 12 --prefill-chunk 0 --layer-split auto
```

## Method

Run by `python tools/strata_bench.py --url http://127.0.0.1:8088 --config strata-deepseek-v4.1-flash-ist-das-2.00bpw.json --runs 1` against the running server; model loading (`22.1 s` for the 77.3 GB CPU expert copy) is not included in request timings. One warm-up chat and a 1K calibration prompt are discarded first. Every fresh-prompt test starts with a unique tag so `ds4-slots` cannot assist it.

- **short chat:** the three questions of `tools/ab_engine.py`, up to 100 new tokens
- **4K / 16K prompts:** generated word lists sized by calibration, 24 new tokens, fresh prompt each
- **agent session:** system prompt and four tools, then 5 turns that each add ~1,500 words of a project file and continue from the model's own earlier answers, up to 64 new tokens per turn
- **deep-context production agent turns (55k–56k tokens):** captured from the same engine session (recorded in `engine-log-excerpt.txt`) showing fresh vs `ds4-slots` cached prefix reuse at 56k context

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 3 | not measured | 23.6 (21.2-24.5) | 0.50 (0.47-0.64) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 1 | 127.9 (127.9-127.9) | 25.2 (25.2-25.2) | 32.26 (32.26-32.26) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 1 | 126.8 (126.8-126.8) | 23.6 (23.6-23.6) | 130.63 (130.63-130.63) |
| agent session, follow-up turns | 14070 (14070-14070) at the last turn | see below | up to 64 | 1 | see below | 25.1 (24.3-25.9) | 31.08 (17.99-53.38) |
| 56K agent turn, cached (`ds4-slots`) | 55887 | 55742 | 400 | 1 | 86.3 (145 read in 1.68 s) | 25.7 | 1.68 |
| 56K agent turn, slot swap + follow-up | 56605 | 56286 | 197 | 1 | 92.6 (319 read in 3.44 s) | 24.7 | 3.44 |

### Agent session (conversation cache — `tools/strata_bench.py`)

Cache hit over the follow-up turns: `69.1 %`; earlier-turn tokens read again per session: `5` (the `13,177` new tokens of the added files are read once); first turn (nothing cached) first token: `11.84 s`.

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s | CPU lane experts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 50.9 | 11.837 | 23.1 | 9.0 % |
| 1 | 3717 | 639 | 3078 | 57.7 | 53.385 | 24.3 | 9.2 % |
| 2 | 5235 | 3780 | 1455 | 81.0 | 17.988 | 25.9 | 9.2 % |
| 3 | 8378 | 5298 | 3080 | 86.4 | 35.660 | 25.1 | 9.2 % |
| 4 | 11249 | 8440 | 2809 | 90.5 | 31.080 | 25.1 | 9.2 % |
| 5 | 14070 | 11310 | 2760 | 91.9 | 30.054 | 25.4 | 9.2 % |

Per-request data: `results.json`. Engine startup and 56k-context log lines: `engine-log-excerpt.txt`.

## Correctness and limitations

- All `tools/strata_bench.py` turns completed with valid answers and tool-call continuations; in live OpenClaw production on this session, 56k-token multi-step tool turns decode at `24.7–27.3 tok/s` with `9.0%–11.0%` of routed experts served by the CPU DDR4 lane and `89%–91%` served directly from the two W6800s' VRAM.
- See issue #6038 for the `>15k` multi-chunk prefill deadlock without `--prefill-chunk 0` and the `DS4_COPY_PIN=1` `kfd_ioctl` stall without `hugetlb`.
