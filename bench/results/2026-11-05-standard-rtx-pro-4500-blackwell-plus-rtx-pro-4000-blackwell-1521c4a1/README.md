# Community benchmark on RTX PRO 4500 + RTX PRO 4000 (HP Z820, dual Xeon E5-2697 v2)

Measured on 2026-11-05 by jkenner111. Tested: `UD-IQ4_XS-v31` and the older `UD-IQ4_XS` of
Qwen3.8-Flash-Next at 262,144 context on two mismatched Blackwell cards (32 GB + 24 GB, PCIe 3.0,
no NVLink) in a 2013 dual-socket Ivy Bridge-EP workstation with **no AVX2**, plus a comparison of the
new and old PCIe share rules.
Main limitation: the engine picks how to divide the model between the two cards itself, and it warns
that on two different cards it cannot tell which division is faster — so these numbers show what it
chose, not necessarily the fastest this box can do.

## Hardware and software

- GPU 0: NVIDIA RTX PRO 4500 Blackwell 32 GB (34,207,694,848 bytes), power limit 200 W, 82 SMs at
  2.41 GHz.
- GPU 1: NVIDIA RTX PRO 4000 Blackwell 24 GB (25,655,508,992 bytes), power limit 145 W, 70 SMs at
  2.06 GHz.
- Both cards on PCIe 3.0, **no NVLink**. Engine link probe 13.1-13.2 GB/s host-to-device. Layer split
  (the split engine needs identical cards of cc 8.6/12.0 with all routed experts resident, so it
  cannot run here).
- CPU: 2x Intel Xeon E5-2697 v2 @ 2.70 GHz (Ivy Bridge-EP, 12C/24T each, 48 logical). **AVX only, no
  AVX2 and no AVX-512**: the engine ran its expert kernels on the AVX1 path (`iq_avx1`), selected by
  a source build with `STRATA_ISA_FLOOR=avx`.
- RAM: 252 GiB DDR3, two NUMA nodes. CPU pool 22 workers, NUMA split 12 threads on node 0 and 11 on
  node 1. Expert complement placed 13.10 GiB on node 0 and 12.89 GiB on node 1 (4 KiB pages).
- Storage: NVMe for the model files and packs.
- OS: Fedora Linux 43, kernel `7.0.14-101.fc43.x86_64`. NVIDIA driver 580.159.03. CUDA 13.0 for the
  source build (`CMAKE_CUDA_ARCHITECTURES=89;120`).
- Strata: **v0.2.11**, commit `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d`, engine 0.1.57, source build.
- Background: every other GPU service on the box was stopped and disabled before the runs; nothing
  else used either card.

## Model and configuration

Both files are the same base model, `Qwen3.8-Flash-Next`, from the Unsloth Dynamic quants.

- A: `UD-IQ4_XS-v31`, 4 shards, 93.67 GB, revision `d96201194f103c906994f283f504209dc8d2a10c`
  (2026-11-04). Native pack built by 0.2.11 setup.
- B: `UD-IQ4_XS`, 3 shards, revision `38bb39ee97821de2c9009abb7e93950eec396e66`. Native pack rebuilt
  by 0.2.11 setup so the older pack from 0.2.9 would not leak into the comparison.

Shared: context 262,144; KV int8 with `--kv-resident 32768` (KV streamed from RAM); expert cache auto;
`--prefill auto`; MTP on (`--spec 4 --spec-min-p 0.5`); vision encoder on the GPU; expert-profile
`data/expert-profile.bin`; resident budget 55 GiB; `--layer-split` and `--pcie-frac` left to the
engine (no override), which is the configuration a user gets from setup.

```text
engine/strata --serve \
  --pack /mnt/nvme/models/strata-0211-data/packs/unsloth-ud-iq4_xs-v31 \
  --native .../Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto \
  --spec 4 --spec-min-p 0.5 --mtp /home/boris/strata-029-data/mtp/rt \
  --max-context 262144 --kv int8 --ple-io ram --kv-resident 32768 \
  --vision --vram-reserve-mib 700
```

The engine chose the split itself, and chose differently for the two files. A: `K=15`, so 15 layers on
the 24 GB card (0.50 ms per layer) and 33 plus the head on the 32 GB card (0.37 ms per layer), with
expert caches of 15.70 GiB and 22.88 GiB. B: `K=13`, 13 layers on the 24 GB card and 35 on the 32 GB
card, with caches of 15.34 GiB and 22.51 GiB. After load A had 531 MiB of VRAM free on the first card
and 404 MiB on the second; B had 398 MiB on the second.

## Method

`strata bench run --suite standard-v1`, the shipped suite. Each run is one warmup request, 12 sanity
checks (four instruction, four retrieval, four tool call) and three timed cases — 77, 4,093 and 16,387
prompt tokens — three repetitions each, one request at a time. Sampling is the protocol's and is not
greedy: temperature 1.0, top_p 0.95, top_k 20, seed 1, thinking on at `reasoning_effort: xhigh`.
Decode is capped at 512 tokens, counts reasoning tokens as generated, and the short case ends at a
natural stop.

Prompt and decode tok/s are the engine's own per-request figures. Prompts are cold on every
repetition (no prefix reuse, 0 reused tokens), so the prompt column is a full read each time.

Both configurations ran back to back on the same box in the same session, with nothing else on the
cards. Run-to-run spread on identical configurations was measured separately at up to 9.3% on 16K
decode, so cells within 10% should not be read as a difference.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| A `UD-IQ4_XS-v31`, short | 77 | 0 | 474 (323-512) | 3 | 73.9 (62.1-92.5) | 117.3 (116.2-127.5) | 1.06 (0.85-1.26) |
| A `UD-IQ4_XS-v31`, 4K | 4,093 | 0 | 512 | 3 | 1,335.5 (1,330.5-1,357.7) | 113.7 (107.9-121.9) | 3.09 (3.04-3.10) |
| A `UD-IQ4_XS-v31`, 16K | 16,387 | 0 | 512 | 3 | 2,339.1 (2,333.6-2,353.1) | 130.0 (118.2-130.8) | 7.04 (7.00-7.06) |
| B `UD-IQ4_XS`, short | 77 | 0 | 362 (227-364) | 3 | 85.8 (80.7-123.2) | 106.9 (99.2-107.4) | 0.92 (0.65-0.97) |
| B `UD-IQ4_XS`, 4K | 4,093 | 0 | 512 (487-512) | 3 | 1,490.4 (1,471.1-1,514.2) | 101.7 (90.3-115.0) | 2.77 (2.73-2.81) |
| B `UD-IQ4_XS`, 16K | 16,387 | 0 | 512 | 3 | 2,572.8 (2,566.4-2,597.0) | 108.3 (103.5-119.9) | 6.41 (6.35-6.43) |

Quality suite: **12/12 for both files**, no failures in any category.

Takeaways: the newer file reads prompts about 10% slower and starts 0.3-0.6 s later on timed
prompts, but decodes 8-20% faster because the engine placed it on a better split; on this hardware
the placement decision is worth more than the quant, and both cards end up full.

## Notes on the two-card split

Recorded because the numbers above are mostly explained by placement rather than by the file.

- **The split dominates decode.** Forcing `--layer-split 13` on file A dropped 16K decode to
  107.9-121.8 across five runs, with prompt unchanged. A at `K=15` measured 128.6 and 130.0 in two
  runs. The engine's `K=15` choice is worth roughly 17% decode here.
- **The PCIe share rule dominates even more.** `pcie_frac` adapts while serving: it started at 0.36
  from the probe, moved to 0.20 once the pool was measured at 81 us per missed expert, then to
  **0.15** in stage 2 (link 13.2 GB/s). The older file settled at 0.20. Restoring the 0.2.10 rule
  (`STRATA_PCIE_AUTO_SPLIT=0`, `STRATA_PCIE_K_RULE=0`, `STRATA_PCIE_SHARE_MAX=0.60`) holds
  `pcie_frac` at 0.35-0.36 and gave 107.7 and 112.1 at 16K decode, against 128.6 and 130.0 under the
  default. Two runs each, same file, same split, consecutively.
- **Prefill chunk stays 8192 on a layer split** (`prompt chunk auto: 8192 tokens, a 96-slot ring`),
  consistent with the 0.2.11 note that a layer split takes the larger ceiling back. Prompt speed is
  therefore unchanged from 0.2.9 on this box, and the v31 prompt cost is not a chunk effect.
- **AVX1 is the likely cause of the prompt cost.** 345 of 1,224 tensors changed type between the two
  files. This CPU has no AVX2, so any type without an AVX1 kernel falls back to a slower path. The
  prompt column moves 9-12% while decode does not, which fits a redistribution of work rather than a
  bandwidth limit.
- **Tool-call sanity moved with the split, but sampling is not controlled.** At `K=13`, A failed the
  tool-call case in 3 of 5 runs (~1,310 tokens generated, `finish_reason: stop`, no structured call);
  at `K=15` it passed 4 of 4, and B passed 6 of 6 at `K=13`. With temperature 1.0 each run is a fresh
  sample, so that split by placement is suggestive rather than established — five runs cannot separate
  placement from sampling noise. The 0.2.11 notes describe sparse block selection sitting on near-ties,
  where a small change flips the pick, which is one way a tool call could flip with it.
- **Both cards are full, and the rest is weights rather than KV.** KV never sits in VRAM here: 32,768
  of 262,144 cells are resident in 0.77 GiB of pinned RAM and the remainder streams. Taking run A: the
  24 GB card has 23.89 GiB usable, its weights leave 16.26 GiB free, the expert cache takes 15.70 GiB
  (66% of the card) and 531 MiB is left. The 32 GB card has 31.37 GiB the engine can use, its weights
  leave 23.94 GiB, the cache takes 22.88 GiB (73%) and 404 MiB is left, with an 835 MiB MTP draft
  layer on top. So roughly 7.5 GiB per card is dense weights, head and draft, not cache.
- **Cache coverage and headroom.** The caches hold 17,361 of 24,576 profiled pairs in A and 16,883 in
  B (98.9% of routed mass either way) with no file reads once warm, and the 32 GB card still ends with
  404 MiB (A) or 398 MiB (B) free. The engine's own LOW warning there suggests
  `--vram-reserve-later-mib 1064` for a build with less headroom than this one.
- **Storage played no part.** The engine fills the two GPU caches first, puts what is left in
  page-locked RAM, and reads from the file only when neither holds an expert. With 252 GiB of RAM the
  RAM copy took 26.0 GiB and the file tier was left 0.1 GiB. The first request after start read
  75.6 GB from the drive while that copy was built; every measured request after it read 0 MB and
  waited 0 ms on storage.

Raw per-run figures are in this folder; the B run is the sibling folder ending `-5ceff2de`.

## Correctness and limitations

- Both files passed the engine's 12 sanity checks in the runs above. No needle test and no external
  quality suite were run.
- Three requests per case, one at a time, no concurrency stage. Sampling was temperature 1.0 with
  thinking on, so each repetition is a different sample and each quality check is a single sample
  rather than a pass rate.
- Run-to-run spread on identical configurations reached 9.3% on 16K decode, so small differences
  between rows are not signal.
- The cards are not identical. The engine prices the split on decode only and says outright that it
  can mis-rank differing cards; the `K=13` / `K=15` comparison in the notes is the empirical check,
  not a prediction.
- Only 404 MiB of VRAM was free on the second card after load. Window graphs are captured on first
  use and need room, so a build with slightly different weights may stall or need a larger reserve.
- Prompt throughput on this CPU is bounded by AVX1, so these prompt numbers are a floor for Qwen3.8-
  Flash-Next on Ivy Bridge-EP rather than a property of the file alone.
