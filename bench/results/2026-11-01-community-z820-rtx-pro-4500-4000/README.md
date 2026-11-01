# Community benchmark on 2x NVIDIA RTX PRO (4500 32 GB + 4000 24 GB), Z820

Measured 2026-11-01 by Jack Kenner (GitHub `jkenner111`). Four configurations of the **same**
machine and model, run back to back against a freshly loaded engine each time: one card alone,
then the pair layer-split with three different splits. The headline is that a **layer split
across two unequal cards nearly doubles decode** on a machine that is otherwise CPU- and
RAM-bound.

**This is a draft-protocol submission.** All four runs carry
`strata-qwen38-thinking-v1-draft.1` and are therefore outside the standard manifest, which
does not exist yet. They are complete runs with 12/12 sanity in every arm.

## Hardware and software

- **GPU 0:** NVIDIA RTX PRO 4500 Blackwell, 32 GB (31.9 GiB usable), sm_120
- **GPU 1:** NVIDIA RTX PRO 4000 Blackwell, 24 GB (23.1 GiB usable), sm_120
- **CPU:** 2x Intel Xeon E5-2697 v2 @ 2.70 GHz (24 cores total), no AVX-512
- **RAM:** 251.8 GiB DDR3
- **Storage:** model on `/dev/sdb3`, a 1.9 TB **SATA** SSD. The runner reports
  "Storage of the model files: not measured" because it cannot see the backing device;
  the true value is SATA, not NVMe, and this matters for load time, not steady state.
- **OS:** Fedora Linux 43 (Workstation Edition), kernel 7.0.14-101.fc43.x86_64
  (the generated READMEs render this as `Linux 43 (Workstation Edition)` with a literal
  `<host>` placeholder — an engine template bug, left uncorrected here rather than edit
  generated evidence)
- **Driver / CUDA:** 580.159.03 / CUDA 13.0
- **Strata:** commit `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`, engine 0.1.55, source build
- **Background work:** **none.** Both cards were idle (2 MiB used) and the host's GPU service
  units were stopped for the duration. The runner reports "Background work: not measured"
  because no `--background` value was passed.
- **Power:** stock, no limits applied.

## Model and configuration

- **Repository:** `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
  `38bb39ee97821de2c9009abb7e93950eec396e66`
- **Source model:** `Qwen/Qwen3.8-Flash-Next`, revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540` — verified independently against the Hugging Face
  API, but **not** captured by the runner (`--source-revision` was not passed on these runs;
  the reports say "source revision not measured")
- **Quantization:** UD-IQ4_XS, three shards, 93.7 GB total
- **Context:** 262,144 tokens (model native), 1 server slot
- **Build:** `STRATA_ISA_FLOOR=avx` — this host has no AVX-512, so the build floors the ISA
  rather than emitting AVX2/512 kernels. Vision enabled. Resident budget 55 GiB.
- **Sampling:** the protocol's fixed set — temperature 1.0, top_p 0.95, top_k 20, min_p 0.0,
  penalties 0.0 / 1.0, `reasoning_effort: xhigh`, thinking on and preserved

### The four configurations

| Folder | Cards | Split | Pool workers | Notes |
|---|---|---|---|---|
| `arm1-4500-nonumactl-20261101` | RTX PRO 4500 alone | — | 22 (default) | single-card baseline |
| `arm4-dualsplit-20261101` | both, 4500 = main | `auto` → layer 13 | 22 (default) | **best result** |
| `arm5-dualsplit-k10-20261101` | both, 4500 = main | explicit layer 10 | 22 (default) | more of the model on the 24 GB card |
| `arm2-dualsplit-w11-20261101` | both, 4500 = main | `auto` → layer 13 | 11 | worker-count sweep |

`"gpu": [1, 0]` — helper card first, main last. The last card runs head, draft and verify, so
the 4500 (more VRAM, more SMs, higher clocks) takes that role. Layer split is by layer index:
the RTX PRO 4000 holds layers 0–12, the 4500 holds 13–47 plus head and draft.

## Results

Medians with (min–max) over three repetitions. Decode and prompt rates are the engine's own
`timings`; TTFT is the client's monotonic clock.

### Single card — RTX PRO 4500 (32 GB)

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | --- | --- | --- |
| short | n/a | 61.3 (56.5–68.8) | 0.82 (0.65–1.01) |
| prompt-4k | 1143.8 (1081.9–1149.7) | 63.5 (61.1–64.9) | 3.60 (3.59–3.81) |
| prompt-16k | 2031.1 (2001.0–2044.0) | 64.6 (62.3–70.4) | 8.11 (8.06–8.23) |

### Dual card, layer 13 — RTX PRO 4000 (0–12) + RTX PRO 4500 (13–47)

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | --- | --- | --- |
| short | n/a | **108.7** (105.2–119.6) | 0.80 (0.58–0.81) |
| prompt-4k | **1520.8** (1504.4–1540.2) | **113.2** (97.8–120.9) | **2.72** (2.69–2.74) |
| prompt-16k | **2581.0** (2579.0–2584.7) | **105.3** (102.4–114.9) | **6.39** (6.38–6.39) |

### Dual card, explicit layer 10

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | --- | --- | --- |
| short | n/a | 95.2 (93.2–103.4) | 1.35 (0.99–1.47) |
| prompt-4k | 1266.1 (1253.9–1306.5) | 103.9 (87.0–107.8) | 3.26 (3.16–3.29) |
| prompt-16k | 2280.7 (2269.8–2281.9) | 105.6 (96.5–107.5) | 7.22 (7.22–7.26) |

### Dual card, layer 13, pool-workers 11

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | --- | --- | --- |
| short | n/a | 113.3 (96.3–114.3) | 0.82 (0.69–0.84) |
| prompt-4k | 1501.5 (1492.4–1535.4) | 103.9 (89.8–108.7) | 2.75 (2.69–2.77) |
| prompt-16k | 2565.0 (2481.8–2574.7) | 100.3 (98.8–116.0) | 6.43 (6.40–6.65) |

Sanity checks: **12 of 12** in all four runs (I1–I4 instruction, R1–R4 retrieval, T1–T4 tool
call).

### What the numbers say

- **Layer split is worth ~78% decode and ~33% prompt.** 61.3 → 108.7 tok/s short,
  63.5 → 113.2 at 4K, 64.6 → 105.3 at 16K. Prompt at 16K rises 2031 → 2581 tok/s and TTFT
  drops 8.11 s → 6.39 s.
- **The engine's own placement predictor was wrong about which split is better.** With
  `layer_split: auto` it chose 13 but flagged K=10 as within 0.7%. K=10 measured 20% worse on
  4K prompt and 13% worse on 16K prompt, and its TTFT was worse at every size. Decode was a
  dead heat at 16K (105.6 vs 105.3). The predictor prices decode, not prompt.
- **`--pool-workers` is not a lever here.** 11 workers vs 22 produced overlapping ranges
  (4K decode 103.9 vs 113.2, both spanning ~98–121).
- These numbers are well below the same hardware's own October history, but that comparison
  is not valid: October ran a different engine version (0.1.41 vs 0.1.55), greedy sampling
  instead of the protocol's temp 1.0 / xhigh, different prompts, and from NVMe rather than
  SATA. Do not compare them.

## Limitations

- **Draft protocol.** Not eligible for the standard manifest; reported per the community guide.
- **Six of nine performance requests hit the 512-token cap** in the single-card arm (3/3 at 4K
  and 3/3 at 16K) and 2/3 at 4K, 3/3 at 16K in the layer-13 arm. A capped request's decode
  rate is measured to the cap, and `first_answer_s` is absent where reasoning consumed the
  whole budget. This is anticipated by the protocol, not a run failure, but it means decode
  figures reflect truncated generations.
- **`--source-revision` was not passed.** The base-model revision above was verified
  out of band; the reports themselves say `null` with that reason.
- **Storage is SATA, not NVMe.** Irrelevant to steady-state decode (the expert arena is
  RAM-resident; requests 2 and 3 read 0 MB from disk) but it does inflate load time.
- **No optional concurrency stage** was run; all figures are concurrency 1.
- **Vision was enabled** but no image case is part of the standard suite.

## Reproducing

```sh
# build (this host has no AVX-512)
STRATA_ISA_FLOOR=avx ./setup.sh --setup --yes --no-start \
  --family qwen --model UD-IQ4_XS --gguf-dir <UD-IQ4_XS dir> \
  --data-dir <data dir> --gpu 0 --context 262144 --low-ram resident

# single card
bin/strata bench run --profile ud-iq4-xs-gpu0 --suite standard-v1 --track custom \
  --model-revision 38bb39ee97821de2c9009abb7e93950eec396e66 --quantization UD-IQ4_XS

# both cards — config has "gpu": [1, 0], layer_split "auto"
bin/strata bench run --profile ud-iq4-xs-dual --suite standard-v1 --track custom \
  --model-revision 38bb39ee97821de2c9009abb7e93950eec396e66 --quantization UD-IQ4_XS
```

Do not wrap in `numactl`: the engine splits its pool by NUMA node itself on this dual-socket
host, and pinning to one socket costs throughput.
