# Community benchmark on RTX PRO 4500 32 GB + RTX PRO 4000 24 GB in a dual Xeon E5-2697 v2 (HP Z820)

Measured on 2026-11-01 by Jack Kenner (jkenner111). Tested: UD-IQ4_XS at 262,144 context in four
configurations — an RTX PRO 4500 alone, both cards layer-split at the automatic boundary (layer 13),
both cards at an explicit layer 10, and a pool-worker comparison at 11 instead of 22. All runs used
`--low-ram resident` and an AVX1 source build. Main limitations: draft protocol; six of nine
performance requests reached the 512-token cap; `--source-revision` not passed. First published
result on a pre-AVX2 CPU.

## Hardware and software

- GPU: NVIDIA RTX PRO 4500 Blackwell 32 GB (31.9 GiB usable, sm_120, 82 SMs @ 2.41 GHz) and NVIDIA
  RTX PRO 4000 Blackwell 24 GB (23.1 GiB usable, sm_120, 70 SMs @ 2.06 GHz). Both PCIe 3.0 x16;
  engine probe 12.7–13.2 GB/s host-to-device. Link idles at gen 1, maximum gen 3.
- CPU: 2x Intel Xeon E5-2697 v2 @ 2.70 GHz (Ivy Bridge-EP, 12C/24T each, 24C/48T). **No AVX2, no
  AVX-512.** Expert kernels ran on the AVX1 128-bit path (`iq_avx1`, older-CPU build) with 22 pool
  workers: 12 on NUMA node 0, 11 on node 1, host on core 0.
- RAM: 251.8 GiB usable of 256 GB installed — 16 x 16 GB DDR3-1866 ECC, 4 channels per socket, two
  NUMA nodes.
- Storage: `/dev/sdb3`, 1.9 TB SATA SSD (model files).
- OS: Fedora Linux 43 (Workstation Edition), kernel 7.0.14-101.fc43.x86_64. NVIDIA driver 580.159.03,
  CUDA 13.0.
- Strata: commit `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`, v0.1.55, source build with
  `STRATA_ISA_FLOOR=avx`.
- Background: none. Both GPUs idle at start (2 MiB each); the two GPU service units were stopped.

## Model and configuration

- Model: `unsloth/Qwen3.8-Flash-Next-GGUF`, UD-IQ4_XS, 93.7 GB in three shards, revision
  `38bb39ee97821de2c9009abb7e93950eec396e66`. Source model `Qwen/Qwen3.8-Flash-Next` at
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`, verified out of band but not captured by the runner
  (`--source-revision` not passed; recorded as null).
- Vision encoder on the GPU (`--vision`, mmproj BF16); shipped expert profile `expert-profile.bin`.
- Context 262,144; KV int8 with 32,768 resident cells; `--low-ram resident` with
  `--resident-budget-gib 55`; PLE table in RAM (`--ple-io ram`); expert cache auto; prefill auto;
  MTP on (`--spec 4 --spec-min-p 0.5`, `--mtp rt`); `--vram-reserve-mib 700`.
- Sampling fixed by the protocol: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, penalties
  0.0 / 1.0, `reasoning_effort: xhigh`, thinking on with `preserve_thinking`.

Effective engine arguments, single-card run:

```text
--pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
--expert-profile expert-profile.bin --expert-cache auto --prefill auto
--spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --ple-io ram
--kv-resident 32768 --resident-budget-gib 55 --vision --vram-reserve-mib 700
```

The dual-card runs use the same arguments with `"gpu": [1, 0]` (helper first, main last) and
`layer_split` either `auto` (resolved to 13) or explicit (10). The 4500 takes layers 13–47 plus head
and draft; the 4000 takes layers 0–12. The split is used because 32 + 24 GB of VRAM is smaller than
the 55 GiB expert arena.

### Local deviations from the default path

- Build with `STRATA_ISA_FLOOR=avx` — this CPU has no AVX2.
- `--low-ram resident`: experts are page-locked in RAM and swapped against the GPU caches instead of
  read from the GGUF on disk. Setup computed `--resident-budget-gib 55` from the model's 59.5 GB
  arena, not from installed RAM.
- Resident mode retained across the layer split, supported from engine 0.1.40
  (`RESIDENT_SPLIT_ENGINE`). Engines before that rewrite `--resident-experts` to `--mmap-experts`,
  which reads experts through the OS file cache.
- No `numactl`. The engine builds its own NUMA-aware pool.

## Method

Four `bench run --suite standard-v1 --track custom` runs, one per configuration, protocol
`strata-qwen38-thinking-v1-draft.1`. Three measured repetitions per case, concurrency 1. Prompt and
decode tok/s are the engine's own per-request figures; TTFT is the client's monotonic clock. Medians
with minimum–maximum range. A separate quality check ran alongside each run.

Prefill plan as logged by the engine, dual-card run:

```text
prefill plan: chunk 8192, ring 96 kept: file set 0.4 GiB fits the page cache 174.5 GiB
  -> about 0.0 GB read from the drive per 32768-token prompt (4 chunks)
prefill plan inputs: 55.4 GiB experts, 38.1 GiB in the GPU caches, 24.8 GiB RAM copy
  (3300 lend slots kept), file tier 0.0 GiB
```

Single-card run: `55.4 GiB experts, 22.2 GiB in the GPU cache, 37.0 GiB RAM copy, file tier 0.0 GiB`.

## Results

### Single card — RTX PRO 4500 32 GB

| Case | Prompt tok/s | Decode tok/s | TTFT s |
|---|---|---|---|
| short | 96.9 (78.5–122.3) | 61.3 (56.5–68.8) | 0.82 (0.65–1.01) |
| prompt-4k | 1143.8 (1081.9–1149.7) | 63.5 (61.1–64.9) | 3.60 (3.59–3.81) |
| prompt-16k | 2031.1 (2001.0–2044.0) | 64.6 (62.3–70.4) | 8.11 (8.06–8.23) |

### Dual card, layer 13 — 4000 (0–12) + 4500 (13–47)

| Case | Prompt tok/s | Decode tok/s | TTFT s |
|---|---|---|---|
| short | 98.0 (96.8–137.1) | 108.7 (105.2–119.6) | 0.80 (0.58–0.81) |
| prompt-4k | 1520.8 (1504.4–1540.2) | 113.2 (97.8–120.9) | 2.72 (2.69–2.74) |
| prompt-16k | 2581.0 (2579.0–2584.7) | 105.3 (102.4–114.9) | 6.39 (6.38–6.39) |

### Dual card, explicit layer 10

| Case | Prompt tok/s | Decode tok/s | TTFT s |
|---|---|---|---|
| short | 57.9 (52.9–79.0) | 95.2 (93.2–103.4) | 1.35 (0.99–1.47) |
| prompt-4k | 1266.1 (1253.9–1306.5) | 103.9 (87.0–107.8) | 3.26 (3.16–3.29) |
| prompt-16k | 2280.7 (2269.8–2281.9) | 105.6 (96.5–107.5) | 7.22 (7.22–7.26) |

### Dual card, layer 13, pool workers 11

| Case | Prompt tok/s | Decode tok/s | TTFT s |
|---|---|---|---|
| short | 96.2 (93.5–115.5) | 113.3 (96.3–114.3) | 0.82 (0.69–0.84) |
| prompt-4k | 1501.5 (1492.4–1535.4) | 103.9 (89.8–108.7) | 2.75 (2.69–2.77) |
| prompt-16k | 2565.0 (2481.8–2574.7) | 100.3 (98.8–116.0) | 6.43 (6.40–6.65) |

### Quality

| Run | Passed | instruction | retrieval | tool_call |
|---|---|---|---|---|
| single card | 12/12 | 4/4 | 4/4 | 4/4 |
| dual, layer 13 | 12/12 | 4/4 | 4/4 | 4/4 |
| dual, layer 10 | 12/12 | 4/4 | 4/4 | 4/4 |
| dual, layer 13, w11 | 12/12 | 4/4 | 4/4 | 4/4 |

## Observations

- Layer split, layer 13 vs single card: decode 61.3 → 108.7 (short), 63.5 → 113.2 (4K), 64.6 → 105.3
  (16K). Prompt at 16K 2,031.1 → 2,581.0 tok/s; TTFT 8.11 → 6.39 s.
- `layer_split: auto` selected 13 and logged that 10 was within 0.7%. Layer 10 measured 1,266.1
  (4K) and 2,280.7 (16K) prompt tok/s against 1,520.8 and 2,581.0 at layer 13; decode was 103.9 vs
  113.2 (4K) and 105.6 vs 105.3 (16K). The log states the predictor prices decode rather than prompt.
- Pool workers 11 vs 22: decode ranges overlap (4K 103.9 [89.8–108.7] vs 113.2 [97.8–120.9]).
- GPU expert cache: 22.2 GiB single card, 38.1 GiB dual. RAM copy: 37.0 GiB single card, 24.8 GiB
  dual. File tier 0.0 GiB in both.

## Limitations

- Draft protocol (`strata-qwen38-thinking-v1-draft.1`); outside the standard manifest.
- Six of nine performance requests reached the 512-token cap (single card: 3/3 at 4K, 3/3 at 16K;
  dual layer 13: 2/3 at 4K, 3/3 at 16K). Decode is measured to the cap on those generations.
- `--source-revision` not passed; base-model revision verified out of band and recorded as null.
- Single-card run logged VRAM headroom of 502 MiB and 372 MiB at request start, below the engine's
  512 MiB threshold where the driver may page GPU memory to host RAM. Decode did not degrade
  visibly, but `--vram-reserve-mib` was not raised.
- No concurrency stage; all figures are concurrency 1.
- Storage is SATA. The drive contributes 0.0 GiB per 32K prompt per the prefill plan, so the figure
  applies to load time only.
- Not comparable to this machine's October results: engine 0.1.41 vs 0.1.55, greedy sampling vs
  temperature 1.0 with `reasoning_effort: xhigh`, different prompts, NVMe vs SATA.

## Reproducing

```sh
STRATA_ISA_FLOOR=avx ./setup.sh --setup --yes --no-start \
  --family qwen --model UD-IQ4_XS --gguf-dir <UD-IQ4_XS dir> \
  --data-dir <data dir> --gpu 0 --context 262144 --low-ram resident

bin/strata bench run --profile ud-iq4-xs-gpu0 --suite standard-v1 --track custom \
  --model-revision 38bb39ee97821de2c9009abb7e93950eec396e66 --quantization UD-IQ4_XS
```
