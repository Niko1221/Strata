# Community benchmark: 2x RTX 2080 Ti 22 GB (layer split), full power limits

Measured on 2026-10-08 by caolonghao, on the same PC as the [2x V100-SXM2 report](../2026-10-08-community-2x-v100-sxm2/)
(the V100s were idle here). This is the **second** 2x 2080 Ti data point after #1225 - and the
first at full power limits: #1225's cards ran a 100 W cap. Main limitations: one machine, one
operator, no needle / correctness runs, shared NVMe with other load.

## Hardware and software

- **GPUs:** 2x RTX 2080 Ti **22 GB** (256-bit TU102 cards with 22 GB memory mods, as in #1225),
  sm_75, power limit 250 W (max 280 W), PCIe x16 (engine probe 13.1 GB/s host->device).
- **CPU and RAM:** 2x Xeon Silver 4316 (80 threads, AVX-512), 125 GB DDR4 ECC.
- **Storage:** shared NVMe (other load present). **OS:** Ubuntu 22.04.5, driver 580.178.04,
  CUDA 12.8 toolkit, gcc 11.4.
- **Strata:** upstream `main` at `e8ca9af`, engine 0.1.40.2, source build into `engine/` for
  `-DCMAKE_CUDA_ARCHITECTURES=75` - the same toolchain as #1225 (CUDA 12.8, driver 580.178.04).

## Model and configuration

- **Model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` **IQ3_XXS** at setup's pinned revision
  (`ed59f92`), two shards (75.8 GB) - the same quant as #1225.
- **Settings:** context 131,072; KV int8 with `--kv-resident 32768`; `--expert-cache auto`;
  `--prefill auto`; MTP `--spec 4 --spec-min-p 0.5`; `--remote-expert-opt`; `--layer-split auto`
  (the engine chose **K=26**, caches holding 20,209 of 24,576 profiled pairs, ~99.5%); greedy,
  reasoning none, vision off.

```text
./setup.sh --yes --family qwen --model IQ3_XXS --gpus 2,3 --context 131072 --data-dir <nvme> --no-start
./run-iq3_xxs.sh
```

## Method

The same unmodified `benchmark.py` as the V100 report (fresh nonce prompts, `reused: 0` logged on
every measured request, 256-token cap, one warm-up excluded, three measured runs, engine-side
throughputs, client-side TTFT).

## Results

| Prompt tokens | Reused | Generated | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 4,096 | 0 | 256 | 3 | 981.3 (960.1-986.2) | 84.1 (80.0-84.7) | 4.22 (4.19-4.31) |
| 32,768 | 0 | 256 | 3 | 1,684.4 (1,677.7-1,703.8) | 75.1 (72.8-79.5) | 19.55 (19.33-19.66) |

- Decode expert-cache hit rate 0.997; MTP draft acceptance ~0.63-0.67; KV streaming hit VRAM on
  97.4-99.4% of block reads at depth.
- **Against #1225** (same cards' 22 GB mods, same quant, same harness, 100 W cap, Ryzen 9 5950X,
  62 GB RAM): 4K decode 84.1 vs 52.8 (+59%), 32K prompt 1,684 vs 477 (+253%). The power cap is the
  obvious first suspect; a server platform with AVX-512 and more RAM may contribute.
- On this PC the pair is also directly comparable with the V100 pair's UD-IQ4_XS run (same
  harness): the 2080 Ti pair reads 4K prompts 23% slower and 32K prompts 29% slower than the V100
  pair does on the larger 4-bit model, and decodes 9% / 6% faster on the smaller 3-bit one.

## Correctness and limitations

- Every request completed; answers were coherent, not graded. No needle, recall or HumanEval runs;
  no 128K-point (the harness cap is unrelated to the card - it is the same ~456K-token synthesizer
  limit as the V100 report, well above this model's measured cells anyway).
- One machine; shared-storage load; single pass of three runs per cell.

## Files

- `data/main/` - per-run JSON and the harness log.

Prepared with an AI assistant (Claude Code) from the raw measurements; numbers are the engine's
and the client's own output, unedited.
