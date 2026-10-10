# Community benchmark: 2x RTX 2080 Ti 22 GB (layer split), full power limits

Measured on 2026-10-08 by caolonghao, on the same PC as the [2x V100-PCIE-32GB report](../2026-10-08-community-2x-v100-pcie-32gb/)
(the V100s were idle here). The **second** 2x 2080 Ti data point after #1225 - and the first at
full power limits: #1225's cards ran a 100 W cap. Main limitations: one machine, one operator, no
needle or correctness runs, shared NVMe with other load.

## Hardware and software

- **GPUs:** 2x RTX 2080 Ti **22 GB** (256-bit TU102 cards with 22 GB memory mods, as in #1225),
  sm_75, PCIe **Gen3 x16** (the engine's probe measured 13.1 GB/s host->device), power limit
  250 W (max 280 W) - #1225 ran the same memory mod at a 100 W cap.
- **CPU and RAM:** 2x Xeon Silver 4316 (2.30 GHz, 80 threads, AVX-512), 125 GB DDR4 ECC.
- **Storage:** shared NVMe (other load present). **OS:** Ubuntu 22.04.5 (kernel 6.8.0-138),
  driver 580.178.04, CUDA 12.8 toolkit, gcc 11.4.
- **Strata:** upstream `main` at `e8ca9af`, engine 0.1.40.2, **source build** into `engine/` for
  `-DCMAKE_CUDA_ARCHITECTURES=75` - the same toolchain as #1225 (CUDA 12.8, driver 580.178.04).
  No NVLink between the cards (consumer TU102 pair, no bridge).

## Model and configuration

- **Model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` **IQ3_XXS** at setup's pinned revision
  `ed59f92`; shards `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` and
  `-00002-of-00002.gguf` (75.8 GB); setup's default pack for the qwen family, its own MTP draft
  layer and expert profile; no custom draft vocabulary; vision off.
- **Settings:** context 131,072; KV int8 with `--kv-resident 32768`; `--expert-cache auto`;
  `--prefill auto`; MTP `--spec 4 --spec-min-p 0.5`; `--remote-expert-opt`; `--layer-split auto`
  (the engine chose **K=26**; the caches hold 20,209 of 24,576 profiled pairs, ~99.5%); **no
  low-RAM mode**; CPU expert pool at its **default** worker count; **no calibration, no speed
  projection**; greedy, reasoning effort none.

```text
./setup.sh --yes --family qwen --model IQ3_XXS --gpus 2,3 --context 131072 --data-dir <nvme> --no-start
engine/strata --serve --pack <data>/packs/iq3_xxs \
  --native <data>/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf <data>/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 \
  --mtp <data>/mtp/rt --max-context 131072 --kv int8 --kv-resident 32768 --remote-expert-opt --layer-split auto
```

## Method

The same unmodified `benchmark.py` as the V100 report ([2026-09-30-community-rtx-5090](../2026-09-30-community-rtx-5090/)):
fresh nonce prompts (the engine logged `reused: 0` on every measured request), 256-token cap,
streaming, client-side TTFT (first token is answer text - reasoning effort none), the engine's own
`prompt_ms` / `decode_ms` for throughput, **model loading excluded**, one warm-up excluded, three
measured runs, one loaded server.

```text
python benchmark.py --url http://127.0.0.1:8080 --pack <data>/packs/iq3_xxs --targets 4096,32768 --runs 3 --out <dir>
```

## Results

| Prompt tokens | Reused | Generated | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 4,096 | 0 | 256 | 3 | 981.3 (960.1-986.2) | 84.1 (80.0-84.7) | 4.22 (4.19-4.31) |
| 32,768 | 0 | 256 | 3 | 1,684.4 (1,677.7-1,703.8) | 75.1 (72.8-79.5) | 19.55 (19.33-19.66) |

- Decode expert-cache hit rate 0.983-0.997 (median 0.996); MTP draft acceptance 0.67-0.74 per run
  (cell medians 0.68 / 0.73); KV streaming hit VRAM on 97.4-99.4% of block reads at depth.
- **Against #1225** (same 22 GB mods, same quant, same harness): changed settings are the power
  cap (100 W there, 250 W here), the host (Ryzen 9 5950X / 62 GB RAM there, 2x Xeon 4316 / 125 GB
  here), the engine (0.1.40 at `82f46a8` there, 0.1.40.2 here) and KV streaming (off there,
  `--kv-resident 32768` here). Measured: 4K decode 84.1 vs 52.8 (+59%), 32K prompt 1,684 vs 477
  (+253%). The power cap is the obvious first suspect.
- On this PC the pair is also directly comparable with the V100 pair's UD-IQ4_XS run (same
  harness, same day): the 2080 Ti pair reads 4K prompts 23% slower and 32K prompts 28% slower than
  the V100 pair does on the larger 4-bit model, and decodes 9% faster at 4K and 6% slower at 32K
  on the smaller 3-bit one.

## Correctness and limitations

- Every request completed; answers were coherent, not graded. No needle, recall or HumanEval runs.
- No 128K point (not run; the harness's ~456K-token synthesis cap was not the limiting factor).
- Total request latency and RAM / VRAM usage were not measured (engine and client throughput only).
- One machine; shared-storage load; a single pass of three runs per cell.

## Files

- `data/main/` - the 4K cell's per-run request records plus `summary.json` / `results.json`
  covering all six runs. Not shipped, per the repository's trimming practice: the harness's raw
  per-run dumps and its progress log (`*.log` is gitignored; the 32K cell's ~63 KB per-request
  files were dropped for size), and the engine log (`strata-iq3_xxs.log`).

Prepared with an AI assistant (Claude Code) from the raw measurements; the numbers are the
engine's and the client's own output, unedited.
