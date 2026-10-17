# Community benchmark: 2x AMD Radeon RX 9070 (RX 9070 + RX 9070 XT, gfx1201), Ryzen 9 5950X

Measured on 2026-10-08 by `HANDLE` on an Arch Linux desktop. This tests **IQ3_S** at the model's
native **262,144-token context** with the layer split across two **RDNA4** cards on ROCm, the PLE
table locked in RAM (`--ple-io ram`), and the native expert profile. It is one of the few RDNA4 /
Linux reports; all numbers are freshly-processed (no prefix reuse) unless stated.

The median throughput was **841.6 prefill / 67.1 decode tok/s at 4,226 prompt tokens**, **1,488.6 /
64.5 at 33,428**, **1,916.4 / 60.0 at 130,542**, and a single run at **248,708 tokens: 1,844.8
prefill / 59.5 decode**. The requests are synthetic code-explanation prompts, greedy, with a
260-token cap. They do not establish general answer quality or performance on other workloads.

## Hardware

- **GPUs:** 2x AMD Radeon RX 9070, 16 GB each, gfx1201 (RDNA4, Navi 48):
  - **GPU 0** (`rocm-smi` index) at PCI `0000:42:00.0`: **RX 9070**, 220 W (stock), 245 W cap.
  - **GPU 1** at PCI `0000:45:00.0`: **RX 9070 XT**, **capped to 225 W** (default 304 W), to keep
    the two cards' power draw even.
  - Both at **PCIe 4.0 x8** (the two CPU slots run x8/x8 when both are populated). `amdgpu_top`
    reports the DPM range `Gen1x8 - Gen4x8`; the engine's own PCIe probe measured **14.4 GB/s**
    host-to-device on each card.
  - The machine's display is driven by these cards; no other GPU workloads ran.
- **CPU and RAM:** AMD Ryzen 9 5950X (16 cores, 32 threads, **AVX2, no AVX-512**). The engine used
  15 expert-pool workers plus its host thread. 126 GiB DDR4 (speed not measured; no root for
  `dmidecode`); a ZFS ARC of ~6 GiB is present.
- **Storage:** NVMe SSD. **Other workloads:** desktop idle.
- **PSU:** 1000 W.

## Software

- **OS:** Arch Linux, kernel `7.2.7-arch1-1`, the kernel's `amdgpu` driver.
- **ROCm:** 7.2.4; hipBLASLt 1.2.2 (package `hipblaslt 7.2.4-1`). `HSA_OVERRIDE_GFX_VERSION` not
  needed (native gfx1201).
- **Strata:** commit [`e8ca9af`](https://github.com/Niko1221/Strata/commit/e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a)
  (2026-10-07), engine **0.1.40.2**, a **local HIP source build for gfx1201** (`engine/BUILD.json` in
  this folder; engine `src 72af3ad7bd9126f6`; `sha256` of the binary in `model-provenance.json`).
  hipBLASLt enabled with the shipped tuning file `tools/hip/gfx1201-hipblaslt-100202.txt`.

## Model and configuration

**Model:** [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
**IQ3_S**. The repository revision was not recorded; the downloaded shard sizes are:

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` — 54,817,524,224 bytes
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` — 28,800,138,432 bytes (this is the **PLE
  table**, passed as `--ple-gguf`)

The native pack (`Strata-data/packs/iq3_s`) was built with `tools/iq_pack.py`
(`conversions.json` included). The expert profile is the bundled `data/expert-profile.bin`
(`sha256 8f59b4aa…`, see `model-provenance.json`). The MTP draft layer is the prepared
`Strata-data/mtp/rt` (`mtp-manifest.json` copied here). No vision.

Configuration (`strata-iq3_s.json` in this folder, API key redacted):

```text
engine/strata --serve
  --pack Strata-data/packs/iq3_s
  --native  .../IQ3_S-00001-of-00002.gguf
  --ple-gguf .../IQ3_S-00002-of-00002.gguf  --ple-io ram
  --expert-profile data/expert-profile.bin  --expert-cache auto
  --prefill auto:32768  --spec 4 --spec-min-p 0.5  --mtp Strata-data/mtp/rt
  --max-context 262144  --kv int8 --kv-resident 32768
  --remote-expert-opt
  --conversation-cache-mib 8192 --conversation-cache-slots 4
```

- `"gpu": [1, 0]` (the RX 9070 XT is the main/first card), `layer_split: "24"` → layers 0-23 on
  the XT, 24-47 on the RX 9070;
- expert cache `auto`: **10,790 slots / 20,869 MiB** total (primary 6,057 slots / 11,153 MiB),
  filled from the profile at start;
- `--prefill auto:32768` chose a **19,456-token chunk** — capped by the second card's 4,733 slots
  (the engine logs the cap);
- `--pcie-frac` is probed per card and set to **0.40** (both links 14.4 GB/s);
- `--pool-workers` left automatic → **15** workers + host; 48 tasks/phase;
- server sampling defaults: `temperature 0.6, top_p 0.95, top_k 20, repetition_penalty 1.05,
  penalty_last_n 256`, `reasoning_budget_tokens 4096`. **The benchmark requests override
  temperature to 0 and set `reasoning_effort: none`**;
- no `--calibrate`, no experimental speed projection, no control vectors.

## Method

[benchmark.py](benchmark.py) (this folder):
- builds deterministic code-like filler, puts a **unique nonce** in each cold request so no prefix
  cache is reused, and posts to the OpenAI-compatible endpoint;
- **cold sweep:** 3 runs each at target 4,096 / 32,768 / 128,000 tokens, **260-token cap**;
- **near-limit:** one run at 244,000 target (248,708 actual tokens), same cap;
- **warm:** one ~8K prompt posted twice (the second reuses the prefix);
- **TTFT:** 3 streaming requests on a short (~200-token) prompt, timing the first non-empty content
  delta over loopback (keep-alives skipped);
- samples **memory every second** into `telemetry.jsonl` (engine `smaps_rollup` RSS/Locked,
  `MemAvailable`, per-card `mem_info_vram_used`).

Full per-run data in `results.json`; aggregates in `summary.json`; TTFT in `ttft.json`; recall in
`needles.json`; memory in `memory-summary.json`. The engine's whole log is `engine.log` (no
credentials). Loading time is excluded. The expert cache was filled at start and kept between runs.

```bash
STRATA_API_KEY=... python3 benchmark.py --url http://127.0.0.1:11634 --out .
python3 ../../../../tools/needle_bench.py --url http://127.0.0.1:11634 \
        --lengths 32k,128k --depths 10,50,90 --out needles.json --api-key ...
```

## Results

Each cell is the median **[minimum-maximum]** of three runs. Cold rows read their whole prompt
(**reused = 0**); decode is `engine_generated / decode_ms`, never generated ÷ total time.

| Prompt tokens | Reused | Generated | Runs | Prefill tok/s | Decode tok/s | TTFT s |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 4,226 | 0 | 260 | 3 | 841.6 [814.8-841.9] | 67.1 [65.7-70.6] | not measured |
| 33,428 | 0 | 260 | 3 | 1,488.6 [1,469.4-1,492.5] | 64.5 [63.4-67.9] | not measured |
| 130,542 | 0 | 260 | 3 | 1,916.4 [1,913.1-1,920.9] | 60.0 [56.5-62.0] | not measured |
| 248,708 (single) | 0 | 260 | 1 | 1,844.8 | 59.5 | not measured |
| ~200 (TTFT) | 0 | ~230 chars | 3 | not measured | not measured | 1.81 [1.80-1.81] |

- Every request reached the 260-token cap (`finish_reason: length`); run durations ~9 s, 27 s, 73 s
  and 140 s. Prefill rises with length (short prompts pay the fixed per-chunk cost), decode falls
  gently.
- **Warm reuse:** the same 8,224-token prompt read cold at **1,004 tok/s / 12.0 s**, then **8,217 of
  8,224 tokens reused** (7 freshly read) at **3.86 s** — the conversation cache works across
  identical prefixes.
- The 248,708-token prompt fits the native context with 13,436 tokens to spare; no request failed,
  paged, or was cancelled.

**Memory** (`memory-summary.json`, 381 one-second samples): engine RSS **86.6-89.8 GiB** (of which
**26.8 GiB locked** — the PLE table, `--ple-io ram`); `MemAvailable` never below **29.3 GiB** of
126 GiB; VRAM used **15.59 GiB** (GPU 0) and **15.69 GiB** (GPU 1) of 16 GiB. These are sampled
values; brief peaks between samples can be missed.

## Correctness and limitations

The repository's unchanged `tools/needle_bench.py` found **all six needles**, depths 10/50/90% at
both 32K and 128K ([needles.json](needles.json)):

- `32k` prompts were 32,343-32,344 tokens; `128k` prompts 126,310-126,312 tokens.
- All six were fully fresh (no reuse in the sweep before them at those exact lengths).

**Limits of this report:**
- One machine, one quantization, one configuration, one synthetic prompt family, greedy decoding.
- The server's configured sampling (temperature 0.6, repetition penalty) was overridden per request;
  different sampling would change decode speed.
- `--ple-io ram` (26.8 GiB permanently locked) is a deliberate choice for this host, not a default.
- Agentic/tool use, coding-task correctness, vision, concurrency, multi-turn quality and sustained
  thermal runs were not measured (the cards are power-capped).
- The 248,708-token case is a single run, not three.

## Notes

- This is the same machine behind a separate PCIe-link A/B (both cards Gen3 x8 vs Gen4 x8); that
  comparison, and an OCuLink x8+x4 run, are planned as a follow-up.
- `--prefill auto:32768` was chosen so the engine could pick the largest fitting chunk; on this pair
  it settles at 19,456 tokens, limited by the second card's expert-cache slots.
